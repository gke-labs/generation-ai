# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A transformers-shaped front for a model running on vXPU.

    from transformers import AutoTokenizer
    from vxpu import AutoModelForCausalLM      # the only changed line

    model = AutoModelForCausalLM.from_pretrained("google/gemma-4-31B-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    inputs = tokenizer("What is your favorite condiment?", return_tensors="pt")
    out = model.generate(inputs.input_ids, max_new_tokens=30)
    tokenizer.batch_decode(out, skip_special_tokens=True)[0]

from_pretrained exports the weightless artifact (meta device, cached
under ~/.cache/vxpu), ships it to the router, and waits for the
executor; generate sends token ids and gets token ids back, with the
same torch loop and sampling rules transformers applies, run on
whatever hardware the router placed the model on. The tokenizer, the
chat template, tools, and streamers are the ordinary transformers
objects; this class only replaces the model.
"""

import os
import warnings

import torch

from .client import Client, PortForward

DEFAULT_ROUTER_TARGET = "pod/vxpu-router"


class VxpuModelForCausalLM:
    """Looks enough like a PreTrainedModel for generate()-based code."""

    def __init__(self, repo_id, client, session, config, generation_config,
                 forward=None, artifact_dir=None):
        self.name_or_path = repo_id
        self.client = client
        self.session = session
        self.config = config
        self.generation_config = generation_config
        self.artifact_dir = artifact_dir
        self._forward = forward
        # Inputs are plain token ids on the CPU; placement is the
        # router's concern, so code that does `.to(model.device)` works.
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.last = None  # statistics of the latest generate()

    @classmethod
    def from_pretrained(cls, repo_id, *, router=None, artifact_dir=None,
                        max_cache_len=2048, revision="main", timeout=3600,
                        progress=print, **unused):
        """Export (if needed), ship, and wait for the model.

        router: "host:port" of a vxpu-router, or None to port-forward
        to pod/vxpu-router with kubectl (VXPU_ROUTER overrides).
        artifact_dir: where the exported artifact lives/goes
        (default ~/.cache/vxpu/artifacts/<repo>/<max_cache_len>).
        max_cache_len: the session's token capacity, fixed at export.
        Unknown keyword arguments (device_map, dtype, attn_implementation
        ...) are accepted and ignored: the executor decides those.
        """
        from transformers import AutoConfig, GenerationConfig

        if unused:
            progress(f"vxpu: ignoring from_pretrained arguments "
                     f"{sorted(unused)}; placement, dtype and attention "
                     "are the executor's decision")
        artifact_dir = artifact_dir or os.path.join(
            os.path.expanduser("~/.cache/vxpu/artifacts"),
            repo_id.replace("/", "--"), str(max_cache_len))
        if not os.path.exists(os.path.join(artifact_dir, "decode.pt2")):
            from .export import export_artifact
            progress(f"vxpu: exporting {repo_id} on the meta device "
                     f"(no weights) to {artifact_dir}")
            export_artifact(repo_id, artifact_dir, max_cache_len, revision)

        forward = None
        router = router or os.environ.get("VXPU_ROUTER")
        if not router:
            forward = PortForward(DEFAULT_ROUTER_TARGET)
            try:
                router = forward.start()
            except RuntimeError as e:
                raise RuntimeError(
                    "no vxpu-router reachable: pass router='host:port', "
                    "set VXPU_ROUTER, or run `vxpu up` first") from e
        client = Client(router, timeout=timeout)
        session = client.load_artifact(artifact_dir, progress=progress)

        config = AutoConfig.from_pretrained(repo_id, revision=revision)
        try:
            generation_config = GenerationConfig.from_pretrained(
                repo_id, revision=revision)
        except OSError:
            generation_config = GenerationConfig()
        return cls(repo_id, client, session, config, generation_config,
                   forward=forward, artifact_dir=artifact_dir)

    # --- the generate() surface --------------------------------------

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None,
                 max_new_tokens=None, max_length=None, do_sample=None,
                 temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, eos_token_id=None,
                 streamer=None, seed=0, generation_config=None,
                 inputs=None, **kwargs):
        """transformers.GenerationMixin.generate for batch size 1.

        Returns a LongTensor of shape [1, prompt + new], prompt first,
        exactly like transformers, so `out[0][input_len:]` and
        `tokenizer.batch_decode(out)` work unchanged. Sampling settings
        default to the model's generation_config, as in transformers.
        A `streamer` (e.g. transformers.TextStreamer) receives the
        prompt, then each new token, then end().
        """
        if input_ids is None:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("generate() needs input_ids")
        ids = torch.as_tensor(input_ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[0] != 1:
            raise NotImplementedError(
                "vxpu sessions generate one sequence at a time "
                f"(got batch size {ids.shape[0]})")
        if attention_mask is not None:
            mask = torch.as_tensor(attention_mask).reshape(-1)
            if mask.numel() == ids.shape[1] and not bool(mask.all()):
                # Left padding would shift positions; strip it.
                ids = ids[:, mask.bool()]
        # Processor outputs (token_type_ids, mm_token_type_ids) and
        # local-execution knobs are accepted silently.
        unsupported = sorted(k for k in kwargs
                             if k not in ("cache_implementation",
                                          "pad_token_id", "use_cache",
                                          "return_dict_in_generate",
                                          "output_scores", "num_beams",
                                          "min_new_tokens", "token_type_ids",
                                          "mm_token_type_ids"))
        if unsupported:
            warnings.warn(f"vxpu: ignoring generate() arguments "
                          f"{unsupported}")
        if kwargs.get("num_beams", 1) not in (None, 1):
            raise NotImplementedError("vxpu generates greedy/sampled "
                                      "sequences; num_beams > 1 is not "
                                      "supported")

        gc = generation_config or self.generation_config
        prompt_len = ids.shape[1]
        if max_new_tokens is None:
            if max_length is None:
                max_length = getattr(gc, "max_length", 20) or 20
            max_new_tokens = max_length - prompt_len
            if max_new_tokens <= 0:
                raise ValueError(
                    f"max_length ({max_length}) is not larger than the "
                    f"prompt ({prompt_len} tokens); pass max_new_tokens")
        if generation_config is not None:
            # Explicit config: forward its settings unless overridden.
            do_sample = gc.do_sample if do_sample is None else do_sample
            temperature = (gc.temperature if temperature is None
                           else temperature)
            top_k = gc.top_k if top_k is None else top_k
            top_p = gc.top_p if top_p is None else top_p
            repetition_penalty = (gc.repetition_penalty
                                  if repetition_penalty is None
                                  else repetition_penalty)
        extra_eos = []
        if eos_token_id is not None:
            extra_eos = ([eos_token_id] if isinstance(eos_token_id, int)
                         else list(eos_token_id))

        prompt = ids[0].tolist()
        if streamer is not None:
            streamer.put(ids)
        new_ids = []
        for token_id in self.session.generate_ids(
                prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=extra_eos, seed=seed):
            new_ids.append(token_id)
            if streamer is not None:
                streamer.put(torch.tensor([token_id]))
        if streamer is not None:
            streamer.end()
        self.last = self.session.last
        return torch.tensor([prompt + new_ids], dtype=torch.long)

    # --- small PreTrainedModel conveniences ---------------------------

    def can_generate(self):
        return True

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        # Placement belongs to the router; `.to("cuda")` is a no-op.
        return self

    def new_session(self):
        """Start a fresh KV cache (e.g. a new, unrelated conversation).
        Not required: generate() reuses whatever prefix matches."""
        self.session = self.client.new_session()
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "vxpu exposes generate(), not forward(): the executor runs "
            "the exported prefill/decode graphs and does not return logits")

    def close(self):
        self.client.close()
        if self._forward is not None:
            self._forward.stop()

    def __repr__(self):
        return (f"VxpuModelForCausalLM({self.name_or_path!r}, "
                f"router={self.client.address!r}, "
                f"session={self.session.session_id!r})")


AutoModelForCausalLM = VxpuModelForCausalLM
