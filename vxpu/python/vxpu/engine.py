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

"""Serving engine: shared weights and compiled graphs, per-session KV.

The binding's three categories map directly onto serving:

    bound   -> fetched once, shared read-only across sessions
    derived -> computed once from config, shared
    state   -> allocated fresh per session; the graphs mutate it in
               place, so a session IS its state tensors

The two graphs (prefill, decode) are built and compiled exactly once
per model. Compilation is the expensive step (tens of seconds), and the
decode graph is identical across sessions — only the state buffers it
reads and mutates differ. So a session owns just its state tensors, and
each turn binds them into the shared modules (a reference swap the
compiled graph re-reads at call time — no recompile). One GPU serves
turns serially anyway, so a lock around bind+run costs nothing real.

Multi-turn conversation costs only the new tokens: the prefill graph
has a dynamic sequence dimension and explicit cache positions, so a
follow-up turn prefills only the tokens beyond the cached prefix.
"""

import json
import os
import threading
import time

import torch

from .rehydrate import (derived_tensor, fetch_tensor, load_program,
                        share_state, strip_asserts)


class Engine:
    def __init__(self, artifact_dir, device="cpu", compile_decode=False,
                 cas_dir=None):
        from transformers import AutoTokenizer
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING)

        self.artifact_dir = artifact_dir
        self.device = device
        self.compile_decode = compile_decode
        with open(os.path.join(artifact_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(artifact_dir, "binding.json")) as f:
            self.binding = json.load(f)
        if cas_dir:
            os.makedirs(cas_dir, exist_ok=True)

        self.config = CONFIG_MAPPING[
            self.manifest["config"]["model_type"]
        ].from_dict(self.manifest["config"])
        self.max_cache_len = int(self.binding.get("max_cache_len", 1024))
        # Tokenizer/processor files are not yet part of the manifest.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.manifest["source"]["repo"])
        self.generation_config = self._generation_config()
        self.eos_ids = self._eos_ids()
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] | capitalize }}: {{ message['content'] }}\\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "Assistant:"
                "{% endif %}"
            )

        # Shared, read-only: fetched/computed exactly once. Shapes for
        # derived/state tensors come from the decode program itself.
        probe = torch.export.load(
            os.path.join(artifact_dir, "decode.pt2"))
        probe_meta = {**probe.state_dict, **probe.constants}

        # Weights move to the device once, here: both programs then
        # alias the same storage (device pass no-ops on tensors already
        # in place), so a session costs cache memory, not a weight copy.
        self.shared = {}
        self.bytes_fetched = 0
        for fqn, ref in self.binding["bound"].items():
            tensor, downloaded = fetch_tensor(
                ref, self.manifest["files"], cas_dir)
            self.shared[fqn] = tensor.to(device)
            self.bytes_fetched += downloaded
        for fqn in self.binding["derived"]:
            self.shared[fqn] = derived_tensor(
                fqn, probe_meta[fqn], self.config).to(device)
        self.state_specs = {
            fqn: (probe_meta[fqn].shape, probe_meta[fqn].dtype)
            for fqn in self.binding["state"]}

        self.sessions = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._build()

    def _generation_config(self):
        """The model's generation_config.json as a dict (may be empty).

        It carries the ids a chat model actually stops on (Gemma 4:
        <end_of_turn>, <turn|>, and <|tool_response> after a tool call)
        and the sampling defaults transformers' generate() would use.
        Like the tokenizer, it is fetched from the source repo until it
        is part of the manifest.
        """
        from transformers import GenerationConfig
        try:
            return GenerationConfig.from_pretrained(
                self.manifest["source"]["repo"]).to_dict()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] no generation_config: {e}", flush=True)
            return {}

    def _eos_ids(self):
        """Every id that ends a turn.

        Chat models stop on more than the tokenizer's eos: Gemma emits
        <end_of_turn> (config eos_token_id lists several ids). Union the
        model config, its text config, and the tokenizer.
        """
        sources = [self.manifest["config"],
                   self.manifest["config"].get("text_config") or {},
                   self.generation_config]
        ids = set()
        for source in sources:
            eos = source.get("eos_token_id")
            if isinstance(eos, int):
                ids.add(eos)
            elif isinstance(eos, (list, tuple)):
                ids.update(int(i) for i in eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return ids

    def _build(self):
        """Build and compile the two graphs once. Scratch state is bound
        now and overwritten per session at chat time."""
        scratch = {
            fqn: torch.zeros(shape, dtype=dtype, device=self.device)
            for fqn, (shape, dtype) in self.state_specs.items()}
        tensors = {**self.shared, **scratch}
        self._prefill = load_program(
            os.path.join(self.artifact_dir, "prefill.pt2"),
            tensors, self.device).module()
        self._decode = load_program(
            os.path.join(self.artifact_dir, "decode.pt2"),
            tensors, self.device).module()
        shared = share_state(self._prefill, self._decode,
                             self.binding["state"])
        assert shared > 0, "prefill/decode share no cache state"
        self._decode_run = self._decode
        if self.compile_decode:
            strip_asserts(self._decode)
            self._decode_run = torch.compile(self._decode)

    def _bind(self, state):
        """Point both graphs' state buffers at this session's tensors.

        A reference swap, O(number of buffers); the compiled decode
        re-reads its buffers on each call, so this triggers no recompile.
        """
        for module in (self._prefill, self._decode):
            for fqn, tensor in state.items():
                parent, name = (fqn.rsplit(".", 1) if "." in fqn
                                else ("", fqn))
                target = (module.get_submodule(parent) if parent
                          else module)
                if name in target._buffers:
                    target._buffers[name] = tensor

    def new_session(self):
        with self._lock:
            session_id = f"s{self._next_id}"
            self._next_id += 1
            self.sessions[session_id] = {
                "state": {
                    fqn: torch.zeros(shape, dtype=dtype,
                                     device=self.device)
                    for fqn, (shape, dtype) in self.state_specs.items()},
                "messages": [],
                # Token ids whose KV entries are in the cache, kept in
                # step with every prefill/decode so an interrupted
                # stream never leaves it describing a cache that is not
                # there.
                "cached_ids": [],
            }
        return session_id

    def chat(self, session_id, text, max_new_tokens=96, raw_prompt=False):
        """One templated (or raw) text turn; greedy, like the CLI expects."""
        with self._lock:
            return self._chat_locked(session_id, text, max_new_tokens,
                                     raw_prompt)

    def _prompt_ids(self, session, text, raw_prompt):
        """Token ids of the whole conversation so far plus this turn.

        Templated mode: the engine owns the transcript and renders it
        with the tokenizer's chat template. Raw mode: the client owns
        the transcript and sends the complete rendered prompt (e.g.
        with tools); it is tokenized verbatim, adding BOS only if the
        rendered text does not already start with it.
        """
        if raw_prompt:
            bos = self.tokenizer.bos_token or ""
            add_special = not (bos and text.startswith(bos))
            return self.tokenizer(
                text, add_special_tokens=add_special,
                return_tensors="pt")["input_ids"][0].tolist()
        messages = session["messages"] + [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"][0].tolist()

    def _chat_locked(self, session_id, text, max_new_tokens, raw_prompt):
        session = self._session(session_id)
        ids = self._prompt_ids(session, text, raw_prompt)
        token_ids, stats = [], None
        for chunk in self._generate_locked(
                session, ids, max_new_tokens, SamplingParams(do_sample=False)):
            if chunk["token_ids"]:
                token_ids.extend(chunk["token_ids"])
            if chunk["done"]:
                stats = chunk

        # Text replies exclude the end-of-turn id itself. Raw clients
        # own the format and need the model's other control tokens
        # (e.g. Gemma's <|tool_call> ... <tool_call|> delimiters);
        # templated replies are plain text.
        if token_ids and token_ids[-1] in self.eos_ids:
            token_ids = token_ids[:-1]
        reply = self.tokenizer.decode(token_ids,
                                      skip_special_tokens=not raw_prompt)
        if not raw_prompt:
            session["messages"].append({"role": "user", "content": text})
            session["messages"].append(
                {"role": "assistant", "content": reply})
        return {
            "text": reply,
            "session_tokens": len(session["cached_ids"]),
            "new_prompt_tokens": stats["prefilled_tokens"],
            "generated": stats["generated"],
            "prefill_ms": stats["prefill_ms"],
            "ms_per_token": stats["ms_per_token"],
        }

    def generate(self, session_id, input_ids, max_new_tokens, params,
                 extra_eos_ids=()):
        """transformers' generate() loop, streamed.

        Yields dicts {"token_ids": [...], "done": bool, ...}; the last
        one has done=True and the statistics. The generation lock is
        held for the whole stream (one accelerator serves turns
        serially anyway).
        """
        with self._lock:
            session = self._session(session_id)
            yield from self._generate_locked(
                session, list(input_ids), max_new_tokens, params,
                extra_eos_ids)

    def _session(self, session_id):
        if session_id not in self.sessions:
            raise ValueError(f"unknown session_id: {session_id}")
        return self.sessions[session_id]

    def _generate_locked(self, session, ids, max_new_tokens, params,
                         extra_eos_ids=()):
        self._bind(session["state"])
        total_len = len(ids)
        if total_len == 0:
            raise ValueError("empty prompt")
        if total_len > self.max_cache_len:
            raise ValueError(
                f"prompt length ({total_len} tokens) exceeds maximum "
                f"cache capacity ({self.max_cache_len} tokens)")
        eos_ids = self.eos_ids | set(int(i) for i in extra_eos_ids)
        sampler = params.resolve(self.generation_config)

        # Reuse the cache only for the prefix that is really there:
        # re-rendering a transcript need not reproduce the generated
        # ids token-for-token, and a client may rewrite history.
        # Prefill from the first differing token: the caches are
        # position-addressed (explicit cache_position, causal masks), so
        # entries from that position on are simply overwritten and
        # nothing stale beyond it is ever attended.
        cached = session["cached_ids"]
        start = 0
        while (start < len(cached) and start < total_len
               and cached[start] == ids[start]):
            start += 1
        if start == total_len:
            # Everything is cached, including the last prompt token;
            # recompute its logits by re-prefilling just that token.
            start = total_len - 1

        allowed_tokens = self.max_cache_len - total_len
        max_tokens_to_generate = max(0, min(max_new_tokens, allowed_tokens))
        new_ids = torch.tensor([ids[start:]], dtype=torch.long,
                               device=self.device)

        # The prefill overwrites cache positions from `start` on; until
        # it has finished only the shared prefix is known to be there.
        session["cached_ids"] = cached[:start]
        prefill_started = time.perf_counter()
        logits = self._prefill(
            input_ids=new_ids,
            cache_position=torch.arange(start, total_len,
                                        device=self.device))
        prefill_s = time.perf_counter() - prefill_started
        cached = session["cached_ids"] = list(ids)

        # The returned sequence matches transformers: the end-of-turn
        # id that stops generation is included (and counted) but, as
        # with transformers, never fed through the model, so it is not
        # part of the cache.
        token_ids = []
        finish_reason = "length"
        next_id = sampler.pick(logits[0, -1], ids)
        loop_started = time.perf_counter()
        for _ in range(max_tokens_to_generate):
            token_ids.append(next_id)
            yield {"token_ids": [next_id], "done": False}
            if next_id in eos_ids:
                finish_reason = "eos"
                break
            if len(cached) >= self.max_cache_len:
                break
            logits = self._decode_run(
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([len(cached)],
                                            device=self.device))
            cached.append(next_id)
            next_id = sampler.pick(logits[0, -1], cached)
        loop_s = time.perf_counter() - loop_started

        yield {
            "token_ids": [],
            "done": True,
            "finish_reason": finish_reason,
            "prompt_tokens": total_len,
            "prefilled_tokens": total_len - start,
            "generated": len(token_ids),
            "prefill_ms": round(prefill_s * 1e3),
            "ms_per_token": round(
                loop_s / max(len(token_ids), 1) * 1e3, 1),
        }


class SamplingParams:
    """Generation parameters as transformers' GenerationConfig names
    them. None means "use the model's generation_config default", so a
    bare request samples exactly as model.generate(input_ids) would."""

    FIELDS = ("do_sample", "temperature", "top_k", "top_p",
              "repetition_penalty")

    def __init__(self, do_sample=None, temperature=None, top_k=None,
                 top_p=None, repetition_penalty=None, seed=None):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed

    def resolve(self, generation_config):
        defaults = {"do_sample": False, "temperature": 1.0, "top_k": 50,
                    "top_p": 1.0, "repetition_penalty": 1.0}
        values = {}
        for name in self.FIELDS:
            value = getattr(self, name)
            if value is None:
                value = generation_config.get(name)
            if value is None:
                value = defaults[name]
            values[name] = value
        return Sampler(seed=self.seed, **values)


class Sampler:
    """The logits processors transformers applies for these settings,
    in its order: repetition penalty, temperature, top-k, top-p, then
    multinomial sampling (or argmax when do_sample is false)."""

    def __init__(self, do_sample, temperature, top_k, top_p,
                 repetition_penalty, seed=None):
        # The same bounds transformers' processors enforce.
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be > 0 when sampling")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be in [0, 1]")
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_k = int(top_k or 0)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))

    def pick(self, logits, context_ids):
        logits = logits.detach().float().cpu()
        if self.repetition_penalty != 1.0 and context_ids:
            ids = torch.tensor(sorted(set(context_ids)), dtype=torch.long)
            scores = logits[ids]
            logits[ids] = torch.where(scores < 0,
                                      scores * self.repetition_penalty,
                                      scores / self.repetition_penalty)
        if not self.do_sample:
            return int(logits.argmax())
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if 0 < self.top_k < logits.numel():
            cutoff = torch.topk(logits, self.top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=False)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative <= (1 - self.top_p)
            remove[-1] = False  # always keep the most likely token
            logits = logits.masked_fill(
                remove.scatter(0, sorted_idx, remove), float("-inf"))
        probs = logits.softmax(-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))
