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
follow-up turn prefills the suffix at positions [cached_len, ...).
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

    def _eos_ids(self):
        """Every id that ends a turn.

        Chat models stop on more than the tokenizer's eos: Gemma emits
        <end_of_turn> (config eos_token_id lists several ids). Union the
        model config, its text config, and the tokenizer.
        """
        from transformers import GenerationConfig

        sources = [self.manifest["config"],
                   self.manifest["config"].get("text_config") or {}]
        # generation_config.json carries the ids a chat model actually
        # stops on (Gemma 4: <end_of_turn>, <turn|>, and <|tool_response>
        # after a tool call); like the tokenizer, it is fetched from the
        # source repo until it is part of the manifest.
        try:
            sources.append(GenerationConfig.from_pretrained(
                self.manifest["source"]["repo"]).to_dict())
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] no generation_config: {e}", flush=True)
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
                "cached_ids": [],  # token ids currently in the cache
                "cached_len": 0,
            }
        return session_id

    def chat(self, session_id, text, max_new_tokens=96, raw_prompt=False):
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
                return_tensors="pt")["input_ids"]
        messages = session["messages"] + [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"]

    def _chat_locked(self, session_id, text, max_new_tokens, raw_prompt):
        if session_id not in self.sessions:
            raise ValueError(f"unknown session_id: {session_id}")
        session = self.sessions[session_id]
        self._bind(session["state"])
        ids = self._prompt_ids(session, text, raw_prompt)
        total_len = ids.shape[1]

        if total_len > self.max_cache_len:
            raise ValueError(
                f"conversation length ({total_len} tokens) exceeds "
                f"maximum cache capacity ({self.max_cache_len} tokens)")

        # Reuse the cache only for the prefix that is really there:
        # re-rendering a transcript need not reproduce the generated
        # ids token-for-token, and a raw client may rewrite history.
        # Prefill from the first differing token: the caches are
        # position-addressed (explicit cache_position, causal masks), so
        # entries from that position on are simply overwritten and
        # nothing stale beyond it is ever attended.
        cached = session["cached_ids"]
        start = 0
        while (start < len(cached) and start < total_len
               and cached[start] == int(ids[0, start])):
            start += 1
        if start == total_len:
            # Everything is cached, including the last prompt token
            # (a retried or truncated prompt); the prefill graph needs
            # at least one token, so recompute the last one's logits.
            start = total_len - 1

        allowed_tokens = self.max_cache_len - total_len
        max_tokens_to_generate = max(0, min(max_new_tokens, allowed_tokens))
        new_ids = ids[:, start:].to(self.device)

        prefill_started = time.perf_counter()
        logits = self._prefill(
            input_ids=new_ids,
            cache_position=torch.arange(start, total_len,
                                        device=self.device))
        prefill_s = time.perf_counter() - prefill_started

        token_ids, position = [], total_len
        next_id = int(logits[0, -1].argmax())
        loop_started = time.perf_counter()
        for _ in range(max_tokens_to_generate):
            if next_id in self.eos_ids:
                break
            token_ids.append(next_id)
            if position >= self.max_cache_len:
                break
            logits = self._decode_run(
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([position],
                                            device=self.device))
            next_id = int(logits[0, -1].argmax())
            position += 1
        loop_s = time.perf_counter() - loop_started

        # Raw clients own the format and need the model's control
        # tokens (e.g. Gemma's <|tool_call> ... <tool_call|> delimiters);
        # templated replies are plain text.
        reply = self.tokenizer.decode(token_ids,
                                      skip_special_tokens=not raw_prompt)
        if not raw_prompt:
            session["messages"].append({"role": "user", "content": text})
            session["messages"].append(
                {"role": "assistant", "content": reply})
        session["cached_ids"] = ids[0].tolist() + token_ids
        session["cached_len"] = position

        return {
            "text": reply,
            "session_tokens": position,
            "new_prompt_tokens": total_len - start,
            "generated": len(token_ids),
            "prefill_ms": round(prefill_s * 1e3),
            "ms_per_token": round(
                loop_s / max(len(token_ids), 1) * 1e3, 1),
        }
