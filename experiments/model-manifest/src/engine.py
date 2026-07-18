"""Long-lived serving engine: shared weights, per-session KV caches.

Deliberately transport-agnostic — no Monarch, no gRPC, no HTTP in this
file. The binding's three categories map directly onto the serving
architecture:

    bound   -> fetched once, shared read-only across every session
    derived -> computed once from config, shared
    state   -> allocated fresh per session; the graphs mutate it in
               place, so a session IS its state tensors

Multi-turn works because the prefill graph has a dynamic sequence
dimension and takes explicit cache positions: a follow-up turn prefills
only the new suffix tokens at positions [cached_len, ...) — the
existing cache rows are the conversation memory.
"""

import json
import os
import time

import torch

from executor import fetch_tensor, rope_inv_freq
from generator import rehydrate, share_state, strip_asserts


class Engine:
    def __init__(self, manifest_path, binding_path="cached.binding.json",
                 device="cpu", compile_decode=False, cas="cas"):
        from transformers import AutoTokenizer
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        self.device = device
        self.compile_decode = compile_decode
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        with open(binding_path) as f:
            self.binding = json.load(f)
        os.makedirs(cas, exist_ok=True)

        self.config = CONFIG_MAPPING[
            self.manifest["config"]["model_type"]
        ].from_dict(self.manifest["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.manifest["source"]["repo"]
        )

        # Shared, read-only: fetched/computed exactly once.
        self.shared = {}
        for fqn, ref in self.binding["bound"].items():
            self.shared[fqn], _ = fetch_tensor(ref, self.manifest["files"], cas)
        for fqn in self.binding["derived"]:
            self.shared[fqn] = rope_inv_freq(self.config)

        probe = torch.export.load("decode.pt2")
        probe_state = {**probe.state_dict, **probe.constants}
        self.state_specs = {
            fqn: (probe_state[fqn].shape, probe_state[fqn].dtype)
            for fqn in self.binding["state"]
        }

        self.sessions = {}
        self._next_id = 0

    def new_session(self) -> str:
        session_id = f"s{self._next_id}"
        self._next_id += 1

        tensors = dict(self.shared)
        for fqn, (shape, dtype) in self.state_specs.items():
            tensors[fqn] = torch.zeros(shape, dtype=dtype)

        t0 = time.perf_counter()
        prefill = rehydrate("prefill.pt2", self.binding, self.manifest,
                            self.config, tensors, self.device).module()
        decode = rehydrate("decode.pt2", self.binding, self.manifest,
                           self.config, tensors, self.device).module()
        share_state(prefill, decode, self.binding["state"])
        if self.compile_decode:
            strip_asserts(decode)
            decode = torch.compile(decode)

        self.sessions[session_id] = {
            "prefill": prefill,
            "decode": decode,
            "messages": [],
            "cached_len": 0,
            "build_s": time.perf_counter() - t0,
        }
        return session_id

    def chat(self, session_id: str, text: str, max_new_tokens: int = 96) -> dict:
        session = self.sessions[session_id]
        session["messages"].append({"role": "user", "content": text})
        ids = self.tokenizer.apply_chat_template(
            session["messages"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )["input_ids"]
        total_len = ids.shape[1]
        start = session["cached_len"]
        new_ids = ids[:, start:].to(self.device)

        t0 = time.perf_counter()
        logits = session["prefill"](
            input_ids=new_ids,
            cache_position=torch.arange(start, total_len, device=self.device),
        )
        prefill_s = time.perf_counter() - t0

        token_ids, position = [], total_len
        next_id = int(logits[0, -1].argmax())
        t1 = time.perf_counter()
        for _ in range(max_new_tokens):
            if next_id == self.tokenizer.eos_token_id:
                break
            token_ids.append(next_id)
            logits = session["decode"](
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([position], device=self.device),
            )
            next_id = int(logits[0, -1].argmax())
            position += 1
        loop_s = time.perf_counter() - t1

        reply = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        session["messages"].append({"role": "assistant", "content": reply})
        session["cached_len"] = position

        return {
            "text": reply,
            "session_tokens": position,
            "new_prompt_tokens": total_len - start,
            "generated": len(token_ids),
            "prefill_ms": round(prefill_s * 1e3),
            "ms_per_token": round(loop_s / max(len(token_ids), 1) * 1e3, 1),
        }

    def stats(self) -> dict:
        return {
            "sessions": {
                sid: {"cached_len": s["cached_len"],
                      "turns": len(s["messages"]) // 2,
                      "build_s": round(s["build_s"], 1)}
                for sid, s in self.sessions.items()
            },
            "state_bytes_per_session": sum(
                torch.zeros(shape, dtype=dtype).element_size()
                * torch.Size(shape).numel()
                for shape, dtype in self.state_specs.values()
            ),
        }
