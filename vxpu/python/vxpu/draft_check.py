# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The drafter against its reference (hero run 13).

The engine runs the checkpoint's drafter (DeepSeek-V4.1's DSpark stages) as an exported program
whose rings the verifying decode step fills with every token row's hidden states, drafts and
rejected drafts alike, and whose output goes through the native loop or the Python-driven rows.
The fork's own drafter has only ever run one token a step. This module feeds the fork's model
the engine's output one token at a time and asks its drafter at every (token, position) the
engine drafted at, so a difference in the drafts names a defect in the export or in the verify
flow (the rings' slack, the positions, the prefix after a partial acceptance), while identical
drafts mean the acceptance rate is the head's and the workload's.
"""
import os
import re

import torch


def eager_model(model_dir, max_cache_len=1024, step_tokens=1):
    """The fork's model with a local checkpoint's weights, eagerly on the CPU in bf16: the expert
    banks stacked from the per-expert tensors (the fork keeps a bank per projection and layer).
    Returns (model, args)."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from . import deepseek_v41 as dv
    torch.set_default_dtype(torch.bfloat16)
    ds = dv._fork()
    args = dv.model_args(model_dir, max_cache_len, step_tokens=step_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ds.Transformer(args, tokenizer).eval()
    state = dict(load_file(os.path.join(model_dir, "model.safetensors")))
    bank = re.compile(r"^(?P<prefix>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<i>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$")
    groups = {}
    for name in list(state):
        m = bank.match(name)
        if m:
            key = f"{m['prefix']}.{m['proj']}" + ("" if m["kind"] == "weight" else "_scale")
            groups.setdefault(key, []).append((int(m["i"]), state.pop(name)))
    for key, parts in groups.items():
        state[key] = torch.stack([t for _, t in sorted(parts, key=lambda p: p[0])], dim=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    shared = {"embed.weight", "head.weight"}  # the stages share the model's embedding and head
    missing = [k for k in missing if not (k.startswith("mtp.") and k.split(".", 2)[-1] in shared)]
    if missing or unexpected:
        raise ValueError(f"eager model: missing {missing[:8]}, unexpected {unexpected[:8]}")
    return model, args


def compare(engine, model, prompts, new_tokens, arrivals=None):
    """Decode `prompts` on `engine` (a speculative artifact: steps of several tokens, the model's
    drafter), recording every draft the engine made per row, then replay each prompt's sequence
    through `model` one token at a time and draft at the same (token, position)s. Returns counts:
    steps drafted, drafts identical, confidence identical, the first differences, and whether the
    engine's tokens equal the eager model's greedy continuation."""
    from . import engine as eng
    from .engine import SamplingParams
    greedy = SamplingParams(do_sample=False)
    block = engine.binding["draft"]["block"]
    records = []  # (prompt index, position, drafts, confidence) per drafting row and step
    original = eng._ModelDrafter.propose

    def propose(self, rows, ids, position, count):
        out = original(self, rows, ids, position, count)
        pos = position.tolist()
        for row, request in enumerate(rows.requests):
            if request is None:
                continue
            conf = rows.confidence[row] if rows.confidence else None
            records.append((request.prompt_index, pos[row], list(out[row][:block]), conf))
        return out

    eng._ModelDrafter.propose = propose
    try:
        # the request's prompt index: the engine builds a _Request per prompt in order
        original_request = eng._Request.__init__

        def init(self, prompt, *a, **kw):
            original_request(self, prompt, *a, **kw)
            self.prompt_index = init.count
            init.count += 1
        init.count = 0
        eng._Request.__init__ = init
        try:
            out = engine.generate_rows(prompts, new_tokens, greedy, arrivals=arrivals)
        finally:
            eng._Request.__init__ = original_request
    finally:
        eng._ModelDrafter.propose = original
    tokens = out["token_ids"]
    result = {"steps": 0, "same": 0, "confidence_same": 0, "first_differences": [], "target_differences": 0,
              "accepted": out.get("accepted", 0), "drafted": out.get("drafted", 0)}
    by_prompt = {}
    for i, p, d, c in records:
        by_prompt.setdefault(i, []).append((p, d, c))
    scale = 1e6
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            seq = prompt + tokens[i]
            x = torch.tensor([seq])
            _, logits, hidden = model(x[:, :len(prompt)], 0)
            model.spec_observe(hidden, 0)
            fed = len(prompt)
            eager_next = [int(logits.argmax(-1))]
            for p, drafts, conf in sorted(by_prompt.get(i, [])):
                while fed < p:
                    _, logits, hidden = model(x[:, fed:fed + 1], fed)
                    model.spec_observe(hidden, fed)
                    eager_next.append(int(logits.argmax(-1)))
                    fed += 1
                if p >= len(seq):
                    continue  # a draft past the last emitted token (the step that finished the request)
                ref, _, c = model.spec_draft(x[:, p], p)
                ref = ref[0, 1:].tolist()
                score = (torch.sigmoid(c.float()) * scale).round().to(torch.int64)[0].tolist()
                got_conf = [round(v * scale) for v in conf] if conf else None
                result["steps"] += 1
                result["same"] += ref == drafts
                result["confidence_same"] += got_conf is None or score[:len(got_conf)] == got_conf
                if ref != drafts and len(result["first_differences"]) < 5:
                    result["first_differences"].append({"prompt": i, "position": p, "engine": drafts, "reference": ref})
            result["target_differences"] += sum(1 for a, b in zip(eager_next, tokens[i]) if a != b)
    return result
