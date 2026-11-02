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

"""Hero run 13, the speculation model's inputs from one run: one artifact (plain or with the
drafter), the first ROWS prompts of a file, greedy, NEW_TOKENS tokens after a 4 s warm-up, in one
process. Prints a RESULT13 line (step time, distinct experts a layer a step, acceptance per
position, the drafter graph's own replay time) and writes, next to the results, the expert trace
(VXPU_EXPERT_TRACE=1: which experts every token row selected, per layer and step) and the native
loop's speculation records, which `python -m vxpu.speculation experts TRACE EVENTS` evaluates
offline for the expert-aware policy.

usage: spec-trace.py LABEL ARTIFACT ROWS PROMPTS_FILE [NEW_TOKENS] [OUT_DIR]"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

label, artifact, rows_wanted, prompts_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
new_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 64
out_dir = sys.argv[6] if len(sys.argv) > 6 else os.path.dirname(os.path.abspath(__file__))

from vxpu.engine import Engine, SamplingParams  # noqa: E402

batch = json.load(open(os.path.join(artifact, "binding.json")))["batch"]
rows = min(rows_wanted, batch)
greedy = SamplingParams(do_sample=False)
t0 = time.time()
engine = Engine(artifact, device="cuda", compile_decode=True)
load_s = time.time() - t0
texts = [p.strip() for p in open(prompts_file) if p.strip()][:rows]
ids = [engine.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)["input_ids"][0].tolist() for t in texts]
engine.generate_rows(ids[:1], 4, greedy)
t_warm = time.time()
while time.time() - t_warm < 4.0:
    engine.generate_rows(ids[:1], 16, greedy)

# the drafter graph alone: its replay time with whatever its input buffers hold (the bytes are the same)
draft_ms = None
draft = getattr(engine, "_draft_run", None)
if draft is not None and hasattr(draft, "native_parts"):
    inputs = draft.native_parts()[0]
    with torch.no_grad():
        for _ in range(3):
            draft(**inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            draft(**inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    draft_ms = round(statistics.median(times), 2)

engine.expert_trace(reset=True)
stats0 = engine.expert_stats() or {}
t = time.time()
out = engine.generate_rows(ids, new_tokens, greedy)
wall = time.time() - t
stats1 = engine.expert_stats() or {}
step_ms = out["step_ms"]
steps = len(step_ms)
distinct = {layer: (stats1.get(layer, 0) - stats0.get(layer, 0)) / steps for layer in stats1} if steps else {}
gen = sum(len(x) for x in out["token_ids"])
rec = {"label": label, "artifact": os.path.basename(artifact), "rows": rows, "prompts": prompts_file,
       "step_tokens": engine.step_tokens, "load_s": round(load_s, 1), "steps": steps, "tokens": gen,
       "median_step_ms": round(statistics.median(step_ms), 2) if step_ms else None,
       "step_ms_p10_p90": [round(np.percentile(step_ms, 10), 1), round(np.percentile(step_ms, 90), 1)] if step_ms else None,
       "tok_s": round(gen / sum(step_ms) * 1e3, 1) if step_ms else None, "wall_tok_s": round(gen / wall, 1),
       "drafted": out.get("drafted", 0), "accepted": out.get("accepted", 0),
       "acceptance": round(out["accepted"] / out["drafted"], 3) if out.get("drafted") else None,
       "tokens_per_step": round(gen / steps, 3) if steps else None,
       "distinct_experts_mean": round(sum(distinct.values()) / len(distinct), 1) if distinct else None,
       "distinct_experts": {str(k): round(v, 1) for k, v in sorted(distinct.items())},
       "draft_graph_ms": draft_ms, "speculation": out.get("speculation")}
print("RESULT13 " + json.dumps(rec), flush=True)
trace = engine.expert_trace(reset=True)
if trace and trace[0]:
    np.savez_compressed(os.path.join(out_dir, f"{label}-trace.npz"), **{f"layer{layer}": arr for layer, arr in trace[0].items()})
events = getattr(engine, "_last_speculation_events", None)
json.dump({"rows": rows, "step_tokens": engine.step_tokens, "steps": steps, "events": events,
           "token_ids": out["token_ids"]}, open(os.path.join(out_dir, f"{label}-events.json"), "w"))
