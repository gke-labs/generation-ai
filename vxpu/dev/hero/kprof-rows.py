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

"""Hero run 13: the decode step's GPU time by kernel family at ROWS rows of real prompts (plain or
with the drafter), averaged over the profiled steps after the first: the expert kernels, the GEMV
templates, the attention and front kernels, the rest. The split the speculation model's terms
need (what a verify token row costs in each family).

usage: kprof-rows.py LABEL ARTIFACT ROWS PROMPTS_FILE [steps]"""
import collections
import json
import os
import statistics
import sys

import torch

label, artifact, rows_wanted, prompts_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 12
from vxpu.engine import Engine, SamplingParams  # noqa: E402

batch = json.load(open(os.path.join(artifact, "binding.json")))["batch"]
rows = min(rows_wanted, batch)
engine = Engine(artifact, device="cuda", compile_decode=True)
greedy = SamplingParams(do_sample=False)
texts = [p.strip() for p in open(prompts_file) if p.strip()][:rows]
ids = [engine.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)["input_ids"][0].tolist() for t in texts]
engine.generate_rows(ids, 4, greedy)  # warm-up: the graphs captured at this row count
from torch.profiler import profile, ProfilerActivity  # noqa: E402
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    out = engine.generate_rows(ids, steps, greedy)
events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
events.sort(key=lambda e: e.time_range.start)
step_ms = out["step_ms"]
n = len(step_ms)
median = statistics.median(step_ms)
# the decode steps after the first: the last (n - 1) steps' worth of the window, by time
end = events[-1].time_range.end
window_us = sum(step_ms[1:]) * 1e3
start = end - window_us
decode = [e for e in events if e.time_range.start >= start]
FAMILIES = [("experts", ("_fp4_", "moe", "expert")), ("gemv", ("_gemv_", "_dual_gemv", "gemv")),
            ("attention", ("sparse_attn", "attn", "window_kv", "rotary")),
            ("front", ("_front_", "ffn_front", "row_post", "swiglu", "x_mxfp8", "act_quant")),
            ("copies", ("Memcpy", "memcpy", "Memset")), ("topk-sort", ("topk", "TopK", "sort", "Sort", "radix"))]


def family(name):
    for fam, needles in FAMILIES:
        if any(nd in name for nd in needles):
            return fam
    return "other"


by = collections.defaultdict(lambda: [0, 0.0])
for e in decode:
    f = family(e.name)
    by[f][0] += 1
    by[f][1] += e.time_range.elapsed_us()
per_step = n - 1 if n > 1 else 1
total_us = sum(v[1] for v in by.values()) / per_step
rec = {"label": label, "artifact": os.path.basename(artifact), "rows": rows, "step_tokens": engine.step_tokens,
       "steps_profiled": n, "median_step_ms": round(median, 2), "gpu_us_per_step": round(total_us),
       "kernels_per_step": round(sum(v[0] for v in by.values()) / per_step),
       "families": {f: {"kernels": round(v[0] / per_step, 1), "us": round(v[1] / per_step)} for f, v in sorted(by.items(), key=lambda kv: -kv[1][1])},
       "drafted": out.get("drafted", 0), "accepted": out.get("accepted", 0)}
print("KPROFROWS " + json.dumps(rec), flush=True)
names = collections.defaultdict(lambda: [0, 0.0])
for e in decode:
    names[e.name[:70]][0] += 1
    names[e.name[:70]][1] += e.time_range.elapsed_us()
print("KPROFROWS by kernel (count a step, us a step), top 25:")
for name, (c, us) in sorted(names.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"  {us / per_step:8.0f} us {c / per_step:7.1f}x  {name}")
