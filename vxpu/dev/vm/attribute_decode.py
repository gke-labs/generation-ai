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

"""Where a split decode step's time goes, by segment.

    python attribute_decode.py --artifact /mnt/ssd/artifact [--device cuda] [--steps 16]

Loads the engine with the decode graph split at its host ops and
compiled (segments.py), warms it up over several sessions (CUDA-graph
recording settles over the first two), and reports:

  clean        decode ms/token with no instrumentation (median and best of
               several sessions; a single session can catch a re-record)
  by kind      time per decode step in dense segments, in the expert host
               op, in the n-gram host op (each segment call timed with a
               device synchronize around it, so the parts sum to a little
               more than the clean step)
  experts      inside the expert op: the routing body (the CPU kernels and
               their glue) against the op's total (the rest is the device
               crossings and dispatch)
  segments     the heaviest dense segments with their node counts, most
               frequent ops and source modules

The prompt is the first prompt of --ids-from (a vxpu-vectors file) or
--ids (comma-separated token ids). Results also go to --out as JSON.
"""

import argparse
import collections
import json
import os
import re
import time

import torch

from vxpu import experts_cpu, segments
from vxpu.engine import Engine, SamplingParams

LAYER = re.compile(r"layers\.\d+")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--artifact", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ids-from", help="a vxpu-vectors file whose first prompt is used")
    p.add_argument("--ids", help="comma-separated token ids")
    p.add_argument("--steps", type=int, default=16, help="decode steps per measured session")
    p.add_argument("--warmup-sessions", type=int, default=3)
    p.add_argument("--clean-sessions", type=int, default=5)
    p.add_argument("--top", type=int, default=8, help="dense segments to list")
    p.add_argument("--out", default="attribute_decode.json")
    args = p.parse_args()

    if args.ids:
        ids = [int(t) for t in args.ids.split(",")]
    elif args.ids_from:
        ids = json.load(open(args.ids_from))["prompts"][0]["input_ids"]
    else:
        raise SystemExit("--ids or --ids-from is required")

    t0 = time.time()
    say = lambda m: print(f"[attribute {time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    os.environ.setdefault("VXPU_COMPILE_MODE", "reduce-overhead")
    say(f"loading {args.artifact} on {args.device}")
    eng = Engine(args.artifact, device=args.device, compile_decode=True)
    if eng._decode_split is None:
        raise SystemExit("the decode graph is not split (VXPU_SPLIT_DECODE=0?)")
    say("loaded")

    def session(n):
        sid = eng.new_session()
        stats = None
        for ev in eng.generate(sid, ids, n, SamplingParams(do_sample=False)):
            if ev.get("done"):
                stats = ev
        return stats["ms_per_token"]

    for i in range(args.warmup_sessions):
        say(f"warm-up session {i + 1}: {session(8):.1f} ms/token")
    # several clean sessions: a single one can catch a CUDA-graph re-record
    # or a collector pause, so report the median and the best
    clean = sorted(session(args.steps) for _ in range(args.clean_sessions))
    res = {"clean_ms_per_token": clean[len(clean) // 2], "clean_best_ms_per_token": clean[0],
           "clean_sessions_ms_per_token": clean}
    say(f"clean: median {res['clean_ms_per_token']:.1f} ms/token, best {clean[0]:.1f} "
        f"(sessions: {', '.join(f'{c:.1f}' for c in clean)})")

    # instrument every segment of the split module
    split = eng._decode_split.split
    kind, original = {}, {}
    for name, sub in segments.segments_of(split):
        targets = {str(n.target) for n in getattr(sub, "_orig_mod", sub).graph.nodes if n.op == "call_function"}
        kind[name] = ("experts" if "vxpu.moe_experts.default" in targets
                      else "engram" if "vxpu.engram_rows.default" in targets else "dense")
        original[name] = sub
    seg_time, seg_calls = collections.Counter(), collections.Counter()
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)

    class Timed(torch.nn.Module):
        def __init__(self, name, sub):
            super().__init__()
            self.name, self.sub = name, sub

        def forward(self, *a, **k):
            sync()
            t = time.perf_counter()
            out = self.sub(*a, **k)
            sync()
            seg_time[self.name] += time.perf_counter() - t
            seg_calls[self.name] += 1
            return out

    for name, sub in original.items():
        split._modules[name] = Timed(name, sub)
    body = collections.Counter()
    routing = experts_cpu.forward_batched

    def timed_routing(*a):
        t = time.perf_counter()
        y = routing(*a)
        body["routing body"] += time.perf_counter() - t
        return y

    experts_cpu.forward_batched = timed_routing
    try:
        res["instrumented_ms_per_token"] = session(args.steps)
    finally:
        experts_cpu.forward_batched = routing
        for name, sub in original.items():
            split._modules[name] = sub

    steps = args.steps - 1  # the first token comes from the prefill
    by_kind = collections.Counter()
    for name, t in seg_time.items():
        by_kind[kind[name]] += t * 1e3 / steps
    res["per_step_ms"] = dict(by_kind)
    res["experts_routing_body_ms"] = body["routing body"] * 1e3 / steps
    say("per decode step: " + ", ".join(f"{k} {v:.1f} ms" for k, v in by_kind.most_common())
        + f" (instrumented step {res['instrumented_ms_per_token']:.1f} ms)")
    say(f"experts: routing body {res['experts_routing_body_ms']:.1f} ms of {by_kind['experts']:.1f} ms in the op")

    dense = sorted(((seg_time[n] * 1e3 / max(1, seg_calls[n]), n) for n in seg_time if kind[n] == "dense"), reverse=True)
    res["dense_segments"] = []
    say(f"dense segments: {len(dense)}; heaviest:")
    for ms, name in dense[:args.top]:
        graph = getattr(original[name], "_orig_mod", original[name]).graph
        ops = collections.Counter(str(n.target).replace("aten.", "").split(".")[0]
                                  for n in graph.nodes if n.op == "call_function")
        mods = collections.Counter()
        for n in graph.nodes:
            stack = n.meta.get("nn_module_stack") or {}
            paths = [v[0] if isinstance(v, (tuple, list)) else str(v) for v in stack.values()]
            if paths:
                mods[LAYER.sub("layers.N", paths[-1])] += 1
        res["dense_segments"].append({"segment": name, "ms": ms, "nodes": sum(ops.values()),
                                      "ops": ops.most_common(8), "modules": mods.most_common(4)})
        say(f"  {name}: {ms:.2f} ms, {sum(ops.values())} nodes; ops {ops.most_common(5)}; modules {mods.most_common(3)}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {args.out}")


if __name__ == "__main__":
    main()
