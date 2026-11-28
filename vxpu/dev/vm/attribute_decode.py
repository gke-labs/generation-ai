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
               op, in the n-gram host op, from one session with the
               engine's telemetry (telemetry.py: CUDA events between
               segments, no added synchronization)
  experts      inside the expert op: the selected implementation's own
               time, the wait for the GPU plus the copy in, the copy out
  segments     the heaviest dense segments with their node counts, most
               frequent ops and source modules

The prompt is the first prompt of --ids-from (a vxpu-vectors file) or
--ids (comma-separated token ids). Results also go to --out as JSON,
with the session's full telemetry and the engine's plan.
"""

import argparse
import collections
import json
import os
import re
import time

from vxpu import segments, vxpu_pb2
from vxpu.engine import Engine, SamplingParams

LAYER = re.compile(r"layers\.\d+")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--artifact", required=True)
    p.add_argument("--plan", required=True, help="path to binary PlanFile (.pb)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ids-from", help="a vxpu-vectors file whose first prompt is used")
    p.add_argument("--ids", help="comma-separated token ids")
    p.add_argument("--steps", type=int, default=16, help="tokens generated per measured session")
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

    from vxpu import physplan
    pf = physplan.read_plan_file(args.plan)

    t0 = time.time()
    say = lambda m: print(f"[attribute {time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    say(f"loading {args.artifact} on {args.device} with plan {args.plan}")
    eng = Engine(args.artifact, device=args.device, plan=pf)
    if eng._decode_split is None:
        raise SystemExit("the decode graph is not split")
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

    # one session with the engine's own telemetry (telemetry.py): CUDA
    # events between segments, read back in batches, so the measurement
    # adds no synchronization of its own
    sid = eng.new_session()
    done = [ev for ev in eng.generate(sid, ids, args.steps, SamplingParams(do_sample=False), telemetry=vxpu_pb2.TELEMETRY_SUMMARY)
            if ev.get("done")][0]
    tel = done["telemetry"]
    res["telemetry"] = tel
    res["instrumented_ms_per_token"] = done["ms_per_token"]
    res["decode_steps"] = tel.get("segment_steps", tel["decode_steps"])
    by_kind = tel.get("by_kind_ms_per_step", {})
    res["per_step_ms"] = by_kind
    say("per decode step: " + ", ".join(f"{k} {v:.1f} ms" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]))
        + f" (instrumented step {res['instrumented_ms_per_token']:.1f} ms)")
    moe = tel["host_ops"].get("decode", {}).get("moe_experts")
    impls = ", ".join(tel["plan"]["selected"].get("vxpu::moe_experts", [])) or "?"
    res["experts_impl_ms"] = moe["impl_ms"] if moe else None
    if moe:
        say(f"experts: {moe['impl_ms']:.1f} ms in the implementation ({impls}), {moe['wait_to_host_ms']:.1f} ms "
            f"waiting for the GPU and copying in, {moe['to_device_ms']:.1f} ms copying out, of "
            f"{by_kind.get('experts', 0):.1f} ms in the op")

    names = {name: sub for name, sub in segments.segments_of(eng._decode_split.split)}
    dense = sorted((s for s in tel.get("segments", []) if s["kind"] == "dense"), key=lambda s: -s["ms_per_step"])
    res["dense_segments"] = []
    say(f"dense segments: {len(dense)}; heaviest:")
    for seg in dense[:args.top]:
        sub = names[seg["name"]]
        graph = getattr(sub, "_orig_mod", sub).graph
        ops = collections.Counter(str(n.target).replace("aten.", "").split(".")[0]
                                  for n in graph.nodes if n.op == "call_function")
        mods = collections.Counter()
        for n in graph.nodes:
            stack = n.meta.get("nn_module_stack") or {}
            paths = [v[0] if isinstance(v, (tuple, list)) else str(v) for v in stack.values()]
            if paths:
                mods[LAYER.sub("layers.N", paths[-1])] += 1
        res["dense_segments"].append({"segment": seg["name"], "ms": seg["ms_per_step"], "layers": seg["layers"],
                                      "nodes": seg["nodes"], "ops": ops.most_common(8), "modules": mods.most_common(4)})
        say(f"  {seg['name']} (layers {seg['layers']}): {seg['ms_per_step']:.2f} ms, {seg['nodes']} nodes; "
            f"ops {ops.most_common(5)}; modules {mods.most_common(3)}")

    res["plan"] = eng.explain()  # the physical plan these numbers measured (plan.py)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {args.out}")


if __name__ == "__main__":
    main()
