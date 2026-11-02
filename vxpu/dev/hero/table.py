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

"""The hero table from batches.py's RESULT lines: one row per (label, mode, batches) group, with the
per-batch latency, the aggregate tokens per second, the floor and the measured share of it.

    python table.py results.jsonl [more.jsonl ...] [--predictions predictions.json] > table.md

The predicted column comes from `vxpu plan matrix` (the planner's cost models: the pipeline's and
the speculative step's), kept in predictions.json next to this script with the command that produced
each number; the last column says whether the planner chose that shape for its load (hero run 10's
acceptance: the choice within about 20% of the best measured shape at every load)."""
import json
import os
import sys
from collections import OrderedDict


def load(paths):
    groups = OrderedDict()
    for path in paths:
        for line in open(path):
            if not line.startswith("RESULT "):
                continue
            r = json.loads(line[7:])
            key = (r["label"], r["mode"], r["batches"])
            groups.setdefault(key, []).append(r)
    return groups


def fmt_ms(r):
    p = r.get("step_ms_p10_p90") or [None, None]
    return f"{r['median_step_ms']} ms ({p[0]}/{p[1]})"


def predictions(path):
    """predictions.json's cases by (rows a batch, mode, batches, drafts a step, parallel); the hero 9 file
    had no draft field (its drafter cases are keyed by a '+drafter' mode suffix), the hero 10 file has
    one, the hero 14 file names the parallelism ('layers' when absent; 'tp4-ep4' for the dense layers
    split across the cards)."""
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path))
    return {(c["rows"], c["mode"].replace("+drafter", ""), c["batches"], c.get("draft", 4 if "+drafter" in c["mode"] else 0),
             c.get("parallel") or "layers"): c
            for c in doc.get("cases", [])}


def main():
    args = sys.argv[1:]
    pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.json")
    if "--predictions" in args:
        i = args.index("--predictions"); pred_path = args[i + 1]; del args[i:i + 2]
    pred = predictions(pred_path)
    groups = load(args)
    print("| shape | mode | batches | per-batch step, median (p10/p90) | tokens a step per row | aggregate tok/s: measured / predicted / ceiling | floor: single batch / pipeline slot | measured ÷ floor | rows identical | tokens vs reference | the planner's choice |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (label, mode, batches), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["batch"])
        first = rs[0]
        shape = f"{first['rows']} rows" + (f" + drafter ({first['step_tokens']} tokens a step)" if first["step_tokens"] > 1 else "")
        latencies = ", ".join(fmt_ms(r) for r in rs) if batches > 1 else fmt_ms(first)
        agg = first.get("aggregate_tokens_per_s")
        single_floor_ms = first.get("floor_single_ms")
        pipe_floor_ms = first.get("floor_pipeline_slot_ms", first.get("floor_pipeline_ms"))
        key = (first["rows"], mode, batches, first["step_tokens"] - 1, first.get("parallel") or "layers")
        p = pred.get(key, {})
        choice = "chosen" if p.get("planner_chooses") else ("rejected" if p else "")
        predicted = f"{agg} / {p.get('predicted_aggregate_tokens_per_s', '-')} / {first.get('aggregate_floor_tokens_per_s', '-')}"
        if batches == 1:
            share = f"{single_floor_ms / first['median_step_ms']:.0%} of bytes" if first.get("median_step_ms") and single_floor_ms else ""
        else:
            share = f"{agg / first['aggregate_floor_tokens_per_s']:.0%} of the aggregate floor" if first.get("aggregate_floor_tokens_per_s") and agg else ""
        if p.get("predicted_latency_ms"):
            latencies += f"; predicted {p['predicted_latency_ms']} ms"
        identical = ", ".join(str(r.get("same_rows")) for r in rs) if mode == "same" else ""
        ref = ", ".join(f"{r['match']}/{r['of']}" if r.get("of") else "no reference" for r in rs)
        print(f"| {shape} | {mode} | {batches} ({first['strategy']}) | {latencies} | {first['tokens_per_row_step']} | "
              f"{predicted} | {single_floor_ms} ms / {pipe_floor_ms} ms ({first.get('floor_bytes_gb')} GB; "
              f"{first.get('floor_distinct_per_layer')} experts a layer, {first.get('floor_distinct_source')}) | {share} | {identical} | {ref} | {choice} |")


if __name__ == "__main__":
    main()
