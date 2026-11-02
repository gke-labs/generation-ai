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

"""The league table from serve.py's RESULT15 lines (hero run 15): our closed-loop serving numbers
at 1, 8 and 16 streams beside the published ones on the same cards and the planner's prediction.

    python league.py serve.jsonl [more.jsonl ...] [--predictions predicted.json] > league.md

The published figures (reported, not measured by us; community RTX PRO 6000 cards at 250 W,
ours 600 W; decode is memory-bound so the comparison holds): the vLLM fork "Karmic Kraken" on
four cards, TP4 with DSpark 7 drafts and adaptive verification, prose prompts: 247 to 252 tok/s
at one stream, 814 to 827 at eight, 1,150 at sixteen (code prompts 494 / 1,188 / 1,669); SGLang on
the same host, TP4: 112 at one stream, 529 at eight. --predictions takes {label: tok/s} from
`vxpu plan serve` runs, one per measured label."""
import argparse
import json
import sys

PUBLISHED = {  # streams -> (stack, aggregate tok/s, note)
    1: [("vLLM fork TP4 + DSpark", 250, "prose; 494 code"), ("SGLang TP4", 112, "prose; 191 code")],
    8: [("vLLM fork TP4 + DSpark", 820, "prose; 1,188 code"), ("SGLang TP4", 529, "prose")],
    16: [("vLLM fork TP4 + DSpark", 1150, "prose; 1,669 code")],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--predictions", default=None, help="{label: predicted aggregate tok/s}")
    args = p.parse_args()
    pred = json.load(open(args.predictions)) if args.predictions else {}
    rows = []
    for path in args.results:
        for line in open(path):
            if line.startswith("RESULT15 "):
                rows.append(json.loads(line[9:]))
    rows.sort(key=lambda r: (r["streams"], r["step_tokens"], r["label"]))
    print("| streams | ours: shape | drafter | aggregate tok/s (steady window) | per stream | TTFT ms (median/p90) | "
          "first prefill ms | step ms per batch | predicted | published |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        shape = f"{r['batches']} x {r['rows']} rows ({r['artifact']})"
        drafter = f"{r['step_tokens'] - 1} drafts, {r['tokens_per_row_step']} tok/step" if r["step_tokens"] > 1 else "no"
        ttft = f"{r['ttft_ms']['median']} / {r['ttft_ms']['p90']}"
        steps = ", ".join(str(s["median_ms"]) for s in r["steps"])
        pub = ""
        if r["streams"] in PUBLISHED and r["streams"] not in seen:
            pub = "; ".join(f"{name} {value}" for name, value, _ in PUBLISHED[r["streams"]])
            seen.add(r["streams"])
        print(f"| {r['streams']} | {shape} | {drafter} | {r['aggregate_tokens_per_s']} ({r['window']}, {r['window_s']} s) | "
              f"{r['per_stream_tokens_per_s']['median']} | {ttft} | {r['prefill_ms']['median']} | {steps} | "
              f"{pred.get(r['label'], '')} | {pub} |")
    if not rows:
        print("no RESULT15 lines", file=sys.stderr)


if __name__ == "__main__":
    main()
