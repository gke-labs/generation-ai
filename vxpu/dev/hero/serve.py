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

"""The league-table instrument: N concurrent streams in a closed loop, measured as the published
serving benchmarks measure (hero run 15).

Each of N streams sends a prompt, receives `--new-tokens` tokens, and at once sends its next prompt,
for `--rounds` rounds; the prompts come from a prose set, every stream a different prompt. The
engine runs them through the same entry the pipeline uses (generate_rows with batches in flight
and continuous admission: a stream's next prompt takes the row its last one freed), under whatever
layout the plan gives the engine (layer ranges, tensor parallelism, the drafter's GPU); this
driver drives the engine, not the layout. N is rows times batches of the artifact in use (one
stream per row), so a 16-stream run on an 8-row artifact is two batches in flight.

    python serve.py --artifact DIR --label L --streams N [--batches B] [--rounds 4]
                    [--new-tokens 256] [--prompts FILE] [--out results.jsonl]

The report is one `RESULT15 {json}` line: the steady-state window (from the moment every stream
has finished one request to the moment the first stream runs out of prompts, so N streams are in
flight throughout), the tokens emitted inside it and the aggregate tokens per second over it; per
stream the decode tokens per second (median, p10, p90 over the requests inside the window), the
time to first token (admission to first token: the prefill and the hand-off; the first round's
queueing wait separately), and the prefill time; the engine's step statistics per batch; the
drafter's counts and tokens a step; the distinct experts a layer. The same summary goes into the
kernel record as the statistic `serving`, so the planner's predictions are scored against it
(`vxpu plan serve` predicts the same shapes). The published figures the acceptance names are in
table.py, beside these lines."""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batches import load_prompts  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--streams", type=int, required=True, help="concurrent requests: rows x batches of the artifact")
    p.add_argument("--batches", type=int, default=None, help="batches in flight (default: streams / rows)")
    p.add_argument("--rounds", type=int, default=4, help="requests per stream; the window spans rounds 2 to rounds-1")
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts-132.txt"))
    p.add_argument("--out", default=None, help="results.jsonl to append the RESULT15 line to")
    p.add_argument("--warmup-seconds", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--synthetic", type=int, default=0,
                   help="use this many synthetic token prompts of 9 to 24 tokens instead of --prompts (the tiny check)")
    p.add_argument("--window", choices=("steady", "all"), default="steady",
                   help="steady: N streams in flight throughout; all: first admission to last token")
    return p.parse_args()


def percentiles(values, keys=(0.5, 0.1, 0.9)):
    if not values:
        return {k: None for k in ("median", "p10", "p90")}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"median": round(statistics.median(s), 1), "p10": round(pick(0.1), 1), "p90": round(pick(0.9), 1)}


def window_of(requests, streams, mode):
    """The steady-state window: starts when `streams` requests have finished (every stream has
    completed its first and is on its second) and ends when `len(requests) - streams + 1` have
    (the first stream has no next prompt, so fewer than `streams` are in flight after it)."""
    done = sorted(r.done_at for r in requests if r.done_at is not None)
    first = min(r.admit_wall for r in requests if r.admit_wall is not None)
    last = max(done) if done else first
    if mode == "all" or len(done) < streams + 1:
        return first, last, "all"
    start = done[streams - 1]
    end_index = len(requests) - streams  # the (M - N + 1)-th finish
    end = done[end_index] if end_index < len(done) else last
    if end <= start:
        return first, last, "all"
    return start, end, "steady"


def main():
    args = parse_args()
    from vxpu.engine import Engine, SamplingParams, _Request
    t0 = time.time()
    engine = Engine(args.artifact, device=args.device, compile_decode=args.device != "cpu")
    load_s = time.time() - t0
    rows = json.load(open(os.path.join(args.artifact, "binding.json")))["batch"]
    batches = args.batches or max(1, args.streams // rows)
    if rows * batches != args.streams:
        sys.exit(f"serve.py: {args.streams} streams need rows x batches = {rows} x {batches}; export an artifact with "
                 f"--batch {args.streams // batches} or choose --batches")
    greedy = SamplingParams(do_sample=False)
    if args.synthetic:
        ids = [[(11 * i + 7 * j + 5) % 30000 + 100 for j in range(9 + (i * 5) % 16)] for i in range(args.synthetic)]
    else:
        ids = load_prompts(engine, args.prompts)[:-1]  # the real prompts, not the profilers' synthetic one
    total = args.streams * args.rounds
    prompts = [ids[i % len(ids)] for i in range(total)]
    batch_of = [i % batches for i in range(total)]  # stream i's requests all live in batch i % batches

    warm = time.time()
    while time.time() - warm < args.warmup_seconds:
        engine.generate_rows(ids[:1], 64, greedy)

    stats = engine if hasattr(engine, "expert_stats") else None
    counts0 = stats.expert_stats() if stats is not None else None
    # the requests themselves, so their timestamps survive: generate_rows builds its own, so wrap it
    made = []
    original = _Request.__init__

    def tracking(self, *a, **k):
        original(self, *a, **k)
        made.append(self)

    _Request.__init__ = tracking
    started = time.perf_counter()
    try:
        if batches > 1:
            out = engine.generate_rows(prompts, args.new_tokens, greedy, batches=batches, batch_of=batch_of)
        else:
            out = engine.generate_rows(prompts, args.new_tokens, greedy)
    finally:
        _Request.__init__ = original
    ended = time.perf_counter()
    requests = [r for r in made if r.shadow_of is None and r.prompt in prompts][:total]
    if len(requests) != total:
        requests = made[:total]
    counts1 = stats.expert_stats() if counts0 is not None else None

    start, end, kind = window_of(requests, args.streams, args.window)
    in_window = [r for r in requests if r.first_token_at is not None and r.done_at is not None
                 and r.admit_wall >= start - 1e-9 and r.done_at <= end + 1e-9]
    window_tokens = sum(sum(1 for t in r.token_times if start <= t <= end) for r in requests)
    per_stream = [len(r.token_ids) / (r.done_at - r.first_token_at) for r in in_window
                  if r.done_at > r.first_token_at and len(r.token_ids) > 1]
    ttft = [(r.first_token_at - r.admit_wall) * 1e3 for r in requests if r.first_token_at and r.admit_wall]
    first_round = requests[:args.streams]
    queue_wait = [(r.first_token_at - started) * 1e3 for r in first_round if r.first_token_at]
    prefill = [r.prefill_ms for r in requests if r.prefill_ms]
    generated = sum(len(r.token_ids) for r in requests)

    step_ms = out.get("batch_step_ms") or [out["step_ms"]]
    steps = [{"median_ms": round(statistics.median(s), 1) if s else None, "steps": len(s),
              "p10_p90": [round(sorted(s)[len(s) // 10]), round(sorted(s)[len(s) * 9 // 10])] if s else None}
             for s in step_ms]
    distinct = None
    if counts0 is not None and counts1 is not None:
        layers = [l for l in counts1 if l < 40] or list(counts1)
        n_steps = max(1, sum(len(s) for s in step_ms))
        distinct = round(sum(counts1.get(l, 0) - counts0.get(l, 0) for l in layers) / n_steps / max(1, len(layers)), 1)
    tokens_per_step = None
    if out.get("drafted"):
        tokens_per_step = round(generated / max(1, sum(len(s) for s in step_ms) * rows), 2)

    rec = {
        "label": args.label, "artifact": os.path.basename(args.artifact.rstrip("/")), "streams": args.streams,
        "rows": rows, "batches": batches, "rounds": args.rounds, "new_tokens": args.new_tokens,
        "requests": len(requests), "generated_tokens": generated, "step_tokens": engine.step_tokens,
        "load_s": round(load_s, 1), "wall_s": round(ended - started, 1),
        "window": kind, "window_s": round(end - start, 2), "window_tokens": window_tokens,
        "aggregate_tokens_per_s": round(window_tokens / (end - start), 1) if end > start else None,
        "requests_in_window": len(in_window),
        "per_stream_tokens_per_s": percentiles(per_stream),
        "ttft_ms": percentiles(ttft), "first_round_queue_ms": percentiles(queue_wait),
        "prefill_ms": percentiles(prefill), "prefill_s_total": round(sum(prefill) / 1e3, 1),
        "steps": steps, "drafted": out.get("drafted", 0), "accepted": out.get("accepted", 0),
        "tokens_per_row_step": tokens_per_step, "distinct_experts_per_layer": distinct,
        "end_to_end_tokens_per_s": round(generated / (ended - started), 1),
        "finish_reasons": {k: sum(1 for r in requests if r.finish_reason == k) for k in ("length", "eos", "error", None)},
    }
    try:  # the record carries the serving measurement, so the planner's prediction is scored against it
        engine.plan.stat("serving", {k: rec[k] for k in ("streams", "rows", "batches", "new_tokens", "window",
                                                         "window_s", "aggregate_tokens_per_s",
                                                         "per_stream_tokens_per_s", "ttft_ms", "prefill_ms",
                                                         "tokens_per_row_step", "distinct_experts_per_layer")})
    except Exception as e:  # noqa: BLE001
        rec["record_error"] = str(e)
    line = "RESULT15 " + json.dumps(rec)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
