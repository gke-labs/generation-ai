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

"""The league benchmark harness (hero run 15).

Drives the engine's generate path using the sparkDash v1.8.9 benchmark method:
- Prose: fixed prompt; at concurrency c > 1, stream i appends ' (stream i/c)'.
- Code: 32 distinct Python tasks (stream i gets task i; c = 1 is binary_search).
- Requests: chat template with thinking off, temperature 0, top_p 1, max_tokens 400
  and min_tokens 400 with EOS ignored, no stop strings, streamed with usage.
  One 32-token warm-up request per job before the measured levels.
- Levels 1, 2, 4, 8, 16 run sequentially; each level is one wave with all streams started together.
- Per stream: tok/s = (completion_tokens - 1) / (client time of the last token - client time of the first token).
  Aggregate for the wave = sum over streams of (completion_tokens - 1) / (latest last-token time - earliest first-token time).
  Prefill and the first token are excluded.

    python league.py --artifact DIR [--type prose|code|all] [--levels 1,2,4,8,16] [--out results.jsonl]
"""

import argparse
import json
import os
import time

PROSE_BASE = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)
DEFAULT_LEVELS = [1, 2, 4, 8, 16]


def stream_tokens_per_s(completion_tokens: int, first_token_time: float, last_token_time: float) -> float:
    """Computes per-stream token rate in tokens per second.

    The formula excludes prefill and the first token: (completion_tokens - 1) / (last_token_time - first_token_time).
    """
    if completion_tokens <= 1 or last_token_time <= first_token_time:
        return 0.0
    return (completion_tokens - 1) / (last_token_time - first_token_time)


def aggregate_tokens_per_s(streams_metrics: list[tuple[int, float, float]]) -> float:
    """Computes aggregate token rate for a wave in tokens per second.

    The formula sums completion tokens minus one across streams and divides by the wave window.
    The wave window spans from the earliest first token to the latest last token.
    """
    valid = [(n, t0, t1) for (n, t0, t1) in streams_metrics if n > 1 and t1 > t0]
    if not valid:
        return 0.0
    total_tokens = sum(n - 1 for (n, _, _) in valid)
    span = max(t1 for (_, _, t1) in valid) - min(t0 for (_, t0, _) in valid)
    return total_tokens / span if span > 0 else 0.0


def build_prose_prompt(stream_index: int, concurrency: int) -> str:
    """Constructs the prose prompt for a stream at a given concurrency.

    Stream 0 at concurrency 1 gets the base prompt. Streams at higher concurrency append their stream index.
    """
    if concurrency <= 1:
        return PROSE_BASE
    return f"{PROSE_BASE} (stream {stream_index + 1}/{concurrency})"


def load_league_prompts(path: str) -> list[dict]:
    """Loads benchmark prompts from a JSON array or JSON lines file."""
    with open(path) as f:
        content = f.read().strip()
    return json.loads(content) if content.startswith("[") else [json.loads(line) for line in content.splitlines() if line.strip()]


def build_prompts(prompts_data: list[dict], bench_type: str, concurrency: int) -> list[str]:
    """Constructs the list of prompt strings for all streams at the given concurrency level."""
    if bench_type == "prose":
        return [build_prose_prompt(i, concurrency) for i in range(concurrency)]
    if bench_type == "code":
        by_stream = {it["stream"]: it["prompt"] for it in prompts_data if it.get("type") == "code" and it.get("concurrency") == concurrency}
        if len(by_stream) < concurrency:
            task_prompts = {it["stream"]: it["prompt"] for it in prompts_data if it.get("type") == "code" and "stream" in it}
            for i in range(concurrency):
                if i not in by_stream and i in task_prompts:
                    by_stream[i] = task_prompts[i]
        return [by_stream[i] for i in range(concurrency)]
    raise ValueError(f"Unknown benchmark type: {bench_type}")


def get_warmup_prompt(prompts_data: list[dict], bench_type: str) -> str:
    """Returns the warmup prompt for the benchmark type."""
    for item in prompts_data:
        if item.get("type") == bench_type and item.get("max_tokens") == 32:
            return item["prompt"]
    if bench_type == "prose":
        return PROSE_BASE
    return ("warmup_noop\ndef warmup_noop(x): return x unchanged.\n"
            "Output only Python source. No comments, no docstrings, no markdown fences. "
            "Then add tests and the helpers this needs. Keep writing code.")


def encode_prompt(tokenizer, text: str) -> list[int]:
    """Applies chat template formatting with thinking disabled and tokenizes the text."""
    if tokenizer is None:
        return [(11 * j + 5) % 30000 + 100 for j in range(32)]
    messages = [{"role": "user", "content": text}]
    try:
        res = tokenizer.apply_chat_template(messages, add_generation_prompt=True, thinking=False,
                                            return_tensors="pt", return_dict=True)
    except (TypeError, Exception):
        res = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
    return res["input_ids"][0].tolist()


def run_level(engine, level_prompts: list[str], max_tokens: int = 400) -> tuple[dict, list[tuple[int, float, float]]]:
    """Runs one wave of streams started together through the engine's generate path."""
    try:
        from vxpu.engine import SamplingParams, _Request
        greedy = SamplingParams(do_sample=False, temperature=0.0, top_p=1.0)
    except (ImportError, ModuleNotFoundError):
        greedy = _Request = None

    c = len(level_prompts)
    tokenized = [encode_prompt(getattr(engine, "tokenizer", None), p) for p in level_prompts]
    made, original_init = [], None
    if _Request is not None:
        original_init = _Request.__init__

        def tracking(self, *a, **k):
            original_init(self, *a, **k)
            made.append(self)

        _Request.__init__ = tracking

    t_start = time.perf_counter()
    try:
        out = engine.generate_rows(tokenized, max_tokens, greedy)
    finally:
        if original_init is not None:
            _Request.__init__ = original_init
    t_end = time.perf_counter()

    streams_metrics = []
    requests = [r for r in made if r.prompt in tokenized][:c]
    if len(requests) == c:
        for r in requests:
            t_first = r.first_token_at if r.first_token_at is not None else (r.token_times[0] if r.token_times else t_start)
            t_last = r.token_times[-1] if r.token_times else (r.done_at if r.done_at is not None else t_end)
            streams_metrics.append((len(r.token_ids), t_first, t_last))
    else:
        tokens_lists, step_ms = out.get("token_ids", [[]] * c), out.get("step_ms", [])
        cum_s = sum(step_ms) / 1e3 if step_ms else (t_end - t_start)
        for i in range(c):
            n_tok = len(tokens_lists[i]) if i < len(tokens_lists) else max_tokens
            streams_metrics.append((n_tok, t_start, t_start + cum_s))

    out["wall_s"] = t_end - t_start
    return out, streams_metrics


def run_benchmark(engine, prompts_data: list[dict], bench_type: str, levels: list[int] = DEFAULT_LEVELS,
                  max_tokens: int = 400, warmup_tokens: int = 32, out_file: str | None = None) -> list[dict]:
    """Executes warmup and sequential levels for the given benchmark type."""
    try:
        from vxpu.engine import SamplingParams
        greedy = SamplingParams(do_sample=False, temperature=0.0, top_p=1.0)
    except (ImportError, ModuleNotFoundError):
        greedy = None

    if hasattr(engine, "eos_ids"):
        engine.eos_ids = set()

    warmup_text = get_warmup_prompt(prompts_data, bench_type)
    warmup_ids = encode_prompt(getattr(engine, "tokenizer", None), warmup_text)
    engine.generate_rows([warmup_ids], warmup_tokens, greedy)

    records = []
    for c in levels:
        prompts = build_prompts(prompts_data, bench_type, c)
        out, streams_metrics = run_level(engine, prompts, max_tokens=max_tokens)
        stream_rates = [round(stream_tokens_per_s(n, t0, t1), 1) for (n, t0, t1) in streams_metrics]
        agg_rate = round(aggregate_tokens_per_s(streams_metrics), 1)

        total_tokens = sum(n for (n, _, _) in streams_metrics)
        n_steps = len(out.get("step_ms", []))
        tok_per_step = round(total_tokens / max(1, n_steps * c), 2) if (out.get("drafted") or n_steps > 0) else 1.0

        spec = out.get("speculation")
        acceptance = [p["accepted"] for p in spec["positions"]] if spec and "positions" in spec else (out.get("accepted") if out.get("drafted") else None)

        rec = {
            "type": bench_type, "level": c, "concurrency": c, "streams": c,
            "per_stream_tokens_per_s": stream_rates, "aggregate_tokens_per_s": agg_rate,
            "tokens_per_step": tok_per_step, "rows_per_step": c,
            "drafted": out.get("drafted", 0), "accepted": out.get("accepted", 0),
            "acceptance": acceptance, "total_tokens": total_tokens,
            "wall_s": round(out.get("wall_s", 0.0), 3),
        }
        line = f"LEAGUE {json.dumps(rec)}"
        print(line, flush=True)
        if out_file:
            with open(out_file, "a") as f:
                f.write(line + "\n")
        records.append(rec)
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--artifact", required=True, help="Path to exported engine artifact")
    p.add_argument("--type", choices=("prose", "code", "all"), default="all", help="Benchmark type: prose, code, or all (both)")
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "league_prompts.json"), help="Path to league_prompts.json")
    p.add_argument("--levels", default="1,2,4,8,16", help="Comma-separated concurrency levels")
    p.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    p.add_argument("--max-tokens", type=int, default=400, help="Completion tokens per stream")
    p.add_argument("--warmup-tokens", type=int, default=32, help="Tokens for warmup request")
    p.add_argument("--out", default=None, help="Output file to append LEAGUE lines to")
    return p.parse_args()


def main():
    args = parse_args()
    from vxpu.engine import Engine

    prompts_data = load_league_prompts(args.prompts)
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    engine = Engine(args.artifact, device=args.device, compile_decode=args.device != "cpu")
    bench_types = ["prose", "code"] if args.type == "all" else [args.type]
    for t in bench_types:
        run_benchmark(engine, prompts_data, t, levels=levels, max_tokens=args.max_tokens,
                      warmup_tokens=args.warmup_tokens, out_file=args.out)


if __name__ == "__main__":
    main()
