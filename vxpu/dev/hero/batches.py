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

"""The hero-run instrument: N batches of prompts through one engine, measured against their floors.

A hero run is judged by one table: for each shape (rows an artifact decodes together) and prompt
mode, the step time, the tokens per second, and the floor those bytes set on this hardware. This
driver produces the rows of that table, for one batch (hero 8's lines) and for N batches at once
(hero 9's: several batches in flight across the GPUs' layer ranges, measured as aggregate tokens
per second with each batch's latency alongside).

    python batches.py --artifact DIR --label L [--batches N] [--mode one|same|different]
                      [--strategy auto|sequential|threads|pipeline] [--new-tokens 64]
                      [--prompts FILE] [--reference FILE] [--out results.jsonl]

Prompt modes: `one` puts one prompt in the first row and leaves the others idle; `same` puts the
same prompt in every row (the rows' tokens must then be identical, and the drafter must accept
rows times what one row accepts); `different` gives every row its own prompt (batch b starts at
prompt b * rows, wrapping). Strategies: `sequential` runs the batches one after another in one
process (the no-overlap baseline); `threads` starts them together as threads (the engine's own
lock decides how much overlaps; the report says how much did); `pipeline` uses the engine's
batched entry point when it has one (`generate_batches`, the pipeline of batches across ranges);
`auto` takes the pipeline when it exists, else threads for N > 1, else sequential.

Each batch prints one `RESULT {json}` line with hero-spec's fields (label, rows, prompts,
step_tokens, median_step_ms, step_ms_p10_p90, drafted, accepted, tokens_per_row_step,
decode_tokens_per_s, end_to_end_tokens_per_s, match, of, first_diff) plus: batch, batches, mode,
strategy, wall_s, the group's aggregate_tokens_per_s (every batch's tokens over the union of their
windows), overlap (the batches' walls summed over the union: 1.0 means no overlap, N means full),
the floors (floor_single_ms: this batch's bytes over one GPU's bandwidth; floor_pipeline_slot_ms: the
slowest range's bytes over one GPU's, the time between batch steps with batches in flight;
aggregate_floor_tokens_per_s: a batch's rows over that slot), the distinct experts a layer when the
engine counts them (VXPU_EXPERT_STATS=1), and the rows' identity (`same_rows` for the same mode).
The single-batch run of a prompt set is the reference: its tokens go to --reference (per prompt
index), and later runs report `match`/`first_diff` against it. A `SAME` line follows a same-mode
batch, as hero-spec prints it. Every line also appends to --out for table.py."""
import argparse
import json
import os
import statistics
import sys
import threading
import time

GPU_BANDWIDTH = 1460e9  # bytes a second one RTX PRO 6000 copies at (measured)
DEFAULT_GPUS = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--mode", choices=("one", "same", "different"), default="different")
    p.add_argument("--strategy", choices=("auto", "sequential", "threads", "pipeline"), default="auto")
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts-32.txt"))
    p.add_argument("--reference", default=None, help="tokens per prompt index from a single-batch run; written when absent")
    p.add_argument("--out", default=None, help="results.jsonl to append RESULT lines to")
    p.add_argument("--gpus", type=int, default=DEFAULT_GPUS, help="GPUs whose bandwidth the pipeline floor adds up")
    p.add_argument("--expert-bytes", type=float, default=None, help="bytes of one expert (default: from the manifest)")
    p.add_argument("--warmup-seconds", type=float, default=4.0)
    p.add_argument("--topk", type=int, default=6, help="experts a token selects (the floor of one or same-prompt rows)")
    return p.parse_args()


def dense_bytes(artifact):
    """Bytes of dense weights a decode step reads, from the artifact's binding: every bound tensor
    that is not state, split into the backbone (layers and engram), the head, and the drafter's
    stages. The embedding is read one row at a time and left out. An alias (the drafter's head is
    the model's) counts once."""
    binding = json.load(open(os.path.join(artifact, "binding.json")))
    state = set(binding.get("state", []))
    aliases = binding.get("aliases", {})
    seen, parts = set(), {"backbone": 0, "head": 0, "drafter": 0}
    for name, t in binding["bound"].items():
        if name in state or name in aliases:
            continue
        key = (t.get("file_sha256"), t.get("offset"), t.get("length"))
        if key in seen:
            continue
        seen.add(key)
        n = t["length"]
        if name.endswith("embed.weight"):
            continue
        if name.endswith("head.weight"):
            parts["head"] += n
        elif ".mtp." in name:
            parts["drafter"] += n
        else:
            parts["backbone"] += n
    return parts


def expert_bytes(artifact):
    """Bytes of one routed expert (its three weights with their scales), from the manifest."""
    manifest = json.load(open(os.path.join(artifact, "manifest.json")))
    total = 0
    for name, t in manifest["tensors"].items():
        if ".experts.0." in name and name.startswith("layers.0."):
            total += t.get("length") or (t.get("size") or 0)
    return total or 18.8e6


def ranges_from_env(layers):
    """The layer ranges per GPU from VXPU_LAYER_DEVICES ("0:8=cuda:0,8:20=cuda:1,..."), as
    {gpu: [layer ids]}; one range with every layer when unset (one GPU)."""
    spec = os.environ.get("VXPU_LAYER_DEVICES", "")
    if not spec:
        return {0: list(range(layers))}
    out = {}
    for item in spec.split(","):
        rng, dev = item.split("=")
        a, b = rng.split(":")
        gpu = int(dev.split(":")[-1]) if ":" in dev else 0
        out.setdefault(gpu, []).extend(range(int(a), int(b) if b else layers))
    return out


def layer_dense_bytes(artifact):
    """Dense bytes per layer id from the binding, the drafter's stages as layers 40.. and the head
    with the last layer (the planner's convention)."""
    binding = json.load(open(os.path.join(artifact, "binding.json")))
    state = set(binding.get("state", []))
    aliases = binding.get("aliases", {})
    per, seen, head, mtp = {}, set(), 0, {}
    for name, t in binding["bound"].items():
        if name in state or name in aliases or name.endswith("embed.weight"):
            continue
        key = (t.get("file_sha256"), t.get("offset"), t.get("length"))
        if key in seen:
            continue
        seen.add(key)
        if name.endswith("head.weight"):
            head += t["length"]
        elif name.startswith("model.mtp."):
            k = int(name.split(".")[2]); mtp[k] = mtp.get(k, 0) + t["length"]
        elif name.startswith("model.layers."):
            k = int(name.split(".")[2]); per[k] = per.get(k, 0) + t["length"]
    return per, head, mtp


def floors(artifact, rows, step_tokens, distinct_per_layer, layers, gpus, per_expert):
    """The byte floors of this batch's decode step. A single batch streams from one GPU at a time:
    its floor is all the bytes over one GPU's bandwidth. With batches in flight, every GPU streams
    its own range, and the aggregate ceiling is a batch's rows over the slowest range's bytes at
    one GPU's bandwidth (the planner's rule), the per-batch latency floor the sum of the ranges.
    distinct_per_layer is measured when the engine counts it, else the bound (rows × step tokens ×
    top-k 6, at most 384)."""
    source = "measured" if distinct_per_layer is not None else "bound"
    if distinct_per_layer is None:
        distinct_per_layer = min(rows * step_tokens * 6, 384)
    per_layer, head, mtp = layer_dense_bytes(artifact)
    experts_layer = distinct_per_layer * per_expert
    ranges = ranges_from_env(layers + len(mtp))
    range_bytes = []
    for gpu in sorted(ranges):
        b = 0
        for k in ranges[gpu]:
            if k < layers:
                b += per_layer.get(k, 0) + experts_layer
            elif step_tokens > 1:  # the drafter's stages run only when the step verifies drafts
                b += mtp.get(k - layers, 0)
        range_bytes.append(b)
    range_bytes[-1] += head
    total = sum(range_bytes)
    slowest = max(range_bytes)
    return {"floor_bytes_gb": round(total / 1e9, 2), "floor_dense_gb": round((sum(per_layer.values()) + head + (sum(mtp.values()) if step_tokens > 1 else 0)) / 1e9, 2),
            "floor_experts_gb": round(experts_layer * layers / 1e9, 2), "floor_distinct_per_layer": round(distinct_per_layer, 1),
            "floor_distinct_source": source, "floor_range_bytes_gb": [round(b / 1e9, 2) for b in range_bytes],
            "floor_single_ms": round(total / GPU_BANDWIDTH * 1e3, 1),
            "floor_pipeline_slot_ms": round(slowest / GPU_BANDWIDTH * 1e3, 1),
            "aggregate_floor_tokens_per_s": round(rows / (slowest / GPU_BANDWIDTH), 1)}


def load_prompts(engine, path):
    texts = [p.strip() for p in open(path) if p.strip()]
    ids = [engine.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
           for t in texts]
    ids.append([(11 * j + 5) % 30000 + 100 for j in range(24)])  # the synthetic prompt of the profilers
    return ids


def batch_prompts(ids, rows, mode, b):
    if mode == "one":
        return [ids[b % len(ids)]], [b % len(ids)]
    if mode == "same":
        k = b % len(ids)
        return [ids[k]] * rows, [k] * rows
    idx = [(b * rows + i) % len(ids) for i in range(rows)]
    return [ids[k] for k in idx], idx


def summarize(out, prompts, idx, reference, new_tokens):
    gen = sum(len(x) for x in out["token_ids"])
    steps = out["step_ms"]
    decode_s = sum(steps) / 1e3
    match, first_diff = 0, []
    for i, k in enumerate(idx):
        ref = reference.get(str(k))
        got = out["token_ids"][i][:new_tokens]
        if ref is None:
            first_diff.append(None)
            continue
        ref = ref[:len(got)]
        first_diff.append(next((j for j, (a, c) in enumerate(zip(got, ref)) if a != c), None))
        match += got == ref
    ss = sorted(steps)
    return {
        "prompts": len(prompts), "decode_steps": len(steps),
        "prefill_s_total": round(sum(out["prefill_ms"]) / 1e3, 1),
        "median_step_ms": round(statistics.median(steps), 1) if steps else None,
        "step_ms_p10_p90": [round(ss[len(ss) // 10]), round(ss[len(ss) * 9 // 10])] if steps else None,
        "drafted": out["drafted"], "accepted": out["accepted"],
        "tokens": gen, "tokens_per_row_step": round(gen / max(1, sum(out["rows_active"])), 2),
        "decode_tokens_per_s": round(gen / decode_s, 1) if decode_s else None,
        "match": match, "of": sum(1 for k in idx if str(k) in reference), "first_diff": first_diff,
        "same_rows": all(t == out["token_ids"][0] for t in out["token_ids"]) if len(out["token_ids"]) > 1 else None,
    }


def main():
    args = parse_args()
    from vxpu.engine import Engine, SamplingParams
    t0 = time.time()
    engine = Engine(args.artifact, device="cuda", compile_decode=True)
    load_s = time.time() - t0
    rows = json.load(open(os.path.join(args.artifact, "binding.json")))["batch"]
    greedy = SamplingParams(do_sample=False)
    ids = load_prompts(engine, args.prompts)
    reference = json.load(open(args.reference)) if args.reference and os.path.exists(args.reference) else {}
    per_expert = args.expert_bytes or expert_bytes(args.artifact)
    layers = len({int(k) for k in json.load(open(os.path.join(args.artifact, "binding.json")))["external"]["experts"]
                  if int(k) < 40}) or 40

    warm = time.time()
    while time.time() - warm < args.warmup_seconds:  # the first seconds after a load are not the engine's steady state
        engine.generate_rows(ids[:1], 64, greedy)

    strategy = args.strategy
    has_pipeline = hasattr(engine, "generate_batches")
    if strategy == "auto":
        strategy = "pipeline" if has_pipeline and args.batches > 1 else ("threads" if args.batches > 1 else "sequential")
    if strategy == "pipeline" and not has_pipeline:
        sys.exit("this engine has no generate_batches: the pipeline is not in this code")

    batches = [batch_prompts(ids, rows, args.mode, b) for b in range(args.batches)]
    # the engine's accessor sums every batch's graph (a pipeline run's batches do not run on _decode_split)
    stats = engine if hasattr(engine, "expert_stats") else getattr(engine, "_decode_split", None)
    counts0 = stats.expert_stats() if stats is not None and hasattr(stats, "expert_stats") else None
    windows, outs = [None] * args.batches, [None] * args.batches

    def run_one(b):
        start = time.time()
        outs[b] = engine.generate_rows(batches[b][0], args.new_tokens, greedy)
        windows[b] = (start, time.time())

    group_start = time.time()
    if strategy == "sequential":
        for b in range(args.batches):
            run_one(b)
    elif strategy == "threads":
        threads = [threading.Thread(target=run_one, args=(b,)) for b in range(args.batches)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:  # pipeline: the engine steps every batch together and returns one result per batch
        results = engine.generate_batches([p for p, _ in batches], args.new_tokens, greedy)
        end = time.time()
        for b, out in enumerate(results):
            outs[b] = out
            # the batches decode together from the group's start; a batch's window is its own decode
            # time (its steps' sum), so the aggregate is tokens over decode time, as the single-batch
            # line's decode_tokens_per_s is, not over the call's wall (which holds the prefill and the
            # loops' set-up); end_to_end_tokens_per_s keeps the wall
            windows[b] = (group_start, group_start + sum(out["step_ms"]) / 1e3)
    group_end = time.time()

    union = max(w[1] for w in windows) - min(w[0] for w in windows)
    summed = sum(w[1] - w[0] for w in windows)
    total_tokens = sum(sum(len(x) for x in o["token_ids"]) for o in outs)
    counts1 = stats.expert_stats() if counts0 is not None else None
    distinct = None
    if counts0 is not None and counts1 is not None:
        # the counters sit in the engine's own step graph, which is batch 0's; the other batches' graphs
        # carry none, so the per-step count divides by batch 0's steps (every batch selects alike)
        steps_all = len(outs[0]["step_ms"]) if strategy == "pipeline" else sum(len(o["step_ms"]) for o in outs)
        deltas = [counts1.get(l, 0) - counts0.get(l, 0) for l in range(layers)]
        distinct = round(sum(deltas) / max(1, steps_all) / layers, 1)
    if distinct is None and args.mode in ("one", "same"):
        distinct = float(args.topk)  # every row chooses the same experts: top-k of them a layer
    fl = floors(args.artifact, rows, engine.step_tokens, distinct, layers, args.gpus, per_expert)
    occupancy = None
    if stats is not None and hasattr(stats, "occupancy"):
        try:
            occupancy = stats.occupancy()
        except Exception:
            occupancy = None

    for b in range(args.batches):
        s = summarize(outs[b], batches[b][0], batches[b][1], reference, args.new_tokens)
        rec = {"label": f"{args.label}-{args.mode}", "batch": b, "batches": args.batches, "mode": args.mode,
               "strategy": strategy, "rows": rows, "step_tokens": engine.step_tokens, "load_s": round(load_s, 1),
               "wall_s": round(windows[b][1] - windows[b][0], 2), **s,
               "aggregate_tokens_per_s": round(total_tokens / union, 1) if union else None,
               "overlap": round(summed / union, 2) if union else None,
               "distinct_experts_per_layer": distinct, **fl, "gpu_occupancy": occupancy,
               "end_to_end_tokens_per_s": round(total_tokens / (group_end - group_start), 1)}
        line = "RESULT " + json.dumps(rec)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")
        if args.mode == "same" and rows > 1:
            one = reference.get(f"accepted:{batches[b][1][0]}")
            msg = (f"SAME rows identical: {s['same_rows']} | accepted == rows * one: "
                   f"{(outs[b]['accepted'] == rows * one) if one is not None else 'no one-row reference'}")
            print(msg, flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(msg + "\n")

    if args.reference and args.batches == 1 and args.mode in ("one", "different"):
        for i, k in enumerate(batches[0][1]):  # the single-batch run is the reference for the next ones
            reference.setdefault(str(k), outs[0]["token_ids"][i][:args.new_tokens])
        if args.mode == "one":
            reference[f"accepted:{batches[0][1][0]}"] = outs[0]["accepted"]
        json.dump(reference, open(args.reference, "w"))


if __name__ == "__main__":
    main()
