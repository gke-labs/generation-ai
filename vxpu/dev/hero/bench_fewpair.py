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

"""Hero 15, the expert kernels at one card's shape under expert parallelism: a layer launches
rows x top-6 pairs, the card holds H of them (the other pairs' programs exit at once), the held
pairs sit on distinct experts unless --share says otherwise, and the slots rotate between calls
so the L2 never serves a repeat. Synthetic banks of the real shapes (K 5120, I 2304; 96 slots, a
card's quarter of a layer). Each form is timed per kernel under the profiler and as a whole call
by CUDA events; the floors printed are the held distinct experts' bytes at the read peak (HBM)
and the held pairs' bytes at the L2 rate.

  python bench_fewpair.py --rows 1,2,3,5,8 [--held auto|1,2,4,6] [--forms direct,gemm,grouped,one,pdl]
                          [--knobs FP4_GU_KS=2,FP4_DN_KS=2] [--share 2] [--held-only] [--per-pair]
                          [--compare]
"""
import argparse
import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from vxpu import triton_ops as T

K, I, TOPK = 5120, 2304, 6
EXPERTS, SLOTS = 384, 96  # a layer's experts; the quarter this card holds (expert e at slot e // 4 when e % 4 == 0)
READ_PEAK = 1535.0  # GB/s, the card's single-tensor read (hero run 11)
L2_RATE = 5700.0  # GB/s, the figure the TP step note used for L2 traffic
W13 = 2 * (I * K // 2 + I * K // 32)  # gate and up of one expert: packed fp4 plus e8m0 scales
W2 = K * I // 2 + K * I // 32


def banks(dev, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)

    def bank(rows, inner):
        wq = torch.randint(0, 256, (SLOTS, rows, inner // 2), device=dev, dtype=torch.uint8, generator=g)
        sc = torch.randint(120, 128, (SLOTS, rows, inner // 32), device=dev, dtype=torch.uint8, generator=g)
        return wq, sc

    w1, s1 = bank(I, K)
    w3, s3 = bank(I, K)
    w2, s2 = bank(K, I)
    return w1, s1, w3, s3, w2, s2


def slot_map(dev):
    m = torch.full((EXPERTS,), -1, dtype=torch.int64)
    m[::4] = torch.arange(SLOTS)
    return m.to(dev)


def draw(rows, held, share, it, g):
    """Expert ids [rows, TOPK] for one call: `held` pairs on this card's experts, on held // share
    distinct slots taken from a window that moves 12 slots a call (so consecutive calls touch
    disjoint experts), the rest on experts of other cards."""
    pairs = rows * TOPK
    ids = torch.randint(0, EXPERTS, (pairs,), generator=g)
    ids = ids + (ids % 4 == 0).to(ids.dtype)  # not held here
    distinct = -(-held // share)
    base = (it * 12) % SLOTS
    slots = [(base + j) % SLOTS for j in range(distinct)]
    where = torch.randperm(pairs, generator=g)[:held]
    for j, p in enumerate(where.tolist()):
        ids[p] = 4 * slots[j % distinct]
    return ids.view(rows, TOPK), distinct


def set_knobs(spec):
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        cur = getattr(T, k)
        setattr(T, k, v if cur is None or isinstance(cur, str) else type(cur)(v))


def form_knobs(form):
    T.FP4_DIRECT, T.FP4_GEMM_MIN_PAIRS, T.FP4_ONE_LAUNCH, T.FP4_PDL = "0", 1 << 30, None, None
    if form in ("direct", "one", "pdl"):
        T.FP4_DIRECT = "1"
    if form == "one":
        T.FP4_ONE_LAUNCH = "1"
    if form == "pdl":
        T.FP4_PDL = "2"
    if form == "gemm":
        T.FP4_GEMM_MIN_PAIRS = 1


def call(x, wt, smap, idx, b, per_pair):
    return T.fp4_experts_decode(x, wt, smap, *b, 7.0, per_pair=per_pair, indices=idx, start=0)


def classify(name):
    for key in ("gate_up", "down", "swiglu", "schedule", "layer_kernel"):
        if key in name:
            return key
    return "other"


def timed(x, wt, smap, idxs, b, iters, per_pair):
    for i in range(2):
        call(x, wt, smap, idxs[i], b, per_pair)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        call(x, wt, smap, idxs[2 + i], b, per_pair)
        ends[i].record()
    torch.cuda.synchronize()
    wall = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))[iters // 2]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            call(x, wt, smap, idxs[2 + i], b, per_pair)
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        per.setdefault(classify(e.name), []).append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
    return {k: sum(v) / iters for k, v in per.items()}, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,5,8")
    ap.add_argument("--held", default="auto", help="held pairs a call: auto (rows x 6 / 4, rounded up) or a list")
    ap.add_argument("--forms", default="direct,gemm")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--knobs", default="", help="NAME=VALUE,... set on triton_ops before the runs")
    ap.add_argument("--share", type=int, default=1, help="held pairs per distinct expert")
    ap.add_argument("--held-only", action="store_true", help="launch only the held pairs (top-1 rows), no exiting programs")
    ap.add_argument("--per-pair", action="store_true", help="per-pair replies (no in-kernel sum over a token's pairs)")
    ap.add_argument("--compare", action="store_true", help="each form's outputs against the direct kernels' on the same inputs")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    b = banks(dev)
    smap = slot_map(dev)
    set_knobs(a.knobs)
    g = torch.Generator().manual_seed(7)
    print(f"forms {a.forms}; knobs {a.knobs}; share {a.share}; held-only {a.held_only}; per-pair {a.per_pair}", flush=True)
    for rows in (int(r) for r in a.rows.split(",")):
        helds = [-(-rows * TOPK // 4)] if a.held == "auto" else [int(h) for h in a.held.split(",")]
        for held in helds:
            held = min(held, rows * TOPK)
            idxs, distinct = [], 0
            for it in range(a.iters + 2):
                ids, distinct = draw(rows, held, a.share, it, g)
                if a.held_only:
                    ids = ids.view(-1)[(ids.view(-1) % 4 == 0)].view(-1, 1)
                idxs.append(ids.to(dev))
            n, topk = idxs[0].shape
            x = (torch.randn(n, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            wt = torch.rand(n, topk, generator=g).to(dev)
            hbm = distinct * (W13 + W2) / READ_PEAK / 1e3
            l2 = held * (W13 + W2) / L2_RATE / 1e3
            ref = None
            for form in a.forms.split(","):
                form_knobs(form)
                try:
                    r, wall = timed(x, wt, smap, idxs, b, a.iters, a.per_pair)
                except Exception as e:  # noqa: BLE001
                    print(f"rows {rows} held {held} {form}: failed: {str(e).splitlines()[0][:160]}", flush=True)
                    continue
                note = ""
                if a.compare:
                    form_knobs("direct")
                    ref = call(x, wt, smap, idxs[2], b, a.per_pair).clone()
                    form_knobs(form)
                    got = call(x, wt, smap, idxs[2], b, a.per_pair)
                    d = (got - ref).abs()
                    ulp = (d / (ref.abs() * 2 ** -8 + 1e-30)).max().item()
                    note = " | bit-identical to direct" if torch.equal(got, ref) else f" | vs direct max |diff| {d.max().item():.3e}, {ulp:.2f} bf16 ulp"
                ksum = sum(r.values())
                parts = " ".join(f"{k} {v:6.1f}" for k, v in sorted(r.items()))
                print(f"rows {rows} pairs {n * topk:3d} held {held:2d} distinct {distinct:2d} | {form:8s} {parts} | kernels {ksum:6.1f} "
                      f"wall {wall:6.1f} us | floors HBM {hbm:5.1f} L2 {l2:5.1f} | {100 * hbm / max(ksum, 1e-9):3.0f}% of HBM{note}", flush=True)
                print("RESULT " + json.dumps({"form": form, "rows": rows, "pairs": n * topk, "held": held, "distinct": distinct,
                                              "kernels_us": r, "kernels_sum_us": ksum, "wall_us": wall, "hbm_floor_us": hbm,
                                              "l2_floor_us": l2, "knobs": a.knobs, "share": a.share, "held_only": a.held_only,
                                              "per_pair": a.per_pair}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
