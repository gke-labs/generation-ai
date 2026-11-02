"""Hero 15, the expert GEMM form: the grouped fp4 expert kernels (many pairs, sorted by expert) timed per
launch on one GPU with synthetic banks of the real shapes (K 5120, I 2304, top-6), the pairs' experts drawn
at random over the slots as a real step's are (so the distinct count per call matches a real step's:
240 pairs over 96 slots give about 88 distinct, 990 over 192 about 190), alternating slot halves so the
L2 never serves a repeat, under the CUDA profiler. Bytes are counted per distinct expert (each expert's
weights are read once per call by both forms), so the percentage is of the read peak.

  python bench_grouped.py --pairs 48,240,990 --iters 12 [--slots 192] [--form grouped|gemm]
"""
import argparse, json, sys, time
import torch
from torch.profiler import profile, ProfilerActivity
from vxpu import triton_ops as T
sys.path.insert(0, "/root/hero")
import bench_fp4 as B

K, I, TOPK = B.K, B.I, B.TOPK
READ_PEAK = 1535.0
W13 = 2 * (I * K // 2 + I * K // 32)  # gate and up, packed fp4 plus e8m0 scales, one expert
W2 = K * I // 2 + K * I // 32

def draws(pairs, S, iters, dev, seed=7):
    g = torch.Generator().manual_seed(seed)
    n = max(1, pairs // TOPK)
    topk = TOPK if pairs % TOPK == 0 else pairs
    out = []
    half = S // 2
    for it in range(iters):
        base = half if it % 2 else 0
        rows = [torch.randperm(half, generator=g)[:topk] + base for _ in range(n)]
        out.append(torch.stack(rows).to(torch.int32).to(dev))
    return n, topk, out

def timed(x, wt, smap, b, idxs, iters, per_pair=False):
    for i in range(2):
        B.one_call(x, wt, smap, idxs[i], b, per_pair=per_pair)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            B.one_call(x, wt, smap, idxs[i], b, per_pair=per_pair)
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = e.name
        if "gate_up" in name: key = "gate_up"
        elif "down" in name: key = "down"
        elif "swiglu" in name: key = "swiglu"
        elif "schedule" in name: key = "schedule"
        else: key = "other"
        per.setdefault(key, []).append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
    return {k: sum(v) / iters for k, v in per.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="6,48,240,990")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--slots", type=int, default=192)
    ap.add_argument("--form", default="grouped")
    ap.add_argument("--knobs", default="", help="NAME=VALUE,... set on triton_ops before the runs")
    ap.add_argument("--compare", action="store_true", help="compare the gemm form's outputs to the grouped form's on the same inputs")
    a = ap.parse_args()
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    S = a.slots
    b = B.banks(S, dev)
    smap = torch.arange(S, dtype=torch.int32, device=dev)
    T.FP4_DIRECT = "0"  # the grouped path whatever the pair count
    for kv in filter(None, a.knobs.split(",")):
        k, v = kv.split("=")
        setattr(T, k, type(getattr(T, k))(v) if not isinstance(getattr(T, k), (type(None), str)) else v)
    if a.form == "gemm":
        T.FP4_GEMM_MIN_PAIRS = 1
    elif hasattr(T, "FP4_GEMM_MIN_PAIRS"):
        T.FP4_GEMM_MIN_PAIRS = 1 << 30
    print(f"form {a.form}; banks {S} slots {sum(t.numel() for t in b) / 1e9:.2f} GB; knobs {a.knobs}", flush=True)
    for pairs in (int(p) for p in a.pairs.split(",")):
        n, topk, idxs = draws(pairs, S, a.iters + 2, dev)
        x = torch.randn(n, K, device=dev).to(torch.bfloat16)
        wt = torch.rand(n, topk, device=dev)
        distinct = sum(len(torch.unique(i)) for i in idxs[2:]) / a.iters
        nbytes = distinct * (W13 + W2)
        floor = nbytes / READ_PEAK / 1e3
        try:
            r = timed(x, wt, smap, b, idxs, a.iters)
        except Exception as e:
            print(f"pairs {pairs}: failed: {str(e).splitlines()[0][:160]}", flush=True)
            continue
        if a.compare:
            T.FP4_GEMM_MIN_PAIRS = 1 << 30
            ref = B.one_call(x, wt, smap, idxs[2], b).clone()
            refp = B.one_call(x, wt, smap, idxs[2], b, per_pair=True).clone()
            T.FP4_GEMM_MIN_PAIRS = 1 if a.form == "gemm" else 1 << 30
            got = B.one_call(x, wt, smap, idxs[2], b)
            gotp = B.one_call(x, wt, smap, idxs[2], b, per_pair=True)
            same = torch.equal(got, ref) and torch.equal(gotp, refp)
            rel = ((got - ref).abs().max() / ref.abs().max().clamp(min=1e-9)).item()
            print(f"pairs {pairs}: gemm vs grouped {'bit-identical' if same else f'differ, max rel {rel:.2e}'}", flush=True)
        gu, dn = r.get("gate_up", 0.0), r.get("down", 0.0)
        tot = sum(r.values())
        print(f"pairs {pairs:4d} rows {n:3d} distinct {distinct:6.1f} bytes {nbytes / 1e6:7.1f} MB floor {floor:7.1f} us | "
              f"gate_up {gu:7.1f} ({100 * distinct * W13 / 1e3 / max(gu, 1e-9) / READ_PEAK:3.0f}%) down {dn:7.1f} "
              f"({100 * distinct * W2 / 1e3 / max(dn, 1e-9) / READ_PEAK:3.0f}%) swiglu {r.get('swiglu', 0):5.1f} "
              f"schedule {r.get('schedule', 0):5.1f} other {r.get('other', 0):5.1f} | all {tot:7.1f} us = {100 * floor / tot:3.0f}% of floor; "
              f"tax {(tot - floor) / pairs:5.2f} us a pair", flush=True)
        print("RESULT15 " + json.dumps({"form": a.form, "pairs": pairs, "rows": n, "distinct": distinct, "bytes": nbytes,
                                        "floor_us": floor, "gate_up_us": gu, "down_us": dn, "all_us": tot, "knobs": a.knobs}), flush=True)

if __name__ == "__main__":
    main()
