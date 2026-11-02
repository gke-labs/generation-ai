# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hero run 15, the resident executor's first measurement: a chain of dependent GEMV nodes
at the real model's dense shapes, run as one resident kernel against the same node body
launched once per node inside a CUDA graph (the step graph's way), on one card.

    python resident-bench.py [--layers 8] [--rows 1] [--programs SMS] [--out results.json]

Three chains: the dense GEMVs of `layers` layers at the real shapes (the bytes case), 400
tiny nodes (K 128, N 256: the per-node overhead case), and the real chain with every
node's dependency removed (how much the spin-waits cost against free claiming). Reports
the per-step time of each way, the per-node cost, and the GEMV bandwidth achieved."""

import argparse
import json
import time

import torch
import triton

from vxpu import resident

# one layer's dense GEMVs at one row, DeepSeek-V4.1-Flash (N, K), from hero 11's ledger
LAYER = [("wq_a", 1280, 5120), ("wkv", 512, 5120), ("wq_b", 32768, 1280), ("wo_a", 8192, 4096),
         ("wo_b", 5120, 8192), ("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]


def make_weight(n, k, device):
    w = (torch.randn(n, k, device=device) * 0.05).to(torch.float8_e4m3fn)
    s = torch.full((n // 32, k // 32), 127, dtype=torch.uint8, device=device)  # scale 1.0
    s[::3] = 126  # a few half scales so the dequantization is exercised
    return w, s


def chain(device, layers, rows, shapes, deps=True, bn=16, bm=16, bk=128):
    wl = resident.WorkList(device, bn=bn, bm=bm, bk=bk)
    prev = None
    for _ in range(layers):
        for _, n, k in shapes:
            x = torch.randn(rows, k, device=device).to(torch.bfloat16)
            w, s = make_weight(n, k, device)
            prev, _ = wl.gemv(x, w, s, deps=(prev,) if (deps and prev is not None) else ())
    return wl.build()


def bytes_of(wl):
    return sum(w.numel() + s.numel() + x.numel() * 2 + y.numel() * 2 for x, w, s, y, _ in wl.nodes)


def time_graph(fn, warm=3, reps=20):
    """fn captured into one CUDA graph; the median replay time in ms."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)]


def check(wl, tag):
    refs = wl.reference()
    worst = 0.0
    for (x, w, s, y, _), r in zip(wl.nodes, refs):
        err = (y.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        worst = max(worst, err)
    print(f"  {tag}: max relative error vs torch {worst:.2e}", flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--programs", type=int, default=0, help="resident programs (default: the card's SMs)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    programs = args.programs or sms
    results = {"device": torch.cuda.get_device_name(device), "sms": sms, "programs": programs, "rows": args.rows}
    print(f"{results['device']}: {sms} SMs, {programs} resident programs, {args.rows} row(s)", flush=True)

    cases = [
        ("real", lambda: chain(device, args.layers, args.rows, LAYER)),
        ("real-nodeps", lambda: chain(device, args.layers, args.rows, LAYER, deps=False)),
        ("tiny", lambda: chain(device, 50, args.rows, [("t", 256, 128)] * 8)),
    ]
    for name, make in cases:
        wl = make()
        nb = bytes_of(wl)
        print(f"{name}: {wl.count} nodes, {nb / 1e9:.3f} GB", flush=True)
        # correctness, both ways
        wl.reset()
        wl.run_resident(programs)
        torch.cuda.synchronize()
        err_r = check(wl, "resident")
        for _, _, _, y, _ in wl.nodes:
            y.zero_()
        wl.run_launched()
        torch.cuda.synchronize()
        err_l = check(wl, "launched")
        # timing: the resident way is one memset of the counters plus one launch, captured
        def resident_step():
            wl.claim.zero_()
            wl.done.zero_()
            wl.run_resident(programs)
        r_med, r_p90 = time_graph(resident_step)
        l_med, l_p90 = time_graph(wl.run_launched)
        print(f"  resident: {r_med:.3f} ms a step (p90 {r_p90:.3f}), {r_med * 1e3 / wl.count:.2f} us a node, "
              f"{nb / r_med / 1e6:.0f} GB/s", flush=True)
        print(f"  launched: {l_med:.3f} ms a step (p90 {l_p90:.3f}), {l_med * 1e3 / wl.count:.2f} us a node, "
              f"{nb / l_med / 1e6:.0f} GB/s", flush=True)
        results[name] = {"nodes": wl.count, "bytes": nb, "resident_ms": r_med, "resident_p90_ms": r_p90,
                         "launched_ms": l_med, "launched_p90_ms": l_p90, "err_resident": err_r, "err_launched": err_l}
        del wl
        torch.cuda.empty_cache()
    # the resident forms against each other on the real chain: dynamic claiming with chunked
    # claims and more programs than SMs, and static striping within the body's occupancy
    wl = chain(device, args.layers, args.rows, LAYER)
    sweep = {}
    for form, p, chunk in [("dyn", sms, 1), ("dyn", 4 * sms, 1), ("dyn", 8 * sms, 1),
                           ("static", 2 * sms, 0), ("static", 4 * sms, 0), ("static", 6 * sms, 0)]:
        def step():
            wl.claim.zero_()
            wl.done.zero_()
            if form == "dyn":
                wl.run_resident(p, chunk=chunk)
            else:
                wl.run_static(p)
        try:
            med, p90 = time_graph(step)
        except Exception as e:  # a static grid past the resident slots would hang: report instead
            print(f"  {form} with {p} programs chunk {chunk}: failed: {str(e)[:80]}", flush=True)
            continue
        torch.cuda.synchronize()
        err = check(wl, f"{form} {p} programs chunk {chunk}")
        sweep[f"{form}-{p}-{chunk}"] = {"ms": med, "p90": p90, "err": err}
        print(f"  {form} with {p} programs, chunk {chunk}: {med:.3f} ms a step ({bytes_of(wl) / med / 1e6:.0f} GB/s)", flush=True)
    results["forms"] = sweep
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
