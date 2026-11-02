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

"""The measuring step for the GEMV templates (hero run 11): every kernel against its own floor.

A GEMV call belongs to a shape class (triton_ops.gemv_class: the template, the rows, the weight's
shape and dtype, the prologue/epilogue flags). The kernel record lists the classes an engine
launched with the blocks that ran (stats.gemv_classes). This module gives each class its bytes
and its floor at the card's bandwidth, and times a class under any blocks on the card, so the
planner (`vxpu kernels measure`, pkg/kernelplan) can sweep the blocks per class and write the
best into the table's "blocks" entries, which triton_ops.gemv_blocks applies.

    python -m vxpu.measure classes RECORD.json          the classes in a record: bytes, floor, what ran
    python -m vxpu.measure run JOBS.json [--out R.jsonl] time each job {"shapes": class, "blocks": {...}}
    python -m vxpu.measure bandwidth                     the card's copy bandwidth, as the floors use it

A job's timing: the kernel is launched over weights cycled through more copies than the L2 holds
(the step reads each weight once, from HBM), captured into a CUDA graph and replayed, so the
number is the kernel's GPU time with no launch gaps, as in the step. Bytes count what the kernel
must move once: the weight and its scales, the activations it reads, what it stores.
"""
import argparse
import json
import math
import statistics
import sys
import time

import torch

from . import triton_ops as T

L2_BYTES_DEFAULT = 128 << 20
CYCLE_BYTES = 320 << 20  # weights cycled through at least this many bytes a replay: past any card's L2


def parse_class(key):
    """"gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1 epi=quant" -> dict(template, rows, n, k, w, flags...)."""
    parts = key.split()
    out = {"template": parts[0]}
    for p in parts[1:]:
        name, _, value = p.partition("=")
        out[name] = int(value) if value.lstrip("-").isdigit() else value
    for name in ("rows", "n", "k"):
        if name not in out:
            raise ValueError(f"shape class {key!r}: missing {name}")
    return out


def class_bytes(key):
    """The bytes a class moves once a call: the weights and their e8m0 scales, the activations
    read (bf16, or the MXFP8 codes and scales on the native path), the outputs stored."""
    c = parse_class(key)
    M, N, K = c["rows"], c["n"], c["k"]
    wbytes = 1 if c.get("w") == "fp8" else 2
    native = c.get("native") == 1
    weights = 2 if c["template"] == "dual_gemv_swiglu" else 1
    total = weights * N * K * wbytes
    if c.get("w") == "fp8":
        total += weights * (N // 32) * (K // 32)
    xk = K * c.get("groups", 1) if c["template"] == "gemv_rows" else K
    total += M * xk * (1 + 1 / 32 if native else 2)
    if c["template"] == "dual_gemv_swiglu":
        total += M * N * 2
    elif c.get("epi", "").startswith("hc"):
        hc = int(c["epi"][2:])
        total += M * hc * N * 2 * 2 + M * N * 4  # the stream read and written, the experts' sum
    else:
        total += M * N * (4 if c.get("out") == "f32" else 2)
    return int(total)


def bandwidth_gbps(device=None, nbytes=1 << 30, repeat=5):
    """The card's copy bandwidth: a device-to-device copy of nbytes, read plus write, best of
    repeat (the floors use it; 1,460 GB/s measured on an RTX PRO 6000)."""
    dev = torch.device(device or "cuda")
    with torch.cuda.device(dev):
        a = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        b = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        b.copy_(a)
        torch.cuda.synchronize(dev)
        best = 0.0
        for _ in range(repeat):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            b.copy_(a)
            e.record()
            torch.cuda.synchronize(dev)
            best = max(best, 2 * nbytes / (s.elapsed_time(e) / 1e3) / 1e9)
    return best


def _e8m0(shape, device):
    """Random e8m0 block scales near one (exponents 120 to 134), as uint8 for the kernels."""
    return torch.randint(120, 135, shape, dtype=torch.uint8, device=device)


class _Job:
    """One class under one set of blocks: the operands (weights cycled through enough copies to
    pass the L2) and a launch of the kernel with the dispatcher's arguments."""

    def __init__(self, key, blocks, device, l2_bytes):
        self.key = key
        self.c = c = parse_class(key)
        self.blocks = dict(blocks)
        self.device = device
        M, N, K = c["rows"], c["n"], c["k"]
        self.M, self.N, self.K = M, N, K
        self.native = c.get("native") == 1
        self.fp8 = c.get("w") == "fp8"
        self.template = c["template"]
        wbytes = (2 if self.template == "dual_gemv_swiglu" else 1) * N * K * (1 if self.fp8 else 2)
        self.copies = max(2, min(64, math.ceil(max(CYCLE_BYTES, 2 * l2_bytes) / max(wbytes, 1))))
        self.weights = []
        for _ in range(self.copies):
            if self.fp8:
                w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                s = _e8m0((N // 32, K // 32), device)
            else:
                w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
                s = None
            if self.template == "dual_gemv_swiglu":
                w3 = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                self.weights.append((w, s, w3, _e8m0((N // 32, K // 32), device)))
            else:
                self.weights.append((w, s))
        groups = c.get("groups", 1) if self.template == "gemv_rows" else 1
        xk = K * groups
        self.x = (torch.randn(M, xk, device=device, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn).to(torch.bfloat16)
        self.xq, self.xe = T.x_mxfp8(self.x) if self.native else (self.x, self.x)
        self.groups = groups
        self.hc = int(c["epi"][2:]) if str(c.get("epi", "")).startswith("hc") else 0
        self.quant = c.get("epi") == "quant"
        self.out_f32 = c.get("out") == "f32"
        self.bm = 16
        while self.bm < min(M, T.GEMV_ROWS_MAX):  # past GEMV_ROWS_MAX rows the kernel runs tiled (launch)
            self.bm *= 2
        self.bn = int(blocks["BN"])
        self.bk = int(blocks["BK"])
        self.ks = int(blocks.get("KSPLIT", 1))
        self.warps = int(blocks.get("warps", 4))
        self.stages = int(blocks.get("stages", 3))
        self.check()
        if self.template == "gemv_rows":
            if self.hc:
                self.residual = torch.randn(M, self.hc, N, device=device, dtype=torch.bfloat16)
                self.post = torch.randn(M, self.hc, device=device)
                self.comb = torch.randn(M, self.hc, self.hc, device=device)
                self.add = torch.randn(M, N, device=device)
                self.y = torch.empty(M, self.hc, N, dtype=torch.bfloat16, device=device)
            else:
                dt = torch.float32 if (self.out_f32 or self.ks > 1) else torch.bfloat16
                self.y = torch.empty((self.ks, M, N) if self.ks > 1 else (M, N), dtype=dt, device=device)
        elif self.template == "dual_gemv_swiglu":
            self.y = (torch.empty(self.ks, 2, M, N, dtype=torch.float32, device=device) if self.ks > 1
                      else torch.empty(M, N, dtype=torch.bfloat16, device=device))
        elif self.template == "gemv_pe":
            self.hd, self.rd = int(c["hd"]), int(c["rd"])
            self.freqs = torch.randn(4096, self.rd // 2, 2, device=device)
            self.start = torch.zeros(M, dtype=torch.int64, device=device)
            self.y = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        else:
            raise ValueError(f"shape class {key!r}: unknown template {self.template}")

    def check(self):
        """The blocks must fit the class the way the dispatchers require."""
        M, N, K = self.M, self.N, self.K
        bn, bk, ks = self.bn, self.bk, self.ks
        if N % bn or K % ks or (K // ks) % bk or bk % 32 or bk > K:
            raise ValueError(f"BN={bn} BK={bk} KSPLIT={ks} do not divide N={N}, K={K}")
        if self.template == "dual_gemv_swiglu" and bn % 32 and ks == 1:
            raise ValueError(f"dual GEMV: BN={bn} must hold whole quantization blocks of 32 when its epilogue is fused")
        if self.template == "gemv_rows":
            if self.quant and bn % 32:
                raise ValueError(f"quantizing epilogue: BN={bn} must hold whole quantization blocks of 32")
            if ks > 1 and (self.quant or self.hc or self.c.get("split") != 1):
                raise ValueError(f"KSPLIT={ks} on a class whose caller takes no partial sums")
            if self.groups > 1 and (self.N // self.groups) % bn:
                raise ValueError(f"BN={bn} must divide a group's rows {self.N // self.groups}")
        if self.template == "gemv_pe":
            hd, rd = int(self.c["hd"]), int(self.c["rd"])
            if (hd - rd) % bn or bn % 2 or ks != 1:
                raise ValueError(f"rotary epilogue: BN={bn} must be even and divide {hd - rd}; no K split")

    def launch(self, i):
        opts = T._launch_opts(self.warps, self.stages)
        M, N, K = self.M, self.N, self.K
        tiles = T.row_tiles(M, self.bm)  # more rows than a pass: the row-tile axis first (hero run 12)
        if self.template == "gemv_rows":
            w, s = self.weights[i]
            s_arg = s if s is not None else w
            if self.hc:
                res, post, comb, add = self.residual, self.post, self.comb, self.add
            else:
                res = post = comb = add = self.x
            R = N // self.groups
            T._gemv_rows_kernel[T._grid(tiles, N // self.bn, self.ks)](
                self.x, self.xq, self.xe, w, s_arg, self.y, res, post, comb, add, M, N, K, self.x.shape[1], R,
                K // 32 if s is not None else 0, HAS_SCALE=s is not None, BM=self.bm, BN=self.bn, BK=self.bk,
                EPI_QUANT=self.quant, KSPLIT=self.ks, EPI_HC=self.hc, HAS_ADD=bool(self.hc), NATIVE=self.native,
                RECODE=False, MT=tiles > 1, **opts)
        elif self.template == "dual_gemv_swiglu":
            w1, s1, w3, s3 = self.weights[i]
            T._dual_gemv_swiglu_kernel[T._grid(tiles, N // self.bn, self.ks)](
                self.x, self.xq, self.xe, w1, s1, w3, s3, self.y, M, N, K, K // 32, 7.0, BM=self.bm, BN=self.bn,
                BK=self.bk, KSPLIT=self.ks, NATIVE=self.native, RECODE=False, MT=tiles > 1, **opts)
        else:
            w, s = self.weights[i]
            T._gemv_pe_kernel[T._grid(tiles, N // self.bn)](
                self.x, self.xq, self.xe, w, s, self.y, self.y, self.x, self.freqs, self.start, 1.0, M, N, K, K // 32,
                N, 1, 0.0, PRO_NORM=False, PRO_ROTARY=False, PRO_QUANT=False, STORE_XN=False, EPI_ROTARY=True,
                HD=self.hd, RD=self.rd, BM=self.bm, BN=self.bn, BK=self.bk, NATIVE=self.native, RECODE=False,
                MT=tiles > 1, **opts)


def time_job(key, blocks, device, l2_bytes=L2_BYTES_DEFAULT, launches=48, replays=3):
    """Microseconds a launch of this class under these blocks: `launches` launches over cycled
    weights captured in one CUDA graph, the graph replayed `replays` times, the median replay
    divided by the launches. Returns (us, us_best_replay)."""
    dev = torch.device(device)
    with torch.cuda.device(dev):
        job = _Job(key, blocks, dev, l2_bytes)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for i in range(3):  # compile and warm
                job.launch(i % job.copies)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for i in range(launches):
                job.launch(i % job.copies)
        torch.cuda.synchronize(dev)
        graph.replay()
        torch.cuda.synchronize(dev)
        times = []
        for _ in range(replays):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize(dev)
            times.append(s.elapsed_time(e) * 1e3 / launches)
        del graph, job
        torch.cuda.empty_cache()
    return statistics.median(times), min(times)


def run_jobs(jobs, device, out, gbps=None):
    """Time each job, one JSON line per job to `out` (a file object): shapes, blocks, us, us_min,
    bytes, gbps, floor_us, efficiency; or error."""
    props = torch.cuda.get_device_properties(torch.device(device))
    l2 = getattr(props, "L2_cache_size", L2_BYTES_DEFAULT) or L2_BYTES_DEFAULT
    bw = gbps or bandwidth_gbps(device)
    done = 0
    for job in jobs:
        key, blocks = job["shapes"], job["blocks"]
        line = {"shapes": key, "blocks": blocks, "gpu": props.name, "bandwidth_gbps": round(bw, 1)}
        try:
            nbytes = class_bytes(key)
            us, best = time_job(key, blocks, device, l2)
            floor = nbytes / bw / 1e3
            line.update(us=round(us, 2), us_min=round(best, 2), bytes=nbytes, gbps=round(nbytes / us / 1e3, 1),
                        floor_us=round(floor, 2), efficiency=round(floor / us, 3))
        except Exception as e:  # a block set the kernel cannot compile or run is a result too
            line["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            torch.cuda.empty_cache()
        out.write(json.dumps(line) + "\n")
        out.flush()
        done += 1
        print(f"[measure] {done}/{len(jobs)} {key} {blocks}: {line.get('us', line.get('error'))}", file=sys.stderr, flush=True)
    return done


def classes_report(record, gbps):
    """The GEMV classes of a record with their bytes, floor and the blocks that ran."""
    classes = (record.get("stats") or {}).get("gemv_classes") or {}
    rows = []
    for key, entry in classes.items():
        nbytes = class_bytes(key)
        rows.append({"shapes": key, "count": entry.get("sites") or entry.get("count"), "blocks": entry.get("blocks"), "bytes": nbytes,
                     "floor_us": round(nbytes / gbps / 1e3, 2), "from_table": entry.get("from_table", False)})
    rows.sort(key=lambda r: -r["bytes"] * (r["count"] or 1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vxpu.measure", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classes", help="the GEMV shape classes in a kernel record, with bytes and floors")
    c.add_argument("record")
    c.add_argument("--gbps", type=float, default=1460.0, help="the card's bandwidth the floors use (default 1,460)")
    r = sub.add_parser("run", help="time jobs: a JSON list of {shapes, blocks}")
    r.add_argument("jobs")
    r.add_argument("--out", default="-")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--gbps", type=float, default=None, help="the floors' bandwidth (default: measured now)")
    b = sub.add_parser("bandwidth", help="the card's copy bandwidth")
    b.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.cmd == "classes":
        with open(args.record) as f:
            record = json.load(f)
        rows = classes_report(record, args.gbps)
        total = sum(r["bytes"] * r["count"] for r in rows)
        print(f"{len(rows)} GEMV shape classes, {total / 1e9:.2f} GB a step at these counts, "
              f"floor {total / args.gbps / 1e6:.2f} ms at {args.gbps:.0f} GB/s")
        for r in rows:
            print(f"{r['count']:4d}x {r['bytes'] / 1e6:8.1f} MB  floor {r['floor_us']:7.2f} us  "
                  f"{r['blocks']}  {r['shapes']}")
        return 0
    if args.cmd == "bandwidth":
        print(json.dumps({"device": args.device, "gbps": round(bandwidth_gbps(args.device), 1)}))
        return 0
    with open(args.jobs) as f:
        jobs = json.load(f)
    out = sys.stdout if args.out == "-" else open(args.out, "a")
    t0 = time.time()
    n = run_jobs(jobs, args.device, out, args.gbps)
    print(f"[measure] {n} jobs in {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
