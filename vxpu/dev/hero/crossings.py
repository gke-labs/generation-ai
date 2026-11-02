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

"""The cost of crossing between the cards of one machine, measured: what a tensor-parallel
dense step would pay per layer on this box (hero run 14's input).

    python -m vxpu.dev.hero.crossings --out crossings.json [--rounds 200] [--chain 86]

For every pair of cards: the one-way latency of a flag (a 4-byte release store from a kernel on
one card into the other's memory, an acquire spin on the other), the peer copy bandwidth and
latency at the payloads a decode step moves (one row of 5120 bf16 up to 132 rows, and 10 MB),
as a peer memcpy and as a kernel doing peer stores. Then a one-shot all-reduce across all four
cards (every card pushes its partial into the other three cards' inboxes and raises a flag per
destination; each card waits for three flags and sums the four partials locally), once and in
a chain of 86 dependent rounds with a small GEMV between them, the per-step shape of a
tensor-parallel dense path (43 layers, two reductions a layer), at 1 row and 132 rows. Every
figure is a median over the rounds with the p90 beside it; the chain is timed eagerly on
streams and, where the capture succeeds, as one CUDA graph per card replayed together.
"""

import argparse
import json
import os
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl

ROW = 5120  # one row's bf16 elements
SIZES = {"10KB (1 row)": ROW, "80KB (8 rows)": 8 * ROW, "330KB (33 rows)": 33 * ROW,
         "1.3MB (132 rows)": 132 * ROW, "10MB": 5 * 1024 * 1024}


@triton.jit
def _signal(flag_ptr, val):
    tl.atomic_xchg(flag_ptr, val, sem="release", scope="sys")


@triton.jit
def _wait(flag_ptr, val):
    while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys") < val:
        pass


@triton.jit
def _pingpong(my_flag_ptr, peer_flag_ptr, rounds, first: tl.constexpr):
    # first: write peer's flag then wait for mine; else wait for mine then write peer's
    for r in range(1, rounds + 1):
        if first:
            tl.atomic_xchg(peer_flag_ptr, r, sem="release", scope="sys")
            while tl.atomic_add(my_flag_ptr, 0, sem="acquire", scope="sys") < r:
                pass
        else:
            while tl.atomic_add(my_flag_ptr, 0, sem="acquire", scope="sys") < r:
                pass
            tl.atomic_xchg(peer_flag_ptr, r, sem="release", scope="sys")


@triton.jit
def _push(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)


@triton.jit
def _push3(src_ptr, d0, d1, d2, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src_ptr + offs, mask=m)
    tl.store(d0 + offs, v, mask=m)
    tl.store(d1 + offs, v, mask=m)
    tl.store(d2 + offs, v, mask=m)


@triton.jit
def _signal3(f0, f1, f2, val):
    tl.atomic_xchg(f0, val, sem="release", scope="sys")
    tl.atomic_xchg(f1, val, sem="release", scope="sys")
    tl.atomic_xchg(f2, val, sem="release", scope="sys")


@triton.jit
def _wait3(f0, f1, f2, val):
    while tl.atomic_add(f0, 0, sem="acquire", scope="sys") < val:
        pass
    while tl.atomic_add(f1, 0, sem="acquire", scope="sys") < val:
        pass
    while tl.atomic_add(f2, 0, sem="acquire", scope="sys") < val:
        pass


@triton.jit
def _sum4(a, b, c, d, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = (tl.load(a + offs, mask=m).to(tl.float32) + tl.load(b + offs, mask=m).to(tl.float32)
         + tl.load(c + offs, mask=m).to(tl.float32) + tl.load(d + offs, mask=m).to(tl.float32))
    tl.store(out + offs, s.to(tl.bfloat16), mask=m)


def _enable_peers(n):
    """A cross-device copy per pair makes torch enable peer access, after which raw peer
    pointers work inside kernels."""
    for i in range(n):
        for j in range(n):
            if i != j:
                torch.empty(1, device=i).copy_(torch.empty(1, device=j))
    torch.cuda.synchronize()


def _med(xs):
    xs = sorted(xs)
    return {"median_us": statistics.median(xs) * 1e3, "p90_us": xs[int(0.9 * (len(xs) - 1))] * 1e3,
            "min_us": xs[0] * 1e3, "n": len(xs)}


def topology(n):
    out = {"pairs": {}}
    try:
        out["nvidia_smi_topo"] = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                                text=True, timeout=30).stdout
    except Exception as e:  # pragma: no cover
        out["nvidia_smi_topo"] = f"unavailable: {e}"
    for i in range(n):
        for j in range(n):
            if i < j:
                out["pairs"][f"{i}-{j}"] = {"peer": bool(torch.cuda.can_device_access_peer(i, j)),
                                           "peer_back": bool(torch.cuda.can_device_access_peer(j, i))}
    out["devices"] = [torch.cuda.get_device_name(i) for i in range(n)]
    return out


def flag_latency(i, j, rounds):
    """Ping-pong of flags between card i and card j, rounds times; the round trip is the total
    over the rounds, one way is half. Eager: two kernels spinning on two cards at once (a
    captured graph could not start both sides together)."""
    fi = torch.zeros(1, dtype=torch.int32, device=i)
    fj = torch.zeros(1, dtype=torch.int32, device=j)
    si, sj = torch.cuda.Stream(device=i), torch.cuda.Stream(device=j)
    times = []
    for _ in range(5):
        fi.zero_()
        fj.zero_()
        torch.cuda.synchronize(i)
        torch.cuda.synchronize(j)
        with torch.cuda.device(j), torch.cuda.stream(sj):
            _pingpong[(1,)](fj, fi, rounds, first=False, num_warps=1)
        with torch.cuda.device(i), torch.cuda.stream(si):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(si)
            _pingpong[(1,)](fi, fj, rounds, first=True, num_warps=1)
            end.record(si)
        torch.cuda.synchronize(i)
        torch.cuda.synchronize(j)
        times.append(start.elapsed_time(end) / rounds)  # ms per round trip
    rt = statistics.median(times)
    return {"round_trip_us": rt * 1e3, "one_way_us": rt * 1e3 / 2, "rounds": rounds,
            "samples_ms_per_rt": times}


def peer_copy(i, j, n, rounds):
    """Card i's tensor into card j's memory: torch copy_ (cudaMemcpyPeerAsync) and a kernel on
    card i storing into the peer; timed with events on card i, median of rounds."""
    src = torch.randn(n, dtype=torch.bfloat16, device=i)
    dst = torch.empty(n, dtype=torch.bfloat16, device=j)
    s = torch.cuda.Stream(device=i)
    res = {}
    for name in ("memcpy", "kernel"):
        times = []
        with torch.cuda.device(i), torch.cuda.stream(s):
            for _ in range(rounds + 10):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(s)
                if name == "memcpy":
                    dst.copy_(src, non_blocking=True)
                else:
                    _push[(triton.cdiv(n, 4096),)](src, dst, n, BLOCK=4096, num_warps=4)
                en.record(s)
                en.synchronize()
                times.append(st.elapsed_time(en))
        times = times[10:]
        m = _med(times)
        m["gb_s"] = (2 * n) / (m["median_us"] * 1e-6) / 1e9
        res[name] = m
    torch.cuda.synchronize(i)
    torch.cuda.synchronize(j)
    assert torch.equal(dst.to(i), src), "peer copy corrupted"
    return res


class AllReduce:
    """One-shot all-reduce across n cards of a bf16 payload of `count` elements: each card
    pushes its partial into the other cards' inboxes (peer stores), raises a flag per
    destination (release, system scope), waits for the others' flags (acquire) and sums the
    four partials into its output. Two kernels per card per round plus the flags."""

    def __init__(self, n, count):
        self.n, self.count = n, count
        self.part = [torch.randn(count, dtype=torch.bfloat16, device=k) for k in range(n)]
        self.out = [torch.empty(count, dtype=torch.bfloat16, device=k) for k in range(n)]
        # inbox[j][k]: card k's partial as seen on card j; flag[j][k] on card j, written by card k
        self.inbox = [[torch.empty(count, dtype=torch.bfloat16, device=j) if k != j else None for k in range(n)]
                      for j in range(n)]
        self.flag = [torch.zeros(n, dtype=torch.int32, device=j) for j in range(n)]
        self.streams = [torch.cuda.Stream(device=k) for k in range(n)]
        self.seq = 0
        self.block = 4096

    def others(self, k):
        return [j for j in range(self.n) if j != k]

    def push(self, k, seq):
        o = self.others(k)
        g = (triton.cdiv(self.count, self.block),)
        with torch.cuda.device(k):
            _push3[g](self.part[k], self.inbox[o[0]][k], self.inbox[o[1]][k], self.inbox[o[2]][k], self.count,
                      BLOCK=self.block, num_warps=4)
            _signal3[(1,)](self.flag[o[0]][k:k + 1], self.flag[o[1]][k:k + 1], self.flag[o[2]][k:k + 1], seq,
                           num_warps=1)

    def gather(self, k, seq):
        o = self.others(k)
        g = (triton.cdiv(self.count, self.block),)
        with torch.cuda.device(k):
            _wait3[(1,)](self.flag[k][o[0]:o[0] + 1], self.flag[k][o[1]:o[1] + 1], self.flag[k][o[2]:o[2] + 1], seq,
                         num_warps=1)
            _sum4[g](self.part[k], self.inbox[k][o[0]], self.inbox[k][o[1]], self.inbox[k][o[2]], self.out[k],
                     self.count, BLOCK=self.block, num_warps=4)

    def round(self, seq):
        for k in range(self.n):
            with torch.cuda.stream(self.streams[k]):
                self.push(k, seq)
        for k in range(self.n):
            with torch.cuda.stream(self.streams[k]):
                self.gather(k, seq)

    def check(self):
        ref = sum(p.to(0).float() for p in self.part).to(torch.bfloat16)
        for k in range(self.n):
            assert torch.equal(self.out[k].to(0), ref), f"all-reduce wrong on card {k}"

    def sync(self):
        for k in range(self.n):
            torch.cuda.synchronize(k)


def allreduce_once(n, count, rounds):
    """The one-shot all-reduce's wall time per round as card 0 sees it (card 0's round cannot
    finish before every card has pushed), eager on streams, plus the push and the sum phases
    timed alone (no waits) for the breakdown."""
    ar = AllReduce(n, count)
    times = []
    s0 = ar.streams[0]
    for r in range(rounds + 10):
        ar.seq += 1
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(s0)
        ar.round(ar.seq)
        en.record(s0)
        ar.sync()
        if r >= 10:
            times.append(st.elapsed_time(en))
    ar.check()
    res = {"wall": _med(times)}
    # breakdown: the push phase alone (stores + flags, no waits) and the sum alone
    for name in ("push", "sum"):
        ts = []
        for r in range(rounds + 10):
            ar.seq += 1
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s0):
                st.record(s0)
                if name == "push":
                    ar.push(0, ar.seq)
                else:
                    o = ar.others(0)
                    with torch.cuda.device(0):
                        _sum4[(triton.cdiv(count, ar.block),)](ar.part[0], ar.inbox[0][o[0]], ar.inbox[0][o[1]],
                                                               ar.inbox[0][o[2]], ar.out[0], count,
                                                               BLOCK=ar.block, num_warps=4)
                en.record(s0)
            en.synchronize()
            if r >= 10:
                ts.append(st.elapsed_time(en))
        res[name] = _med(ts)
    res["wait_us"] = res["wall"]["median_us"] - res["push"]["median_us"] - res["sum"]["median_us"]
    return res


def chain(n, rows, steps, rounds, graph):
    """`steps` dependent all-reduces of rows × 5120 bf16 with a small GEMV between them on every
    card (rows × 5120 by 5120 × 1280, the attention front's shape), the per-step shape of a
    tensor-parallel dense path; wall time per chain on card 0, eager and as one CUDA graph per
    card replayed together."""
    count = rows * ROW
    ar = AllReduce(n, count)
    w = [torch.randn(ROW, 1280, dtype=torch.bfloat16, device=k) for k in range(n)]
    x = [torch.randn(rows, ROW, dtype=torch.bfloat16, device=k) for k in range(n)]

    def body(seq0):
        for s in range(steps):
            seq = seq0 + s
            for k in range(n):
                with torch.cuda.device(k), torch.cuda.stream(ar.streams[k]):
                    torch.matmul(x[k], w[k])  # the layer's work between two reductions
                    ar.push(k, seq)
            for k in range(n):
                with torch.cuda.stream(ar.streams[k]):
                    ar.gather(k, seq)

    res = {"rows": rows, "steps": steps}
    for k in range(n):  # cuBLAS handles and kernels warm before any capture
        with torch.cuda.device(k), torch.cuda.stream(ar.streams[k]):
            torch.matmul(x[k], w[k])
    ar.sync()
    # the GEMVs alone on card 0, captured, so the crossing share of the chain is the difference
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.device(0), torch.cuda.graph(g0, stream=ar.streams[0]):
        for s in range(steps):
            torch.matmul(x[0], w[0])
    torch.cuda.synchronize(0)
    times = []
    for r in range(rounds + 3):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(0), torch.cuda.stream(ar.streams[0]):
            st.record(ar.streams[0])
            g0.replay()
            en.record(ar.streams[0])
        torch.cuda.synchronize(0)
        if r >= 3:
            times.append(st.elapsed_time(en))
    m = _med(times)
    m["per_step_us"] = m["median_us"] / steps
    res["gemv_only_graph"] = m
    times = []
    for r in range(rounds + 3):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(ar.streams[0])
        body(ar.seq + 1)
        ar.seq += steps
        en.record(ar.streams[0])
        ar.sync()
        if r >= 3:
            times.append(st.elapsed_time(en))
    ar.check()
    m = _med(times)
    m["per_step_us"] = m["median_us"] / steps
    res["eager"] = m
    if graph:
        try:
            # one graph per card: the flags advance by `steps` per replay, so the flag values
            # baked into the capture must be re-based: capture with seq values 1..steps on
            # fresh flags and reset the flags to zero before every replay
            for k in range(n):
                ar.flag[k].zero_()
            ar.sync()
            graphs = []
            for k in range(n):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.device(k), torch.cuda.graph(g, stream=ar.streams[k]):
                    for s in range(steps):
                        torch.matmul(x[k], w[k])
                        ar.push(k, s + 1)
                        ar.gather(k, s + 1)
                graphs.append(g)
            ar.sync()
            times = []
            for r in range(rounds + 3):
                for k in range(n):
                    ar.flag[k].zero_()
                ar.sync()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(ar.streams[0])
                for k in range(n):
                    with torch.cuda.device(k), torch.cuda.stream(ar.streams[k]):
                        graphs[k].replay()
                en.record(ar.streams[0])
                ar.sync()
                if r >= 3:
                    times.append(st.elapsed_time(en))
            ar.check()
            m = _med(times)
            m["per_step_us"] = m["median_us"] / steps
            res["graph"] = m
        except Exception as e:  # the capture or the replay refused: say so, keep the eager number
            res["graph"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    return res


def ring_allreduce(n, count, rounds):
    """A ring all-reduce (reduce-scatter then all-gather over 2(n-1) steps of count/n elements)
    with peer stores and flags, for the bandwidth-bound sizes; wall time per round on card 0."""
    chunk = count // n
    part = [torch.randn(count, dtype=torch.bfloat16, device=k) for k in range(n)]
    acc = [p.float() for p in part]  # fp32 accumulators per card
    inbox = [torch.empty(chunk, dtype=torch.float32, device=k) for k in range(n)]
    flag = [torch.zeros(1, dtype=torch.int32, device=k) for k in range(n)]
    streams = [torch.cuda.Stream(device=k) for k in range(n)]
    seq = 0

    def step(k, src_chunk, dst_card, s, add):
        # card k sends its chunk src_chunk to dst_card's inbox, signals, then waits for its own inbox
        with torch.cuda.device(k), torch.cuda.stream(streams[k]):
            _push[(triton.cdiv(chunk, 4096),)](acc[k][src_chunk * chunk:(src_chunk + 1) * chunk], inbox[dst_card],
                                               chunk, BLOCK=4096, num_warps=4)
            _signal[(1,)](flag[dst_card], s, num_warps=1)

    def recv(k, into_chunk, s, add):
        with torch.cuda.device(k), torch.cuda.stream(streams[k]):
            _wait[(1,)](flag[k], s, num_warps=1)
            seg = acc[k][into_chunk * chunk:(into_chunk + 1) * chunk]
            if add:
                seg.add_(inbox[k])
            else:
                seg.copy_(inbox[k])

    times = []
    for r in range(rounds + 5):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(streams[0])
        # reduce-scatter
        for i in range(n - 1):
            seq += 1
            for k in range(n):
                step(k, (k - i) % n, (k + 1) % n, seq, True)
            for k in range(n):
                recv(k, (k - i - 1) % n, seq, True)
        # all-gather
        for i in range(n - 1):
            seq += 1
            for k in range(n):
                step(k, (k - i + 1) % n, (k + 1) % n, seq, False)
            for k in range(n):
                recv(k, (k - i) % n, seq, False)
        en.record(streams[0])
        for k in range(n):
            torch.cuda.synchronize(k)
        if r >= 5:
            times.append(st.elapsed_time(en))
    return _med(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--chain", type=int, default=86)
    p.add_argument("--chain-rounds", type=int, default=20)
    p.add_argument("--skip-ring", action="store_true")
    p.add_argument("--only-chain", action="store_true", help="the chain and its GEMV baseline only")
    args = p.parse_args()
    n = torch.cuda.device_count()
    assert n >= 2, "two cards at least"
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cards": n}

    def save(stage):
        res["stage"] = stage
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[crossings] {stage} written", flush=True)

    res["topology"] = topology(n)
    _enable_peers(n)
    save("topology")
    if args.only_chain:
        res["chain"] = {}
        for rows in (1, 8, 33, 132):
            r = chain(4, rows, args.chain, args.chain_rounds, graph=True)
            res["chain"][str(rows)] = r
            g = r.get("graph", {})
            print(f"[crossings] chain {rows} rows x {args.chain}: gemv alone {r['gemv_only_graph']['per_step_us']:.1f} us a step; "
                  f"graph {g.get('per_step_us', 0):.1f} us a step; eager {r['eager']['per_step_us']:.1f}", flush=True)
        save("done")
        return

    res["flag_latency"] = {}
    for i in range(n):
        for j in range(i + 1, n):
            res["flag_latency"][f"{i}-{j}"] = flag_latency(i, j, args.rounds)
            print(f"[crossings] flag {i}-{j}: one way {res['flag_latency'][f'{i}-{j}']['one_way_us']:.1f} us", flush=True)
    save("flags")

    res["peer_copy"] = {}
    for pair in ((0, 1), (0, 2)):  # a PIX pair and a SYS pair
        i, j = pair
        res["peer_copy"][f"{i}-{j}"] = {}
        for name, count in SIZES.items():
            r = peer_copy(i, j, count, args.rounds)
            res["peer_copy"][f"{i}-{j}"][name] = r
            print(f"[crossings] copy {i}-{j} {name}: memcpy {r['memcpy']['median_us']:.1f} us {r['memcpy']['gb_s']:.1f} GB/s, "
                  f"kernel {r['kernel']['median_us']:.1f} us {r['kernel']['gb_s']:.1f} GB/s", flush=True)
    save("copies")

    if n >= 4:
        res["allreduce_once"] = {}
        for name, count in SIZES.items():
            r = allreduce_once(4, count, args.rounds)
            res["allreduce_once"][name] = r
            print(f"[crossings] all-reduce {name}: wall {r['wall']['median_us']:.1f} us (p90 {r['wall']['p90_us']:.1f}); "
                  f"push {r['push']['median_us']:.1f}, sum {r['sum']['median_us']:.1f}, wait {r['wait_us']:.1f}", flush=True)
        save("allreduce")

        res["chain"] = {}
        for rows in (1, 8, 33, 132):
            r = chain(4, rows, args.chain, args.chain_rounds, graph=True)
            res["chain"][str(rows)] = r
            g = r.get("graph", {})
            print(f"[crossings] chain {rows} rows x {args.chain}: gemv alone {r['gemv_only_graph']['per_step_us']:.1f} us a step; "
                  f"eager {r['eager']['median_us'] / 1e3:.2f} ms "
                  f"({r['eager']['per_step_us']:.1f} us a step); graph "
                  f"{(g.get('median_us', 0) / 1e3):.2f} ms ({g.get('per_step_us', 0):.1f} us a step) {g.get('error', '')}",
                  flush=True)
        save("chain")

        if not args.skip_ring:
            res["ring"] = {}
            for name in ("1.3MB (132 rows)", "10MB"):
                r = ring_allreduce(4, SIZES[name], max(20, args.rounds // 4))
                res["ring"][name] = r
                print(f"[crossings] ring {name}: {r['median_us']:.1f} us (p90 {r['p90_us']:.1f})", flush=True)
            save("ring")
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save("done")


if __name__ == "__main__":
    main()
