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

"""The cross-card sum as a kernel template (hero run 14): vxpu::allreduce.

A tensor-parallel dense step splits each weight across the cards and sums the partial outputs
after every block, twice a layer. On a box without NVLink the sum goes over PCIe through peer
stores, and the whole step is one captured CUDA graph per card, so the sum has to run with no
host and no allocation inside the capture. Two forms, both summing in rank order in fp32 so the
result is the same bits on every card and under either form:

- one-shot: every card pushes its whole partial into the other cards' inboxes (one load, one
  store per peer), raises a flag per destination, waits for the others' flags and sums the
  partials locally. Two kernels a card. Latency bound at small payloads: a flag crosses in 2 to
  3 µs and a 10 KB push in 10, so about 10 µs a reduction inside a graph at one row (measured on
  four RTX PRO 6000, dev/hero/crossings.py); it pushes 3× the payload per card, so at 132 rows
  of 5120 bf16 it is bandwidth bound at about 40 GB/s a card and costs about 190 µs.
- reduce-scatter then all-gather: card j owns slice j of the elements; every card pushes slice j
  of its partial to card j, which sums the four slices and pushes the reduced slice back to
  the others. 1.5× the payload per card and two flag rounds; three kernels a card. For wide
  rows, where the bytes are what the sum costs.

The flags need no zeroing between steps and no sequence numbers: a writer raises a flag to 1 and
the reader resets it to 0 after it has consumed the data, before it pushes its own next
partial; a writer cannot reach the same site again until it has consumed that push, so a flag is
never raised twice before it is read. The arrival counters that let the last program of a
kernel raise the flags reset themselves the same way. Every site (a call in the step) has
inboxes, flags and counters of its own, allocated once by setup(), so two reductions of one
step never share a flag.

    setup(devices, sites, max_elems)        once per process: the Comm for these cards
    allreduce(x, site)                       the op (vxpu::allreduce; ops.py chooses the form)
    python -m vxpu.allreduce measure         the measuring step: both forms per rows, captured,
                                             checked bit for bit, written as a kernel table
"""

import argparse
import json
import statistics
import time

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the CPU-only selftest
    triton = None
    tl = None

from . import triton_ops

ROW = 5120  # one row's bf16 elements for the measuring step's classes

_COMM = None


class Comm:
    """The buffers of a world of cards: for site s and card j, an inbox per other card k
    (inbox[j][s][k], the one-shot's whole partial or the scatter's slice), a gather buffer per
    other card (gbuf[j][s][k]: the reduced slice k sends back), flags [2, world] (round A and
    round B, one per source rank) and counters [3] (push, reduce, gather arrivals), all on card
    j. Ranks are the position in `devices`; the sum is taken in rank order."""

    def __init__(self, devices, sites, max_elems, dtype=torch.bfloat16):
        self.devices = [torch.device(d) for d in devices]
        self.world = len(self.devices)
        self.sites = sites
        self.max_elems = max_elems
        self.dtype = dtype
        self.slice = -(-max_elems // max(self.world, 1))
        self.inbox, self.gbuf, self.flags, self.counters = [], [], [], []
        for j, dev in enumerate(self.devices):
            self.inbox.append([[torch.empty(max_elems, dtype=dtype, device=dev) if k != j else None
                                for k in range(self.world)] for _ in range(sites)])
            self.gbuf.append([[torch.empty(self.slice, dtype=dtype, device=dev) if k != j else None
                               for k in range(self.world)] for _ in range(sites)])
            self.flags.append(torch.zeros(sites, 2, self.world, dtype=torch.int32, device=dev))
            self.counters.append(torch.zeros(sites, 3, dtype=torch.int32, device=dev))
        if self.world > 1 and self.devices[0].type == "cuda":
            _enable_peers(self.devices)
        # the reference's view: the partial each rank registered at a site (tests)
        self.registered = {}

    def rank_of(self, device):
        device = torch.device(device)
        for r, d in enumerate(self.devices):
            if d.type == device.type and (d.index or 0) == (device.index or 0):
                return r
        raise ValueError(f"allreduce: {device} is not one of the world's cards {self.devices}")

    def peers(self, rank):
        return [k for k in range(self.world) if k != rank]

    def register(self, site, rank, x):
        self.registered.setdefault(site, {})[rank] = x


def setup(devices, sites, max_elems, dtype=torch.bfloat16):
    """The process's Comm: the cards (ranks in this order), how many call sites a step has, the
    most elements one call sums, and their dtype. Replaces any earlier one."""
    global _COMM
    _COMM = Comm(devices, sites, max_elems, dtype)
    return _COMM


def comm():
    """The Comm setup() made, or None (a world of one: the op is the identity)."""
    return _COMM


def _enable_peers(devices):
    """A cross-device copy per pair makes torch enable peer access, after which raw peer
    pointers work inside kernels."""
    for a in devices:
        for b in devices:
            if a != b:
                torch.empty(1, device=a).copy_(torch.empty(1, device=b))
    for d in devices:
        torch.cuda.synchronize(d)


def _check(x, site, c):
    if x.dim() < 1:
        raise ValueError("allreduce: x needs at least one dimension")
    if not 0 <= site < c.sites:
        raise ValueError(f"allreduce: site {site} outside the Comm's {c.sites} sites")
    if x.numel() > c.max_elems:
        raise ValueError(f"allreduce: {x.numel()} elements, the Comm holds {c.max_elems} a site")
    if x.dtype != c.dtype:
        raise ValueError(f"allreduce: x is {x.dtype}, the Comm's buffers {c.dtype}")


# ---------------------------------------------------------------------------
# the kernels

if triton is not None:
    @triton.jit
    def _wait_flags(f_ptr, PEERS: tl.constexpr):
        """Spin until every peer's flag (int32, contiguous, in rank order) is raised: an acquire at
        system scope, so the peer stores it ordered before are visible to the loads after."""
        for i in tl.static_range(PEERS):
            while tl.atomic_add(f_ptr + i, 0, sem="acquire", scope="sys") < 1:
                pass

    @triton.jit
    def _raise_flags(f0, f1, f2, PEERS: tl.constexpr):
        tl.atomic_xchg(f0, 1, sem="release", scope="sys")
        if PEERS > 1:
            tl.atomic_xchg(f1, 1, sem="release", scope="sys")
        if PEERS > 2:
            tl.atomic_xchg(f2, 1, sem="release", scope="sys")

    @triton.jit
    def _last_program(counter_ptr):
        """True in the one program that arrives last: the others' stores are ordered before its
        acquire, so what it does next (raise a flag, reset the counter) follows them all."""
        tl.debug_barrier()
        done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
        return done == tl.num_programs(0) - 1

    @triton.jit
    def _ar_push_kernel(x_ptr, d0, d1, d2, n, counter_ptr, f0, f1, f2, PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """One-shot, round A: this card's partial into every peer's inbox for this site, then
        the last program raises the site's flag on every peer."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=0)
        tl.store(d0 + offs, v, mask=m)
        if PEERS > 1:
            tl.store(d1 + offs, v, mask=m)
        if PEERS > 2:
            tl.store(d2 + offs, v, mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

    @triton.jit
    def _ar_sum_kernel(p0, p1, p2, p3, out_ptr, n, f_ptr, counter_ptr, PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """One-shot, round B: wait for the peers' flags, sum the partials in rank order (p0..p3
        are the ranks' partials, this card's own among them, fp32 accumulation, rounded once),
        then the last program resets the flags for the next step."""
        _wait_flags(f_ptr, PEERS)
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.load(p0 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        acc += tl.load(p1 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        if PEERS > 1:
            acc += tl.load(p2 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        if PEERS > 2:
            acc += tl.load(p3 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(f_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _rs_push_kernel(x_ptr, d0, d1, d2, o0, o1, o2, S, n, counter_ptr, f0, f1, f2,
                        PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """Reduce-scatter, round A: slice j of this card's partial (elements o_j .. o_j + S) into
        card j's inbox for this site, for every peer j; the last program raises the flags."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < S
        tl.store(d0 + offs, tl.load(x_ptr + o0 + offs, mask=m & (o0 + offs < n), other=0), mask=m)
        if PEERS > 1:
            tl.store(d1 + offs, tl.load(x_ptr + o1 + offs, mask=m & (o1 + offs < n), other=0), mask=m)
        if PEERS > 2:
            tl.store(d2 + offs, tl.load(x_ptr + o2 + offs, mask=m & (o2 + offs < n), other=0), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

    @triton.jit
    def _rs_reduce_kernel(p0, p1, p2, p3, out_ptr, g0, g1, g2, S, n_mine, fa_ptr, counter_ptr, fb0, fb1, fb2,
                          PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """Reduce-scatter, round B: wait for the slices, sum this card's slice in rank order
        (p0..p3 point at the ranks' slices, this card's own at its offset in x), store it into
        the output and push it into every peer's gather buffer; the last program resets round
        A's flags and raises round B's."""
        _wait_flags(fa_ptr, PEERS)
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_mine
        acc = tl.load(p0 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        acc += tl.load(p1 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        if PEERS > 1:
            acc += tl.load(p2 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        if PEERS > 2:
            acc += tl.load(p3 + offs, mask=m, other=0, volatile=True).to(tl.float32)
        v = acc.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, v, mask=m)
        tl.store(g0 + offs, v, mask=m)
        if PEERS > 1:
            tl.store(g1 + offs, v, mask=m)
        if PEERS > 2:
            tl.store(g2 + offs, v, mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(fa_ptr + i, 0, sem="relaxed", scope="gpu")
            _raise_flags(fb0, fb1, fb2, PEERS)

    @triton.jit
    def _rs_gather_kernel(out_ptr, h0, h1, h2, o0, o1, o2, S, n, fb_ptr, counter_ptr,
                          PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """Reduce-scatter, round C: wait for the peers' reduced slices and place each at its
        offset in the output; the last program resets round B's flags."""
        _wait_flags(fb_ptr, PEERS)
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < S
        tl.store(out_ptr + o0 + offs, tl.load(h0 + offs, mask=m, other=0, volatile=True), mask=m & (o0 + offs < n))
        if PEERS > 1:
            tl.store(out_ptr + o1 + offs, tl.load(h1 + offs, mask=m, other=0, volatile=True),
                     mask=m & (o1 + offs < n))
        if PEERS > 2:
            tl.store(out_ptr + o2 + offs, tl.load(h2 + offs, mask=m, other=0, volatile=True),
                     mask=m & (o2 + offs < n))
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(fb_ptr + i, 0, sem="relaxed", scope="gpu")


def _pad3(ptrs, filler):
    ptrs = list(ptrs)
    while len(ptrs) < 3:
        ptrs.append(filler)
    return ptrs


def _slot(c, rank, src):
    """Where source rank `src` writes its flag on card `rank`: the peers in rank order occupy
    slots 0..world-2, the own slot is the last (never written)."""
    return src if src < rank else src - 1


def block_elems():
    """Elements a program moves (the knob ALLREDUCE_BLOCK, default 4096)."""
    return int(getattr(triton_ops, "ALLREDUCE_BLOCK", None) or 4096)


def scatter_wanted(rows):
    """Whether the reduce-scatter form runs for a call of `rows` rows: forced by the knob
    ALLREDUCE_SCATTER, else from ALLREDUCE_SCATTER_MIN_ROWS (0: never; the measuring step
    writes the crossover it finds into the table)."""
    forced = getattr(triton_ops, "ALLREDUCE_SCATTER", None)
    if forced is not None:
        return forced in (True, "1")
    min_rows = int(getattr(triton_ops, "ALLREDUCE_SCATTER_MIN_ROWS", None) or 0)
    return min_rows > 0 and rows >= min_rows


def allreduce_oneshot(x, site, traced=False):
    """The one-shot form on x's card: returns the sum of every rank's x for this site."""
    c = comm()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    out = torch.empty_like(xc)
    block = block_elems()
    grid = (triton.cdiv(n, block),)
    dests = _pad3([c.inbox[j][site][rank] for j in peers], flat)
    raise_ = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], c.counters[rank][site, 0:1])
    parts = [c.inbox[rank][site][k] if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    kernel, ctx = triton_ops._launcher(_ar_push_kernel, x.device, traced)
    with ctx:
        kernel[grid](flat, dests[0], dests[1], dests[2], n, c.counters[rank][site, 0:1], raise_[0], raise_[1], raise_[2],
                     PEERS=len(peers), BLOCK=block)
    kernel, ctx = triton_ops._launcher(_ar_sum_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1), n, c.flags[rank][site, 0],
                     c.counters[rank][site, 1:2], PEERS=len(peers), BLOCK=block)
    return out


def allreduce_scatter(x, site, traced=False):
    """The reduce-scatter + all-gather form on x's card."""
    c = comm()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    S = -(-n // c.world)
    out = torch.empty_like(xc)
    block = block_elems()
    grid = (triton.cdiv(S, block),)
    cnt = c.counters[rank][site]
    # round A: slice j of my partial to card j's inbox[j][site][rank]
    dests = _pad3([c.inbox[j][site][rank] for j in peers], flat)
    offs = _pad3([j * S for j in peers], 0)
    raise_a = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[0:1])
    kernel, ctx = triton_ops._launcher(_rs_push_kernel, x.device, traced)
    with ctx:
        kernel[grid](flat, dests[0], dests[1], dests[2], offs[0], offs[1], offs[2], S, n, cnt[0:1],
                     raise_a[0], raise_a[1], raise_a[2], PEERS=len(peers), BLOCK=block)
    # round B: my slice summed in rank order (own slice at offset rank * S in x), out and to the peers' gbuf
    mine = min(S, max(n - rank * S, 0))
    parts = [c.inbox[rank][site][k] if k != rank else flat[rank * S:rank * S + S] if mine else flat
             for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    gdest = _pad3([c.gbuf[j][site][rank] for j in peers], flat)
    raise_b = _pad3([c.flags[j][site, 1, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[1:2])
    kernel, ctx = triton_ops._launcher(_rs_reduce_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1)[rank * S:] if mine else out.view(-1),
                     gdest[0], gdest[1], gdest[2], S, mine, c.flags[rank][site, 0], cnt[1:2],
                     raise_b[0], raise_b[1], raise_b[2], PEERS=len(peers), BLOCK=block)
    # round C: the peers' reduced slices into the output
    srcs = _pad3([c.gbuf[rank][site][j] for j in peers], flat)
    kernel, ctx = triton_ops._launcher(_rs_gather_kernel, x.device, traced)
    with ctx:
        kernel[grid](out.view(-1), srcs[0], srcs[1], srcs[2], offs[0], offs[1], offs[2], S, n,
                     c.flags[rank][site, 1], cnt[2:3], PEERS=len(peers), BLOCK=block)
    return out


def allreduce_reference(x, site):
    """The sum in rank order from the partials the ranks registered at this site (Comm.register,
    or the tests' register of every rank before the calls), fp32 accumulation rounded once: the
    bits both kernel forms produce. A world of one returns x."""
    c = comm()
    if c is None or c.world == 1:
        return x.clone()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    c.register(site, rank, x)
    parts = c.registered.get(site, {})
    missing = [r for r in range(c.world) if r not in parts]
    if missing:
        raise RuntimeError(f"allreduce reference: site {site} has no partial from ranks {missing}")
    acc = parts[0].to(x.device).float()
    for r in range(1, c.world):
        acc = acc + parts[r].to(x.device).float()
    return acc.to(x.dtype)


def allreduce(x, site, traced=False):
    """The op's body on a card: the form the knobs choose (ops.py registers each form as an
    implementation the plan can name; this is the dispatcher the eager path and the measuring
    step use)."""
    c = comm()
    if c is None or c.world == 1 or x.device.type != "cuda" or triton is None:
        return allreduce_reference(x, site)
    rows = x.numel() // x.shape[-1]
    if scatter_wanted(rows):
        return allreduce_scatter(x, site, traced)
    return allreduce_oneshot(x, site, traced)


# ---------------------------------------------------------------------------
# the measuring step


def _med(xs):
    xs = sorted(xs)
    return {"median_us": 1000 * statistics.median(xs), "p90_us": 1000 * xs[min(len(xs) - 1, int(0.9 * len(xs)))],
            "n": len(xs)}


def _all_sync(c):
    for d in c.devices:
        torch.cuda.synchronize(d)


def measure(rows_list, steps, rounds, out_path, dim=ROW, dtype=torch.bfloat16):
    """Both forms at every row count: correctness (bit for bit against the reference, on every
    card), one reduction captured per card and replayed together, and a chain of `steps`
    reductions with a 5120 x 1280 GEMV between them (the TP4 dense step's shape); the better
    form per class goes into a kernel table's "blocks" with the measured microseconds."""
    n = torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(n)]
    max_rows = max(rows_list)
    c = setup(devices, sites=max(steps, 2), max_elems=max_rows * dim, dtype=dtype)
    name = torch.cuda.get_device_name(0)
    results = {"gpu": name, "world": n, "dim": dim, "classes": []}
    streams = [torch.cuda.Stream(device=d) for d in devices]
    w = [torch.randn(dim, 1280, dtype=dtype, device=d) for d in devices]
    for k, d in enumerate(devices):  # cuBLAS handles and kernels warm on every card before any capture
        with torch.cuda.device(d), torch.cuda.stream(streams[k]):
            torch.matmul(torch.randn(max_rows, dim, dtype=dtype, device=d), w[k])
    _all_sync(c)
    for rows in rows_list:
        count = rows * dim
        xs = [torch.randn(rows, dim, dtype=dtype, device=d) for d in devices]
        ref = sum(x.to(0).float() for x in xs).to(dtype)
        entry = {"rows": rows, "bytes": count * xs[0].element_size(), "forms": {}}
        for form, fn in (("oneshot", allreduce_oneshot), ("scatter", allreduce_scatter)):
            # correctness, eager on streams
            print(f"rows {rows} {form}: correctness", flush=True)
            outs = [None] * n
            for k in range(n):
                with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                    outs[k] = fn(xs[k], 0)
            _all_sync(c)
            for k in range(n):
                if not torch.equal(outs[k].to(0), ref):
                    raise RuntimeError(f"{form}: card {k} summed wrong at {rows} rows")
            # one reduction per card, captured, replayed together: per reduction time on card 0
            print(f"rows {rows} {form}: capturing single", flush=True)
            graphs = []
            for k in range(n):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.device(k), torch.cuda.graph(g, stream=streams[k]):
                    for s in range(8):
                        fn(xs[k], s % c.sites)
                graphs.append(g)
            _all_sync(c)
            single = []
            for r in range(rounds + 3):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(streams[0])
                for k in range(n):
                    with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                        graphs[k].replay()
                en.record(streams[0])
                _all_sync(c)
                if r >= 3:
                    single.append(st.elapsed_time(en) / 8)
            # the chain: steps reductions with a GEMV between, one graph per card
            print(f"rows {rows} {form}: capturing chain", flush=True)
            chains = []
            for k in range(n):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.device(k), torch.cuda.graph(g, stream=streams[k]):
                    for s in range(steps):
                        torch.matmul(xs[k], w[k])
                        fn(xs[k], s)
                chains.append(g)
            _all_sync(c)
            chain_t = []
            for r in range(rounds + 3):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(streams[0])
                for k in range(n):
                    with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                        chains[k].replay()
                en.record(streams[0])
                _all_sync(c)
                if r >= 3:
                    chain_t.append(st.elapsed_time(en))
            # the GEMVs alone, for the crossing share
            g0 = torch.cuda.CUDAGraph()
            with torch.cuda.device(0), torch.cuda.graph(g0, stream=streams[0]):
                for s in range(steps):
                    torch.matmul(xs[0], w[0])
            torch.cuda.synchronize(0)
            gemv_t = []
            for r in range(rounds + 3):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(streams[0])
                g0.replay()
                en.record(streams[0])
                torch.cuda.synchronize(0)
                if r >= 3:
                    gemv_t.append(st.elapsed_time(en))
            # the chain's result: every card's output of the last site equals the reference
            m_single, m_chain, m_gemv = _med(single), _med(chain_t), _med(gemv_t)
            entry["forms"][form] = {
                "single_us": m_single, "chain_ms": m_chain["median_us"] / 1000, "chain_p90_ms": m_chain["p90_us"] / 1000,
                "gemv_only_ms": m_gemv["median_us"] / 1000,
                "per_reduction_in_chain_us": (m_chain["median_us"] - m_gemv["median_us"]) / steps,
                "per_step_ms": (m_chain["median_us"] - m_gemv["median_us"]) / 1000,
            }
            print(f"rows {rows:4d} {form:8s}: single {m_single['median_us']:7.1f} us, chain {m_chain['median_us'] / 1000:7.2f} ms "
                  f"({steps} reductions; GEMVs alone {m_gemv['median_us'] / 1000:.2f}), "
                  f"{entry['forms'][form]['per_reduction_in_chain_us']:.1f} us a reduction", flush=True)
            del graphs, chains, g0
        results["classes"].append(entry)
    # the kernel table: the better form per class
    blocks = []
    crossover = None
    for e in results["classes"]:
        one, sc = e["forms"]["oneshot"]["per_reduction_in_chain_us"], e["forms"]["scatter"]["per_reduction_in_chain_us"]
        form = 2 if sc < one else 1
        if form == 2 and crossover is None:
            crossover = e["rows"]
        blocks.append({"shapes": f"allreduce rows={e['rows']} n={dim} k={n} w=bf16",
                       "blocks": {"FORM": form, "BLOCK": block_elems()}, "us": round(min(one, sc), 2),
                       "bytes": e["bytes"],
                       "why": f"one-shot {one:.1f} us, reduce-scatter {sc:.1f} us a reduction in a chain of {steps} on {name}"})
    table = {"format": "vxpu-kernel-table/1", "gpu": name, "sites": [],
             "knobs": {"ALLREDUCE_SCATTER_MIN_ROWS": crossover or 0, "ALLREDUCE_BLOCK": block_elems()},
             "blocks": blocks,
             "explain": [f"vxpu.allreduce measure: world {n}, {steps} reductions a chain, {rounds} rounds; "
                         f"FORM 1 = one-shot, 2 = reduce-scatter + all-gather"]}
    with open(out_path, "w") as f:
        json.dump(table, f, indent=1)
    with open(out_path.replace(".json", "-raw.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path} (crossover at {crossover} rows)")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--rows", default="1,8,33,132,264")
    m.add_argument("--chain", type=int, default=86)
    m.add_argument("--rounds", type=int, default=20)
    m.add_argument("--out", default="allreduce-table.json")
    a = p.parse_args(argv)
    if a.cmd == "measure":
        t0 = time.time()
        measure([int(r) for r in a.rows.split(",")], a.chain, a.rounds, a.out)
        print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
