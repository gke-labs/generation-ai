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
    python -m vxpu.allreduce measure-push    the fused push's GEMV per tile width at the step's
                                             two mixing shapes (hero run 15)
"""

import argparse
import json
import statistics
import time

import contextlib
import os
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
_EARLIER = []  # Comms a larger one replaced: graphs captured against them keep their buffers' addresses


class Comm:
    """The buffers of a world of cards: for site s and card j, an inbox per other card k
    (inbox[j][s][k], the one-shot's whole partial or the scatter's slice), a gather buffer per
    other card (gbuf[j][s][k]: the reduced slice k sends back), flags [2, world] (round A and
    round B, one per source rank) and counters [3] (push, reduce, gather arrivals), all on card
    j. Ranks are the position in `devices`; the sum is taken in rank order."""

    def __init__(self, devices, sites, max_elems, dtype=torch.bfloat16, dtypes=None, hcs=()):
        self.devices = [torch.device(d) for d in devices]
        self.world = len(self.devices)
        self.hcs = tuple(hcs)  # the hc widths the mixing sums (allreduce_hc) use: their kernels primed too
        self.bound = False  # set once a captured graph holds this Comm's buffers: from then on it is never replaced
        self.sites = sites
        self.max_elems = max_elems
        # the dtypes the sites sum (a tensor-parallel step reduces fp32 streams and bf16 scores):
        # the buffers are bytes, sized for max_elems of the widest, viewed as a call's dtype (box);
        # every kernel is primed per dtype, since Triton specializes it on the pointers' type
        self.dtypes = tuple(dtypes) if dtypes else (dtype,)
        self.dtype = self.dtypes[0]
        width = max(d.itemsize for d in self.dtypes)
        self.max_bytes = -(-max_elems * width // 16) * 16
        self.slice = -(-max_elems // max(self.world, 1))
        slice_bytes = -(-self.slice * width // 16) * 16
        self.inbox, self.gbuf, self.flags, self.counters, self.slots, self.trace = [], [], [], [], [], []
        for j, dev in enumerate(self.devices):
            self.inbox.append([[torch.empty(self.max_bytes, dtype=torch.uint8, device=dev) if k != j else None
                                for k in range(self.world)] for _ in range(sites)])
            # the pull form's export slot a site: this card's partial, published here, read by the peers over PCIe
            self.slots.append([torch.empty(self.max_bytes, dtype=torch.uint8, device=dev) for _ in range(sites)])
            self.gbuf.append([[torch.empty(slice_bytes, dtype=torch.uint8, device=dev) if k != j else None
                               for k in range(self.world)] for _ in range(sites)])
            self.flags.append(torch.zeros(sites, 2, self.world, dtype=torch.int32, device=dev))
            self.counters.append(torch.zeros(sites, 3, dtype=torch.int32, device=dev))
            self.trace.append(torch.zeros(sites, 2, dtype=torch.int64, device=dev))  # SUM_TRACE: arrival, flags up (ns)
        if self.world > 1 and self.devices[0].type == "cuda":
            _enable_peers(self.devices)
            self.prime()
        # the reference's view: the partial each rank registered at a site (tests)
        self.registered = {}


    def reset(self):
        """Every card's flags and counters back to zero: the clean slate a first real step needs after
        steps that pushed with nobody consuming (the prime's solo-mode eager steps, hero run 15)."""
        for dev, f, c in zip(self.devices, self.flags, self.counters):
            with torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext():
                f.zero_()
                c.zero_()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

    def prime(self):
        """Launch every kernel once on every card against local scratch with the flags already
        raised, so each card loads the kernels' modules now. Triton loads a kernel's module on a
        card at its first launch there, and that load blocks while a kernel is spinning on that
        card: a card whose sum kernel is already waiting for its peers would block the host from
        launching the peers' pushes, and the step would hang (measured: the host sat in
        _init_handles while card 0 spun). Captured graphs never hit it; eager calls would."""
        if triton is None:
            return
        peers = self.world - 1
        block = block_elems()
        for rank, dev in enumerate(self.devices):
            for dtype in self.dtypes:
                self._prime_one(dev, dtype, peers, block, rank)
                for hc in self.hcs:  # the mixing kernels (the sum with hc_post in its epilogue) too
                    _PRIMED_HC.discard((dev.index, dtype, hc))
                    _prime_hc_now(dev, dtype, hc, peers, rank)
            torch.cuda.synchronize(dev)

    def _prime_one(self, dev, dtype, peers, block, rank=0):
        # the fp32 partials' kernels twice when they cross as bf16 (push_bf16): the inboxes bf16, this
        # rank's own partial fp32 in its slot, the partials rounded (ROUND); the specialization the step runs
        layouts = [(dtype, False)]
        if dtype == torch.float32 and push_bf16():
            layouts.append((torch.bfloat16, True))
        for box_dtype, rnd in layouts:
            with torch.cuda.device(dev):
                x = torch.zeros(block, dtype=dtype, device=dev)
                out = torch.empty_like(x)
                sc = [torch.zeros(block, dtype=box_dtype, device=dev) for _ in range(3)]
                parts = [sc[0] if k != rank else x for k in range(4)]
                cnt = torch.zeros(3, dtype=torch.int32, device=dev)
                ones = [torch.ones(4, dtype=torch.int32, device=dev) for _ in range(3)]
                n = x.numel()
                raise_ = [ones[0][i:i + 1] for i in range(3)]
                _ar_push_kernel[(1,)](x, sc[0], sc[1], sc[2], n, cnt[0:1], raise_[0], raise_[1], raise_[2],
                                      PEERS=peers, BLOCK=block)
                _ar_publish_kernel[(1,)](x, sc[0], n, cnt[0:1], raise_[0], raise_[1], raise_[2], PEERS=peers, BLOCK=block)
                _ar_sum_kernel[(1,)](parts[0], parts[1], parts[2], parts[3], out, n, ones[1], cnt[1:2], PEERS=peers,
                                     BLOCK=block, ROUND=rnd)
                ones[1].fill_(1)
                _rs_push_kernel[(1,)](x, sc[0], sc[1], sc[2], 0, 0, 0, n, n, cnt[0:1], raise_[0], raise_[1], raise_[2],
                                      PEERS=peers, BLOCK=block)
                gx = [torch.zeros_like(x) for _ in range(3)]
                _rs_reduce_kernel[(1,)](parts[0], parts[1], parts[2], parts[3], out, gx[0], gx[1], gx[2], n, n, ones[1],
                                        cnt[1:2], raise_[0], raise_[1], raise_[2], PEERS=peers, BLOCK=block, ROUND=rnd, trace_ptr=torch.zeros(2, dtype=torch.int64, device=dev))
                sc = gx
                _rs_gather_kernel[(1,)](out, sc[0], sc[1], sc[2], 0, 0, 0, n, n, ones[2], cnt[2:3],
                                        PEERS=peers, BLOCK=block)

    def rank_of(self, device):
        device = torch.device(device)
        for r, d in enumerate(self.devices):
            if d.type == device.type and (d.index or 0) == (device.index or 0):
                return r
        raise ValueError(f"allreduce: {device} is not one of the world's cards {self.devices}")

    def peers(self, rank):
        return [k for k in range(self.world) if k != rank]

    def box(self, j, site, k, dtype):
        """Card j's inbox for source k at `site`, as `dtype` (the bytes viewed)."""
        return self.inbox[j][site][k].view(dtype)

    def slot(self, j, site, dtype):
        """Card j's export slot at `site` (the pull form: j's partial, read by every peer), as `dtype`."""
        return self.slots[j][site].view(dtype)

    def gather_box(self, j, site, k, dtype):
        """Card j's gather buffer for the slice card k reduced at `site`, as `dtype`."""
        return self.gbuf[j][site][k].view(dtype)

    def register(self, site, rank, x):
        self.registered.setdefault(site, {})[rank] = x


def setup(devices, sites, max_elems, dtype=torch.bfloat16, dtypes=None, hcs=()):
    """The process's Comm: the cards (ranks in this order), how many call sites a step has, the
    most elements one call sums, their dtype (`dtypes`: several, each primed), and `hcs`, the hc
    widths of the sums that mix the stream (their kernels primed here too, not lazily under a
    synchronize). A Comm that already fits (the same cards, as many sites and elements, the dtypes
    and hc widths) is kept: the step graphs of several programs share it, one program stepping at
    a time (the verifies per draft length, hero run 15); a larger one replaces it and the earlier
    stays alive, since the graphs captured against it hold its buffers' addresses (replacing it
    freed them under a captured graph: a segmentation fault at the second program's first step)."""
    global _COMM
    want = tuple(dtypes) if dtypes else (dtype,)
    hcs = tuple(hcs)
    c = _COMM
    if c is not None and c.devices == [torch.device(d) for d in devices] and c.sites >= sites \
            and c.max_elems >= max_elems and set(want) <= set(c.dtypes) and set(hcs) <= set(c.hcs):
        return c
    if c is not None:
        if c.bound:  # a graph captured against it: replacing it now would leave that graph pushing into boxes
            # nobody waits on (a rank-pair wait in the first step, hero run 15); size it once before any capture
            raise RuntimeError(f"allreduce: the Comm ({c.sites} sites, {c.max_elems} elements, {c.dtypes}, hc {c.hcs}) is "
                               f"bound by a captured graph and {sites} sites x {max_elems} elements ({want}, hc {hcs}) "
                               f"do not fit it; the engine sizes it once for every program before the first capture")
        _EARLIER.append(c)
        sites, max_elems = max(sites, c.sites), max(max_elems, c.max_elems)
        want = tuple(sorted(set(want) | set(c.dtypes), key=str))
        hcs = tuple(sorted(set(hcs) | set(c.hcs)))
    _COMM = Comm(devices, sites, max_elems, want[0], want, hcs)
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
    if x.dtype not in c.dtypes:
        raise ValueError(f"allreduce: x is {x.dtype}, the Comm was set up for {c.dtypes}")
    if x.numel() * x.element_size() > c.max_bytes:
        raise ValueError(f"allreduce: {x.numel()} elements of {x.dtype}, the Comm holds {c.max_bytes} bytes a site")


# ---------------------------------------------------------------------------
# the kernels

# the kernels read these at compile time (Triton takes module globals only as tl.constexpr instances)
WAIT_VECTOR = tl.constexpr(os.environ.get("VXPU_ALLREDUCE_WAIT_VECTOR", "1") == "1") if tl is not None else False  # the spin's form (an A/B)
WAIT_NONE = tl.constexpr(os.environ.get("VXPU_ALLREDUCE_WAIT_NONE", "0") == "1") if tl is not None else False  # no spin: the timing split only
# the sites' arrival trace (measurement, hero run 15): each hc sum's first program stamps the GPU's global
# timer when it arrives and when its peers' flags are up, into Comm.trace[rank][site] (ns); a run's last step
# is left in the buffers and a harness reads the ranks' arrivals against rank 0's without the profiler
SUM_TRACE = tl.constexpr(os.environ.get("VXPU_SUM_TRACE", "0") == "1") if tl is not None else False
FENCE_ONCE = tl.constexpr(os.environ.get("VXPU_ALLREDUCE_FENCE_ONCE", "1") == "1") if tl is not None else True  # one fence then relaxed flags (VXPU_ALLREDUCE_FENCE_ONCE=0 the A/B)
PUSH_BLOCK = 512  # elements a push program moves: 8x the programs of the 4,096 block, -2 to -3 us a site at 8 to 16 rows
PUSH_PEER_MAJOR = os.environ.get("VXPU_ALLREDUCE_PUSH_PM", "0") == "1"  # the push's grid: a program per (block, peer) (an A/B)


if triton is not None:
    @triton.jit
    def _wait_flags(f_ptr, PEERS: tl.constexpr):
        """Spin until every peer's flag (int32, contiguous, in rank order) is raised: an acquire at
        system scope, so the peer stores it ordered before are visible to the loads after. With
        WAIT_VECTOR (the default; VXPU_ALLREDUCE_WAIT_VECTOR=0 the A/B back) the spin reads the PEERS
        flags in one volatile load, and only once all are up does an acquire per flag pair with the
        peer's release, instead of PEERS system-scope atomics spinning in turn: -0.6 us a site on
        every form (hero run 15, measured in the fused push's harness; the B12X all-reduce spins
        the same way, plain loads then one fence)."""
        if WAIT_NONE:  # timing only: the data is static in the measuring harness's chains
            tl.atomic_add(f_ptr, 0, sem="acquire", scope="sys")
        elif WAIT_VECTOR:
            idx = tl.arange(0, 4)
            while tl.min(tl.load(f_ptr + idx, mask=idx < PEERS, other=1, volatile=True), axis=0) < 1:
                pass
            for i in tl.static_range(PEERS):
                tl.atomic_add(f_ptr + i, 0, sem="acquire", scope="sys")
        else:
            for i in tl.static_range(PEERS):
                while tl.atomic_add(f_ptr + i, 0, sem="acquire", scope="sys") < 1:
                    pass

    @triton.jit
    def _raise_flags(f0, f1, f2, PEERS: tl.constexpr):
        if FENCE_ONCE:  # one system-scope fence orders the stores before every flag; the flags then go relaxed
            tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
            tl.atomic_xchg(f0, 1, sem="relaxed", scope="sys")
            if PEERS > 1:
                tl.atomic_xchg(f1, 1, sem="relaxed", scope="sys")
            if PEERS > 2:
                tl.atomic_xchg(f2, 1, sem="relaxed", scope="sys")
        else:
            tl.atomic_xchg(f0, 1, sem="release", scope="sys")
            if PEERS > 1:
                tl.atomic_xchg(f1, 1, sem="release", scope="sys")
            if PEERS > 2:
                tl.atomic_xchg(f2, 1, sem="release", scope="sys")

    @triton.jit
    def _ar_push_pm_kernel(x_ptr, d0, d1, d2, n, counter_ptr, f0, f1, f2, PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """The push with a program per (block, peer): three streams of remote stores in flight instead
        of one program storing its block to the peers in turn (the per-byte layout question, hero run
        15: the push ran at 3.4 GB/s a site). The last program of the grid raises the flags."""
        pid = tl.program_id(0)
        j = tl.program_id(1)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=0)
        if j == 0:
            tl.store(d0 + offs, v.to(d0.dtype.element_ty), mask=m)
        elif j == 1:
            tl.store(d1 + offs, v.to(d1.dtype.element_ty), mask=m)
        else:
            tl.store(d2 + offs, v.to(d2.dtype.element_ty), mask=m)
        tl.debug_barrier()
        done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
        if done == tl.num_programs(0) * tl.num_programs(1) - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

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
        tl.store(d0 + offs, v.to(d0.dtype.element_ty), mask=m)  # the inbox's dtype: bf16 for an fp32 partial
        if PEERS > 1:  # under ALLREDUCE_PUSH_BF16 (hero run 15), half the bytes over PCIe
            tl.store(d1 + offs, v.to(d1.dtype.element_ty), mask=m)
        if PEERS > 2:
            tl.store(d2 + offs, v.to(d2.dtype.element_ty), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

    @triton.jit
    def _ar_publish_kernel(x_ptr, s_ptr, n, counter_ptr, f0, f1, f2, PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """The pull form, round A: this card's partial into its own export slot (a local write, in the
        slot's dtype), then the last program raises the site's flag on every peer; the peers read the
        slot over PCIe in their sum kernel (hero run 15: no bytes pushed, one PCIe pass a reader)."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=0)
        tl.store(s_ptr + offs, v.to(s_ptr.dtype.element_ty), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

    @triton.jit
    def _ar_pull_hc_kernel(p0, p1, p2, p3, out_ptr, res_ptr, post_ptr, comb_ptr, n, d, f_ptr, counter_ptr,
                           PEERS: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, UNROLL: tl.constexpr,
                           ROUND: tl.constexpr, ROWBLOCK: tl.constexpr = False):
        """The pull form's sum with the block's hc_post in its epilogue (hero run 15), in the geometry
        the B12X all-reduce measured best on PCIe: a few programs (3 at one row, 8 from five rows up)
        of 256 threads, each walking its share of the partial in BLOCK-element tiles, UNROLL tiles'
        loads of every rank's slot issued before any is summed, so each program keeps many PCIe reads
        in flight; one flag barrier at the start, the ranks' partials added in rank order in fp32,
        the total rounded to the stream's dtype and mixed into the HC copies of the stream."""
        _wait_flags(f_ptr, PEERS)
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        nblocks = tl.cdiv(n, BLOCK)
        for first in range(pid * UNROLL, nblocks, nprog * UNROLL):
            for u in tl.static_range(UNROLL):
                blk = first + u
                offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                m = offs < n
                acc = _partial(p0, offs, m, ROUND)
                acc += _partial(p1, offs, m, ROUND)
                if PEERS > 1:
                    acc += _partial(p2, offs, m, ROUND)
                if PEERS > 2:
                    acc += _partial(p3, offs, m, ROUND)
                total = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
                _hc_mix(total, offs, m, out_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, blk.to(tl.int64) * BLOCK)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(f_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _partial(p, offs, m, ROUND: tl.constexpr):
        """A rank's partial for the sum, as fp32; with ROUND every partial, this card's own fp32 one
        included, is rounded to bf16 first, as the peers' arrived (the inboxes' dtype), so the four
        ranks add the same four values in the same order and their totals agree bit for bit."""
        v = tl.load(p + offs, mask=m, other=0, volatile=True).to(tl.float32)
        if ROUND:
            v = v.to(tl.bfloat16).to(tl.float32)
        return v

    @triton.jit
    def _ar_sum_kernel(p0, p1, p2, p3, out_ptr, n, f_ptr, counter_ptr, PEERS: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr):
        """One-shot, round B: wait for the peers' flags, sum the partials in rank order (p0..p3
        are the ranks' partials, this card's own among them, fp32 accumulation, rounded once),
        then the last program resets the flags for the next step."""
        _wait_flags(f_ptr, PEERS)
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = _partial(p0, offs, m, ROUND)
        acc += _partial(p1, offs, m, ROUND)
        if PEERS > 1:
            acc += _partial(p2, offs, m, ROUND)
        if PEERS > 2:
            acc += _partial(p3, offs, m, ROUND)
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(f_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _ar_sum2_hc_kernel(p0, p1, p2, p3, q0, q1, q2, q3, out_ptr, res_ptr, post_ptr, comb_ptr, n, d, f_ptr, g_ptr,
                           counter_ptr, PEERS: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, ROWBLOCK: tl.constexpr = False):
        """_ar_sum_hc_kernel for the two-part push sum (hero run 15): each rank's partial arrives as
        two parts, the routed experts' sum (p, fp32, pushed by the expert kernel from its epilogue,
        the flags at f_ptr) and the shared expert's output (q, bf16, pushed by its GEMV from the
        parallel branch, the flags at g_ptr); the sum adds p_r + q_r in fp32, the value the fused
        push's GEMV epilogue made, then the ranks in rank order: the same numbers in the same order."""
        pid = tl.program_id(0)
        _wait_flags(f_ptr, PEERS)
        _wait_flags(g_ptr, PEERS)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = _partial(p0, offs, m, False) + _partial(q0, offs, m, False)
        acc += _partial(p1, offs, m, False) + _partial(q1, offs, m, False)
        if PEERS > 1:
            acc += _partial(p2, offs, m, False) + _partial(q2, offs, m, False)
        if PEERS > 2:
            acc += _partial(p3, offs, m, False) + _partial(q3, offs, m, False)
        total = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
        _hc_mix(total, offs, m, out_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, pid.to(tl.int64) * BLOCK)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(f_ptr + i, 0, sem="relaxed", scope="gpu")
                tl.atomic_xchg(g_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _ar_sum_hc_kernel(p0, p1, p2, p3, out_ptr, res_ptr, post_ptr, comb_ptr, n, d, f_ptr, counter_ptr, trace_ptr,
                          PEERS: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr, ROWBLOCK: tl.constexpr = False):
        """_ar_sum_kernel with the block's hc_post as its epilogue (hero run 15): the total of row t,
        rounded to the stream's dtype as the fork's all_reduce output is, becomes HC copies of the
        stream, out[t, h, :] = post[t, h] * total + sum_g comb[t, h, g] * residual[t, g, :], so the
        two casts, the copy and the mix's kernel after the sum are not launched. trace_ptr: two
        int64 the first program stamps with the global timer at arrival and after the wait (SUM_TRACE)."""
        pid = tl.program_id(0)
        if SUM_TRACE:
            if pid == 0:
                tl.store(trace_ptr, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                              is_pure=False, pack=1))
        _wait_flags(f_ptr, PEERS)
        if SUM_TRACE:
            if pid == 0:
                tl.store(trace_ptr + 1, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                                  is_pure=False, pack=1))
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = _partial(p0, offs, m, ROUND)
        acc += _partial(p1, offs, m, ROUND)
        if PEERS > 1:
            acc += _partial(p2, offs, m, ROUND)
        if PEERS > 2:
            acc += _partial(p3, offs, m, ROUND)
        total = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
        _hc_mix(total, offs, m, out_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, pid.to(tl.int64) * BLOCK)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(f_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _hc_mix(total, g, m, y_ptr, res_ptr, post_ptr, comb_ptr, d, HC: tl.constexpr, ROWBLOCK: tl.constexpr = False,
                g0=0):
        """The block's hc_post for the totals `total` (fp32, already rounded to the stream's dtype)
        at flat element indexes g of the summed [.., d]: y[t, h, :] = post[t, h] * total + sum_g
        comb[t, g, h] * residual[t, g, :], stored into the stream y [.., HC, d]. With ROWBLOCK the
        block lies within one row (the caller checked BLOCK divides d and the block's start), so the
        row's post and comb are loaded once as scalars instead of per element: 20 loads an element
        fewer, the same arithmetic in the same order (hero run 15: the mix was most of the sum
        kernel's 8 us a site at one row)."""
        if ROWBLOCK:  # g0: the block's first flat index, a scalar the caller knows
            t0 = g0 // d
            col = g - t0 * d
            for h in tl.static_range(HC):
                y = tl.load(post_ptr + t0 * HC + h).to(tl.float32) * total
                for k in tl.static_range(HC):
                    c = tl.load(comb_ptr + t0 * (HC * HC) + k * HC + h).to(tl.float32)
                    y += c * tl.load(res_ptr + (t0 * HC + k) * d + col, mask=m, other=0).to(tl.float32)
                tl.store(y_ptr + (t0 * HC + h) * d + col, y.to(y_ptr.dtype.element_ty), mask=m)
        else:
            t = g // d
            col = g - t * d
            for h in tl.static_range(HC):
                y = tl.load(post_ptr + t * HC + h, mask=m, other=0).to(tl.float32) * total
                for k in tl.static_range(HC):
                    c = tl.load(comb_ptr + t * (HC * HC) + k * HC + h, mask=m, other=0).to(tl.float32)
                    y += c * tl.load(res_ptr + (t * HC + k) * d + col, mask=m, other=0).to(tl.float32)
                tl.store(y_ptr + (t * HC + h) * d + col, y.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _rs_reduce_hc_kernel(p0, p1, p2, p3, y_ptr, res_ptr, post_ptr, comb_ptr, g0, g1, g2, S, n_mine, base, d,
                             fa_ptr, counter_ptr, fb0, fb1, fb2, PEERS: tl.constexpr, HC: tl.constexpr,
                             BLOCK: tl.constexpr, ROUND: tl.constexpr, ROWBLOCK: tl.constexpr = False, trace_ptr=None):
        """_rs_reduce_kernel with the block's hc_post as its epilogue: this card's slice of the total
        (elements base .. base + n_mine of the summed tensor) goes to the peers' gather buffers as
        the stream's dtype and, mixed, into the stream y."""
        if SUM_TRACE:  # the skew harness (hero-skew.py): arrival and the end of the wait, as the one-shot sum stamps
            if tl.program_id(0) == 0:
                tl.store(trace_ptr, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                              is_pure=False, pack=1))
        _wait_flags(fa_ptr, PEERS)
        if SUM_TRACE:
            if tl.program_id(0) == 0:
                tl.store(trace_ptr + 1, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                                  is_pure=False, pack=1))
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_mine
        acc = _partial(p0, offs, m, ROUND)
        acc += _partial(p1, offs, m, ROUND)
        if PEERS > 1:
            acc += _partial(p2, offs, m, ROUND)
        if PEERS > 2:
            acc += _partial(p3, offs, m, ROUND)
        v = acc.to(g0.dtype.element_ty)
        tl.store(g0 + offs, v, mask=m)
        if PEERS > 1:
            tl.store(g1 + offs, v, mask=m)
        if PEERS > 2:
            tl.store(g2 + offs, v, mask=m)
        _hc_mix(v.to(y_ptr.dtype.element_ty).to(tl.float32), base + offs, m, y_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK,
                base + pid.to(tl.int64) * BLOCK)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(fa_ptr + i, 0, sem="relaxed", scope="gpu")
            _raise_flags(fb0, fb1, fb2, PEERS)

    @triton.jit
    def _rs_gather_hc_kernel(y_ptr, res_ptr, post_ptr, comb_ptr, h0, h1, h2, o0, o1, o2, S, n, d, fb_ptr, counter_ptr,
                             PEERS: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, ROWBLOCK: tl.constexpr = False):
        """_rs_gather_kernel with the block's hc_post as its epilogue: each peer's reduced slice,
        mixed, into the stream y at its offset."""
        _wait_flags(fb_ptr, PEERS)
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < S
        v0 = tl.load(h0 + offs, mask=m, other=0, volatile=True).to(y_ptr.dtype.element_ty).to(tl.float32)
        _hc_mix(v0, o0 + offs, m & (o0 + offs < n), y_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, o0 + pid.to(tl.int64) * BLOCK)
        if PEERS > 1:
            v1 = tl.load(h1 + offs, mask=m, other=0, volatile=True).to(y_ptr.dtype.element_ty).to(tl.float32)
            _hc_mix(v1, o1 + offs, m & (o1 + offs < n), y_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, o1 + pid.to(tl.int64) * BLOCK)
        if PEERS > 2:
            v2 = tl.load(h2 + offs, mask=m, other=0, volatile=True).to(y_ptr.dtype.element_ty).to(tl.float32)
            _hc_mix(v2, o2 + offs, m & (o2 + offs < n), y_ptr, res_ptr, post_ptr, comb_ptr, d, HC, ROWBLOCK, o2 + pid.to(tl.int64) * BLOCK)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            for i in tl.static_range(PEERS):
                tl.atomic_xchg(fb_ptr + i, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _rs_push_kernel(x_ptr, d0, d1, d2, o0, o1, o2, S, n, counter_ptr, f0, f1, f2,
                        PEERS: tl.constexpr, BLOCK: tl.constexpr):
        """Reduce-scatter, round A: slice j of this card's partial (elements o_j .. o_j + S) into
        card j's inbox for this site, for every peer j; the last program raises the flags."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < S
        tl.store(d0 + offs, tl.load(x_ptr + o0 + offs, mask=m & (o0 + offs < n), other=0).to(d0.dtype.element_ty), mask=m)
        if PEERS > 1:
            tl.store(d1 + offs, tl.load(x_ptr + o1 + offs, mask=m & (o1 + offs < n), other=0).to(d1.dtype.element_ty), mask=m)
        if PEERS > 2:
            tl.store(d2 + offs, tl.load(x_ptr + o2 + offs, mask=m & (o2 + offs < n), other=0).to(d2.dtype.element_ty), mask=m)
        if _last_program(counter_ptr):
            tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
            _raise_flags(f0, f1, f2, PEERS)

    @triton.jit
    def _rs_reduce_kernel(p0, p1, p2, p3, out_ptr, g0, g1, g2, S, n_mine, fa_ptr, counter_ptr, fb0, fb1, fb2,
                          PEERS: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr, trace_ptr=None):
        """Reduce-scatter, round B: wait for the slices, sum this card's slice in rank order
        (p0..p3 point at the ranks' slices, this card's own at its offset in x), store it into
        the output and push it into every peer's gather buffer; the last program resets round
        A's flags and raises round B's."""
        if SUM_TRACE:  # the skew harness (hero-skew.py): arrival and the end of the wait, as the one-shot sum stamps
            if tl.program_id(0) == 0:
                tl.store(trace_ptr, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                              is_pure=False, pack=1))
        _wait_flags(fa_ptr, PEERS)
        if SUM_TRACE:
            if tl.program_id(0) == 0:
                tl.store(trace_ptr + 1, tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                                                  is_pure=False, pack=1))
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_mine
        acc = _partial(p0, offs, m, ROUND)
        acc += _partial(p1, offs, m, ROUND)
        if PEERS > 1:
            acc += _partial(p2, offs, m, ROUND)
        if PEERS > 2:
            acc += _partial(p3, offs, m, ROUND)
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
    """Elements a program moves (the knob ALLREDUCE_BLOCK, default 4096; VXPU_ALLREDUCE_BLOCK the A/B)."""
    return int(os.environ.get("VXPU_ALLREDUCE_BLOCK", "0")) or int(getattr(triton_ops, "ALLREDUCE_BLOCK", None) or 4096)


def scatter_wanted(rows):
    """Whether the reduce-scatter form runs for a call of `rows` rows: forced by the knob
    ALLREDUCE_SCATTER, else from ALLREDUCE_SCATTER_MIN_ROWS (0: never; the measuring step
    writes the crossover it finds into the table)."""
    forced = os.environ.get("VXPU_ALLREDUCE_SCATTER") or getattr(triton_ops, "ALLREDUCE_SCATTER", None)
    if forced is not None:  # the table's knob, or the A/B switch VXPU_ALLREDUCE_SCATTER=1 (hero run 15)
        return forced in (True, "1")
    min_rows = int(getattr(triton_ops, "ALLREDUCE_SCATTER_MIN_ROWS", None) or 0)
    return min_rows > 0 and rows >= min_rows


def push_bf16():
    """Whether fp32 partials cross PCIe as bf16 (the knob ALLREDUCE_PUSH_BF16, VXPU_ALLREDUCE_PUSH_BF16
    the A/B override): a numerics class, each rank's partial rounded to bf16 before the fp32 sum
    instead of the total once after it (hero run 15's many-row account: the FFN sites' fp32 partials
    were half the pushes' bytes). bf16 partials are unaffected."""
    v = os.environ.get("VXPU_ALLREDUCE_PUSH_BF16") or getattr(triton_ops, "ALLREDUCE_PUSH_BF16", None)
    return v in (True, "1")


def push_dtype(x):
    """The dtype a partial x crosses PCIe in."""
    return torch.bfloat16 if x.dtype == torch.float32 and push_bf16() else x.dtype


def _launch_push(flat, dests, n, counter, raise_, peers, block, traced):
    """The one-shot push in its layout: a program per block storing to the peers in turn (the form of
    record), or with PUSH_PEER_MAJOR a program per (block, peer)."""
    if PUSH_PEER_MAJOR:
        kernel, ctx = triton_ops._launcher(_ar_push_pm_kernel, flat.device, traced)
        with ctx:
            kernel[(triton.cdiv(n, block), peers)](flat, dests[0], dests[1], dests[2], n, counter, raise_[0], raise_[1],
                                                   raise_[2], PEERS=peers, BLOCK=block)
    else:
        kernel, ctx = triton_ops._launcher(_ar_push_kernel, flat.device, traced)
        with ctx:
            kernel[(triton.cdiv(n, block),)](flat, dests[0], dests[1], dests[2], n, counter, raise_[0], raise_[1], raise_[2],
                                             PEERS=peers, BLOCK=block)


def push_only_hc(x, site, traced=False):
    """The harness's push-only form: the one-shot push kernel of x at `site`, no sum (the flags stay
    raised; the next push over them is the same store), to time the push alone."""
    c = comm()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    flat = x.contiguous().view(-1)
    n = flat.numel()
    pd = push_dtype(x)
    dests = _pad3([c.box(j, site, rank, pd) for j in peers], flat)
    raise_ = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], c.counters[rank][site, 0:1])
    _launch_push(flat, dests, n, c.counters[rank][site, 0:1], raise_, len(peers), min(block_elems(), PUSH_BLOCK), traced)
    return x


PULL_SCOPE = None  # "draft" while the drafter's program is lowered (engine): its sites take the pull form under DRAFT_ALLREDUCE_PULL


def pull_wanted():
    """Whether the mixing sums take the pull form (the knob ALLREDUCE_PULL, VXPU_ALLREDUCE_PULL the A/B
    override): each card publishes its partial in its own slot and raises the peers' flags, and the
    sum kernel reads the peers' slots over PCIe instead of inboxes the peers pushed (hero run 15).
    The drafter's sites alone take it under DRAFT_ALLREDUCE_PULL while PULL_SCOPE is "draft" (the
    engine sets it around the drafter's lowering): the width drafter is exchange-bound where the
    verify's sums measured better pushed."""
    v = os.environ.get("VXPU_ALLREDUCE_PULL") or getattr(triton_ops, "ALLREDUCE_PULL", None)
    if v in (True, "1"):
        return True
    if PULL_SCOPE == "draft":
        d = os.environ.get("VXPU_DRAFT_ALLREDUCE_PULL") or getattr(triton_ops, "DRAFT_ALLREDUCE_PULL", None)
        return d in (True, "1")
    return False


PULL_BLOCK = 2048  # elements a program's tile: 256 threads, 16 bytes of bf16 each
PULL_UNROLL = 2  # tiles whose loads a program issues before summing any
PULL_MAX_PROGRAMS = 8  # the B12X geometry: 3 programs at one row of 5120, 8 from five rows up


def _pull_geometry(n):
    """Programs and unroll for a partial of n elements (VXPU_ALLREDUCE_PULL_CTAS / _UNROLL the A/B overrides)."""
    block = PULL_BLOCK
    programs = int(os.environ.get("VXPU_ALLREDUCE_PULL_CTAS", "0")) or min(PULL_MAX_PROGRAMS, triton.cdiv(n, block))
    unroll = int(os.environ.get("VXPU_ALLREDUCE_PULL_UNROLL", "0")) or PULL_UNROLL
    return block, max(programs, 1), unroll


def _rowblock(d, block, start=0):
    """Whether the mix loads its coefficients once a block (VXPU_ALLREDUCE_ROWBLOCK=1; every block of
    `block` elements from `start` must lie within one row of d). Off by default: measured -0.3 us a site
    (fused push 15.0 -> 14.7 at one row), and the scalar coefficients change the FMA contraction, so the
    stream differs from the per-element form by a last bit here and there (two of four cards at three
    rows), which is not worth a tokens entry for 0.03 ms a step (hero run 15)."""
    return os.environ.get("VXPU_ALLREDUCE_ROWBLOCK", "0") == "1" and d % block == 0 and start % block == 0


def _pull_sum(c, rank, peers, site, flat, pd, out, residual, post, comb, n, d, hc, rnd, traced):
    """The pull form's sum: this card's partial `flat` and the peers' slots at `site`, in rank order."""
    block, programs, unroll = _pull_geometry(n)
    parts = [c.slot(k, site, pd)[:n] if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    kernel, ctx = triton_ops._launcher(_ar_pull_hc_kernel, flat.device, traced)
    with ctx:
        kernel[(programs,)](parts[0], parts[1], parts[2], parts[3], out.view(-1), residual.contiguous().view(-1),
                            post.contiguous().view(-1), comb.contiguous().view(-1), n, d, c.flags[rank][site, 0],
                            c.counters[rank][site, 1:2], PEERS=len(peers), HC=hc, BLOCK=block, UNROLL=unroll,
                            ROUND=rnd, ROWBLOCK=_rowblock(d, block), num_warps=8)
    return out


def allreduce_pull_hc(x, site, residual, post, comb, traced=False):
    """The pull form with the block's hc_post in the sum's epilogue: x [.., d] goes into this card's
    export slot (in the wire dtype) and the peers' flags are raised; the sum kernel waits for the
    peers' flags and reads their slots over PCIe. The same values in the same order as the one-shot
    form, so the same bits."""
    c = comm()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    d = x.shape[-1]
    hc = residual.shape[-2]
    pd = push_dtype(x)
    if not traced:
        _prime_hc(x.device, x.dtype, hc, len(peers))
    out = torch.empty(tuple(x.shape[:-1]) + (hc, d), dtype=residual.dtype, device=x.device)
    block = block_elems()
    raise_ = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], c.counters[rank][site, 0:1])
    mine = c.slot(rank, site, pd)[:n]
    kernel, ctx = triton_ops._launcher(_ar_publish_kernel, x.device, traced)
    with ctx:
        kernel[(triton.cdiv(n, block),)](flat, mine, n, c.counters[rank][site, 0:1], raise_[0], raise_[1], raise_[2],
                                         PEERS=len(peers), BLOCK=block)
    # the sum reads this card's own slot too (the published, rounded value), so the four parts agree
    return _pull_sum(c, rank, peers, site, mine, pd, out, residual, post, comb, n, d, hc, False, traced)


def gemv_allreduce_pull_hc(x, weight, scale, add, site, residual, post, comb, traced=False):
    """The fused pull: the GEMV that makes the partial writes it into this card's export slot and its
    last program raises the peers' flags (no stores into the peers' inboxes); the sum kernel reads
    the peers' slots over PCIe. The same numbers as gemv_allreduce_hc."""
    c = comm()
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], weight.shape[0]
    dtype = torch.float32 if add is not None else x.dtype
    n = M * N
    if n * dtype.itemsize > c.max_bytes or not 0 <= site < c.sites:
        raise ValueError(f"gemv_allreduce_pull_hc: {M}x{N} {dtype} at site {site} does not fit the Comm "
                         f"({c.max_bytes} bytes a site, {c.sites} sites)")
    hc = residual.shape[-2]
    if not traced:
        _prime_hc(x.device, dtype, hc, len(peers))
    flags = [c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers]
    mine = c.slot(rank, site, dtype)[:n].view(M, N)
    y = triton_ops.gemv_rows(x2, weight, scale, out_dtype=dtype, traced=traced,
                             add=add.reshape(M, N) if add is not None else None,
                             out=mine, peers=(None, c.counters[rank][site, 0:1], flags))
    out = torch.empty((M, hc, N), dtype=residual.dtype, device=x.device)
    _pull_sum(c, rank, peers, site, y.view(-1), dtype, out, residual, post, comb, n, N, hc, False, traced)
    return out.reshape(tuple(residual.shape))


def allreduce_oneshot(x, site, traced=False):
    """The one-shot form on x's card: returns the sum of every rank's x for this site."""
    c = comm()
    if c is not None and (x.numel() > c.max_elems or not 0 <= site < c.sites):
        raise RuntimeError(f"allreduce: a sum of {x.numel()} elements at site {site} does not fit the Comm "
                           f"({c.sites} sites, {c.max_elems} elements): sized too small at setup")
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    out = torch.empty_like(xc)
    block = block_elems()
    grid = (triton.cdiv(n, block),)
    pd = push_dtype(x)
    dests = _pad3([c.box(j, site, rank, pd) for j in peers], flat)
    raise_ = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], c.counters[rank][site, 0:1])
    parts = [c.box(rank, site, k, pd) if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    kernel, ctx = triton_ops._launcher(_ar_push_kernel, x.device, traced)
    with ctx:
        kernel[grid](flat, dests[0], dests[1], dests[2], n, c.counters[rank][site, 0:1], raise_[0], raise_[1], raise_[2],
                     PEERS=len(peers), BLOCK=block)
    kernel, ctx = triton_ops._launcher(_ar_sum_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1), n, c.flags[rank][site, 0],
                     c.counters[rank][site, 1:2], PEERS=len(peers), BLOCK=block, ROUND=pd != x.dtype)
    return out


_PRIMED_HC = set()
_PRIME_LAZILY = False  # the tests' harness sets it: they call the mixing kernels eagerly with no step running


def _prime_hc(dev, dtype, hc, peers):
    """Load the mixing kernels' modules on `dev` before their first real launch there (Comm.prime
    does it for the plain kernels): a module load blocks while a kernel spins on the card, and the
    reduce-scatter form's gather is first launched while its reduce is already waiting."""
    key = (dev.index, dtype, hc)
    if key in _PRIMED_HC or torch.cuda.is_current_stream_capturing():
        # under a capture the launches load their modules without running (nothing spins), and a
        # synchronize would invalidate the capture
        return
    if not _PRIME_LAZILY:
        # never here from a step: the synchronize below blocks forever when another rank program's
        # graph is spinning in a sum on this device (the drafted TP step, hero run 15); Comm.prime
        # loads these kernels at setup for the hc values it is told
        return
    _prime_hc_now(dev, dtype, hc, peers)


def _prime_hc_now(dev, dtype, hc, peers, rank=None):
    _PRIMED_HC.add((dev.index, dtype, hc))
    if rank is None:
        rank = comm().rank_of(dev)
    block = 512
    layouts = [(dtype, False)]
    if dtype == torch.float32 and push_bf16():  # the fp32 partials crossing as bf16: the step's specialization
        layouts.append((torch.bfloat16, True))
    with torch.cuda.device(dev):
        for box_dtype, rnd in layouts:
            x = torch.zeros(block, dtype=dtype, device=dev)
            y = torch.empty(hc * block, dtype=torch.bfloat16, device=dev)
            res = torch.zeros(hc * block, dtype=torch.bfloat16, device=dev)
            pst = torch.zeros(hc, dtype=torch.float32, device=dev)
            cmb = torch.zeros(hc * hc, dtype=torch.float32, device=dev)
            bx = torch.zeros(block, dtype=box_dtype, device=dev)
            parts = [bx if k != rank else x for k in range(4)]
            gs = [torch.zeros(block, dtype=torch.bfloat16, device=dev) for _ in range(3)]  # the gather, in the stream's dtype
            cnt = torch.zeros(3, dtype=torch.int32, device=dev)
            ones = [torch.ones(4, dtype=torch.int32, device=dev) for _ in range(3)]
            raise_ = [ones[0][i:i + 1] for i in range(3)]
            n = x.numel()
            _ar_sum_hc_kernel[(1,)](parts[0], parts[1], parts[2], parts[3], y, res, pst, cmb, n, block, ones[1], cnt[1:2],
                                    torch.zeros(2, dtype=torch.int64, device=dev), PEERS=peers, HC=hc, BLOCK=block, ROUND=rnd)
            ones[1].fill_(1)
            _rs_reduce_hc_kernel[(1,)](parts[0], parts[1], parts[2], parts[3], y, res, pst, cmb, gs[0], gs[1], gs[2], n, n,
                                       0, block, ones[1], cnt[1:2], raise_[0], raise_[1], raise_[2], PEERS=peers, HC=hc,
                                       BLOCK=block, ROUND=rnd, trace_ptr=torch.zeros(2, dtype=torch.int64, device=dev))
            _rs_gather_hc_kernel[(1,)](y, res, pst, cmb, gs[0], gs[1], gs[2], 0, 0, 0, n, n, block, ones[2], cnt[2:3],
                                       PEERS=peers, HC=hc, BLOCK=block)
            ones[1].fill_(1)
            pb, _, pu = _pull_geometry(block)
            yp = torch.empty(hc * pb, dtype=torch.bfloat16, device=dev)
            rp = torch.zeros(hc * pb, dtype=torch.bfloat16, device=dev)
            xp = torch.zeros(pb, dtype=dtype, device=dev)
            bp = torch.zeros(pb, dtype=box_dtype, device=dev)
            pparts = [bp if k != rank else xp for k in range(4)]
            _ar_pull_hc_kernel[(1,)](pparts[0], pparts[1], pparts[2], pparts[3], yp, rp, pst, cmb, pb, pb, ones[1], cnt[1:2],
                                     PEERS=peers, HC=hc, BLOCK=pb, UNROLL=pu, ROUND=rnd, num_warps=8)
        torch.cuda.synchronize(dev)


def allreduce_oneshot_hc(x, site, residual, post, comb, traced=False):
    """The one-shot form with the block's hc_post in the sum kernel's epilogue: x [.., d] the
    partial, residual [.., HC, d] the stream, post [.., HC], comb [.., HC, HC]; returns the new
    stream [.., HC, d] in residual's dtype (hero run 15)."""
    c = comm()
    if c is not None and (x.numel() > c.max_elems or not 0 <= site < c.sites):
        raise RuntimeError(f"allreduce: a sum of {x.numel()} elements at site {site} does not fit the Comm "
                           f"({c.sites} sites, {c.max_elems} elements): sized too small at setup")
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    d = x.shape[-1]
    hc = residual.shape[-2]
    if not traced:
        _prime_hc(x.device, x.dtype, hc, len(peers))
    out = torch.empty(tuple(x.shape[:-1]) + (hc, d), dtype=residual.dtype, device=x.device)
    # smaller programs than the plain sum's: each element's epilogue reads hc * (hc + 1) more values
    # and writes hc, so a 5120-wide row is spread over ten programs instead of two
    block = min(block_elems(), int(os.environ.get("VXPU_ALLREDUCE_HC_BLOCK", "512")))
    grid = (triton.cdiv(n, block),)
    pd = push_dtype(x)
    dests = _pad3([c.box(j, site, rank, pd) for j in peers], flat)
    raise_ = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], c.counters[rank][site, 0:1])
    parts = [c.box(rank, site, k, pd) if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    _launch_push(flat, dests, n, c.counters[rank][site, 0:1], raise_, len(peers), min(block_elems(), PUSH_BLOCK), traced)
    kernel, ctx = triton_ops._launcher(_ar_sum_hc_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1), residual.contiguous().view(-1),
                     post.contiguous().view(-1), comb.contiguous().view(-1), n, d, c.flags[rank][site, 0],
                     c.counters[rank][site, 1:2], c.trace[rank][site], PEERS=len(peers), HC=hc, BLOCK=block,
                     ROUND=pd != x.dtype, ROWBLOCK=_rowblock(d, block))
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
    dests = _pad3([c.box(j, site, rank, x.dtype) for j in peers], flat)
    offs = _pad3([j * S for j in peers], 0)
    raise_a = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[0:1])
    kernel, ctx = triton_ops._launcher(_rs_push_kernel, x.device, traced)
    with ctx:
        kernel[grid](flat, dests[0], dests[1], dests[2], offs[0], offs[1], offs[2], S, n, cnt[0:1],
                     raise_a[0], raise_a[1], raise_a[2], PEERS=len(peers), BLOCK=block)
    # round B: my slice summed in rank order (own slice at offset rank * S in x), out and to the peers' gbuf
    mine = min(S, max(n - rank * S, 0))
    parts = [c.box(rank, site, k, x.dtype) if k != rank else flat[rank * S:rank * S + S] if mine else flat
             for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    gdest = _pad3([c.gather_box(j, site, rank, x.dtype) for j in peers], flat)
    raise_b = _pad3([c.flags[j][site, 1, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[1:2])
    kernel, ctx = triton_ops._launcher(_rs_reduce_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1)[rank * S:] if mine else out.view(-1),
                     gdest[0], gdest[1], gdest[2], S, mine, c.flags[rank][site, 0], cnt[1:2],
                     raise_b[0], raise_b[1], raise_b[2], PEERS=len(peers), BLOCK=block, ROUND=False,
                     trace_ptr=c.trace[rank][site])
    # round C: the peers' reduced slices into the output
    srcs = _pad3([c.gather_box(rank, site, j, x.dtype) for j in peers], flat)
    kernel, ctx = triton_ops._launcher(_rs_gather_kernel, x.device, traced)
    with ctx:
        kernel[grid](out.view(-1), srcs[0], srcs[1], srcs[2], offs[0], offs[1], offs[2], S, n,
                     c.flags[rank][site, 1], cnt[2:3], PEERS=len(peers), BLOCK=block)
    return out


def allreduce_scatter_hc(x, site, residual, post, comb, traced=False):
    """The reduce-scatter + all-gather form with the block's hc_post in its last two kernels'
    epilogues (hero run 15): at five to eight rows the sums' PCIe bytes are the step's largest
    per-row term, and this form moves half the one-shot's. x [.., d] the partial, residual
    [.., HC, d], post [.., HC], comb [.., HC, HC]; returns the new stream [.., HC, d]."""
    c = comm()
    _check(x, site, c)
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    xc = x.contiguous()
    flat = xc.view(-1)
    n = flat.numel()
    d = x.shape[-1]
    hc = residual.shape[-2]
    if not traced:
        _prime_hc(x.device, x.dtype, hc, len(peers))
    S = -(-n // c.world)
    y = torch.empty(tuple(x.shape[:-1]) + (hc, d), dtype=residual.dtype, device=x.device)
    res, pst, cmb = residual.contiguous().view(-1), post.contiguous().view(-1), comb.contiguous().view(-1)
    block = min(block_elems(), int(os.environ.get("VXPU_ALLREDUCE_HC_BLOCK", "512")))
    grid = (triton.cdiv(S, block),)
    cnt = c.counters[rank][site]
    pd = push_dtype(x)
    dests = _pad3([c.box(j, site, rank, pd) for j in peers], flat)
    offs = _pad3([j * S for j in peers], 0)
    raise_a = _pad3([c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[0:1])
    kernel, ctx = triton_ops._launcher(_rs_push_kernel, x.device, traced)
    with ctx:
        kernel[grid](flat, dests[0], dests[1], dests[2], offs[0], offs[1], offs[2], S, n, cnt[0:1],
                     raise_a[0], raise_a[1], raise_a[2], PEERS=len(peers), BLOCK=block)
    mine = min(S, max(n - rank * S, 0))
    parts = [c.box(rank, site, k, pd) if k != rank else flat[rank * S:rank * S + S] if mine else flat
             for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    # the gathered slices of the total travel in the stream's dtype: the mix rounds the total to it anyway
    gdest = _pad3([c.gather_box(j, site, rank, residual.dtype) for j in peers], flat)
    raise_b = _pad3([c.flags[j][site, 1, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], cnt[1:2])
    kernel, ctx = triton_ops._launcher(_rs_reduce_hc_kernel, x.device, traced)
    with ctx:
        rowblock = _rowblock(d, block, S)  # every block within one row: the slices start on blocks too
        kernel[grid](parts[0], parts[1], parts[2], parts[3], y.view(-1), res, pst, cmb, gdest[0], gdest[1], gdest[2],
                     S, mine, rank * S, d, c.flags[rank][site, 0], cnt[1:2], raise_b[0], raise_b[1], raise_b[2],
                     PEERS=len(peers), HC=hc, BLOCK=block, ROUND=pd != x.dtype, ROWBLOCK=rowblock,
                     trace_ptr=c.trace[rank][site])
    srcs = _pad3([c.gather_box(rank, site, j, residual.dtype) for j in peers], flat)
    kernel, ctx = triton_ops._launcher(_rs_gather_hc_kernel, x.device, traced)
    with ctx:
        kernel[grid](y.view(-1), res, pst, cmb, srcs[0], srcs[1], srcs[2], offs[0], offs[1], offs[2], S, n, d,
                     c.flags[rank][site, 1], cnt[2:3], PEERS=len(peers), HC=hc, BLOCK=block, ROWBLOCK=rowblock)
    return y


def gemv_allreduce(x, weight, scale, site, traced=False):
    """The fused form: the row GEMV (triton_ops.gemv_rows, the plain epilogue) writes its partial
    [M, N] into every peer's inbox for this site as it stores it locally and raises the flags
    from its last program, so the reduction is the sum kernel alone. x [M, K_slice] bf16, weight
    [N, K_slice] (bf16, or fp8 with its e8m0 scales): the card's slice of a row-split linear.
    Returns the summed [M, N]."""
    c = comm()
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    M, N = x.shape[0], weight.shape[0]
    if M * N > c.max_elems or not 0 <= site < c.sites:
        raise ValueError(f"gemv_allreduce: {M}x{N} at site {site} does not fit the Comm ({c.max_elems} elements, {c.sites} sites)")
    dests = [c.box(j, site, rank, x.dtype)[:M * N].view(M, N) for j in peers]
    flags = [c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers]
    y = triton_ops.gemv_rows(x, weight, scale, traced=traced, peers=(dests, c.counters[rank][site, 0:1], flags))
    flat = y.view(-1)
    n = flat.numel()
    out = torch.empty_like(y)
    block = block_elems()
    parts = [c.box(rank, site, k, x.dtype) if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    kernel, ctx = triton_ops._launcher(_ar_sum_kernel, x.device, traced)
    with ctx:
        kernel[(triton.cdiv(n, block),)](parts[0], parts[1], parts[2], parts[3], out.view(-1), n, c.flags[rank][site, 0],
                                         c.counters[rank][site, 1:2], PEERS=len(peers), BLOCK=block, ROUND=False)
    return out


def two_part_layout(c, site, j, k, n):
    """The two-part push sum's layout of card j's inbox for source k at `site`: the routed partial
    as fp32 [n] first, the shared expert's output as bf16 [n] after it (6 n bytes of the box)."""
    raw = c.inbox[j][site][k]
    if 6 * n > raw.numel():
        raise ValueError(f"two-part push sum: {n} elements need {6 * n} bytes a box, the Comm has {raw.numel()}")
    return raw[:4 * n].view(torch.float32), raw[4 * n:6 * n].view(torch.bfloat16)


def two_part_push(c, site, rank, n):
    """What the expert kernel needs to push its routed partial (the first part): the peers' routed
    regions as flat fp32 [n], the push counter (the site's third counter) and the peers' first-round
    flags for this rank. Returns the dict triton_ops.fp4_experts_decode takes as `push`."""
    peers = c.peers(rank)
    return {"dests": [two_part_layout(c, site, j, rank, n)[0] for j in peers],
            "counter": c.counters[rank][site, 2:3],
            "flags": [c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers], "done": False}


def gemv_allreduce_hc2(x, weight, scale, add, site, residual, post, comb, traced=False):
    """The two-part push sum (hero run 15): the routed experts' partial `add` ([.., N] fp32) was
    pushed into the peers' inboxes by the expert kernel's epilogue (two_part_push) with the
    first-round flags; here the shared expert's GEMV over x pushes its bf16 output into the second
    part of the boxes with the second-round flags, from the parallel branch where it runs beside the
    routed experts, and the sum kernel waits for both and adds p_r + q_r per rank, then the ranks in
    rank order: the numbers the fused push made, nothing of the step waiting for the shared expert
    but the sum."""
    c = comm()
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], weight.shape[0]
    n = M * N
    if not 0 <= site < c.sites:
        raise ValueError(f"gemv_allreduce_hc2: site {site} of {c.sites}")
    hc = residual.shape[-2]
    if not traced:
        _prime_hc(x.device, torch.float32, hc, len(peers))
    dests = [two_part_layout(c, site, j, rank, n)[1][:n].view(M, N) for j in peers]
    flags = [c.flags[j][site, 1, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers]
    y = triton_ops.gemv_rows(x2, weight, scale, out_dtype=torch.bfloat16, traced=traced,
                             peers=(dests, c.counters[rank][site, 0:1], flags))
    yflat = y.view(-1)
    aflat = add.reshape(-1)
    out = torch.empty((M, hc, N), dtype=residual.dtype, device=x.device)
    block = min(block_elems(), int(os.environ.get("VXPU_ALLREDUCE_HC_BLOCK", "512")))
    grid = (triton.cdiv(n, block),)
    ps, qs = [], []
    for k in range(c.world):
        if k == rank:
            ps.append(aflat)
            qs.append(yflat)
        else:
            pr, qr = two_part_layout(c, rank, site, k, n)
            ps.append(pr)
            qs.append(qr)
    while len(ps) < 4:
        ps.append(aflat)
        qs.append(yflat)
    kernel, ctx = triton_ops._launcher(_ar_sum2_hc_kernel, x.device, traced)
    with ctx:
        kernel[grid](ps[0], ps[1], ps[2], ps[3], qs[0], qs[1], qs[2], qs[3], out.view(-1), residual.contiguous().view(-1),
                     post.contiguous().view(-1), comb.contiguous().view(-1), n, N, c.flags[rank][site, 0],
                     c.flags[rank][site, 1], c.counters[rank][site, 1:2], PEERS=len(peers), HC=hc, BLOCK=block,
                     ROWBLOCK=_rowblock(N, block))
    return out.reshape(tuple(residual.shape))


def gemv_allreduce_hc(x, weight, scale, add, site, residual, post, comb, traced=False):
    """The fused push (hero run 15): the mixing sum whose partial a row GEMV produces, with the
    GEMV inside. The GEMV over this rank's slice of a row-split linear (triton_ops.gemv_rows)
    writes its partial into every peer's inbox for `site` as it stores it locally and raises the
    peers' flags from its last program, so the one-shot form's push kernel and its pass over the
    partial are gone and the sum kernel (_ar_sum_hc_kernel) only waits and mixes. x [.., K] bf16
    on the fp8 grid (the attention's grouped projection, the shared expert's SwiGLU), weight
    [N, K] fp8 with its e8m0 scales (or bf16, scale None). add [.., N] fp32 or None: with it the
    partial is add + bf16(x W) in fp32, the routed experts' sum plus the shared expert's down
    projection as forward_routed adds them; without it the linear's bf16 output (the attention's
    wo_b), which the sum loads as fp32 exactly, so the peers carry half the bytes. residual
    [.., HC, N], post [.., HC], comb [.., HC, HC]. Returns the stream [.., HC, N] in residual's
    dtype: the same numbers as allreduce_oneshot_hc of the partial, added in the same order."""
    c = comm()
    rank = c.rank_of(x.device)
    peers = c.peers(rank)
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], weight.shape[0]
    dtype = torch.float32 if add is not None else x.dtype
    n = M * N
    if n * dtype.itemsize > c.max_bytes or not 0 <= site < c.sites:
        raise ValueError(f"gemv_allreduce_hc: {M}x{N} {dtype} at site {site} does not fit the Comm "
                         f"({c.max_bytes} bytes a site, {c.sites} sites)")
    hc = residual.shape[-2]
    if not traced:
        _prime_hc(x.device, dtype, hc, len(peers))
    dests = [c.box(j, site, rank, dtype)[:n].view(M, N) for j in peers]
    flags = [c.flags[j][site, 0, _slot(c, j, rank):_slot(c, j, rank) + 1] for j in peers]
    y = triton_ops.gemv_rows(x2, weight, scale, out_dtype=dtype, traced=traced,
                             add=add.reshape(M, N) if add is not None else None,
                             peers=(dests, c.counters[rank][site, 0:1], flags))
    flat = y.view(-1)
    out = torch.empty((M, hc, N), dtype=residual.dtype, device=x.device)
    block = min(block_elems(), int(os.environ.get("VXPU_ALLREDUCE_HC_BLOCK", "512")))
    grid = (triton.cdiv(n, block),)
    parts = [c.box(rank, site, k, dtype)[:n] if k != rank else flat for k in range(c.world)]
    while len(parts) < 4:
        parts.append(flat)
    kernel, ctx = triton_ops._launcher(_ar_sum_hc_kernel, x.device, traced)
    with ctx:
        kernel[grid](parts[0], parts[1], parts[2], parts[3], out.view(-1), residual.contiguous().view(-1),
                     post.contiguous().view(-1), comb.contiguous().view(-1), n, N, c.flags[rank][site, 0],
                     c.counters[rank][site, 1:2], c.trace[rank][site], PEERS=len(peers), HC=hc, BLOCK=block,
                     ROUND=False, ROWBLOCK=_rowblock(N, block))
    return out.reshape(tuple(residual.shape))


def warm_gemv_push(x, weight, scale, add, peers):
    """One eager launch of the fused push's GEMV against scratch on its own card, so its module is
    loaded before the capture (a capture forbids a first launch, and a module load blocks while a
    kernel spins on the card: Comm.prime). Shapes as gemv_allreduce_hc's; nothing crosses to a
    peer. Synchronizes the card."""
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], weight.shape[0]
    dtype = torch.float32 if add is not None else x.dtype
    with torch.cuda.device(x.device):
        dests = [torch.empty(M, N, dtype=dtype, device=x.device) for _ in range(peers)]
        flags = [torch.zeros(1, dtype=torch.int32, device=x.device) for _ in range(peers)]
        counter = torch.zeros(1, dtype=torch.int32, device=x.device)
        triton_ops.gemv_rows(x2, weight, scale, out_dtype=dtype, add=add.reshape(M, N) if add is not None else None,
                             peers=(dests, counter, flags))
        torch.cuda.synchronize(x.device)


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


def _sync_or_dump(c, streams, what, limit=20.0):
    """Wait for every card's stream, and if one does not finish within `limit` seconds print the
    flags and counters of every site on every card (a hung reduction is a flag that was never
    raised or was reset too early) and raise."""
    events = []
    for k in range(c.world):
        e = torch.cuda.Event()
        e.record(streams[k])
        events.append(e)
    t0 = time.time()
    while time.time() - t0 < limit:
        if all(e.query() for e in events):
            return
        time.sleep(0.02)
    pending = [k for k, e in enumerate(events) if not e.query()]
    for k in range(c.world):
        if k in pending:
            continue  # its flags can be read; the pending cards' streams are busy, read them too (host copies)
    for k in range(c.world):
        try:
            print(f"  card {k}: flags {c.flags[k].cpu().tolist()} counters {c.counters[k].cpu().tolist()}", flush=True)
        except Exception as e:  # a card busy spinning still answers a copy on another stream
            print(f"  card {k}: unreadable ({e})", flush=True)
    raise RuntimeError(f"allreduce hung: {what}; cards still running: {pending}")


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
            _sync_or_dump(c, streams, f"{form} at {rows} rows, eager")
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
                with torch.cuda.device(0), torch.cuda.stream(streams[0]):  # the replay goes to the current stream
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


def measure_fused(rows_list, steps, rounds, out_path, n_out=5120, k_slice=576, dtype=torch.bfloat16):
    """The fused form against the separate one at the row-split linear's shape (x [rows, 576] by
    weight [5120, 576] fp8 with e8m0 scales: a 5120 x 2304 linear split four ways): a chain of
    `steps` GEMV + reduction per card, captured, GEMV alone / GEMV then one-shot / fused; the
    outputs checked bit for bit against each other and the reference."""
    n = torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(n)]
    max_rows = max(rows_list)
    c = setup(devices, sites=max(steps, 2), max_elems=max_rows * n_out, dtype=dtype)
    streams = [torch.cuda.Stream(device=d) for d in devices]
    ws = [torch.randn(n_out, k_slice, device=d).to(torch.float8_e4m3fn) for d in devices]
    scales = [torch.full((n_out // 32, k_slice // 32), 127, dtype=torch.uint8, device=d).view(torch.float8_e8m0fnu)
              for d in devices]
    results = []
    for rows in rows_list:
        xs = [torch.randn(rows, k_slice, dtype=dtype, device=d) for d in devices]
        for k, d in enumerate(devices):  # warm the GEMV variants on every card
            with torch.cuda.device(d), torch.cuda.stream(streams[k]):
                triton_ops.gemv_rows(xs[k], ws[k], scales[k])
        _all_sync(c)
        # correctness: separate and fused against the rank-ordered sum of the cards' own GEMVs
        ys = [None] * n
        sep = [None] * n
        fused = [None] * n
        for k in range(n):
            with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                ys[k] = triton_ops.gemv_rows(xs[k], ws[k], scales[k])
                sep[k] = allreduce_oneshot(ys[k], 0)
        _sync_or_dump(c, streams, f"separate at {rows} rows")
        for k in range(n):
            with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                fused[k] = gemv_allreduce(xs[k], ws[k], scales[k], 1)
        _sync_or_dump(c, streams, f"fused at {rows} rows")
        ref = sum(y.to(0).float() for y in ys).to(dtype)
        for k in range(n):
            if not torch.equal(sep[k].to(0), ref) or not torch.equal(fused[k].to(0), ref):
                raise RuntimeError(f"fused form wrong on card {k} at {rows} rows")
        timings = {}
        for name in ("gemv", "separate", "fused"):
            graphs = []
            for k in range(n):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.device(k), torch.cuda.graph(g, stream=streams[k]):
                    for s in range(steps):
                        if name == "gemv":
                            triton_ops.gemv_rows(xs[k], ws[k], scales[k])
                        elif name == "separate":
                            allreduce_oneshot(triton_ops.gemv_rows(xs[k], ws[k], scales[k]), s)
                        else:
                            gemv_allreduce(xs[k], ws[k], scales[k], s)
                graphs.append(g)
            _all_sync(c)
            ts = []
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
                    ts.append(st.elapsed_time(en))
            timings[name] = _med(ts)["median_us"] / steps
            del graphs
        entry = {"rows": rows, "gemv_us": timings["gemv"], "separate_us": timings["separate"], "fused_us": timings["fused"],
                 "reduction_separate_us": timings["separate"] - timings["gemv"],
                 "reduction_fused_us": timings["fused"] - timings["gemv"]}
        results.append(entry)
        print(f"rows {rows:4d}: GEMV alone {timings['gemv']:.1f} us, GEMV + one-shot {timings['separate']:.1f}, "
              f"fused {timings['fused']:.1f}: the reduction costs {entry['reduction_separate_us']:.1f} separate, "
              f"{entry['reduction_fused_us']:.1f} fused (a step of {steps})", flush=True)
    with open(out_path, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "world": n, "steps": steps, "shape": f"{n_out}x{k_slice} fp8",
                   "results": results}, f, indent=1)
    return results


def measure_push(rows_list, steps, rounds, out_path, shapes=((2048, False), (576, True)), bns=(0, 32, 64, 128),
                 n_out=5120, hc=4):
    """The fused push's GEMV per tile width (hero run 15): at the step's two mixing sites, the
    attention's wo_b (x [rows, K] bf16 by weight [5120, K] fp8, a bf16 partial) and the shared
    expert's w2 (an fp32 partial with the routed sum added), a chain of `steps` sites per card,
    captured and replayed together: the GEMV alone, the GEMV then the one-shot mixing sum, and
    the fused push, for each weight-rows-per-program BN of the pushing GEMV (0: the GEMV's own
    choice). A program's tile is the span its epilogue stores into each peer over PCIe, BN
    elements a row, so BN sets the piece size. Outputs are checked bit for bit against the
    one-shot form on each card (nothing is copied between cards: an eager cross-device copy came
    back stale in this harness). Writes a table naming the best BN per pushing class."""
    n = torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(n)]
    max_rows = max(rows_list)
    c = setup(devices, sites=max(steps, 2), max_elems=max_rows * n_out, dtypes=[torch.float32, torch.bfloat16], hcs=[hc])
    streams = [torch.cuda.Stream(device=d) for d in devices]
    name = torch.cuda.get_device_name(0)
    results, blocks = [], []
    for k_slice, with_add in shapes:
        ws = [torch.randn(n_out, k_slice, device=d).to(torch.float8_e4m3fn) for d in devices]
        scales = [torch.full((n_out // 32, k_slice // 32), 127, dtype=torch.uint8, device=d).view(torch.float8_e8m0fnu)
                  for d in devices]
        for rows in rows_list:
            xs = [triton_ops.act_quant_dequant(torch.randn(rows, k_slice, dtype=torch.bfloat16, device=d)) for d in devices]
            adds = [torch.randn(rows, n_out, device=d) if with_add else None for d in devices]
            mixes = [(torch.randn(rows, hc, n_out, device=d).to(torch.bfloat16), torch.randn(rows, hc, device=d),
                      torch.randn(rows, hc, hc, device=d)) for d in devices]
            _all_sync(c)
            dtype = torch.float32 if with_add else torch.bfloat16
            bytes_site = rows * n_out * dtype.itemsize * (n - 1)  # what each card pushes a site
            for bn in bns:
                triton_ops.GEMV_PUSH_BN = bn or None
                try:
                    for k, d in enumerate(devices):  # warm every variant on every card before any capture
                        with torch.cuda.device(d), torch.cuda.stream(streams[k]):
                            from .ops import gemv_partial
                            gemv_partial(xs[k], ws[k], scales[k], adds[k])
                            warm_gemv_push(xs[k], ws[k], scales[k], adds[k], n - 1)
                    _all_sync(c)
                    sep, fused, pulled = [None] * n, [None] * n, [None] * n
                    for k in range(n):
                        with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                            sep[k] = allreduce_oneshot_hc(gemv_partial(xs[k], ws[k], scales[k], adds[k]).float(), 0, *mixes[k])
                    _sync_or_dump(c, streams, f"one-shot at {rows} rows, K {k_slice}")
                    for k in range(n):
                        with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                            fused[k] = gemv_allreduce_hc(xs[k], ws[k], scales[k], adds[k], 1, *mixes[k])
                    _sync_or_dump(c, streams, f"fused push at {rows} rows, K {k_slice}, BN {bn}")
                    for k in range(n):
                        with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                            pulled[k] = gemv_allreduce_pull_hc(xs[k], ws[k], scales[k], adds[k], 0, *mixes[k])
                    _sync_or_dump(c, streams, f"fused pull at {rows} rows, K {k_slice}, BN {bn}")
                    for k in range(n):
                        with torch.cuda.device(k):
                            for name, got in (("fused push", fused[k]), ("fused pull", pulled[k])):
                                if not torch.equal(sep[k], got):
                                    diff = (sep[k].float() - got.float()).abs().max().item()
                                    if not bn:  # the GEMV's own tile: the same K order as gemv_partial's, so the same bits
                                        raise RuntimeError(f"{name} differs from the one-shot form on card {k} at {rows} rows, K {k_slice}: max {diff}")
                                    # a forced BN changes the pushing GEMV's tiles and so its fp32 order; the bits may differ
                                    print(f"note: {name} with BN {bn} differs from the one-shot form on card {k} at {rows} rows, K {k_slice} by {diff:.3g} (the forced tile's order)", flush=True)
                    timings = {}
                    for form in ("gemv", "separate", "fused", "pull", "pullsep", "push"):
                        graphs = []
                        for k in range(n):
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.device(k), torch.cuda.graph(g, stream=streams[k]):
                                for st in range(steps):
                                    if form == "gemv":
                                        gemv_partial(xs[k], ws[k], scales[k], adds[k])
                                    elif form == "separate":
                                        allreduce_oneshot_hc(gemv_partial(xs[k], ws[k], scales[k], adds[k]).float(), st, *mixes[k])
                                    elif form == "fused":
                                        gemv_allreduce_hc(xs[k], ws[k], scales[k], adds[k], st, *mixes[k])
                                    elif form == "pull":
                                        gemv_allreduce_pull_hc(xs[k], ws[k], scales[k], adds[k], st, *mixes[k])
                                    elif form == "push":  # the push kernel alone after the GEMV, no sum
                                        push_only_hc(gemv_partial(xs[k], ws[k], scales[k], adds[k]), st)
                                    else:  # the plain pull: the partial published by its own kernel, then pulled
                                        allreduce_pull_hc(gemv_partial(xs[k], ws[k], scales[k], adds[k]).float(), st, *mixes[k])
                            graphs.append(g)
                        _all_sync(c)
                        ts = []
                        for r in range(rounds + 3):
                            st_ev = torch.cuda.Event(enable_timing=True)
                            en_ev = torch.cuda.Event(enable_timing=True)
                            st_ev.record(streams[0])
                            for k in range(n):
                                with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                                    graphs[k].replay()
                            en_ev.record(streams[0])
                            _all_sync(c)
                            if r >= 3:
                                ts.append(st_ev.elapsed_time(en_ev))
                        timings[form] = _med(ts)["median_us"] / steps
                        del graphs
                finally:
                    triton_ops.GEMV_PUSH_BN = None
                entry = {"k": k_slice, "add": with_add, "rows": rows, "bn": bn, "bytes_site": bytes_site,
                         "gemv_us": timings["gemv"], "separate_us": timings["separate"], "fused_us": timings["fused"],
                         "pull_us": timings["pull"], "pullsep_us": timings["pullsep"], "push_us": timings["push"],
                         "push_only_us": timings["push"] - timings["gemv"],
                         "sum_separate_us": timings["separate"] - timings["gemv"],
                         "sum_fused_us": timings["fused"] - timings["gemv"],
                         "sum_pull_us": timings["pull"] - timings["gemv"]}
                results.append(entry)
                print(f"K {k_slice} add={int(with_add)} rows {rows} BN {bn or 'own':>4}: GEMV alone {timings['gemv']:.1f} us, "
                      f"GEMV + one-shot {timings['separate']:.1f}, fused push {timings['fused']:.1f}, fused pull {timings['pull']:.1f}, "
                      f"GEMV + plain pull {timings['pullsep']:.1f}, push alone {timings['push'] - timings['gemv']:.1f} "
                      f"(the sum costs {entry['sum_separate_us']:.1f} separate, {entry['sum_fused_us']:.1f} fused, "
                      f"{entry['sum_pull_us']:.1f} pull; "
                      f"{bytes_site / 1024:.0f} KB pushed a site, {bytes_site / max(entry['sum_fused_us'], 1e-3) / 1e3:.1f} GB/s)",
                      flush=True)
            best = min((e for e in results if e["k"] == k_slice and e["rows"] == rows), key=lambda e: e["fused_us"])
            if best["bn"]:
                blocks.append({"shapes": f"gemv_rows rows={rows} n={n_out} k={k_slice} w=fp8 native=1"
                               + (" epi=add out=f32" if with_add else "") + " push=1",
                               "blocks": {"BN": best["bn"]}, "us": round(best["fused_us"], 2), "bytes": bytes_site,
                               "why": f"the fused push's GEMV at BN {best['bn']}: {best['fused_us']:.1f} us a site against "
                                      f"{[round(e['fused_us'], 1) for e in results if e['k'] == k_slice and e['rows'] == rows]} on {name}"})
    table = {"format": "vxpu-kernel-table/1", "gpu": name, "sites": [], "knobs": {}, "blocks": blocks,
             "explain": [f"vxpu.allreduce measure-push: world {n}, {steps} sites a chain, {rounds} rounds; BN is the pushing "
                         f"GEMV's weight rows per program, the span its epilogue stores per peer"]}
    with open(out_path, "w") as f:
        json.dump(table, f, indent=1)
    with open(out_path.replace(".json", "-raw.json"), "w") as f:
        json.dump({"gpu": name, "world": n, "steps": steps, "results": results}, f, indent=1)
    print(f"wrote {out_path}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--rows", default="1,8,33,132,264")
    m.add_argument("--chain", type=int, default=86)
    m.add_argument("--rounds", type=int, default=20)
    m.add_argument("--out", default="allreduce-table.json")
    f = sub.add_parser("measure-fused")
    f.add_argument("--rows", default="1,33")
    f.add_argument("--chain", type=int, default=86)
    f.add_argument("--rounds", type=int, default=20)
    f.add_argument("--out", default="allreduce-fused.json")
    q = sub.add_parser("measure-push")
    q.add_argument("--rows", default="1,3,5")
    q.add_argument("--chain", type=int, default=80)
    q.add_argument("--rounds", type=int, default=20)
    q.add_argument("--bns", default="0,32,64,128")
    q.add_argument("--attn-k", type=int, default=2048)  # wo_b [5120, 8 x 1024] split four ways
    q.add_argument("--ffn-k", type=int, default=576)  # the shared expert's w2 [5120, 2304] split four ways
    q.add_argument("--out", default="allreduce-push.json")
    q.add_argument("--shapes", default="both", choices=("both", "attn", "ffn"))
    a = p.parse_args(argv)
    t0 = time.time()
    if a.cmd == "measure":
        measure([int(r) for r in a.rows.split(",")], a.chain, a.rounds, a.out)
    elif a.cmd == "measure-fused":
        measure_fused([int(r) for r in a.rows.split(",")], a.chain, a.rounds, a.out)
    elif a.cmd == "measure-push":
        measure_push([int(r) for r in a.rows.split(",")], a.chain, a.rounds, a.out,
                     shapes={"both": ((a.attn_k, False), (a.ffn_k, True)), "attn": ((a.attn_k, False),),
                             "ffn": ((a.ffn_k, True),)}[a.shapes], bns=[int(b) for b in a.bns.split(",")])
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
