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

"""The resident executor (hero run 15): a step's nodes walked by one kernel per card.

Today a decode step is a CUDA graph of one kernel per node: about 1,400 launches a step
when the layers are sharded, 662 a card under tensor parallelism, each paying the graph's
per-node floor (0.75 us a dependent node measured in hero 11, 1.47 when it touches a row)
and its own ramp and tail, with the host handing the step between the cards' graphs. The
resident executor is a second lowering target for the same step graph: the planner writes
the step as a work list (one entry per node: which body, its tensors, its tile count, the
nodes it waits on), and one persistent kernel per card, with one program per SM, walks the
list. A program claims a node's tiles from a counter, computes them with the node type's
body (the same Triton template code as the standalone kernel), and counts each finished
tile into the node's arrival counter; a node that depends on another spins on the
producer's counter (an acquire read) until every tile has arrived. Nothing returns to the
host between nodes. Under tensor parallelism the per-layer sum is a node like any other:
push to the peers' inboxes, raise a flag, wait, add.

This module holds the work list, the resident kernel and, for the measurement, the same
node body launched once per node the way the graph does it, so the per-node cost inside
the resident kernel is measured against the graph's floor with the arithmetic held equal.
The first body is the GEMV template at one to BM rows (bf16 activations, fp8 e4m3 weights
with e8m0 block scales per 32, fp32 accumulation, bf16 output), the body that carries
most of a dense step's bytes; the other templates follow the same pattern (a NODE_KIND
value and a branch in the dispatcher)."""

import torch
import triton
import triton.language as tl

KIND_GEMV = 0
KIND_SUM = 1  # placeholder for the cross-card sum; not in this version's dispatcher


@triton.jit
def _gemv_tile(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, KS, pid,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One tile of y[M, N] = x[M, K] @ w[N, K]^T: the program owns BN weight rows (rows
    pid*BN..), streams them once over K in BK steps, dequantizes the fp8 tile by its e8m0
    block scales (KS scales a weight row) and multiplies by all M rows of x on the tensor
    cores in bf16 with fp32 accumulation. The same arithmetic as triton_ops._gemv_rows_kernel's
    converting body."""
    rows = pid.to(tl.int64) * BN + tl.arange(0, BN)
    live = rows < N
    ms = tl.arange(0, BM)
    mlive = ms < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kin = ks < K
        w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
        kb = k0 // 32 + tl.arange(0, BK // 32)
        e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                    mask=live[:, None] & (kb < KS)[None, :], other=127).to(tl.int32)
        sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
        w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
        x = tl.load(x_ptr + ms[:, None] * K + ks[None, :], mask=mlive[:, None] & kin[None, :], other=0.0)
        acc += tl.dot(w.to(tl.bfloat16), tl.trans(x.to(tl.bfloat16)))
    tl.store(y_ptr + ms[None, :] * N + rows[:, None], acc.to(tl.bfloat16), mask=live[:, None] & mlive[None, :])


@triton.jit
def _node_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, KS,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The graph's way: one launch per node, one program per tile."""
    _gemv_tile(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, KS, tl.program_id(0), BM, BN, BK)


@triton.jit
def _resident_kernel(xp_ptr, wp_ptr, sp_ptr, yp_ptr, m_ptr, n_ptr, k_ptr, ks_ptr, tiles_ptr,
                     dep0_ptr, dep1_ptr, claim_ptr, done_ptr, NODES, CHUNK: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The resident way, dynamic claiming: every program walks the whole work list. For node i
    it waits until its producers' arrival counters have counted every tile (an acquire read
    in a spin), then claims CHUNK tiles at a time from claim[i] until none are left, computing
    each and counting the chunk into done[i] with a release. claim and done are zero at the
    launch (step_counters' rule); the tensors' addresses are int64 in the pointer arrays."""
    for i in range(0, NODES):
        d0 = tl.load(dep0_ptr + i)
        d1 = tl.load(dep1_ptr + i)
        if d0 >= 0:
            need0 = tl.load(tiles_ptr + d0)
            while tl.atomic_add(done_ptr + d0, 0, sem="acquire") < need0:
                pass
        if d1 >= 0:
            need1 = tl.load(tiles_ptr + d1)
            while tl.atomic_add(done_ptr + d1, 0, sem="acquire") < need1:
                pass
        x_ptr = tl.multiple_of(tl.load(xp_ptr + i), 256).to(tl.pointer_type(tl.bfloat16))
        w_ptr = tl.multiple_of(tl.load(wp_ptr + i), 256).to(tl.pointer_type(tl.float8e4nv))
        s_ptr = tl.multiple_of(tl.load(sp_ptr + i), 256).to(tl.pointer_type(tl.uint8))
        y_ptr = tl.multiple_of(tl.load(yp_ptr + i), 256).to(tl.pointer_type(tl.bfloat16))
        M = tl.load(m_ptr + i)
        N = tl.multiple_of(tl.load(n_ptr + i), 16)
        K = tl.multiple_of(tl.load(k_ptr + i), 128)
        KS = tl.multiple_of(tl.load(ks_ptr + i), 4)
        tiles = tl.load(tiles_ptr + i)
        t0 = tl.atomic_add(claim_ptr + i, CHUNK)
        while t0 < tiles:
            for t in range(t0, t0 + CHUNK):
                if t < tiles:
                    _gemv_tile(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, KS, t, BM, BN, BK)
            tl.atomic_add(done_ptr + i, CHUNK, sem="release")
            t0 = tl.atomic_add(claim_ptr + i, CHUNK)


@triton.jit
def _resident_static_kernel(xp_ptr, wp_ptr, sp_ptr, yp_ptr, m_ptr, n_ptr, k_ptr, ks_ptr, tiles_ptr,
                            dep0_ptr, dep1_ptr, done_ptr, NODES, P,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The resident way, static striping: program p computes tiles p, p + P, p + 2P, ... of
    every node (no claim counter), and counts itself once into done[i] when its share is
    stored; a consumer waits for done[producer] == P. One atomic per program per node."""
    pid = tl.program_id(0)
    for i in range(0, NODES):
        d0 = tl.load(dep0_ptr + i)
        d1 = tl.load(dep1_ptr + i)
        if d0 >= 0:
            while tl.atomic_add(done_ptr + d0, 0, sem="acquire") < P:
                pass
        if d1 >= 0:
            while tl.atomic_add(done_ptr + d1, 0, sem="acquire") < P:
                pass
        x_ptr = tl.multiple_of(tl.load(xp_ptr + i), 256).to(tl.pointer_type(tl.bfloat16))
        w_ptr = tl.multiple_of(tl.load(wp_ptr + i), 256).to(tl.pointer_type(tl.float8e4nv))
        s_ptr = tl.multiple_of(tl.load(sp_ptr + i), 256).to(tl.pointer_type(tl.uint8))
        y_ptr = tl.multiple_of(tl.load(yp_ptr + i), 256).to(tl.pointer_type(tl.bfloat16))
        M = tl.load(m_ptr + i)
        N = tl.multiple_of(tl.load(n_ptr + i), 16)
        K = tl.multiple_of(tl.load(k_ptr + i), 128)
        KS = tl.multiple_of(tl.load(ks_ptr + i), 4)
        tiles = tl.load(tiles_ptr + i)
        for t in range(pid, tiles, P):
            _gemv_tile(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, KS, t, BM, BN, BK)
        tl.atomic_add(done_ptr + i, 1, sem="release")


class WorkList:
    """A step's dense GEMV nodes as the resident kernel's arrays. Each node: x [M, K] bf16,
    w [N, K] fp8 e4m3, s [N/32, K/32] e8m0 (uint8 view), y [M, N] bf16 (allocated here),
    deps: up to two node indices it waits on (-1: none). Tiles are weight-row tiles of BN."""

    def __init__(self, device, bn=16, bm=16, bk=128):
        self.device = torch.device(device)
        self.bn, self.bm, self.bk = bn, bm, bk
        self.nodes = []  # (x, w, s, y, deps)

    def gemv(self, x, w, s, deps=()):
        M, K = x.shape
        N = w.shape[0]
        assert M <= self.bm and K % self.bk == 0 and w.shape[1] == K and s.shape == (N // 32, K // 32)
        y = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
        self.nodes.append((x, w, s, y, tuple(deps)))
        return len(self.nodes) - 1, y

    def build(self):
        n = len(self.nodes)
        i64 = lambda vals: torch.tensor(vals, dtype=torch.int64, device=self.device)
        i32 = lambda vals: torch.tensor(vals, dtype=torch.int32, device=self.device)
        self.xp = i64([t[0].data_ptr() for t in self.nodes])
        self.wp = i64([t[1].data_ptr() for t in self.nodes])
        self.sp = i64([t[2].data_ptr() for t in self.nodes])
        self.yp = i64([t[3].data_ptr() for t in self.nodes])
        self.m = i32([t[0].shape[0] for t in self.nodes])
        self.n = i32([t[1].shape[0] for t in self.nodes])
        self.k = i32([t[1].shape[1] for t in self.nodes])
        self.ks = i32([t[2].shape[1] for t in self.nodes])
        self.tiles = i32([triton.cdiv(t[1].shape[0], self.bn) for t in self.nodes])
        deps = [list(t[4]) + [-1, -1] for t in self.nodes]
        self.dep0 = i32([d[0] for d in deps])
        self.dep1 = i32([d[1] for d in deps])
        self.claim = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.count = n
        return self

    def reset(self):
        """The counters back to zero: a graph node (memset) when captured, one call eagerly."""
        self.claim.zero_()
        self.done.zero_()

    def run_resident(self, programs, num_warps=4, num_stages=2, chunk=1):
        """One launch of the dynamic-claiming kernel with `programs` programs; more programs
        than resident slots is safe (a program claims a tile only when it runs, so every
        claimed tile completes)."""
        _resident_kernel[(programs,)](self.xp, self.wp, self.sp, self.yp, self.m, self.n, self.k, self.ks,
                                      self.tiles, self.dep0, self.dep1, self.claim, self.done, self.count,
                                      CHUNK=chunk, BM=self.bm, BN=self.bn, BK=self.bk,
                                      num_warps=num_warps, num_stages=num_stages)

    def run_static(self, programs, num_warps=4, num_stages=2):
        """One launch of the static-striping kernel: all `programs` must be resident at once
        (a consumer waits for every program's share), so keep it within the body's occupancy."""
        _resident_static_kernel[(programs,)](self.xp, self.wp, self.sp, self.yp, self.m, self.n, self.k, self.ks,
                                             self.tiles, self.dep0, self.dep1, self.done, self.count, programs,
                                             BM=self.bm, BN=self.bn, BK=self.bk,
                                             num_warps=num_warps, num_stages=num_stages)

    def run_launched(self, num_warps=4, num_stages=2):
        """The graph's way: one launch per node in list order (dependencies by stream order)."""
        for x, w, s, y, _ in self.nodes:
            M, K = x.shape
            N = w.shape[0]
            _node_kernel[(triton.cdiv(N, self.bn),)](x, w, s, y, M, N, K, s.shape[1],
                                                     BM=self.bm, BN=self.bn, BK=self.bk,
                                                     num_warps=num_warps, num_stages=num_stages)

    def reference(self):
        """torch's answer per node, for the check (dequantized weight, fp32)."""
        outs = []
        for x, w, s, _, _ in self.nodes:
            N, K = w.shape
            sc = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
            wf = w.to(torch.float32).view(N // 32, 32, K // 32, 32) * sc.view(N // 32, 1, K // 32, 1)
            outs.append(x.float() @ wf.reshape(N, K).t())
        return outs
