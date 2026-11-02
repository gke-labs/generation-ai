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

"""GPU kernels (Triton) for the custom ops' CUDA paths.

The fork's activation quantization (kernel_aten.act_quant followed by
the dequantization its callers do) is a dozen small aten kernels per
call; inside vxpu::fp8_linear, which is opaque to inductor, they ran
one by one, 290 calls per DeepSeek-V4.1-Flash decode token. One Triton
kernel does the same arithmetic in a single pass: per 32-element block,
amax floored at 1e-4, the smallest power of two at or above amax/448 as
the scale (a power of two maps to itself), the block divided by it,
clamped to +-448, rounded to the e4m3 grid (nearest even, as a cast
does), and multiplied back. The result is the exact value the reference
multiplies with, returned in bf16 (where it is exact).

The hyper-connection mixing (the fork's hc_split_sinkhorn) is the other
latency-bound piece: per token a 4x4 matrix normalized by rows and
columns 39 times, which inductor turns into about 80 tiny kernels per
layer. hc_split_sinkhorn below does it in one kernel per call, with the
same operations in the same order (sums over four elements may round
differently from torch's, so it matches to rounding, not bitwise).

The output head is a matrix-vector product with fp32 accumulation over
a bf16 checkpoint weight; the fork declares the weight fp32, so it was
held widened and read at twice the bytes (2.6 GB a token for
DeepSeek-V4.1-Flash). linear_f32 reads the bf16 weight and accumulates
in fp32: the same products (bf16 to fp32 is exact), summed in a
different order, so it matches to fp32 rounding.

The dense fp8 linears of a decode step are matrix-vector products, so
their cost is the weight bytes read. fp8_gemv reads the fp8 weight and
its e8m0 32x32 block scales directly (half the bytes of a bf16 copy),
scales in registers and accumulates in fp32. With group_rows it is
block-diagonal: groups of rows each read their own slice of x (the
fork's grouped output projection, wo_a).

``available()`` is False without Triton or a GPU; callers keep the aten
path then.

Each function also runs traced (traced=True): the kernel launched through
torch.library's wrap_triton, so a compiler tracing the call (inductor,
AOTInductor) sees a Triton kernel it can place in its own generated code
rather than an opaque Python call. The `vxpu_triton::` ops register the
traced forms, and `lowered` exposes them under this module's names: the
plan's lowering (lower.py) runs an implementation with it in place of
this module.
"""

import contextlib
import functools
import os
import types
import typing

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton: the aten path remains
    triton = None


def available():
    return triton is not None and torch.cuda.is_available()


def _launcher(kernel, device, traced):
    """The kernel to launch and the context to launch it in. Eager, Triton
    launches on the current device, so that is made the tensors' device (an
    engine on cuda:1 with the process's current device still 0); traced,
    the launch goes through wrap_triton and the compiler places it."""
    if traced:
        return torch.library.wrap_triton(kernel), contextlib.nullcontext()
    return kernel, torch.cuda.device(device)


if triton is not None:
    @triton.jit
    def _round_e4m3(q):
        """q (fp32, within +-448) rounded to the nearest fp8 e4m3 value, ties to even, as a cast to
        float8_e4m3fn and back does: the hardware's cast (sm_89 and later, which every GPU that
        runs these kernels is: the fp8 GEMVs load fp8 weights). The same values as the fp32 bit
        arithmetic this replaced (bit-identical outputs on the four region ops at 1, 9 and 33
        rows), and cheaper: the arithmetic ran on the padded rows of a tile too."""
        return q.to(tl.float8e4nv).to(tl.float32)

    @triton.jit
    def _act_quant_dequant_kernel(x_ptr, y_ptr, n_groups, G: tl.constexpr):
        pid = tl.program_id(0)
        g = pid.to(tl.int64) * G + tl.arange(0, G)  # int64: offsets of activations past 2^31 elements
        live = g < n_groups
        offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(x_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        v = amax / 448.0
        # the smallest power of two >= v: v's exponent, plus one unless
        # v is a power of two already (mantissa bits zero); v is normal
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
        y = _round_e4m3(q) * scale[:, None]
        tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=live[:, None])


if triton is not None:
    @triton.jit
    def _x_mxfp8_kernel(x_ptr, xq_ptr, xe_ptr, K, KS, BK: tl.constexpr):
        """A row of bf16 activations on the fp8 grid (act_quant's output) as the MXFP8 operand the
        tensor cores take: its fp8 e4m3 codes and one e8m0 scale per 32 elements. The same
        arithmetic as _act_quant_dequant_kernel finds the block's scale, so a value that was
        round_e4m3(q) * scale comes back as exactly q and scale; nothing is rounded again. One
        program per (row, BK elements)."""
        m = tl.program_id(0).to(tl.int64)
        k0 = tl.program_id(1) * BK
        ks = k0 + tl.arange(0, BK)
        kin = ks < K
        x = tl.load(x_ptr + m * K + ks, mask=kin, other=0.0).to(tl.float32)
        xg = tl.reshape(x, (BK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(tl.reshape(xg / scale[:, None], (BK,)), -448.0), 448.0)
        tl.store(xq_ptr + m * K + ks, q.to(tl.float8e4nv), mask=kin)
        kb = k0 // 32 + tl.arange(0, BK // 32)
        tl.store(xe_ptr + m * KS + kb, exponent.to(tl.uint8), mask=kb < KS)

    @triton.jit
    def _mx_recode(x, BM: tl.constexpr, BK: tl.constexpr):
        """_x_mxfp8_kernel's arithmetic on a tile x [BM, BK] fp32 (bf16 activations on the fp8
        grid, as loaded) in registers: the fp8 e4m3 codes [BM, BK] and the e8m0 scales [BM, BK /
        32] uint8 the tensor cores take, the same bits the kernel writes. The native GEMV bodies
        call it on each x tile they load (RECODE), so the recoding is the GEMV template's
        prologue and not a launch before every GEMV: at one row the x tile is 16 KB against a
        weight tile of 128 KB, and the arithmetic is a reduction over 32 elements."""
        xg = tl.reshape(x, (BM, BK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        # x times the exact reciprocal of the block's scale (a power of two: the same bits as the
        # division by it, without the divide per element; the exponent stays well inside the
        # normal range, 105 to about 135)
        inv = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(tl.reshape(xg * inv[:, :, None], (BM, BK)), -448.0), 448.0)
        return q.to(tl.float8e4nv), exponent.to(tl.uint8)


# Counters a step's kernels arrive on (hero run 11): a kernel whose last program finishes a
# cross-program reduction (window_kv's row norm, rotary and quantization after its split-K GEMV)
# counts its programs' arrivals in an int32 that must be zero when the kernel starts. The caller
# provides it: under the lowering a slice of the segment's scratch (one zeros per segment graph,
# so every captured graph owns its counters and two batches in flight on one GPU cannot share
# one, as the expert kernels' did in hero run 9); eagerly, zeros of the call's own.
FOLD_POST = os.environ.get("VXPU_FOLD_POST")  # "1": window_kv's row finish runs in its GEMV's last program (knob FOLD_POST)
FOLD_POST_RB = 1  # rows the finishing program handles at a time


def _fold_post_on():
    """Whether window_kv's finish is folded into its GEMV (FOLD_POST; VXPU_FOLD_POST=1 is the A/B)."""
    return FOLD_POST in (True, "1")


def step_counters(n, device, given=None):
    """n int32 counters at zero for one kernel call: `given` (the lowering's scratch slice) when
    the caller has them, else zeros of this call's own (a fill launch)."""
    if given is not None:
        if given.numel() < n or given.dtype != torch.int32:
            raise ValueError(f"step_counters: {n} int32 wanted, {tuple(given.shape)} {given.dtype} given")
        return given
    return torch.zeros(n, dtype=torch.int32, device=device)


MXFP8_NATIVE = os.environ.get("VXPU_MXFP8_NATIVE")  # override: force the native (1) or converting (0) GEMV body (A/B)
# Where the native bodies recode x to MXFP8: "1" in the GEMV's own prologue (_mx_recode), "0" (the default) in
# a kernel of its own before every GEMV (x_mxfp8: 284 launches of the one-row decode step). A planner knob
# (MXFP8_RECODE in the kernel table); VXPU_MXFP8_RECODE=1 is the A/B switch. Measured on four RTX PRO 6000
# (hero 10, Nov 1): the prologue takes 290 launches out of the step and leaves it at 13.3 ms at one row (13.2
# with the reciprocal multiply below), because every program of a GEMV recodes the whole x tile at every K
# step and the GEMVs lengthen by what the launches cost (gemv_rows 21.4 -> 22.6 us, gemv_pe 30.1 -> 35.1); at
# nine rows it costs 1.1 ms (21.2 -> 22.4 ms a step; gemv_rows 21.9 -> 26.3 us, gemv_pe 31.9 -> 47.7). The fold
# that pays is the producer's: the kernel that makes the activation writes the codes and scales once.
MXFP8_RECODE = os.environ.get("VXPU_MXFP8_RECODE")


def _mx_recode_on():
    """Whether the native bodies recode x themselves (MXFP8_RECODE, default off: see above)."""
    return MXFP8_RECODE == "1"


@functools.lru_cache(maxsize=None)
def _mx_native(device_index):
    """Whether the fp8 GEMV templates hand MXFP8 to the tensor cores on this GPU (Blackwell and
    later take fp8 operands with e8m0 scales directly) or convert the weight to bf16 in software
    first. Chosen from the card, not configured; VXPU_MXFP8_NATIVE only overrides it for an A/B.
    Measured cold on an RTX PRO 6000 (eight weights cycled a replay), the one-weight kernel on a
    5,120 x 8,192 weight: converting 32.6 us at one row and 44.6 at 33, native 31.6 and 33.3
    with the conversion kernel included; on 1,280 x 5,120: 12.6 -> 9.5 and 17.4 -> 12.8."""
    if MXFP8_NATIVE is not None:
        return MXFP8_NATIVE == "1"
    if triton is None or not hasattr(tl, "dot_scaled"):
        return False
    return _blackwell(device_index)


def x_mxfp8(x, traced=False):
    """x [M, K] bf16 on the fp8 grid -> (codes [M, K] float8_e4m3fn, scales [M, K / 32] uint8
    e8m0): the activations' MXFP8 form, made once per GEMV call for the native body."""
    M, K = x.shape
    xq = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    xe = torch.empty((M, K // 32), dtype=torch.uint8, device=x.device)
    if M:
        bk = 256 if K % 256 == 0 else 32
        kernel, ctx = _launcher(_x_mxfp8_kernel, x.device, traced)
        with ctx:
            kernel[(M, triton.cdiv(K, bk))](x, xq, xe, K, K // 32, BK=bk)
    return xq, xe


# A wave's crossing copies and its flag as one kernel (the ring write), "0" for cuMemcpyAsync per value and a
# stream memory operation for the flag (the A/B; stepgraph.py). Measured on four RTX PRO 6000 (hero 10, Nov 1):
# the sharded step's 24 copy nodes become 2 and three kernels (3.6 us a wave at one row against 8 copies of
# 0.6 us; 17.6 us against 8 of 2.1 at nine rows); the step time is unchanged at both, the tokens identical.
RING_WRITE = os.environ.get("VXPU_RING_WRITE", "1") != "0"
RING_VALUES = 8  # values one ring write kernel carries at most
RING_BLOCK = 4096  # int32 words a program copies


if triton is not None:
    @triton.jit
    def _ring_copy(src, dst, n, chunk, BLOCK: tl.constexpr):
        offs = chunk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m, other=0), mask=m)

    @triton.jit
    def _ring_write_kernel(s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3, s4, d4, n4, s5, d5, n5, s6, d6, n6,
                           s7, d7, n7, counter_ptr, flag_ptr, total, VALUES: tl.constexpr, FLAG: tl.constexpr,
                           BLOCK: tl.constexpr):
        """The ring write: a wave of values crossing to another GPU, copied by this GPU's programs
        straight into the reader's inboxes (peer access: dst is the reader GPU's memory) and, with
        FLAG, the wave's flag raised in the reader's memory by the last program to finish. One
        program per (BLOCK words, value); value v copies s_v -> d_v, n_v int32 words. Every
        program counts itself on counter_ptr (this GPU's memory, release at system scope after its
        stores), and the one that makes the count `total` (the grid) resets the counter for the
        next replay and writes the flag: its acquire of the counter has made every program's
        stores visible before the flag store, which is what the reader's wait relies on. The
        step graph captured one copy node per value and a memory operation for the flag before
        (eight copies a range boundary); this is one kernel a wave."""
        chunk = tl.program_id(0)
        v = tl.program_id(1)
        if v == 0:
            _ring_copy(s0, d0, n0, chunk, BLOCK)
        if VALUES > 1:
            if v == 1:
                _ring_copy(s1, d1, n1, chunk, BLOCK)
        if VALUES > 2:
            if v == 2:
                _ring_copy(s2, d2, n2, chunk, BLOCK)
        if VALUES > 3:
            if v == 3:
                _ring_copy(s3, d3, n3, chunk, BLOCK)
        if VALUES > 4:
            if v == 4:
                _ring_copy(s4, d4, n4, chunk, BLOCK)
        if VALUES > 5:
            if v == 5:
                _ring_copy(s5, d5, n5, chunk, BLOCK)
        if VALUES > 6:
            if v == 6:
                _ring_copy(s6, d6, n6, chunk, BLOCK)
        if VALUES > 7:
            if v == 7:
                _ring_copy(s7, d7, n7, chunk, BLOCK)
        if FLAG:
            tl.debug_barrier()
            done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
            if done == total - 1:
                tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
                tl.store(flag_ptr, 1)


def ring_write(pairs, counter, flag, stream):
    """Launch the ring write for `pairs` [(src, dst)] (contiguous CUDA tensors, src on this GPU,
    dst its inbox on the reader GPU, equal bytes, a multiple of 4) on `stream`: one kernel per
    RING_VALUES pairs, the last one raising `flag` (int32 [1] on the reader GPU, 1) once every
    copy is visible; `counter` is an int32 [1] zero on this GPU, kept across replays. Returns the
    number of kernels launched."""
    launched = 0
    for i in range(0, len(pairs), RING_VALUES):
        group = pairs[i:i + RING_VALUES]
        last = i + RING_VALUES >= len(pairs)
        words = [src.numel() * src.element_size() // 4 for src, _ in group]
        args = []
        for (src, dst), n in zip(group, words):
            args += [src.view(torch.uint8).view(torch.int32), dst.view(torch.uint8).view(torch.int32), n]
        while len(args) < 3 * RING_VALUES:
            args += [args[0], args[1], 0]
        chunks = triton.cdiv(max(words), RING_BLOCK)
        with torch.cuda.stream(stream), torch.cuda.device(group[0][0].device):
            _ring_write_kernel[(chunks, len(group))](*args, counter, flag, chunks * len(group), VALUES=len(group),
                                                     FLAG=last, BLOCK=RING_BLOCK)
        launched += 1
    return launched


def act_quant_dequant(x, traced=False):
    """The fork's act_quant (32-element blocks, ue8m0 scales) followed by
    the dequantization, for a CUDA tensor whose last dim is a multiple
    of 32: bf16 values on the fp8 grid times each block's scale."""
    assert x.is_cuda and x.shape[-1] % 32 == 0
    xc = x.contiguous()
    y = torch.empty(xc.shape, dtype=torch.bfloat16, device=x.device)
    n_groups = xc.numel() // 32
    G = 16
    kernel, ctx = _launcher(_act_quant_dequant_kernel, x.device, traced)
    with ctx:
        kernel[(triton.cdiv(n_groups, G),)](xc, y, n_groups, G=G)
    return y


if triton is not None:
    @triton.jit
    def _hc_sinkhorn_kernel(m_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, eps,
                            HC: tl.constexpr, ITERS: tl.constexpr):
        t = tl.program_id(0)
        # Triton's launcher passes a Python float as fp32, inductor's (the
        # traced form) as fp64, which would make the arithmetic below fp64:
        # fix it to fp32 so both forms compute the same bits
        eps = eps.to(tl.float32)
        W: tl.constexpr = (2 + HC) * HC
        r = tl.arange(0, HC)
        row = m_ptr + t * W
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        pre = tl.sigmoid(tl.load(row + r).to(tl.float32) * s0 + tl.load(base_ptr + r)) + eps
        post = 2 * tl.sigmoid(tl.load(row + HC + r).to(tl.float32) * s1 + tl.load(base_ptr + HC + r))
        ij = r[:, None] * HC + r[None, :]
        c = tl.load(row + 2 * HC + ij).to(tl.float32) * s2 + tl.load(base_ptr + 2 * HC + ij)
        c = tl.exp(c - tl.max(c, axis=1)[:, None])  # softmax over the last dim
        c = c / tl.sum(c, axis=1)[:, None] + eps
        c = c / (tl.sum(c, axis=0)[None, :] + eps)
        for _ in tl.static_range(ITERS - 1):
            c = c / (tl.sum(c, axis=1)[:, None] + eps)
            c = c / (tl.sum(c, axis=0)[None, :] + eps)
        tl.store(pre_ptr + t * HC + r, pre)
        tl.store(post_ptr + t * HC + r, post)
        tl.store(comb_ptr + t * HC * HC + ij, c)


def hc_split_sinkhorn_supported(hc_mult):
    """The kernel's blocks are hc wide, and Triton needs power-of-two block
    shapes; other widths take the aten path."""
    return hc_mult > 0 and hc_mult & (hc_mult - 1) == 0


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, traced=False):
    """The fork's hc_split_sinkhorn for CUDA tensors, one kernel per call:
    mixes [..., (2 + hc) * hc] -> pre [..., hc], post [..., hc],
    comb [..., hc, hc] (fp32), comb made doubly stochastic. hc must be a
    power of two (hc_split_sinkhorn_supported)."""
    hc = hc_mult
    lead = mixes.shape[:-1]
    m = mixes.reshape(-1, (2 + hc) * hc).contiguous()  # widened in the kernel (see linear_f32)
    T = m.shape[0]
    pre = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    post = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    comb = torch.empty(T, hc, hc, dtype=torch.float32, device=mixes.device)
    if T:
        kernel, ctx = _launcher(_hc_sinkhorn_kernel, mixes.device, traced)
        with ctx:
            kernel[(T,)](m, hc_scale.float().contiguous(), hc_base.float().contiguous(),
                         pre, post, comb, float(eps), HC=hc, ITERS=sinkhorn_iters)
    return pre.view(*lead, hc), post.view(*lead, hc), comb.view(*lead, hc, hc)


if triton is not None:
    @triton.jit
    def _linear_f32_kernel(x_ptr, w_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)  # int64: weights past 2^31 elements
        live = rows < N
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
            x = tl.load(x_ptr + m * K + ks, mask=kin, other=0.0).to(tl.float32)
            acc += w.to(tl.float32) * x[None, :]
        tl.store(y_ptr + m * N + rows, tl.sum(acc, axis=1), mask=live)


def linear_f32(x, weight, block_n=16, block_k=512, traced=False):
    """x [M, K] (any float dtype, taken as fp32) times weight [N, K]
    (bf16, read as is) transposed -> fp32 [M, N], fp32 accumulation. For
    a decode step's rows: up to GEMV_ROWS_MAX bf16 rows, one pass over the
    weight serves them all (gemv_rows); otherwise each row reads it."""
    M, K = x.shape
    N = weight.shape[0]
    if weight.dim() != 2 or weight.shape[1] != K:
        raise ValueError(f"linear_f32: x {tuple(x.shape)} and weight {tuple(weight.shape)} disagree on K")
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and M <= GEMM_ROWS_MAX and K % 32 == 0 and N % 32 == 0:
        return gemv_rows(x, weight, None, out_dtype=torch.float32, traced=traced)
    # x is widened to fp32 in the kernel, not here: traced, a torch-side
    # cast would let inductor fuse it into x's producer and skip the
    # rounding to x's dtype that the eager path does
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    if M and N:
        kernel, ctx = _launcher(_linear_f32_kernel, x.device, traced)
        with ctx:
            kernel[(triton.cdiv(N, block_n), M)](x, weight.contiguous(), y, N, K, BN=block_n, BK=block_k)
    return y


if triton is not None:
    @triton.jit
    def _fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, XS, R, KS,
                         BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)  # int64: weights past 2^31 elements
        live = rows < N
        g = (pid * BN) // R  # BN divides R: one group per program
        xb = x_ptr + m * XS + g * K
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
            kb = k0 // 32 + tl.arange(0, BK // 32)
            e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                        mask=live[:, None] & (kb < KS)[None, :], other=127).to(tl.int32)
            # e8m0: 2^(e-127) from the exponent bits; e = 0 is 2^-127, below
            # fp32's normal range; e = 255 is the format's NaN
            sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
            sc = tl.where(e == 255, float("nan"), sc)
            w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
            x = tl.load(xb + ks, mask=kin, other=0.0).to(tl.float32)
            acc += w * x[None, :]
        # rounded to the output's dtype at the store: no separate cast kernel
        tl.store(y_ptr + m * N + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=live)


if triton is not None:
    @triton.jit
    def _gemv_rows_kernel(x_ptr, xq_ptr, xe_ptr, w_ptr, s_ptr, y_ptr, res_ptr, post_ptr, comb_ptr, add_ptr, M, N, K, XS, R, KS,
                          cnt_ptr, nw_ptr, freqs_ptr, start_ptr, out_ptr, S, eps,
                          HAS_SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          EPI_QUANT: tl.constexpr = False, KSPLIT: tl.constexpr = 1, EPI_HC: tl.constexpr = 0,
                          HAS_ADD: tl.constexpr = False, NATIVE: tl.constexpr = False,
                          RECODE: tl.constexpr = False, POST: tl.constexpr = False, RD: tl.constexpr = 0,
                          PBD: tl.constexpr = 0, RB: tl.constexpr = 8, MT: tl.constexpr = False):
        """x [M, K] (bf16) times weight [N, K] transposed, every row of x by one pass over the
        weight: a program owns BN weight rows, streams them once, and multiplies each tile by all
        M rows of x (with MT, by the BM rows of its row tile: the grid's first axis is the row
        tile, program_id(0), and the weight tile is streamed once per row tile, from L2 after the
        first) on the tensor cores (tl.dot, bf16 in, fp32 accumulation; an fp8 weight times
        its e8m0 block scale is exact in bf16, so the products are exact). With NATIVE the tensor
        cores take the MXFP8 operands as they are: the fp8 weight tile with its e8m0 scales and
        x's fp8 codes and e8m0 scales (xq_ptr, xe_ptr: x_mxfp8's output, the same layout as x
        with K / 32 scales a row), tl.dot_scaled, no conversion in software; the products are
        the same exact values, the sums come in the tensor core's order. With RECODE the native
        body makes x's codes and scales itself from the bf16 tile it loads (_mx_recode; xq_ptr
        and xe_ptr are then unused). Up to BM rows. With
        EPI_QUANT the bf16 outputs are quantized to the fp8 grid per 32 consecutive outputs of a
        row (the fork's act_quant before the next linear) before the store; BN is then a multiple
        of 32, so a program holds whole blocks. With KSPLIT > 1, that many programs share a row
        tile, each reducing a K/KSPLIT slice and storing its fp32 partial sums in y[split] (y is
        [KSPLIT, M, N] fp32); the consumer adds them. A narrow weight (1,280 or 512 rows) gives
        too few programs otherwise: 80 or 32 for 188 SMs, and the kernel runs at a third of the
        card's bandwidth. With EPI_HC = hc > 0 the epilogue is the fork's hc_post: the linear's
        bf16 output x (with HAS_ADD, x = bf16(add + x) first: the experts' fp32 sum plus the
        shared expert, as forward_routed adds them) is expanded back to the hc copies of the
        stream, out[m, h, n] = post[m, h] * x[m, n] + sum_k comb[m, k, h] * residual[m, k, n],
        in fp32 and stored as bf16 in y [M, hc, N]; the sum runs over k first, as torch's does.
        With POST (window_kv's finish, hero run 11), after its store every program adds one to
        cnt_ptr[0] and the one that arrives last sums the KSPLIT partials of every row and
        finishes it (_rows_finish: norm with nw_ptr, the rotary of the last RD elements at the
        row's position, the fp8-grid quantization) into out_ptr [M, N] bf16: the row_post launch
        that followed this kernel runs inside it. cnt_ptr[0] must be zero at the launch
        (step_counters)."""
        if MT:
            mt = tl.program_id(0)
            pid = tl.program_id(1)
            kp = tl.program_id(2)
        else:
            mt = 0
            pid = tl.program_id(0)
            kp = tl.program_id(1)
        ybase = y_ptr
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)  # int64: weights past 2^31 elements
        live = rows < N
        g = (pid * BN) // R  # BN divides R: one group per program
        ms = mt * BM + tl.arange(0, BM)
        mlive = ms < M
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        span = K // KSPLIT
        k_start = kp * span
        for k0 in range(k_start, k_start + span, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
            if NATIVE:
                kb = k0 // 32 + tl.arange(0, BK // 32)
                e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                            mask=live[:, None] & (kb < KS)[None, :], other=127)
                if RECODE:
                    x = tl.load(x_ptr + ms[:, None] * XS + g * K + ks[None, :],
                                mask=mlive[:, None] & kin[None, :], other=0.0).to(tl.float32)
                    xq, xe = _mx_recode(x, BM, BK)
                else:
                    xq = tl.load(xq_ptr + ms[:, None] * XS + g * K + ks[None, :],
                                 mask=mlive[:, None] & kin[None, :], other=0.0)
                    xe = tl.load(xe_ptr + ms[:, None] * (XS // 32) + g * (K // 32) + kb[None, :],
                                 mask=mlive[:, None] & (kb < KS)[None, :], other=127)
                acc = tl.dot_scaled(w, e, "e4m3", tl.trans(xq), xe, "e4m3", acc)
            else:
                if HAS_SCALE:
                    kb = k0 // 32 + tl.arange(0, BK // 32)
                    e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                                mask=live[:, None] & (kb < KS)[None, :], other=127).to(tl.int32)
                    sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
                    sc = tl.where(e == 255, float("nan"), sc)
                    w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
                x = tl.load(x_ptr + ms[:, None] * XS + g * K + ks[None, :],
                            mask=mlive[:, None] & kin[None, :], other=0.0)
                acc += tl.dot(w.to(tl.bfloat16), tl.trans(x.to(tl.bfloat16)))
        if EPI_QUANT:
            hg = tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (BN // 32, 32, BM))  # the linear's bf16 output
            amax = tl.maximum(tl.max(tl.abs(hg), axis=1), 1e-4)
            v = amax / 448.0
            bits = v.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) & 0xFF
            exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(hg / scale[:, None, :], -448.0), 448.0)
            acc = tl.reshape(_round_e4m3(q) * scale[:, None, :], (BN, BM))
        if EPI_HC > 0:
            omask = live[:, None] & mlive[None, :]
            xo = acc.to(tl.bfloat16).to(tl.float32)  # the linear's bf16 output
            if HAS_ADD:
                add = tl.load(add_ptr + ms[None, :] * N + rows[:, None], mask=omask, other=0.0)
                xo = (add + xo).to(tl.bfloat16).to(tl.float32)  # forward_routed: (experts + shared).type_as(x)
            for h in tl.static_range(EPI_HC):
                mix = tl.zeros((BN, BM), dtype=tl.float32)
                for k in tl.static_range(EPI_HC):
                    c = tl.load(comb_ptr + ms * (EPI_HC * EPI_HC) + k * EPI_HC + h, mask=mlive, other=0.0)
                    res = tl.load(res_ptr + (ms * EPI_HC + k)[None, :] * N + rows[:, None], mask=omask,
                                  other=0.0).to(tl.float32)
                    mix += c[None, :] * res
                post = tl.load(post_ptr + ms * EPI_HC + h, mask=mlive, other=0.0)
                out = post[None, :] * xo + mix
                tl.store(y_ptr + (ms * EPI_HC + h)[None, :] * N + rows[:, None], out.to(tl.bfloat16), mask=omask)
        else:
            if KSPLIT > 1:
                y_ptr += kp.to(tl.int64) * M * N
            tl.store(y_ptr + ms[None, :] * N + rows[:, None], acc.to(y_ptr.dtype.element_ty),
                     mask=live[:, None] & mlive[None, :])
            if POST:
                tl.debug_barrier()  # every thread's partials are stored before the program's arrival
                arrived = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
                if arrived == tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2) - 1:  # the last to arrive finishes every row
                    _rows_finish(ybase, nw_ptr, freqs_ptr, start_ptr, out_ptr, M, N, S, eps, 0, 1, KSPLIT, RD, PBD, RB)

    @triton.jit
    def _rows_finish(p_ptr, nw_ptr, freqs_ptr, start_ptr, out_ptr, M, N, S, eps, first, stride,
                     KSUM: tl.constexpr, RD: tl.constexpr, BD: tl.constexpr, RB: tl.constexpr):
        """_row_post_kernel's finish for window_kv (NORM, the rotary of the last RD elements, QUANT,
        one row a token) over rows first, first + stride, ... of the fp32 partials p [KSUM, M, N],
        after every program of the GEMV has arrived: program p of P finishes every P-th row, RB
        of them at a time, so the rows finish in parallel as the separate kernel's programs did.
        The partials were written by other programs during this kernel: this SM has not loaded
        their lines before, so the loads come from L2 after the acquire (volatile loads, which
        also forbid vectorizing, measured 3x slower). The same arithmetic as the kernel."""
        eps = eps.to(tl.float32)
        d = tl.arange(0, BD)
        live = d < N
        for m0 in range(first, M, stride * RB):
            ms = m0 + tl.arange(0, RB) * stride
            mlive = ms < M
            mask = mlive[:, None] & live[None, :]
            x = tl.zeros((RB, BD), dtype=tl.float32)
            for k in tl.static_range(KSUM):
                x += tl.load(p_ptr + (k * M + ms)[:, None] * N + d[None, :], mask=mask, other=0.0)
            x = x.to(tl.bfloat16).to(tl.float32)
            rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / N + eps)
            nw = tl.load(nw_ptr + d, mask=live, other=0.0).to(tl.float32)
            x = (nw[None, :] * (x * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
            if RD > 0:
                ev, od = tl.split(tl.reshape(x, (RB, BD // 2, 2)))
                even_dim = 2 * tl.arange(0, BD // 2)
                tail = even_dim >= N - RD
                pair = (even_dim - (N - RD)) // 2
                pos = tl.load(start_ptr + ms // S, mask=mlive, other=0) + ms % S
                fb = freqs_ptr + pos[:, None] * RD + 2 * pair[None, :]
                fmask = mlive[:, None] & tail[None, :]
                c = tl.load(fb, mask=fmask, other=1.0)
                sn = tl.load(fb + 1, mask=fmask, other=0.0)
                x = tl.reshape(tl.join(ev * c - od * sn, ev * sn + od * c), (RB, BD))
                x = x.to(tl.bfloat16).to(tl.float32)  # the fork writes the rotated tail back as bf16
            xg = tl.reshape(x, (RB, BD // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-4)
            v = amax / 448.0
            bits = v.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) & 0xFF
            exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(xg / scale[:, :, None], -448.0), 448.0)
            x = tl.reshape(_round_e4m3(q) * scale[:, :, None], (RB, BD))
            tl.store(out_ptr + ms[:, None] * N + d[None, :], x.to(tl.bfloat16), mask=mask)


# rows one pass over the weight serves (BM); beyond, a GEMM. VXPU_GEMV_ROWS_MAX overrides it, to
# measure where the one-pass kernel should hand over (hero 6: 33 rows were slower at 64 than at 32)
GEMV_ROWS_MAX = int(os.environ.get("VXPU_GEMV_ROWS_MAX", "64"))
# Beyond GEMV_ROWS_MAX the same kernels run tiled over rows (hero run 12): a program owns one row
# tile of at most GEMV_ROWS_MAX rows, and the grid gains a row-tile axis, the fastest, so the
# programs that share a weight tile launch together and the weight comes from DRAM about once
# (the tile set sits in L2 for the next row tile). Up to GEMM_ROWS_MAX rows; longer inputs
# (a prefill) take the bf16 copies and torch's GEMM as before.
GEMM_ROWS_MAX = int(os.environ.get("VXPU_GEMM_ROWS_MAX", "1024"))


def row_tiles(M, bm):
    """How many row tiles of bm rows M rows take (1 for M <= bm: the one-pass GEMV)."""
    return max(1, -(-M // bm))


def _grid(tiles, *rest):
    """A kernel's grid: the row-tile axis first (fastest) when there is more than one tile, else
    the untiled grid, so a one-pass launch is the same launch as before."""
    return (tiles,) + tuple(rest) if tiles > 1 else tuple(rest)

# The GEMV templates' block parameters per shape class (hero 11). A shape class names a template
# and what fixes its tiles: the rows, the weight's shape and dtype, the prologue/epilogue flags.
# GEMV_BLOCKS holds the kernel table's choice per class (plan.load_table fills it from the
# table's "blocks" entries, which `vxpu kernels measure` wrote from a sweep on the card);
# a class the table does not name runs the dispatcher's defaults. GEMV_CLASSES counts every
# class launched since the process started with the blocks that ran, and goes into the kernel
# record's stats (stats.gemv_classes): the measuring step's input, and the ledger's.
GEMV_BLOCKS = {}
GEMV_CLASSES = {}
GEMV_BLOCK_NAMES = ("BN", "BK", "KSPLIT", "warps", "stages")


def gemv_class(template, M, N, K, w, **flags):
    """The shape class key of a GEMV call: "gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1 epi=quant".
    Flags that are False, None or 0 are left out, so a key reads the same from the dispatcher and
    from the measuring step."""
    parts = [template, f"rows={M}", f"n={N}", f"k={K}", f"w={w}"]
    parts += [f"{k}={1 if v is True else v}" for k, v in sorted(flags.items()) if v not in (False, None, 0, "")]
    return " ".join(parts)


def gemv_blocks(key, bm, bn, bk, ksplit=1, warps=4, stages=3, allow_ksplit=False, traced=False):
    """The blocks a GEMV call of shape class `key` runs with: the table's (GEMV_BLOCKS) where it
    names the class, else the defaults given. KSPLIT is taken from the table only where the
    caller can consume partial sums (allow_ksplit). Counts the class for the record: `count` is
    every launch since the process started (warm-up and prefill included), `sites` the call sites
    the lowering traced (traced=True), which is the class's launches a decode step."""
    chosen = GEMV_BLOCKS.get(key)
    if chosen:
        bn = int(chosen.get("BN", bn))
        bk = int(chosen.get("BK", bk))
        warps = int(chosen.get("warps", warps))
        stages = int(chosen.get("stages", stages))
        if allow_ksplit:
            ksplit = int(chosen.get("KSPLIT", ksplit))
    entry = GEMV_CLASSES.setdefault(key, {"count": 0, "sites": 0, "blocks": {}, "from_table": bool(chosen)})
    entry["count"] += 1
    entry["sites"] += int(bool(traced))
    entry["blocks"] = {"BM": bm, "BN": bn, "BK": bk, "KSPLIT": ksplit, "warps": warps, "stages": stages}
    return bm, bn, bk, ksplit, warps, stages


def _launch_opts(warps, stages):
    """Triton's launch options for a GEMV: given only when they differ from the compiler's
    defaults (4 warps, 3 stages), so a class the table does not name launches as before."""
    opts = {}
    if warps != 4:
        opts["num_warps"] = warps
    if stages != 3:
        opts["num_stages"] = stages
    return opts


def _rows_blocks(M, K):
    """Block shape for the GEMV kernels: rows padded to a tensor-core tile, 16 weight rows per
    program (32 above 32 rows: measured on an L4, 16 fell to 130-175 GB/s at 33-64 rows where 32
    held 180-255), a K step of 256 where K allows it. The one-weight kernel takes 32 rows per
    program on Blackwell (gemv_rows); the kernels holding two weights or a prologue keep 16."""
    bm = 16
    while bm < min(M, GEMV_ROWS_MAX):  # more rows than GEMV_ROWS_MAX: tiles of this many (row_tiles)
        bm *= 2
    bn = 32 if bm > 32 else 16
    for bk in (256, 128, 64, 32):
        if K % bk == 0:
            return bm, bn, bk
    return bm, bn, 32


def _pe_blocks(M, K, tiles=3, device=None):
    """Block shape for the GEMV kernels that hold `tiles` fp32 tiles of BK (x, and one or two
    weights) and their temporaries: _rows_blocks' choice with BK halved until twice the tiles fit
    in half the device's shared memory (an L4 has about 100 KB and needed 113 to 123 KB at 32 and
    33 rows with BK 256; the hero's cards have 227 KB). A stopgap for the planner's autotuning."""
    bm, bn, bk = _rows_blocks(M, K)
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    budget = getattr(props, "shared_memory_per_block_optin", 101376) // 2
    while bk > 32 and (bm + (tiles - 1) * bn) * bk * 4 * 2 > budget:
        bk //= 2
    return bm, bn, bk


def gemv_rows(x, weight, scale=None, group_rows=None, out_dtype=torch.bfloat16, blocks=None, traced=False,
              quant_out=False, ksplit=1, hc=None, post=None):
    """x [M, G*K] bf16 times weight [N, K] (bf16, or fp8 e4m3 with [N/32, K/32] e8m0 block
    scales) transposed -> [M, N] in out_dtype, the weight read once for all M <= GEMV_ROWS_MAX
    rows. group_rows as fp8_gemv. quant_out quantizes the bf16 result to the fp8 grid per 32
    outputs in the kernel's epilogue (the fork's act_quant before the next linear) instead of in
    a kernel of its own; the programs then own 32 rows of the weight. ksplit > 1 returns the
    fp32 partial sums [ksplit, M, N] of that many K slices for the consumer to add (a narrow
    weight's programs, see the kernel); the K step is shrunk until it divides the slice. hc =
    (residual [M, hc, N] bf16, post [M, hc] fp32, comb [M, hc, hc] fp32, add [M, N] fp32 or None)
    makes the epilogue the fork's hc_post (see the kernel) and the result the stream [M, hc, N]
    bf16: the block's output mixed back into the hyper-connection stream without a kernel of its
    own. post = (counter [1] int32 at zero, norm_weight [N], freqs [positions, rope_dim / 2, 2]
    fp32, start [b] int64, seqlen, eps, rope_dim) makes the last program to arrive finish every
    row of the result as window_kv's row_post would (the kernel's POST) and returns that [M, N]
    bf16 instead of the partials."""
    M = x.shape[0]
    N, K = weight.shape
    R = group_rows or N
    bm, bn, bk = blocks or _rows_blocks(M, K)
    native = scale is not None and _mx_native(x.device.index)
    if blocks is None and native:
        # the native body's shape: 16 weight rows per program at every row count (measured cold on
        # an RTX PRO 6000 through this dispatcher: 5,120 x 8,192 at one row 31.6 us against 32.6
        # with 32 rows and 33.1 with 64; at 33 rows 33.2 against 33.6; 1,280 x 5,120 at 33 rows
        # 12.1 against 14.9), with the 256-wide K step where K allows
        bn = 16
    elif blocks is None and _blackwell(x.device):
        # 32 weight rows per program on this card for the converting body (measured cold on an RTX
        # PRO 6000, 5120 x 8192 fp8: 16 rows 33.3 us at one row of x and 36.8 at six; 32 rows 32.8
        # and 32.9), so the six-row verify step of a speculative decode stops paying 10 to 20% over
        # one row. In the step: 26.6 -> 24.2 us a launch over 250 launches. The dual GEMV lost with
        # 32 (28 -> 38 us), so only this kernel takes it. The planner's choice per shape and GPU.
        bn = max(bn, 32)
    if quant_out:
        bn = max(bn, 32)
    # the shape class and its blocks: the table's where it names the class (ksplit "narrow": the
    # caller takes fp32 partial sums, so the split count is the dispatcher's or the table's)
    narrow = ksplit == "narrow"
    if narrow:
        ksplit = _narrow_ksplit(N, bn, K, bk, x.device)
    HC0 = hc[0].shape[1] if hc is not None else 0
    key = gemv_class("gemv_rows", M, N, K, "fp8" if scale is not None else "bf16", native=native,
                     epi=("quant" if quant_out else f"hc{HC0}" if HC0 else None),
                     groups=(N // R if R != N else 0), out=("f32" if out_dtype == torch.float32 and not narrow else None),
                     split=narrow, tiles=(row_tiles(M, bm) if M > bm else 0))
    bm, bn, bk, ksplit, warps, stages = gemv_blocks(key, bm, bn, bk, ksplit, allow_ksplit=narrow, traced=traced)
    if ksplit > 1:
        if K % ksplit or quant_out:
            raise ValueError(f"gemv_rows: a K split of {ksplit} needs K {K} divisible and no quant_out")
        while bk > 32 and (K // ksplit) % bk:
            bk //= 2
        out_dtype = torch.float32
    tiles = row_tiles(M, bm)  # more rows than one pass serves: the kernel runs once per row tile
    if M > GEMM_ROWS_MAX or N % R or R % bn or x.shape[1] != (N // R) * K or K % 32 or N % 32 or bk % 32:
        raise ValueError(f"gemv_rows: x {tuple(x.shape)}, weight {tuple(weight.shape)}, group_rows {group_rows}")
    if scale is not None and (weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu
                              or tuple(scale.shape) != (N // 32, K // 32)):
        raise ValueError(f"gemv_rows: weight {weight.dtype} {tuple(weight.shape)} needs e8m0 scales "
                         f"[{N // 32}, {K // 32}], got {scale.dtype} {tuple(scale.shape)}")
    HC = 0
    if hc is not None:
        residual, hpost, comb, add = hc
        HC = residual.shape[1]
        if (ksplit > 1 or quant_out or tuple(residual.shape) != (M, HC, N) or tuple(hpost.shape) != (M, HC)
                or tuple(comb.shape) != (M, HC, HC) or (add is not None and tuple(add.shape) != (M, N))):
            raise ValueError(f"gemv_rows: hc epilogue with residual {tuple(residual.shape)}, post {tuple(post.shape)}, "
                             f"comb {tuple(comb.shape)}, add {None if add is None else tuple(add.shape)} for [{M}, {N}]")
        residual = residual.contiguous()
        hpost = hpost.contiguous().float()
        comb = comb.contiguous().float()
        add = add.contiguous().float() if add is not None else residual
        y = torch.empty((M, HC, N), dtype=torch.bfloat16, device=x.device)
    else:
        residual = hpost = comb = add = x
        y = torch.empty((ksplit, M, N) if ksplit > 1 else (M, N), dtype=out_dtype, device=x.device)
    cnt = nw = freqs = start = out = x
    seqlen, eps, rope_dim = 1, 0.0, 0
    if post is not None:
        if hc is not None or quant_out or N % 32:
            raise ValueError("gemv_rows: the post finish needs a plain result with N a multiple of 32")
        cnt, nw, freqs, start, seqlen, eps, rope_dim = post
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M and N:
        w = weight.contiguous()
        s = scale.contiguous().view(torch.uint8) if scale is not None else w
        xc = x.contiguous()
        recode = native and _mx_recode_on()
        xq, xe = x_mxfp8(xc, traced=traced) if native and not recode else (xc, xc)
        kernel, ctx = _launcher(_gemv_rows_kernel, x.device, traced)
        with ctx:
            kernel[_grid(tiles, triton.cdiv(N, bn), ksplit)](
                xc, xq, xe, w, s, y, residual, hpost, comb, add,
                M, N, K, x.shape[1], R, scale.shape[1] if scale is not None else 0,
                cnt, nw, freqs, start, out, seqlen, float(eps),
                HAS_SCALE=scale is not None, BM=bm, BN=bn, BK=bk,
                EPI_QUANT=quant_out, KSPLIT=ksplit, EPI_HC=HC,
                HAS_ADD=hc is not None and hc[3] is not None, NATIVE=native,
                RECODE=recode, POST=post is not None, RD=rope_dim,
                PBD=triton.next_power_of_2(N) if post is not None else 0,
                RB=FOLD_POST_RB, MT=tiles > 1, **_launch_opts(warps, stages))
    return out if post is not None else y


def _narrow_ksplit(N, bn, K, bk, device=None):
    """How many programs share a row tile's reduction in a GEMV over a weight of N rows: the
    smallest power of two that gives at least two programs per SM, as long as the slices keep
    whole K steps. The planner's choice per shape and GPU; this is the stopgap."""
    sms = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device()).multi_processor_count
    ks = 1
    while triton.cdiv(N, bn) * ks < 2 * sms and ks < 16 and K % (2 * ks * bk) == 0:
        ks *= 2
    return ks


if triton is not None:
    @triton.jit
    def _dual_gemv_swiglu_kernel(x_ptr, xq_ptr, xe_ptr, w1_ptr, s1_ptr, w3_ptr, s3_ptr, h_ptr, M, N, K, KS, limit,
                                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr = 1,
                                 NATIVE: tl.constexpr = False, RECODE: tl.constexpr = False, MT: tl.constexpr = False):
        """The SwiGLU front of an FFN for up to BM rows (a row tile of BM with MT: the grid's
        first axis, as _gemv_rows_kernel) in one pass over both weights: a program
        owns BN rows of w1 and of w3 (fp8, e8m0 block scales), streams them once against all M rows
        of x (bf16 on the fp8 grid), and in its epilogue rounds the two products to bf16 as the
        fork's linears do, clamps them (the swiglu limit; inf for none), takes silu(gate) * up,
        rounds to bf16, and quantizes the result to the fp8 grid per 32 outputs (act_quant with
        ue8m0 scales, as the w2 linear's input), stored as bf16. BN is a multiple of 32, so a
        program holds whole quantization blocks. With KSPLIT > 1, that many programs share a row
        tile, each reducing a K slice and storing its two fp32 partial sums in h_ptr as [KSPLIT, 2,
        M, N] (gate then up); _swiglu_rows_kernel adds them and does the epilogue (a shared expert
        of 2,304 rows gives 72 programs otherwise). NATIVE as in _gemv_rows_kernel: the tensor
        cores take both weights' fp8 tiles with their e8m0 scales and x's codes and scales (made
        from the bf16 tile in the kernel with RECODE)."""
        limit = limit.to(tl.float32)
        if MT:
            mt = tl.program_id(0)
            pid = tl.program_id(1)
            kp = tl.program_id(2)
        else:
            mt = 0
            pid = tl.program_id(0)
            kp = tl.program_id(1)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)
        live = rows < N
        ms = mt * BM + tl.arange(0, BM)
        mlive = ms < M
        acc1 = tl.zeros((BN, BM), dtype=tl.float32)
        acc3 = tl.zeros((BN, BM), dtype=tl.float32)
        span = K // KSPLIT
        k_start = kp * span
        for k0 in range(k_start, k_start + span, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            kb = k0 // 32 + tl.arange(0, BK // 32)
            smask = live[:, None] & (kb < KS)[None, :]
            wmask = live[:, None] & kin[None, :]
            if NATIVE:
                if RECODE:
                    x = tl.load(x_ptr + ms[:, None] * K + ks[None, :], mask=mlive[:, None] & kin[None, :],
                                other=0.0).to(tl.float32)
                    xq, xe = _mx_recode(x, BM, BK)
                else:
                    xq = tl.load(xq_ptr + ms[:, None] * K + ks[None, :], mask=mlive[:, None] & kin[None, :], other=0.0)
                    xe = tl.load(xe_ptr + ms[:, None] * KS + kb[None, :], mask=mlive[:, None] & (kb < KS)[None, :], other=127)
                xqt = tl.trans(xq)
                w = tl.load(w1_ptr + rows[:, None] * K + ks[None, :], mask=wmask, other=0.0)
                e = tl.load(s1_ptr + (rows // 32)[:, None] * KS + kb[None, :], mask=smask, other=127)
                acc1 = tl.dot_scaled(w, e, "e4m3", xqt, xe, "e4m3", acc1)
                w = tl.load(w3_ptr + rows[:, None] * K + ks[None, :], mask=wmask, other=0.0)
                e = tl.load(s3_ptr + (rows // 32)[:, None] * KS + kb[None, :], mask=smask, other=127)
                acc3 = tl.dot_scaled(w, e, "e4m3", xqt, xe, "e4m3", acc3)
            else:
                x = tl.load(x_ptr + ms[:, None] * K + ks[None, :], mask=mlive[:, None] & kin[None, :], other=0.0)
                xt = tl.trans(x.to(tl.bfloat16))
                w = tl.load(w1_ptr + rows[:, None] * K + ks[None, :], mask=wmask, other=0.0)
                e = tl.load(s1_ptr + (rows // 32)[:, None] * KS + kb[None, :], mask=smask, other=127).to(tl.int32)
                sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
                sc = tl.where(e == 255, float("nan"), sc)
                w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
                acc1 += tl.dot(w.to(tl.bfloat16), xt)
                w = tl.load(w3_ptr + rows[:, None] * K + ks[None, :], mask=wmask, other=0.0)
                e = tl.load(s3_ptr + (rows // 32)[:, None] * KS + kb[None, :], mask=smask, other=127).to(tl.int32)
                sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
                sc = tl.where(e == 255, float("nan"), sc)
                w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
                acc3 += tl.dot(w.to(tl.bfloat16), xt)
        if KSPLIT > 1:
            part = h_ptr + kp.to(tl.int64) * 2 * M * N
            omask = live[:, None] & mlive[None, :]
            tl.store(part + ms[None, :] * N + rows[:, None], acc1, mask=omask)
            tl.store(part + M * N + ms[None, :] * N + rows[:, None], acc3, mask=omask)
            return
        gate = acc1.to(tl.bfloat16).to(tl.float32)  # the linears' bf16 outputs, as the fork sees them
        up = acc3.to(tl.bfloat16).to(tl.float32)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        gate = tl.minimum(gate, limit)
        h = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
        # act_quant of the bf16 result per 32 consecutive outputs of a row (the w2 linear's input)
        hg = tl.reshape(h, (BN // 32, 32, BM))
        amax = tl.maximum(tl.max(tl.abs(hg), axis=1), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(hg / scale[:, None, :], -448.0), 448.0)
        hq = tl.reshape(_round_e4m3(q) * scale[:, None, :], (BN, BM))
        tl.store(h_ptr + ms[None, :] * N + rows[:, None], hq.to(tl.bfloat16), mask=live[:, None] & mlive[None, :])


if triton is not None:
    @triton.jit
    def _swiglu_rows_kernel(p_ptr, h_ptr, M, N, limit, KSUM: tl.constexpr, BN: tl.constexpr):
        """The dual GEMV's epilogue over its KSUM fp32 partial sums [KSUM, 2, M, N]: one program
        per (row, BN outputs; BN a multiple of 32): the sums rounded to bf16 as the linears'
        outputs, the clamps, silu(gate) * up rounded to bf16, quantized to the fp8 grid per 32
        (act_quant), stored as bf16 h [M, N]."""
        limit = limit.to(tl.float32)
        m = tl.program_id(0).to(tl.int64)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        live = n < N
        g = tl.zeros((BN,), dtype=tl.float32)
        u = tl.zeros((BN,), dtype=tl.float32)
        for k in tl.static_range(KSUM):
            g += tl.load(p_ptr + (k * 2 * M + m) * N + n, mask=live, other=0.0)
            u += tl.load(p_ptr + ((k * 2 + 1) * M + m) * N + n, mask=live, other=0.0)
        gate = g.to(tl.bfloat16).to(tl.float32)
        up = u.to(tl.bfloat16).to(tl.float32)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        gate = tl.minimum(gate, limit)
        h = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
        hg = tl.reshape(h, (BN // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(hg), axis=1), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(hg / scale[:, None], -448.0), 448.0)
        hq = tl.reshape(_round_e4m3(q) * scale[:, None], (BN,))
        tl.store(h_ptr + m * N + n, hq.to(tl.bfloat16), mask=live)


def shared_ffn(x, w1, s1, w3, s3, w2, s2, swiglu_limit, quantized=False, y_experts=None, residual=None, post=None,
               comb=None, traced=False):
    """An fp8 SwiGLU FFN (the shared expert) for a decode step's rows, in two to four kernels:
    x [.., K] bf16 quantized to the fp8 grid (act_quant_dequant, unless `quantized` says the
    caller already did: the FFN front's x_in; with y_experts [.., K] fp32 and residual [.., hc,
    K], post, comb the w2 GEMV's epilogue adds the experts' sum and mixes the result into the
    stream, returning [.., hc, K] bf16, see gemv_rows), the dual GEMV over w1 and w3
    [I, K] with the SwiGLU and the w2 input's quantization in its epilogue (or, split over K for
    enough programs, in a small kernel after it), and the w2 [K, I] GEMV. Returns
    [.., K] in x's dtype. The fork's Expert.forward launches about nine kernels for the same."""
    K = x.shape[-1]
    I = w1.shape[0]
    xq = (x.contiguous() if quantized else act_quant_dequant(x, traced=traced)).reshape(-1, K)
    M = xq.shape[0]
    if M > GEMM_ROWS_MAX or K % 32 or I % 32 or w3.shape != w1.shape or tuple(w2.shape) != (K, I):
        raise ValueError(f"shared_ffn: x {tuple(x.shape)}, w1 {tuple(w1.shape)}, w2 {tuple(w2.shape)}")
    bm, _, bk = _pe_blocks(M, K, tiles=3, device=x.device)
    tiles = row_tiles(M, bm)
    bn = 32  # whole quantization blocks per program
    ks = _narrow_ksplit(I, bn, K, bk, x.device)
    native = _mx_native(x.device.index)
    key = gemv_class("dual_gemv_swiglu", M, I, K, "fp8", native=native, tiles=(tiles if tiles > 1 else 0))
    bm, bn, bk, ks, warps, stages = gemv_blocks(key, bm, bn, bk, ks, allow_ksplit=True, traced=traced)
    if (bn % 32 and ks == 1) or I % bn or K % ks or (K // ks) % bk:
        raise ValueError(f"shared_ffn: blocks BN={bn} BK={bk} KSPLIT={ks} do not fit I={I}, K={K} (whole quantization "
                         f"blocks per program unless the epilogue runs apart, whole K steps per slice)")
    h = torch.empty(M, I, dtype=torch.bfloat16, device=x.device)
    limit = float(swiglu_limit) if swiglu_limit > 0 else float("inf")
    if M:
        kernel, ctx = _launcher(_dual_gemv_swiglu_kernel, x.device, traced)
        parts = torch.empty(ks, 2, M, I, dtype=torch.float32, device=x.device) if ks > 1 else h
        recode = native and _mx_recode_on()
        xq8, xe8 = x_mxfp8(xq, traced=traced) if native and not recode else (xq, xq)
        with ctx:
            kernel[_grid(tiles, triton.cdiv(I, bn), ks)](
                xq, xq8, xe8, w1.contiguous(), s1.contiguous().view(torch.uint8),
                w3.contiguous(), s3.contiguous().view(torch.uint8), parts, M, I, K,
                s1.shape[1], limit, BM=bm, BN=bn, BK=bk, KSPLIT=ks, NATIVE=native,
                RECODE=recode, MT=tiles > 1, **_launch_opts(warps, stages))
        if ks > 1:  # the epilogue over the partial sums
            epi, ctx = _launcher(_swiglu_rows_kernel, x.device, traced)
            with ctx:
                epi[(M, triton.cdiv(I, 256))](parts, h, M, I, limit, KSUM=ks, BN=256)
    if residual is not None:
        HC = residual.shape[-2]
        out = gemv_rows(h, w2, s2, traced=traced, hc=(residual.reshape(M, HC, K), post.reshape(M, HC),
                                                      comb.reshape(M, HC, HC), y_experts.reshape(M, K)))
        return out.reshape(tuple(x.shape[:-1]) + (HC, K))
    y = gemv_rows(h, w2, s2, traced=traced)
    return y.reshape(tuple(x.shape[:-1]) + (K,)).to(x.dtype)


if triton is not None:
    @triton.jit
    def _gemv_pe_kernel(x_ptr, xq_ptr, xe_ptr, w_ptr, s_ptr, y_ptr, xn_ptr, nw_ptr, freqs_ptr, start_ptr, sign, M, N, K, KS, R, S, eps,
                        PRO_NORM: tl.constexpr, PRO_ROTARY: tl.constexpr, PRO_QUANT: tl.constexpr,
                        STORE_XN: tl.constexpr, EPI_ROTARY: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr = False,
                        RECODE: tl.constexpr = False, MT: tl.constexpr = False):
        """_gemv_rows_kernel with the work around the GEMV folded in (and its row tiles, MT). Prologue, per row of x as it
        is read: RMSNorm with nw_ptr's weight (PRO_NORM; the fork's: fp32, rounded to bf16), then
        quantization to the fp8 grid per 32 elements (PRO_QUANT, act_quant with ue8m0 scales), the
        normalized row stored by program 0 (STORE_XN, for whoever else reads it). Epilogue: the
        bf16 outputs rotated in adjacent pairs over the last RD of every HD outputs (EPI_ROTARY,
        freqs_ptr the whole table of complex frequencies as its real view [positions, RD / 2, 2],
        read at each row's position, start_ptr[row // S] + row % S; sign -1 for the inverse), as
        the fork's apply_rotary_emb. BN must divide
        HD - RD and be even. PRO_ROTARY rotates the input the same way instead (the pairs along K,
        the last RD of every HD; sign picks the direction), for the
        attention's output. Weight rows come in groups of R, group g reading x[:, g * K:(g + 1)
        * K] (R = N for one group; BN divides R). NATIVE (no prologue: x already on the fp8
        grid) as in _gemv_rows_kernel: the tensor cores take the weight's fp8 tile with its scales
        and x's codes and scales (xq_ptr, xe_ptr; made from the bf16 tile in the kernel with
        RECODE)."""
        eps = eps.to(tl.float32)
        sign = sign.to(tl.float32)
        if MT:
            mt = tl.program_id(0)
            pid = tl.program_id(1)
        else:
            mt = 0
            pid = tl.program_id(0)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)
        live = rows < N
        g = (pid * BN) // R  # the group: BN divides R, so one group per program
        XS = (N // R) * K  # x's row stride: one K-slice per group
        xb = x_ptr + g * K
        ms = mt * BM + tl.arange(0, BM)
        mlive = ms < M
        pos = tl.load(start_ptr + ms // S, mask=mlive, other=0) + ms % S  # each row's position in the table
        rstd = tl.zeros((BM,), dtype=tl.float32)
        if PRO_NORM:
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                x = tl.load(xb + ms[:, None] * XS + ks[None, :], mask=mlive[:, None] & (ks < K)[None, :],
                            other=0.0).to(tl.float32)
                rstd += tl.sum(x * x, axis=1)
            rstd = tl.math.rsqrt(rstd / K + eps)
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            xmask = mlive[:, None] & kin[None, :]
            if NATIVE:
                kb = k0 // 32 + tl.arange(0, BK // 32)
                w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
                e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                            mask=live[:, None] & (kb < KS)[None, :], other=127)
                if RECODE:
                    x = tl.load(xb + ms[:, None] * XS + ks[None, :], mask=xmask, other=0.0).to(tl.float32)
                    xq, xe = _mx_recode(x, BM, BK)
                else:
                    xq = tl.load(xq_ptr + g * K + ms[:, None] * XS + ks[None, :], mask=xmask, other=0.0)
                    xe = tl.load(xe_ptr + g * (K // 32) + ms[:, None] * (XS // 32) + kb[None, :],
                                 mask=mlive[:, None] & (kb < KS)[None, :], other=127)
                acc = tl.dot_scaled(w, e, "e4m3", tl.trans(xq), xe, "e4m3", acc)
            else:
                x = tl.load(xb + ms[:, None] * XS + ks[None, :], mask=xmask, other=0.0).to(tl.float32)
                if PRO_NORM:
                    nw = tl.load(nw_ptr + ks, mask=kin, other=0.0).to(tl.float32)
                    x = (nw[None, :] * (x * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
                if PRO_ROTARY:  # adjacent pairs along K, in the last RD of every HD elements
                    ev, od = tl.split(tl.reshape(x, (BM, BK // 2, 2)))  # (rows, pairs) each
                    even_k = k0 + 2 * tl.arange(0, BK // 2)
                    jk = even_k % HD
                    tail = (jk >= HD - RD) & (even_k < K)
                    pair = (jk - (HD - RD)) // 2
                    pm = mlive[:, None] & tail[None, :]
                    fb = freqs_ptr + pos[:, None] * RD + 2 * pair[None, :]
                    c = tl.load(fb, mask=pm, other=1.0)
                    sn = tl.load(fb + 1, mask=pm, other=0.0) * sign
                    x = tl.reshape(tl.join(ev * c - od * sn, ev * sn + od * c), (BM, BK))
                    x = x.to(tl.bfloat16).to(tl.float32)
                if STORE_XN:
                    if pid == 0:
                        tl.store(xn_ptr + ms[:, None] * K + ks[None, :], x.to(tl.bfloat16), mask=xmask)
                if PRO_QUANT:
                    xg = tl.reshape(x, (BM, BK // 32, 32))
                    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-4)
                    v = amax / 448.0
                    bits = v.to(tl.int32, bitcast=True)
                    exponent = (bits >> 23) & 0xFF
                    exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
                    scale = (exponent << 23).to(tl.float32, bitcast=True)
                    q = tl.minimum(tl.maximum(xg / scale[:, :, None], -448.0), 448.0)
                    x = tl.reshape(_round_e4m3(q) * scale[:, :, None], (BM, BK))
                w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
                kb = k0 // 32 + tl.arange(0, BK // 32)
                e = tl.load(s_ptr + (rows // 32)[:, None] * KS + kb[None, :],
                            mask=live[:, None] & (kb < KS)[None, :], other=127).to(tl.int32)
                sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
                sc = tl.where(e == 255, float("nan"), sc)
                w = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
                acc += tl.dot(w.to(tl.bfloat16), tl.trans(x.to(tl.bfloat16)))
        y = acc.to(tl.bfloat16).to(tl.float32)  # the linear's bf16 output, as the fork rotates it
        if EPI_ROTARY:
            y3 = tl.permute(tl.reshape(y, (BN // 2, 2, BM)), (0, 2, 1))  # (pairs, rows, 2)
            ev, od = tl.split(y3)
            even_dim = pid.to(tl.int64) * BN + 2 * tl.arange(0, BN // 2)
            j = even_dim % HD
            tail = j >= HD - RD
            pair = (j - (HD - RD)) // 2
            pm = tail[:, None] & mlive[None, :]
            fb = freqs_ptr + pos[None, :] * RD + 2 * pair[:, None]
            c = tl.load(fb, mask=pm, other=1.0)
            sn = tl.load(fb + 1, mask=pm, other=0.0) * sign
            y3 = tl.join(ev * c - od * sn, ev * sn + od * c)
            y = tl.reshape(tl.permute(y3, (0, 2, 1)), (BN, BM))
        tl.store(y_ptr + ms[None, :] * N + rows[:, None], y.to(tl.bfloat16), mask=live[:, None] & mlive[None, :])


def attn_q(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs, start, seqlen, heads, head_dim, rope_dim,
           quantized=False, traced=False):
    """The attention's query path for a decode step's rows, in three or four kernels: x [.., K]
    quantized to the fp8 grid (skipped when `quantized`: the front already did it), the
    down-projection wq_a [R, K] (fp8, e8m0 block scales), one program per row for q_norm and the
    fp8-grid quantization of its result (the normalized query also kept, for the indexer), then
    the up-projection wq_b [N, R] with the rotary of every head's last rope_dim dims in its
    epilogue (freqs [positions, rope_dim / 2, 2] fp32: the whole table of complex frequencies as
    its real view, read at each row's position, start [b] int64 + the token's index of seqlen:
    no gather before the kernels). Returns (qn [.., R] bf16, the normalized query; q [.., N] bf16,
    rotated). The per-row work runs once a row,
    not once per program of the wide GEMV, which recomputed it 2,048 times and lost to the
    compiler's version. The fork launches about nine kernels for the same."""
    K = x.shape[-1]
    R, N = wq_a.shape[0], wq_b.shape[0]
    xq = (x.contiguous() if quantized else act_quant_dequant(x, traced=traced)).reshape(-1, K)
    M = xq.shape[0]
    bm, bn, bk = _pe_blocks(M, R, tiles=2, device=x.device)
    BR = triton.next_power_of_2(R)  # the per-row kernel masks past R
    tiles = row_tiles(M, bm)
    if (M > GEMM_ROWS_MAX or K % 32 or R % 32 or N % head_dim or head_dim % bn
            or (head_dim - rope_dim) % bn or rope_dim % 2 or wq_b.shape[1] != R
            or tuple(freqs.shape[1:]) != (rope_dim // 2, 2) or start.numel() * seqlen != M):
        raise ValueError(f"attn_q: x {tuple(x.shape)}, wq_a {tuple(wq_a.shape)}, wq_b {tuple(wq_b.shape)}, "
                         f"heads {heads} x {head_dim}, rope {rope_dim}, freqs {tuple(freqs.shape)}, "
                         f"start {tuple(start.shape)} x {seqlen}")
    freqs = freqs.contiguous()
    start = start.contiguous()
    qr = gemv_rows(xq, wq_a, sa, traced=traced, ksplit="narrow")
    ks = qr.shape[0] if qr.dim() == 3 else 1  # the split the dispatcher (or the table) chose
    qn = torch.empty(M, R, dtype=torch.bfloat16, device=x.device)
    qq = torch.empty(M, R, dtype=torch.bfloat16, device=x.device)
    q = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if M:
        post, ctx = _launcher(_row_post_kernel, x.device, traced)
        with ctx:
            post[(M,)](qr, norm_weight.contiguous(), freqs, start, 1.0, qq, qn, M, R, 1, seqlen, float(norm_eps),
                       NORM=True, RD=0, QUANT=True, STORE_PRE=True, BD=BR, KSUM=ks)
        kernel, ctx = _launcher(_gemv_pe_kernel, x.device, traced)
        native = _mx_native(x.device.index)  # no prologue here: qq is on the fp8 grid already
        recode = native and _mx_recode_on()
        key = gemv_class("gemv_pe", M, N, R, "fp8", native=native, epi="rotary", hd=head_dim, rd=rope_dim,
                         tiles=(tiles if tiles > 1 else 0))
        bm, bn, bk, _, warps, stages = gemv_blocks(key, bm, bn, bk, traced=traced)
        if N % bn or (head_dim - rope_dim) % bn or bn % 2 or R % bk:
            raise ValueError(f"attn_q: blocks BN={bn} BK={bk} do not fit N={N}, heads of {head_dim} with rope {rope_dim}, R={R}")
        qq8, qe8 = x_mxfp8(qq, traced=traced) if native and not recode else (qq, qq)
        with ctx:
            kernel[_grid(tiles, triton.cdiv(N, bn))](
                qq, qq8, qe8, wq_b.contiguous(), sb.contiguous().view(torch.uint8), q, q, qq,
                freqs, start, 1.0, M, N, R, sb.shape[1], N, seqlen, 0.0,
                PRO_NORM=False, PRO_ROTARY=False, PRO_QUANT=False, STORE_XN=False,
                EPI_ROTARY=True, HD=head_dim, RD=rope_dim, BM=bm, BN=bn, BK=bk,
                NATIVE=native, RECODE=recode, MT=tiles > 1, **_launch_opts(warps, stages))
    lead = tuple(x.shape[:-1])
    return qn.reshape(lead + (R,)), q.reshape(lead + (N,))


if triton is not None:
    @triton.jit
    def _row_post_kernel(x_ptr, nw_ptr, freqs_ptr, start_ptr, sign, y_ptr, pre_ptr, M, D, HPR, S, eps,
                         NORM: tl.constexpr, RD: tl.constexpr, QUANT: tl.constexpr, STORE_PRE: tl.constexpr,
                         BD: tl.constexpr, KSUM: tl.constexpr = 1):
        """A row's finishing after a projection, one program per row of x [M, D] (bf16): RMSNorm
        with nw_ptr's weight (NORM; fp32, rounded to bf16), the rotary of the last RD elements in
        adjacent pairs (freqs_ptr: the whole table of complex frequencies as its real view
        [positions, RD / 2, 2], cos and sin interleaved, read in place at the token's position:
        token t of row b sits at start_ptr[b] + t, S tokens a row, so no gather of the table runs
        before the kernel; sign -1 for the inverse rotation; RD = 0 for none),
        quantization to the fp8 grid per 32 elements (QUANT, act_quant with ue8m0 scales), stored
        as bf16. The fork's kv_norm, apply_rotary_emb and act_quant, each a kernel or more, as
        one. BD is D rounded up to a power of two; RD divides D and is even. HPR consecutive rows
        share a frequency row (the heads of one token). STORE_PRE also stores the row before the
        quantization (pre_ptr), for a reader that wants it unquantized. With KSUM > 1, x holds
        that many fp32 partial sums [KSUM, M, D] of a split GEMV (gemv_rows(ksplit)); their sum,
        rounded to bf16 as the linear's output, is the row."""
        eps = eps.to(tl.float32)
        sign = sign.to(tl.float32)
        m = tl.program_id(0)
        d = tl.arange(0, BD)
        live = d < D
        if KSUM > 1:
            x = tl.zeros((BD,), dtype=tl.float32)
            for k in tl.static_range(KSUM):
                x += tl.load(x_ptr + (k * M + m) * D + d, mask=live, other=0.0)
            x = x.to(tl.bfloat16).to(tl.float32)
        else:
            x = tl.load(x_ptr + m * D + d, mask=live, other=0.0).to(tl.float32)
        if NORM:
            rstd = tl.math.rsqrt(tl.sum(x * x) / D + eps)
            nw = tl.load(nw_ptr + d, mask=live, other=0.0).to(tl.float32)
            x = (nw * (x * rstd)).to(tl.bfloat16).to(tl.float32)
        if RD > 0:
            ev, od = tl.split(tl.reshape(x, (BD // 2, 2)))
            even_dim = 2 * tl.arange(0, BD // 2)
            tail = even_dim >= D - RD
            pair = (even_dim - (D - RD)) // 2
            tok = m // HPR  # the token this row belongs to (HPR heads a token)
            pos = tl.load(start_ptr + tok // S) + tok % S
            fb = freqs_ptr + pos * RD + 2 * pair
            c = tl.load(fb, mask=tail, other=1.0)
            sn = tl.load(fb + 1, mask=tail, other=0.0) * sign
            x = tl.reshape(tl.join(ev * c - od * sn, ev * sn + od * c), (BD,))
            x = x.to(tl.bfloat16).to(tl.float32)  # the fork writes the rotated tail back as bf16
        if STORE_PRE:
            tl.store(pre_ptr + m * D + d, x.to(tl.bfloat16), mask=live)
        if QUANT:
            xg = tl.reshape(x, (BD // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-4)
            v = amax / 448.0
            bits = v.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) & 0xFF
            exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(xg / scale[:, None], -448.0), 448.0)
            x = tl.reshape(_round_e4m3(q) * scale[:, None], (BD,))
        tl.store(y_ptr + m * D + d, x.to(tl.bfloat16), mask=live)


def window_kv(x, wkv, scale, norm_weight, norm_eps, freqs, start, seqlen, rope_dim, quantized=False, traced=False,
              counters=None):
    """The attention's window key for a decode step's rows, in two or three kernels: x [.., K]
    quantized to the fp8 grid (skipped when `quantized`), the projection wkv [D, K] (fp8, e8m0
    block scales), and the row's finishing: kv_norm, the rotary of the last rope_dim elements
    (freqs [positions, rope_dim / 2, 2] fp32: the whole table's real view, read at the row's
    position start[b] + t, seqlen tokens a row) and the quantization to the fp8 grid the cache
    holds. Returns [.., D] bf16."""
    K = x.shape[-1]
    D = wkv.shape[0]
    xq = (x.contiguous() if quantized else act_quant_dequant(x, traced=traced)).reshape(-1, K)
    M = xq.shape[0]
    BD = triton.next_power_of_2(D)  # the per-row kernel masks past D
    if (M > GEMM_ROWS_MAX or K % 32 or D % 32 or rope_dim % 2 or tuple(freqs.shape[1:]) != (rope_dim // 2, 2)
            or start.numel() * seqlen != M):
        raise ValueError(f"window_kv: x {tuple(x.shape)}, wkv {tuple(wkv.shape)}, rope {rope_dim}, "
                         f"freqs {tuple(freqs.shape)}, start {tuple(start.shape)} x {seqlen}")
    if _fold_post_on() and M:  # the row's finish in the GEMV's last program (FOLD_POST, hero run 11)
        cnt = step_counters(1, x.device, counters)
        y = gemv_rows(xq, wkv, scale, traced=traced, ksplit="narrow",
                      post=(cnt, norm_weight.contiguous(), freqs.contiguous(), start.contiguous(), seqlen,
                            float(norm_eps), rope_dim))
        return y.reshape(tuple(x.shape[:-1]) + (D,))
    kv = gemv_rows(xq, wkv, scale, traced=traced, ksplit="narrow")
    ks = kv.shape[0] if kv.dim() == 3 else 1  # the split the dispatcher (or the table) chose
    y = torch.empty(M, D, dtype=torch.bfloat16, device=x.device)
    if M:
        kernel, ctx = _launcher(_row_post_kernel, x.device, traced)
        with ctx:
            kernel[(M,)](kv, norm_weight.contiguous(), freqs.contiguous(), start.contiguous(), 1.0, y, y, M, D, 1,
                         seqlen, float(norm_eps), NORM=True, RD=rope_dim, QUANT=True, STORE_PRE=False, BD=BD, KSUM=ks)
    return y.reshape(tuple(x.shape[:-1]) + (D,))


def attn_out(o, freqs, start, seqlen, wo_a, sa, groups, rank, wo_b, sb, head_dim, rope_dim, residual=None, post=None,
             comb=None, prepared=False, traced=False):
    """The attention's output path for a decode step's rows, in three kernels: one program per
    (row, head) for the inverse rotary of the head's last rope_dim dims (freqs [positions,
    rope_dim / 2, 2] fp32: the whole table's real view, read at the row's position start[b] + t,
    seqlen tokens a row) and the fp8-grid quantization; the
    grouped projection wo_a [groups * rank, heads_per_group * head_dim] (fp8, e8m0 block scales,
    block-diagonal over the groups) quantizing its result in its epilogue; wo_b [dim, groups * rank]. Returns
    [.., dim] bf16, or with residual [.., hc, dim], post [.., hc] and comb [.., hc, hc] the
    stream [.., hc, dim] with the output mixed in by wo_b's epilogue (the fork's hc_post). With
    `prepared`, o comes rotated back and quantized already (sparse_attn's finish) and the per-row
    kernel is skipped. The fork launches about five kernels; the fold of the rotary into the
    grouped GEMV's prologue recomputed it per program and was dropped."""
    *lead, heads, hd = o.shape
    M = o.numel() // (heads * hd)
    Kg = heads * hd // groups  # a group's slice of o
    N1, dim = wo_a.shape[0], wo_b.shape[0]
    if (M > GEMM_ROWS_MAX or hd != head_dim or N1 != groups * rank or triton.next_power_of_2(hd) != hd
            or (hd - rope_dim) % 2 or wo_a.shape[1] != Kg or wo_b.shape[1] != N1
            or tuple(freqs.shape[1:]) != (rope_dim // 2, 2) or start.numel() * seqlen != M):
        raise ValueError(f"attn_out: o {tuple(o.shape)}, wo_a {tuple(wo_a.shape)}, wo_b {tuple(wo_b.shape)}, "
                         f"groups {groups} rank {rank}, rope {rope_dim}, freqs {tuple(freqs.shape)}, "
                         f"start {tuple(start.shape)} x {seqlen}")
    x = o.reshape(M * heads, hd).contiguous()
    oq = x if prepared else torch.empty_like(x)
    if M and not prepared:
        rowk, ctx = _launcher(_row_post_kernel, o.device, traced)  # not `post`: that name is the mix's weights
        with ctx:
            rowk[(M * heads,)](x, x, freqs.contiguous(), start.contiguous(), -1.0, oq, oq, M * heads, hd, heads,
                               seqlen, 0.0, NORM=False, RD=rope_dim, QUANT=True, STORE_PRE=False, BD=hd, KSUM=1)
    mid = gemv_rows(oq.view(M, heads * hd), wo_a, sa, group_rows=rank, traced=traced, quant_out=True)
    if residual is not None:
        HC = residual.shape[-2]
        out = gemv_rows(mid, wo_b, sb, traced=traced, hc=(residual.reshape(M, HC, dim), post.reshape(M, HC),
                                                          comb.reshape(M, HC, HC), None))
        return out.reshape(tuple(lead) + (HC, dim))
    y = gemv_rows(mid, wo_b, sb, traced=traced)
    return y.reshape(tuple(lead) + (dim,))


# ---------------------------------------------------------------------------
# a block's FFN front half as two kernels (ops.ffn_front): the mix dots, the
# stream's collapse, the norm, the quantized input and the gate's scores in
# the first (its programs share the gate's projection), the Sinkhorn
# iterations and the top-k choice in the second (one warp per row)

if triton is not None:
    @triton.jit
    def _ffn_front_a_kernel(stream_ptr, premix_ptr, hcfn_ptr, normw_ptr, gatew_ptr,
                            mixes_ptr, xn_ptr, xin_ptr, scores_ptr,
                            D, E, norm_eps,
                            HC: tl.constexpr, NMIX: tl.constexpr, P: tl.constexpr, EB: tl.constexpr,
                            BD: tl.constexpr, GATE: tl.constexpr):
        """Row r of the stream [rows, HC, D] (bf16), program p of P (without GATE: no scores, no
        quantized input; hc_front):
          mixes[r, j] = dot(hc_fn[j], stream[r].flatten().float()) * rsqrt(mean(stream[r]^2) + eps)
                        for j = p, p + P, ... (the fork's hc_mixes)
          x = sum_h pre_mix[r, h] * stream[r, h]          -> bf16 (hc_pre)
          xn = norm_w * x / sqrt(mean(x^2) + eps)          -> bf16 (RMSNorm; every program recomputes it)
          program 0 stores xn and x_in = act_quant(xn) (the experts' input, per 32, fp8 grid, bf16)
          scores[r, e] = dot(gate_w[e].float(), xn.float()) for the program's EB experts
        Passes over the row BD elements at a time: the norms' statistics first, then the mix dots
        this program owns (one pass each), then xn, x_in and the gate's scores."""
        norm_eps = norm_eps.to(tl.float32)  # inductor passes a Python float as fp64 (lower.py: Numerics)
        r = tl.program_id(0).to(tl.int64)
        p = tl.program_id(1)
        ds = tl.arange(0, BD)
        hs = tl.arange(0, HC)
        premix = tl.load(premix_ptr + r * HC + hs)  # [HC] fp32
        base = stream_ptr + r * HC * D
        sumsq_stream = 0.0
        sumsq_x = 0.0
        for d0 in range(0, D, BD):
            st = tl.load(base + hs[:, None] * D + (d0 + ds)[None, :]).to(tl.float32)  # [HC, BD]
            sumsq_stream += tl.sum(tl.sum(st * st, axis=1), axis=0)
            x = tl.sum(premix[:, None] * st, axis=0).to(tl.bfloat16).to(tl.float32)  # hc_pre, rounded to bf16
            sumsq_x += tl.sum(x * x, axis=0)
        rs_stream = 1.0 / tl.sqrt(sumsq_stream / (HC * D) + norm_eps)
        rs_x = 1.0 / tl.sqrt(sumsq_x / D + norm_eps)
        # the mix dots this program owns: hc_fn [NMIX, HC * D] in the flattened stream's order
        for j in range(p, NMIX, P):
            acc = tl.zeros((HC, BD), dtype=tl.float32)
            for d0 in range(0, D, BD):
                fn = tl.load(hcfn_ptr + j * (HC * D) + hs[:, None] * D + (d0 + ds)[None, :])
                st = tl.load(base + hs[:, None] * D + (d0 + ds)[None, :]).to(tl.float32)
                acc += fn * st
            tl.store(mixes_ptr + r * NMIX + j, tl.sum(tl.sum(acc, axis=1), axis=0) * rs_stream)
        es = p * EB + tl.arange(0, EB)
        el = es < E
        acc_scores = tl.zeros((EB,), dtype=tl.float32)
        for d0 in range(0, D, BD):
            st = tl.load(base + hs[:, None] * D + (d0 + ds)[None, :]).to(tl.float32)
            x = tl.sum(premix[:, None] * st, axis=0).to(tl.bfloat16).to(tl.float32)
            w = tl.load(normw_ptr + d0 + ds).to(tl.float32)
            xn = (w * (x * rs_x)).to(tl.bfloat16)  # the normalized row, bf16 as the fork rounds it
            if p == 0:
                tl.store(xn_ptr + r * D + d0 + ds, xn)
                xq = _quant_tile(tl.reshape(xn.to(tl.float32), (1, BD)), 1, BD)
                tl.store(xin_ptr + r * D + d0 + ds, tl.reshape(xq, (BD,)).to(tl.bfloat16))
            if GATE:
                gw = tl.load(gatew_ptr + es[:, None] * D + (d0 + ds)[None, :], mask=el[:, None], other=0.0).to(tl.float32)
                acc_scores += tl.sum(gw * xn.to(tl.float32)[None, :], axis=1)
        if GATE:
            tl.store(scores_ptr + r * E + es, acc_scores, mask=el)

    @triton.jit
    def _ffn_front_b_kernel(mixes_ptr, hcscale_ptr, hcbase_ptr, scores_ptr, bias_ptr,
                            pre_ptr, post_ptr, comb_ptr, w_ptr, i_ptr,
                            E, hc_eps, temp, route_scale, iters,
                            HC: tl.constexpr, NMIX: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                            BE: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, GATE: tl.constexpr,
                            NC: tl.constexpr = 1, NCP: tl.constexpr = 1):
        """Row r: the fork's hc_split_sinkhorn on its NMIX mixes (pre, post, and comb made doubly
        stochastic by `iters` Sinkhorn iterations), then Gate.forward's choice on its E raw scores
        (the scoring function, the K largest of score + bias one at a time, their unbiased
        scores normalized when NORM, times route_scale). One warp. The mixes and scores come as
        NC partial sums per row ([rows, NC, NMIX] and [rows, NC, E]: the split front's chunks of
        the row; NC = 1 for the one-pass kernel), summed here."""
        hc_eps = hc_eps.to(tl.float32)  # inductor passes Python floats as fp64 (lower.py: Numerics)
        temp = temp.to(tl.float32)
        route_scale = route_scale.to(tl.float32)
        r = tl.program_id(0).to(tl.int64)
        hs = tl.arange(0, HC)
        cs = tl.arange(0, NCP)  # the chunks, padded to a power of two
        cl = cs < NC
        mbase = mixes_ptr + r * NC * NMIX + cs[:, None] * NMIX
        m_pre = tl.sum(tl.load(mbase + hs[None, :], mask=cl[:, None], other=0.0), axis=0)
        m_post = tl.sum(tl.load(mbase + HC + hs[None, :], mask=cl[:, None], other=0.0), axis=0)
        m_comb = tl.sum(tl.load(mixes_ptr + r * NC * NMIX + cs[:, None, None] * NMIX + 2 * HC
                                + hs[None, :, None] * HC + hs[None, None, :], mask=cl[:, None, None], other=0.0), axis=0)
        s0 = tl.load(hcscale_ptr)
        s1 = tl.load(hcscale_ptr + 1)
        s2 = tl.load(hcscale_ptr + 2)
        pre = tl.sigmoid(m_pre * s0 + tl.load(hcbase_ptr + hs)) + hc_eps
        post = 2.0 * tl.sigmoid(m_post * s1 + tl.load(hcbase_ptr + HC + hs))
        c = m_comb * s2 + tl.load(hcbase_ptr + 2 * HC + hs[:, None] * HC + hs[None, :])
        c = c - tl.max(c, axis=1)[:, None]
        c = tl.exp(c)
        c = c / tl.sum(c, axis=1)[:, None] + hc_eps
        c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
        for _ in range(iters - 1):
            c = c / (tl.sum(c, axis=1)[:, None] + hc_eps)
            c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
        tl.store(pre_ptr + r * HC + hs, pre)
        tl.store(post_ptr + r * HC + hs, post)
        tl.store(comb_ptr + r * HC * HC + hs[:, None] * HC + hs[None, :], c)
        if not GATE:
            return
        # the gate's choice
        e = tl.arange(0, BE)
        live = e < E
        raw = tl.sum(tl.load(scores_ptr + r * NC * E + cs[:, None] * E + e[None, :],
                             mask=cl[:, None] & live[None, :], other=0.0), axis=0) / temp
        sc = _score(raw, live, SCORE)
        b = tl.load(bias_ptr + e, mask=live, other=0.0).to(tl.float32)
        cand = tl.where(live, sc + b, float("-inf"))
        ks = tl.arange(0, KP)
        chosen = tl.zeros((KP,), dtype=tl.int32)
        for k in tl.static_range(K):
            idx = tl.argmax(cand, axis=0)
            chosen = tl.where(ks == k, idx, chosen)
            cand = tl.where(e == idx, float("-inf"), cand)
        kin = ks < K
        picked_raw = tl.sum(tl.load(scores_ptr + r * NC * E + cs[:, None] * E + chosen[None, :],
                                    mask=cl[:, None] & kin[None, :], other=0.0), axis=0) / temp
        if SCORE == 2:
            mx = tl.max(tl.where(live, raw, float("-inf")), axis=0)
            picked = tl.where(kin, tl.exp(picked_raw - mx), 0.0) / tl.sum(tl.where(live, tl.exp(raw - mx), 0.0), axis=0)
        else:
            picked = tl.where(kin, _score(picked_raw, kin, SCORE), 0.0)
        if NORM:
            picked = picked / (tl.sum(picked, axis=0) + 1e-20)
        tl.store(w_ptr + r * K + ks, picked * route_scale, mask=kin)
        tl.store(i_ptr + r * K + ks, chosen.to(tl.int64), mask=kin)

    @triton.jit
    def _score(s, live, SCORE: tl.constexpr):
        """The gate's scoring function over a row's raw scores (SCORE 0: sqrt(softplus), 1: sigmoid,
        2: softmax)."""
        if SCORE == 2:
            m = tl.max(tl.where(live, s, float("-inf")), axis=0)
            p = tl.where(live, tl.exp(s - m), 0.0)
            return p / tl.sum(p, axis=0)
        elif SCORE == 1:
            return tl.sigmoid(s)
        else:  # torch's softplus: x itself above 20, else log1p(exp(x))
            return tl.sqrt(tl.where(s > 20.0, s, tl.extra.cuda.libdevice.log1p(tl.exp(s))))

    @triton.jit
    def _quant_tile(x, BM: tl.constexpr, BK: tl.constexpr):
        """x [BM, BK] fp32 through the fork's act_quant and back (per 32 along K: a power-of-two
        scale putting the block's amax at most at 448, the values rounded to fp8 e4m3), the
        arithmetic of _act_quant_dequant_kernel: the same values."""
        xb = tl.reshape(x, (BM, BK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(xb), axis=2), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(xb / scale[:, :, None], -448.0), 448.0)
        y = _round_e4m3(q) * scale[:, :, None]
        return tl.reshape(y.to(tl.bfloat16).to(tl.float32), (BM, BK))


if triton is not None:
    @triton.jit
    def _front_stats_kernel(stream_ptr, premix_ptr, x_ptr, part_ptr, D,
                            HC: tl.constexpr, NC: tl.constexpr, BD: tl.constexpr):
        """Chunk c of row r of the stream [rows, HC, D], one pass: x = sum_h pre_mix[r, h] *
        stream[r, h] rounded to bf16 (hc_pre), stored, and the chunk's sums of squares of the
        stream and of x (the two norms' statistics), stored as part[r, c] = (ss_stream, ss_x)."""
        r = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1)
        hs = tl.arange(0, HC)
        ds = c * BD + tl.arange(0, BD)
        st = tl.load(stream_ptr + r * HC * D + hs[:, None] * D + ds[None, :]).to(tl.float32)  # [HC, BD]
        premix = tl.load(premix_ptr + r * HC + hs)
        x = tl.sum(premix[:, None] * st, axis=0).to(tl.bfloat16)
        tl.store(x_ptr + r * D + ds, x)
        xf = x.to(tl.float32)
        tl.store(part_ptr + (r * NC + c) * 2, tl.sum(tl.sum(st * st, axis=1), axis=0))
        tl.store(part_ptr + (r * NC + c) * 2 + 1, tl.sum(xf * xf, axis=0))

    @triton.jit
    def _front_dots_kernel(stream_ptr, x_ptr, part_ptr, hcfn_ptr, normw_ptr, gatew_ptr,
                           mixp_ptr, xn_ptr, xin_ptr, scorep_ptr, D, E, norm_eps,
                           HC: tl.constexpr, NMIX: tl.constexpr, NC: tl.constexpr, NCP: tl.constexpr,
                           BD: tl.constexpr, G: tl.constexpr, EB: tl.constexpr, GATE: tl.constexpr):
        """Chunk c of row r, group g of G: from the row's statistics (the NC partials summed), the
        normalized chunk xn = norm_w * x * rsqrt(mean(x^2) + eps) in bf16 (group 0 stores it and,
        with GATE, x_in = act_quant(xn), the experts' input); the partial mix dots this group owns,
        mixp[r, c, j] = dot(hc_fn[j], stream[r]) over the chunk times the stream's rsqrt, for
        j = g, g + G, ...; and with GATE the partial gate scores of its EB experts,
        scorep[r, c, e] = dot(gate_w[e], xn) over the chunk. Every program reads about 20 KB and
        writes a few numbers: many short programs instead of a few long chains."""
        norm_eps = norm_eps.to(tl.float32)  # inductor passes a Python float as fp64 (lower.py: Numerics)
        r = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1)
        g = tl.program_id(2)
        hs = tl.arange(0, HC)
        ds = c * BD + tl.arange(0, BD)
        cs = tl.arange(0, NCP)  # the chunks, padded to a power of two
        part = tl.load(part_ptr + (r * NC + cs[:, None]) * 2 + tl.arange(0, 2)[None, :],
                       mask=(cs < NC)[:, None], other=0.0)  # [NCP, 2]
        ss = tl.sum(part, axis=0)
        ss_stream = tl.sum(tl.where(tl.arange(0, 2) == 0, ss, 0.0), axis=0)
        ss_x = tl.sum(tl.where(tl.arange(0, 2) == 1, ss, 0.0), axis=0)
        rs_stream = 1.0 / tl.sqrt(ss_stream / (HC * D) + norm_eps)
        rs_x = 1.0 / tl.sqrt(ss_x / D + norm_eps)
        x = tl.load(x_ptr + r * D + ds).to(tl.float32)
        w = tl.load(normw_ptr + ds).to(tl.float32)
        xn = (w * (x * rs_x)).to(tl.bfloat16)  # the normalized row, bf16 as the fork rounds it
        if g == 0:
            tl.store(xn_ptr + r * D + ds, xn)
            xq = _quant_tile(tl.reshape(xn.to(tl.float32), (1, BD)), 1, BD)
            tl.store(xin_ptr + r * D + ds, tl.reshape(xq, (BD,)).to(tl.bfloat16))
        st = tl.load(stream_ptr + r * HC * D + hs[:, None] * D + ds[None, :]).to(tl.float32)  # [HC, BD]
        for j in range(g, NMIX, G):
            fn = tl.load(hcfn_ptr + j * (HC * D) + hs[:, None] * D + ds[None, :])
            tl.store(mixp_ptr + (r * NC + c) * NMIX + j, tl.sum(tl.sum(fn * st, axis=1), axis=0) * rs_stream)
        if GATE:
            es = g * EB + tl.arange(0, EB)
            el = es < E
            gw = tl.load(gatew_ptr + es[:, None] * D + ds[None, :], mask=el[:, None], other=0.0).to(tl.float32)
            tl.store(scorep_ptr + (r * NC + c) * E + es, tl.sum(gw * xn.to(tl.float32)[None, :], axis=1), mask=el)


FFN_FRONT_EB = 8  # experts per program of the first kernel: E / EB programs a row (measured on an L4:
FFN_FRONT_BD = 1024  # 36 us a row against 60 at 16 x 256), and the elements of the row per pass


def hc_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
             traced=False):
    """ops.hc_front's outputs: ffn_front's kernels without the gate."""
    pre, post, comb, xn, x_in, _, _ = ffn_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters,
                                                hc_eps, norm_weight, norm_eps, None, None, 1, "", 1.0, False, 1.0,
                                                traced=traced)
    return pre, post, comb, xn, x_in


def ffn_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
              gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale, traced=False,
              eb=None, bd=None):
    """ops.ffn_front's outputs from the two kernels above. With gate_weight None (hc_front), the
    gate's outputs are empty and not computed."""
    eb = eb or FFN_FRONT_EB
    bd = bd or FFN_FRONT_BD
    while bd > 32 and stream.shape[-1] % bd:  # a narrow model (the tiny fixture): the widest chunk that divides
        bd //= 2
    b, s_, hc, d = stream.shape
    rows = b * s_
    gate = gate_weight is not None
    E = gate_weight.shape[0] if gate else 0
    nmix = (2 + hc) * hc
    if d % bd:
        raise ValueError(f"ffn_front: dim {d} is not a multiple of {bd}")
    P = max(triton.cdiv(E, eb), nmix)  # at least a program per mix dot: a model with few experts
                                      # (the fixtures) must not serialize them on two programs
    dev = stream.device
    mixes = torch.empty(rows, nmix, dtype=torch.float32, device=dev)
    xn = torch.empty(rows, d, dtype=stream.dtype, device=dev)
    x_in = torch.empty(rows, d, dtype=torch.bfloat16, device=dev)  # act_quant(xn): the next linears' input
    scores = torch.empty(rows, max(E, 1), dtype=torch.float32, device=dev)
    pre = torch.empty(rows, hc, dtype=torch.float32, device=dev)
    post = torch.empty(rows, hc, dtype=torch.float32, device=dev)
    comb = torch.empty(rows, hc, hc, dtype=torch.float32, device=dev)
    weights = torch.empty(rows, topk, dtype=torch.float32, device=dev)
    indices = torch.empty(rows, topk, dtype=torch.int64, device=dev)
    be = kp = 1
    while be < E:
        be *= 2
    while kp < topk:
        kp *= 2
    if rows:
        ka, ctx = _launcher(_ffn_front_a_kernel, dev, traced)
        with ctx:
            ka[(rows, P)](stream.contiguous(), pre_mix.contiguous().float(), hc_fn.contiguous().float(),
                          norm_weight.contiguous(), gate_weight.contiguous() if gate else xn, mixes, xn, x_in, scores,
                          d, E, float(norm_eps), HC=hc, NMIX=nmix, P=P, EB=eb, BD=bd, GATE=gate)
        kb, ctx = _launcher(_ffn_front_b_kernel, dev, traced)
        with ctx:
            kb[(rows,)](mixes, hc_scale.contiguous().float(), hc_base.contiguous().float(), scores,
                        gate_bias.contiguous() if gate else scores, pre, post, comb, weights, indices, E,
                        float(hc_eps), float(gate_temp), float(route_scale), int(sinkhorn_iters), HC=hc, NMIX=nmix,
                        K=topk, KP=kp, BE=be, SCORE={"softmax": 2, "sigmoid": 1}.get(score_func, 0),
                        NORM=bool(norm_topk_prob and topk > 1), GATE=gate, NC=1, NCP=1, num_warps=1)
    return (pre.view(b, s_, hc), post.view(b, s_, hc), comb.view(b, s_, hc, hc), xn.view(b, s_, d), x_in,
            weights, indices)


FRONT_SPLIT_BD = 512  # the chunk of the row per program of the split front's kernels
FRONT_SPLIT_EB = 8  # experts per program of its second kernel


def hc_front_split(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
                   traced=False):
    """ops.hc_front's outputs: ffn_front_split's kernels without the gate."""
    pre, post, comb, xn, x_in, _, _ = ffn_front_split(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult,
                                                      sinkhorn_iters, hc_eps, norm_weight, norm_eps, None, None, 1, "",
                                                      1.0, False, 1.0, traced=traced)
    return pre, post, comb, xn, x_in


def ffn_front_split(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
                    gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale, traced=False,
                    bd=None, eb=None):
    """ops.ffn_front's outputs from three kernels, the row split into chunks: the norms'
    statistics and x per chunk (_front_stats_kernel), then per chunk and group of experts the
    normalized row, the mix dots and the gate scores as partial sums (_front_dots_kernel), then
    the Sinkhorn iterations and the top-k choice per row over the summed partials
    (_ffn_front_b_kernel). The one-pass pair (ffn_front) walks the whole row per program in a
    dependent chain; here a program reads one chunk and the card runs hundreds of them at once."""
    bd = bd or FRONT_SPLIT_BD
    eb = eb or FRONT_SPLIT_EB
    b, s_, hc, d = stream.shape
    while bd > 32 and d % bd:  # a narrow model (the tiny fixture): the widest chunk that divides
        bd //= 2
    if d % bd:
        raise ValueError(f"ffn_front_split: dim {d} is not a multiple of {bd}")
    rows = b * s_
    gate = gate_weight is not None
    E = gate_weight.shape[0] if gate else 0
    nmix = (2 + hc) * hc
    nc = d // bd
    ncp = triton.next_power_of_2(nc)
    G = triton.cdiv(E, eb) if gate else nmix
    dev = stream.device
    xpre = torch.empty(rows, d, dtype=torch.bfloat16, device=dev)
    part = torch.empty(rows, nc, 2, dtype=torch.float32, device=dev)
    mixp = torch.empty(rows, nc, nmix, dtype=torch.float32, device=dev)
    xn = torch.empty(rows, d, dtype=stream.dtype, device=dev)
    x_in = torch.empty(rows, d, dtype=torch.bfloat16, device=dev)  # act_quant(xn): the next linears' input
    scorep = torch.empty(rows, nc, max(E, 1), dtype=torch.float32, device=dev)
    pre = torch.empty(rows, hc, dtype=torch.float32, device=dev)
    post = torch.empty(rows, hc, dtype=torch.float32, device=dev)
    comb = torch.empty(rows, hc, hc, dtype=torch.float32, device=dev)
    weights = torch.empty(rows, topk, dtype=torch.float32, device=dev)
    indices = torch.empty(rows, topk, dtype=torch.int64, device=dev)
    be = kp = 1
    while be < E:
        be *= 2
    while kp < topk:
        kp *= 2
    if rows:
        st = stream.contiguous()
        k1, ctx = _launcher(_front_stats_kernel, dev, traced)
        with ctx:
            k1[(rows, nc)](st, pre_mix.contiguous().float(), xpre, part, d, HC=hc, NC=nc, BD=bd)
        k2, ctx = _launcher(_front_dots_kernel, dev, traced)
        with ctx:
            k2[(rows, nc, G)](st, xpre, part, hc_fn.contiguous().float(), norm_weight.contiguous(),
                              gate_weight.contiguous() if gate else xn, mixp, xn, x_in, scorep, d, E, float(norm_eps),
                              HC=hc, NMIX=nmix, NC=nc, NCP=ncp, BD=bd, G=G, EB=eb, GATE=gate)
        kb, ctx = _launcher(_ffn_front_b_kernel, dev, traced)
        with ctx:
            kb[(rows,)](mixp, hc_scale.contiguous().float(), hc_base.contiguous().float(), scorep,
                        gate_bias.contiguous() if gate else scorep, pre, post, comb, weights, indices, E,
                        float(hc_eps), float(gate_temp), float(route_scale), int(sinkhorn_iters), HC=hc, NMIX=nmix,
                        K=topk, KP=kp, BE=be, SCORE={"softmax": 2, "sigmoid": 1}.get(score_func, 0),
                        NORM=bool(norm_topk_prob and topk > 1), GATE=gate, NC=nc, NCP=ncp, num_warps=1)
    return (pre.view(b, s_, hc), post.view(b, s_, hc), comb.view(b, s_, hc, hc), xn.view(b, s_, d), x_in,
            weights, indices)


# ---------------------------------------------------------------------------
# the sampler's two argmaxes (ops.choose_argmax): one pass over a row's logits

if triton is not None:
    @triton.jit
    def _choose_blocks_kernel(logits_ptr, temp_ptr, seed_ptr, pmax_ptr, pidx_ptr, smax_ptr, sidx_ptr, V, T, NB,
                              BLOCK: tl.constexpr):
        """Program (p, b): block b of position p's logits (row p // T): the block's largest logit
        and its index (the first on ties, as torch.argmax), and the largest of logit / temperature
        + Gumbel noise, -log(-log(u)) with u uniform per element (fresh each call: the seed tensor
        is stepped by the caller), and its index. The second kernel combines the blocks."""
        p = tl.program_id(0).to(tl.int64)
        b = tl.program_id(1)
        temp = tl.maximum(tl.load(temp_ptr + p // T), 1e-5)
        seed = tl.load(seed_ptr)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        live = offs < V
        x = tl.load(logits_ptr + p * V + offs, mask=live, other=float("-inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        i = tl.min(tl.where(x == m, offs, V), axis=0)
        u = tl.maximum(tl.rand(seed, offs + p * V), 1e-20)
        y = tl.where(live, x / temp - tl.log(-tl.log(u)), float("-inf"))
        sm = tl.max(y, axis=0)
        si = tl.min(tl.where(y == sm, offs, V), axis=0)
        tl.store(pmax_ptr + p * NB + b, m)
        tl.store(pidx_ptr + p * NB + b, i)
        tl.store(smax_ptr + p * NB + b, sm)
        tl.store(sidx_ptr + p * NB + b, si)

    @triton.jit
    def _choose_reduce_kernel(pmax_ptr, pidx_ptr, smax_ptr, sidx_ptr, greedy_ptr, full_ptr, NB, V,
                              NBP: tl.constexpr):
        """Position p: the block maxima combined, the smallest index among equal maxima."""
        p = tl.program_id(0).to(tl.int64)
        bs = tl.arange(0, NBP)
        live = bs < NB
        m = tl.load(pmax_ptr + p * NB + bs, mask=live, other=float("-inf"))
        i = tl.load(pidx_ptr + p * NB + bs, mask=live, other=V)
        best = tl.max(m, axis=0)
        tl.store(greedy_ptr + p, tl.min(tl.where(m == best, i, V), axis=0).to(tl.int64))
        sm = tl.load(smax_ptr + p * NB + bs, mask=live, other=float("-inf"))
        si = tl.load(sidx_ptr + p * NB + bs, mask=live, other=V)
        sbest = tl.max(sm, axis=0)
        tl.store(full_ptr + p, tl.min(tl.where(sm == sbest, si, V), axis=0).to(tl.int64))


def choose_argmax(logits, temperature, seed, traced=False):
    """ops.choose_argmax's outputs: greedy and full [rows, tokens] int64 for logits [rows, tokens,
    vocab], temperature [rows] fp32, seed a one-element int64 tensor on the device. Two kernels: the
    vocabulary in blocks of 2,048 across programs (a 129k vocabulary is 64 programs a position; one
    program a position serialized the noise and took 120 us), then the blocks combined."""
    rows, toks, V = logits.shape
    dev = logits.device
    greedy = torch.empty(rows, toks, dtype=torch.int64, device=dev)
    full = torch.empty_like(greedy)
    P = rows * toks
    if P:
        BLOCK = 2048
        NB = triton.cdiv(V, BLOCK)
        NBP = triton.next_power_of_2(NB)
        pmax = torch.empty(P, NB, dtype=torch.float32, device=dev)
        smax = torch.empty_like(pmax)
        pidx = torch.empty(P, NB, dtype=torch.int32, device=dev)
        sidx = torch.empty_like(pidx)
        k1, ctx = _launcher(_choose_blocks_kernel, dev, traced)
        with ctx:
            k1[(P, NB)](logits.contiguous(), temperature.contiguous().float(), seed, pmax, pidx, smax, sidx, V, toks,
                        NB, BLOCK=BLOCK, num_warps=4)
        k2, ctx = _launcher(_choose_reduce_kernel, dev, traced)
        with ctx:
            k2[(P,)](pmax, pidx, smax, sidx, greedy, full, NB, V, NBP=NBP, num_warps=1)
    return greedy, full


# ---------------------------------------------------------------------------
# sparse attention for decode (ops.sparse_attn): one kernel per (query, group
# of heads) gathers the selected KV rows, forms the scores, the softmax with
# the sink term and the output, in fp32 as the fork's aten version does

if triton is not None:
    @triton.jit
    def _sparse_attn_kernel(q_ptr, kv_ptr, idx_ptr, m_ptr, l_ptr, acc_ptr, kv2_ptr, start_ptr, M, H, T, N, S, scale,
                            T1, N2, WIN,
                            D: tl.constexpr, HB: tl.constexpr, TB: tl.constexpr, TWO: tl.constexpr):
        """Query row r (of b * M: row b's token m), heads hg * HB ..., and split sp of its T selected
        KV rows (idx, -1 for an absent slot; TB rows per split): scores = q . kv * scale, and the
        split's softmax partials: the max, the sum of exp, and sum exp * kv. On the tensor cores:
        the scores from bf16 inputs (exact products, fp32 sums), the output from the
        probabilities split into a bf16 high and low part (about 16 bits each), so the result
        differs from the fork's fp32 computation by summation order and that split. With TWO the
        KV comes from two caches and the selection is formed here, so neither the window's ids
        nor the concatenated KV is built before the kernel: entries 0 .. T1 - 1 are the window's
        T1 slots of kv_ptr [b, N = T1, D], entry k holding position newest - T1 + 1 + k with
        newest = start[b] + M - 1 the row's last position of the step (the fork's
        get_window_topk_idxs for a decode step: seen when at most the query's position, within
        WIN of it and not before the sequence), and entries T1 .. T - 1 the compressed cache
        kv2_ptr [b, N2, D] at idx[r, entry - T1] (-1 for none)."""
        scale = scale.to(tl.float32)  # inductor passes a Python float as fp64 (lower.py: Numerics)
        r = tl.program_id(0).to(tl.int64)
        hg = tl.program_id(1)
        sp = tl.program_id(2)
        b = r // M
        ds = tl.arange(0, D)
        hs = hg * HB + tl.arange(0, HB)
        hl = hs < H
        q = tl.load(q_ptr + (r * H + hs)[:, None] * D + ds[None, :], mask=hl[:, None], other=0.0).to(tl.bfloat16)
        ts = sp * TB + tl.arange(0, TB)
        if TWO:
            st = tl.load(start_ptr + b)
            p = st + r % M  # the query's position
            newest = st + M - 1
            held = newest - T1 + 1 + ts  # the window slot's position, oldest first
            win = ts < T1
            seen = win & (held <= p) & (held > p - WIN) & (held >= 0)
            ids1 = tl.where(seen, held % T1, -1)
            ids2 = tl.load(idx_ptr + r * (T - T1) + (ts - T1), mask=(ts >= T1) & (ts < T), other=-1)
            valid = tl.where(win, ids1 >= 0, ids2 >= 0)
            rows1 = tl.load(kv_ptr + (b * N + tl.maximum(ids1, 0).to(tl.int64))[:, None] * D + ds[None, :],
                            mask=seen[:, None], other=0.0)
            rows2 = tl.load(kv2_ptr + (b * N2 + tl.maximum(ids2, 0).to(tl.int64))[:, None] * D + ds[None, :],
                            mask=((~win) & (ids2 >= 0))[:, None], other=0.0)
            rows = tl.where(win[:, None], rows1, rows2).to(tl.bfloat16)  # [TB, D]
        else:
            ids = tl.load(idx_ptr + r * T + ts, mask=ts < T, other=-1)
            valid = ids >= 0
            rows = tl.load(kv_ptr + (b * N + tl.maximum(ids, 0).to(tl.int64))[:, None] * D + ds[None, :],
                           mask=valid[:, None], other=0.0).to(tl.bfloat16)  # [TB, D]
        sc = tl.dot(q, tl.trans(rows)) * scale  # [HB, TB] fp32
        sc = tl.where(valid[None, :], sc, float("-inf"))
        m_i = tl.max(sc, axis=1)
        m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
        pr = tl.exp(sc - m_safe[:, None])
        l_i = tl.sum(pr, axis=1)
        p_hi = pr.to(tl.bfloat16)
        p_lo = (pr - p_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(p_hi, rows) + tl.dot(p_lo, rows)  # [HB, D]
        part = (r * H + hs) * S + sp
        tl.store(m_ptr + part, m_i, mask=hl)
        tl.store(l_ptr + part, l_i, mask=hl)
        tl.store(acc_ptr + part[:, None] * D + ds[None, :], acc, mask=hl[:, None])

    @triton.jit
    def _sparse_attn_combine_kernel(m_ptr, l_ptr, acc_ptr, sink_ptr, o_ptr, freqs_ptr, start_ptr, H, S, M, SQ,
                                    D: tl.constexpr, SP: tl.constexpr, RD: tl.constexpr, FINISH: tl.constexpr):
        """Query row r, head h: the S splits' partials combined under the global max, the sink term
        added to the denominator (only there), o = acc / denominator, rounded to bf16 as the fork
        returns it. A query with no valid slot gives zeros. With FINISH the output path's first
        step follows in the same program (the fork's apply_rotary_emb inverse and act_quant, as
        _row_post_kernel does them): the head's last RD elements rotated back by the frequencies
        at the token's position (freqs_ptr the table's real view, start_ptr[row // SQ] + row % SQ
        with row = r // M... the query rows run b-major, M a row's queries, SQ the step's length)
        and the head quantized to the fp8 grid per 32, so the grouped projection reads it as is."""
        r = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        ss = tl.arange(0, SP)
        sl = ss < S
        part = (r * H + h) * S + ss
        m_s = tl.load(m_ptr + part, mask=sl, other=float("-inf"))
        m = tl.max(m_s, axis=0)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        w = tl.where(sl, tl.exp(m_s - m_safe), 0.0)
        l = tl.sum(w * tl.load(l_ptr + part, mask=sl, other=0.0), axis=0)
        ds = tl.arange(0, D)
        acc = tl.sum(w[:, None] * tl.load(acc_ptr + part[:, None] * D + ds[None, :], mask=sl[:, None], other=0.0), axis=0)
        denom = l + tl.exp(tl.load(sink_ptr + h).to(tl.float32) - m_safe)
        o = acc / denom
        if FINISH:
            o = o.to(tl.bfloat16).to(tl.float32)  # the attention's bf16 output, as the fork rotates it
            ev, od = tl.split(tl.reshape(o, (D // 2, 2)))
            even_dim = 2 * tl.arange(0, D // 2)
            tail = even_dim >= D - RD
            pair = (even_dim - (D - RD)) // 2
            tok = r  # query rows run b-major: row b = r // M, token t = r % M
            pos = tl.load(start_ptr + tok // M) + tok % M
            fb = freqs_ptr + pos * RD + 2 * pair
            c = tl.load(fb, mask=tail, other=1.0)
            sn = -tl.load(fb + 1, mask=tail, other=0.0)  # the inverse rotation
            o = tl.reshape(tl.join(ev * c - od * sn, ev * sn + od * c), (D,))
            o = o.to(tl.bfloat16).to(tl.float32)  # the fork writes the rotated tail back as bf16
            og = tl.reshape(o, (D // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(og), axis=1), 1e-4)
            v = amax / 448.0
            bits = v.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) & 0xFF
            exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(og / scale[:, None], -448.0), 448.0)
            o = tl.reshape(_round_e4m3(q) * scale[:, None], (D,))
        tl.store(o_ptr + (r * H + h) * D + ds, o.to(o_ptr.dtype.element_ty))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, freqs=None, start=None, seqlen=1, rope_dim=0,
                finish=False, kv2=None, window=0, traced=False):
    """The fork's sparse_attn: q [b, m, h, d], kv [b, n, d] (K and V the same rows), attn_sink [h],
    topk_idxs [b, m, t] int32 with -1 for an absent slot -> o [b, m, h, d] in q's dtype. With
    finish, the output path's first step runs in the combine kernel: each head's last rope_dim
    elements rotated back by the frequencies at the token's position (freqs the table's real
    view [positions, rope_dim / 2, 2], start [b] int64, seqlen the step's length: query token t
    of row b at start[b] + t) and the head quantized to the fp8 grid per 32, so attn_out takes
    the result as prepared (one launch fewer a layer). With kv2 [b, n2, d] (the compressed KV)
    and window (the attention's window size), kv is the window ring [b, slots, d] and
    topk_idxs the compressed selection only: the kernel forms the window's selection from the
    positions and reads both caches, so the model builds neither the window's ids nor the
    concatenated KV (two launches fewer a layer); a decode step's form (start, seqlen as above)."""
    b, m, h, d = q.shape
    if finish and (freqs is None or start is None or rope_dim % 2 or tuple(freqs.shape[1:]) != (rope_dim // 2, 2)
                   or start.numel() * seqlen != b * m or d % 32):
        raise ValueError(f"sparse_attn: finish needs the table and positions: freqs "
                         f"{None if freqs is None else tuple(freqs.shape)}, start {None if start is None else tuple(start.shape)} "
                         f"x {seqlen} for {b} x {m} queries, rope {rope_dim}, d {d}")
    n = kv.shape[1]
    two = kv2 is not None
    t1 = n if two else 0  # the window's slots come first
    n2 = kv2.shape[1] if two else 0
    t = t1 + topk_idxs.shape[-1]
    if kv.shape[-1] != d or (d & (d - 1)) or (two and (kv2.shape[-1] != d or start is None or seqlen != m
                                                        or start.numel() != b)):
        raise ValueError(f"sparse_attn: q {tuple(q.shape)} against kv {tuple(kv.shape)}"
                         f"{'' if not two else f' and kv2 {tuple(kv2.shape)}, start {tuple(start.shape)} x {seqlen}'}; "
                         f"d must be a power of two")
    o = torch.empty_like(q)
    if q.numel():
        HB, TB = (16 if h >= 16 else h), 32  # 64-row tiles overflow shared memory at d = 512
        S = triton.cdiv(t, TB)
        sp = 1
        while sp < S:
            sp *= 2
        dev = q.device
        m_part = torch.empty(b * m * h * S, dtype=torch.float32, device=dev)
        l_part = torch.empty_like(m_part)
        acc_part = torch.empty(b * m * h * S, d, dtype=torch.float32, device=dev)
        kernel, ctx = _launcher(_sparse_attn_kernel, dev, traced)
        with ctx:
            kernel[(b * m, triton.cdiv(h, HB), S)](q.contiguous(), kv.contiguous(), topk_idxs.contiguous(), m_part,
                                                  l_part, acc_part, kv2.contiguous() if two else kv,
                                                  start.contiguous() if two else topk_idxs, m, h, t, n, S,
                                                  float(softmax_scale), t1, n2, int(window),
                                                  D=d, HB=HB, TB=TB, TWO=two, num_warps=4)
        combine, ctx = _launcher(_sparse_attn_combine_kernel, dev, traced)
        fr = freqs.contiguous() if finish else o
        st = start.contiguous() if finish else topk_idxs
        with ctx:
            combine[(b * m, h)](m_part, l_part, acc_part, attn_sink.contiguous(), o, fr, st, h, S, m, seqlen, D=d,
                                SP=sp, RD=rope_dim if finish else 2, FINISH=finish, num_warps=4)
    return o


def sparse_attn_finish(q, kv, attn_sink, topk_idxs, softmax_scale, freqs, start, seqlen, rope_dim, traced=False):
    """sparse_attn with finish (ops.attn_core): the output rotated back and quantized per head."""
    return sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, freqs, start, seqlen, rope_dim, True, traced=traced)


def attn_select(q, kv_window, kv_compress, attn_sink, compress_idxs, softmax_scale, freqs, start, seqlen, rope_dim,
                window, traced=False):
    """ops.attn_select's Triton body: sparse_attn over the window ring and the compressed cache with
    the window's selection formed in the kernel, finishing the output (see sparse_attn)."""
    return sparse_attn(q, kv_window, attn_sink, compress_idxs, softmax_scale, freqs, start, seqlen, rope_dim, True,
                       kv2=kv_compress, window=window, traced=traced)


def _gemv_blocks(K):
    """Block shape for fp8_gemv: 8 rows per program and the widest K step
    that divides K, measured best or within 3% of best for every
    DeepSeek-V4.1-Flash dense shape (inside a CUDA graph, cache-cold
    weights); a 1024 step lost at K = 8192."""
    for bk in (1024, 512, 256, 128):
        if K % bk == 0 and not (bk == 1024 and K >= 8192):
            return 8, bk
    return 8, 32


def fp8_gemv(x, weight, scale, group_rows=None, out_dtype=torch.bfloat16, block_n=None, block_k=None, traced=False):
    """x [M, G*K] times weight [N, K] (fp8 e4m3 with [N/32, K/32] e8m0
    block scales) transposed -> [M, N] in out_dtype, fp32 accumulation.
    With group_rows R, rows n read x[:, g*K:(g+1)*K] for g = n // R (a
    block-diagonal weight, G = N / R groups); otherwise G = 1. For a
    decode step's rows: up to GEMV_ROWS_MAX bf16 rows, one pass over the
    weight serves them all (gemv_rows); otherwise each row reads it."""
    M = x.shape[0]
    N, K = weight.shape
    R = group_rows or N
    if x.dtype == torch.bfloat16 and M <= GEMM_ROWS_MAX and block_n is None and block_k is None:
        return gemv_rows(x, weight, scale, group_rows, out_dtype, traced=traced)
    if block_n is None or block_k is None:
        block_n, block_k = _gemv_blocks(K)
    if N % R or R % block_n or x.shape[1] != (N // R) * K or K % 32 or N % 32 or block_k % 32:
        raise ValueError(f"fp8_gemv: x {tuple(x.shape)}, weight {tuple(weight.shape)}, group_rows {group_rows}")
    # the kernel reads the scales as one byte per 32x32 block
    if (weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu
            or tuple(scale.shape) != (N // 32, K // 32)):
        raise ValueError(f"fp8_gemv: weight {weight.dtype} {tuple(weight.shape)} needs e8m0 scales "
                         f"[{N // 32}, {K // 32}], got {scale.dtype} {tuple(scale.shape)}")
    y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    if M and N:
        kernel, ctx = _launcher(_fp8_gemv_kernel, x.device, traced)
        with ctx:
            kernel[(triton.cdiv(N, block_n), M)](
                x.contiguous(), weight.contiguous(), scale.contiguous().view(torch.uint8), y, N, K,
                x.shape[1], R, scale.shape[1], BN=block_n, BK=block_k)
    return y


if triton is not None:
    @triton.jit
    def _rotary_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                       d, H, S,
                       BD: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        D = d // 2
        s_idx = (pid // H) % S
        cols = tl.arange(0, BD)
        mask = cols < D

        off_re = pid * d + cols * 2
        off_im = pid * d + cols * 2 + 1
        x_re = tl.load(x_ptr + off_re, mask=mask).to(tl.float32)
        x_im = tl.load(x_ptr + off_im, mask=mask).to(tl.float32)

        off_cs = s_idx * D + cols
        c = tl.load(cos_ptr + off_cs, mask=mask).to(tl.float32)
        s = tl.load(sin_ptr + off_cs, mask=mask).to(tl.float32)

        out_re = x_re * c - x_im * s
        out_im = x_re * s + x_im * c

        tl.store(y_ptr + off_re, out_re.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(y_ptr + off_im, out_im.to(y_ptr.dtype.element_ty), mask=mask)


def rotary(x, cos, sin, traced=False):
    """Rotate element pairs in x (last dim d) by cos and sin using a single
    Triton kernel: [s, d/2], shared by every row, or [b, s, d/2], per row."""
    assert x.is_cuda and x.shape[-1] % 2 == 0
    if x.ndim not in (3, 4):
        raise ValueError(f"rotary: x must have rank 3 [b, s, d] or 4 [b, s, h, d], got {x.ndim} {tuple(x.shape)}")
    d = x.shape[-1]
    D = d // 2
    S = x.shape[1]
    shared, per_row = (S, D), (x.shape[0], S, D)
    if cos.shape not in (shared, per_row) or sin.shape != cos.shape:
        raise ValueError(f"rotary: x {tuple(x.shape)} needs cos and sin {list(shared)} or {list(per_row)}, "
                         f"got cos {tuple(cos.shape)} and sin {tuple(sin.shape)}")
    # the kernel picks frequency row (row // H) % S: with per-row frequencies there are b * s of
    # them, in the same order as x's rows, so the modulo leaves each row its own
    S = cos.numel() // D
    H = x.shape[2] if x.ndim == 4 else 1
    n_rows = x.numel() // d
    xc = x.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    y = torch.empty(xc.shape, dtype=x.dtype, device=x.device)
    if n_rows and D:
        BD = triton.next_power_of_2(D)
        kernel, ctx = _launcher(_rotary_kernel, x.device, traced)
        with ctx:
            kernel[(n_rows,)](xc, cos_c, sin_c, y, d, H, S, BD=BD)
    return y


# ---------------------------------------------------------------------------
# the traced forms, as torch ops (the plan's lowering calls these)

if triton is not None:
    @torch.library.triton_op("vxpu_triton::act_quant_dequant", mutates_args=())
    def _act_quant_dequant_op(x: torch.Tensor) -> torch.Tensor:
        return act_quant_dequant(x, traced=True)

    @torch.library.triton_op("vxpu_triton::hc_split_sinkhorn", mutates_args=())
    def _hc_split_sinkhorn_op(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int,
                              sinkhorn_iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, traced=True)

    @torch.library.triton_op("vxpu_triton::linear_f32", mutates_args=())
    def _linear_f32_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return linear_f32(x, weight, traced=True)

    @torch.library.triton_op("vxpu_triton::fp8_gemv", mutates_args=())
    def _fp8_gemv_op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, group_rows: int,
                     out_dtype: torch.dtype) -> torch.Tensor:
        return fp8_gemv(x, weight, scale, group_rows or None, out_dtype, traced=True)

    @torch.library.triton_op("vxpu_triton::ffn_front", mutates_args=())
    def _ffn_front_op(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                      hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float,
                      norm_weight: torch.Tensor, norm_eps: float, gate_weight: torch.Tensor, gate_bias: torch.Tensor,
                      topk: int, score_func: str, gate_temp: float, norm_topk_prob: bool, route_scale: float
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor]:
        return ffn_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                         norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale,
                         traced=True)

    @torch.library.triton_op("vxpu_triton::hc_front", mutates_args=())
    def _hc_front_op(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                     hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_weight: torch.Tensor,
                     norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                        norm_eps, traced=True)

    @torch.library.triton_op("vxpu_triton::ffn_front_split", mutates_args=())
    def _ffn_front_split_op(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                            hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float,
                            norm_weight: torch.Tensor, norm_eps: float, gate_weight: torch.Tensor,
                            gate_bias: torch.Tensor, topk: int, score_func: str, gate_temp: float,
                            norm_topk_prob: bool, route_scale: float
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]:
        return ffn_front_split(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                               norm_weight, norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp,
                               norm_topk_prob, route_scale, traced=True)

    @torch.library.triton_op("vxpu_triton::hc_front_split", mutates_args=())
    def _hc_front_split_op(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                           hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float,
                           norm_weight: torch.Tensor, norm_eps: float
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_front_split(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                              norm_weight, norm_eps, traced=True)

    @torch.library.triton_op("vxpu_triton::shared_ffn", mutates_args=())
    def _shared_ffn_op(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor,
                       w2: torch.Tensor, s2: torch.Tensor, swiglu_limit: float, quantized: bool,
                       y_experts: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
                       ) -> torch.Tensor:
        return shared_ffn(x, w1, s1, w3, s3, w2, s2, swiglu_limit, quantized, y_experts, residual, post, comb,
                          traced=True)

    @torch.library.triton_op("vxpu_triton::attn_q", mutates_args=())
    def _attn_q_op(x: torch.Tensor, wq_a: torch.Tensor, sa: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
                   wq_b: torch.Tensor, sb: torch.Tensor, freqs: torch.Tensor, start: torch.Tensor, seqlen: int,
                   heads: int, head_dim: int, rope_dim: int, quantized: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return attn_q(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs, start, seqlen, heads, head_dim, rope_dim,
                      quantized, traced=True)

    @torch.library.triton_op("vxpu_triton::window_kv", mutates_args=())
    def _window_kv_op(x: torch.Tensor, wkv: torch.Tensor, scale: torch.Tensor, norm_weight: torch.Tensor,
                      norm_eps: float, freqs: torch.Tensor, start: torch.Tensor, seqlen: int, rope_dim: int,
                      quantized: bool, counters: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        return window_kv(x, wkv, scale, norm_weight, norm_eps, freqs, start, seqlen, rope_dim, quantized, traced=True,
                         counters=counters)

    @torch.library.triton_op("vxpu_triton::attn_out", mutates_args=())
    def _attn_out_op(o: torch.Tensor, freqs: torch.Tensor, start: torch.Tensor, seqlen: int, wo_a: torch.Tensor,
                     sa: torch.Tensor, groups: int, rank: int, wo_b: torch.Tensor, sb: torch.Tensor, head_dim: int,
                     rope_dim: int, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, prepared: bool
                     ) -> torch.Tensor:
        return attn_out(o, freqs, start, seqlen, wo_a, sa, groups, rank, wo_b, sb, head_dim, rope_dim, residual, post,
                        comb, prepared, traced=True)

    @torch.library.triton_op("vxpu_triton::sparse_attn", mutates_args=())
    def _sparse_attn_op(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                        softmax_scale: float) -> torch.Tensor:
        return sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, traced=True)

    @torch.library.triton_op("vxpu_triton::attn_select", mutates_args=())
    def _attn_select_op(q: torch.Tensor, kv_window: torch.Tensor, kv_compress: torch.Tensor, attn_sink: torch.Tensor,
                        compress_idxs: torch.Tensor, softmax_scale: float, freqs: torch.Tensor, start: torch.Tensor,
                        seqlen: int, rope_dim: int, window: int) -> torch.Tensor:
        return attn_select(q, kv_window, kv_compress, attn_sink, compress_idxs, softmax_scale, freqs, start, seqlen,
                           rope_dim, window, traced=True)

    @torch.library.triton_op("vxpu_triton::sparse_attn_finish", mutates_args=())
    def _sparse_attn_finish_op(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                               softmax_scale: float, freqs: torch.Tensor, start: torch.Tensor, seqlen: int,
                               rope_dim: int) -> torch.Tensor:
        return sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, freqs, start, seqlen, rope_dim, True,
                           traced=True)

    @torch.library.triton_op("vxpu_triton::rotary", mutates_args=())
    def _rotary_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return rotary(x, cos, sin, traced=True)


def _lowered_fp8_gemv(x, weight, scale, group_rows=None, out_dtype=torch.bfloat16):
    return torch.ops.vxpu_triton.fp8_gemv(x, weight, scale, group_rows or 0, out_dtype)


# this module's functions under the same names, through the vxpu_triton ops
lowered = types.SimpleNamespace(
    available=available,
    hc_split_sinkhorn_supported=hc_split_sinkhorn_supported,
    act_quant_dequant=lambda x: torch.ops.vxpu_triton.act_quant_dequant(x),
    hc_split_sinkhorn=lambda *a: torch.ops.vxpu_triton.hc_split_sinkhorn(*a),
    linear_f32=lambda x, weight: torch.ops.vxpu_triton.linear_f32(x, weight),
    fp8_gemv=_lowered_fp8_gemv,
    rotary=lambda x, cos, sin: torch.ops.vxpu_triton.rotary(x, cos, sin),
    shared_ffn=lambda *a: torch.ops.vxpu_triton.shared_ffn(*a),
    attn_q=lambda *a: torch.ops.vxpu_triton.attn_q(*a),
    window_kv=lambda *a, counters=None: torch.ops.vxpu_triton.window_kv(*a, counters),
    attn_out=lambda *a: torch.ops.vxpu_triton.attn_out(*a),
    sparse_attn_finish=lambda *a: torch.ops.vxpu_triton.sparse_attn_finish(*a),
    attn_select=lambda *a: torch.ops.vxpu_triton.attn_select(*a),
    ffn_front=lambda *a: torch.ops.vxpu_triton.ffn_front(*a),
    hc_front=lambda *a: torch.ops.vxpu_triton.hc_front(*a),
    ffn_front_split=lambda *a: torch.ops.vxpu_triton.ffn_front_split(*a),
    hc_front_split=lambda *a: torch.ops.vxpu_triton.hc_front_split(*a),
    sparse_attn=lambda *a: torch.ops.vxpu_triton.sparse_attn(*a),
)


# ---------------------------------------------------------------------------
# routed experts on the GPU (the hot ones; experts_gpu.py)

if triton is not None:
    @triton.jit
    def _swiglu_quant_kernel(g_ptr, u_ptr, wt_ptr, slot_ptr, h_ptr, I, limit, G: tl.constexpr):
        """The fork's _ffn for one pair's 32-blocks: clamps, w * silu(g) * u
        in fp32, rounded to bf16, then act_quant per 32 (fp8 grid, ue8m0
        scales) and dequantized, as _act_quant_dequant_kernel does."""
        p = tl.program_id(1)
        if tl.load(slot_ptr + p) < 0:  # not held here
            return
        blocks = tl.program_id(0) * G + tl.arange(0, G)
        live = blocks < I // 32
        offs = p.to(tl.int64) * I + blocks[:, None] * 32 + tl.arange(0, 32)[None, :]
        g = tl.load(g_ptr + offs, mask=live[:, None], other=0.0).to(tl.bfloat16).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=live[:, None], other=0.0).to(tl.bfloat16).to(tl.float32)
        if limit > 0:
            u = tl.minimum(tl.maximum(u, -limit), limit)
            g = tl.minimum(g, limit)
        w = tl.load(wt_ptr + p).to(tl.float32)
        h = w * ((g * tl.sigmoid(g)) * u)
        h = h.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(h), axis=1), 1e-4)
        v = amax / 448.0
        bits = v.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(h / scale[:, None], -448.0), 448.0)
        y = _round_e4m3(q) * scale[:, None]
        tl.store(h_ptr + offs, y.to(tl.bfloat16), mask=live[:, None])


if triton is not None:
    @triton.jit
    def _e2m1_bf16_bits(v, e):
        """fp4 e2m1 codes (0..15, int32) times 2^(e - 127) (e8m0 scales, int32, broadcast), as bf16
        bits in int32, by integer operations alone. A code's exponent field f (2 bits) and mantissa
        bit m give 2^(f-1) * (1 + m/2) for f > 0, so bf16 exponent f + 126 and top mantissa bit m;
        f = 0 is 0 or 0.5 (exponent 126, mantissa 0). The scale adds e - 127 to the exponent. Exact
        while the result stays a normal bf16 (scales of real checkpoints keep it there)."""
        f = (v >> 1) & 3
        m = v & 1
        sub = f == 0
        exp = tl.where(sub, 126, f + 126) + e - 127
        mant = tl.where(sub, 0, m << 6)
        mag = (exp << 7) | mant
        mag = tl.where((v & 7) == 0, 0, mag)  # +-0
        return ((v & 8) << 12) | mag

    @triton.jit
    def _fp4_tile(w_ptr, s_ptr, rows, live, k0, KB, KS, BN: tl.constexpr, BK: tl.constexpr):
        """Rows of an fp4 bank ([S*N, K/2] bytes, e8m0 scales [S*N, K/32]), columns k0..k0+BK, decoded
        to bf16 [BN, BK]: exact, since an fp4 value times a power of two fits bf16."""
        kb = k0 // 2 + tl.arange(0, BK // 2)
        b = tl.load(w_ptr + rows[:, None] * KB + kb[None, :], mask=live[:, None], other=0).to(tl.int32)
        ks = k0 // 32 + tl.arange(0, BK // 32)
        e = tl.load(s_ptr + rows[:, None] * KS + ks[None, :], mask=live[:, None], other=127).to(tl.int32)
        e = tl.reshape(tl.broadcast_to(e[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2))  # each byte's scale
        bits = tl.join(_e2m1_bf16_bits(b & 15, e), _e2m1_bf16_bits(b >> 4, e))  # low nibble first
        return tl.reshape(bits, (BN, BK)).to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _fp4_raw(w_ptr, s_ptr, rows, live, k0, KB, KS, BN: tl.constexpr, BK: tl.constexpr):
        """The same rows and columns as _fp4_tile, undecoded: the packed bytes [BN, BK/2] and the
        e8m0 scales [BN, BK/32], the operands of a scaled dot on a tensor core that takes MXFP4."""
        kb = k0 // 2 + tl.arange(0, BK // 2)
        b = tl.load(w_ptr + rows[:, None] * KB + kb[None, :], mask=live[:, None], other=0)
        ks = k0 // 32 + tl.arange(0, BK // 32)
        e = tl.load(s_ptr + rows[:, None] * KS + ks[None, :], mask=live[:, None], other=127)
        return b, e

    @triton.jit
    def _fp4_scales(s_ptr, rows, live, k0, KS, BN: tl.constexpr, BK: tl.constexpr):
        """The e8m0 scales [BN, BK/32] of _fp4_raw alone, for a kernel whose packed bytes arrive
        through a tensor descriptor (TMA): a scale tile of 8 bytes a row is below the descriptor's
        16-byte minimum at BK 256, and plain loads of it cost nothing measurable."""
        ks = k0 // 32 + tl.arange(0, BK // 32)
        return tl.load(s_ptr + rows[:, None] * KS + ks[None, :], mask=live[:, None], other=127)

    @triton.jit
    def _grouped_block(order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr, b, BM: tl.constexpr):
        """Block b of the schedule: its expert's slot, its pairs (up to BM, in the sorted order) and
        which of them are real."""
        cnt = tl.load(blk_count_ptr + b)
        m = tl.arange(0, BM)
        live = m < cnt
        pairs = tl.load(order_ptr + tl.load(blk_first_ptr + b) + m, mask=live, other=0)
        return tl.load(blk_slot_ptr + b).to(tl.int64), pairs, live, cnt

    @triton.jit
    def _fp4_grouped_gate_up_kernel(x_ptr, order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr,
                                    w1_ptr, s1_ptr, w3_ptr, s3_ptr, g_ptr, u_ptr, I, K, TOPK,
                                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr,
                                    NATIVE: tl.constexpr):
        """Gate and up for one block of pairs sharing an expert, BN of its rows, the products on the
        tensor cores (fp32 sums). NATIVE: the packed fp4 and the e8m0 scales go to the tensor core
        as they are (MXFP4, Blackwell and later); otherwise each weight tile is decoded to bf16 in
        software once for all the block's tokens (the decode, not the bytes, bounds that body: 666
        against 1,220 GB/s native at one token on an RTX PRO 6000). With KS > 1 the reduction over K is split across KS programs (a decode step's few pairs
        leave too few programs otherwise: at one token, six experts and BN 32 gave 432 programs
        on 188 SMs and 45% of the card's bandwidth); each adds its partial sums into fp32 g and u,
        zeroed by the caller, and the SwiGLU kernel rounds them to bf16 as the module does."""
        slot, pairs, live_m, cnt = _grouped_block(order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr,
                                                  tl.program_id(0), BM)
        if cnt <= 0:
            return
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        live_n = rows < I
        bank_rows = slot * I + rows
        tok = (pairs // TOPK).to(tl.int64)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        span = K // KS
        k_start = tl.program_id(2) * span
        for k0 in range(k_start, k_start + span, BK):
            x = tl.load(x_ptr + tok[:, None] * K + k0 + tl.arange(0, BK)[None, :], mask=live_m[:, None], other=0.0)
            if NATIVE:
                b1, e1 = _fp4_raw(w1_ptr, s1_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)
                b3, e3 = _fp4_raw(w3_ptr, s3_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)
                acc_g = tl.dot_scaled(x, None, "bf16", tl.trans(b1), e1, "e2m1", acc_g)
                acc_u = tl.dot_scaled(x, None, "bf16", tl.trans(b3), e3, "e2m1", acc_u)
            else:
                acc_g = tl.dot(x, tl.trans(_fp4_tile(w1_ptr, s1_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)), acc_g)
                acc_u = tl.dot(x, tl.trans(_fp4_tile(w3_ptr, s3_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)), acc_u)
        out = pairs.to(tl.int64)[:, None] * I + rows[None, :]
        mask = live_m[:, None] & live_n[None, :]
        if KS == 1:
            tl.store(g_ptr + out, acc_g.to(g_ptr.dtype.element_ty), mask=mask)
            tl.store(u_ptr + out, acc_u.to(u_ptr.dtype.element_ty), mask=mask)
        else:
            tl.atomic_add(g_ptr + out, acc_g, mask=mask)
            tl.atomic_add(u_ptr + out, acc_u, mask=mask)

    @triton.jit
    def _fp4_grouped_down_kernel(h_ptr, order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr, w2_ptr, s2_ptr,
                                 d_ptr, D, I, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr,
                                 NATIVE: tl.constexpr):
        """The down projection, the two bodies of the gate/up kernel; with KS > 1 the reduction over I is split across KS programs, each
        adding its partial sums into d (fp32, zeroed by the caller). The module rounds a pair's
        reply to bf16; the split keeps it in fp32 (a rounding-level difference)."""
        slot, pairs, live_m, cnt = _grouped_block(order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr,
                                                  tl.program_id(0), BM)
        if cnt <= 0:
            return
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        live_n = rows < D
        bank_rows = slot * D + rows
        prow = pairs.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        span = I // KS
        k_start = tl.program_id(2) * span
        for k0 in range(k_start, k_start + span, BK):
            h = tl.load(h_ptr + prow[:, None] * I + k0 + tl.arange(0, BK)[None, :], mask=live_m[:, None], other=0.0)
            if NATIVE:
                b2, e2 = _fp4_raw(w2_ptr, s2_ptr, bank_rows, live_n, k0, I // 2, I // 32, BN, BK)
                acc = tl.dot_scaled(h, None, "bf16", tl.trans(b2), e2, "e2m1", acc)
            else:
                acc = tl.dot(h, tl.trans(_fp4_tile(w2_ptr, s2_ptr, bank_rows, live_n, k0, I // 2, I // 32, BN, BK)), acc)
        mask = live_m[:, None] & live_n[None, :]
        if KS == 1:
            tl.store(d_ptr + prow[:, None] * D + rows[None, :], acc.to(tl.bfloat16).to(tl.float32), mask=mask)
        else:
            tl.atomic_add(d_ptr + prow[:, None] * D + rows[None, :], acc, mask=mask)


if triton is not None:
    @triton.jit
    def _fp4_direct_one(x_ptr, w_ptr, s_ptr, o_ptr, p, tok, bank_rows, rows, live_n, I, K, NROWS,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr,
                        TMA: tl.constexpr):
        """One pair's product with BN rows of one fp4 matrix (gate or up), stored as bf16. With TMA
        the packed bytes stream through a tensor descriptor over the bank's rows ([NROWS, K/2]),
        num_stages tiles in flight in shared memory, and the scales come by plain loads: measured
        cold at one token on an RTX PRO 6000, 60.2 us for 75 MB against 75.6 with plain loads of
        both (the loads alone reach 1,400 GB/s either way; the descriptor's pipeline keeps the
        one-live-row dot from stalling them)."""
        live_m = tl.arange(0, BM) == 0
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if TMA:  # w_ptr is the bank's descriptor (made on the host, _fp4_descriptor)
            base = tl.min(bank_rows, axis=0).to(tl.int32)
        for k0 in range(0, K, BK):
            # x's one live row, loaded once and broadcast in registers: a [BM, BK] masked load of
            # it was staged per pipeline stage (8 KB a stage at BK 256, the weight tile's twice),
            # which held the resident programs to four an SM and made a third stage a loss
            xr = tl.load(x_ptr + tok * K + k0 + tl.arange(0, BK))
            x = tl.where(live_m[:, None], xr[None, :], 0.0).to(tl.bfloat16)
            if TMA:
                b = w_ptr.load([base, k0 // 2])
                e = _fp4_scales(s_ptr, bank_rows, live_n, k0, K // 32, BN, BK)
                acc = tl.dot_scaled(x, None, "bf16", tl.trans(b), e, "e2m1", acc)
            elif NATIVE:
                b, e = _fp4_raw(w_ptr, s_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)
                acc = tl.dot_scaled(x, None, "bf16", tl.trans(b), e, "e2m1", acc)
            else:
                acc = tl.dot(x, tl.trans(_fp4_tile(w_ptr, s_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)), acc)
        # the other BM - 1 rows of the accumulator are exactly zero (their x rows were), so the sum is row 0
        tl.store(o_ptr + p.to(tl.int64) * I + rows, tl.sum(acc, axis=0).to(tl.bfloat16), mask=live_n)

    @triton.jit
    def _fp4_direct_gate_up_kernel(x_ptr, idx_ptr, map_ptr, start, w1_ptr, s1_ptr, w3_ptr, s3_ptr, g_ptr, u_ptr,
                                   I, K, TOPK, NROWS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                   NATIVE: tl.constexpr, TMA: tl.constexpr):
        """Gate or up (the third grid axis) for one pair (a token and one of its chosen experts), BN
        of the expert's rows: the decode step's path, with no schedule. The program finds its
        expert's slot itself from the pair's expert id and the pool's slot map (a pair whose expert
        is not held here writes nothing; the down kernel writes its reply as zero). Each pair reads
        its expert's weights whole; a decode step's few pairs rarely share an expert, and the
        grouped kernels take the prefill's many. Gate and up are separate programs on purpose: one
        tensor-core dot per load keeps the loads flowing (measured cold at one token on an RTX PRO
        6000: 70 us for 75 MB against 78 with both in one program)."""
        p = tl.program_id(0)
        slot = tl.load(map_ptr + tl.load(idx_ptr + p) - start).to(tl.int64)
        if slot < 0:
            return
        tok = (p // TOPK).to(tl.int64)
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        live_n = rows < I
        bank_rows = slot * I + rows
        if tl.program_id(2) == 0:
            _fp4_direct_one(x_ptr, w1_ptr, s1_ptr, g_ptr, p, tok, bank_rows, rows, live_n, I, K, NROWS, BM, BN, BK, NATIVE, TMA)
        else:
            _fp4_direct_one(x_ptr, w3_ptr, s3_ptr, u_ptr, p, tok, bank_rows, rows, live_n, I, K, NROWS, BM, BN, BK, NATIVE, TMA)

    @triton.jit
    def _fp4_direct_down_kernel(g_ptr, u_ptr, wt_ptr, idx_ptr, map_ptr, start, w2_ptr, s2_ptr, part_ptr, cnt_ptr,
                                y_ptr, D, I, limit, TOPK, NT, NROWS,
                                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr,
                                SUM: tl.constexpr, TMA: tl.constexpr):
        """The down projection for one pair, BN of its expert's rows, with the SwiGLU folded into the
        loads: each BK tile of the pair's input is w * silu(g) * u from the gate/up outputs, rounded
        and quantized per 32 exactly as _swiglu_quant_kernel does (the tile is a multiple of 32, the
        quantization's block). The pair's reply, rounded to bf16 as the module rounds it, goes to
        part[pair]. With SUM, the last of a token's TOPK programs on this row tile (a counter per
        token and tile, which it resets) adds the token's replies in k order into y: the combine
        with no launch of its own and a fixed order of summation."""
        limit = limit.to(tl.float32)
        p = tl.program_id(0)
        tile = tl.program_id(1)
        t = (p // TOPK).to(tl.int64)
        rows = tile * BN + tl.arange(0, BN)
        live_n = rows < D
        slot = tl.load(map_ptr + tl.load(idx_ptr + p) - start).to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if slot >= 0:
            bank_rows = slot * D + rows
            w = tl.load(wt_ptr + p).to(tl.float32)
            live_m = tl.arange(0, BM) == 0
            if TMA:  # w2_ptr is the bank's descriptor (made on the host, _fp4_descriptor)
                base = (slot * D + tile * BN).to(tl.int32)
            for i0 in range(0, I, BK):
                offs = p.to(tl.int64) * I + i0 + tl.arange(0, BK)
                g = tl.load(g_ptr + offs).to(tl.bfloat16).to(tl.float32)
                u = tl.load(u_ptr + offs).to(tl.bfloat16).to(tl.float32)
                if limit > 0:
                    u = tl.minimum(tl.maximum(u, -limit), limit)
                    g = tl.minimum(g, limit)
                h = (w * ((g * tl.sigmoid(g)) * u)).to(tl.bfloat16).to(tl.float32)
                h = tl.reshape(_quant_tile(tl.reshape(h, (1, BK)), 1, BK), (BK,))
                hx = tl.where(live_m[:, None], h[None, :], 0.0).to(tl.bfloat16)
                if TMA:
                    b2 = w2_ptr.load([base, i0 // 2])
                    e2 = _fp4_scales(s2_ptr, bank_rows, live_n, i0, I // 32, BN, BK)
                    acc = tl.dot_scaled(hx, None, "bf16", tl.trans(b2), e2, "e2m1", acc)
                elif NATIVE:
                    b2, e2 = _fp4_raw(w2_ptr, s2_ptr, bank_rows, live_n, i0, I // 2, I // 32, BN, BK)
                    acc = tl.dot_scaled(hx, None, "bf16", tl.trans(b2), e2, "e2m1", acc)
                else:
                    acc = tl.dot(hx, tl.trans(_fp4_tile(w2_ptr, s2_ptr, bank_rows, live_n, i0, I // 2, I // 32, BN, BK)), acc)
        r = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)
        tl.store(part_ptr + p.to(tl.int64) * D + rows, r, mask=live_n)
        if SUM:
            tl.debug_barrier()
            old = tl.atomic_add(cnt_ptr + t * NT + tile, 1, sem="acq_rel")
            if old == TOPK - 1:
                y = tl.zeros((BN,), dtype=tl.float32)
                for k in range(TOPK):
                    y += tl.load(part_ptr + (t * TOPK + k) * D + rows, mask=live_n, other=0.0, cache_modifier=".cg")
                tl.store(y_ptr + t * D + rows, y, mask=live_n)
                tl.atomic_xchg(cnt_ptr + t * NT + tile, 0)


if triton is not None:
    @triton.jit
    def _max_combine(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def _grouped_schedule_kernel(slot_ptr, scratch_ptr, order_ptr, bslot_ptr, bfirst_ptr, bcount_ptr, P, S, NB,
                                 BM: tl.constexpr, PP: tl.constexpr, NBP: tl.constexpr):
        """_grouped_schedule in one program, for up to 8192 pairs: sort the pairs by slot (stable:
        the key carries the pair index), cut each slot's run into blocks of BM, and count each
        block's pairs."""
        i = tl.arange(0, PP)
        live = i < P
        sl = tl.load(slot_ptr + i, mask=live, other=-1)
        sl = tl.where(sl < 0, S, sl)  # not held here: after every held slot
        sl = tl.where(live, sl, S + 1)  # padding: after those
        key = tl.sort(sl * 16384 + i)
        ks = key >> 14
        held = ks < S
        tl.store(order_ptr + i, key & 16383, mask=live)
        b = tl.arange(0, NBP)
        tl.store(bcount_ptr + b, tl.zeros((NBP,), dtype=tl.int32), mask=b < NB)
        tl.store(scratch_ptr + i, ks)
        tl.debug_barrier()
        prev = tl.load(scratch_ptr + i - 1, mask=i > 0, other=-1)
        run_start = tl.associative_scan(tl.where(ks != prev, i, 0), 0, _max_combine)
        starts = held & ((i - run_start) % BM == 0)
        bid = tl.cumsum(starts.to(tl.int32), 0) - 1
        tl.store(bfirst_ptr + bid, i, mask=starts)
        tl.store(bslot_ptr + bid, ks, mask=starts)
        tl.atomic_add(bcount_ptr + bid, 1, mask=held)


def _grouped_schedule(slot, S, BM):
    """Blocks of up to BM pairs sharing a held expert, from each pair's slot (-1: not held here).
    All on the device and of fixed size, so a CUDA graph can capture it: (order of the pairs
    sorted by slot, and per block its slot, first index into the order, and pair count; blocks
    past the last real one have count <= 0)."""
    P = slot.numel()
    blocks = triton.cdiv(P, BM) + min(S, P)  # an upper bound: every held slot adds at most one partial block
    if P <= 8192 and S + 1 < 16384:  # one program sorts and cuts: one launch instead of a dozen
        # two sizes only (a decode step or a short prompt; a long prompt), so that prompts of every
        # length share two compiled kernels instead of one per power of two
        PP = 1024 if P <= 1024 else 8192
        dev = slot.device
        order = torch.empty(P, dtype=torch.int32, device=dev)
        scratch = torch.empty(PP, dtype=torch.int32, device=dev)
        bslot = torch.empty(blocks, dtype=torch.int32, device=dev)
        bfirst = torch.empty(blocks, dtype=torch.int32, device=dev)
        bcount = torch.empty(blocks, dtype=torch.int32, device=dev)
        with torch.cuda.device(dev):
            _grouped_schedule_kernel[(1,)](slot.contiguous(), scratch, order, bslot, bfirst, bcount, P, S, blocks,
                                           BM=BM, PP=PP, NBP=2 * PP, num_warps=8)
        return order, bslot, bfirst, bcount, blocks
    key = torch.where(slot < 0, S, slot).to(torch.int64)
    order = torch.argsort(key, stable=True).to(torch.int32)
    counts = torch.zeros(S + 1, dtype=torch.int64, device=slot.device).scatter_add_(0, key, torch.ones_like(key))
    counts = counts[:S]
    starts = torch.cumsum(counts, 0) - counts  # each slot's first index into the order
    nblocks = (counts + BM - 1) // BM
    blk_end = torch.cumsum(nblocks, 0)
    b = torch.arange(blocks, device=slot.device)
    s_of_b = torch.searchsorted(blk_end, b, right=True).clamp(max=S - 1)
    j = b - (blk_end[s_of_b] - nblocks[s_of_b])  # the block's index within its slot
    first = starts[s_of_b] + j * BM
    count = torch.where(b < blk_end[-1], torch.clamp(counts[s_of_b] - j * BM, max=BM), 0)
    return order, s_of_b.to(torch.int32), first.to(torch.int32), count.to(torch.int32), blocks


def _block_k(dim, want):
    """The K tile for a reduction over dim: want, or the largest smaller power of two that divides
    dim (at least 32, the scale block, which every bank's inner sizes are multiples of)."""
    bk = want
    while bk > 32 and dim % bk:
        bk //= 2
    if dim % bk:
        raise ValueError(f"fp4 experts: {dim} is not a multiple of 32")
    return bk


FP4_KSPLIT = os.environ.get("VXPU_FP4_KSPLIT")  # override: the K split of the fp4 expert kernels (measurement)
FP4_NATIVE = os.environ.get("VXPU_FP4_NATIVE")  # override: force the native (1) or software-decode (0) body (A/B)
FP4_DIRECT = os.environ.get("VXPU_FP4_DIRECT")  # override: force the direct (1) or grouped (0) decode path (A/B)
FP4_TMA = os.environ.get("VXPU_FP4_TMA")  # override: force the descriptor (1) or plain (0) loads in the direct kernels (A/B)
# the direct kernels' shapes with descriptor loads (rows a program, the K tile, warps, stages), measured cold at one
# token on an RTX PRO 6000, bit-identical to the plain loads: gate/up 60.2 us for 75 MB (plain loads 75.6; floor 52),
# down 30.6 us for 38 MB (plain 32.6; floor 26); the best of 36 and 36 shapes, each timed twice
# The direct expert kernels' tiles (rows of the expert per program, K per iteration, warps, pipeline
# stages), table knobs (plan.KNOBS): measured on an RTX PRO 6000 at six pairs with the L2 defeated
# (hero run 11; the sweep over 108 sets per kernel is in dev/hero/fp4-sweep-rtx6000.json). Gate/up
# with x as one row takes three stages (57.6 us for 75.2 MB, 1,305 GB/s, against 60.4 at two);
# down is best at two (32.9 us for 37.6 MB; every other set measured slower).
FP4_GU_BN, FP4_GU_BK, FP4_GU_WARPS, FP4_GU_STAGES = 32, 256, 2, 3
FP4_DN_BN, FP4_DN_BK, FP4_DN_WARPS, FP4_DN_STAGES = 32, 256, 2, 2
_FP4_DESCRIPTORS = {}


def _fp4_descriptor(bank, rows, cols, bn, bk_bytes):
    """The TMA descriptor over a bank seen as [rows, cols] bytes with boxes of [bn, bk_bytes], made
    on the host once per bank and tile and passed to the kernels as an argument: a kernel making
    its own per program paid about 1 us a launch (measured at six pairs). Keyed by the bank's
    address and shape, which name its contents for the descriptor's purposes."""
    key = (bank.data_ptr(), rows, cols, bn, bk_bytes)
    d = _FP4_DESCRIPTORS.get(key)
    if d is None:
        from triton.tools.tensor_descriptor import TensorDescriptor
        d = TensorDescriptor.from_tensor(bank.view(rows, cols), block_shape=[bn, bk_bytes])
        _FP4_DESCRIPTORS[key] = d
    return d


@functools.lru_cache(maxsize=None)
def _fp4_tma(device_index):
    """Whether the direct expert kernels stream their bytes through tensor descriptors (TMA) on
    this GPU: Hopper and later, with a Triton that has them; the descriptors' scratch comes from
    Triton's allocator, set here once. VXPU_FP4_TMA only overrides it for an A/B."""
    if FP4_TMA is not None and FP4_TMA != "1":
        return False
    if triton is None or not hasattr(tl, "make_tensor_descriptor"):
        return False
    major, _ = torch.cuda.get_device_capability(device_index)
    if major < 9 and FP4_TMA is None:
        return False
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    return True
FP4_DIRECT_MAX = 8  # pairs up to which the direct kernels run (no schedule; each pair reads its expert whole).
# One row's pairs (top-6) and no more: the direct path streams an expert once per pair, so rows that share experts
# multiply its bytes (one prompt in 9 rows: 54 pairs over 6 experts read each nine times, 19.1 ms of experts a step
# against 4.5 at one row and 7.9 grouped); the grouped path reads each expert once for all its pairs.


@functools.lru_cache(maxsize=None)
def _blackwell(device=None):
    """Whether the GPU (the current one by default) is Blackwell or later (compute capability 10+)."""
    if not torch.cuda.is_available():
        return False
    index = device.index if isinstance(device, torch.device) else device
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device() if index is None else index)
    return major >= 10


@functools.lru_cache(maxsize=None)
def _fp4_native(device_index):
    """Whether the expert kernels hand fp4 to the tensor cores on this GPU: Blackwell (compute
    capability 10 and 12) and later take MXFP4 directly; earlier cards get the software decode.
    Chosen from the card, not configured; VXPU_FP4_NATIVE only overrides it for an A/B."""
    if FP4_NATIVE is not None:
        return FP4_NATIVE == "1"
    if triton is None or not hasattr(tl, "dot_scaled"):
        return False
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 10


def _fp4_ksplit(pairs):
    """How many programs share the reduction of one (pair block, row tile) in the fp4 expert
    kernels. A decode step's few pairs give few programs otherwise; a prefill's many need none.
    The right value is the planner's, per shape and GPU; this is the stopgap it will replace."""
    if FP4_KSPLIT:
        return int(FP4_KSPLIT)
    if pairs <= 16:
        return 4
    if pairs <= 64:
        return 2
    return 1



def fp4_experts_decode(x_in, weights, slots, w1, s1, w3, s3, w2, s2, swiglu_limit, per_pair=False,
                       indices=None, start=0):
    """The routed experts of some tokens, for the pairs whose expert is held on the GPU: x_in [n, K]
    bf16 (already on the fp8 grid, as the module quantizes it), weights [n, topk] fp32, slots
    [n, topk] int (the expert's slot in the GPU banks, or -1 for experts computed elsewhere), banks
    w1/w3 [S, I, K/2] and w2 [S, K, I/2] packed fp4 with e8m0 scales per 32 along the inner
    dimension. Returns y [n, K] fp32, the sum over the token's held pairs (0 where none), or with
    per_pair=True each pair's reply, [n, topk, K] (0 where not held; mailboxes.py sums them in a
    fixed order). With indices ([n, topk] expert ids), slots is instead the pool's slot map (one
    entry per expert from `start`).

    The pairs are grouped by expert (_grouped_schedule): each block of up to BM pairs that chose
    the same held expert decodes the expert's weights once, a tile at a time, and multiplies them
    on the tensor cores with fp32 sums. The rounding points are the module's _forward_batched:
    gate and up rounded to bf16, the clamps, SiLU and routing weight, the down projection's input
    re-quantized per 32-block, its output rounded to bf16. Tile shapes follow the number of pairs
    (measured on an L4: narrow, deep tiles for a few pairs; taller blocks for many)."""
    pairs_n = (indices if indices is not None else slots).numel()
    native = _fp4_native(x_in.device.index)
    direct = indices is not None and pairs_n <= FP4_DIRECT_MAX if FP4_DIRECT is None else FP4_DIRECT == "1"
    if direct and indices is not None:
        return _fp4_experts_direct(x_in, weights, slots, indices, start, w1, s1, w3, s3, w2, s2, swiglu_limit,
                                   per_pair, native)
    BM = 32 if pairs_n >= 768 else 16
    if native:  # measured on an RTX PRO 6000 at one token with a cold L2: BN 64, BK 256, no K split (75.6 us for 75 MB)
        BN, BK = (64, 256) if pairs_n <= 64 else (64, 128)
        KS = int(FP4_KSPLIT) if FP4_KSPLIT else 1
    else:
        BN, BK = (32, 256) if pairs_n <= 64 else (64, 128)
        KS = _fp4_ksplit(pairs_n)
    n, topk = (indices if indices is not None else slots).shape
    K = x_in.shape[1]
    S, I, _ = w1.shape
    pairs = n * topk
    dev = x_in.device
    if indices is not None:  # the pool's slot map, one entry per expert from `start`
        slot = slots[(indices.reshape(-1) - start).long()].to(torch.int32)
    else:
        slot = slots.reshape(-1).to(torch.int32)
    order, blk_slot, blk_first, blk_count, blocks = _grouped_schedule(slot, S, BM)
    if KS == 1:
        g = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
        u = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    else:  # the K splits add their partial sums here
        g = torch.zeros(pairs, I, dtype=torch.float32, device=dev)
        u = torch.zeros(pairs, I, dtype=torch.float32, device=dev)
    h = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    d = torch.zeros(pairs, K, dtype=torch.float32, device=dev)  # 0 for the pairs not held here
    x = x_in.contiguous()
    wt = weights.to(torch.float32).contiguous().view(-1)
    with torch.cuda.device(dev):
        bk1, bk2 = _block_k(K, BK), _block_k(I, BK)
        ks1 = KS if K % (KS * bk1) == 0 else 1
        ks2 = KS if I % (KS * bk2) == 0 else 1
        _fp4_grouped_gate_up_kernel[(blocks, triton.cdiv(I, BN), ks1)](
            x, order, blk_slot, blk_first, blk_count, w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), g, u,
            I, K, topk, BM=BM, BN=BN, BK=bk1, KS=ks1, NATIVE=native)
        _swiglu_quant_kernel[(triton.cdiv(I // 32, 16), pairs)](g, u, wt, slot, h, I, float(swiglu_limit), G=16)
        _fp4_grouped_down_kernel[(blocks, triton.cdiv(K, BN), ks2)](
            h, order, blk_slot, blk_first, blk_count, w2, s2.view(torch.uint8), d, K, I, BM=BM, BN=BN,
            BK=bk2, KS=ks2, NATIVE=native)
    return d.view(n, topk, K) if per_pair else d.view(n, topk, K).sum(1)


_FP4_COUNTERS = {}


FP4_COUNTER_SCOPE = None  # set by a step graph while it captures: its counters live with its graphs


def _fp4_counters(dev, size):
    """The down kernel's arrival counters, one buffer per GPU and size, zero between calls: the
    last program of each (token, row tile) resets its counter, so no fill launch is needed per
    layer. The layers of a step run in order on their stream, so they share the buffer. Two steps
    in flight on one GPU (hero run 9: a batch per layer range) must not share it, or their
    arrivals add up across batches and the sums come out wrong (with four one-row batches three
    produced other tokens than the single-batch run). A step graph capturing sets
    FP4_COUNTER_SCOPE to a list of its own: the buffers made under it are keyed by it, handed to
    the list (they live as long as the graphs) and forgotten here when the capture ends, so a
    later graph never reuses memory a freed graph's pool gave back (an illegal address)."""
    scope = FP4_COUNTER_SCOPE
    key = (str(dev), size, id(scope) if scope is not None else None)
    buf = _FP4_COUNTERS.get(key)
    if buf is None:
        buf = _FP4_COUNTERS[key] = torch.zeros(size, dtype=torch.int32, device=dev)
        if scope is not None:
            scope.append(buf)
    return buf


def fp4_counter_scope_end(scope):
    """The capture under `scope` is over: its counters belong to it alone now."""
    for key in [k for k in _FP4_COUNTERS if k[2] == id(scope)]:
        del _FP4_COUNTERS[key]


def _fp4_experts_direct(x_in, weights, slot_map, indices, start, w1, s1, w3, s3, w2, s2, swiglu_limit, per_pair,
                        native):
    """fp4_experts_decode's decode path: two launches a layer. The gate/up kernel per (pair, row
    tile) with the slot looked up in the kernel, then the down kernel per (pair, row tile) with the
    SwiGLU in its loads and, unless per_pair, the token's sum done by the last of its programs.
    No schedule, no SwiGLU kernel, no zeroed reply buffer, no separate sum."""
    n, topk = indices.shape
    K = x_in.shape[1]
    S, I, _ = w1.shape
    pairs = n * topk
    dev = x_in.device
    BM = 16
    tma = native and _fp4_tma(dev.index) and (K // 2) % 16 == 0 and (I // 2) % 16 == 0  # descriptor strides: 16 bytes
    if tma:
        bn1, bk1, warps1, stages1 = int(FP4_GU_BN), int(FP4_GU_BK), int(FP4_GU_WARPS), int(FP4_GU_STAGES)
        bn2, bk2, warps2, stages2 = int(FP4_DN_BN), int(FP4_DN_BK), int(FP4_DN_WARPS), int(FP4_DN_STAGES)
    else:
        bn1, bk1, warps1, stages1 = 64, 256, 4, 3
        bn2, bk2, warps2, stages2 = 64, 256, 4, 3
    g = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    u = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    part = torch.empty(pairs, K, dtype=torch.float32, device=dev)
    x = x_in.contiguous()
    wt = weights.to(torch.float32).contiguous().view(-1)
    idx = indices.contiguous().view(-1)
    smap = slot_map.contiguous()
    with torch.cuda.device(dev):
        bk1, bk2 = _block_k(K, bk1), _block_k(I, bk2)
        nt = triton.cdiv(K, bn2)
        if tma:  # the banks go in as descriptors
            w1 = _fp4_descriptor(w1, S * I, K // 2, bn1, bk1 // 2)
            w3 = _fp4_descriptor(w3, S * I, K // 2, bn1, bk1 // 2)
            w2 = _fp4_descriptor(w2, S * K, I // 2, bn2, bk2 // 2)
        _fp4_direct_gate_up_kernel[(pairs, triton.cdiv(I, bn1), 2)](
            x, idx, smap, start, w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), g, u, I, K, topk, S * I,
            BM=BM, BN=bn1, BK=bk1, NATIVE=native, TMA=tma, num_warps=warps1, num_stages=stages1)
        if per_pair:
            _fp4_direct_down_kernel[(pairs, nt)](
                g, u, wt, idx, smap, start, w2, s2.view(torch.uint8), part, part, part, K, I, float(swiglu_limit),
                topk, nt, S * K, BM=BM, BN=bn2, BK=bk2, NATIVE=native, SUM=False, TMA=tma, num_warps=warps2,
                num_stages=stages2)
            return part.view(n, topk, K)
        y = torch.empty(n, K, dtype=torch.float32, device=dev)
        counters = _fp4_counters(dev, n * nt)
        _fp4_direct_down_kernel[(pairs, nt)](
            g, u, wt, idx, smap, start, w2, s2.view(torch.uint8), part, counters, y, K, I, float(swiglu_limit),
            topk, nt, S * K, BM=BM, BN=bn2, BK=bk2, NATIVE=native, SUM=True, TMA=tma, num_warps=warps2,
            num_stages=stages2)
    return y


if triton is not None:
    @triton.jit
    def _combine_kernel(cpu_ptr, loc_ptr, peer_ptr, idx_ptr, loc_map_ptr, peer_map_ptr, y_ptr, start, K,
                        TOPK: tl.constexpr, HAS_LOC: tl.constexpr, HAS_PEER: tl.constexpr, BK: tl.constexpr):
        t = tl.program_id(1)
        offs = tl.program_id(0) * BK + tl.arange(0, BK)
        live = offs < K
        acc = tl.zeros((BK,), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            p = t * TOPK + k
            r = tl.load(cpu_ptr + p.to(tl.int64) * K + offs, mask=live, other=0.0)
            e = tl.load(idx_ptr + p) - start
            if HAS_PEER:
                held = tl.load(peer_map_ptr + e) >= 0
                r = tl.where(held, tl.load(peer_ptr + p.to(tl.int64) * K + offs, mask=live & held, other=0.0), r)
            if HAS_LOC:
                held = tl.load(loc_map_ptr + e) >= 0
                r = tl.where(held, tl.load(loc_ptr + p.to(tl.int64) * K + offs, mask=live & held, other=0.0), r)
            acc = acc + r  # in k order, from zero: the fused CPU kernel's sum
        tl.store(y_ptr + t.to(tl.int64) * K + offs, acc, mask=live)


def combine_replies(cpu, indices, start, local=None, local_map=None, peer=None, peer_map=None):
    """One experts node's replies [n, topk, K] fp32 combined into y [n, K]:
    each pair's reply from the pool on this GPU if its slot map holds the
    expert, else from the other GPU's if that one does, else the CPU's,
    summed in k order into fp32 from zero. One kernel; the same sums as
    mailboxes.Layer's torch.where and adds, in the same order."""
    n, topk, K = cpu.shape
    y = torch.empty(n, K, dtype=torch.float32, device=cpu.device)
    BK = 1024
    dummy = cpu
    with torch.cuda.device(cpu.device):
        _combine_kernel[(triton.cdiv(K, BK), n)](
            cpu, local if local is not None else dummy, peer if peer is not None else dummy,
            indices.contiguous().view(-1), local_map if local_map is not None else indices,
            peer_map if peer_map is not None else indices, y, start, K,
            TOPK=topk, HAS_LOC=local is not None, HAS_PEER=peer is not None, BK=BK)
    return y
