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
import types

import os

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
        float8_e4m3fn and back does, in fp32 arithmetic: a normal value keeps 3 of its 23 mantissa
        bits (round to nearest even on the 20 dropped), a subnormal one (below 2^-6) becomes a
        multiple of 2^-9. Any GPU (Triton's fp8e4nv type needs sm_89)."""
        bits = q.to(tl.int32, bitcast=True)
        lsb = (bits >> 20) & 1
        normal = ((bits + 0x7FFFF + lsb) & ~0xFFFFF).to(tl.float32, bitcast=True)
        sub = tl.extra.cuda.libdevice.rint(q * 512.0) / 512.0
        return tl.where(tl.abs(q) < 0.015625, sub, normal)

    @triton.jit
    def _round_e2m1(x):
        """x (fp32, within [-6, 6]) rounded to the nearest fp4 e2m1 value, ties to even index."""
        a = tl.abs(x)
        v = tl.where(a <= 5.0, 4.0, 6.0)
        v = tl.where(a < 3.5, 3.0, v)
        v = tl.where(a <= 2.5, 2.0, v)
        v = tl.where(a < 1.75, 1.5, v)
        v = tl.where(a <= 1.25, 1.0, v)
        v = tl.where(a < 0.75, 0.5, v)
        v = tl.where(a <= 0.25, 0.0, v)
        return tl.where(x.to(tl.int32, bitcast=True) < 0, -v, v)

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
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and M <= GEMV_ROWS_MAX and K % 32 == 0 and N % 32 == 0:
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
    def _gemv_rows_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, XS, R, KS,
                          HAS_SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """x [M, K] (bf16) times weight [N, K] transposed, every row of x by one pass over the
        weight: a program owns BN weight rows, streams them once, and multiplies each tile by all
        M rows of x on the tensor cores (tl.dot, bf16 in, fp32 accumulation; an fp8 weight times
        its e8m0 block scale is exact in bf16, so the products are exact). Up to BM rows."""
        pid = tl.program_id(0)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)  # int64: weights past 2^31 elements
        live = rows < N
        g = (pid * BN) // R  # BN divides R: one group per program
        ms = tl.arange(0, BM)
        mlive = ms < M
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            w = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None] & kin[None, :], other=0.0)
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
        tl.store(y_ptr + ms[None, :] * N + rows[:, None], acc.to(y_ptr.dtype.element_ty),
                 mask=live[:, None] & mlive[None, :])


# rows one pass over the weight serves (BM); beyond, a GEMM. VXPU_GEMV_ROWS_MAX overrides it, to
# measure where the one-pass kernel should hand over (hero 6: 33 rows were slower at 64 than at 32)
GEMV_ROWS_MAX = int(os.environ.get("VXPU_GEMV_ROWS_MAX", "64"))


def _rows_blocks(M, K):
    """Block shape for _gemv_rows_kernel: rows padded to a tensor-core tile, 16 weight rows per
    program (32 above 32 rows: measured on an L4, 16 fell to 130-175 GB/s at 33-64 rows where 32
    held 180-255), a K step of 256 where K allows it."""
    bm = 16
    while bm < M:
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


def gemv_rows(x, weight, scale=None, group_rows=None, out_dtype=torch.bfloat16, blocks=None, traced=False):
    """x [M, G*K] bf16 times weight [N, K] (bf16, or fp8 e4m3 with [N/32, K/32] e8m0 block
    scales) transposed -> [M, N] in out_dtype, the weight read once for all M <= GEMV_ROWS_MAX
    rows. group_rows as fp8_gemv."""
    M = x.shape[0]
    N, K = weight.shape
    R = group_rows or N
    bm, bn, bk = blocks or _rows_blocks(M, K)
    if M > bm or N % R or R % bn or x.shape[1] != (N // R) * K or K % 32 or N % 32 or bk % 32:
        raise ValueError(f"gemv_rows: x {tuple(x.shape)}, weight {tuple(weight.shape)}, group_rows {group_rows}")
    if scale is not None and (weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu
                              or tuple(scale.shape) != (N // 32, K // 32)):
        raise ValueError(f"gemv_rows: weight {weight.dtype} {tuple(weight.shape)} needs e8m0 scales "
                         f"[{N // 32}, {K // 32}], got {scale.dtype} {tuple(scale.shape)}")
    y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    if M and N:
        w = weight.contiguous()
        s = scale.contiguous().view(torch.uint8) if scale is not None else w
        kernel, ctx = _launcher(_gemv_rows_kernel, x.device, traced)
        with ctx:
            kernel[(triton.cdiv(N, bn),)](x.contiguous(), w, s, y, M, N, K, x.shape[1], R,
                                         scale.shape[1] if scale is not None else 0,
                                         HAS_SCALE=scale is not None, BM=bm, BN=bn, BK=bk)
    return y


if triton is not None:
    @triton.jit
    def _dual_gemv_swiglu_kernel(x_ptr, w1_ptr, s1_ptr, w3_ptr, s3_ptr, h_ptr, M, N, K, KS, limit,
                                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """The SwiGLU front of an FFN for up to BM rows in one pass over both weights: a program
        owns BN rows of w1 and of w3 (fp8, e8m0 block scales), streams them once against all M rows
        of x (bf16 on the fp8 grid), and in its epilogue rounds the two products to bf16 as the
        fork's linears do, clamps them (the swiglu limit; inf for none), takes silu(gate) * up,
        rounds to bf16, and quantizes the result to the fp8 grid per 32 outputs (act_quant with
        ue8m0 scales, as the w2 linear's input), stored as bf16. BN is a multiple of 32, so a
        program holds whole quantization blocks."""
        limit = limit.to(tl.float32)
        pid = tl.program_id(0)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)
        live = rows < N
        ms = tl.arange(0, BM)
        mlive = ms < M
        acc1 = tl.zeros((BN, BM), dtype=tl.float32)
        acc3 = tl.zeros((BN, BM), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kin = ks < K
            kb = k0 // 32 + tl.arange(0, BK // 32)
            smask = live[:, None] & (kb < KS)[None, :]
            wmask = live[:, None] & kin[None, :]
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


def shared_ffn(x, w1, s1, w3, s3, w2, s2, swiglu_limit, traced=False):
    """An fp8 SwiGLU FFN (the shared expert) for a decode step's rows, in three kernels: x [.., K]
    bf16 quantized to the fp8 grid (act_quant_dequant), the dual GEMV over w1 and w3 [I, K] with
    the SwiGLU and the w2 input's quantization in its epilogue, and the w2 [K, I] GEMV. Returns
    [.., K] in x's dtype. The fork's Expert.forward launches about nine kernels for the same."""
    K = x.shape[-1]
    I = w1.shape[0]
    xq = act_quant_dequant(x, traced=traced).reshape(-1, K)
    M = xq.shape[0]
    if M > GEMV_ROWS_MAX or K % 32 or I % 32 or w3.shape != w1.shape or tuple(w2.shape) != (K, I):
        raise ValueError(f"shared_ffn: x {tuple(x.shape)}, w1 {tuple(w1.shape)}, w2 {tuple(w2.shape)}")
    bm, _, bk = _pe_blocks(M, K, tiles=3, device=x.device)
    bn = 32  # whole quantization blocks per program
    h = torch.empty(M, I, dtype=torch.bfloat16, device=x.device)
    if M:
        kernel, ctx = _launcher(_dual_gemv_swiglu_kernel, x.device, traced)
        with ctx:
            kernel[(triton.cdiv(I, bn),)](xq, w1.contiguous(), s1.contiguous().view(torch.uint8), w3.contiguous(),
                                         s3.contiguous().view(torch.uint8), h, M, I, K, s1.shape[1],
                                         float(swiglu_limit) if swiglu_limit > 0 else float("inf"),
                                         BM=bm, BN=bn, BK=bk)
    y = gemv_rows(h, w2, s2, traced=traced)
    return y.reshape(tuple(x.shape[:-1]) + (K,)).to(x.dtype)


if triton is not None:
    @triton.jit
    def _gemv_pe_kernel(x_ptr, w_ptr, s_ptr, y_ptr, xn_ptr, nw_ptr, freqs_ptr, sign, M, N, K, KS, R, eps,
                        PRO_NORM: tl.constexpr, PRO_ROTARY: tl.constexpr, PRO_QUANT: tl.constexpr,
                        STORE_XN: tl.constexpr, EPI_ROTARY: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """_gemv_rows_kernel with the work around the GEMV folded in. Prologue, per row of x as it
        is read: RMSNorm with nw_ptr's weight (PRO_NORM; the fork's: fp32, rounded to bf16), then
        quantization to the fp8 grid per 32 elements (PRO_QUANT, act_quant with ue8m0 scales), the
        normalized row stored by program 0 (STORE_XN, for whoever else reads it). Epilogue: the
        bf16 outputs rotated in adjacent pairs over the last RD of every HD outputs (EPI_ROTARY,
        freqs_ptr the complex frequencies' real view [M, RD / 2, 2] per row, sign -1 for the
        inverse), as the fork's apply_rotary_emb. BN must divide
        HD - RD and be even. PRO_ROTARY rotates the input the same way instead (the pairs along K,
        the last RD of every HD; sign picks the direction), for the
        attention's output. Weight rows come in groups of R, group g reading x[:, g * K:(g + 1)
        * K] (R = N for one group; BN divides R)."""
        eps = eps.to(tl.float32)
        sign = sign.to(tl.float32)
        pid = tl.program_id(0)
        rows = pid.to(tl.int64) * BN + tl.arange(0, BN)
        live = rows < N
        g = (pid * BN) // R  # the group: BN divides R, so one group per program
        XS = (N // R) * K  # x's row stride: one K-slice per group
        xb = x_ptr + g * K
        ms = tl.arange(0, BM)
        mlive = ms < M
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
                fb = freqs_ptr + ms[:, None] * RD + 2 * pair[None, :]
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
            fb = freqs_ptr + ms[None, :] * RD + 2 * pair[:, None]
            c = tl.load(fb, mask=pm, other=1.0)
            sn = tl.load(fb + 1, mask=pm, other=0.0) * sign
            y3 = tl.join(ev * c - od * sn, ev * sn + od * c)
            y = tl.reshape(tl.permute(y3, (0, 2, 1)), (BN, BM))
        tl.store(y_ptr + ms[None, :] * N + rows[:, None], y.to(tl.bfloat16), mask=live[:, None] & mlive[None, :])


def attn_q(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs, heads, head_dim, rope_dim, traced=False):
    """The attention's query path for a decode step's rows, in four kernels: x [.., K] quantized
    to the fp8 grid, the down-projection wq_a [R, K] (fp8, e8m0 block scales), one program per row
    for q_norm and the fp8-grid quantization of its result (the normalized query also kept, for
    the indexer), then the up-projection wq_b [N, R] with the rotary of every head's last
    rope_dim dims in its epilogue (freqs [rows, rope_dim / 2, 2] fp32: the complex frequencies'
    real view, per row). Returns (qn [.., R] bf16, the normalized query; q [.., N] bf16,
    rotated). The per-row work runs once a row,
    not once per program of the wide GEMV, which recomputed it 2,048 times and lost to the
    compiler's version. The fork launches about nine kernels for the same."""
    K = x.shape[-1]
    R, N = wq_a.shape[0], wq_b.shape[0]
    xq = act_quant_dequant(x, traced=traced).reshape(-1, K)
    M = xq.shape[0]
    bm, bn, bk = _pe_blocks(M, R, tiles=2, device=x.device)
    BR = triton.next_power_of_2(R)  # the per-row kernel masks past R
    if (M > GEMV_ROWS_MAX or K % 32 or R % 32 or N % head_dim or head_dim % bn
            or (head_dim - rope_dim) % bn or rope_dim % 2 or wq_b.shape[1] != R
            or tuple(freqs.shape) != (M, rope_dim // 2, 2)):
        raise ValueError(f"attn_q: x {tuple(x.shape)}, wq_a {tuple(wq_a.shape)}, wq_b {tuple(wq_b.shape)}, "
                         f"heads {heads} x {head_dim}, rope {rope_dim}, freqs {tuple(freqs.shape)}")
    freqs = freqs.contiguous()
    qr = gemv_rows(xq, wq_a, sa, traced=traced)
    qn = torch.empty(M, R, dtype=torch.bfloat16, device=x.device)
    qq = torch.empty(M, R, dtype=torch.bfloat16, device=x.device)
    q = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if M:
        post, ctx = _launcher(_row_post_kernel, x.device, traced)
        with ctx:
            post[(M,)](qr, norm_weight.contiguous(), freqs, 1.0, qq, qn, M, R, 1, float(norm_eps),
                       NORM=True, RD=0, QUANT=True, STORE_PRE=True, BD=BR)
        kernel, ctx = _launcher(_gemv_pe_kernel, x.device, traced)
        with ctx:
            kernel[(triton.cdiv(N, bn),)](qq, wq_b.contiguous(), sb.contiguous().view(torch.uint8), q, q, qq,
                                         freqs, 1.0, M, N, R, sb.shape[1], N, 0.0,
                                         PRO_NORM=False, PRO_ROTARY=False, PRO_QUANT=False, STORE_XN=False,
                                         EPI_ROTARY=True, HD=head_dim, RD=rope_dim, BM=bm, BN=bn, BK=bk)
    lead = tuple(x.shape[:-1])
    return qn.reshape(lead + (R,)), q.reshape(lead + (N,))


if triton is not None:
    @triton.jit
    def _row_post_kernel(x_ptr, nw_ptr, freqs_ptr, sign, y_ptr, pre_ptr, M, D, HPR, eps,
                         NORM: tl.constexpr, RD: tl.constexpr, QUANT: tl.constexpr, STORE_PRE: tl.constexpr,
                         BD: tl.constexpr, FP4: tl.constexpr = False, FP4_E4M3: tl.constexpr = False):
        """A row's finishing after a projection, one program per row of x [M, D] (bf16): RMSNorm
        with nw_ptr's weight (NORM; fp32, rounded to bf16), the rotary of the last RD elements in
        adjacent pairs (freqs_ptr: the complex frequencies' real view [M, RD / 2, 2], cos and sin
        interleaved, read in place; sign -1 for the inverse rotation; RD = 0 for none),
        quantization to the fp8 grid per 32 elements (QUANT, act_quant with ue8m0 scales), or
        fp4 quantization to e2m1 values (FP4, with e4m3 scales per 16 elements if FP4_E4M3 else
        e8m0 scales per 32 elements), stored as bf16. The fork's kv_norm, apply_rotary_emb and
        act_quant / fp4_quant_dequant, each a kernel or more, as one. BD is D rounded up to a power
        of two; RD divides D and is even. HPR consecutive rows share a frequency row (the heads of
        one token). STORE_PRE also stores the row before the quantization (pre_ptr), for a reader
        that wants it unquantized."""
        eps = eps.to(tl.float32)
        sign = sign.to(tl.float32)
        m = tl.program_id(0)
        d = tl.arange(0, BD)
        live = d < D
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
            fb = freqs_ptr + (m // HPR) * RD + 2 * pair
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
        elif FP4:
            if FP4_E4M3:
                xg = tl.reshape(x, (BD // 16, 16))
                amax = tl.maximum(tl.max(tl.abs(xg), axis=1), 0.01171875)
                scale = _round_e4m3(tl.minimum(amax / 6.0, 448.0))
                q = tl.minimum(tl.maximum(xg / scale[:, None], -6.0), 6.0)
                x = tl.reshape(_round_e2m1(q) * scale[:, None], (BD,))
            else:
                xg = tl.reshape(x, (BD // 32, 32))
                amax = tl.maximum(tl.max(tl.abs(xg), axis=1), 7.052966100933725e-38)
                v = amax / 6.0
                bits = v.to(tl.int32, bitcast=True)
                exponent = (bits >> 23) & 0xFF
                exponent = tl.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                q = tl.minimum(tl.maximum(xg / scale[:, None], -6.0), 6.0)
                x = tl.reshape(_round_e2m1(q) * scale[:, None], (BD,))
        tl.store(y_ptr + m * D + d, x.to(tl.bfloat16), mask=live)


def window_kv(x, wkv, scale, norm_weight, norm_eps, freqs, rope_dim, traced=False):
    """The attention's window key for a decode step's rows, in three kernels: x [.., K] quantized
    to the fp8 grid, the projection wkv [D, K] (fp8, e8m0 block scales), and the row's finishing:
    kv_norm, the rotary of the last rope_dim elements (freqs [rows, rope_dim / 2, 2] fp32: the
    complex frequencies' real view, per row) and the quantization to the fp8 grid the cache
    holds. Returns [.., D] bf16."""
    K = x.shape[-1]
    D = wkv.shape[0]
    xq = act_quant_dequant(x, traced=traced).reshape(-1, K)
    M = xq.shape[0]
    BD = triton.next_power_of_2(D)  # the per-row kernel masks past D
    if M > GEMV_ROWS_MAX or K % 32 or D % 32 or rope_dim % 2 or tuple(freqs.shape) != (M, rope_dim // 2, 2):
        raise ValueError(f"window_kv: x {tuple(x.shape)}, wkv {tuple(wkv.shape)}, rope {rope_dim}, freqs {tuple(freqs.shape)}")
    kv = gemv_rows(xq, wkv, scale, traced=traced)
    y = torch.empty(M, D, dtype=torch.bfloat16, device=x.device)
    if M:
        kernel, ctx = _launcher(_row_post_kernel, x.device, traced)
        with ctx:
            kernel[(M,)](kv, norm_weight.contiguous(), freqs.contiguous(), 1.0, y, y, M, D, 1,
                         float(norm_eps), NORM=True, RD=rope_dim, QUANT=True, STORE_PRE=False, BD=BD)
    return y.reshape(tuple(x.shape[:-1]) + (D,))


def attn_out(o, freqs, wo_a, sa, groups, rank, wo_b, sb, head_dim, rope_dim, traced=False):
    """The attention's output path for a decode step's rows, in four kernels: one program per
    (row, head) for the inverse rotary of the head's last rope_dim dims (freqs [rows, rope_dim /
    2, 2] fp32: the complex frequencies' real view, per row) and the fp8-grid quantization; the
    grouped
    projection wo_a [groups * rank, heads_per_group * head_dim] (fp8, e8m0 block scales,
    block-diagonal over the groups); its result quantized; wo_b [dim, groups * rank]. Returns
    [.., dim] bf16. The fork launches about five kernels; the fold of the rotary into the grouped
    GEMV's prologue recomputed it per program and was dropped."""
    *lead, heads, hd = o.shape
    M = o.numel() // (heads * hd)
    Kg = heads * hd // groups  # a group's slice of o
    N1, dim = wo_a.shape[0], wo_b.shape[0]
    if (M > GEMV_ROWS_MAX or hd != head_dim or N1 != groups * rank or triton.next_power_of_2(hd) != hd
            or (hd - rope_dim) % 2 or wo_a.shape[1] != Kg or wo_b.shape[1] != N1
            or tuple(freqs.shape) != (M, rope_dim // 2, 2)):
        raise ValueError(f"attn_out: o {tuple(o.shape)}, wo_a {tuple(wo_a.shape)}, wo_b {tuple(wo_b.shape)}, "
                         f"groups {groups} rank {rank}, rope {rope_dim}, freqs {tuple(freqs.shape)}")
    x = o.reshape(M * heads, hd).contiguous()
    oq = torch.empty_like(x)
    if M:
        post, ctx = _launcher(_row_post_kernel, o.device, traced)
        with ctx:
            post[(M * heads,)](x, x, freqs.contiguous(), -1.0, oq, oq, M * heads, hd, heads, 0.0,
                               NORM=False, RD=rope_dim, QUANT=True, STORE_PRE=False, BD=hd)
    mid = gemv_rows(oq.view(M, heads * hd), wo_a, sa, group_rows=rank, traced=traced)
    y = gemv_rows(act_quant_dequant(mid, traced=traced), wo_b, sb, traced=traced)
    return y.reshape(tuple(lead) + (dim,))


def compress_post(x, freqs, rope_dim, quant_kind, traced=False):
    """The compressed KV or index key finishing for a decode step's rows, in one kernel: the
    rotary of the last rope_dim elements (freqs [rows, rope_dim / 2, 2] fp32: the complex
    frequencies' real view, per row) and fp4 quantization (quant_kind: 'e4m3' per 16 elements,
    or 'e8m0' per 32 elements). Returns [.., D] bf16."""
    D = x.shape[-1]
    lead = tuple(x.shape[:-1])
    M = x.numel() // D
    BD = triton.next_power_of_2(D)
    e4m3 = quant_kind in ("e4m3", "16_e4m3", torch.float8_e4m3fn)
    if M > GEMV_ROWS_MAX or D % 32 != 0 or rope_dim % 2 != 0 or tuple(freqs.shape) != (M, rope_dim // 2, 2):
        raise ValueError(f"compress_post: x {tuple(x.shape)}, rope {rope_dim}, freqs {tuple(freqs.shape)}, quant_kind {quant_kind}")
    y = torch.empty(M, D, dtype=torch.bfloat16, device=x.device)
    if M:
        kernel, ctx = _launcher(_row_post_kernel, x.device, traced)
        with ctx:
            kernel[(M,)](x.contiguous(), x, freqs.contiguous(), 1.0, y, y, M, D, 1,
                         0.0, NORM=False, RD=rope_dim, QUANT=False, STORE_PRE=False, BD=BD,
                         FP4=True, FP4_E4M3=e4m3)
    return y.reshape(lead + (D,))


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
                if GATE:
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
                            BE: tl.constexpr, SCORE: tl.constexpr, NORM: tl.constexpr, GATE: tl.constexpr):
        """Row r: the fork's hc_split_sinkhorn on its NMIX mixes (pre, post, and comb made doubly
        stochastic by `iters` Sinkhorn iterations), then Gate.forward's choice on its E raw scores
        (the scoring function, the K largest of score + bias one at a time, their unbiased
        scores normalized when NORM, times route_scale). One warp."""
        hc_eps = hc_eps.to(tl.float32)  # inductor passes Python floats as fp64 (lower.py: Numerics)
        temp = temp.to(tl.float32)
        route_scale = route_scale.to(tl.float32)
        r = tl.program_id(0).to(tl.int64)
        hs = tl.arange(0, HC)
        m_pre = tl.load(mixes_ptr + r * NMIX + hs)
        m_post = tl.load(mixes_ptr + r * NMIX + HC + hs)
        m_comb = tl.load(mixes_ptr + r * NMIX + 2 * HC + hs[:, None] * HC + hs[None, :])
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
        raw = tl.load(scores_ptr + r * E + e, mask=live, other=0.0) / temp
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
        picked_raw = tl.load(scores_ptr + r * E + chosen, mask=kin, other=0.0) / temp
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


FFN_FRONT_EB = 8  # experts per program of the first kernel: E / EB programs a row (measured on an L4:
FFN_FRONT_BD = 1024  # 36 us a row against 60 at 16 x 256), and the elements of the row per pass


def hc_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
             traced=False):
    """ops.hc_front's outputs: ffn_front's kernels without the gate."""
    pre, post, comb, xn, _, _, _ = ffn_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                                             norm_weight, norm_eps, None, None, 1, "", 1.0, False, 1.0, traced=traced)
    return pre, post, comb, xn


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
    x_in = torch.empty(rows if gate else 0, d, dtype=torch.bfloat16, device=dev)
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
                        NORM=bool(norm_topk_prob and topk > 1), GATE=gate, num_warps=1)
    return (pre.view(b, s_, hc), post.view(b, s_, hc), comb.view(b, s_, hc, hc), xn.view(b, s_, d), x_in,
            weights, indices)


# ---------------------------------------------------------------------------
# sparse attention for decode (ops.sparse_attn): one kernel per (query, group
# of heads) gathers the selected KV rows, forms the scores, the softmax with
# the sink term and the output, in fp32 as the fork's aten version does

if triton is not None:
    @triton.jit
    def _sparse_attn_kernel(q_ptr, kv_ptr, idx_ptr, m_ptr, l_ptr, acc_ptr, M, H, T, N, S, scale,
                            D: tl.constexpr, HB: tl.constexpr, TB: tl.constexpr):
        """Query row r (of b * M: row b's token m), heads hg * HB ..., and split sp of its T selected
        KV rows (idx, -1 for an absent slot; TB rows per split): scores = q . kv * scale, and the
        split's softmax partials: the max, the sum of exp, and sum exp * kv. On the tensor cores:
        the scores from bf16 inputs (exact products, fp32 sums), the output from the
        probabilities split into a bf16 high and low part (about 16 bits each), so the result
        differs from the fork's fp32 computation by summation order and that split."""
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
    def _sparse_attn_combine_kernel(m_ptr, l_ptr, acc_ptr, sink_ptr, o_ptr, H, S,
                                    D: tl.constexpr, SP: tl.constexpr):
        """Query row r, head h: the S splits' partials combined under the global max, the sink term
        added to the denominator (only there), o = acc / denominator. A query with no valid slot
        gives zeros."""
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
        tl.store(o_ptr + (r * H + h) * D + ds, (acc / denom).to(o_ptr.dtype.element_ty))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, traced=False):
    """The fork's sparse_attn: q [b, m, h, d], kv [b, n, d] (K and V the same rows), attn_sink [h],
    topk_idxs [b, m, t] int32 with -1 for an absent slot -> o [b, m, h, d] in q's dtype."""
    b, m, h, d = q.shape
    n = kv.shape[1]
    t = topk_idxs.shape[-1]
    if kv.shape[-1] != d or (d & (d - 1)):
        raise ValueError(f"sparse_attn: q {tuple(q.shape)} against kv {tuple(kv.shape)}; d must be a power of two")
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
                                                  l_part, acc_part, m, h, t, n, S, float(softmax_scale),
                                                  D=d, HB=HB, TB=TB, num_warps=4)
        combine, ctx = _launcher(_sparse_attn_combine_kernel, dev, traced)
        with ctx:
            combine[(b * m, h)](m_part, l_part, acc_part, attn_sink.contiguous(), o, h, S, D=d, SP=sp, num_warps=4)
    return o


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
    if x.dtype == torch.bfloat16 and M <= GEMV_ROWS_MAX and block_n is None and block_k is None:
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
                     norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                        norm_eps, traced=True)

    @torch.library.triton_op("vxpu_triton::shared_ffn", mutates_args=())
    def _shared_ffn_op(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor,
                       w2: torch.Tensor, s2: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
        return shared_ffn(x, w1, s1, w3, s3, w2, s2, swiglu_limit, traced=True)

    @torch.library.triton_op("vxpu_triton::attn_q", mutates_args=())
    def _attn_q_op(x: torch.Tensor, wq_a: torch.Tensor, sa: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
                   wq_b: torch.Tensor, sb: torch.Tensor, freqs: torch.Tensor, heads: int, head_dim: int,
                   rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        return attn_q(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs, heads, head_dim, rope_dim, traced=True)

    @torch.library.triton_op("vxpu_triton::window_kv", mutates_args=())
    def _window_kv_op(x: torch.Tensor, wkv: torch.Tensor, scale: torch.Tensor, norm_weight: torch.Tensor,
                      norm_eps: float, freqs: torch.Tensor, rope_dim: int) -> torch.Tensor:
        return window_kv(x, wkv, scale, norm_weight, norm_eps, freqs, rope_dim, traced=True)

    @torch.library.triton_op("vxpu_triton::compress_post", mutates_args=())
    def _compress_post_op(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, quant_kind: str) -> torch.Tensor:
        return compress_post(x, freqs, rope_dim, quant_kind, traced=True)

    @torch.library.triton_op("vxpu_triton::attn_out", mutates_args=())
    def _attn_out_op(o: torch.Tensor, freqs: torch.Tensor, wo_a: torch.Tensor, sa: torch.Tensor, groups: int,
                     rank: int, wo_b: torch.Tensor, sb: torch.Tensor, head_dim: int, rope_dim: int) -> torch.Tensor:
        return attn_out(o, freqs, wo_a, sa, groups, rank, wo_b, sb, head_dim, rope_dim, traced=True)

    @torch.library.triton_op("vxpu_triton::sparse_attn", mutates_args=())
    def _sparse_attn_op(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                        softmax_scale: float) -> torch.Tensor:
        return sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, traced=True)

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
    window_kv=lambda *a: torch.ops.vxpu_triton.window_kv(*a),
    compress_post=lambda *a: torch.ops.vxpu_triton.compress_post(*a),
    attn_out=lambda *a: torch.ops.vxpu_triton.attn_out(*a),
    ffn_front=lambda *a: torch.ops.vxpu_triton.ffn_front(*a),
    hc_front=lambda *a: torch.ops.vxpu_triton.hc_front(*a),
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
        g = tl.load(g_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
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
                                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Gate and up for one block of pairs sharing an expert, BN of its rows: each weight tile
        decoded once for all the block's tokens, the products on the tensor cores (fp32 sums)."""
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
        for k0 in range(0, K, BK):
            x = tl.load(x_ptr + tok[:, None] * K + k0 + tl.arange(0, BK)[None, :], mask=live_m[:, None], other=0.0)
            acc_g = tl.dot(x, tl.trans(_fp4_tile(w1_ptr, s1_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)), acc_g)
            acc_u = tl.dot(x, tl.trans(_fp4_tile(w3_ptr, s3_ptr, bank_rows, live_n, k0, K // 2, K // 32, BN, BK)), acc_u)
        out = pairs.to(tl.int64)[:, None] * I + rows[None, :]
        mask = live_m[:, None] & live_n[None, :]
        tl.store(g_ptr + out, acc_g.to(tl.bfloat16), mask=mask)
        tl.store(u_ptr + out, acc_u.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _fp4_grouped_down_kernel(h_ptr, order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr, w2_ptr, s2_ptr,
                                 d_ptr, D, I, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        slot, pairs, live_m, cnt = _grouped_block(order_ptr, blk_slot_ptr, blk_first_ptr, blk_count_ptr,
                                                  tl.program_id(0), BM)
        if cnt <= 0:
            return
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        live_n = rows < D
        bank_rows = slot * D + rows
        prow = pairs.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, I, BK):
            h = tl.load(h_ptr + prow[:, None] * I + k0 + tl.arange(0, BK)[None, :], mask=live_m[:, None], other=0.0)
            acc = tl.dot(h, tl.trans(_fp4_tile(w2_ptr, s2_ptr, bank_rows, live_n, k0, I // 2, I // 32, BN, BK)), acc)
        mask = live_m[:, None] & live_n[None, :]
        tl.store(d_ptr + prow[:, None] * D + rows[None, :], acc.to(tl.bfloat16).to(tl.float32), mask=mask)


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
    BM = 32 if pairs_n >= 768 else 16
    BN, BK = (32, 256) if pairs_n <= 64 else (64, 128)
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
    g = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    u = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    h = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    d = torch.zeros(pairs, K, dtype=torch.float32, device=dev)  # 0 for the pairs not held here
    x = x_in.contiguous()
    wt = weights.to(torch.float32).contiguous().view(-1)
    with torch.cuda.device(dev):
        _fp4_grouped_gate_up_kernel[(blocks, triton.cdiv(I, BN))](
            x, order, blk_slot, blk_first, blk_count, w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), g, u,
            I, K, topk, BM=BM, BN=BN, BK=_block_k(K, BK))
        _swiglu_quant_kernel[(triton.cdiv(I // 32, 16), pairs)](g, u, wt, slot, h, I, float(swiglu_limit), G=16)
        _fp4_grouped_down_kernel[(blocks, triton.cdiv(K, BN))](
            h, order, blk_slot, blk_first, blk_count, w2, s2.view(torch.uint8), d, K, I, BM=BM, BN=BN,
            BK=_block_k(I, BK))
    return d.view(n, topk, K) if per_pair else d.view(n, topk, K).sum(1)


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
