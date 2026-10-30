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
import os
import types

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.standard as tls
except ImportError:  # no Triton: the aten path remains
    triton = None
    tls = None


def available():
    return triton is not None and torch.cuda.is_available()


def _launcher(kernel, device, traced):
    """The kernel to launch and the context to launch it in. Eager, Triton
    launches on the current device, so that is made the tensors' device (an
    engine on cuda:1 with the process's current device still 0); traced,
    the launch goes through wrap_triton and the compiler places it."""
    if traced:
        return torch.library.wrap_triton(kernel), contextlib.nullcontext()
    if getattr(device, "type", None) == "cuda":
        return kernel, torch.cuda.device(device)
    return kernel, contextlib.nullcontext()


if triton is not None:
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
        y = q.to(tl.float8e4nv).to(tl.float32) * scale[:, None]
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
    the few rows of a decode step: each row reads the whole weight."""
    M, K = x.shape
    N = weight.shape[0]
    if weight.dim() != 2 or weight.shape[1] != K:
        raise ValueError(f"linear_f32: x {tuple(x.shape)} and weight {tuple(weight.shape)} disagree on K")
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
    block-diagonal weight, G = N / R groups); otherwise G = 1. For the
    few rows of a decode step: each row of x reads the whole weight."""
    M = x.shape[0]
    N, K = weight.shape
    R = group_rows or N
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
    """Rotate element pairs in x (last dim d) by cos and sin [s, d/2] using
    a single Triton kernel."""
    assert x.is_cuda and x.shape[-1] % 2 == 0
    if x.ndim not in (3, 4):
        raise ValueError(f"rotary: x must have rank 3 [b, s, d] or 4 [b, s, h, d], got {x.ndim} {tuple(x.shape)}")
    d = x.shape[-1]
    D = d // 2
    S = x.shape[1]
    if cos.shape != (S, D) or sin.shape != (S, D):
        raise ValueError(f"rotary: x {tuple(x.shape)} needs cos and sin [{S}, {D}], "
                         f"got cos {tuple(cos.shape)} and sin {tuple(sin.shape)}")
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


if triton is not None:
    @triton.jit
    def _index_score_kernel(
        q_ptr, k_ptr, w_ptr, cl_ptr, cand_ptr, out_ptr,
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kd,
        stride_wb, stride_ws, stride_wh,
        stride_cb, stride_cs, stride_ct,
        stride_ob, stride_os, stride_ot,
        b_size, s_size, T, H: tl.constexpr, D: tl.constexpr,
        HAS_CAND: tl.constexpr, IS_INTERP: tl.constexpr,
        BT: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_t = tl.program_id(1)

        b_idx = pid_row // s_size
        s_idx = pid_row % s_size

        offs_h = tl.arange(0, BH)
        offs_d = tl.arange(0, BD)
        offs_t = pid_t * BT + tl.arange(0, BT)

        mask_q = (offs_h[:, None] < H) & (offs_d[None, :] < D)
        ptr_q = q_ptr + b_idx * stride_qb + s_idx * stride_qs + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(ptr_q, mask=mask_q, other=0.0)

        mask_k = (offs_t[None, :] < T) & (offs_d[:, None] < D)
        ptr_k = k_ptr + b_idx * stride_kb + offs_t[None, :] * stride_kt + offs_d[:, None] * stride_kd
        k = tl.load(ptr_k, mask=mask_k, other=0.0)

        if IS_INTERP:
            dot = tl.dot(q.to(tl.float32), k.to(tl.float32))
        else:
            dot = tl.dot(q, k)
        dot = tl.maximum(dot, 0.0)

        mask_w = offs_h < H
        ptr_w = w_ptr + b_idx * stride_wb + s_idx * stride_ws + offs_h * stride_wh
        w = tl.load(ptr_w, mask=mask_w, other=0.0)

        weighted = dot * w[:, None]
        score = tl.sum(weighted, axis=0)

        cl = tl.load(cl_ptr + pid_row)

        mask = (offs_t < T) & (offs_t < cl)
        if HAS_CAND:
            ptr_c = cand_ptr + b_idx * stride_cb + s_idx * stride_cs + offs_t * stride_ct
            cand = tl.load(ptr_c, mask=(offs_t < T), other=False)
            mask = mask & cand

        final_score = tl.where(mask, score, -float("inf"))

        ptr_out = out_ptr + b_idx * stride_ob + s_idx * stride_os + offs_t * stride_ot
        tl.store(ptr_out, final_score, mask=(offs_t < T))

    @triton.jit
    def _index_topk_kernel(
        scores_ptr, cl_ptr, out_ptr, offset,
        stride_sb, stride_ss, stride_st,
        stride_ob, stride_os, stride_ok,
        b_size, s_size, T, K,
        BT: tl.constexpr, BK: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        b_idx = pid_row // s_size
        s_idx = pid_row % s_size

        cols = tl.arange(0, BT)
        mask = cols < T
        ptr_s = scores_ptr + b_idx * stride_sb + s_idx * stride_ss + cols * stride_st
        score = tl.load(ptr_s, mask=mask, other=-float("inf"))

        u = score.to(tl.int32, bitcast=True)
        pos_mask = tl.full(cols.shape, -2147483648, tl.int32)
        neg_mask = tl.full(cols.shape, -1, tl.int32)
        m = tl.where(u >= 0, pos_mask, neg_mask)
        k_val = (u ^ m).to(tl.uint32)

        packed = (k_val.to(tl.uint64) << 32) | (cols.to(tl.uint64) & 0xffffffff)

        topk_packed = tls.sort_impl(packed, k=BK, descending=True)
        chosen_idxs = (topk_packed & 0xffffffff).to(tl.int32)
        sorted_idxs = tl.sort(chosen_idxs, descending=False)

        cl = tl.load(cl_ptr + pid_row)
        neg_one = tl.full(sorted_idxs.shape, -1, tl.int32)
        result = tl.where(sorted_idxs < cl, sorted_idxs + offset, neg_one)

        out_cols = tl.arange(0, BK)
        ptr_out = out_ptr + b_idx * stride_ob + s_idx * stride_os + out_cols * stride_ok
        tl.store(ptr_out, result, mask=out_cols < K)


def index_topk(q, index_k, weights, compress_lens, candidates=None, index_topk=512, offset=0, traced=False):
    """Score compressed positions and select the top-k best per query.
    Takes q [b, s, H, D] bf16, index_k [b, T, D] bf16, weights [b, s, H] bf16,
    compress_lens (broadcastable to [b, s, 1]), candidates mask [b, s, T] bool (or None),
    index_topk, and offset. Returns int32 indices [b, s, index_topk] sorted ascending.
    Decode rows b*s <= 64."""
    if q.ndim != 4 or index_k.ndim != 3 or weights.ndim != 3:
        raise ValueError(f"index_topk: bad input shapes q {tuple(q.shape)}, index_k {tuple(index_k.shape)}, weights {tuple(weights.shape)}")
    b, s, H, D = q.shape
    b_k, T, D_k = index_k.shape
    if b != b_k or D != D_k:
        raise ValueError(f"index_topk: shape mismatch between q {tuple(q.shape)} and index_k {tuple(index_k.shape)}")
    if weights.shape != (b, s, H):
        raise ValueError(f"index_topk: weights shape {tuple(weights.shape)} != expected {(b, s, H)}")
    n_rows = b * s
    if n_rows > 64:
        raise ValueError(f"index_topk: decode rows b*s must be <= 64, got {n_rows}")

    qc = q.contiguous()
    kc = index_k.contiguous()
    wc = weights.contiguous()

    if compress_lens.ndim == 3 and compress_lens.shape[-1] == 1:
        cl = compress_lens
    else:
        n = compress_lens.numel()
        if n == b * s:
            cl = compress_lens.view(b, s, 1)
        elif n == s:
            cl = compress_lens.view(1, s, 1)
        elif n == b:
            cl = compress_lens.view(b, 1, 1)
        else:
            cl = compress_lens.view(1, 1, 1)
    cl_flat = cl.expand(b, s, 1).reshape(-1).contiguous().to(torch.int32)

    has_cand = candidates is not None
    if has_cand:
        if candidates.shape != (b, s, T):
            raise ValueError(f"index_topk: candidates shape {tuple(candidates.shape)} != expected {(b, s, T)}")
        cand_c = candidates.contiguous()
    else:
        cand_c = qc  # unused dummy

    scores = torch.empty((b, s, T), dtype=torch.float32, device=q.device)
    out_idxs = torch.empty((b, s, index_topk), dtype=torch.int32, device=q.device)

    if n_rows and T:
        BT_SCORE = 64
        BH = max(16, triton.next_power_of_2(H))
        BD = max(16, triton.next_power_of_2(D))
        is_interp = os.environ.get("TRITON_INTERPRET") == "1"

        k1, ctx1 = _launcher(_index_score_kernel, q.device, traced)
        grid1 = (n_rows, triton.cdiv(T, BT_SCORE))
        with ctx1:
            k1[grid1](
                qc, kc, wc, cl_flat, cand_c, scores,
                qc.stride(0), qc.stride(1), qc.stride(2), qc.stride(3),
                kc.stride(0), kc.stride(1), kc.stride(2),
                wc.stride(0), wc.stride(1), wc.stride(2),
                cand_c.stride(0), cand_c.stride(1), cand_c.stride(2),
                scores.stride(0), scores.stride(1), scores.stride(2),
                b, s, T, H=H, D=D,
                HAS_CAND=has_cand, IS_INTERP=is_interp,
                BT=BT_SCORE, BH=BH, BD=BD,
            )

        BK = max(2, triton.next_power_of_2(index_topk))
        BT_TOPK = max(16, triton.next_power_of_2(T), BK)

        k2, ctx2 = _launcher(_index_topk_kernel, q.device, traced)
        grid2 = (n_rows,)
        with ctx2:
            k2[grid2](
                scores, cl_flat, out_idxs, int(offset),
                scores.stride(0), scores.stride(1), scores.stride(2),
                out_idxs.stride(0), out_idxs.stride(1), out_idxs.stride(2),
                b, s, T, index_topk,
                BT=BT_TOPK, BK=BK,
            )

    return out_idxs


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

    @torch.library.triton_op("vxpu_triton::rotary", mutates_args=())
    def _rotary_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return rotary(x, cos, sin, traced=True)

    @torch.library.triton_op("vxpu_triton::index_topk", mutates_args=())
    def _index_topk_op(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
                       compress_lens: torch.Tensor, candidates: torch.Tensor | None,
                       index_topk: int, offset: int = 0) -> torch.Tensor:
        return index_topk(q, index_k, weights, compress_lens, candidates, index_topk, offset, traced=True)


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
    index_topk=lambda q, k, w, cl, cand, topk, off=0: torch.ops.vxpu_triton.index_topk(q, k, w, cl, cand, topk, off),
)


# ---------------------------------------------------------------------------
# routed experts on the GPU (the hot ones; experts_gpu.py)

if triton is not None:
    @triton.jit
    def _e2m1(v):
        """fp4 e2m1 codes (0..15, int32) -> fp32: sign, 2-bit exponent,
        1-bit mantissa; exponent 0 is subnormal (0 or 0.5)."""
        sign = tl.where((v & 8) != 0, -1.0, 1.0)
        e = (v >> 1) & 3
        m = (v & 1).to(tl.float32)
        mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        return sign * mag

    @triton.jit
    def _fp4_rows_dot(x_ptr, w_ptr, s_ptr, rows, live, K, KB, KS, BN: tl.constexpr, BK: tl.constexpr):
        """sum over K of x[k] * W[row, k] for rows of an fp4 bank viewed as
        [S*N, K/2] (rows are absolute: slot * N + row), with e8m0 scales
        [S*N, K/32]; fp32 accumulation -> [BN]. BK divides K."""
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kb = k0 // 2 + tl.arange(0, BK // 2)
            b = tl.load(w_ptr + rows[:, None] * KB + kb[None, :], mask=live[:, None], other=0).to(tl.int32)
            w = tl.reshape(tl.join(_e2m1(b & 15), _e2m1(b >> 4)), (BN, BK))  # low nibble first
            ks = k0 // 32 + tl.arange(0, BK // 32)
            e = tl.load(s_ptr + rows[:, None] * KS + ks[None, :], mask=live[:, None], other=127).to(tl.int32)
            sc = tl.where(e == 0, 5.877471754111438e-39, (e << 23).to(tl.float32, bitcast=True))
            w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
            x = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
            acc += w * x[None, :]
        return tl.sum(acc, axis=1)

    @triton.jit
    def _slot(slot_ptr, idx_ptr, start, p, MAPPED: tl.constexpr):
        """Pair p's slot in the pool: given per pair, or looked up in the
        pool's slot map (slot_ptr) by the pair's expert (idx_ptr)."""
        if MAPPED:
            return tl.load(slot_ptr + (tl.load(idx_ptr + p) - start))
        return tl.load(slot_ptr + p)

    @triton.jit
    def _fp4_gate_up_kernel(x_ptr, slot_ptr, idx_ptr, start, w1_ptr, s1_ptr, w3_ptr, s3_ptr, g_ptr, u_ptr,
                            I, K, TOPK, BN: tl.constexpr, BK: tl.constexpr, MAPPED: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.program_id(1)  # (token, k) pair
        slot = _slot(slot_ptr, idx_ptr, start, p, MAPPED)
        if slot < 0:  # a cold expert: the CPU computes it
            return
        rows = pid * BN + tl.arange(0, BN)
        live = rows < I
        bank_rows = slot.to(tl.int64) * I + rows
        xrow = x_ptr + (p // TOPK).to(tl.int64) * K
        g = _fp4_rows_dot(xrow, w1_ptr, s1_ptr, bank_rows, live, K, K // 2, K // 32, BN, BK)
        u = _fp4_rows_dot(xrow, w3_ptr, s3_ptr, bank_rows, live, K, K // 2, K // 32, BN, BK)
        tl.store(g_ptr + p.to(tl.int64) * I + rows, g.to(tl.bfloat16), mask=live)
        tl.store(u_ptr + p.to(tl.int64) * I + rows, u.to(tl.bfloat16), mask=live)

    @triton.jit
    def _swiglu_quant_kernel(g_ptr, u_ptr, wt_ptr, slot_ptr, idx_ptr, start, h_ptr, I, limit, G: tl.constexpr,
                             MAPPED: tl.constexpr):
        """The fork's _ffn for one pair's 32-blocks: clamps, w * silu(g) * u
        in fp32, rounded to bf16, then act_quant per 32 (fp8 grid, ue8m0
        scales) and dequantized, as _act_quant_dequant_kernel does."""
        p = tl.program_id(1)
        slot = _slot(slot_ptr, idx_ptr, start, p, MAPPED)
        if slot < 0:
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
        y = q.to(tl.float8e4nv).to(tl.float32) * scale[:, None]
        tl.store(h_ptr + offs, y.to(tl.bfloat16), mask=live[:, None])

    @triton.jit
    def _fp4_down_kernel(h_ptr, slot_ptr, idx_ptr, start, w2_ptr, s2_ptr, d_ptr, D, I, BN: tl.constexpr,
                         BK: tl.constexpr, MAPPED: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.program_id(1)
        slot = _slot(slot_ptr, idx_ptr, start, p, MAPPED)
        rows = pid * BN + tl.arange(0, BN)
        live = rows < D
        if slot < 0:  # not held here: its reply is 0 (the output is not zeroed beforehand)
            tl.store(d_ptr + p.to(tl.int64) * D + rows, tl.zeros((BN,), dtype=tl.float32), mask=live)
            return
        bank_rows = slot.to(tl.int64) * D + rows
        d = _fp4_rows_dot(h_ptr + p.to(tl.int64) * I, w2_ptr, s2_ptr, bank_rows, live, I, I // 2, I // 32, BN, BK)
        tl.store(d_ptr + p.to(tl.int64) * D + rows, d.to(tl.bfloat16).to(tl.float32), mask=live)


def _fp4_block_k(K):
    for bk in (512, 256, 128, 64, 32):
        if K % bk == 0:
            return bk
    raise ValueError(f"fp4 experts: K={K} is not a multiple of 32")


def fp4_experts_decode(x_in, weights, slots, w1, s1, w3, s3, w2, s2, swiglu_limit, per_pair=False,
                       indices=None, start=0):
    """The routed experts of a few tokens, for the pairs whose expert is
    held on the GPU: x_in [n, K] bf16 (already on the fp8 grid, as the
    module quantizes it), weights [n, topk] fp32, slots [n, topk] int (the
    expert's slot in the GPU banks, or -1 for experts computed elsewhere),
    banks w1/w3 [S, I, K/2] and w2 [S, K, I/2] packed fp4 with e8m0
    scales per 32 along the inner dimension. The arithmetic is the
    module's _forward_batched per pair: fp32 accumulation rounded to bf16,
    the clamps, SiLU, the routing weight, the down projection's input
    re-quantized per 32 block, its output rounded to bf16. Returns y
    [n, K] fp32, the sum over the token's held pairs (0 where none), or
    with per_pair=True each pair's reply, [n, topk, K] (0 where not held;
    mailboxes.py sums them in a fixed order). With indices ([n, topk]
    expert ids), slots is instead the pool's slot map (one entry per
    expert from `start`), read by the kernels themselves: no lookup
    launched first."""
    n, topk = (indices if indices is not None else slots).shape
    K = x_in.shape[1]
    S, I, _ = w1.shape
    pairs = n * topk
    dev = x_in.device
    mapped = indices is not None
    sl = slots.contiguous().view(-1) if mapped else slots.to(torch.int32).contiguous().view(-1)
    idx = indices.contiguous().view(-1) if mapped else sl
    g = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    u = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    h = torch.empty(pairs, I, dtype=torch.bfloat16, device=dev)
    d = torch.empty(pairs, K, dtype=torch.float32, device=dev)  # the down kernel writes every pair, 0 where not held
    x = x_in.contiguous()
    wt = weights.to(torch.float32).contiguous().view(-1)
    BN = 8
    with torch.cuda.device(dev):
        _fp4_gate_up_kernel[(triton.cdiv(I, BN), pairs)](
            x, sl, idx, start, w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), g, u, I, K, topk, BN=BN,
            BK=_fp4_block_k(K), MAPPED=mapped)
        _swiglu_quant_kernel[(triton.cdiv(I // 32, 16), pairs)](g, u, wt, sl, idx, start, h, I, float(swiglu_limit),
                                                               G=16, MAPPED=mapped)
        _fp4_down_kernel[(triton.cdiv(K, BN), pairs)](
            h, sl, idx, start, w2, s2.view(torch.uint8), d, K, I, BN=BN, BK=_fp4_block_k(I), MAPPED=mapped)
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
