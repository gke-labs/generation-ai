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
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton: the aten path remains
    triton = None


def available():
    return triton is not None and torch.cuda.is_available()


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


def act_quant_dequant(x):
    """The fork's act_quant (32-element blocks, ue8m0 scales) followed by
    the dequantization, for a CUDA tensor whose last dim is a multiple
    of 32: bf16 values on the fp8 grid times each block's scale."""
    assert x.is_cuda and x.shape[-1] % 32 == 0
    xc = x.contiguous()
    y = torch.empty(xc.shape, dtype=torch.bfloat16, device=x.device)
    n_groups = xc.numel() // 32
    G = 16
    # Triton launches on the current device; make that the tensors' device
    # (an engine on cuda:1 with the process's current device still 0)
    with torch.cuda.device(x.device):
        _act_quant_dequant_kernel[(triton.cdiv(n_groups, G),)](xc, y, n_groups, G=G)
    return y


if triton is not None:
    @triton.jit
    def _hc_sinkhorn_kernel(m_ptr, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr, eps,
                            HC: tl.constexpr, ITERS: tl.constexpr):
        t = tl.program_id(0)
        W: tl.constexpr = (2 + HC) * HC
        r = tl.arange(0, HC)
        row = m_ptr + t * W
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        pre = tl.sigmoid(tl.load(row + r) * s0 + tl.load(base_ptr + r)) + eps
        post = 2 * tl.sigmoid(tl.load(row + HC + r) * s1 + tl.load(base_ptr + HC + r))
        ij = r[:, None] * HC + r[None, :]
        c = tl.load(row + 2 * HC + ij) * s2 + tl.load(base_ptr + 2 * HC + ij)
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


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """The fork's hc_split_sinkhorn for CUDA tensors, one kernel per call:
    mixes [..., (2 + hc) * hc] -> pre [..., hc], post [..., hc],
    comb [..., hc, hc] (fp32), comb made doubly stochastic. hc must be a
    power of two (hc_split_sinkhorn_supported)."""
    hc = hc_mult
    lead = mixes.shape[:-1]
    m = mixes.float().reshape(-1, (2 + hc) * hc).contiguous()
    T = m.shape[0]
    pre = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    post = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    comb = torch.empty(T, hc, hc, dtype=torch.float32, device=mixes.device)
    if T:
        with torch.cuda.device(mixes.device):  # Triton launches on the current device
            _hc_sinkhorn_kernel[(T,)](m, hc_scale.float().contiguous(), hc_base.float().contiguous(),
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
            x = tl.load(x_ptr + m * K + ks, mask=kin, other=0.0)
            acc += w.to(tl.float32) * x[None, :]
        tl.store(y_ptr + m * N + rows, tl.sum(acc, axis=1), mask=live)


def linear_f32(x, weight, block_n=16, block_k=512):
    """x [M, K] (any float dtype, taken as fp32) times weight [N, K]
    (bf16, read as is) transposed -> fp32 [M, N], fp32 accumulation. For
    the few rows of a decode step: each row reads the whole weight."""
    M, K = x.shape
    N = weight.shape[0]
    if weight.dim() != 2 or weight.shape[1] != K:
        raise ValueError(f"linear_f32: x {tuple(x.shape)} and weight {tuple(weight.shape)} disagree on K")
    x = x.float().contiguous()
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    if M and N:
        with torch.cuda.device(x.device):  # Triton launches on the current device
            _linear_f32_kernel[(triton.cdiv(N, block_n), M)](x, weight.contiguous(), y, N, K, BN=block_n, BK=block_k)
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


def fp8_gemv(x, weight, scale, group_rows=None, out_dtype=torch.bfloat16, block_n=None, block_k=None):
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
        with torch.cuda.device(x.device):  # Triton launches on the current device
            _fp8_gemv_kernel[(triton.cdiv(N, block_n), M)](
                x.contiguous(), weight.contiguous(), scale.contiguous().view(torch.uint8), y, N, K,
                x.shape[1], R, scale.shape[1], BN=block_n, BK=block_k)
    return y
