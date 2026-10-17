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
        g = pid * G + tl.arange(0, G)
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


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """The fork's hc_split_sinkhorn for CUDA tensors, one kernel per call:
    mixes [..., (2 + hc) * hc] -> pre [..., hc], post [..., hc],
    comb [..., hc, hc] (fp32), comb made doubly stochastic."""
    hc = hc_mult
    lead = mixes.shape[:-1]
    m = mixes.float().reshape(-1, (2 + hc) * hc).contiguous()
    T = m.shape[0]
    pre = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    post = torch.empty(T, hc, dtype=torch.float32, device=mixes.device)
    comb = torch.empty(T, hc, hc, dtype=torch.float32, device=mixes.device)
    if T:
        _hc_sinkhorn_kernel[(T,)](m, hc_scale.float().contiguous(), hc_base.float().contiguous(),
                                  pre, post, comb, float(eps), HC=hc, ITERS=sinkhorn_iters)
    return pre.view(*lead, hc), post.view(*lead, hc), comb.view(*lead, hc, hc)
