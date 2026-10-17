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

"""The routed experts on the CPU with the fused fp4 kernels.

The same routings as the fork's ``MoE._forward_batched`` (decode: gather
the chosen experts) and ``_forward_grouped`` (prefill: tokens sorted by
expert), with the three matmuls over packed fp4 banks done by the
kernels in csrc/ instead of a dequantized copy. In decode the whole
step is one C++ call (fp4_moe_decode), which also does the nonlinear
part between the matmuls (the module's ``_ffn``: clamps, SiLU, routing
weight, input requantization) with the reference's rounding points; in
prefill, and with VXPU_FUSED_EXPERTS=0, ``_ffn`` is the module's own
method. Dequantization is exact either way, so the paths differ only in
fp32 accumulation order.

Only fp4 banks with e8m0 scales are handled here; a module whose banks
are something else keeps its own paths.

Banks held in RAM can be repacked once at load (``repack``) into groups
of 16 rows, the layout the decode kernel reaches memory bandwidth on
(csrc/, "16-row repacked banks"). A repacked module's banks are only
readable by the kernels: the fork's own routings cannot run on it.
"""

import os

import torch

from . import kernels


def applicable(moe):
    """Whether the kernels can serve this module's banks."""
    k = kernels.load()
    return (k is not None and moe.w1.dtype == torch.uint8 and moe.w1.device.type == "cpu"
            and moe.w1_scale is not None and moe.w1_scale.dtype == torch.float8_e8m0fnu)


BANKS = (("w1", "w1_scale"), ("w3", "w3_scale"), ("w2", "w2_scale"))


def repacked(moe):
    """Whether this module's banks are in the 16-row layout."""
    return getattr(moe, "vxpu_repacked16", False)


def repack_by_default():
    """Whether the 16-row decode kernel is the faster one here: it has an
    AVX-512 path; elsewhere its scalar path loses to the row-major NEON
    kernel."""
    k = kernels.load()
    return k is not None and hasattr(k, "fp4_repack16") and k.has_avx512()


def repack(moe):
    """Repack the module's three banks in place into the 16-row layout,
    one bank at a time (the old copy goes as each new one lands). Returns
    False, changing nothing, where the kernels cannot serve the module."""
    k = kernels.load()
    if repacked(moe) or not applicable(moe) or not hasattr(k, "fp4_repack16"):
        return repacked(moe)
    if any(getattr(moe, w).shape[1] % 16 for w, _ in BANKS):
        return False
    for w, sc in BANKS:
        wp, sp = k.fp4_repack16(getattr(moe, w).contiguous(), getattr(moe, sc).contiguous())
        setattr(moe, w, torch.nn.Parameter(wp, requires_grad=False))
        setattr(moe, sc, torch.nn.Parameter(sp, requires_grad=False))
    moe.vxpu_repacked16 = True
    return True


def forward_batched(moe, x_in, weights, indices):
    """Decode routing on the kernels; mirrors MoE._forward_batched.

    One C++ call (fp4_moe_decode) runs the whole step, the fork's _ffn
    included, when the extension has it (fp4_moe_decode16 on repacked
    banks); VXPU_FUSED_EXPERTS=0, which also stops the engine repacking, keeps the
    three kernel calls with the module's own _ffn between them."""
    k = kernels.load()
    if repacked(moe):
        sel = (indices - moe.experts_start_idx).to(torch.int64)
        return k.fp4_moe_decode16(x_in.contiguous(), weights.float().contiguous(), sel.contiguous(),
                                  moe.w1, moe.w1_scale, moe.w3, moe.w3_scale, moe.w2, moe.w2_scale,
                                  float(moe.swiglu_limit))
    if hasattr(k, "fp4_moe_decode") and os.environ.get("VXPU_FUSED_EXPERTS", "1").lower() not in ("0", "off", "false"):
        sel = (indices - moe.experts_start_idx).to(torch.int64)
        return k.fp4_moe_decode(x_in.contiguous(), weights.float().contiguous(), sel.contiguous(),
                                moe.w1, moe.w1_scale, moe.w3, moe.w3_scale, moe.w2, moe.w2_scale,
                                float(moe.swiglu_limit))
    n, topk = indices.shape
    sel = (indices - moe.experts_start_idx).flatten().to(torch.int64)
    xi = x_in.repeat_interleave(topk, dim=0).contiguous()  # [n * topk, dim]
    gate = k.fp4_gather_mm(xi, moe.w1, moe.w1_scale, sel)
    up = k.fp4_gather_mm(xi, moe.w3, moe.w3_scale, sel)
    h = moe._ffn(gate, up, weights.flatten()).contiguous()
    d = k.fp4_gather_mm(h, moe.w2, moe.w2_scale, sel).float()  # [n * topk, dim]
    y = torch.zeros(n, moe.dim, dtype=torch.float32, device=x_in.device)
    return y.index_add_(0, torch.arange(n, device=x_in.device).repeat_interleave(topk), d)


def forward_grouped(moe, x_in, weights, indices):
    """Prefill routing on the kernels; mirrors MoE._forward_grouped
    (every expert at once: the kernel decodes each expert once and runs
    torch's GEMM over its tokens, so no chunking is needed)."""
    k = kernels.load()
    n, topk = indices.shape
    E = moe.n_local_experts
    flat = (indices - moe.experts_start_idx).flatten()
    order = flat.argsort(stable=True)
    token = order // topk
    wgt = weights.flatten()[order]
    counts = torch.zeros(E, dtype=torch.int64, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    offs = counts.cumsum(0).to(torch.int32)
    x_sorted = x_in[token].contiguous()  # [n * topk, dim]
    mm = k.fp4_grouped_mm16 if repacked(moe) else k.fp4_grouped_mm
    gate = mm(x_sorted, moe.w1, moe.w1_scale, offs)
    up = mm(x_sorted, moe.w3, moe.w3_scale, offs)
    h = moe._ffn(gate, up, wgt).contiguous()
    d = mm(h, moe.w2, moe.w2_scale, offs).float()
    y = torch.zeros(n, moe.dim, dtype=torch.float32, device=x_in.device)
    return y.index_add_(0, token, d)
