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
"""


import torch

from . import kernels


def applicable(moe):
    """Whether the kernels can serve this module's banks."""
    k = kernels.load()
    if k is None:
        return False
    table = getattr(moe, "expert_table", None)
    if table is not None:
        return (table.w1_dtype == torch.uint8 and table.device.type == "cpu"
                and table.w1_scale_dtype == torch.float8_e8m0fnu)
    return (hasattr(moe, "w1") and moe.w1.dtype == torch.uint8 and moe.w1.device.type == "cpu"
            and getattr(moe, "w1_scale", None) is not None and moe.w1_scale.dtype == torch.float8_e8m0fnu)


def forward_batched(moe, x_in, weights, indices, fused=True):
    """Decode routing on the kernels; mirrors MoE._forward_batched.

    One C++ call (fp4_moe_decode) runs the whole step, the fork's _ffn
    included, when the extension has it; fused=False (the plan's
    fused_experts choice, VXPU_FUSED_EXPERTS=0) keeps the three kernel
    calls with the module's own _ffn between them."""
    k = kernels.load()
    table = getattr(moe, "expert_table", None)
    w1 = table.ptrs("w1") if table is not None else moe.w1
    s1 = table.ptrs("w1_scale") if table is not None else moe.w1_scale
    w3 = table.ptrs("w3") if table is not None else moe.w3
    s3 = table.ptrs("w3_scale") if table is not None else moe.w3_scale
    w2 = table.ptrs("w2") if table is not None else moe.w2
    s2 = table.ptrs("w2_scale") if table is not None else moe.w2_scale
    I = table.intermediate_size if table is not None else 0

    if fused and hasattr(k, "fp4_moe_decode"):
        sel = (indices - moe.experts_start_idx).to(torch.int64)
        return k.fp4_moe_decode(x_in.contiguous(), weights.float().contiguous(), sel.contiguous(),
                                w1, s1, w3, s3, w2, s2, float(moe.swiglu_limit), I)
    n, topk = indices.shape
    sel = (indices - moe.experts_start_idx).flatten().to(torch.int64)
    xi = x_in.repeat_interleave(topk, dim=0).contiguous()  # [n * topk, dim]
    gate = k.fp4_gather_mm(xi, w1, s1, sel, I)
    up = k.fp4_gather_mm(xi, w3, s3, sel, I)
    h = moe._ffn(gate, up, weights.flatten()).contiguous()
    d = k.fp4_gather_mm(h, w2, s2, sel, table.dim if table is not None else 0).float()  # [n * topk, dim]
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
    table = getattr(moe, "expert_table", None)
    w1 = table.ptrs("w1") if table is not None else moe.w1
    s1 = table.ptrs("w1_scale") if table is not None else moe.w1_scale
    w3 = table.ptrs("w3") if table is not None else moe.w3
    s3 = table.ptrs("w3_scale") if table is not None else moe.w3_scale
    w2 = table.ptrs("w2") if table is not None else moe.w2
    s2 = table.ptrs("w2_scale") if table is not None else moe.w2_scale
    I = table.intermediate_size if table is not None else 0

    gate = k.fp4_grouped_mm(x_sorted, w1, s1, offs, I)
    up = k.fp4_grouped_mm(x_sorted, w3, s3, offs, I)
    h = moe._ffn(gate, up, wgt).contiguous()
    d = k.fp4_grouped_mm(h, w2, s2, offs, table.dim if table is not None else 0).float()
    y = torch.zeros(n, moe.dim, dtype=torch.float32, device=x_in.device)
    return y.index_add_(0, token, d)
