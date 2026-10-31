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
    return (table is not None and table.w1_dtype == torch.uint8 and table.device.type == "cpu"
            and table.w1_scale_dtype == torch.float8_e8m0fnu)


def forward_batched(moe, x_in, weights, indices, fused=True):
    """Decode routing on the kernels; mirrors MoE._forward_batched.

    One C++ call (fp4_moe_decode) runs the whole step, the fork's _ffn
    included, when the extension has it; fused=False (the plan's
    fused_experts choice, VXPU_FUSED_EXPERTS=0) keeps the three kernel
    calls with the module's own _ffn between them."""
    k = kernels.load()
    table = moe.expert_table
    w1, s1, w3, s3, w2, s2 = table.kernel_args()
    I, dim = table.intermediate_size, table.dim

    if fused and hasattr(k, "fp4_moe_decode"):
        sel = (indices - moe.experts_start_idx).to(torch.int64)
        return k.fp4_moe_decode(x_in.contiguous(), weights.float().contiguous(), sel.contiguous(),
                                w1, s1, w3, s3, w2, s2, float(moe.swiglu_limit), I, dim)
    n, topk = indices.shape
    sel = (indices - moe.experts_start_idx).flatten().to(torch.int64)
    xi = x_in.repeat_interleave(topk, dim=0).contiguous()  # [n * topk, dim]
    gate = k.fp4_gather_mm(xi, w1, s1, sel, I, dim)
    up = k.fp4_gather_mm(xi, w3, s3, sel, I, dim)
    h = moe._ffn(gate, up, weights.flatten()).contiguous()
    d = k.fp4_gather_mm(h, w2, s2, sel, dim, I).float()  # [n * topk, dim]
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
    table = moe.expert_table
    w1, s1, w3, s3, w2, s2 = table.kernel_args()
    I, dim = table.intermediate_size, table.dim

    gate = k.fp4_grouped_mm(x_sorted, w1, s1, offs, I, dim)
    up = k.fp4_grouped_mm(x_sorted, w3, s3, offs, I, dim)
    h = moe._ffn(gate, up, wgt).contiguous()
    d = k.fp4_grouped_mm(h, w2, s2, offs, dim, I).float()
    y = torch.zeros(n, moe.dim, dtype=torch.float32, device=x_in.device)
    return y.index_add_(0, token, d)
