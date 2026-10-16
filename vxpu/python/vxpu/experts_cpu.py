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
kernels in csrc/ instead of a dequantized copy; the nonlinear part
between them (``_ffn``: clamps, SiLU, routing weight, input
requantization) is the module's own method, unchanged. Dequantization
is exact either way, so the two paths differ only in fp32 accumulation
order.

Only fp4 banks with e8m0 scales are handled here; a module whose banks
are something else keeps its own paths.
"""

import torch

from . import kernels


def applicable(moe):
    """Whether the kernels can serve this module's banks."""
    k = kernels.load()
    return (k is not None and moe.w1.dtype == torch.uint8 and moe.w1.device.type == "cpu"
            and moe.w1_scale is not None and moe.w1_scale.dtype == torch.float8_e8m0fnu)


def forward_batched(moe, x_in, weights, indices):
    """Decode routing on the kernels; mirrors MoE._forward_batched."""
    k = kernels.load()
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
    gate = k.fp4_grouped_mm(x_sorted, moe.w1, moe.w1_scale, offs)
    up = k.fp4_grouped_mm(x_sorted, moe.w3, moe.w3_scale, offs)
    h = moe._ffn(gate, up, wgt).contiguous()
    d = k.fp4_grouped_mm(h, moe.w2, moe.w2_scale, offs).float()
    y = torch.zeros(n, moe.dim, dtype=torch.float32, device=x_in.device)
    return y.index_add_(0, token, d)
