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

The fork's ``MoE._forward_batched`` routing (every token's chosen experts),
with the three matmuls over packed fp4 banks done by the kernels in csrc/
instead of a dequantized copy, for any number of tokens: a decode step and
a prefill alike. The whole layer is one C++ call (fp4_moe_decode), which
groups each expert's tokens, works in tiles, and also does the nonlinear
part between the matmuls (the module's ``_ffn``: clamps, SiLU, routing
weight, input requantization) with the reference's rounding points; with
VXPU_FUSED_EXPERTS=0, ``_ffn`` is the module's own method. Dequantization
is exact either way, so the paths differ only in fp32 accumulation order.

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


def forward_batched(moe, x_in, weights, indices, fused=True):
    """Decode routing on the kernels; mirrors MoE._forward_batched.

    One C++ call (fp4_moe_decode) runs the whole step, the fork's _ffn
    included, when the extension has it; fused=False (the plan's
    fused_experts choice, VXPU_FUSED_EXPERTS=0) keeps the three kernel
    calls with the module's own _ffn between them."""
    k = kernels.load()
    if fused and hasattr(k, "fp4_moe_decode"):
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
