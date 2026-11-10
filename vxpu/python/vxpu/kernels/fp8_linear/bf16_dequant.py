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

"""Variant bf16_dequant of vxpu::fp8_linear."""

# pyright: strict

from typing import Any

import torch
import torch.nn.functional as F

from .. import variant


@variant(
    devices=("cpu", "cuda"),
    numerics="rounding",
    when="torch's GEMM over the weight dequantized to bf16 for this call (exact), "
         "the activation quantized by the fork's act_quant",
)
def bf16_dequant(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Computes fp8 linear by dequantizing weights to bf16 inline and running torch GEMM."""
    from ... import ops

    ops_mod: Any = ops
    ds, ka = ops_mod._fork_kernels()
    K = x.shape[-1]
    xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
    xq_t: torch.Tensor = xq.to(torch.bfloat16).unflatten(-1, (K // 32, 32))
    xs_t: torch.Tensor = xs.to(torch.bfloat16).unsqueeze(-1)
    xf: torch.Tensor = (xq_t * xs_t).flatten(-2)
    N = weight.shape[0]
    w: torch.Tensor = (weight.to(torch.bfloat16).view(N // 32, 32, K // 32, 32)
                       * scale.to(torch.bfloat16).view(N // 32, 1, K // 32, 1)).view(N, K)
    return F.linear(xf, w).to(x.dtype)
