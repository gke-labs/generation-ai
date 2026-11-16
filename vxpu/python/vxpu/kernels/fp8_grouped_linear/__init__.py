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

"""Input builder for vxpu::fp8_grouped_linear."""

# pyright: strict

import torch

SCORE_SHAPES: tuple[str, ...] = ("1x8x4096->8x1024",)


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds synthetic inputs (o, weight, scale) for fp8_grouped_linear.

    Stands for the attention output projection (wo_a) in decode (rows=1, g=8
    groups, d=4096 input features per group, r=1024 projection rank per group)
    in the released model.

    Parses shapes in '{rows}x{g}x{d}->{g}x{r}' format.
    """
    left, right = shapes.split("->")
    r_str, g_str, d_str = left.split("x")
    g2_str, rank_str = right.split("x")
    rows, g, d, r = int(r_str), int(g_str), int(d_str), int(rank_str)
    if g != int(g2_str):
        raise ValueError(f"group count mismatch: {g} vs {g2_str}")
    if d % 32 != 0:
        raise ValueError(f"d ({d}) must be a multiple of 32")
    if (g * r) % 32 != 0:
        raise ValueError(f"total rows ({g * r}) must be a multiple of 32")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    o = torch.randn((1, rows, g, d), generator=gen, dtype=torch.bfloat16).to(device)
    w_fp32 = torch.randn((g * r, d), generator=gen, dtype=torch.float32)
    weight = w_fp32.to(torch.float8_e4m3fn).to(device)

    exp = torch.randint(-2, 3, ((g * r) // 32, d // 32), generator=gen).float()
    scale = (2.0 ** exp).to(torch.float8_e8m0fnu).to(device)

    return o, weight, scale
