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

"""Input builder for vxpu::fp8_linear."""

# pyright: strict

import torch

SCORE_SHAPES: tuple[str, ...] = ("1x6144->25600",)


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds synthetic inputs (x, weight, scale) for fp8_linear from a shape string.

    Parses shapes in '{rows}x{K}->{N}' format. K and N must be multiples of 32.
    """
    left, n_str = shapes.split("->")
    r_str, k_str = left.split("x")
    rows, K, N = int(r_str), int(k_str), int(n_str)
    if K % 32 != 0:
        raise ValueError(f"K ({K}) must be a multiple of 32")
    if N % 32 != 0:
        raise ValueError(f"N ({N}) must be a multiple of 32")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, K), generator=gen, dtype=torch.bfloat16).to(device)
    w_fp32 = torch.randn((N, K), generator=gen, dtype=torch.float32)
    weight = w_fp32.to(torch.float8_e4m3fn).to(device)

    # Scale in e8m0 [N/32, K/32] from powers of two near one.
    exp = torch.randint(-2, 3, (N // 32, K // 32), generator=gen).float()
    scale = (2.0 ** exp).to(torch.float8_e8m0fnu).to(device)

    return x, weight, scale
