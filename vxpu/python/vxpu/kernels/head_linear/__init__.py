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

"""Input builder for vxpu::head_linear."""

# pyright: strict

import torch

SCORE_SHAPES: tuple[str, ...] = ("1x5120->129280 bfloat16",)


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds synthetic inputs (x, weight) for head_linear.

    Stands for the language model output head projection (model.head) in decode
    (rows=1 token, K=5120 hidden dimension, N=129280 vocabulary size) in the
    released model. The released DeepSeek-V4.1 checkpoint stores the head
    weight in bfloat16; the fork widens it to float32 at construction, but vXPU
    preserves the checkpoint's bfloat16 weight to avoid doubling memory traffic.

    Parses shapes in '{rows}x{K}->{N}' or '{rows}x{K}->{N} {dtype}' format.
    """
    parts = shapes.split()
    if len(parts) == 1:
        shape_part, dtype_str = parts[0], "bfloat16"
    elif len(parts) == 2:
        shape_part, dtype_str = parts
    else:
        raise ValueError(f"invalid head_linear shapes {shapes!r}")

    left, n_str = shape_part.split("->")
    r_str, k_str = left.split("x")
    rows, K, N = int(r_str), int(k_str), int(n_str)
    dtype = getattr(torch, dtype_str, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {dtype_str!r} in {shapes!r}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, K), generator=gen, dtype=torch.bfloat16).to(device)
    weight = torch.randn((N, K), generator=gen, dtype=dtype).to(device)

    return x, weight
