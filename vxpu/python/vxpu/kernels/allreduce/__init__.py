# pyright: strict
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

"""Kernel definitions and measurement inputs for allreduce."""

import torch

SCORE_SHAPES: tuple[str, ...] = ("4x1x5120", "8x3x7_f32")


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor]:
    """Generates synthetic input buffers for allreduce benchmarking and checks.

    The shapes string specifies dimensions as "{ranks}x{rows}x{dim}" or "{ranks}x{dim}".
    An optional suffix "_f32" or "_bf16" selects torch.float32 or torch.bfloat16 (default: bfloat16).
    """
    dtype = torch.bfloat16
    clean_shapes = shapes
    if "_f32" in shapes:
        dtype = torch.float32
        clean_shapes = shapes.replace("_f32", "")
    elif "_bf16" in shapes:
        dtype = torch.bfloat16
        clean_shapes = shapes.replace("_bf16", "")

    parts = [int(p) for p in clean_shapes.split("x")]
    if len(parts) == 3:
        ranks, rows, dim = parts
        shape = (ranks, rows, dim)
    elif len(parts) == 2:
        ranks, dim = parts
        shape = (ranks, dim)
    elif len(parts) == 1:
        shape = (4, 1, parts[0])
    else:
        shape = tuple(parts)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    buffers = torch.randn(shape, generator=gen, dtype=dtype).to(device)
    return (buffers,)
