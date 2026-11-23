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

"""Input builder for vxpu::rotary."""

# pyright: strict

import torch

SCORE_SHAPES: tuple[str, ...] = ("1x1x64x64",)


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Builds synthetic inputs (x, freqs_cis, inverse) for rotary.

    Stands for the attention query rotary embedding (q[..., -rd:] in
    Attention.forward) in decode (b=1, s=1 token, h=64 attention heads, d=64
    rope head dimension) in the released model.

    Parses shapes in '{b}x{s}x{h}x{d}' or '{b}x{s}x{d}' format, optionally with
    trailing 'inv' and/or 'per_row' modifiers.
    """
    inv = False
    per_row = False
    parts = shapes.split()
    if not parts:
        raise ValueError(f"rotary shapes cannot be empty, got {shapes!r}")
    core = parts[0]
    for modifier in parts[1:]:
        if modifier == "inv":
            inv = True
        elif modifier in ("per_row", "per-row"):
            per_row = True
        else:
            raise ValueError(f"unrecognized rotary shape modifier {modifier!r} in {shapes!r}")

    dims = tuple(int(x) for x in core.split("x"))
    if len(dims) not in (3, 4):
        raise ValueError(f"rotary shapes must have rank 3 or 4, got {shapes!r}")
    b = dims[0]
    s = dims[1]
    d = dims[-1]
    if d % 2 != 0:
        raise ValueError(f"rotary feature dimension ({d}) must be even")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(dims, generator=gen, dtype=torch.bfloat16).to(device)
    angle_shape = (b, s, d // 2) if per_row else (s, d // 2)
    angles = torch.randn(angle_shape, generator=gen, dtype=torch.float32).to(device)
    freqs_cis = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)

    return x, freqs_cis, inv
