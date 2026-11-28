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

"""Input builder and packing helpers for vxpu::int4_linear."""

import torch

DEFAULT_GROUP_SIZE: int = 128

SCORE_SHAPES: tuple[str, ...] = ("1x4096->14336",)


def pack_int4(weight_q: torch.Tensor, zero_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs unsigned 4-bit integer weights and zero points into int32 tensors.

    Accepts weight_q [K, N] and zero_q [n_groups, N] with integer values in 0..15.
    Packs 8 consecutive K values per int32 into qweight [K // 8, N] and 8 consecutive
    N values per int32 into qzeros [n_groups, N // 8] in little-endian nibble order.
    """
    if weight_q.ndim != 2 or zero_q.ndim != 2:
        raise ValueError(f"pack_int4 expects rank-2 tensors, got {tuple(weight_q.shape)} and {tuple(zero_q.shape)}")
    K, N = int(weight_q.shape[0]), int(weight_q.shape[1])
    n_groups, N_z = int(zero_q.shape[0]), int(zero_q.shape[1])
    if K % 8 != 0 or N % 8 != 0 or N_z != N:
        raise ValueError(f"pack_int4 requires K ({K}) and N ({N}) divisible by 8 and matching N ({N_z})")
    if n_groups <= 0 or K % n_groups != 0 or (K // n_groups) % 8 != 0:
        raise ValueError(f"pack_int4 requires n_groups ({n_groups}) to divide K ({K}) with group_size divisible by 8")

    shifts = torch.arange(0, 32, 4, device=weight_q.device, dtype=torch.int32)
    w_nibbles = (weight_q.to(torch.int32) & 0xF).reshape(K // 8, 8, N)
    qweight = torch.sum(w_nibbles << shifts.view(1, 8, 1), dim=1, dtype=torch.int32)

    z_nibbles = (zero_q.to(torch.int32) & 0xF).reshape(n_groups, N // 8, 8)
    qzeros = torch.sum(z_nibbles << shifts.view(1, 1, 8), dim=2, dtype=torch.int32)
    return qweight.contiguous(), qzeros.contiguous()


def inputs(
    shapes: str,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds synthetic inputs (x, qweight, scales, qzeros) for int4_linear.

    Parses shapes in '{rows}x{K}->{N}' format with optional whitespace-separated
    modifiers 'g={group_size}' (default 128) and a torch dtype name such as
    'bfloat16' or 'float16' (default bfloat16). K must be a multiple of group_size,
    and both group_size and N must be multiples of 8.
    """
    tokens = shapes.strip().split()
    if not tokens:
        raise ValueError(f"empty int4_linear shapes {shapes!r}")

    shape_part = tokens[0]
    group_size = DEFAULT_GROUP_SIZE
    dtype: torch.dtype = torch.bfloat16

    for tok in tokens[1:]:
        if tok.startswith("g="):
            group_size = int(tok[2:])
        else:
            parsed = getattr(torch, tok, None)
            if not isinstance(parsed, torch.dtype):
                raise ValueError(f"unknown modifier {tok!r} in int4_linear shapes {shapes!r}")
            dtype = parsed

    if "->" not in shape_part:
        raise ValueError(f"invalid int4_linear shapes {shapes!r}; expected '{{rows}}x{{K}}->{{N}}'")
    left, n_str = shape_part.split("->", 1)
    if "x" not in left:
        raise ValueError(f"invalid int4_linear shapes {shapes!r}; expected '{{rows}}x{{K}}->{{N}}'")
    r_str, k_str = left.split("x", 1)
    rows, K, N = int(r_str), int(k_str), int(n_str)

    if rows <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"rows ({rows}), K ({K}), and N ({N}) must be positive")
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size ({group_size}) must be a positive multiple of 8")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be a multiple of group_size ({group_size})")
    if N % 8 != 0:
        raise ValueError(f"N ({N}) must be a multiple of 8")

    n_groups = K // group_size
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, K), generator=gen, dtype=dtype).to(device)
    qweight = torch.randint(-(2**31), 2**31, (K // 8, N), generator=gen, dtype=torch.int32).to(device)
    qzeros = torch.randint(-(2**31), 2**31, (n_groups, N // 8), generator=gen, dtype=torch.int32).to(device)
    scales = torch.empty((n_groups, N), dtype=torch.float32).uniform_(0.005, 0.05, generator=gen).to(dtype).to(device)
    return x, qweight, scales, qzeros
