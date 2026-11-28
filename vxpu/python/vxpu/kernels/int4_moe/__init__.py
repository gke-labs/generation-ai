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

"""Input builder, argument validation, packing helpers, and active-bytes accounting for vxpu::int4_moe."""

from typing import Any

import torch

from ..int4_linear import DEFAULT_GROUP_SIZE, pack_int4

SCORE_SHAPES: tuple[str, ...] = ("1x2048->768 e=128 k=8",)


def validate_int4_moe_args(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_qzeros: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_qzeros: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    down_qzeros: torch.Tensor,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Validates tensor ranks, dtypes, and dimensions for vxpu::int4_moe.

    Returns (E, K, I, top_k, n_groups_k, group_size_k, n_groups_i, group_size_i).
    """
    if x.ndim < 1:
        raise ValueError(f"int4_moe: x must have rank >= 1, got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError(f"int4_moe: x must be floating point, got {x.dtype}")
    if topk_ids.ndim != x.ndim or topk_weights.ndim != x.ndim:
        raise ValueError(
            f"int4_moe: topk_ids {tuple(topk_ids.shape)} and topk_weights {tuple(topk_weights.shape)} "
            f"must match x rank {x.ndim}"
        )
    if topk_ids.shape[:-1] != x.shape[:-1] or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"int4_moe: topk_ids {tuple(topk_ids.shape)} and topk_weights {tuple(topk_weights.shape)} "
            f"must match x leading dims {tuple(x.shape[:-1])}"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"int4_moe: topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if not topk_weights.is_floating_point():
        raise ValueError(f"int4_moe: topk_weights must be floating point, got {topk_weights.dtype}")

    if (
        gate_qweight.ndim != 3
        or gate_qzeros.ndim != 3
        or up_qweight.ndim != 3
        or up_qzeros.ndim != 3
        or down_qweight.ndim != 3
        or down_qzeros.ndim != 3
        or gate_scales.ndim != 3
        or up_scales.ndim != 3
        or down_scales.ndim != 3
    ):
        raise ValueError("int4_moe: all expert weight, scale, and zero tensors must be rank-3")
    if (
        gate_qweight.dtype != torch.int32
        or gate_qzeros.dtype != torch.int32
        or up_qweight.dtype != torch.int32
        or up_qzeros.dtype != torch.int32
        or down_qweight.dtype != torch.int32
        or down_qzeros.dtype != torch.int32
    ):
        raise ValueError("int4_moe: all packed qweight and qzeros tensors must be int32")
    if (
        not gate_scales.is_floating_point()
        or not up_scales.is_floating_point()
        or not down_scales.is_floating_point()
    ):
        raise ValueError("int4_moe: all scales tensors must be floating point")

    E = gate_scales.shape[0]
    K = x.shape[-1]
    I = gate_scales.shape[2]
    top_k = topk_ids.shape[-1]
    n_groups_k = gate_scales.shape[1]
    n_groups_i = down_scales.shape[1]

    if E <= 0 or top_k <= 0:
        raise ValueError(f"int4_moe: E ({E}) and top_k ({top_k}) must be positive")
    if K <= 0 or I <= 0 or K % 8 != 0 or I % 8 != 0:
        raise ValueError(f"int4_moe: K ({K}) and I ({I}) must be positive multiples of 8")
    if n_groups_k <= 0 or K % n_groups_k != 0:
        raise ValueError(f"int4_moe: gate_scales groups ({n_groups_k}) must divide K ({K})")
    group_size_k = K // n_groups_k
    if group_size_k % 8 != 0:
        raise ValueError(f"int4_moe: gate/up group_size ({group_size_k}) must be a multiple of 8")
    if n_groups_i <= 0 or I % n_groups_i != 0:
        raise ValueError(f"int4_moe: down_scales groups ({n_groups_i}) must divide I ({I})")
    group_size_i = I // n_groups_i
    if group_size_i % 8 != 0:
        raise ValueError(f"int4_moe: down group_size ({group_size_i}) must be a multiple of 8")

    k_pack = K // 8
    i_pack = I // 8
    if (
        gate_qweight.shape != (E, k_pack, I)
        or gate_qzeros.shape != (E, n_groups_k, i_pack)
        or up_qweight.shape != (E, k_pack, I)
        or up_scales.shape != (E, n_groups_k, I)
        or up_qzeros.shape != (E, n_groups_k, i_pack)
        or down_qweight.shape != (E, i_pack, K)
        or down_scales.shape != (E, n_groups_i, K)
        or down_qzeros.shape != (E, n_groups_i, k_pack)
    ):
        raise ValueError(
            f"int4_moe: expert tensor shapes do not match E={E}, K={K}, I={I}, "
            f"n_groups_k={n_groups_k}, n_groups_i={n_groups_i}"
        )

    return E, K, I, top_k, n_groups_k, group_size_k, n_groups_i, group_size_i


def pack_int4_experts(weight_q: torch.Tensor, zero_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs stacked unsigned 4-bit expert weights and zero points into int32 tensors.

    Accepts weight_q [E, K, N] and zero_q [E, n_groups, N] with integer values in 0..15.
    Reuses the exact vxpu::int4_linear packed layout along each expert slice, returning
    qweight [E, K // 8, N] and qzeros [E, n_groups, N // 8] in little-endian nibble order.
    """
    if weight_q.ndim != 3 or zero_q.ndim != 3:
        raise ValueError(
            f"pack_int4_experts expects rank-3 tensors, got {tuple(weight_q.shape)} and {tuple(zero_q.shape)}"
        )
    E = int(weight_q.shape[0])
    if E <= 0 or int(zero_q.shape[0]) != E:
        raise ValueError(f"pack_int4_experts requires matching positive E, got {weight_q.shape[0]} and {zero_q.shape[0]}")

    qw_list: list[torch.Tensor] = []
    qz_list: list[torch.Tensor] = []
    for e in range(E):
        qw_e, qz_e = pack_int4(weight_q[e], zero_q[e])
        qw_list.append(qw_e)
        qz_list.append(qz_e)
    return torch.stack(qw_list, dim=0).contiguous(), torch.stack(qz_list, dim=0).contiguous()


def active_bytes(args: tuple[Any, ...], out: Any) -> int:
    """Computes HBM bytes read and written by active experts and activations for vxpu::int4_moe.

    Counts input activations, routing indices and weights, output activations, and
    the packed weights, scales, and zero points of the unique active experts selected
    in topk_ids rather than all E experts in the bank.
    """
    (
        x,
        topk_ids,
        topk_weights,
        gate_qweight,
        gate_scales,
        gate_qzeros,
        up_qweight,
        up_scales,
        up_qzeros,
        down_qweight,
        down_scales,
        down_qzeros,
    ) = args
    if topk_ids.numel() > 0:
        torch_mod: Any = torch
        unique_ids: torch.Tensor = torch_mod.unique(topk_ids)
        n_active = int(unique_ids.numel())
    else:
        n_active = 0
    num_experts = int(gate_scales.shape[0])

    expert_tensors = (
        gate_qweight,
        gate_scales,
        gate_qzeros,
        up_qweight,
        up_scales,
        up_qzeros,
        down_qweight,
        down_scales,
        down_qzeros,
    )
    per_expert_bytes = sum((t.numel() // num_experts) * t.element_size() for t in expert_tensors)
    act_bytes = (
        x.numel() * x.element_size()
        + topk_ids.numel() * topk_ids.element_size()
        + topk_weights.numel() * topk_weights.element_size()
    )
    if isinstance(out, torch.Tensor):
        act_bytes += out.numel() * out.element_size()
    return act_bytes + n_active * per_expert_bytes


def inputs(
    shapes: str,
    device: str,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Builds synthetic inputs for vxpu::int4_moe.

    Parses shapes in '{rows}x{K}->{I} e={E} k={top_k}' format with optional
    whitespace-separated modifiers 'g={group_size}' (default 128) and a torch dtype
    name such as 'bfloat16' or 'float16' (default bfloat16). Both K and I must be
    positive multiples of group_size, and group_size must be a multiple of 8.
    """
    tokens = shapes.strip().split()
    if not tokens:
        raise ValueError(f"empty int4_moe shapes {shapes!r}")

    shape_part = tokens[0]
    num_experts: int | None = None
    top_k: int | None = None
    group_size = DEFAULT_GROUP_SIZE
    dtype: torch.dtype = torch.bfloat16

    for tok in tokens[1:]:
        if tok.startswith("e="):
            num_experts = int(tok[2:])
        elif tok.startswith("k="):
            top_k = int(tok[2:])
        elif tok.startswith("g="):
            group_size = int(tok[2:])
        else:
            parsed = getattr(torch, tok, None)
            if not isinstance(parsed, torch.dtype):
                raise ValueError(f"unknown modifier {tok!r} in int4_moe shapes {shapes!r}")
            dtype = parsed

    if "->" not in shape_part:
        raise ValueError(f"invalid int4_moe shapes {shapes!r}; expected '{{rows}}x{{K}}->{{I}} e={{E}} k={{top_k}}'")
    left, i_str = shape_part.split("->", 1)
    if "x" not in left:
        raise ValueError(f"invalid int4_moe shapes {shapes!r}; expected '{{rows}}x{{K}}->{{I}} e={{E}} k={{top_k}}'")
    r_str, k_str = left.split("x", 1)
    rows, K, I = int(r_str), int(k_str), int(i_str)

    if num_experts is None or top_k is None:
        raise ValueError(f"int4_moe shapes {shapes!r} must specify both e={{E}} and k={{top_k}}")
    if rows <= 0 or K <= 0 or I <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError(f"rows ({rows}), K ({K}), I ({I}), E ({num_experts}), and top_k ({top_k}) must be positive")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must not exceed E ({num_experts})")
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size ({group_size}) must be a positive multiple of 8")
    if K % group_size != 0 or I % group_size != 0:
        raise ValueError(f"K ({K}) and I ({I}) must be multiples of group_size ({group_size})")

    n_groups_k = K // group_size
    n_groups_i = I // group_size
    gen = torch.Generator(device="cpu").manual_seed(seed)

    x = torch.randn((rows, K), generator=gen, dtype=dtype).to(device)
    router_logits = torch.rand((rows, num_experts), generator=gen, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1).to(device)
    topk_ids = topk_ids.to(torch.int32).to(device)

    gate_qweight = torch.randint(-(2**31), 2**31, (num_experts, K // 8, I), generator=gen, dtype=torch.int32).to(device)
    gate_qzeros = torch.randint(-(2**31), 2**31, (num_experts, n_groups_k, I // 8), generator=gen, dtype=torch.int32).to(device)
    gate_scales = (
        torch.empty((num_experts, n_groups_k, I), dtype=torch.float32)
        .uniform_(0.005, 0.05, generator=gen)
        .to(dtype)
        .to(device)
    )

    up_qweight = torch.randint(-(2**31), 2**31, (num_experts, K // 8, I), generator=gen, dtype=torch.int32).to(device)
    up_qzeros = torch.randint(-(2**31), 2**31, (num_experts, n_groups_k, I // 8), generator=gen, dtype=torch.int32).to(device)
    up_scales = (
        torch.empty((num_experts, n_groups_k, I), dtype=torch.float32)
        .uniform_(0.005, 0.05, generator=gen)
        .to(dtype)
        .to(device)
    )

    down_qweight = torch.randint(-(2**31), 2**31, (num_experts, I // 8, K), generator=gen, dtype=torch.int32).to(device)
    down_qzeros = torch.randint(-(2**31), 2**31, (num_experts, n_groups_i, K // 8), generator=gen, dtype=torch.int32).to(device)
    down_scales = (
        torch.empty((num_experts, n_groups_i, K), dtype=torch.float32)
        .uniform_(0.005, 0.05, generator=gen)
        .to(dtype)
        .to(device)
    )

    return (
        x,
        topk_ids,
        topk_weights,
        gate_qweight,
        gate_scales,
        gate_qzeros,
        up_qweight,
        up_scales,
        up_qzeros,
        down_qweight,
        down_scales,
        down_qzeros,
    )
