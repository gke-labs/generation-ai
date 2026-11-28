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

"""Fused Triton CUDA grouped INT4 MoE variant (gpu_triton) of vxpu::int4_moe."""

import importlib
from typing import Any

import torch

from .. import variant
from . import validate_int4_moe_args

INT4_MOE_MAX_ROWS: int = 8

_triton: Any = None
_gate_up_kernel: Any = None
_down_reduce_kernel: Any = None
try:
    _triton = importlib.import_module("triton")
    _kernel_mod: Any = importlib.import_module(f"{__package__}._triton_kernel")
    _gate_up_kernel = _kernel_mod.int4_moe_gate_up_swiglu_kernel
    _down_reduce_kernel = _kernel_mod.int4_moe_down_reduce_kernel
except ImportError:
    _triton = None
    _gate_up_kernel = None
    _down_reduce_kernel = None


def _applies(plan: Any, rows: int = 0, **_: Any) -> bool:
    """Returns True when Triton CUDA is available and row count fits decode MoE."""
    return bool(
        plan["triton"]
        and rows <= INT4_MOE_MAX_ROWS
        and _triton is not None
        and _gate_up_kernel is not None
        and torch.cuda.is_available()
    )


def _largest_pow2_divisor(n: int, cap: int) -> int:
    """Returns the largest power of two <= cap that divides n."""
    p = 1
    while p * 2 <= cap and n % (p * 2) == 0:
        p *= 2
    return p


def _launch_int4_moe(
    x2: torch.Tensor,
    ids_flat: torch.Tensor,
    weights_flat: torch.Tensor,
    gate_qw: torch.Tensor,
    gate_sc: torch.Tensor,
    gate_qz: torch.Tensor,
    up_qw: torch.Tensor,
    up_sc: torch.Tensor,
    up_qz: torch.Tensor,
    down_qw: torch.Tensor,
    down_sc: torch.Tensor,
    down_qz: torch.Tensor,
    y: torch.Tensor,
    M: int,
    E: int,
    K: int,
    I: int,
    top_k: int,
    n_groups_k: int,
    group_pack_k: int,
    n_groups_i: int,
    group_pack_i: int,
) -> None:
    """Launches fused gate+up+SwiGLU followed by fused down projection and weighted reduction."""
    pairs = M * top_k
    h = torch.empty((pairs, I), dtype=torch.float32, device=x2.device)

    bn = 32
    bk_pack_k = _largest_pow2_divisor(group_pack_k, 16)
    _gate_up_kernel[((I + bn - 1) // bn, pairs)](
        x2,
        ids_flat,
        gate_qw,
        gate_sc,
        gate_qz,
        up_qw,
        up_sc,
        up_qz,
        h,
        E,
        I,
        K,
        n_groups_k,
        top_k,
        GROUP_PACK=group_pack_k,
        BK_PACK=bk_pack_k,
        BN=bn,
        num_warps=4,
        num_stages=2,
    )

    bk_pack_i = _largest_pow2_divisor(group_pack_i, 16)
    _down_reduce_kernel[((K + bn - 1) // bn, M)](
        h,
        ids_flat,
        weights_flat,
        down_qw,
        down_sc,
        down_qz,
        y,
        E,
        K,
        I,
        n_groups_i,
        top_k,
        GROUP_PACK=group_pack_i,
        BK_PACK=bk_pack_i,
        BN=bn,
        num_warps=4,
        num_stages=2,
    )


@variant(
    devices=("cuda",),
    numerics="rounding",
    when=(
        "fused Triton CUDA grouped INT4 MoE unpacking gate/up/down weights in registers "
        f"for decode (M <= {INT4_MOE_MAX_ROWS})"
    ),
    applies=_applies,
    condition={"max_rows": INT4_MOE_MAX_ROWS},
)
def gpu_triton(
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
) -> torch.Tensor:
    """Computes fused grouped INT4 SwiGLU MoE on CUDA without materializing unpacked weights."""
    if _gate_up_kernel is None or _down_reduce_kernel is None:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise ValueError(f"gpu_triton requires CUDA tensors, got {x.device}")

    E, K, I, top_k, n_groups_k, group_size_k, n_groups_i, group_size_i = validate_int4_moe_args(
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

    is_2d = x.ndim == 2
    x2 = x if is_2d else x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty((M, K), dtype=x.dtype, device=x.device)
    if M == 0:
        return y if is_2d else y.reshape(x.shape)

    ids_flat = topk_ids.reshape(-1)
    if not ids_flat.is_contiguous():
        ids_flat = ids_flat.contiguous()
    weights_flat = topk_weights.reshape(-1)
    if not weights_flat.is_contiguous():
        weights_flat = weights_flat.contiguous()

    gate_qw = gate_qweight if gate_qweight.is_contiguous() else gate_qweight.contiguous()
    gate_sc = gate_scales if gate_scales.is_contiguous() else gate_scales.contiguous()
    gate_qz = gate_qzeros if gate_qzeros.is_contiguous() else gate_qzeros.contiguous()
    up_qw = up_qweight if up_qweight.is_contiguous() else up_qweight.contiguous()
    up_sc = up_scales if up_scales.is_contiguous() else up_scales.contiguous()
    up_qz = up_qzeros if up_qzeros.is_contiguous() else up_qzeros.contiguous()
    down_qw = down_qweight if down_qweight.is_contiguous() else down_qweight.contiguous()
    down_sc = down_scales if down_scales.is_contiguous() else down_scales.contiguous()
    down_qz = down_qzeros if down_qzeros.is_contiguous() else down_qzeros.contiguous()

    group_pack_k = group_size_k >> 3
    group_pack_i = group_size_i >> 3

    with torch.cuda.device(x.device):
        _launch_int4_moe(
            x2,
            ids_flat,
            weights_flat,
            gate_qw,
            gate_sc,
            gate_qz,
            up_qw,
            up_sc,
            up_qz,
            down_qw,
            down_sc,
            down_qz,
            y,
            M,
            E,
            K,
            I,
            top_k,
            n_groups_k,
            group_pack_k,
            n_groups_i,
            group_pack_i,
        )
    return y if is_2d else y.reshape(x.shape)
