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

"""Fused Triton CUDA GEMV variant (gpu_triton) of vxpu::int4_linear."""

import importlib
from typing import Any

import torch

from .. import variant

INT4_GEMV_MAX_ROWS: int = 8

_triton: Any = None
_m1_kernel: Any = None
_tc_kernel: Any = None
try:
    _triton = importlib.import_module("triton")
    _kernel_mod: Any = importlib.import_module(f"{__package__}._triton_kernel")
    _m1_kernel = _kernel_mod.int4_gemv_m1_kernel
    _tc_kernel = _kernel_mod.int4_gemm_tc_kernel
except ImportError:
    _triton = None
    _m1_kernel = None
    _tc_kernel = None


def _applies(plan: Any, rows: int = 0, **_: Any) -> bool:
    """Returns True when Triton CUDA is available and row count fits decode GEMV."""
    return bool(
        plan["triton"]
        and rows <= INT4_GEMV_MAX_ROWS
        and _triton is not None
        and _m1_kernel is not None
        and torch.cuda.is_available()
    )


def _largest_pow2_divisor(n: int, cap: int) -> int:
    """Returns the largest power of two <= cap that divides n."""
    p = 1
    while p * 2 <= cap and n % (p * 2) == 0:
        p *= 2
    return p


def _select_m1_config(N: int, group_pack: int) -> tuple[int, int, int, int]:
    """Selects (BN, BK_PACK, num_warps, num_stages) tuned for NVIDIA L4 (58 SMs)."""
    if N > 16384:
        return 128, _largest_pow2_divisor(group_pack, 8), 8, 2
    if N >= 8192:
        return 32, _largest_pow2_divisor(group_pack, 16), 2, 2
    return 32, _largest_pow2_divisor(group_pack, 16), 4, 2


def _select_tc_config(N: int, group_pack: int) -> tuple[int, int, int]:
    """Selects (BN, BK_PACK, num_warps) for Tensor Core small-batch decode on NVIDIA L4."""
    if N > 16384:
        return 64, _largest_pow2_divisor(group_pack, 4), 4
    if N >= 8192:
        return 64, _largest_pow2_divisor(group_pack, 8), 4
    return 32, _largest_pow2_divisor(group_pack, 16), 4


def _launch_int4_linear(
    x2: torch.Tensor,
    qw_c: torch.Tensor,
    sc_c: torch.Tensor,
    qz_c: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    n_groups: int,
    group_pack: int,
) -> None:
    """Launches the appropriate Triton kernel for M=1 GEMV or M>1 Tensor Core decode."""
    if M == 1:
        block_n, bk_pack, num_warps, num_stages = _select_m1_config(N, group_pack)
        _m1_kernel[((N + block_n - 1) // block_n,)](
            x2,
            qw_c,
            sc_c,
            qz_c,
            y,
            N,
            n_groups,
            GROUP_PACK=group_pack,
            BK_PACK=bk_pack,
            BN=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif group_pack % 2 != 0:
        block_n, bk_pack, num_warps, num_stages = _select_m1_config(N, group_pack)
        grid_m1 = ((N + block_n - 1) // block_n,)
        for m in range(M):
            _m1_kernel[grid_m1](
                x2[m],
                qw_c,
                sc_c,
                qz_c,
                y[m],
                N,
                n_groups,
                GROUP_PACK=group_pack,
                BK_PACK=bk_pack,
                BN=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        block_n, bk_pack, num_warps = _select_tc_config(N, group_pack)
        block_m = 16
        grid_tc = ((N + block_n - 1) // block_n, (M + block_m - 1) // block_m)
        _tc_kernel[grid_tc](
            x2,
            qw_c,
            sc_c,
            qz_c,
            y,
            M,
            N,
            K,
            n_groups,
            GROUP_PACK=group_pack,
            BK_PACK=bk_pack,
            BN=block_n,
            BM=block_m,
            num_warps=num_warps,
        )


@variant(
    devices=("cuda",),
    numerics="rounding",
    when=f"fused Triton CUDA GEMV unpacking INT4 weights and zero points in registers for decode (M <= {INT4_GEMV_MAX_ROWS})",
    applies=_applies,
    condition={"max_rows": INT4_GEMV_MAX_ROWS},
)
def gpu_triton(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    """Computes packed INT4 linear on CUDA by fusing 4-bit unpack, dequant, and accumulation in registers."""
    if _m1_kernel is None or _tc_kernel is None:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise ValueError(f"gpu_triton requires CUDA tensors, got {x.device}")
    if x.ndim < 1:
        raise ValueError(f"int4_linear: x must have rank >= 1, got {tuple(x.shape)}")
    if qweight.ndim != 2 or scales.ndim != 2 or qzeros.ndim != 2:
        raise ValueError(
            f"int4_linear: qweight {tuple(qweight.shape)}, scales {tuple(scales.shape)}, "
            f"and qzeros {tuple(qzeros.shape)} must be rank-2 tensors"
        )
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError(
            f"int4_linear: qweight ({qweight.dtype}) and qzeros ({qzeros.dtype}) must be int32"
        )

    K = x.shape[-1]
    n_groups = scales.shape[0]
    N = scales.shape[1]
    if K <= 0 or N <= 0 or (K & 7) != 0 or (N & 7) != 0:
        raise ValueError(f"int4_linear: K ({K}) and N ({N}) must be positive multiples of 8")
    if qweight.shape[0] * 8 != K or qweight.shape[1] != N:
        raise ValueError(
            f"int4_linear: qweight shape {tuple(qweight.shape)} does not match expected ({K // 8}, {N})"
        )
    if n_groups <= 0 or K % n_groups != 0:
        raise ValueError(f"int4_linear: scales groups ({n_groups}) must divide K ({K})")
    group_size = K // n_groups
    if (group_size & 7) != 0:
        raise ValueError(f"int4_linear: group_size ({group_size}) must be a multiple of 8")
    if qzeros.shape[0] != n_groups or qzeros.shape[1] * 8 != N:
        raise ValueError(
            f"int4_linear: qzeros shape {tuple(qzeros.shape)} does not match expected ({n_groups}, {N // 8})"
        )

    is_2d = x.ndim == 2
    x2 = x if is_2d else x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return y if is_2d else y.reshape(tuple(x.shape[:-1]) + (N,))

    qw_c = qweight if qweight.is_contiguous() else qweight.contiguous()
    sc_c = scales if scales.is_contiguous() else scales.contiguous()
    qz_c = qzeros if qzeros.is_contiguous() else qzeros.contiguous()
    group_pack = group_size >> 3

    if x.device.index == torch.cuda.current_device():
        _launch_int4_linear(x2, qw_c, sc_c, qz_c, y, M, N, K, n_groups, group_pack)
    else:
        with torch.cuda.device(x.device):
            _launch_int4_linear(x2, qw_c, sc_c, qz_c, y, M, N, K, n_groups, group_pack)
    return y if is_2d else y.reshape(tuple(x.shape[:-1]) + (N,))
