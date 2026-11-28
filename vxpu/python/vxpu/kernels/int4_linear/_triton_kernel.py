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

"""Triton JIT kernels for packed INT4 linear decode GEMV (M = 1 and M = 2..8)."""

import triton
import triton.language as tl


@triton.jit
def int4_gemv_m1_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    qz_ptr,
    y_ptr,
    N,
    N_GROUPS,
    GROUP_PACK: tl.constexpr,
    BK_PACK: tl.constexpr,
    BN: tl.constexpr,
):
    """Fused single-row (M = 1) packed INT4 GEMV unpacking weights in registers."""
    pid_n = tl.program_id(0)
    n_offs = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    nz_offs = n_offs >> 3
    nz_shifts = ((n_offs & 7) * 4).to(tl.int32)
    n_z = N >> 3
    n64 = tl.cast(N, tl.int64)
    nz64 = tl.cast(n_z, tl.int64)

    acc = tl.zeros((BN,), dtype=tl.float32)
    kp_arange = tl.arange(0, BK_PACK)

    for g in range(0, N_GROUPS):
        s = tl.load(sc_ptr + g * n64 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        qz = tl.load(qz_ptr + g * nz64 + nz_offs, mask=n_mask, other=0)
        z = (qz >> nz_shifts) & 0xF

        g_tile = tl.zeros((BK_PACK, BN), dtype=tl.float32)
        kp_base = g * GROUP_PACK
        for kp in range(0, GROUP_PACK, BK_PACK):
            kp_offs = kp_base + kp + kp_arange
            qw = tl.load(qw_ptr + kp_offs[:, None] * n64 + n_offs[None, :], mask=n_mask[None, :], other=0)
            k_offs = kp_offs * 8
            for b in tl.static_range(8):
                x_b = tl.load(x_ptr + k_offs + b).to(tl.float32)
                q_b = (((qw >> (b * 4)) & 0xF) - z[None, :]).to(tl.float32)
                g_tile += q_b * x_b[:, None]
        acc += tl.sum(g_tile, axis=0) * s

    tl.store(y_ptr + n_offs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def int4_gemm_tc_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    qz_ptr,
    y_ptr,
    M,
    N,
    K,
    N_GROUPS,
    GROUP_PACK: tl.constexpr,
    BK_PACK: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    """Fused small-batch (M = 2..8) packed INT4 GEMM using Tensor Core tl.dot with fp32 accumulation."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offs = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    nz_offs = n_offs >> 3
    nz_shifts = ((n_offs & 7) * 4).to(tl.int32)
    n_z = N >> 3
    n64 = tl.cast(N, tl.int64)
    nz64 = tl.cast(n_z, tl.int64)
    k64 = tl.cast(K, tl.int64)

    m_offs = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    BK: tl.constexpr = BK_PACK * 8
    kp_arange = tl.arange(0, BK_PACK)
    b_shifts = (tl.arange(0, 8) * 4).to(tl.int32)
    k_arange = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, N_GROUPS):
        s = tl.load(sc_ptr + g * n64 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        qz = tl.load(qz_ptr + g * nz64 + nz_offs, mask=n_mask, other=0)
        z = (qz >> nz_shifts) & 0xF

        g_acc = tl.zeros((BM, BN), dtype=tl.float32)
        kp_base = g * GROUP_PACK
        for kp in range(0, GROUP_PACK, BK_PACK):
            kp_offs = kp_base + kp + kp_arange
            qw = tl.load(qw_ptr + kp_offs[:, None] * n64 + n_offs[None, :], mask=n_mask[None, :], other=0)
            # (q - z) is an integer in [-15, 15], which is bit-exact in both bfloat16 and float16.
            q = (((qw[:, None, :] >> b_shifts[None, :, None]) & 0xF) - z[None, None, :]).reshape(BK, BN)
            q_cast = q.to(x_ptr.dtype.element_ty)
            k_base = (kp_base + kp) * 8
            x_tile = tl.load(
                x_ptr + m_offs[:, None] * k64 + (k_base + k_arange[None, :]),
                mask=m_mask[:, None],
                other=0.0,
            )
            g_acc = tl.dot(x_tile, q_cast, g_acc)
        acc += g_acc * s[None, :]

    y_offs = m_offs[:, None] * n64 + n_offs[None, :]
    tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
