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

"""Triton JIT kernels for fused grouped packed INT4 SwiGLU MoE decode."""

import triton
import triton.language as tl


@triton.jit
def int4_moe_gate_up_swiglu_kernel(
    x_ptr,
    topk_ids_ptr,
    gate_qw_ptr,
    gate_sc_ptr,
    gate_qz_ptr,
    up_qw_ptr,
    up_sc_ptr,
    up_qz_ptr,
    h_ptr,
    I,
    K,
    N_GROUPS_K,
    TOP_K,
    GROUP_PACK: tl.constexpr,
    BK_PACK: tl.constexpr,
    BN: tl.constexpr,
):
    """Computes fused gate, up, and SwiGLU across all active (token, top_k) pairs.

    Evaluates the gate and up projections sequentially per output tile so each pass
    holds only one (BK_PACK, BN) accumulator tile in registers while reusing the
    L1-resident activation vector x[tok_idx]. Never writes gate or up intermediates
    to HBM; stores only silu(gate) * up into h [pairs, I].
    """
    pid_n = tl.program_id(0)
    pair_idx = tl.program_id(1).to(tl.int64)
    tok_idx = pair_idx // TOP_K
    expert_id = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)

    n_offs = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    n_mask = n_offs < I
    nz_offs = n_offs >> 3
    nz_shifts = ((n_offs & 7) * 4).to(tl.int32)

    i64 = tl.cast(I, tl.int64)
    iz64 = tl.cast(I >> 3, tl.int64)
    k64 = tl.cast(K, tl.int64)
    k_pack64 = tl.cast(K >> 3, tl.int64)
    ng64 = tl.cast(N_GROUPS_K, tl.int64)

    qw_expert_off = expert_id * k_pack64 * i64
    sc_expert_off = expert_id * ng64 * i64
    qz_expert_off = expert_id * ng64 * iz64
    x_base = x_ptr + tok_idx * k64
    kp_arange = tl.arange(0, BK_PACK)

    gate_acc = tl.zeros((BN,), dtype=tl.float32)
    for g in range(0, N_GROUPS_K):
        s_g = tl.load(gate_sc_ptr + sc_expert_off + g * i64 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        z_g = (tl.load(gate_qz_ptr + qz_expert_off + g * iz64 + nz_offs, mask=n_mask, other=0) >> nz_shifts) & 0xF
        tile = tl.zeros((BK_PACK, BN), dtype=tl.float32)
        kp_base = g * GROUP_PACK
        for kp in range(0, GROUP_PACK, BK_PACK):
            kp_offs = kp_base + kp + kp_arange
            qw = tl.load(
                gate_qw_ptr + qw_expert_off + kp_offs[:, None] * i64 + n_offs[None, :],
                mask=n_mask[None, :],
                other=0,
            )
            k_offs = kp_offs * 8
            for b in tl.static_range(8):
                x_b = tl.load(x_base + k_offs + b).to(tl.float32)
                q_b = (((qw >> (b * 4)) & 0xF) - z_g[None, :]).to(tl.float32)
                tile += q_b * x_b[:, None]
        gate_acc += tl.sum(tile, axis=0) * s_g

    up_acc = tl.zeros((BN,), dtype=tl.float32)
    for g in range(0, N_GROUPS_K):
        s_u = tl.load(up_sc_ptr + sc_expert_off + g * i64 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        z_u = (tl.load(up_qz_ptr + qz_expert_off + g * iz64 + nz_offs, mask=n_mask, other=0) >> nz_shifts) & 0xF
        tile = tl.zeros((BK_PACK, BN), dtype=tl.float32)
        kp_base = g * GROUP_PACK
        for kp in range(0, GROUP_PACK, BK_PACK):
            kp_offs = kp_base + kp + kp_arange
            qw = tl.load(
                up_qw_ptr + qw_expert_off + kp_offs[:, None] * i64 + n_offs[None, :],
                mask=n_mask[None, :],
                other=0,
            )
            k_offs = kp_offs * 8
            for b in tl.static_range(8):
                x_b = tl.load(x_base + k_offs + b).to(tl.float32)
                q_b = (((qw >> (b * 4)) & 0xF) - z_u[None, :]).to(tl.float32)
                tile += q_b * x_b[:, None]
        up_acc += tl.sum(tile, axis=0) * s_u

    h_val = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
    tl.store(h_ptr + pair_idx * i64 + n_offs, h_val.to(h_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def int4_moe_down_reduce_kernel(
    h_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    down_qw_ptr,
    down_sc_ptr,
    down_qz_ptr,
    y_ptr,
    K,
    I,
    N_GROUPS_I,
    TOP_K,
    GROUP_PACK: tl.constexpr,
    BK_PACK: tl.constexpr,
    BN: tl.constexpr,
):
    """Computes fused down projection and deterministic weighted top-k reduction per token."""
    pid_n = tl.program_id(0)
    m = tl.program_id(1).to(tl.int64)

    n_offs = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    n_mask = n_offs < K
    nz_offs = n_offs >> 3
    nz_shifts = ((n_offs & 7) * 4).to(tl.int32)

    k64 = tl.cast(K, tl.int64)
    kz64 = tl.cast(K >> 3, tl.int64)
    i64 = tl.cast(I, tl.int64)
    i_pack64 = tl.cast(I >> 3, tl.int64)
    ng64 = tl.cast(N_GROUPS_I, tl.int64)
    kp_arange = tl.arange(0, BK_PACK)

    y_acc = tl.zeros((BN,), dtype=tl.float32)
    base_pair = m * TOP_K
    for r in range(0, TOP_K):
        pair_idx = base_pair + r
        expert_id = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
        weight = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)

        qw_expert_off = expert_id * i_pack64 * k64
        sc_expert_off = expert_id * ng64 * k64
        qz_expert_off = expert_id * ng64 * kz64
        h_base = h_ptr + pair_idx * i64

        acc = tl.zeros((BN,), dtype=tl.float32)
        for g in range(0, N_GROUPS_I):
            s = tl.load(down_sc_ptr + sc_expert_off + g * k64 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
            z = (tl.load(down_qz_ptr + qz_expert_off + g * kz64 + nz_offs, mask=n_mask, other=0) >> nz_shifts) & 0xF
            g_tile = tl.zeros((BK_PACK, BN), dtype=tl.float32)
            kp_base = g * GROUP_PACK
            for kp in range(0, GROUP_PACK, BK_PACK):
                kp_offs = kp_base + kp + kp_arange
                qw = tl.load(
                    down_qw_ptr + qw_expert_off + kp_offs[:, None] * k64 + n_offs[None, :],
                    mask=n_mask[None, :],
                    other=0,
                )
                i_offs = kp_offs * 8
                for b in tl.static_range(8):
                    h_b = tl.load(h_base + i_offs + b).to(tl.float32)
                    q_b = (((qw >> (b * 4)) & 0xF) - z[None, :]).to(tl.float32)
                    g_tile += q_b * h_b[:, None]
            acc += tl.sum(g_tile, axis=0) * s
        y_acc += acc * weight

    tl.store(y_ptr + m * k64 + n_offs, y_acc.to(y_ptr.dtype.element_ty), mask=n_mask)
