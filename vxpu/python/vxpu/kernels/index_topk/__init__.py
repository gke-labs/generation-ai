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

"""Kernel contract definition for vxpu::index_topk.

Provides the input builder, scorable shapes, and verification logic with exact score tie handling.
"""

# pyright: strict

import math
from typing import Any

import torch

# Scorable decode shapes at the released model's configuration:
# index_n_heads=32, index_head_dim=128, T=4096, index_topk=512.
SCORE_SHAPES: tuple[str, ...] = (
    "1x32x128->T=4096 k=512",
    "1x32x128->T=4096 k=512 cand",
)


def inputs(
    shapes: str, device: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Builds synthetic inputs for vxpu::index_topk from a shape string.

    Parses shapes in '{rows}x{H}x{D}->T={T} k={topk}' format (with optional 'cand').
    Returns (q, index_k, weights, compress_lens, candidates, index_topk, offset).
    """
    has_cand = "cand" in shapes
    s = shapes.replace("cand", "").strip()
    left, right = s.split("->")
    rows_str, h_str, d_str = left.strip().split("x")
    rows, h_val, d_val = int(rows_str), int(h_str), int(d_str)

    t_val = 4096
    k_val = 512
    for p in right.strip().split():
        if p.startswith("T="):
            t_val = int(p.split("=")[1])
        elif p.startswith("k="):
            k_val = int(p.split("=")[1])

    gen = torch.Generator(device="cpu").manual_seed(seed)

    q = torch.randn(rows, 1, h_val, d_val, generator=gen, dtype=torch.bfloat16).to(device)
    index_k = torch.randn(rows, t_val, d_val, generator=gen, dtype=torch.bfloat16).to(device)
    weights = torch.randn(rows, 1, h_val, generator=gen, dtype=torch.bfloat16).to(device)

    # Vary compress_lens per row between k_val and t_val.
    cl_list: list[int] = []
    for r in range(rows):
        if t_val > k_val:
            cl = min(t_val, max(k_val, t_val * 3 // 4 + (r * 17) % max(1, t_val // 4)))
        else:
            cl = t_val
        cl_list.append(cl)
    compress_lens = torch.tensor(cl_list, dtype=torch.int32).view(rows, 1, 1).to(device)

    candidates: torch.Tensor | None = None
    if has_cand:
        candidates = torch.randint(0, 2, (rows, 1, t_val), generator=gen, dtype=torch.bool).to(device)

    return q, index_k, weights, compress_lens, candidates, k_val, 0


def judge(
    numerics: str,
    got: Any,
    want: Any,
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    index_topk: int,
    offset: int = 0,
) -> tuple[bool, float]:
    """Verifies got against want for vxpu::index_topk.

    Indices must be strictly ascending, with unreachable picks mapped to -1 at the tail.
    Where scores are distinct, the selection matches the reference exactly. Where two
    positions share an identical score (including -inf picks among non-candidate and
    unreachable slots), either choice is accepted provided the sorted multiset of pick
    scores matches the reference exactly.
    """
    _ = numerics
    if not isinstance(got, torch.Tensor) or not isinstance(want, torch.Tensor):
        return False, float("inf")

    if got.shape != want.shape or got.dtype != want.dtype:
        return False, float("inf")

    if torch.equal(got, want):
        return True, 0.0

    b, s, k = got.shape
    if k != index_topk:
        return False, float("inf")

    t_val = index_k.shape[1]
    from ... import ops

    ops_mod: Any = ops
    scores_ref: torch.Tensor = ops_mod._index_scores_reference(q, index_k, weights, compress_lens, candidates)

    for bi in range(b):
        for si in range(s):
            g_t = got[bi, si]
            w_t = want[bi, si]
            g_row = [int(g_t[i].item()) for i in range(k)]
            w_row = [int(w_t[i].item()) for i in range(k)]
            sc_row = scores_ref[bi, si]
            cl_val = int(compress_lens[bi, si, 0].item())
            n_reachable = max(0, min(cl_val, t_val))
            n_unreachable = t_val - n_reachable

            if len(g_row) != k or len(w_row) != k:
                return False, float("inf")

            # Check valid picks are strictly ascending and -1 appears only at the tail.
            for i in range(k - 1):
                if g_row[i] == -1 and g_row[i + 1] != -1:
                    return False, float("inf")
                if g_row[i] != -1 and g_row[i + 1] != -1 and g_row[i] >= g_row[i + 1]:
                    return False, float("inf")

            # Check bounds on valid picks and count of unreachable -1 picks.
            n_neg1 = 0
            for idx in g_row:
                if idx == -1:
                    n_neg1 += 1
                else:
                    pos = idx - offset
                    if pos < 0 or pos >= n_reachable:
                        return False, float("inf")
            if n_neg1 > n_unreachable:
                return False, float("inf")

            # Compare the exact multiset of pick scores; -1 stands for an unreachable (-inf) pick.
            g_scores = sorted(float(sc_row[idx - offset].item()) if idx != -1 else -math.inf for idx in g_row)
            w_scores = sorted(float(sc_row[idx - offset].item()) if idx != -1 else -math.inf for idx in w_row)
            if g_scores != w_scores:
                return False, float("inf")

    return True, 0.0
