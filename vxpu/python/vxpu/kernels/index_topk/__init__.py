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

Inputs builder, scorable shapes, and verification logic allowing exact tie equivalence."""

import math
from typing import Any
import torch

from ...ops import _index_scores_reference

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
    rows, H, D = int(rows_str), int(h_str), int(d_str)

    t_val = 4096
    k_val = 512
    for p in right.strip().split():
        if p.startswith("T="):
            t_val = int(p.split("=")[1])
        elif p.startswith("k="):
            k_val = int(p.split("=")[1])

    gen = torch.Generator(device="cpu").manual_seed(seed)

    q = torch.randn(rows, 1, H, D, generator=gen, dtype=torch.bfloat16).to(device)
    index_k = torch.randn(rows, t_val, D, generator=gen, dtype=torch.bfloat16).to(device)
    weights = torch.randn(rows, 1, H, generator=gen, dtype=torch.bfloat16).to(device)

    # Vary compress_lens per row between k_val and t_val
    cl_list: list[int] = []
    for r in range(rows):
        if t_val > k_val:
            cl = min(t_val, max(k_val, t_val * 3 // 4 + (r * 17) % (t_val // 4)))
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

    Indices must be sorted ascending, with unreachable positions mapped to -1 at the tail.
    Where scores are distinct, the selection is exact (torch.equal).
    Where two positions score identically, the check accepts either choice provided the
    differing indices have the exact same score.
    """
    if not isinstance(got, torch.Tensor) or not isinstance(want, torch.Tensor):
        return False, float("inf")

    if got.shape != want.shape or got.dtype != want.dtype:
        return False, float("inf")

    if torch.equal(got, want):
        return True, 0.0

    # If not bit-identical, check if differences are solely due to tied scores
    b, s, k = got.shape
    scores_ref = _index_scores_reference(q, index_k, weights, compress_lens, candidates)

    for bi in range(b):
        for si in range(s):
            g_row = got[bi, si].tolist()
            w_row = want[bi, si].tolist()
            sc_row = scores_ref[bi, si]
            cl_val = compress_lens[bi, si, 0].item()

            # Must have index_topk entries
            if len(g_row) != k or len(w_row) != k:
                return False, float("inf")

            # Check valid picks sorted ascending, -1 only at the end
            for i in range(k - 1):
                if g_row[i] == -1 and g_row[i + 1] != -1:
                    return False, float("inf")
                if g_row[i] != -1 and g_row[i + 1] != -1 and g_row[i] >= g_row[i + 1]:
                    return False, float("inf")

            # Check bounds on valid picks
            for idx in g_row:
                if idx != -1:
                    pos = idx - offset
                    if pos < 0 or pos >= cl_val:
                        return False, float("inf")

            # Count of -1s must match
            if g_row.count(-1) != w_row.count(-1):
                return False, float("inf")

            g_finite = [x for x in g_row if x != -1]
            w_finite = [x for x in w_row if x != -1]

            diff_g = set(g_finite) - set(w_finite)
            diff_w = set(w_finite) - set(g_finite)

            if len(diff_g) != len(diff_w):
                return False, float("inf")

            if diff_g:
                # Compare scores of differing picks: differing picks must have the exact same score (tie)
                g_scores = sorted([sc_row[x - offset].item() for x in diff_g])
                w_scores = sorted([sc_row[x - offset].item() for x in diff_w])
                for gs, ws in zip(g_scores, w_scores):
                    # In bfloat16 arithmetic, allow exact tie or tiny ulp difference at boundary
                    if gs != ws and abs(gs - ws) > 1e-4:
                        return False, float("inf")

    return True, 0.0
