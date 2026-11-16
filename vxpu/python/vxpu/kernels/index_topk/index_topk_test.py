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

"""Unit tests for vxpu::index_topk."""

# pyright: strict

from typing import Any
import unittest

import torch

from . import inputs, judge
from ... import ops, plan, triton_ops


def _run_reference(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    k: int,
    offset: int = 0,
) -> torch.Tensor:
    impl = plan.reference_of("vxpu::index_topk")
    assert impl is not None
    ops_mod: Any = ops
    out: torch.Tensor = ops_mod.call("vxpu::index_topk", impl.fn, q, index_k, weights, compress_lens, candidates, k, offset)
    return out


def _ints(t: torch.Tensor) -> list[int]:
    flat = t.reshape(-1)
    return [int(flat[i].item()) for i in range(flat.numel())]


def _fork_tail(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    k: int,
    offset: int = 0,
) -> torch.Tensor:
    """Literal tail of deepseek_v41 Indexer.forward for reference parity testing."""
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    slots = torch.arange(index_k.size(1), device=q.device)
    index_score = index_score.masked_fill(slots >= compress_lens, -torch.inf)
    if candidates is not None:
        index_score = index_score.masked_fill(~candidates, -torch.inf)
    idxs = index_score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


class IndexTopkOpTest(unittest.TestCase):
    """Verifies input builder, fork parity, judge tie rules, and Triton variant."""

    def test_inputs_tiny_cache(self) -> None:
        """inputs() handles small shapes where k < T < 4 without dividing by zero."""
        for shape in ("1x4x16->T=2 k=1", "1x4x16->T=3 k=2", "3x4x16->T=3 k=1 cand"):
            q, index_k, weights, compress_lens, candidates, k_val, offset = inputs(shape, device="cpu", seed=7)
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertEqual(index_k.dtype, torch.bfloat16)
            self.assertEqual(weights.dtype, torch.bfloat16)
            self.assertEqual(compress_lens.dtype, torch.int32)
            self.assertEqual(offset, 0)
            out = torch.ops.vxpu.index_topk(q, index_k, weights, compress_lens, candidates, k_val, offset)
            self.assertEqual(out.shape, (q.shape[0], 1, k_val))
            self.assertEqual(out.dtype, torch.int32)

    def test_reference_matches_fork_all_regimes(self) -> None:
        """Reference matches the fork's tail, including sparse candidate masks leaving < k positions."""
        gen = torch.Generator(device="cpu").manual_seed(42)
        b, s, h, d, t, k = 2, 1, 8, 32, 64, 8
        q = torch.randn(b, s, h, d, generator=gen, dtype=torch.bfloat16)
        index_k = torch.randn(b, t, d, generator=gen, dtype=torch.bfloat16)
        weights = torch.randn(b, s, h, generator=gen, dtype=torch.bfloat16)

        # Case 1: no candidate mask, normal compress_lens
        cl_norm = torch.tensor([[[40]], [[50]]], dtype=torch.int32)
        got = _run_reference(q, index_k, weights, cl_norm, None, k, offset=5)
        want = _fork_tail(q, index_k, weights, cl_norm, None, k, offset=5)
        self.assertTrue(torch.equal(got, want))

        # Case 2: compress_lens < k
        cl_short = torch.tensor([[[3]], [[0]]], dtype=torch.int32)
        got = _run_reference(q, index_k, weights, cl_short, None, k, offset=2)
        want = _fork_tail(q, index_k, weights, cl_short, None, k, offset=2)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(_ints(got[1, 0]), [-1] * k)

        # Case 3: dense candidate mask (>= k candidates)
        cand_dense = torch.ones(b, s, t, dtype=torch.bool)
        cand_dense[:, :, ::2] = False
        got = _run_reference(q, index_k, weights, cl_norm, cand_dense, k, offset=0)
        want = _fork_tail(q, index_k, weights, cl_norm, cand_dense, k, offset=0)
        self.assertTrue(torch.equal(got, want))

        # Case 4: sparse candidate mask (< k candidates) with cl < t and cl == t
        cand_sparse = torch.zeros(b, s, t, dtype=torch.bool)
        for pos in (13, 20, 35, 43):
            cand_sparse[:, :, pos] = True
        for cl in (torch.tensor([[[50]], [[50]]], dtype=torch.int32), torch.tensor([[[t]], [[t]]], dtype=torch.int32)):
            got = _run_reference(q, index_k, weights, cl, cand_sparse, k, offset=0)
            want = _fork_tail(q, index_k, weights, cl, cand_sparse, k, offset=0)
            self.assertTrue(torch.equal(got, want))
            # With cl == t, no position is >= compress_lens, so no pick maps to -1.
            if int(cl[0, 0, 0].item()) == t:
                self.assertEqual(_ints(got[0, 0]).count(-1), 0)

    def test_judge_rejects_near_zero_wrong_pick(self) -> None:
        """judge() compares scores for exact equality and rejects close but distinct near-zero scores."""
        gen = torch.Generator(device="cpu").manual_seed(123)
        b, s, h, d, t, k = 1, 1, 8, 32, 32, 4
        q = (torch.randn(b, s, h, d, generator=gen, dtype=torch.float32) * 0.01).to(torch.bfloat16)
        index_k = (torch.randn(b, t, d, generator=gen, dtype=torch.float32) * 0.01).to(torch.bfloat16)
        weights = (torch.randn(b, s, h, generator=gen, dtype=torch.float32) * 0.01).to(torch.bfloat16)
        cl = torch.tensor([[[t]]], dtype=torch.int32)

        ops_mod: Any = ops
        scores: torch.Tensor = ops_mod._index_scores_reference(q, index_k, weights, cl, None)[0, 0]
        order = _ints(scores.argsort(descending=True))
        want = _run_reference(q, index_k, weights, cl, None, k, 0)

        # Find a non-selected position with a strictly lower score within 1e-4 of the k-th pick.
        kth_pos = order[k - 1]
        kth_score = float(scores[kth_pos].item())
        replacement = None
        for pos in order[k:]:
            sc = float(scores[pos].item())
            if sc < kth_score and abs(kth_score - sc) < 1e-4:
                replacement = pos
                break
        self.assertIsNotNone(replacement, "expected a distinct near-zero non-top-k score within 1e-4")
        assert replacement is not None

        wrong_list = sorted([x for x in _ints(want[0, 0]) if x != kth_pos] + [replacement])
        wrong = torch.tensor(wrong_list, dtype=torch.int32).view(1, 1, k)
        passed, _ = judge("rounding", wrong, want, q, index_k, weights, cl, None, k, 0)
        self.assertFalse(passed)

    def test_judge_tie_and_structural_rules(self) -> None:
        """judge() accepts exact finite and -inf ties and rejects structural violations."""
        b, s, h, d, t, k = 1, 1, 4, 16, 16, 4
        # Zero weights produce identical 0.0 scores for all reachable candidate positions.
        q = torch.ones(b, s, h, d, dtype=torch.bfloat16)
        index_k = torch.ones(b, t, d, dtype=torch.bfloat16)
        weights = torch.zeros(b, s, h, dtype=torch.bfloat16)
        cl = torch.tensor([[[10]]], dtype=torch.int32)

        want = _run_reference(q, index_k, weights, cl, None, k, 0)
        # Alternate finite tie choice among reachable positions [0..9].
        alt_tie = torch.tensor([[[2, 5, 7, 9]]], dtype=torch.int32)
        self.assertTrue(judge("rounding", alt_tie, want, q, index_k, weights, cl, None, k, 0)[0])

        # Unsorted or duplicate or out-of-bounds picks must fail.
        unsorted = torch.tensor([[[5, 2, 7, 9]]], dtype=torch.int32)
        self.assertFalse(judge("rounding", unsorted, want, q, index_k, weights, cl, None, k, 0)[0])
        dup = torch.tensor([[[2, 2, 7, 9]]], dtype=torch.int32)
        self.assertFalse(judge("rounding", dup, want, q, index_k, weights, cl, None, k, 0)[0])
        oob = torch.tensor([[[2, 5, 7, 10]]], dtype=torch.int32)
        self.assertFalse(judge("rounding", oob, want, q, index_k, weights, cl, None, k, 0)[0])

        # Sparse candidates (< k finite scores) with cl < t: swapping a reachable -inf pick for -1 is a valid tie.
        cand = torch.zeros(b, s, t, dtype=torch.bool)
        cand[0, 0, 3] = True
        cand[0, 0, 7] = True
        want_sparse = _run_reference(q, index_k, weights, cl, cand, k, 0)
        alt_with_neg1 = torch.tensor([[[3, 7, -1, -1]]], dtype=torch.int32)
        self.assertTrue(judge("rounding", alt_with_neg1, want_sparse, q, index_k, weights, cl, cand, k, 0)[0])
        alt_with_reachable = torch.tensor([[[0, 1, 3, 7]]], dtype=torch.int32)
        self.assertTrue(judge("rounding", alt_with_reachable, want_sparse, q, index_k, weights, cl, cand, k, 0)[0])
        # Leading -1 must fail.
        leading_neg1 = torch.tensor([[[-1, 3, 7, -1]]], dtype=torch.int32)
        self.assertFalse(judge("rounding", leading_neg1, want_sparse, q, index_k, weights, cl, cand, k, 0)[0])

        # With cl == t (no unreachable slots), -1 is never valid even under sparse candidates.
        cl_full = torch.tensor([[[t]]], dtype=torch.int32)
        want_full = _run_reference(q, index_k, weights, cl_full, cand, k, 0)
        self.assertFalse(judge("rounding", alt_with_neg1, want_full, q, index_k, weights, cl_full, cand, k, 0)[0])

    def test_triton_variant(self) -> None:
        """Triton variant agrees with reference across sparse, dense, short, and scaled inputs."""
        if not torch.cuda.is_available() or not triton_ops.available():
            self.skipTest("CUDA or Triton is not available")

        gen = torch.Generator(device="cpu").manual_seed(99)
        b, s, h, d, t = 2, 1, 16, 64, 64
        q = torch.randn(b, s, h, d, generator=gen, dtype=torch.bfloat16).cuda()
        index_k = torch.randn(b, t, d, generator=gen, dtype=torch.bfloat16).cuda()
        weights = torch.randn(b, s, h, generator=gen, dtype=torch.bfloat16).cuda()

        cand_sparse = torch.zeros(b, s, t, dtype=torch.bool, device="cuda")
        for pos in (4, 17, 29):
            cand_sparse[:, :, pos] = True

        cases: list[tuple[torch.Tensor, torch.Tensor | None, int, int]] = [
            (torch.tensor([[[48]], [[32]]], dtype=torch.int32, device="cuda"), None, 8, 0),
            (torch.tensor([[[3]], [[0]]], dtype=torch.int32, device="cuda"), None, 8, 12),
            (torch.tensor([[[50]], [[t]]], dtype=torch.int32, device="cuda"), cand_sparse, 6, 0),
            (torch.tensor([[[50]], [[t]]], dtype=torch.int32, device="cuda"), cand_sparse, 8, 7),
        ]
        triton_mod: Any = triton_ops
        for cl, cand, k_val, offset in cases:
            want = _run_reference(q, index_k, weights, cl, cand, k_val, offset)
            got = triton_ops.index_topk(q, index_k, weights, cl, cand, k_val, offset)
            passed, _ = judge("rounding", got, want, q, index_k, weights, cl, cand, k_val, offset)
            self.assertTrue(passed, f"triton mismatch for cl={_ints(cl)}, k={k_val}, offset={offset}")
            got_lowered: torch.Tensor = triton_mod.lowered.index_topk(q, index_k, weights, cl, cand, k_val, offset)
            self.assertTrue(torch.equal(got, got_lowered))
