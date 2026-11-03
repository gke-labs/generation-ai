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

"""vxpu::allreduce without a GPU: the identity in a world of one, the reference's rank-ordered
sum over a fake world on the CPU, the form choice from the knobs, and the Comm's layout; and on
two or more GPUs the fused push against the one-shot form (hero run 15)."""

import unittest

import torch

from . import allreduce, ops, plan, triton_ops


class AllReduceTest(unittest.TestCase):
    def tearDown(self):
        allreduce._COMM = None
        triton_ops.ALLREDUCE_SCATTER = None
        triton_ops.ALLREDUCE_SCATTER_MIN_ROWS = None

    def test_world_of_one_is_the_identity(self):
        allreduce._COMM = None
        x = torch.randn(3, 8)
        y = torch.ops.vxpu.allreduce(x, 0)
        self.assertTrue(torch.equal(x, y))
        self.assertIsNot(x, y)

    def test_reference_sums_in_rank_order_once_rounded(self):
        # a fake world of two cards on the CPU: both ranks map to cpu, so rank_of cannot tell them
        # apart; the test registers the partials by rank itself, as the kernels' inboxes would hold them
        c = allreduce.setup(["cpu", "cpu"], sites=2, max_elems=64)
        self.assertEqual(c.world, 2)
        a = torch.randn(2, 16, dtype=torch.bfloat16)
        b = torch.randn(2, 16, dtype=torch.bfloat16)
        c.register(1, 0, a)
        c.register(1, 1, b)
        y = allreduce.allreduce_reference(a, 1)
        want = (a.float() + b.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(y, want))
        # a missing rank is an error, not a partial sum
        c.register(0, 0, a)
        with self.assertRaises(RuntimeError):
            allreduce.allreduce_reference(a, 0)

    def test_checks(self):
        c = allreduce.setup(["cpu", "cpu"], sites=1, max_elems=8)
        with self.assertRaises(ValueError):
            allreduce._check(torch.zeros(3, 4, dtype=torch.bfloat16), 0, c)  # too many elements
        with self.assertRaises(ValueError):
            allreduce._check(torch.zeros(4, dtype=torch.bfloat16), 1, c)  # no such site
        with self.assertRaises(ValueError):
            allreduce._check(torch.zeros(4, dtype=torch.float32), 0, c)  # not the Comm's dtype

    def test_form_choice_from_the_knobs(self):
        self.assertFalse(allreduce.scatter_wanted(132))  # no crossover in the table: one-shot
        triton_ops.ALLREDUCE_SCATTER_MIN_ROWS = 64
        self.assertFalse(allreduce.scatter_wanted(33))
        self.assertTrue(allreduce.scatter_wanted(132))
        triton_ops.ALLREDUCE_SCATTER = "0"
        self.assertFalse(allreduce.scatter_wanted(132))  # forced off wins over the rows
        triton_ops.ALLREDUCE_SCATTER = "1"
        self.assertTrue(allreduce.scatter_wanted(1))

    def test_slots_keep_the_own_rank_last(self):
        c = allreduce.setup(["cpu"] * 4, sites=1, max_elems=4)
        # on card 2 the peers 0, 1, 3 write slots 0, 1, 2; the wait scans PEERS = 3 slots from 0
        self.assertEqual([allreduce._slot(c, 2, src) for src in (0, 1, 3)], [0, 1, 2])
        self.assertEqual(c.peers(2), [0, 1, 3])
        self.assertEqual(c.flags[2].shape, (1, 2, 4))
        self.assertEqual(c.slice, 1)

    def test_registered_with_the_plan(self):
        self.assertIn("vxpu::allreduce", ops.SELECTION)
        names = [i.name for i in plan.REGISTRY["vxpu::allreduce"]]
        self.assertEqual(names, ["gpu-scatter", "gpu-oneshot", "reference"])
        for knob in ("ALLREDUCE_SCATTER", "ALLREDUCE_SCATTER_MIN_ROWS", "ALLREDUCE_BLOCK"):
            self.assertIn(knob, plan.KNOBS)
        self.assertIn("FORM", triton_ops.GEMV_BLOCK_NAMES)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2 and triton_ops.available(),
                     "needs two GPUs and Triton")
class FusedPushTest(unittest.TestCase):
    """The fused push (hero run 15) against the one-shot form of the partial it would have pushed,
    and both against the total formed on the card itself: every rank draws the same inputs, so the
    total is the partial added world times in rank order (fp32, rounded once to bf16), and the mix
    is the identity on copy 0 (post[.., 0] = 1, comb = 0), so out[.., 0, :] must equal it bit for
    bit. Nothing is copied between cards for the comparison: a cross-device copy of a card's
    tensor came back stale while the sums were right on the card (hero run 15's harness)."""

    def tearDown(self):
        allreduce._COMM = None

    def test_same_bits_as_the_one_shot_form_and_the_total(self):
        devs = [torch.device(f"cuda:{i}") for i in range(min(torch.cuda.device_count(), 4))]
        hc, N = 4, 5120
        allreduce.setup(devs, 2, 5 * N, dtypes=[torch.float32, torch.bfloat16], hcs=[hc])
        for M, K, with_add in ((1, 1024, False), (3, 1024, False), (2, 2048, True), (5, 2048, True)):
            per = []
            for dv in devs:
                torch.manual_seed(M * 7 + K)  # the same inputs on every card
                with torch.cuda.device(dv):
                    x = triton_ops.act_quant_dequant(torch.randn(1, M, K, device=dv, dtype=torch.bfloat16))
                    w = (torch.randn(N, K, device=dv) * 0.05).to(torch.float8_e4m3fn)
                    s = torch.full((N // 32, K // 32), 127, dtype=torch.uint8, device=dv).view(torch.float8_e8m0fnu)
                    add = torch.randn(1, M, N, device=dv) if with_add else None
                    post = torch.zeros(1, M, hc, device=dv)
                    post[..., 0] = 1.0
                    mix = (torch.randn(1, M, hc, N, device=dv).to(torch.bfloat16), post, torch.zeros(1, M, hc, hc, device=dv))
                    per.append((x, w, s, add, mix))
            for dv in devs:
                torch.cuda.synchronize(dv)
            control = []
            for dv, (x, w, s, add, mix) in zip(devs, per):
                with torch.cuda.device(dv):
                    control.append(allreduce.allreduce_oneshot_hc(ops.gemv_partial(x, w, s, add).float(), 0, *mix))
            for dv in devs:
                torch.cuda.synchronize(dv)
            fused = []
            for dv, (x, w, s, add, mix) in zip(devs, per):
                with torch.cuda.device(dv):
                    fused.append(allreduce.gemv_allreduce_hc(x, w, s, add, 1, *mix))
            for dv in devs:
                torch.cuda.synchronize(dv)
            for r, (dv, (x, w, s, add, mix), c, f) in enumerate(zip(devs, per, control, fused)):
                with torch.cuda.device(dv):
                    part = ops.gemv_partial(x, w, s, add).float()
                    total = part
                    for _ in range(len(devs) - 1):
                        total = total + part
                    total = total.to(torch.bfloat16)
                    self.assertEqual(tuple(f.shape), (1, M, hc, N))
                    self.assertTrue(torch.equal(c, f), f"rank {r}: fused differs from the one-shot form at M={M} K={K} add={with_add}")
                    self.assertTrue(torch.equal(f[:, :, 0, :], total), f"rank {r}: the sum is off at M={M} K={K} add={with_add}")


if __name__ == "__main__":
    unittest.main()
