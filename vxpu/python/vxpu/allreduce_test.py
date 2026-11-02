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
sum over a fake world on the CPU, the form choice from the knobs, and the Comm's layout."""

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


if __name__ == "__main__":
    unittest.main()
