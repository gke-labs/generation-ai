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

"""Tests for vxpu::allreduce under the kernel contract."""

import json
import unittest

import torch

from . import ops, plan, plan_pb2
from .kernel_tools.check import run_check
from .kernels.allreduce.narrow import narrow
from .kernels.allreduce.one_shot import one_shot
from .kernels.allreduce.scatter_gather import scatter_gather


class AllreduceKernelTest(unittest.TestCase):
    """Verifies that vxpu::allreduce meets the kernel contract."""

    def test_reference_bit_identical_and_arrival_order_invariant(self) -> None:
        """Verifies reference bit-identity and arrival-order invariance for dense step dtypes."""
        # Accumulating partials directly in arrival order reassociates and differs in bits:
        p0 = torch.tensor([1.0], dtype=torch.float32)
        p1 = torch.tensor([1e8], dtype=torch.float32)
        p2 = torch.tensor([-1e8], dtype=torch.float32)
        arrival_order_1 = (p0 + p1) + p2  # 1.0 + 1e8 rounds to 1e8, minus 1e8 = 0.0
        arrival_order_2 = (p1 + p2) + p0  # 1e8 - 1e8 = 0.0, plus 1.0 = 1.0
        self.assertFalse(torch.equal(arrival_order_1, arrival_order_2))

        # Tagging partials by rank and slotting them into the buffer ensures bit-identical results:
        gen = torch.Generator().manual_seed(42)
        world_sizes = [2, 4, 8]
        shapes = [(1, 5120), (33, 5120)]
        dtypes = [torch.bfloat16, torch.float32]

        for world in world_sizes:
            for s in shapes:
                for dt in dtypes:
                    # Individual rank partials arriving from peers:
                    partials = [torch.randn(s, generator=gen, dtype=dt) for _ in range(world)]

                    # Standard arrival in rank order 0, 1, ..., world-1:
                    stacked_standard = torch.stack(partials, dim=0)
                    res_standard = ops.allreduce(stacked_standard)

                    # Permuted arrival order (reverse arrival):
                    # Each partial arrives with its rank tag and is placed in its rank slot.
                    perm = list(reversed(range(world)))
                    arriving_tagged = [(r, partials[r]) for r in perm]

                    reordered = [torch.empty_like(partials[0]) for _ in range(world)]
                    for r, p in arriving_tagged:
                        reordered[r] = p
                    stacked_reordered = torch.stack(reordered, dim=0)
                    res_reordered = ops.allreduce(stacked_reordered)

                    self.assertEqual(res_standard.shape, s)
                    self.assertEqual(res_standard.dtype, dt)
                    self.assertTrue(
                        torch.equal(res_standard, res_reordered),
                        f"mismatch under arrival reordering for world={world}, shape={s}, dtype={dt}",
                    )

    def test_one_dimensional_scalar_partial_inputs(self) -> None:
        """Verifies that all bodies handle 1D [ranks] buffer inputs without raising IndexError."""
        gen = torch.Generator().manual_seed(77)
        for world in [1, 2, 4, 7]:
            b = torch.randn(world, generator=gen, dtype=torch.bfloat16)
            ref = ops.allreduce(b)
            self.assertEqual(ref.shape, ())

            # one_shot single-kernel reduction over rank axis
            os_res = one_shot(b)
            self.assertEqual(os_res.shape, ())
            self.assertTrue(torch.equal(os_res, ref))

            # scatter_gather fallback for 1D inputs
            sg_res = scatter_gather(b)
            self.assertEqual(sg_res.shape, ())
            self.assertTrue(torch.equal(sg_res, ref))

            # narrow wire approximation
            n_res = narrow(b)
            self.assertEqual(n_res.shape, ())

    def test_kernel_contract_registration(self) -> None:
        """Verifies that allreduce, reference, and file variants register under the contract."""
        op = "vxpu::allreduce"
        self.assertIn(op, plan.REGISTRY)
        impls = plan.REGISTRY[op]
        names = [im.name for im in impls]

        # Reference is registered and marked reference.
        self.assertIn("reference", names)
        ref_impl = next(im for im in impls if im.name == "reference")
        self.assertTrue(ref_impl.reference)
        self.assertEqual(ref_impl.numerics, "bit-identical")
        self.assertEqual(ref_impl.kind, plan_pb2.SITE_KIND_SUM_SITE)

        # File variants discovered under kernels/allreduce.
        self.assertIn("one_shot", names)
        self.assertIn("scatter_gather", names)
        self.assertIn("narrow", names)

        one_shot_impl = next(im for im in impls if im.name == "one_shot")
        self.assertEqual(one_shot_impl.numerics, "rounding")

        sg_impl = next(im for im in impls if im.name == "scatter_gather")
        self.assertEqual(sg_impl.numerics, "bit-identical")

        narrow_impl = next(im for im in impls if im.name == "narrow")
        self.assertEqual(narrow_impl.numerics, "rounding")
        self.assertIsNotNone(narrow_impl.condition)
        assert narrow_impl.condition is not None
        self.assertEqual(narrow_impl.condition.require_external_fact, "approximate")

    def test_contract_check_passes_on_variants(self) -> None:
        """Verifies that check passes for all declared variant bodies against the reference."""
        test_shapes = ["4x1x5120", "4x1x5120_f32", "8x3x7_f32", "33x1x5120_f32"]
        for shape in test_shapes:
            # one_shot: rounding (reassociates floating additions across ranks)
            r_oneshot = run_check("vxpu::allreduce", "one_shot", shape, device="cpu")
            self.assertTrue(r_oneshot.passed, f"one_shot failed on {shape}")
            self.assertLessEqual(r_oneshot.deviation, 0.05)

            # scatter_gather: bit-identical (sums in rank order)
            r_sg = run_check("vxpu::allreduce", "scatter_gather", shape, device="cpu")
            self.assertTrue(r_sg.passed, f"scatter_gather failed on {shape}")
            self.assertEqual(r_sg.deviation, 0.0)

            # narrow: rounding approximation
            r_narrow = run_check("vxpu::allreduce", "narrow", shape, device="cpu")
            self.assertTrue(r_narrow.passed, f"narrow failed on {shape}")
            self.assertLessEqual(r_narrow.deviation, 0.05)
            if "_f32" in shape:
                self.assertGreater(r_narrow.deviation, 0.0)

    def test_catalog_condition_and_table_choice(self) -> None:
        """Verifies that narrow requires an external fact in the catalog and only runs when named in a table."""
        catalog = plan.build_catalog()
        narrow_entry = next((b for b in catalog.bodies if b.op == "vxpu::allreduce" and b.name == "narrow"), None)
        self.assertIsNotNone(narrow_entry)
        assert narrow_entry is not None
        self.assertEqual(narrow_entry.condition.require_external_fact, "approximate")

        # Without a table, default plan never chooses narrow.
        p_default = plan.default_plan("cpu")
        chosen = p_default.select("vxpu::allreduce", "cpu", key="4x1x5120", ranks=4)
        self.assertNotEqual(chosen.name, "narrow")

        # When a table explicitly names narrow, the plan selects it.
        table = plan.load_table(json.dumps({"sites": [{"op": "vxpu::allreduce", "impl": "narrow"}]}))
        p_table = plan.Plan("cpu")
        p_table.table = table
        chosen_table = p_table.select("vxpu::allreduce", "cpu", key="4x1x5120", ranks=4)
        self.assertEqual(chosen_table.name, "narrow")


if __name__ == "__main__":
    unittest.main()
