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

"""The GEMV templates' row tiling (hero run 12): the shape arithmetic that needs no GPU."""

import unittest

from . import triton_ops as T


class RowTiles(unittest.TestCase):
    def test_one_pass_up_to_the_gemv_limit(self):
        for m in (1, 9, 33, 64):
            bm, bn, bk = T._rows_blocks(m, 8192)
            self.assertGreaterEqual(bm, m)
            self.assertLessEqual(bm, T.GEMV_ROWS_MAX)
            self.assertEqual(T.row_tiles(m, bm), 1)
            self.assertEqual(T._grid(1, 160, 1), (160, 1))  # the untiled launch is the launch it was

    def test_more_rows_tile_at_the_gemv_limit(self):
        for m, tiles in ((65, 2), (66, 2), (99, 2), (128, 2), (129, 3), (132, 3), (1024, 16)):
            bm, bn, bk = T._rows_blocks(m, 8192)
            self.assertEqual(bm, T.GEMV_ROWS_MAX, m)
            self.assertEqual(T.row_tiles(m, bm), tiles, m)
            self.assertEqual(T._grid(tiles, 160, 2), (tiles, 160, 2))  # the row tile is the fastest axis

    def test_the_class_names_the_tiles_only_when_there_are_several(self):
        one = T.gemv_class("gemv_rows", 33, 5120, 8192, "fp8", native=True, tiles=0)
        self.assertNotIn("tiles", one)
        wide = T.gemv_class("gemv_rows", 132, 5120, 8192, "fp8", native=True, tiles=3)
        self.assertEqual(wide, "gemv_rows rows=132 n=5120 k=8192 w=fp8 native=1 tiles=3")


if __name__ == "__main__":
    unittest.main()

class ExpertKnobsTest(unittest.TestCase):
    """Every fp4 expert knob the plan's table can carry is an attribute the kernels read (hero
    run 15 added the GEMM form's), and the GEMM form's env override shapes its tiles."""

    def test_table_knobs_are_kernel_attributes(self):
        from vxpu import plan
        for name, kind in plan.KNOBS.items():
            if not name.startswith(("FP4_", "GEMV_", "FFN_", "FRONT_", "MXFP8_", "FOLD_", "ALLREDUCE_")):
                continue
            self.assertTrue(hasattr(T, name), name)
            if name.startswith(("FP4_GG_", "FP4_GD_", "FP4_GEMM_")):  # the GEMM form's tiles are numbers by default
                self.assertIsInstance(getattr(T, name), int, name)

    def test_gemm_form_defaults_and_threshold(self):
        self.assertEqual(T.FP4_GEMM_MIN_PAIRS, T.FP4_DIRECT_MAX + 1)
        for prefix in ("GG", "GD"):
            bm, bn, bk = (getattr(T, f"FP4_{prefix}_{k}") for k in ("BM", "BN", "BK"))
            self.assertTrue(bm in (16, 32, 64) and bn % 16 == 0 and bk % 32 == 0, (prefix, bm, bn, bk))



class DirectCompactTest(unittest.TestCase):
    """The direct expert kernels' compact grid (hero run 15): the host arithmetic that needs no GPU."""

    def test_scan_width_is_a_power_of_two_at_or_above_the_pairs(self):
        for pairs, width in ((1, 8), (6, 8), (8, 8), (9, 16), (18, 32), (30, 32), (48, 64), (64, 64), (65, 128)):
            self.assertEqual(T._fp4_scan_width(pairs), width, pairs)

    def test_the_knobs_are_declared(self):
        from vxpu import plan
        self.assertIs(plan.KNOBS["FP4_DIRECT_COMPACT"], bool)
        self.assertIn(T.FP4_DIRECT_COMPACT, ("0", "1"))


@unittest.skipUnless(__import__("torch").cuda.is_available(), "needs a GPU")
class DirectCompactGPUTest(unittest.TestCase):
    """The compact grid against the plain direct kernels on random banks: the same arithmetic in
    another dispatch order, so bit-identical, for the token sums and the per-pair replies, with
    pairs held, not held, and tokens with no held pair."""

    def setUp(self):
        import torch
        self.torch = torch
        dev = torch.device("cuda:0")
        torch.manual_seed(1)
        n, topk, S, I, K = 3, 6, 3, 2304, 5120
        self.x = (torch.randn(n, K, device=dev) * 0.5).to(torch.bfloat16)
        self.w = torch.rand(n, topk, device=dev)
        # experts 0, 1 and 6 held; the third token has none
        self.idx = torch.tensor([[0, 1, 2, 3, 4, 6], [1, 6, 7, 2, 0, 3], [2, 3, 4, 5, 7, 9]], device=dev)
        self.smap = torch.tensor([0, 1, -1, -1, -1, -1, 2, -1, -1, -1], device=dev)

        def bank(rows, inner):
            wq = torch.randint(0, 256, (S, rows, inner // 2), device=dev, dtype=torch.uint8)
            sc = torch.randint(120, 128, (S, rows, inner // 32), device=dev, dtype=torch.uint8)
            return wq, sc

        self.banks = (*bank(I, K), *bank(I, K), *bank(K, I))
        self.saved = {k: getattr(T, k) for k in ("FP4_DIRECT", "FP4_DIRECT_COMPACT", "FP4_GU_KS", "FP4_DN_KS")}
        T.FP4_DIRECT = "1"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(T, k, v)

    def run_once(self, per_pair):
        return T.fp4_experts_decode(self.x, self.w, self.smap, *self.banks, 7.0, per_pair=per_pair, indices=self.idx,
                                    start=0)

    def test_compact_grids_are_bit_identical_to_the_direct_kernels(self):
        for ks in ("1", "2"):
            T.FP4_GU_KS = T.FP4_DN_KS = ks
            T.FP4_DIRECT_COMPACT = "0"
            ref = [self.run_once(False).clone(), self.run_once(True).clone()]
            self.assertTrue(ref[0].abs().sum() > 0)
            self.assertTrue(self.torch.equal(ref[1][2], self.torch.zeros_like(ref[1][2])))  # no held pair
            self.assertTrue(self.torch.equal(ref[0][2], self.torch.zeros_like(ref[0][2])))
            T.FP4_DIRECT_COMPACT = "1"
            for want, per_pair in zip(ref, (False, True)):
                got = self.run_once(per_pair)
                self.assertTrue(self.torch.equal(got, want), (ks, per_pair))
