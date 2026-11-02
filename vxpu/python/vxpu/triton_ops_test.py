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

