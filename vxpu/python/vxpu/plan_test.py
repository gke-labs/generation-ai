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

"""Unit tests for plan.py: effective knobs resolution, environment overrides,
and result line formatting (GPU-free)."""

import os
import unittest

from vxpu.plan import default_table, effective_knobs, format_knobs_line


class TestEffectiveKnobs(unittest.TestCase):

    def setUp(self):
        self._saved_env = dict(os.environ)
        # Clear any VXPU_* environment variables that could interfere
        for k in list(os.environ):
            if k.startswith("VXPU_"):
                del os.environ[k]

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("VXPU_") and k not in self._saved_env:
                del os.environ[k]
        os.environ.update(self._saved_env)

    def test_scatter_threshold_by_rows(self):
        """Given a table with a scatter threshold of 16, a run at 40 rows has the
        scatter form on; at 8 rows, off; and the printed line shows it."""
        table = {
            "format": "vxpu-kernel-table/1",
            "knobs": {
                "ALLREDUCE_SCATTER_MIN_ROWS": 16,
            },
        }

        # 40 rows: scatter form is on
        eff_40 = effective_knobs(table, rows=40)
        self.assertTrue(eff_40["scatter"])
        self.assertEqual(eff_40["sum"], "scatter")
        self.assertEqual(eff_40["sum_threshold"], 16)
        line_40 = format_knobs_line(eff_40)
        self.assertIn("sum=scatter:16", line_40)
        self.assertNotIn("override", line_40)

        # 8 rows: scatter form is off (one-shot sum)
        eff_8 = effective_knobs(table, rows=8)
        self.assertFalse(eff_8["scatter"])
        self.assertEqual(eff_8["sum"], "one-shot")
        self.assertEqual(eff_8["sum_threshold"], 16)
        line_8 = format_knobs_line(eff_8)
        self.assertIn("sum=one-shot:16", line_8)
        self.assertNotIn("override", line_8)

    def test_environment_override_wins_and_marked(self):
        """An environment override wins over the table and is marked as an
        override in the line."""
        table = {
            "format": "vxpu-kernel-table/1",
            "knobs": {
                "ALLREDUCE_SCATTER_MIN_ROWS": 16,
            },
        }

        # Override forcing scatter off at 40 rows
        os.environ["VXPU_ALLREDUCE_SCATTER"] = "0"
        eff_override_off = effective_knobs(table, rows=40)
        self.assertFalse(eff_override_off["scatter"])
        self.assertEqual(eff_override_off["sum"], "one-shot")
        line_override_off = format_knobs_line(eff_override_off)
        self.assertIn("sum=one-shot:16(override)", line_override_off)
        self.assertIn("sum", eff_override_off["overrides"])

        # Override forcing scatter on at 8 rows
        os.environ["VXPU_ALLREDUCE_SCATTER"] = "1"
        eff_override_on = effective_knobs(table, rows=8)
        self.assertTrue(eff_override_on["scatter"])
        self.assertEqual(eff_override_on["sum"], "scatter")
        line_override_on = format_knobs_line(eff_override_on)
        self.assertIn("sum=scatter:16(override)", line_override_on)
        self.assertIn("sum", eff_override_on["overrides"])

        # Override scatter threshold via VXPU_ALLREDUCE_SCATTER_MIN_ROWS
        del os.environ["VXPU_ALLREDUCE_SCATTER"]
        os.environ["VXPU_ALLREDUCE_SCATTER_MIN_ROWS"] = "64"
        eff_thresh_override = effective_knobs(table, rows=40)
        self.assertFalse(eff_thresh_override["scatter"])
        self.assertEqual(eff_thresh_override["sum"], "one-shot")
        self.assertEqual(eff_thresh_override["sum_threshold"], 64)
        line_thresh = format_knobs_line(eff_thresh_override)
        self.assertIn("sum=one-shot:64(override)", line_thresh)

    def test_default_planner_table_when_none_provided(self):
        """With no table given, the planner default table provides the scatter
        threshold of 16."""
        # 40 rows with default planner table
        eff_40 = effective_knobs(None, rows=40)
        self.assertTrue(eff_40["scatter"])
        self.assertEqual(eff_40["sum"], "scatter")
        self.assertEqual(eff_40["sum_threshold"], 16)
        line_40 = format_knobs_line(eff_40)
        self.assertIn("sum=scatter:16", line_40)

        # 8 rows with default planner table
        eff_8 = effective_knobs(None, rows=8)
        self.assertFalse(eff_8["scatter"])
        self.assertEqual(eff_8["sum"], "one-shot")
        self.assertEqual(eff_8["sum_threshold"], 16)
        line_8 = format_knobs_line(eff_8)
        self.assertIn("sum=one-shot:16", line_8)

    def test_printed_line_contains_all_knobs_in_effect(self):
        """Every result line prints the knobs in effect (sum form and threshold,
        expert form, fused push threshold, FP4_PDL, expert_tp, class digest)."""
        table = default_table()
        eff = effective_knobs(table, rows=1, expert_tp=4)
        line = format_knobs_line(eff, prefix="LEAGUE-CLASS")

        self.assertTrue(line.startswith("LEAGUE-CLASS "))
        self.assertIn("sum=one-shot:16", line)
        self.assertIn("expert=direct:48", line)
        self.assertIn("push=32", line)
        self.assertIn("pdl=0", line)
        self.assertIn("expert_tp=4", line)
        self.assertIn(f"digest={eff['digest'][:8]}", line)

        # CLASS prefix for step harness
        class_line = format_knobs_line(eff, prefix="CLASS")
        self.assertTrue(class_line.startswith("CLASS "))

    def test_expert_and_pdl_env_overrides(self):
        """Test expert, push, pdl and expert_tp environment overrides."""
        table = default_table()
        os.environ["VXPU_FP4_PDL"] = "2"
        os.environ["VXPU_GEMV_PUSH_BN"] = "16"
        os.environ["VXPU_EXPERT_TP"] = "4"
        os.environ["VXPU_FP4_DIRECT"] = "0"

        eff = effective_knobs(table, rows=1)
        line = format_knobs_line(eff)

        self.assertEqual(eff["pdl"], 2)
        self.assertIn("pdl=2(override)", line)
        self.assertEqual(eff["push"], 16)
        self.assertIn("push=16(override)", line)
        self.assertEqual(eff["expert_tp"], 4)
        self.assertIn("expert_tp=4(override)", line)
        self.assertEqual(eff["expert"], "gemm")
        self.assertIn("expert=gemm(override)", line)


if __name__ == "__main__":
    unittest.main()
