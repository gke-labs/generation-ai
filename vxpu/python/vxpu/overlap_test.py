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

"""Tests for vxpu.overlap (GPU-free fixture acceptance)."""

import unittest

from vxpu import overlap


class OverlapTest(unittest.TestCase):

    def test_fixture_hero15_block7(self):
        """A fixture matching the Hero 15 8-row timeline:
        - expert host node: 6.34 ms
        - shared-expert island: 2.31 ms, completely hidden inside the expert node
        - attention projections: attn_q (1.08 ms) and window_kv (0.48 ms side-branch work)
        - attention core and out: 1.26 ms
        - sum sites: 2.63 ms
        - ffn front: 0.78 ms
        - hc front: 0.76 ms
        - dense other: 0.75 ms
        """
        # Spans in nanoseconds (1 ms = 1,000,000 ns)
        nodes = [
            # Routed experts on main chain: [0.0, 6.34 ms]
            {"name": "experts_0", "kind": "experts", "class": "routed_experts", "branch": "main",
             "start_ns": 0, "end_ns": 6340000},
            # Shared expert island on side branch: [0.50 ms, 2.81 ms] (duration 2.31 ms) inside experts_0
            {"name": "shared_ffn_front_0", "kind": "dense", "class": "shared_expert", "branch": "par",
             "start_ns": 500000, "end_ns": 2810000},
            # Attention projections: attn_q on main [7.00 ms, 8.08 ms] (1.08 ms), window_kv on par [7.00 ms, 7.48 ms] (0.48 ms)
            {"name": "attn_q_0", "kind": "dense", "class": "attention_projections", "branch": "main",
             "start_ns": 7000000, "end_ns": 8080000},
            {"name": "window_kv_0", "kind": "dense", "class": "attention_projections", "branch": "par",
             "start_ns": 7000000, "end_ns": 7480000},
            # Attention core: [8.10 ms, 9.36 ms] (1.26 ms)
            {"name": "attn_core_0", "kind": "dense", "class": "attention_core", "branch": "main",
             "start_ns": 8100000, "end_ns": 9360000},
            # Sum sites: [9.40 ms, 12.03 ms] (2.63 ms)
            {"name": "allreduce_0", "kind": "allreduce", "class": "sums", "branch": "main",
             "start_ns": 9400000, "end_ns": 12030000},
            # FFN front: [12.10 ms, 12.88 ms] (0.78 ms)
            {"name": "ffn_front_0", "kind": "dense", "class": "ffn_front", "branch": "main",
             "start_ns": 12100000, "end_ns": 12880000},
            # HC front: [12.90 ms, 13.66 ms] (0.76 ms)
            {"name": "hc_front_0", "kind": "dense", "class": "hc_front", "branch": "main",
             "start_ns": 12900000, "end_ns": 13660000},
            # Dense other: [13.70 ms, 14.45 ms] (0.75 ms)
            {"name": "dense_other_0", "kind": "dense", "class": "dense_other", "branch": "main",
             "start_ns": 13700000, "end_ns": 14450000},
        ]

        result = overlap.measure_step_overlap(nodes)

        # Expected busy: sum of all durations = 6.34 + 2.31 + 1.08 + 0.48 + 1.26 + 2.63 + 0.78 + 0.76 + 0.75 = 16.39 ms
        self.assertAlmostEqual(result["busy_ms"], 16.39, places=2)

        # Expected covered: union of all intervals:
        # [0.0, 6.34] = 6.34
        # [7.0, 8.08] = 1.08
        # [8.10, 9.36] = 1.26
        # [9.40, 12.03] = 2.63
        # [12.10, 12.88] = 0.78
        # [12.90, 13.66] = 0.76
        # [13.70, 14.45] = 0.75
        # Total covered = 13.60 ms
        self.assertAlmostEqual(result["covered_ms"], 13.60, places=2)

        # Overlap = busy - covered = 16.39 - 13.60 = 2.79 ms (~3 ms)
        self.assertAlmostEqual(result["overlap_ms"], 2.79, places=2)

        # Wall = 14.45 ms - 0.0 ms = 14.45 ms
        self.assertAlmostEqual(result["wall_ms"], 14.45, places=2)

        # Hidden work: shared_ffn_front_0 lies entirely inside experts_0
        hidden_names = [h["name"] for h in result["hidden_nodes"]]
        self.assertIn("shared_ffn_front_0", hidden_names)
        shared_hidden = next(h for h in result["hidden_nodes"] if h["name"] == "shared_ffn_front_0")
        self.assertEqual(shared_hidden["inside"], "experts_0")
        self.assertAlmostEqual(shared_hidden["duration_ms"], 2.31, places=2)

        # Window_kv is also hidden inside attn_q_0 in this fixture
        self.assertIn("window_kv_0", hidden_names)
        window_hidden = next(h for h in result["hidden_nodes"] if h["name"] == "window_kv_0")
        self.assertEqual(window_hidden["inside"], "attn_q_0")

        # Per-class verification
        classes = result["classes"]
        self.assertIn("routed_experts", classes)
        self.assertAlmostEqual(classes["routed_experts"]["busy_ms"], 6.34, places=2)
        self.assertAlmostEqual(classes["routed_experts"]["side_branch_share"], 0.0, places=2)

        self.assertIn("shared_expert", classes)
        self.assertAlmostEqual(classes["shared_expert"]["busy_ms"], 2.31, places=2)
        self.assertAlmostEqual(classes["shared_expert"]["side_branch_ms"], 2.31, places=2)
        self.assertAlmostEqual(classes["shared_expert"]["side_branch_share"], 1.0, places=2)

        self.assertIn("attention_projections", classes)
        # 1.08 + 0.48 = 1.56 ms busy, 0.48 ms side branch
        self.assertAlmostEqual(classes["attention_projections"]["busy_ms"], 1.56, places=2)
        self.assertAlmostEqual(classes["attention_projections"]["side_branch_ms"], 0.48, places=2)
        self.assertAlmostEqual(classes["attention_projections"]["side_branch_share"], 0.48 / 1.56, places=2)


if __name__ == "__main__":
    unittest.main()
