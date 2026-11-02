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

"""The speculation policies on synthetic inputs: acceptance, the confidence stop, the expert
preference and budget, the stats the record carries, and the plan section's checks."""
import unittest

from vxpu import speculation as sp


class AcceptanceTest(unittest.TestCase):
    def test_keeps_drafts_until_the_first_that_differs(self):
        # the verify chose 7, 8, 9, 10 at the four positions; the drafts were 7, 8, 5, 10
        self.assertEqual(sp.accept([7, 8, 5, 10], [7, 8, 9, 10]), 2)
        self.assertEqual(sp.accept([7, 8, 9, 10], [7, 8, 9, 10]), 4)
        self.assertEqual(sp.accept([1, 8, 9, 10], [7, 8, 9, 10]), 0)
        self.assertEqual(sp.accept([], []), 0)


class DraftLengthTest(unittest.TestCase):
    def test_fixed_issues_every_draft(self):
        self.assertEqual(sp.draft_length([0.9, 0.1, 0.1, 0.1], sp.Policy(draft=4)), 4)
        self.assertEqual(sp.draft_length([0.9, 0.1], sp.Policy(draft=4)), 2)

    def test_confidence_stops_at_the_first_unsure_draft(self):
        p = sp.Policy(draft=4, policy=sp.CONFIDENCE, min_confidence=0.5)
        self.assertEqual(sp.draft_length([0.9, 0.6, 0.4, 0.8], p), 2)
        self.assertEqual(sp.draft_length([0.2, 0.9, 0.9, 0.9], p), 0)
        self.assertEqual(sp.draft_length([0.9, 0.9, 0.9, 0.9], p), 4)


class ExpertsPolicyTest(unittest.TestCase):
    reading = [[1, 2, 3], [4, 5, 6]]  # two layers, the experts the batch reads for its first tokens

    def test_prefers_the_candidate_already_read(self):
        a = [[1, 2, 9], [4, 7, 8]]  # adds 1 + 2 experts
        b = [[1, 2, 3], [4, 5, 9]]  # adds 0 + 1
        self.assertEqual(sp.prefer([a, b], self.reading), 1)
        self.assertEqual(sp.prefer([b, a], self.reading), 0)
        self.assertEqual(sp.new_experts(a, self.reading), 3)
        # a tie goes to the drafter's first (most probable) candidate
        self.assertEqual(sp.prefer([a, a], self.reading), 0)

    def test_budget_cuts_the_drafts(self):
        p = sp.Policy(draft=3, policy=sp.EXPERTS, max_new_experts=1, candidates=2)  # one new expert a layer: 2 in all
        cheap = [[1, 2, 3], [4, 5, 9]]  # 1 new
        dear = [[7, 8, 9], [10, 11, 12]]  # 6 new
        positions = [[dear, cheap], [dear, cheap], [dear, cheap]]
        # cheap at each position; the second cheap draft repeats expert 9 so it adds nothing; the third too
        self.assertEqual(sp.plan_drafts(positions, self.reading, p), [1, 1, 1])
        positions = [[dear, cheap], [dear], [cheap]]
        self.assertEqual(sp.plan_drafts(positions, self.reading, p), [1])  # the dear-only position breaks the budget
        fixed = sp.Policy(draft=3, policy=sp.FIXED)
        self.assertEqual(sp.plan_drafts(positions, self.reading, fixed), [0, 0, 0])  # fixed takes the first candidate throughout


class StatsTest(unittest.TestCase):
    def test_per_position_rates_and_confidence(self):
        s = sp.Stats(positions=4)
        s.record(issued=4, accepted=2, draft_probs=[0.9, 0.8, 0.3, 0.2])
        s.record(issued=4, accepted=0, draft_probs=[0.4, 0.8, 0.3, 0.2])
        s.record(issued=2, accepted=2, draft_probs=[0.7, 0.6])
        out = s.summary()
        self.assertEqual((out["steps"], out["issued"], out["accepted"]), (3, 10, 4))
        self.assertEqual(out["alpha"], 0.4)
        first = out["positions"][0]
        self.assertEqual((first["issued"], first["accepted"], first["rate"]), (3, 2, 0.667))
        self.assertEqual(first["confidence_accepted"], 0.8)
        self.assertEqual(first["confidence_rejected"], 0.4)
        third = out["positions"][2]
        self.assertEqual((third["issued"], third["accepted"]), (2, 0))
        self.assertIsNone(third["confidence_accepted"])


class PlanSectionTest(unittest.TestCase):
    def test_reads_and_checks_the_plan_section(self):
        self.assertEqual(sp.Policy.from_plan(None), sp.Policy())
        p = sp.Policy.from_plan({"draft": 4, "policy": "confidence", "min_confidence": 0.5})
        self.assertEqual((p.draft, p.policy, p.min_confidence), (4, "confidence", 0.5))
        with self.assertRaises(ValueError):
            sp.Policy.from_plan({"draft": 4, "policy": "lucky"})
        with self.assertRaises(ValueError):
            sp.Policy.from_plan({"draft": 4, "policy": "confidence"})
        with self.assertRaises(ValueError):
            sp.Policy.from_plan({"draft": 9, "policy": "fixed"})
        with self.assertRaises(ValueError):
            sp.Policy.from_plan({"draft": 2, "policy": "experts", "candidates": 0})


class RecordsTest(unittest.TestCase):
    """The native loop's records (hero run 12) replayed under every policy."""

    def events(self):
        # four drafts a step; rows of [accepted, score_1..score_4] at the 1e6 scale
        return [[4, 900000, 850000, 800000, 700000],   # all kept
                [1, 950000, 200000, 600000, 100000],   # the second draft rejected with low confidence
                [0, 300000, 900000, 900000, 900000],   # the first rejected, unsure
                [2, 800000, 750000, 450000, 900000]]   # the third rejected at 0.45

    def test_stats_from_events(self):
        st = sp.stats_from_events(self.events(), 4, 1e6)
        s = st.summary()
        self.assertEqual((s["steps"], s["issued"], s["accepted"]), (4, 16, 7))
        self.assertEqual([p["rate"] for p in s["positions"]], [0.75, 0.5, 0.25, 0.25])
        # thresholds: of the four first-rejected drafts (0.2, 0.3, 0.45, and none in the first row) at 0.5 three
        # are foreseen; of the seven accepted drafts none is under 0.5
        self.assertEqual(s["thresholds"]["0.5"], {"precision": 1.0, "loss": 0.0})
        self.assertEqual(s["thresholds"]["0.3"]["precision"], round(1 / 3, 3))

    def test_no_confidence(self):
        st = sp.stats_from_events([[2, -1, -1, -1, -1], [0, -1, -1, -1, -1]], 4, 1e6)
        self.assertNotIn("thresholds", st.summary())
        self.assertIsNone(sp.evaluate(st, sp.Policy(draft=4, policy=sp.CONFIDENCE, min_confidence=0.5)))
        fixed = sp.evaluate(st, sp.Policy(draft=1))
        self.assertEqual((fixed["tokens_per_step"], fixed["token_rows_per_step"]), (1.5, 2.0))

    def test_evaluate_policies(self):
        st = sp.stats_from_events(self.events(), 4, 1e6)
        fixed4 = sp.evaluate(st, sp.Policy(draft=4))
        self.assertEqual(fixed4["tokens_per_step"], round((5 + 2 + 1 + 3) / 4, 3))
        fixed2 = sp.evaluate(st, sp.Policy(draft=2))
        self.assertEqual((fixed2["tokens_per_step"], fixed2["token_rows_per_step"]), (round((3 + 2 + 1 + 3) / 4, 3), 3.0))
        conf = sp.evaluate(st, sp.Policy(draft=4, policy=sp.CONFIDENCE, min_confidence=0.5))
        # issued: 4 (none under 0.5), 1 (stops at 0.2), 0 (stops at 0.3), 2 (stops at 0.45): kept 4, 1, 0, 2
        self.assertEqual((conf["tokens_per_step"], conf["token_rows_per_step"]), (round(11 / 4, 3), round(1 + 7 / 4, 3)))
        self.assertEqual(conf["acceptance"], 1.0)
        names = [(o["policy"], o["draft"], o.get("min_confidence")) for o in sp.evaluate_all(st, 4)]
        self.assertEqual(names[:4], [("fixed", 1, None), ("fixed", 2, None), ("fixed", 3, None), ("fixed", 4, None)])
        self.assertEqual(len(names), 4 + len(sp.THRESHOLDS))


class ComposeStepTest(unittest.TestCase):
    """The verify's token rows under a policy (hero run 13): the export fixes the step's token rows,
    so the positions past the issued drafts carry the last issued token; the count issued is what
    the statistics see."""

    def test_no_policy_issues_every_draft(self):
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], None, None, 5), ([7, 1, 2, 3, 4], 4))

    def test_fixed_shorter_than_the_block_pads_with_the_last_draft(self):
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], None, sp.Policy(draft=2), 5), ([7, 1, 2, 2, 2], 2))
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], None, sp.Policy(draft=0), 5), ([7, 7, 7, 7, 7], 0))

    def test_confidence_stops_and_pads(self):
        policy = sp.Policy(draft=4, policy=sp.CONFIDENCE, min_confidence=0.5)
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], [0.9, 0.6, 0.4, 0.8], policy, 5), ([7, 1, 2, 2, 2], 2))
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], [0.1, 0.6, 0.4, 0.8], policy, 5), ([7, 7, 7, 7, 7], 0))
        self.assertEqual(sp.compose_step(7, [1, 2, 3, 4], [0.9, 0.9, 0.9, 0.9], policy, 5), ([7, 1, 2, 3, 4], 4))

    def test_padded_positions_are_not_issued_in_the_records(self):
        # the native loop's record of a row that issued two of four drafts, both accepted, the two padded
        # positions marked -2 (the verify agreed with one of them: n is 3 in the loop, 2 in the record)
        st = sp.stats_from_events([[2, 900000, 600000, -2, -2]], 4, 1e6)
        self.assertEqual((st.issued, st.accepted), ([1, 1, 0, 0], [1, 1, 0, 0]))
        self.assertEqual(st.events, [(2, [0.9, 0.6])])


if __name__ == "__main__":
    unittest.main()


class ExpertsReplayTest(unittest.TestCase):
    """The expert-aware policy replayed offline over a synthetic trace: two rows, a block of
    three (two drafts), two layers, top-2 of 8 experts."""

    def trace(self):
        import numpy as np
        S, rows, k = 3, 2, 2
        # step 0: row 0's drafts reuse the read experts (0 new), row 1's first draft adds two
        # experts a layer and its second two more; step 1: everything new
        t0 = np.array([[[0, 1], [0, 1], [1, 0]], [[0, 1], [2, 3], [4, 5]]]).reshape(rows * S, k)
        t1 = np.array([[[0, 1], [2, 3], [4, 5]], [[0, 1], [6, 7], [2, 3]]]).reshape(rows * S, k)
        trace = {0: np.stack([t0, t1]), 1: np.stack([t0, t1])}
        events = [[2, -1, -1], [1, -1, -1], [1, -1, -1], [2, -1, -1]]  # step-major: (step 0, row 0), (0, 1), (1, 0), (1, 1)
        return trace, events, rows, S

    def test_budgets_cut_the_costly_drafts_and_keep_their_tokens_accordingly(self):
        trace, events, rows, S = self.trace()
        out = sp.experts_replay(trace, events, rows, S, budgets=(1, 2, 4))
        self.assertEqual(out["plain"]["tokens_per_step"], 1.0)
        self.assertEqual(out["plain"]["distinct_experts"], 2.0)  # both rows' first tokens read experts 0 and 1
        # fixed: every draft kept; tokens = 1 + accepted: (1+2 + 1+1)/2 and (1+1 + 1+2)/2
        self.assertEqual(out["fixed"]["tokens_per_step"], 2.5)
        self.assertEqual(out["fixed"]["token_rows_per_step"], 6.0)
        self.assertEqual(out["fixed"]["distinct_experts"], (6 + 8) / 2)
        # budget 1 a layer: row 0's drafts in step 0 are free (kept), row 1's first adds 2 (dropped);
        # step 1: every draft adds 2 (dropped)
        b1 = out["experts-1"]
        self.assertEqual(b1["tokens_per_step"], ((1 + 2 + 1) / 2 + (1 + 1) / 2) / 2)
        self.assertEqual(b1["distinct_experts"], 2.0)
        self.assertEqual(b1["kept_by_position"], [0.25, 0.25])
        # budget 2 a layer: one costly draft a row fits
        b2 = out["experts-2"]
        self.assertEqual(b2["kept_by_position"][0], 1.0)
        self.assertEqual(b2["new_experts_by_position"][0], 1.5)  # (0 + 2 + 2 + 2) / 4
        # budget 4: everything fits, same as fixed
        self.assertEqual(out["experts-4"]["tokens_per_step"], out["fixed"]["tokens_per_step"])
