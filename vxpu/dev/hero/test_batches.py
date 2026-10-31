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

import unittest
import os
import sys

# Ensure dev/hero is in sys.path
hero_dir = os.path.dirname(os.path.abspath(__file__))
if hero_dir not in sys.path:
    sys.path.insert(0, hero_dir)

from batches import batch_prompts, summarize


class FakeEngine:
    """Fake engine for unit testing batching harness without GPU or real weights."""

    def __init__(self, step_tokens=1):
        self.step_tokens = step_tokens

    def generate_batches(self, prompt_lists, max_new_tokens, params=None):
        results = []
        for lst in prompt_lists:
            # Deterministic token generation: each prompt produces its own distinct tokens
            token_ids = []
            for p in lst:
                prompt_key = p[0] if p else 0
                token_ids.append([prompt_key * 100 + j for j in range(max_new_tokens)])
            step_ms = [25.0] * max_new_tokens
            rows_active = [len(lst)] * max_new_tokens
            results.append({
                "token_ids": token_ids,
                "step_ms": step_ms,
                "rows_active": rows_active,
                "prefill_ms": [5.0] * len(lst),
                "drafted": 0,
                "accepted": 0,
            })
        return results


class BatchesTest(unittest.TestCase):
    def test_batch_prompts_same_mode(self):
        """In 'same' mode, every row in every batch must hold prompt 0 and index 0."""
        ids = [[10, 11], [20, 21], [30, 31]]
        rows = 4
        for b in range(4):
            prompts, idx = batch_prompts(ids, rows=rows, mode="same", b=b)
            self.assertEqual(prompts, [ids[0]] * rows, f"batch {b} must hold prompt 0 in every row")
            self.assertEqual(idx, [0] * rows, f"batch {b} must have prompt index 0 in every row")

    def test_batch_prompts_one_mode(self):
        """In 'one' mode, batch b must hold prompt b % len(ids) and index b % len(ids)."""
        ids = [[10, 11], [20, 21]]
        for b in (0, 1):
            prompts, idx = batch_prompts(ids, rows=4, mode="one", b=b)
            self.assertEqual(prompts, [ids[b % len(ids)]])
            self.assertEqual(idx, [b % len(ids)])

    def test_batch_prompts_different_mode(self):
        """In 'different' mode, rows hold distinct prompts wrapping across batches."""
        ids = [[0], [1], [2], [3]]
        prompts0, idx0 = batch_prompts(ids, rows=2, mode="different", b=0)
        self.assertEqual(idx0, [0, 1])
        self.assertEqual(prompts0, [ids[0], ids[1]])
        prompts1, idx1 = batch_prompts(ids, rows=2, mode="different", b=1)
        self.assertEqual(idx1, [2, 3])
        self.assertEqual(prompts1, [ids[2], ids[3]])

    def test_drafter_throughput_calculation(self):
        """decode_tokens_per_s must equal tokens a step over median step time within 1%."""
        new_tokens = 64
        num_steps = 25
        median_ms = 25.7
        # 25 steps of 25.7 ms each
        step_ms = [median_ms] * num_steps
        rows_active = [1] * num_steps
        out = {
            "token_ids": [[i for i in range(new_tokens)]],
            "step_ms": step_ms,
            "rows_active": rows_active,
            "prefill_ms": [10.0],
            "drafted": 100,
            "accepted": 39,
        }
        s = summarize(out, prompts=[[0]], idx=[0], reference={"0": list(range(new_tokens))}, new_tokens=new_tokens)
        self.assertEqual(s["tokens"], 64)
        self.assertEqual(s["tokens_per_row_step"], 2.56)
        self.assertEqual(s["median_step_ms"], 25.7)

        # decode_tokens_per_s = 64 / (25 * 0.0257) = 99.6
        expected_tps = s["tokens_per_row_step"] / (s["median_step_ms"] / 1e3)
        self.assertIsNotNone(s["decode_tokens_per_s"])
        rel_diff = abs(s["decode_tokens_per_s"] - expected_tps) / expected_tps
        self.assertLess(rel_diff, 0.01, f"decode_tokens_per_s ({s['decode_tokens_per_s']}) must equal {expected_tps:.1f} within 1%")

    def test_drafter_throughput_ignores_idle_steps(self):
        """Idle steps (rows_active == 0) must be excluded from step_ms in summarize."""
        new_tokens = 64
        num_active = 25
        num_idle = 7
        step_ms = [25.7] * num_active + [25.7] * num_idle
        rows_active = [1] * num_active + [0] * num_idle
        out = {
            "token_ids": [[i for i in range(new_tokens)]],
            "step_ms": step_ms,
            "rows_active": rows_active,
            "prefill_ms": [10.0],
            "drafted": 100,
            "accepted": 39,
        }
        s = summarize(out, prompts=[[0]], idx=[0], reference={"0": list(range(new_tokens))}, new_tokens=new_tokens)
        self.assertEqual(s["tokens_per_row_step"], 2.56)
        self.assertEqual(s["decode_steps"], 25)
        expected_tps = s["tokens_per_row_step"] / (s["median_step_ms"] / 1e3)
        rel_diff = abs(s["decode_tokens_per_s"] - expected_tps) / expected_tps
        self.assertLess(rel_diff, 0.01)

    def test_fake_engine_two_batches_two_prompts(self):
        """A fake engine with two batches and two prompts asserts every batch compares against the right prompt."""
        engine = FakeEngine(step_tokens=1)
        prompts = [[1, 2], [2, 3]]  # prompt 0 starts with 1, prompt 1 starts with 2
        rows = 2
        new_tokens = 16

        # Reference dictionary holding the expected tokens per prompt index
        reference = {
            "0": [1 * 100 + j for j in range(new_tokens)],
            "1": [2 * 100 + j for j in range(new_tokens)],
        }

        # 1. Test "same" mode (two batches): every row of every batch holds prompt 0
        batches_same = [batch_prompts(prompts, rows, "same", b) for b in range(2)]
        outs_same = engine.generate_batches([p for p, _ in batches_same], new_tokens)
        for b in range(2):
            s = summarize(outs_same[b], batches_same[b][0], batches_same[b][1], reference, new_tokens)
            self.assertEqual(s["of"], rows, f"batch {b} must have reference for all rows")
            self.assertEqual(s["match"], rows, f"batch {b} in same mode must compare against prompt 0 and match {rows}/{rows}")
            self.assertEqual(s["same_rows"], True)

        # 2. Test "one" mode (two batches): batch 0 holds prompt 0, batch 1 holds prompt 1
        batches_one = [batch_prompts(prompts, rows, "one", b) for b in range(2)]
        outs_one = engine.generate_batches([p for p, _ in batches_one], new_tokens)
        for b in range(2):
            s = summarize(outs_one[b], batches_one[b][0], batches_one[b][1], reference, new_tokens)
            self.assertEqual(s["of"], 1)
            self.assertEqual(s["match"], 1, f"batch {b} in one mode must compare against prompt {b} and match 1/1")


if __name__ == "__main__":
    unittest.main()
