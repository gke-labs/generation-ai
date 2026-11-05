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

"""Unit tests for the league benchmark harness (vxpu/dev/hero/league.py)."""

import json
import os
import sys
import unittest

hero_dir = os.path.dirname(os.path.abspath(__file__))
if hero_dir not in sys.path:
    sys.path.insert(0, hero_dir)

from league import (
    PROSE_BASE,
    aggregate_tokens_per_s,
    build_prompts,
    build_prose_prompt,
    encode_prompt,
    load_league_prompts,
    run_benchmark,
    stream_tokens_per_s,
)

PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "league_prompts.json")

EXPECTED_CODE_TASKS = [
    "binary_search",
    "merge_sort",
    "lru_cache",
    "token_bucket",
    "ring_buffer",
    "dijkstra",
    "edit_distance",
    "semver_cmp",
    "url_parse",
    "json_pointer",
    "glob_match",
    "csv_parse",
    "rle",
    "top_k",
    "interval_merge",
    "topo_sort",
]


class TestLeagueFormulas(unittest.TestCase):
    """Verifies per-stream and aggregate tok/s formulas, including the exclusion of prefill and the first token."""

    def test_stream_tokens_per_s_formula(self):
        # 400 completion tokens emitted from t = 1.0 to t = 5.0 (span = 4.0s).
        # (400 - 1) / 4.0 = 99.75 tok/s.
        rate = stream_tokens_per_s(400, 1.0, 5.0)
        self.assertAlmostEqual(rate, 99.75)

    def test_first_token_and_prefill_excluded(self):
        # Prefill taking 0.2s vs 10.0s: both have client duration 4.0s from first token to last token.
        # Prefill time before first_token_time is completely excluded.
        rate_fast_prefill = stream_tokens_per_s(400, 0.2, 4.2)
        rate_slow_prefill = stream_tokens_per_s(400, 10.0, 14.0)
        self.assertAlmostEqual(rate_fast_prefill, 99.75)
        self.assertAlmostEqual(rate_slow_prefill, 99.75)

        # Single completion token means 0 tokens generated after first token: returns 0.0.
        self.assertEqual(stream_tokens_per_s(1, 1.0, 2.0), 0.0)
        self.assertEqual(stream_tokens_per_s(0, 1.0, 2.0), 0.0)

        # Zero or negative duration returns 0.0.
        self.assertEqual(stream_tokens_per_s(400, 5.0, 5.0), 0.0)
        self.assertEqual(stream_tokens_per_s(400, 5.0, 4.0), 0.0)

    def test_aggregate_tokens_per_s_formula(self):
        # Two streams in a wave:
        # Stream 0: 401 completion tokens (400 counted), first token at 2.0s, last at 6.0s.
        # Stream 1: 401 completion tokens (400 counted), first token at 1.0s, last at 5.0s.
        # Sum of (completion_tokens - 1) = 400 + 400 = 800.
        # Earliest first token = 1.0s, latest last token = 6.0s.
        # Total wave window = 6.0 - 1.0 = 5.0s.
        # Aggregate tok/s = 800 / 5.0 = 160.0.
        streams = [
            (401, 2.0, 6.0),
            (401, 1.0, 5.0),
        ]
        rate = aggregate_tokens_per_s(streams)
        self.assertAlmostEqual(rate, 160.0)

    def test_aggregate_single_stream_matches_stream_rate(self):
        streams = [(401, 1.0, 5.0)]
        self.assertAlmostEqual(aggregate_tokens_per_s(streams), 100.0)
        self.assertAlmostEqual(stream_tokens_per_s(401, 1.0, 5.0), 100.0)

    def test_aggregate_empty_or_invalid(self):
        self.assertEqual(aggregate_tokens_per_s([]), 0.0)
        self.assertEqual(aggregate_tokens_per_s([(1, 1.0, 2.0)]), 0.0)
        self.assertEqual(aggregate_tokens_per_s([(400, 5.0, 5.0)]), 0.0)


class TestPromptConstruction(unittest.TestCase):
    """Verifies prose prompt suffixing, code task assignment by stream, and reported server token counts."""

    @classmethod
    def setUpClass(cls):
        cls.prompts_data = load_league_prompts(PROMPTS_PATH)

    def test_prose_prompt_construction(self):
        # Concurrency 1: stream 0 has no suffix.
        p_c1 = build_prose_prompt(0, 1)
        self.assertEqual(p_c1, PROSE_BASE)
        self.assertNotIn("(stream", p_c1)

        # Concurrency c > 1: stream i has suffix " (stream {i+1}/{c})".
        for c in (2, 4, 8, 16):
            prompts = build_prompts(self.prompts_data, "prose", c)
            self.assertEqual(len(prompts), c)
            for i, p in enumerate(prompts):
                suffix = f" (stream {i + 1}/{c})"
                self.assertTrue(p.endswith(suffix), f"Expected {p} to end with {suffix}")
                self.assertEqual(p, f"{PROSE_BASE}{suffix}")

    def test_code_task_assignment_by_stream(self):
        # Stream i receives task i across all concurrency levels.
        for c in (1, 2, 4, 8, 16):
            prompts = build_prompts(self.prompts_data, "code", c)
            self.assertEqual(len(prompts), c)
            for i, p in enumerate(prompts):
                expected_task = EXPECTED_CODE_TASKS[i]
                self.assertTrue(p.startswith(f"{expected_task}\n"), f"Stream {i} should start with {expected_task}")

    def test_reported_server_token_counts(self):
        # Published server reported: prose 32 tokens at c = 1, 39 with a suffix; code 56 to 65.
        for item in self.prompts_data:
            t = item.get("type")
            if "prompt_tokens_server" not in item:
                continue
            server_tokens = item["prompt_tokens_server"]
            if t == "prose":
                c = item.get("concurrency")
                if c == 1:
                    self.assertEqual(server_tokens, 32, "Prose c=1 prompt must be 32 tokens")
                else:
                    self.assertEqual(server_tokens, 39, f"Prose c={c} prompt must be 39 tokens")
            elif t == "code":
                self.assertGreaterEqual(server_tokens, 56, f"Code task {item.get('task')} tokens < 56")
                self.assertLessEqual(server_tokens, 65, f"Code task {item.get('task')} tokens > 65")

    def test_model_tokenizer_token_counts_if_cached(self):
        """Uses the model's tokenizer from cache if present, otherwise skips."""
        tokenizer = None
        try:
            from transformers import AutoTokenizer

            # Check potential local cache locations or hub repo
            repo = os.environ.get("VXPU_MODEL_REPO", "deepseek-ai/DeepSeek-V4.1-Flash")
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
            except Exception:
                # Try finding under HF_HOME
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                if os.path.exists(hf_home):
                    tokenizer = AutoTokenizer.from_pretrained(hf_home, local_files_only=True)
        except Exception:
            pass

        if tokenizer is None:
            raise unittest.SkipTest("Model tokenizer not present in cache")

        # Prose c = 1: 32 tokens
        ids_c1 = encode_prompt(tokenizer, build_prose_prompt(0, 1))
        self.assertEqual(len(ids_c1), 32)

        # Prose c > 1: 39 tokens with suffix
        ids_c2 = encode_prompt(tokenizer, build_prose_prompt(0, 2))
        self.assertEqual(len(ids_c2), 39)

        # Code tasks: 56 to 65 tokens
        for i, task in enumerate(EXPECTED_CODE_TASKS):
            prompts = build_prompts(self.prompts_data, "code", i + 1)
            ids_code = encode_prompt(tokenizer, prompts[i])
            self.assertGreaterEqual(len(ids_code), 56)
            self.assertLessEqual(len(ids_code), 65)


class MockEngine:
    """Mock engine simulating generate_rows with custom timestamps and speculation counters."""

    def __init__(self, step_tokens=1, drafted=0, accepted=0, positions=None):
        self.tokenizer = None
        self.step_tokens = step_tokens
        self.eos_ids = {100}
        self.drafted = drafted
        self.accepted = accepted
        self.positions = positions

    def generate_rows(self, prompts, max_new_tokens, params):
        c = len(prompts)
        out = {
            "token_ids": [[j for j in range(max_new_tokens)] for _ in range(c)],
            "step_ms": [12.5] * max_new_tokens,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "speculation": {"positions": self.positions} if self.positions else None,
        }
        return out


class TestBenchmarkExecution(unittest.TestCase):
    """Tests the benchmark runner with synthetic/mock engine."""

    def test_run_benchmark_mock_prose(self):
        engine = MockEngine()
        prompts_data = load_league_prompts(PROMPTS_PATH)
        levels = [1, 2]
        records = run_benchmark(engine, prompts_data, "prose", levels=levels, max_tokens=10, warmup_tokens=5)
        self.assertEqual(len(records), 2)
        for i, rec in enumerate(records):
            self.assertEqual(rec["type"], "prose")
            self.assertEqual(rec["level"], levels[i])
            self.assertEqual(rec["concurrency"], levels[i])
            self.assertEqual(rec["streams"], levels[i])
            self.assertEqual(len(rec["per_stream_tokens_per_s"]), levels[i])
            self.assertGreater(rec["aggregate_tokens_per_s"], 0)
            self.assertEqual(rec["rows_per_step"], levels[i])
            self.assertIsNone(rec["acceptance"])

    def test_run_benchmark_mock_code(self):
        engine = MockEngine()
        prompts_data = load_league_prompts(PROMPTS_PATH)
        levels = [1, 4]
        records = run_benchmark(engine, prompts_data, "code", levels=levels, max_tokens=10, warmup_tokens=5)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "code")
        self.assertEqual(records[0]["streams"], 1)
        self.assertEqual(records[1]["type"], "code")
        self.assertEqual(records[1]["streams"], 4)

    def test_run_benchmark_with_drafter(self):
        positions = [{"position": 1, "accepted": 8}, {"position": 2, "accepted": 4}]
        engine = MockEngine(step_tokens=3, drafted=16, accepted=12, positions=positions)
        prompts_data = load_league_prompts(PROMPTS_PATH)
        records = run_benchmark(engine, prompts_data, "prose", levels=[2], max_tokens=10, warmup_tokens=5)
        rec = records[0]
        self.assertEqual(rec["drafted"], 16)
        self.assertEqual(rec["accepted"], 12)
        self.assertEqual(rec["acceptance"], [8, 4])
        self.assertGreater(rec["tokens_per_step"], 0)

    def test_run_benchmark_out_file(self):
        import tempfile
        engine = MockEngine()
        prompts_data = load_league_prompts(PROMPTS_PATH)
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            temp_path = f.name
        try:
            run_benchmark(engine, prompts_data, "prose", levels=[1], max_tokens=5, warmup_tokens=2, out_file=temp_path)
            with open(temp_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("LEAGUE {"))
            parsed = json.loads(lines[0][7:])
            self.assertEqual(parsed["level"], 1)
            self.assertIn("checks", parsed)
            self.assertTrue(parsed["checks"].startswith("checks: "))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_run_benchmark_carries_checks_field(self):
        engine = MockEngine()
        prompts_data = load_league_prompts(PROMPTS_PATH)
        records = run_benchmark(engine, prompts_data, "prose", levels=[1], max_tokens=5, warmup_tokens=2)
        self.assertEqual(len(records), 1)
        self.assertIn("checks", records[0])
        self.assertEqual(records[0]["checks"], "checks: golden=off, duprow=off, ref=off")


if __name__ == "__main__":
    unittest.main()
