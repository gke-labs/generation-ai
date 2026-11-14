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

"""Unit tests for physical plan mapping and per-site strategies (#529, #583)."""

import os
import re
import tempfile
import unittest
from google.protobuf import json_format

from . import physplan, plan, plan_pb2
from .engine import Engine, SamplingParams
from .export import export_artifact
from .plan import ConfigError


def find_knob_readers(knob: str, pkg_dir: str) -> list[str]:
    """Finds files in pkg_dir that read knob, excluding physplan.py and tests."""
    readers: list[str] = []
    pattern = re.compile(rf"\b{re.escape(knob)}\b")
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            if f in ("physplan.py", "physplan_test.py") or f.endswith("_test.py"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if pattern.search(content):
                readers.append(path)
    return readers


class PhysPlanTest(unittest.TestCase):

    def test_plan_with_two_strategies_for_one_site_class(self):
        pf = plan_pb2.PlanFile(
            artifact_md5="abc123md5",
            tree_commit="commit_xyz",
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    shape=[384, 2304, 2048],
                    row_counts=[1, 5, 8],
                    strategy="fused",
                ),
                plan_pb2.PlanSite(
                    id="layers.1.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    shape=[384, 2304, 2048],
                    row_counts=[1, 5, 8],
                    strategy="mailbox",
                ),
            ],
        )

        switches = physplan.apply_env(pf)
        self.assertEqual(switches, {"VXPU_FUSED_EXPERTS": "1", "VXPU_EXPERT_MAILBOXES": "1"})

    def test_per_expert_strategy(self):
        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    strategy="per-expert",
                ),
            ],
        )
        switches = physplan.apply_env(pf)
        self.assertEqual(switches, {"VXPU_FUSED_EXPERTS": "0"})

    def test_unknown_strategy_raises(self):
        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    strategy="unsupported_magic_strategy",
                ),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(pf)
        self.assertIn("unknown strategy", str(ctx.exception))

    def test_non_default_strategy_for_kind_without_strategies_raises(self):
        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(
                    id="model.layers.0.attn.wo_a.weight",
                    kind=plan_pb2.SITE_KIND_DENSE_GEMV,
                    strategy="gemv_rows",
                ),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(pf)
        self.assertIn("unknown strategy 'gemv_rows'", str(ctx.exception))

    def test_unknown_site_kind_raises(self):
        with tempfile.TemporaryDirectory() as td:
            plan_path = os.path.join(td, "bad_kind.json")
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write('{"sites": [{"id": "some.site", "kind": "unsupported_kind"}]}')
            with self.assertRaises(ValueError) as ctx:
                physplan.read_plan_file(plan_path)
            self.assertIn("invalid plan JSON", str(ctx.exception))

        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(
                    id="some.site",
                    kind=plan_pb2.SITE_KIND_UNSPECIFIED,
                    strategy="default",
                ),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(pf)
        self.assertIn("unknown site kind", str(ctx.exception))

    def test_conflicting_strategies_for_same_knob_raises(self):
        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(id="layers.0.experts", kind=plan_pb2.SITE_KIND_EXPERT_OP, strategy="fused"),
                plan_pb2.PlanSite(id="layers.1.experts", kind=plan_pb2.SITE_KIND_EXPERT_OP, strategy="per-expert"),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(pf)
        self.assertIn("conflicting values for knob VXPU_FUSED_EXPERTS", str(ctx.exception))

    def test_settable_knobs_have_readers_in_tree(self):
        pkg_dir = os.path.dirname(physplan.__file__)
        for knob in physplan.SETTABLE_KNOBS:
            readers = find_knob_readers(knob, pkg_dir)
            self.assertTrue(readers, f"knob {knob} has no reader in {pkg_dir}")

    def test_knob_without_reader_in_tree_is_rejected(self):
        pkg_dir = os.path.dirname(physplan.__file__)
        unread_knob = "VXPU_NONEXISTENT_KNOB_FOR_TESTING"
        readers = find_knob_readers(unread_knob, pkg_dir)
        self.assertEqual(readers, [], f"expected no readers for {unread_knob}, found {readers}")
        with self.assertRaises(ValueError) as ctx:
            physplan._set_knob({}, unread_knob, "1", "test.site")
        self.assertIn("has no reader in the tree", str(ctx.exception))

    def test_missing_plan_path_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            physplan.read_plan_file("/nonexistent/file/path/plan.json")
        self.assertIn("/nonexistent/file/path/plan.json", str(ctx.exception))

    def test_golden_plan_round_trip(self):
        golden_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "golden_plan.json")
        )
        if not os.path.exists(golden_path):
            self.skipTest(f"golden plan fixture not found at {golden_path}")
        pf = physplan.read_plan_file(golden_path)
        switches = physplan.apply_env(pf)
        self.assertEqual(switches, {})

    def test_engine_with_plan_file_tiny_model(self):
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-model")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)

            plan_path = os.path.join(td, "plan.json")
            pf = plan_pb2.PlanFile(
                artifact_md5="dummy_md5",
                tree_commit="dummy_commit",
                sites=[
                    plan_pb2.PlanSite(
                        id="layers.0.experts",
                        kind=plan_pb2.SITE_KIND_EXPERT_OP,
                        strategy="per-expert",
                    ),
                ],
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(json_format.MessageToJson(pf))

            # Engine loaded with plan_file parameter runs with fused_experts False and prints it
            record_path = os.path.join(td, "run_record.json")
            eng = Engine(
                artifact, device="cpu", plan_file=plan_path, run_record_file=record_path
            )
            self.assertFalse(eng.plan["fused_experts"])
            self.assertIn("fused_experts=False (coordinator)", eng.plan.summary())

            sid = eng.new_session()
            events = list(eng.generate(sid, [1, 2], max_new_tokens=1, params=SamplingParams(do_sample=False)))
            self.assertTrue(events)

            self.assertTrue(os.path.exists(record_path))
            rec = physplan.read_plan_file(record_path)
            self.assertTrue(rec.HasField("stats"))
            self.assertEqual(rec.stats.plan_hash, physplan.plan_hash(plan_path))
            self.assertEqual(rec.tree_commit, "dummy_commit")
            self.assertEqual(rec.artifact_md5, "dummy_md5")
            self.assertGreater(rec.stats.median_step_ms, 0)
            self.assertEqual(rec.stats.tokens_a_stream_a_step, 1.0)
            self.assertFalse(rec.stats.HasField("accepted_drafts_a_step"))
            self.assertFalse(rec.stats.HasField("knobs"))
            self.assertEqual(len(rec.stats.per_card_peak_memory), 0)
            self.assertEqual(len(rec.stats.per_class_device_times), 0)
            self.assertEqual(len(rec.stats.checks), 0)

            # A run ending on an EOS token also records 1.0 token per stream per step
            sid_eos = eng.new_session()
            eos_events = list(
                eng.generate(
                    sid_eos, [1, 2], max_new_tokens=4, extra_eos_ids=[17245], params=SamplingParams(do_sample=False)
                )
            )
            self.assertEqual(eos_events[-1]["finish_reason"], "eos")
            eos_rec = physplan.read_plan_file(record_path)
            self.assertEqual(eos_rec.stats.tokens_a_stream_a_step, 1.0)

            # Engine loaded via VXPU_PLAN and VXPU_RUN_RECORD environment switches
            env_record_path = os.path.join(td, "env_record.json")
            saved = os.environ.get("VXPU_PLAN")
            saved_rec = os.environ.get("VXPU_RUN_RECORD")
            os.environ["VXPU_PLAN"] = plan_path
            os.environ["VXPU_RUN_RECORD"] = env_record_path
            try:
                env_eng = Engine(artifact, device="cpu")
                self.assertFalse(env_eng.plan["fused_experts"])
                env_sid = env_eng.new_session()
                list(env_eng.generate(env_sid, [1, 2], max_new_tokens=1, params=SamplingParams(do_sample=False)))
                self.assertTrue(os.path.exists(env_record_path))
                env_rec = physplan.read_plan_file(env_record_path)
                self.assertEqual(env_rec.stats.plan_hash, physplan.plan_hash(plan_path))
            finally:
                if saved is None:
                    os.environ.pop("VXPU_PLAN", None)
                else:
                    os.environ["VXPU_PLAN"] = saved
                if saved_rec is None:
                    os.environ.pop("VXPU_RUN_RECORD", None)
                else:
                    os.environ["VXPU_RUN_RECORD"] = saved_rec

            # Plan with unknown strategy refuses the load
            bad_plan_path = os.path.join(td, "bad_plan.json")
            bad_pf = plan_pb2.PlanFile(
                sites=[
                    plan_pb2.PlanSite(
                        id="layers.0.experts",
                        kind=plan_pb2.SITE_KIND_EXPERT_OP,
                        strategy="unsupported_strategy",
                    ),
                ],
            )
            with open(bad_plan_path, "w", encoding="utf-8") as f:
                f.write(json_format.MessageToJson(bad_pf))

            with self.assertRaises(ValueError) as ctx:
                Engine(artifact, device="cpu", plan_file=bad_plan_path)
            self.assertIn("unknown strategy", str(ctx.exception))

            # Conflicting coordinator switch raises ConfigError
            with self.assertRaises(ConfigError) as ctx:
                Engine(artifact, device="cpu", plan_file=plan_path, plan_given={"VXPU_FUSED_EXPERTS": "1"})
            self.assertIn("conflicts with coordinator switch", str(ctx.exception))

    def test_run_record_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "record.json")
            rec = plan_pb2.PlanFile(
                artifact_md5="artifact789",
                tree_commit="commit456",
                sites=[
                    plan_pb2.PlanSite(
                        id="layers.0.experts",
                        kind=plan_pb2.SITE_KIND_EXPERT_OP,
                        shape=[2, 2048, 768],
                        row_counts=[1, 5, 8],
                        strategy="fused",
                        stats=plan_pb2.SiteExecutionStats(measured_us=42.5),
                    ),
                    plan_pb2.PlanSite(
                        id="model.layers.0.attn.wo_a.weight",
                        kind=plan_pb2.SITE_KIND_DENSE_GEMV,
                        shape=[8192, 4096],
                        row_counts=[1, 5, 8],
                        strategy="default",
                    ),
                ],
                stats=plan_pb2.PlanExecutionStats(
                    plan_hash="hash123",
                    knobs=plan_pb2.Knobs(GEMV_ROWS_MAX=32, FP4_NATIVE=True),
                    median_step_ms=15.5,
                    tokens_a_stream_a_step=1.0,
                    accepted_drafts_a_step=0.75,
                    checks={"bounds": False, "exactness": True},
                    per_card_peak_memory=[1024 * 1024 * 500, 1024 * 1024 * 600],
                    per_class_device_times={"fp8_linear": 12.3, "experts": 4.5},
                ),
            )
            physplan.write_plan_file(path, rec)
            got = physplan.read_plan_file(path)
            self.assertEqual(got.tree_commit, "commit456")
            self.assertEqual(got.artifact_md5, "artifact789")
            self.assertEqual(len(got.sites), 2)
            self.assertEqual(got.sites[0].id, "layers.0.experts")
            self.assertEqual(got.sites[0].strategy, "fused")
            self.assertTrue(got.sites[0].HasField("stats"))
            self.assertEqual(got.sites[0].stats.measured_us, 42.5)
            self.assertEqual(got.sites[1].id, "model.layers.0.attn.wo_a.weight")
            self.assertEqual(got.sites[1].strategy, "default")
            self.assertFalse(got.sites[1].HasField("stats"))
            self.assertTrue(got.HasField("stats"))
            self.assertEqual(got.stats.plan_hash, "hash123")
            self.assertEqual(got.stats.knobs.GEMV_ROWS_MAX, 32)
            self.assertTrue(got.stats.knobs.FP4_NATIVE)
            self.assertEqual(got.stats.median_step_ms, 15.5)
            self.assertEqual(got.stats.tokens_a_stream_a_step, 1.0)
            self.assertEqual(got.stats.accepted_drafts_a_step, 0.75)
            self.assertEqual(dict(got.stats.checks), {"exactness": True, "bounds": False})
            self.assertEqual(
                list(got.stats.per_card_peak_memory), [1024 * 1024 * 500, 1024 * 1024 * 600]
            )
            self.assertEqual(
                dict(got.stats.per_class_device_times), {"fp8_linear": 12.3, "experts": 4.5}
            )

    def test_fill_run_record_preserves_plan_and_distinguishes_unset(self):
        pf = plan_pb2.PlanFile(
            artifact_md5="art1",
            tree_commit="com1",
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    strategy="fused",
                ),
                plan_pb2.PlanSite(
                    id="layers.1.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    strategy="per-expert",
                ),
            ],
        )
        self.assertFalse(pf.HasField("stats"))
        self.assertFalse(pf.sites[0].HasField("stats"))
        self.assertFalse(pf.sites[1].HasField("stats"))

        rec = physplan.fill_run_record(
            pf,
            median_step_ms=8.5,
            site_measurements={"layers.0.experts": 120.0},
        )
        self.assertEqual(rec.artifact_md5, "art1")
        self.assertEqual(rec.tree_commit, "com1")
        self.assertEqual(len(rec.sites), 2)
        self.assertEqual(rec.sites[0].strategy, "fused")
        self.assertTrue(rec.sites[0].HasField("stats"))
        self.assertEqual(rec.sites[0].stats.measured_us, 120.0)
        self.assertEqual(rec.sites[1].strategy, "per-expert")
        self.assertFalse(rec.sites[1].HasField("stats"))
        self.assertTrue(rec.HasField("stats"))
        self.assertEqual(rec.stats.median_step_ms, 8.5)
        self.assertFalse(rec.stats.HasField("tokens_a_stream_a_step"))

    def test_fill_run_record_clears_previous_measurements_on_already_measured_plan(self):
        plan = plan_pb2.PlanFile(
            artifact_md5="art1",
            tree_commit="com1",
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    strategy="fused",
                ),
                plan_pb2.PlanSite(
                    id="model.layers.0.attn.wo_a.weight",
                    kind=plan_pb2.SITE_KIND_DENSE_GEMV,
                    strategy="default",
                ),
            ],
        )
        first_record = physplan.fill_run_record(
            plan=plan,
            plan_hash="hash1",
            knobs={"GEMV_ROWS_MAX": 16},
            median_step_ms=20.0,
            tokens_a_stream_a_step=1.0,
            accepted_drafts_a_step=0.5,
            checks={"golden": True, "exactness": True},
            per_card_peak_memory=[111, 222],
            per_class_device_times={"dense": 10.0, "experts": 5.0},
            site_measurements={"layers.0.experts": 50.0},
        )
        self.assertEqual(list(first_record.stats.per_card_peak_memory), [111, 222])
        self.assertTrue(first_record.sites[0].HasField("stats"))

        # Hand first_record back as the plan for a second run that measures less:
        # per_card_peak_memory=[333], no accepted_drafts,
        # checks={"exactness": False}, per_class_device_times={"dense": 8.0}, no site_measurements.
        second_record = physplan.fill_run_record(
            plan=first_record,
            plan_hash="hash2",
            median_step_ms=12.0,
            checks={"exactness": False},
            per_card_peak_memory=[333],
            per_class_device_times={"dense": 8.0},
        )

        # per_card_peak_memory must NOT append to old list:
        self.assertEqual(list(second_record.stats.per_card_peak_memory), [333])
        # checks must NOT keep old run's keys:
        self.assertEqual(dict(second_record.stats.checks), {"exactness": False})
        # per_class_device_times must NOT keep old run's keys:
        self.assertEqual(dict(second_record.stats.per_class_device_times), {"dense": 8.0})
        # unmeasured fields in second run must be unset:
        self.assertFalse(second_record.stats.HasField("accepted_drafts_a_step"))
        self.assertFalse(second_record.stats.HasField("knobs"))
        # site measurements from earlier run must be cleared:
        self.assertFalse(second_record.sites[0].HasField("stats"))
        self.assertFalse(second_record.sites[1].HasField("stats"))

    def test_checked_in_run_record_fixture(self):
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "run_record.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"run record fixture not found at {fixture_path}")
        plan = plan_pb2.PlanFile(
            artifact_md5="artifact123",
            tree_commit="commit456",
            sites=[
                plan_pb2.PlanSite(
                    id="layers.0.experts",
                    kind=plan_pb2.SITE_KIND_EXPERT_OP,
                    shape=[2, 2048, 768],
                    row_counts=[1, 5, 8],
                    strategy="fused",
                ),
                plan_pb2.PlanSite(
                    id="model.layers.0.attn.wo_a.weight",
                    kind=plan_pb2.SITE_KIND_DENSE_GEMV,
                    shape=[8192, 4096],
                    row_counts=[1, 5, 8],
                    strategy="default",
                ),
            ],
        )
        rec = physplan.fill_run_record(
            plan=plan,
            plan_hash="hash789",
            knobs={"GEMV_ROWS_MAX": 32, "FP4_NATIVE": True},
            median_step_ms=15.5,
            tokens_a_stream_a_step=1.0,
            accepted_drafts_a_step=0.75,
            checks={"bounds": False, "exactness": True},
            per_card_peak_memory=[524288000],
            per_class_device_times={"dense": 12.3},
            site_measurements={"layers.0.experts": 42.5},
        )
        with tempfile.TemporaryDirectory() as td:
            written_path = os.path.join(td, "run_record.json")
            physplan.write_plan_file(written_path, rec)
            with open(written_path, "r", encoding="utf-8") as f:
                written_text = f.read()
            with open(fixture_path, "r", encoding="utf-8") as f:
                expected_text = f.read()
            self.assertEqual(written_text, expected_text)

    def test_run_record_unknown_field_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad_record.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"unknown_field": "disallowed", "artifact_md5": "abc"}')
            with self.assertRaises(ValueError) as ctx:
                physplan.read_plan_file(path)
            self.assertIn("invalid plan JSON", str(ctx.exception))

    def test_run_record_with_knobs_and_checks(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-model")
            export_artifact(
                "trl-internal-testing/tiny-random-LlamaForCausalLM",
                artifact,
                max_cache_len=64,
            )
            record_path = os.path.join(td, "record.json")
            table_json = json.dumps({"knobs": {"GEMV_ROWS_MAX": 16}})
            saved = os.environ.get("VXPU_PLAN_TABLE")
            os.environ["VXPU_PLAN_TABLE"] = table_json
            try:
                eng = Engine(artifact, device="cpu", run_record_file=record_path)
                eng.checks["exactness"] = True
                sid = eng.new_session()
                list(
                    eng.generate(
                        sid, [1, 2], max_new_tokens=1, params=SamplingParams(do_sample=False)
                    )
                )
                rec = physplan.read_plan_file(record_path)
                self.assertTrue(rec.HasField("stats"))
                self.assertEqual(rec.stats.knobs.GEMV_ROWS_MAX, 16)
                self.assertEqual(dict(rec.stats.checks), {"exactness": True})
            finally:
                if saved is None:
                    os.environ.pop("VXPU_PLAN_TABLE", None)
                else:
                    os.environ["VXPU_PLAN_TABLE"] = saved


if __name__ == "__main__":
    unittest.main()
