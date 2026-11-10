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

from . import physplan, plan_pb2
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
                    kind="expert op",
                    shape=[384, 2304, 2048],
                    row_counts=[1, 5, 8],
                    strategy="fused",
                ),
                plan_pb2.PlanSite(
                    id="layers.1.experts",
                    kind="expert op",
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
                    kind="expert op",
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
                    kind="expert op",
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
                    kind="dense GEMV",
                    strategy="gemv_rows",
                ),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            physplan.apply_env(pf)
        self.assertIn("unknown strategy 'gemv_rows'", str(ctx.exception))

    def test_unknown_site_kind_raises(self):
        pf = plan_pb2.PlanFile(
            sites=[
                plan_pb2.PlanSite(
                    id="some.site",
                    kind="unsupported_kind",
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
                plan_pb2.PlanSite(id="layers.0.experts", kind="expert op", strategy="fused"),
                plan_pb2.PlanSite(id="layers.1.experts", kind="expert op", strategy="per-expert"),
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
                        kind="expert op",
                        strategy="per-expert",
                    ),
                ],
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(json_format.MessageToJson(pf))

            # Engine loaded with plan_file parameter runs with fused_experts False and prints it
            eng = Engine(artifact, device="cpu", plan_file=plan_path)
            self.assertFalse(eng.plan["fused_experts"])
            self.assertIn("fused_experts=False (coordinator)", eng.plan.summary())

            sid = eng.new_session()
            events = list(eng.generate(sid, [1, 2], max_new_tokens=1, params=SamplingParams(do_sample=False)))
            self.assertTrue(events)

            # Engine loaded via VXPU_PLAN environment switch
            saved = os.environ.get("VXPU_PLAN")
            os.environ["VXPU_PLAN"] = plan_path
            try:
                env_eng = Engine(artifact, device="cpu")
                self.assertFalse(env_eng.plan["fused_experts"])
            finally:
                if saved is None:
                    os.environ.pop("VXPU_PLAN", None)
                else:
                    os.environ["VXPU_PLAN"] = saved

            # Plan with unknown strategy refuses the load
            bad_plan_path = os.path.join(td, "bad_plan.json")
            bad_pf = plan_pb2.PlanFile(
                sites=[
                    plan_pb2.PlanSite(
                        id="layers.0.experts",
                        kind="expert op",
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


if __name__ == "__main__":
    unittest.main()
