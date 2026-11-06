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

"""The plan file's strategies section: read as pkg/kernelplan validates it and rendered
as the engine's switches."""

import json
import os
import tempfile
import unittest

from vxpu import physplan


def _plan(strategies, ranges=None):
    return {
        "format": physplan.FORMAT,
        "ranges": ranges or [{"gpu": 0, "layers": "0:43"}],
        "pipeline": {"batches": 1, "schedule": "range-order"},
        "strategies": strategies,
    }


class StrategySection(unittest.TestCase):
    def setUp(self):
        physplan._loaded = None
        self.saved_env = {
            k: os.environ.get(k)
            for k in ("VXPU_PLAN", "VXPU_ALLREDUCE_PUSH_BF16", "VXPU_FP4_DIRECT")
        }

    def tearDown(self):
        physplan._loaded = None
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_loads_and_renders_sum_wire_bf16(self):
        section = {"sum_wire": "bf16"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plan.json")
            with open(path, "w") as f:
                json.dump(_plan(section), f)
            plan = physplan.load(path)
            self.assertEqual(plan["strategies"], section)
            self.assertEqual(
                physplan.strategy_env(section),
                {"VXPU_ALLREDUCE_PUSH_BF16": "1"},
            )

            os.environ["VXPU_PLAN"] = path
            os.environ.pop("VXPU_ALLREDUCE_PUSH_BF16", None)
            physplan.apply_env()
            self.assertEqual(os.environ["VXPU_ALLREDUCE_PUSH_BF16"], "1")

    def test_loads_and_renders_sum_wire_fp32(self):
        section = {"sum_wire": "fp32"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plan.json")
            with open(path, "w") as f:
                json.dump(_plan(section), f)

            os.environ["VXPU_PLAN"] = path
            os.environ.pop("VXPU_ALLREDUCE_PUSH_BF16", None)
            physplan.apply_env()
            self.assertEqual(os.environ["VXPU_ALLREDUCE_PUSH_BF16"], "0")

    def test_loads_and_renders_expert_direct_and_grouped(self):
        self.assertEqual(
            physplan.strategy_env({"expert": "direct"}),
            {"VXPU_FP4_DIRECT": "1"},
        )
        self.assertEqual(
            physplan.strategy_env({"expert": "grouped"}),
            {"VXPU_FP4_DIRECT": "0"},
        )
        self.assertEqual(
            physplan.strategy_env({"expert": "per-expert"}),
            {"VXPU_FP4_DIRECT": "0"},
        )

    def test_refuses_unknown_strategy(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_plan.json")
            with open(path, "w") as f:
                json.dump(_plan({"sum_wire": "unknown_strategy"}), f)

            with self.assertRaises(ValueError) as cm:
                physplan.load(path)
            self.assertIn("unknown strategy 'unknown_strategy'", str(cm.exception))

    def test_refuses_unknown_site_class(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_plan.json")
            with open(path, "w") as f:
                json.dump(_plan({"unknown_class": "fast"}), f)

            with self.assertRaises(ValueError) as cm:
                physplan.load(path)
            self.assertIn("unknown site class 'unknown_class'", str(cm.exception))

    def test_hand_switch_overrides_plan(self):
        section = {"sum_wire": "bf16"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plan.json")
            with open(path, "w") as f:
                json.dump(_plan(section), f)

            os.environ["VXPU_PLAN"] = path
            os.environ["VXPU_ALLREDUCE_PUSH_BF16"] = "0"
            physplan.apply_env()
            # Hand override wins
            self.assertEqual(os.environ["VXPU_ALLREDUCE_PUSH_BF16"], "0")


if __name__ == "__main__":
    unittest.main()
