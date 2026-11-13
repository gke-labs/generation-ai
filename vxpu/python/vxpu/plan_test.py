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

"""Unit tests for vxpu kernel tables and plan selection."""

import json
import os
import tempfile
import unittest

from google.protobuf import json_format
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from . import lower, ops, plan, plan_pb2, triton_ops
from .plan import ConfigError, Plan, load_table, table_entry


class PlanTableTest(unittest.TestCase):

    def setUp(self):
        self.old_kernels = os.environ.get("VXPU_KERNELS")
        os.environ["VXPU_KERNELS"] = "0"
        self.old_gemv = getattr(triton_ops, "GEMV_ROWS_MAX", None)
        self.old_native = getattr(triton_ops, "FP4_NATIVE", None)
        self.good_table = {
            "knobs": {"GEMV_ROWS_MAX": 16, "FP4_NATIVE": False},
            "sites": [
                {"op": "vxpu::fp8_linear", "site": "model.layers.1.engram.wkv", "impl": "reference"},
                {"op": "vxpu::fp8_linear", "shapes": "1x8->16", "impl": "reference"},
                {"op": "vxpu::rotary", "impl": "reference"},
            ],
        }

    def tearDown(self):
        if self.old_kernels is None:
            os.environ.pop("VXPU_KERNELS", None)
        else:
            os.environ["VXPU_KERNELS"] = self.old_kernels
        if self.old_gemv is None:
            if hasattr(triton_ops, "GEMV_ROWS_MAX"):
                delattr(triton_ops, "GEMV_ROWS_MAX")
        else:
            triton_ops.GEMV_ROWS_MAX = self.old_gemv
        if self.old_native is None:
            if hasattr(triton_ops, "FP4_NATIVE"):
                delattr(triton_ops, "FP4_NATIVE")
        else:
            triton_ops.FP4_NATIVE = self.old_native

    def test_refusals(self):
        bad_cases = [
            {**self.good_table, "sites": [{"op": "vxpu::no_such", "impl": "x"}]},
            {**self.good_table, "sites": [{"op": "vxpu::fp8_linear", "impl": "no-such-body"}]},
            {**self.good_table, "knobs": {"NO_SUCH_KNOB": 1}},
            {**self.good_table, "knobs": {"GEMV_ROWS_MAX": "16"}},
            {**self.good_table, "knobs": {"FP4_NATIVE": 1}},
            {**self.good_table, "sites": [
                {"op": "vxpu::fp8_linear", "shapes": "1x8->16", "impl": "reference"},
                {"op": "vxpu::fp8_linear", "shapes": "1x8->16", "impl": "cpu-kernel"},
            ]},
            {**self.good_table, "sites": [
                {"op": "vxpu::rotary", "impl": "reference"},
                {"op": "vxpu::rotary", "impl": "gpu-triton"},
            ]},
            {**self.good_table, "knobs_by_rows": [{"knobs": {}}]},
            {**self.good_table, "knobs_by_rows": [{"rows_max": "16", "knobs": {}}]},
            {**self.good_table, "knobs_by_rows": [{"rows_max": 16, "knobs": {"GEMV_ROWS_MAX": "16"}}]},
            {**self.good_table, "knobs_by_rows": [{"rows_max": True, "knobs": {}}]},
            {**self.good_table, "knobs_by_rows": [{"rows_max": 16, "knobs": {"NO_SUCH_KNOB": 1}}]},
            {**self.good_table, "blocks": []},
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigError):
                    load_table(json.dumps(bad))

        # Coordinator-given tables must be inline JSON objects, not file paths.
        with self.assertRaises(ConfigError):
            Plan("cpu", triton_available=False, given={"VXPU_PLAN_TABLE": "/etc/table.json"})

    def test_precedence(self):
        table = load_table(json.dumps(self.good_table))
        # A site entry beats a shapes entry.
        site_match = table_entry(table, "vxpu::fp8_linear", "model.layers.1.engram.wkv", "1x8->16")
        self.assertIsNotNone(site_match)
        self.assertEqual(site_match.get("site"), "model.layers.1.engram.wkv")

        # A shapes entry applies when site does not match.
        shapes_match = table_entry(table, "vxpu::fp8_linear", "model.layers.3.attn.wo", "1x8->16")
        self.assertIsNotNone(shapes_match)
        self.assertEqual(shapes_match.get("shapes"), "1x8->16")

        # An op with neither the site nor the shapes matches nothing.
        no_match = table_entry(table, "vxpu::fp8_linear", "model.layers.3.attn.wo", "1x9->16")
        self.assertIsNone(no_match)

        # An op-level entry matches everywhere.
        op_match = table_entry(table, "vxpu::rotary", "anywhere", "any")
        self.assertIsNotNone(op_match)
        self.assertEqual(op_match.get("impl"), "reference")

    def test_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.json")
            with open(path, "w") as f:
                json.dump(self.good_table, f)

            old_env = os.environ.get("VXPU_PLAN_TABLE")
            try:
                os.environ["VXPU_PLAN_TABLE"] = path
                p = Plan("cpu", triton_available=False)
                self.assertEqual(p.choices["plan_table"].source, "env")
                self.assertIn("3 entries", str(p["plan_table"]))
                self.assertEqual(p.select("vxpu::fp8_linear", "cpu", key="1x8->16").name, "reference")
                self.assertEqual(p.select("vxpu::rotary", "cpu", key="whatever").name, "reference")
            finally:
                if old_env is None:
                    os.environ.pop("VXPU_PLAN_TABLE", None)
                else:
                    os.environ["VXPU_PLAN_TABLE"] = old_env

        # Coordinator provides an inline table.
        given = Plan("cpu", triton_available=False, given={"VXPU_PLAN_TABLE": json.dumps(self.good_table)})
        self.assertEqual(given.choices["plan_table"].source, "coordinator")

    def test_lowering_cpu(self):
        with FakeTensorMode() as mode:
            x = mode.from_tensor(torch.randn(1, 8, dtype=torch.bfloat16))
            w = mode.from_tensor(torch.randn(16, 8).to(torch.float8_e4m3fn))
            sc = mode.from_tensor(torch.ones(1, 1, dtype=torch.float8_e8m0fnu))

            def f(x):
                return torch.ops.vxpu.fp8_linear(x, w, sc)

            gm = make_fx(f, tracing_mode="fake")(x)
            node = next(n for n in gm.graph.nodes if str(n.target).startswith("vxpu.fp8_linear"))
            node.meta["nn_module_stack"] = {
                "L__self___model": ("model", "Model"),
                "L__self___model_layers_1": ("model.layers.1", "Block"),
                "wkv": ("model.layers.1.engram.wkv", "Linear"),
            }

            self.assertEqual(lower._site(node), ("model.layers.1.engram.wkv", 1))

            # A table naming reference: reference body is not lowerable, so it is kept.
            with tempfile.TemporaryDirectory() as tmp:
                record_path = os.path.join(tmp, "record.json")

                os.environ["VXPU_PLAN_RECORD"] = record_path
                try:
                    p = Plan(
                        "cpu",
                        triton_available=False,
                        given={"VXPU_PLAN_TABLE": json.dumps(self.good_table)},
                    )
                    report = lower.lower(gm, p)
                    self.assertEqual(report, {"vxpu::fp8_linear": {"kept": {"reference": 1}}})
                    self.assertGreater(p._selections.get(("vxpu::fp8_linear", "reference", "1x8->16"), 0), 0)

                    rec = p.kernel_record()
                    self.assertEqual(len(rec.sites), 1)
                    row = rec.sites[0]
                    self.assertEqual(row.source, "table")
                    self.assertEqual(row.chosen, "reference")
                    self.assertEqual(row.site, "model.layers.1.engram.wkv")
                    self.assertEqual(row.shapes, "1x8->16")
                    self.assertEqual(row.layer, 1)
                    self.assertEqual(list(row.kernels), [])
                    self.assertEqual(row.bytes.weights, 129)
                    self.assertEqual(row.bytes.activations, 48)
                    self.assertEqual(row.bytes.state, 0)

                    # File at VXPU_PLAN_RECORD has knobs in effect and no gpu_capability on CPU.
                    self.assertTrue(os.path.exists(record_path))
                    disk_rec = plan_pb2.KernelRecord()
                    with open(record_path, "r", encoding="utf-8") as fh:
                        json_format.Parse(fh.read(), disk_rec)
                    self.assertEqual(disk_rec.knobs.GEMV_ROWS_MAX, 16)
                    self.assertFalse(disk_rec.HasField("gpu_capability"))

                    # Repeated lowering on the same plan resets plan.sites and does not accumulate duplicates.
                    lower.lower(gm, p)
                    self.assertEqual(len(p.kernel_record().sites), 1)
                finally:
                    os.environ.pop("VXPU_PLAN_RECORD", None)

            # A table naming gpu-fp8-gemv: lowering raises ConfigError, eager call falls back with a note.
            table_gpu = {
                "sites": [{"op": "vxpu::fp8_linear", "site": "model.layers.1.engram.wkv", "impl": "gpu-fp8-gemv"}],
            }
            p_gpu = Plan("cpu", triton_available=False, given={"VXPU_PLAN_TABLE": json.dumps(table_gpu)})

            gm_gpu = make_fx(f, tracing_mode="fake")(x)
            node_gpu = next(n for n in gm_gpu.graph.nodes if str(n.target).startswith("vxpu.fp8_linear"))
            node_gpu.meta["nn_module_stack"] = dict(node.meta["nn_module_stack"])

            with self.assertRaises(ConfigError):
                lower.lower(gm_gpu, p_gpu)

            eager_impl = p_gpu.select("vxpu::fp8_linear", "cpu", key="1x8->16", site="model.layers.1.engram.wkv")
            self.assertNotEqual(eager_impl.name, "gpu-fp8-gemv")
            self.assertEqual(eager_impl.name, "reference")
            self.assertTrue(any("could not run it and took the registry's choice" in note for note in p_gpu.notes))

    def test_apply_knobs_and_effective(self):
        table1 = {
            "knobs": {"GEMV_ROWS_MAX": 16, "FP4_NATIVE": False},
        }
        Plan("cpu", triton_available=False, given={"VXPU_PLAN_TABLE": json.dumps(table1)})
        self.assertEqual(triton_ops.GEMV_ROWS_MAX, 16)
        self.assertEqual(triton_ops.FP4_NATIVE, "0")
        eff1 = plan.effective_knobs()
        self.assertEqual(eff1["GEMV_ROWS_MAX"], 16)
        self.assertIs(eff1["FP4_NATIVE"], False)

        # A second table without FP4_NATIVE resets it to its import-time value.
        table2 = {
            "knobs": {"GEMV_ROWS_MAX": 8},
        }
        Plan("cpu", triton_available=False, given={"VXPU_PLAN_TABLE": json.dumps(table2)})
        self.assertEqual(triton_ops.GEMV_ROWS_MAX, 8)
        self.assertFalse(hasattr(triton_ops, "FP4_NATIVE"))
        eff2 = plan.effective_knobs()
        self.assertEqual(eff2["GEMV_ROWS_MAX"], 8)
        self.assertIsNone(eff2["FP4_NATIVE"])

        # Building a plan without a table resets all knobs.
        Plan("cpu", triton_available=False)
        self.assertFalse(hasattr(triton_ops, "GEMV_ROWS_MAX"))
        self.assertIsNone(plan.effective_knobs()["GEMV_ROWS_MAX"])

    def test_env_override(self):
        old_env_gemv = os.environ.get("VXPU_GEMV_ROWS_MAX")
        old_env_native = os.environ.get("VXPU_FP4_NATIVE")
        try:
            # Environment override takes precedence over table value.
            os.environ["VXPU_GEMV_ROWS_MAX"] = "7"
            triton_ops.GEMV_ROWS_MAX = 7
            table = {
                "knobs": {"GEMV_ROWS_MAX": 16},
            }
            plan.apply_knobs(table["knobs"])
            self.assertEqual(triton_ops.GEMV_ROWS_MAX, 7)
            eff = plan.effective_knobs()
            self.assertEqual(eff["GEMV_ROWS_MAX"], 7)

            # An empty override is treated as unset, not an error.
            os.environ["VXPU_GEMV_ROWS_MAX"] = ""
            plan.apply_knobs(table["knobs"])
            self.assertEqual(triton_ops.GEMV_ROWS_MAX, 16)
            self.assertEqual(plan.effective_knobs()["GEMV_ROWS_MAX"], 16)

            # Malformed boolean override raises ConfigError.
            os.environ["VXPU_FP4_NATIVE"] = "ture"
            with self.assertRaises(ConfigError):
                plan.effective_knobs()
            os.environ.pop("VXPU_FP4_NATIVE", None)

            # Coordinator switches in given are respected.
            os.environ.pop("VXPU_GEMV_ROWS_MAX", None)
            token = plan._given.set({"VXPU_GEMV_ROWS_MAX": "9"})
            try:
                self.assertEqual(plan.effective_knobs()["GEMV_ROWS_MAX"], 9)
            finally:
                plan._given.reset(token)
        finally:
            if old_env_gemv is None:
                os.environ.pop("VXPU_GEMV_ROWS_MAX", None)
            else:
                os.environ["VXPU_GEMV_ROWS_MAX"] = old_env_gemv
            if old_env_native is None:
                os.environ.pop("VXPU_FP4_NATIVE", None)
            else:
                os.environ["VXPU_FP4_NATIVE"] = old_env_native

    def test_knobs_for_rows(self):
        entries = [
            {"rows_max": 16, "knobs": {"GEMV_ROWS_MAX": 16}},
            {"rows_max": 5, "knobs": {"GEMV_ROWS_MAX": 5}},
        ]
        table = {"knobs_by_rows": entries}
        self.assertEqual(plan.knobs_for_rows(table, 1), {"GEMV_ROWS_MAX": 5})
        self.assertEqual(plan.knobs_for_rows(table, 5), {"GEMV_ROWS_MAX": 5})
        self.assertEqual(plan.knobs_for_rows(table, 6), {"GEMV_ROWS_MAX": 16})
        self.assertEqual(plan.knobs_for_rows(table, 16), {"GEMV_ROWS_MAX": 16})
        self.assertEqual(plan.knobs_for_rows(table, 17), {})


if __name__ == "__main__":
    unittest.main()
