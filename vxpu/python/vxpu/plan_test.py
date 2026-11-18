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

import io
import json
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout

from google.protobuf import json_format
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from . import lower, ops, plan, plan_pb2, triton_ops
from .plan import ConfigError, Plan, check_record, describe_mismatches, load_table, table_entry


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
                record_path = os.path.join(tmp, "record.pb")

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
                    disk_rec = plan.read_kernel_record(record_path)
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

    def test_write_and_read_kernel_record_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            rec_path = os.path.join(td, "kernel_record.pb")
            rec = plan_pb2.KernelRecord(device="cpu")
            plan.write_kernel_record(rec_path, rec)
            self.assertTrue(os.path.exists(rec_path))

            loaded = plan.read_kernel_record(rec_path)
            self.assertEqual(loaded.device, "cpu")

            # Verify temp file cleanup on replace failure
            with unittest.mock.patch("os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    plan.write_kernel_record(rec_path, rec)
            files = os.listdir(td)
            self.assertEqual(files, ["kernel_record.pb"])

    def test_write_catalog_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = os.path.join(td, "kernel_catalog.pb")
            cat = plan_pb2.KernelCatalog(
                bodies=[plan_pb2.KernelBody(name="test_body", op="vxpu::fp8_linear")]
            )
            plan.write_catalog(cat_path, cat)
            self.assertTrue(os.path.exists(cat_path))

            with open(cat_path, "rb") as f:
                loaded = plan_pb2.KernelCatalog()
                loaded.ParseFromString(f.read())
            self.assertEqual(len(loaded.bodies), 1)
            self.assertEqual(loaded.bodies[0].name, "test_body")

            with unittest.mock.patch("os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    plan.write_catalog(cat_path, cat)
            files = os.listdir(td)
            self.assertEqual(files, ["kernel_catalog.pb"])

    def test_kernel_record_and_catalog_refuse_json_path(self):
        """read_kernel_record, write_kernel_record and write_catalog refuse binary paths ending in .json."""
        with self.assertRaises(ValueError) as ctx:
            plan.read_kernel_record("record.json")
        self.assertIn("must not end in .json", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            plan.write_kernel_record("record.json", plan_pb2.KernelRecord())
        self.assertIn("must not end in .json", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            plan.write_catalog("catalog.json", plan_pb2.KernelCatalog())
        self.assertIn("must not end in .json", str(ctx.exception))

    def test_aoti_parallel_builds_switch(self):
        """aoti_parallel_builds defaults to cpu_threads and can be overridden."""
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VXPU_AOTI_PARALLEL_BUILDS", None)
            p_default = Plan(device="cpu")
            self.assertEqual(p_default["aoti_parallel_builds"], os.cpu_count() or 1)

        with unittest.mock.patch.dict(os.environ, {"VXPU_AOTI_PARALLEL_BUILDS": "4"}):
            p_par = Plan(device="cpu")
            self.assertEqual(p_par["aoti_parallel_builds"], 4)

        with unittest.mock.patch.dict(os.environ, {"VXPU_AOTI_PARALLEL_BUILDS": "1"}):
            p_one = Plan(device="cpu")
            self.assertEqual(p_one["aoti_parallel_builds"], 1)

    def test_require_model_source(self):
        """require_model_source reads VXPU_REQUIRE_MODEL_SOURCE."""
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VXPU_REQUIRE_MODEL_SOURCE", None)
            self.assertFalse(plan.require_model_source())

        with unittest.mock.patch.dict(os.environ, {"VXPU_REQUIRE_MODEL_SOURCE": "1"}):
            self.assertTrue(plan.require_model_source())

        with unittest.mock.patch.dict(os.environ, {"VXPU_REQUIRE_MODEL_SOURCE": "0"}):
            self.assertFalse(plan.require_model_source())


class PlanRecordCheckTest(unittest.TestCase):
    """The engine checks what each site launched against the table at load,
    and refuses a mismatch (issue #495)."""

    def setUp(self):
        self.table_data = {
            "sites": [
                {"op": "vxpu::fp8_linear", "site": "model.layers.1.engram.wkv", "impl": "reference"},
                {"op": "vxpu::fp8_linear", "shapes": "1x8->16", "impl": "reference"},
                {"op": "vxpu::rotary", "impl": "reference"},
            ],
        }
        self.table = load_table(json.dumps(self.table_data))
        self.matching_record = {
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "layer": 1,
                    "chosen": "reference",
                    "ran": "reference",
                    "source": "table",
                },
                {
                    "op": "vxpu::rotary",
                    "site": "model.layers.0.attn.rotary",
                    "shapes": "1x8->8",
                    "layer": 0,
                    "chosen": "reference",
                    "ran": "reference",
                    "source": "table",
                },
            ],
        }

    def test_matching_record_passes_and_summarizes(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bad = check_record(self.table, self.matching_record)
        self.assertEqual(bad, [])
        output = buf.getvalue()
        self.assertIn("[vxpu] kernel table: all 2 sites matched the table", output)

    def test_mismatched_record_raises_config_error_naming_site(self):
        mismatched = {
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "layer": 1,
                    "chosen": "cpu-kernel",
                    "ran": "cpu-kernel",
                    "source": "table",
                }
            ],
        }
        with self.assertRaises(ConfigError) as ctx:
            check_record(self.table, mismatched)
        msg = str(ctx.exception)
        self.assertIn("model.layers.1.engram.wkv", msg)
        self.assertIn("ran cpu-kernel", msg)
        self.assertIn("the table names reference", msg)

    def test_mismatched_record_with_raise_false_returns_bad(self):
        mismatched = {
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "cpu-kernel",
                }
            ],
        }
        bad = check_record(self.table, mismatched, raise_on_mismatch=False)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0][1], "reference")
        described = describe_mismatches(bad, self.table)
        self.assertIn("model.layers.1.engram.wkv", described)
        self.assertIn("ran cpu-kernel", described)

    def test_no_table_warns_on_mismatch_against_chosen(self):
        no_table_record = {
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "reference",
                    "ran": "cpu-kernel",
                }
            ],
        }
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as ws, redirect_stdout(buf):
            warnings.simplefilter("always")
            bad = check_record(None, no_table_record)
            self.assertEqual(len(bad), 1)
            self.assertEqual(len(ws), 1)
            self.assertIn("model.layers.1.engram.wkv", str(ws[0].message))
            self.assertIn("ran cpu-kernel", str(ws[0].message))
            self.assertIn("the lowering chose reference", str(ws[0].message))
        self.assertIn("warning:", buf.getvalue())

    def test_no_table_matching_emits_no_warnings(self):
        no_table_record = {
            "sites": [
                {
                    "op": "vxpu::fp8_linear",
                    "site": "model.layers.1.engram.wkv",
                    "shapes": "1x8->16",
                    "chosen": "reference",
                    "ran": "reference",
                }
            ],
        }
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            bad = check_record(None, no_table_record)
            self.assertEqual(bad, [])
            self.assertEqual(len(ws), 0)

    def test_protobuf_kernel_record_checks(self):
        rec = plan_pb2.KernelRecord()
        s1 = rec.sites.add()
        s1.op = "vxpu::fp8_linear"
        s1.site = "model.layers.1.engram.wkv"
        s1.shapes = "1x8->16"
        s1.chosen = "reference"
        s1.ran = "reference"

        buf = io.StringIO()
        with redirect_stdout(buf):
            bad = check_record(self.table, rec)
        self.assertEqual(bad, [])
        self.assertIn("[vxpu] kernel table: all 1 sites matched the table", buf.getvalue())

        # Mismatch against table raises ConfigError
        s1.ran = "cpu-kernel"
        with self.assertRaises(ConfigError) as ctx:
            check_record(self.table, rec)
        msg = str(ctx.exception)
        self.assertIn("model.layers.1.engram.wkv", msg)
        self.assertIn("ran cpu-kernel", msg)
        self.assertIn("the table names reference", msg)

        # No table: mismatch against chosen warns
        s1.chosen = "reference"
        s1.ran = "cpu-kernel"
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            bad = check_record(None, rec)
            self.assertEqual(len(bad), 1)
            self.assertEqual(len(ws), 1)
            self.assertIn("model.layers.1.engram.wkv", str(ws[0].message))

    def test_cli_check_record(self):
        with tempfile.TemporaryDirectory() as td:
            table_path = os.path.join(td, "table.json")
            with open(table_path, "w") as f:
                json.dump(self.table_data, f)

            good_rec_path = os.path.join(td, "good_record.json")
            with open(good_rec_path, "w") as f:
                json.dump(self.matching_record, f)

            bad_rec_path = os.path.join(td, "bad_record.json")
            bad_data = {
                "sites": [
                    {
                        "op": "vxpu::fp8_linear",
                        "site": "model.layers.1.engram.wkv",
                        "shapes": "1x8->16",
                        "chosen": "reference",
                        "ran": "cpu-kernel",
                    }
                ]
            }
            with open(bad_rec_path, "w") as f:
                json.dump(bad_data, f)

            # Matching record with table: exits 0
            with self.assertRaises(SystemExit) as ctx:
                plan.main(["--check-record", good_rec_path, "--table", table_path])
            self.assertEqual(ctx.exception.code, 0)

            # Mismatched record with table: exits 1
            with self.assertRaises(SystemExit) as ctx:
                plan.main(["--check-record", bad_rec_path, "--table", table_path])
            self.assertEqual(ctx.exception.code, 1)

            # Record without table: exits 0 (warnings only)
            with self.assertRaises(SystemExit) as ctx:
                plan.main(["--check-record", bad_rec_path])
            self.assertEqual(ctx.exception.code, 0)

    def test_engine_check_plan(self):
        from .engine import Engine

        eng = object.__new__(Engine)
        eng.plan = Plan("cpu", given={"VXPU_PLAN_TABLE": json.dumps(self.table_data)})
        eng.plan.sites.clear()
        s1 = plan_pb2.SiteRecord(
            op="vxpu::fp8_linear",
            site="model.layers.1.engram.wkv",
            shapes="1x8->16",
            chosen="reference",
            ran="reference",
        )
        eng.plan.sites.append(s1)

        buf = io.StringIO()
        with redirect_stdout(buf):
            eng._check_plan()
        self.assertIn("all 1 sites matched the table", eng.plan.notes[-1])

        # Record mismatching site raises ConfigError
        s1.ran = "cpu-kernel"
        with self.assertRaises(ConfigError) as ctx:
            eng._check_plan()
        self.assertIn("model.layers.1.engram.wkv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
