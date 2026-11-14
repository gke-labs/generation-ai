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

"""Unit tests for graph lowering and kernel digest computation."""

import json
import os
import tempfile
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from . import lower, plan, plan_pb2, triton_ops


class LowerTest(unittest.TestCase):

    def setUp(self):
        self.old_gemv = getattr(triton_ops, "GEMV_ROWS_MAX", None)
        self.old_fp4_direct_max = getattr(triton_ops, "FP4_DIRECT_MAX", None)
        self.old_allreduce_block = getattr(triton_ops, "ALLREDUCE_BLOCK", None)
        self.old_base_cache_dir = lower._base_cache_dir
        self.old_memo_digest = lower._memo_digest
        self.old_keyed_cache_dir = lower._keyed_cache_dir
        self.old_cache_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        lower._base_cache_dir = None
        lower._memo_digest = None
        lower._keyed_cache_dir = None

    def tearDown(self):
        lower._base_cache_dir = self.old_base_cache_dir
        lower._memo_digest = self.old_memo_digest
        lower._keyed_cache_dir = self.old_keyed_cache_dir
        if self.old_cache_env is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = self.old_cache_env

        if self.old_gemv is None:
            if hasattr(triton_ops, "GEMV_ROWS_MAX"):
                delattr(triton_ops, "GEMV_ROWS_MAX")
        else:
            triton_ops.GEMV_ROWS_MAX = self.old_gemv

        if self.old_fp4_direct_max is None:
            if hasattr(triton_ops, "FP4_DIRECT_MAX"):
                delattr(triton_ops, "FP4_DIRECT_MAX")
        else:
            triton_ops.FP4_DIRECT_MAX = self.old_fp4_direct_max

        if self.old_allreduce_block is None:
            if hasattr(triton_ops, "ALLREDUCE_BLOCK"):
                delattr(triton_ops, "ALLREDUCE_BLOCK")
        else:
            triton_ops.ALLREDUCE_BLOCK = self.old_allreduce_block

    def test_kernel_digest_and_key_cache_dir(self):
        triton_ops.GEMV_ROWS_MAX = 8
        d1 = lower.kernel_digest()
        self.assertEqual(len(d1), 16)
        cache_dir_1 = lower.key_cache_dir()
        self.assertTrue(cache_dir_1.endswith(f"vxpu-kernels-{d1}"))
        self.assertEqual(os.environ.get("TORCHINDUCTOR_CACHE_DIR"), cache_dir_1)

        # Recomputing with a new knob updates the cache directory and environment variable.
        triton_ops.GEMV_ROWS_MAX = 16
        d2 = lower.kernel_digest()
        self.assertEqual(len(d2), 16)
        self.assertNotEqual(d1, d2)

        cache_dir_2 = lower.key_cache_dir()
        self.assertTrue(cache_dir_2.endswith(f"vxpu-kernels-{d2}"))
        self.assertNotEqual(cache_dir_1, cache_dir_2)
        self.assertEqual(os.environ.get("TORCHINDUCTOR_CACHE_DIR"), cache_dir_2)
        # Verify base directory was not nested.
        self.assertNotIn("vxpu-kernels-", os.path.dirname(cache_dir_2))

        # Idempotent when digest does not change.
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # kernel_digest does not change when FP4_DIRECT_MAX changes.
        triton_ops.FP4_DIRECT_MAX = 10
        d3 = lower.kernel_digest()
        self.assertEqual(d3, d2)
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # kernel_digest does not change when ALLREDUCE_BLOCK changes.
        triton_ops.ALLREDUCE_BLOCK = 1024
        d4 = lower.kernel_digest()
        self.assertEqual(d4, d2)
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # An inherited keyed directory in TORCHINDUCTOR_CACHE_DIR strips to its parent.
        lower._base_cache_dir = None
        lower._memo_digest = None
        lower._keyed_cache_dir = None
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ti/vxpu-kernels-0123456789abcdef"
        cache_dir_inherited = lower.key_cache_dir()
        self.assertEqual(os.path.dirname(cache_dir_inherited), "/tmp/ti")
        self.assertTrue(cache_dir_inherited.startswith("/tmp/ti/vxpu-kernels-"))

    def test_site_bytes(self):
        with FakeTensorMode():
            g = torch.fx.Graph()
            x = g.placeholder("x")
            x.meta["val"] = torch.empty((1, 16), dtype=torch.bfloat16)

            p_model_w = g.placeholder("p_model_w")
            p_model_w.meta["val"] = torch.empty((16, 8), dtype=torch.float8_e4m3fn)

            b_model_cache = g.placeholder("b_model_cache")
            b_model_cache.meta["val"] = torch.empty((64, 8), dtype=torch.bfloat16)

            perm = g.call_function(torch.ops.aten.permute.default, (p_model_w, [1, 0]))
            perm.meta["val"] = torch.empty((8, 16), dtype=torch.float8_e4m3fn)

            mm = g.call_function(torch.ops.aten.mm.default, (x, perm))
            mm.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            add = g.call_function(torch.ops.aten.add.Tensor, (b_model_cache, mm))
            add.meta["val"] = torch.empty((64, 8), dtype=torch.bfloat16)

            buf = g.get_attr("model_buf")

            buf_op = g.call_function(torch.ops.aten.add.Tensor, (mm, buf))
            buf_op.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model_buf = torch.empty((1, 8), dtype=torch.bfloat16)

            gm = torch.fx.GraphModule(Mod(), g)

            # mm site reads: weights: 128 (16x8 fp8), activations: 48 (1x16 bf16 input [32] + 1x8 bf16 output [16]), state: 0
            sb_mm = lower.site_bytes(gm, mm)
            self.assertEqual(sb_mm.weights, 128)
            self.assertEqual(sb_mm.activations, 48)
            self.assertEqual(sb_mm.state, 0)

            # add site reads: weights: 0, activations: 1040 (1x8 bf16 input [16] + 64x8 bf16 output [1024]), state: 1024 (64x8 bf16)
            sb_add = lower.site_bytes(gm, add)
            self.assertEqual(sb_add.weights, 0)
            self.assertEqual(sb_add.activations, 1040)
            self.assertEqual(sb_add.state, 1024)

            # a get_attr buffer counts as a parameter (weights: 16 bytes for 1x8 bf16)
            sb_buf = lower.site_bytes(gm, buf_op)
            self.assertEqual(sb_buf.weights, 16)
            self.assertEqual(sb_buf.activations, 32)
            self.assertEqual(sb_buf.state, 0)

            # an intermediate call_function result counts as activation, never weights or state
            cond = g.placeholder("cond")
            cond.meta["val"] = torch.empty((1, 8), dtype=torch.bool)
            where_node = g.call_function(torch.ops.aten.where.self, (cond, mm, buf_op))
            where_node.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            consumer = g.call_function(torch.ops.aten.add.Tensor, (where_node, mm))
            consumer.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            sb_consumer = lower.site_bytes(gm, consumer)
            self.assertEqual(sb_consumer.weights, 0)
            self.assertEqual(sb_consumer.activations, 48)
            self.assertEqual(sb_consumer.state, 0)

    def test_describe_sites_no_choice_and_facts(self):
        with FakeTensorMode():
            g = torch.fx.Graph()
            x = g.placeholder("x")
            x.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            w = g.placeholder("p_w")
            w.meta["val"] = torch.empty((16, 8), dtype=torch.float8_e4m3fn)

            s = g.placeholder("p_s")
            s.meta["val"] = torch.empty((1,), dtype=torch.float32)

            node = g.call_function(torch.ops.vxpu.fp8_linear.default, (x, w, s))
            node.meta["val"] = torch.empty((1, 16), dtype=torch.bfloat16)
            node.meta["nn_module_stack"] = {
                "L__self___model_layers_0": ("model.layers.0", "Block"),
                "linear": ("model.layers.0.attn.wo_a", "Linear"),
            }

            class Mod(torch.nn.Module):
                pass

            gm = torch.fx.GraphModule(Mod(), g)

            rec = lower.describe_sites(gm)
            self.assertEqual(len(rec.sites), 1)
            site = rec.sites[0]
            self.assertEqual(site.op, "vxpu::fp8_linear")
            self.assertEqual(site.site, "model.layers.0.attn.wo_a")
            self.assertEqual(site.node, node.name)
            self.assertEqual(site.shapes, "1x8->16")
            self.assertEqual(site.layer, 0)
            self.assertEqual(site.bytes.weights, 132)
            self.assertEqual(site.bytes.activations, 48)

            # Facts only: no choices recorded
            self.assertEqual(site.chosen, "")
            self.assertEqual(site.source, "")
            self.assertFalse(site.HasField("why"))
            self.assertEqual(list(site.kernels), [])
            self.assertFalse(site.HasField("fused"))
            self.assertFalse(site.HasField("measured_us"))
            self.assertEqual(site.rows, 1)

            # Bodies are reported in the catalog
            cat = plan.build_catalog()
            body_names = [b.name for b in cat.bodies if b.op == "vxpu::fp8_linear"]
            self.assertIn("reference", body_names)
            ref_body = next(b for b in cat.bodies if b.op == "vxpu::fp8_linear" and b.name == "reference")
            self.assertEqual(list(ref_body.devices), ["cpu", "cuda"])
            self.assertEqual(ref_body.numerics, "bit-identical")
            self.assertFalse(ref_body.lowerable)
            self.assertEqual(ref_body.kind, plan_pb2.SITE_KIND_DENSE_GEMV)

    def test_throwaway_variant_appears_in_candidates(self):
        import importlib
        from . import plan

        kernels_dir = os.path.join(os.path.dirname(__file__), "kernels", "fp8_linear")
        var_file = os.path.join(kernels_dir, "throwaway_test_cand.py")
        code = '''from .. import variant
import torch

@variant(
    devices=("cpu",),
    numerics="rounding",
    when="throwaway candidate for testing",
)
def throwaway_test_cand(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x
'''
        with FakeTensorMode():
            g = torch.fx.Graph()
            x = g.placeholder("x")
            x.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)
            w = g.placeholder("p_w")
            w.meta["val"] = torch.empty((16, 8), dtype=torch.float8_e4m3fn)
            s = g.placeholder("p_s")
            s.meta["val"] = torch.empty((1,), dtype=torch.float32)
            node = g.call_function(torch.ops.vxpu.fp8_linear.default, (x, w, s))
            node.meta["val"] = torch.empty((1, 16), dtype=torch.bfloat16)

            class Mod(torch.nn.Module):
                pass

            gm = torch.fx.GraphModule(Mod(), g)

            try:
                with open(var_file, "w", encoding="utf-8") as f:
                    f.write(code)
                importlib.invalidate_caches()

                cat = plan.build_catalog()
                cand = next((b for b in cat.bodies if b.name == "throwaway_test_cand"), None)
                self.assertIsNotNone(cand, "new kernel variant under kernels/fp8_linear/ must appear in catalog")
                assert cand is not None
                self.assertEqual(cand.name, "throwaway_test_cand")
                self.assertEqual(list(cand.devices), ["cpu"])
                self.assertEqual(cand.numerics, "rounding")
                self.assertFalse(cand.lowerable)
            finally:
                if os.path.exists(var_file):
                    os.remove(var_file)
                plan.REGISTRY["vxpu::fp8_linear"] = [
                    impl for impl in plan.REGISTRY["vxpu::fp8_linear"] if impl.name != "throwaway_test_cand"
                ]
                import sys
                sys.modules.pop("vxpu.kernels.fp8_linear.throwaway_test_cand", None)

    def test_tiny_deepseek_call_sites_description(self):
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 fixture unavailable: {e}") from e

        import tempfile
        from google.protobuf import json_format
        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir = tiny_deepseek_v41.write(os.path.join(td, "model"))
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

            from .export import export_artifact
            artifact = os.path.join(td, "artifact")
            export_artifact(model_dir, artifact, max_cache_len=64)
            ep = torch.export.load(os.path.join(artifact, "decode.pt2"))
            desc_file = os.path.join(td, "kernel_record.json")

            rec = lower.describe_sites(ep.module(), out=desc_file)
            self.assertEqual(len(rec.sites), 67)
            self.assertTrue(os.path.exists(desc_file))

            # Verify no choices are recorded across all sites
            for s in rec.sites:
                self.assertEqual(s.chosen, "")
                self.assertEqual(s.source, "")
                self.assertFalse(s.HasField("why"))
                self.assertEqual(len(s.kernels), 0)
                self.assertFalse(s.HasField("fused"))
                self.assertFalse(s.HasField("measured_us"))

            # Round-trip read
            from . import plan
            reloaded = plan.read_kernel_record(desc_file)
            self.assertEqual(len(reloaded.sites), 67)
            self.assertEqual(reloaded.sites[0].op, rec.sites[0].op)

            # Pin against checked-in testdata fixture when available
            fixture_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "kernel_record.json")
            )
            if os.path.exists(fixture_path):
                expected = plan.read_kernel_record(fixture_path)
                self.assertEqual(
                    json_format.MessageToDict(rec),
                    json_format.MessageToDict(expected),
                    "describe_sites output differs from checked-in testdata/kernel_record.json; "
                    "regenerate with: python -m vxpu.lower --artifact <tiny_artifact> --out vxpu/pkg/kernelplan/testdata/kernel_record.json",
                )

    def test_checked_in_kernel_record_fixture(self):
        from . import plan
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "kernel_record.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"kernel record fixture not found at {fixture_path}")

        rec = plan.read_kernel_record(fixture_path)
        self.assertEqual(len(rec.sites), 67)
        self.assertEqual(rec.sites[0].op, "vxpu::hc_sinkhorn")
        self.assertTrue(rec.sites[0].sinkhorn_supported)
        self.assertEqual(rec.sites[1].op, "vxpu::fp8_linear")
        self.assertEqual(rec.sites[1].rows, 1)
        self.assertEqual(rec.sites[10].op, "vxpu::moe_experts")
        self.assertEqual(rec.sites[14].op, "vxpu::engram_rows")

    def test_checked_in_kernel_catalog_fixture(self):
        from . import plan
        from google.protobuf import json_format

        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "kernel_catalog.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"kernel catalog fixture not found at {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            expected = plan_pb2.KernelCatalog()
            json_format.Parse(f.read(), expected)

        current = plan.build_catalog()
        self.assertEqual(
            json_format.MessageToDict(current),
            json_format.MessageToDict(expected),
            "build_catalog() output differs from checked-in testdata/kernel_catalog.json; "
            "regenerate with: python -m vxpu.lower --catalog vxpu/pkg/kernelplan/testdata/kernel_catalog.json",
        )

    def test_checked_in_golden_call_sites_plan_fixture(self):
        from . import physplan
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "golden_call_sites_plan.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"golden call sites plan fixture not found at {fixture_path}")

        golden_plan = physplan.read_plan_file(fixture_path)
        self.assertEqual(len(golden_plan.sites), 67)
        for s in golden_plan.sites:
            self.assertTrue(s.id)
            self.assertNotEqual(s.kind, plan_pb2.SITE_KIND_UNSPECIFIED)
            self.assertEqual(s.strategy, "")
            self.assertTrue(s.body)

    def test_tiny_deepseek_plan_sites_agree_with_lowering(self):
        """Verifies that the bodies chosen by today's Python lowering agree with the planner's plan."""
        fixture_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata")
        )
        plan_path = os.path.join(fixture_dir, "golden_call_sites_plan.json")
        table_path = os.path.join(fixture_dir, "kernel_table.json")
        if not os.path.exists(plan_path) or not os.path.exists(table_path):
            self.skipTest("fixtures not found")

        try:
            from vxpu.fixtures import tiny_deepseek_v41
            from vxpu.export import export_artifact
        except ImportError as e:
            self.skipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}")

        from . import physplan, plan

        golden_plan = physplan.read_plan_file(plan_path)
        plan_bodies = {s.id: s.body for s in golden_plan.sites}

        with open(table_path) as f:
            table_dict = json.load(f)

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir = tiny_deepseek_v41.write(os.path.join(td, "model"))
            except (ImportError, ModuleNotFoundError) as e:
                self.skipTest(f"tiny_deepseek_v41 fork unavailable: {e}")
            artifact = os.path.join(td, "artifact")
            export_artifact(model_dir, artifact, max_cache_len=64)

            # 1. Lowering with kernel_table.json (silent for rotary, sinkhorn, grouped_linear, head_linear)
            ep = torch.export.load(os.path.join(artifact, "decode.pt2"))
            p = plan.Plan("cpu", given={"VXPU_PLAN_TABLE": json.dumps(table_dict)})
            lower.lower(ep.module(), p)

            lowered_sites = [s for s in p.sites if s.chosen]
            self.assertEqual(len(lowered_sites), 62)
            for s in lowered_sites:
                self.assertIn(s.node, plan_bodies)
                self.assertEqual(
                    s.chosen,
                    plan_bodies[s.node],
                    f"site {s.node} ({s.op}): lowering chose {s.chosen!r}, plan named {plan_bodies[s.node]!r}",
                )

            # 2. Lowering with no table on CPU (planner and lowering both select default preference order)
            ep_notable = torch.export.load(os.path.join(artifact, "decode.pt2"))
            p_notable = plan.Plan("cpu")
            lower.lower(ep_notable.module(), p_notable)
            notable_sites = [s for s in p_notable.sites if s.chosen]
            self.assertEqual(len(notable_sites), 62)
            for s in notable_sites:
                expected_choice = "f32-copy" if s.op == "vxpu::head_linear" else "reference"
                self.assertEqual(
                    s.chosen,
                    expected_choice,
                    f"no-table site {s.node} ({s.op}): lowering chose {s.chosen!r}, want {expected_choice!r}",
                )

    def test_lowering_does_not_latch_kernels_error_when_disabled(self):
        from . import kernels, plan

        saved_error = kernels._error
        try:
            kernels._error = None
            with FakeTensorMode():
                g = torch.fx.Graph()
                x = g.placeholder("x")
                x.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)
                w = g.placeholder("p_w")
                w.meta["val"] = torch.empty((16, 8), dtype=torch.float8_e4m3fn)
                s = g.placeholder("p_s")
                s.meta["val"] = torch.empty((1,), dtype=torch.float32)
                node = g.call_function(torch.ops.vxpu.fp8_linear.default, (x, w, s))
                node.meta["val"] = torch.empty((1, 16), dtype=torch.bfloat16)

                class Mod(torch.nn.Module):
                    pass

                gm = torch.fx.GraphModule(Mod(), g)

                with unittest.mock.patch.dict(os.environ, {"VXPU_KERNELS": "0"}):
                    p = plan.Plan("cpu")
                    lower.lower(gm, p)
                self.assertIsNone(kernels._error, "lowering with VXPU_KERNELS=0 must not call kernels.load() or latch _error")
        finally:
            kernels._error = saved_error


if __name__ == "__main__":
    unittest.main()
