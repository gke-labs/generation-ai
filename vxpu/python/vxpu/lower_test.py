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

from google.protobuf import json_format
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

    def test_package_cache_key_includes_kernel_digest(self):
        from . import aoti

        g = torch.fx.Graph()
        x = g.placeholder("x")
        g.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        args = [torch.empty((1, 8), dtype=torch.bfloat16)]

        triton_ops.GEMV_ROWS_MAX = 8
        k1 = aoti.cache_key(gm, args)

        # In-segment knob change updates aoti.cache_key (#490).
        triton_ops.GEMV_ROWS_MAX = 16
        k2 = aoti.cache_key(gm, args)
        self.assertNotEqual(k1, k2)

        # Outside-segment knob does not change aoti.cache_key.
        triton_ops.FP4_DIRECT_MAX = 10
        k3 = aoti.cache_key(gm, args)
        self.assertEqual(k3, k2)

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
            self.assertEqual(list(site.kernels), [])
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
            desc_file = os.path.join(td, "kernel_record.pb")

            rec = lower.describe_sites(ep.module(), out=desc_file)
            self.assertEqual(len(rec.sites), 67)
            self.assertTrue(os.path.exists(desc_file))

            # Verify no choices are recorded across all sites
            for s in rec.sites:
                self.assertEqual(len(s.kernels), 0)

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
                expected = plan_pb2.KernelRecord()
                with open(fixture_path, "r", encoding="utf-8") as f:
                    json_format.Parse(f.read(), expected)
                self.assertEqual(
                    json_format.MessageToDict(rec),
                    json_format.MessageToDict(expected),
                    "describe_sites output differs from checked-in testdata/kernel_record.json; "
                    "regenerate with: python -m vxpu.lower --json --artifact <tiny_artifact> --out vxpu/pkg/kernelplan/testdata/kernel_record.json",
                )

    def test_checked_in_kernel_record_fixture(self):
        from . import plan
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "kernel_record.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"kernel record fixture not found at {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            rec = plan_pb2.KernelRecord()
            json_format.Parse(f.read(), rec)
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
            "regenerate with: python -m vxpu.lower --json --catalog vxpu/pkg/kernelplan/testdata/kernel_catalog.json",
        )

    def test_checked_in_golden_call_sites_plan_fixture(self):
        from . import physplan
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "golden_call_sites_plan.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"golden call sites plan fixture not found at {fixture_path}")

        golden_plan = plan_pb2.PlanFile()
        with open(fixture_path, "r", encoding="utf-8") as f:
            json_format.Parse(f.read(), golden_plan)
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

        golden_plan = plan_pb2.PlanFile()
        with open(plan_path, "r", encoding="utf-8") as f:
            json_format.Parse(f.read(), golden_plan)
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
            p = plan.Plan("cpu", plan_file=golden_plan)
            rep = lower.lower(ep.module(), p)
            self.assertIsNotNone(p.plan_file)
            assert p.plan_file is not None
            p_bodies = {s.id: s.body for s in p.plan_file.sites if s.body}
            self.assertEqual(len(p_bodies), 67)
            for s in p.sites:
                self.assertIn(s.node, p_bodies)
                self.assertEqual(
                    p_bodies[s.node],
                    plan_bodies[s.node],
                    f"site {s.node} ({s.op}): lowering executed {p_bodies[s.node]!r}, plan named {plan_bodies[s.node]!r}",
                )

            # 2. Lowering with no table on CPU: compare against the plan the Go planner wrote
            import subprocess

            plan_path_notable = os.path.join(td, "plan_notable.pb")
            sites_path = os.path.join(td, "sites.pb")
            cat_path = os.path.join(td, "cat.pb")
            desc = lower.describe_sites(ep.module())
            plan.write_kernel_record(sites_path, desc)
            plan.write_catalog(cat_path, plan.build_catalog())
            subprocess.run(
                [
                    "vxpu",
                    "plan",
                    "emit",
                    "--sites",
                    sites_path,
                    "--catalog",
                    cat_path,
                    "--device",
                    "cpu",
                    "--tree-commit",
                    "0000000000000000000000000000000000000000",
                    "--out",
                    plan_path_notable,
                ],
                check=True,
            )
            pf_notable = physplan.read_plan_file(plan_path_notable)
            ep_notable = torch.export.load(os.path.join(artifact, "decode.pt2"))
            p_notable = plan.Plan("cpu", plan_file=pf_notable)
            rep_notable = lower.lower(ep_notable.module(), p_notable)
            notable_plan_bodies = {s.id: s.body for s in pf_notable.sites if s.body}
            self.assertEqual(len(notable_plan_bodies), 67)
            for s in p_notable.sites:
                self.assertIn(s.node, notable_plan_bodies)
                expected_body = (
                    "f32-copy"
                    if s.op == "vxpu::head_linear"
                    else ("host-lookup" if s.op == "vxpu::engram_rows" else "reference")
                )
                self.assertEqual(
                    notable_plan_bodies[s.node],
                    expected_body,
                    f"site {s.node} ({s.op}): plan named {notable_plan_bodies[s.node]!r}, want {expected_body!r}",
                )

            # 3. Verify token equality against main with kernels enabled
            from .engine import Engine, SamplingParams

            eng1 = Engine(artifact, device="cpu", compile_decode=True, plan_file=plan_path_notable)
            sid1 = eng1.new_session()
            tokens1 = [
                item["token_ids"][0]
                for item in eng1.generate(sid1, [1, 2, 3], 5, SamplingParams(do_sample=False))
                if item.get("token_ids")
            ]
            self.assertEqual(tokens1, [20509, 4345, 2558, 15854, 7309])

            # 4. Verify token equality against main with kernels disabled
            from . import kernels

            kernels.set_enabled(False)
            try:
                eng2 = Engine(artifact, device="cpu", compile_decode=True, plan_file=plan_path_notable)
                sid2 = eng2.new_session()
                tokens2 = [
                    item["token_ids"][0]
                    for item in eng2.generate(sid2, [1, 2, 3], 5, SamplingParams(do_sample=False))
                    if item.get("token_ids")
                ]
                self.assertEqual(tokens2, [20509, 4345, 2558, 15854, 7309])
            finally:
                kernels.set_enabled(True)

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
                node.name = "test_fp8_linear"

                class Mod(torch.nn.Module):
                    pass

                gm = torch.fx.GraphModule(Mod(), g)

                with unittest.mock.patch.dict(os.environ, {"VXPU_KERNELS": "0"}):
                    p = plan.Plan(
                        "cpu",
                        plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="test_fp8_linear", body="reference")]),
                    )
                    lower.lower(gm, p)
                self.assertIsNone(kernels._error, "lowering with VXPU_KERNELS=0 must not call kernels.load() or latch _error")
        finally:
            kernels._error = saved_error

    def test_plan_refusals(self):
        """Verifies that lowering refuses plans naming non-existent bodies,
        bodies incompatible with the device or site conditions, or omitting bodies."""
        from . import plan

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
            node.name = "test_fp8_linear"

            class Mod(torch.nn.Module):
                pass

            gm = torch.fx.GraphModule(Mod(), g)

            # 1. Unknown body
            p1 = plan.Plan("cpu", plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="test_fp8_linear", body="nonexistent-body")]))
            with self.assertRaises(plan.ConfigError) as ctx:
                lower.lower(gm, p1)
            self.assertIn("test_fp8_linear", str(ctx.exception))
            self.assertIn("nonexistent-body", str(ctx.exception))

            # 2. Body cannot run on device (e.g. gpu-fp8-gemv on CPU)
            p2 = plan.Plan("cpu", plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="test_fp8_linear", body="gpu-fp8-gemv")]))
            with self.assertRaises(plan.ConfigError) as ctx:
                lower.lower(gm, p2)
            self.assertIn("test_fp8_linear", str(ctx.exception))
            self.assertIn("gpu-fp8-gemv", str(ctx.exception))

            # 3. Body whose condition does not hold (cpu-kernel when VXPU_KERNELS=0)
            with unittest.mock.patch.dict(os.environ, {"VXPU_KERNELS": "0"}):
                p3 = plan.Plan("cpu", plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="test_fp8_linear", body="cpu-kernel")]))
                with self.assertRaises(plan.ConfigError) as ctx:
                    lower.lower(gm, p3)
                self.assertIn("test_fp8_linear", str(ctx.exception))
                self.assertIn("cpu-kernel", str(ctx.exception))

            # 4. Plan leaves a call site without a body
            p4 = plan.Plan("cpu", plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="test_fp8_linear", body="")]))
            with self.assertRaises(plan.ConfigError) as ctx:
                lower.lower(gm, p4)
            self.assertIn("test_fp8_linear", str(ctx.exception))
            self.assertIn("plan has no body", str(ctx.exception))

            # 5. Host op body whose condition does not hold is refused during lowering
            g_host = torch.fx.Graph()
            x_h = g_host.placeholder("x")
            x_h.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)
            w_h = g_host.placeholder("w")
            w_h.meta["val"] = torch.empty((4, 8, 8), dtype=torch.bfloat16)
            i_h = g_host.placeholder("i")
            i_h.meta["val"] = torch.zeros((1, 1), dtype=torch.int64)
            node_h = g_host.call_function(torch.ops.vxpu.moe_experts.default, (x_h, w_h, i_h, 0))
            node_h.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)
            node_h.name = "moe_experts"
            gm_host = torch.fx.GraphModule(Mod(), g_host)

            with unittest.mock.patch.dict(os.environ, {"VXPU_KERNELS": "0"}):
                p_host = plan.Plan("cpu", plan_file=plan_pb2.PlanFile(sites=[plan_pb2.PlanSite(id="moe_experts", body="cpu-fused")]))
                with self.assertRaises(plan.ConfigError) as ctx:
                    lower.lower(gm_host, p_host)
                self.assertIn("moe_experts", str(ctx.exception))
                self.assertIn("cpu-fused", str(ctx.exception))

            # 6. Lowering without a plan raises clear ConfigError
            p_noplan = plan.Plan("cpu")
            with self.assertRaises(plan.ConfigError) as ctx:
                lower.lower(gm, p_noplan)
            self.assertIn("no physical plan provided", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
