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

import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import zipfile

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

    def test_kernel_digest_tiny_model_preserved(self):
        """Verifies that kernel_digest for the tiny model's knobs is preserved."""
        saved = {
            "GEMV_ROWS_MAX": getattr(triton_ops, "GEMV_ROWS_MAX", None),
            "FP4_NATIVE": getattr(triton_ops, "FP4_NATIVE", None),
            "ALLREDUCE_BLOCK": getattr(triton_ops, "ALLREDUCE_BLOCK", None),
            "FP4_DIRECT_MAX": getattr(triton_ops, "FP4_DIRECT_MAX", None),
            "FP4_GU_BN": getattr(triton_ops, "FP4_GU_BN", None),
        }
        try:
            triton_ops.GEMV_ROWS_MAX = 16
            triton_ops.FP4_NATIVE = False
            digest = lower.kernel_digest()
            self.assertEqual(digest, "ac5ba01f41af2ccc")

            # Outside-segment knobs do not alter the kernel digest.
            triton_ops.ALLREDUCE_BLOCK = 1024
            triton_ops.FP4_DIRECT_MAX = 10
            triton_ops.FP4_GU_BN = 64
            self.assertEqual(lower.kernel_digest(), digest)
        finally:
            for k, v in saved.items():
                if v is None:
                    if hasattr(triton_ops, k):
                        delattr(triton_ops, k)
                else:
                    setattr(triton_ops, k, v)

    def test_is_compiled_in_knob_unknown_refused(self):
        with self.assertRaises(KeyError):
            lower.is_compiled_in_knob("UNKNOWN_KNOB")

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

    def test_tiny_deepseek_go_plan_emit_plans_all_graph_call_sites(self):
        """Go vxpu plan emit plans every call site from decode.pt2 without Python description."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 fixture unavailable: {e}") from e

        import subprocess
        import tempfile
        from google.protobuf import json_format

        with tempfile.TemporaryDirectory() as td:
            try:
                artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

            fixture_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata")
            )
            cat_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
            )
            table_path = os.path.join(fixture_dir, "kernel_table.json")
            plan_file = os.path.join(td, "plan.json")

            cmd = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
            cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

            env = dict(os.environ, GOTOOLCHAIN="auto")
            res = subprocess.run(
                [*cmd, "plan", "emit", "--artifact", artifact, "--catalog", cat_path, "--table", table_path, "--out", plan_file, "--json", "--tree-commit", "commit123"],
                cwd=cwd if cmd[0] == "go" else None,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(res.returncode, 0, f"plan emit failed: {res.stderr}")

            plan = plan_pb2.PlanFile()
            with open(plan_file, "r", encoding="utf-8") as f:
                json_format.Parse(f.read(), plan)

            # Load golden call sites plan to compare bodies
            golden_path = os.path.join(fixture_dir, "golden_call_sites_plan.json")
            golden_plan = plan_pb2.PlanFile()
            with open(golden_path, "r", encoding="utf-8") as f:
                json_format.Parse(f.read(), golden_plan)

            # PlanFile contains 141 total sites (74 macroscopic family sites from binding.json + 67 call sites)
            self.assertEqual(len(plan.sites), 141)

            call_site_entries = [s for s in plan.sites if s.body]
            self.assertEqual(len(call_site_entries), 67)
            self.assertEqual(len(golden_plan.sites), 67)

            for i in range(len(call_site_entries)):
                self.assertEqual(call_site_entries[i].id, golden_plan.sites[i].id)
                self.assertEqual(call_site_entries[i].body, golden_plan.sites[i].body)
                self.assertEqual(call_site_entries[i].kind, golden_plan.sites[i].kind)

    def test_tiny_deepseek_go_plan_segments_matches_python_split(self):
        """Go vxpu plan segments finds the same segments and call sites as split_at_host_ops."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 fixture unavailable: {e}") from e

        try:
            artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
        except (ImportError, ModuleNotFoundError) as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

        from . import ops, segments
        ep = torch.export.load(os.path.join(artifact, "decode.pt2"))
        gm = ep.module()
        split, host_names = segments.split_at_host_ops(gm)

        # Extract Python submodules and call sites with their original graph node names
        py_segments: list[dict[str, object]] = []
        for name, sub in segments.segments_of(split):
            cs = lower.call_sites(sub)
            sites: list[tuple[str, str, str]] = []
            for node, rec, _, _ in cs:
                from_node = node.meta.get("from_node")
                node_name = from_node[0].name if from_node else node.name
                sites.append((rec.op, node_name, rec.site))
            py_segments.append({
                "name": name,
                "host": name in host_names,
                "sites": sites,
            })

        # Run Go vxpu plan segments
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd_prefix = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "auto"

        # Verify that Go's HostOps and Python's HOST_OPS agree
        res_host = subprocess.run(cmd_prefix + ["plan", "segments", "--host-ops", "--json"], cwd=cwd, capture_output=True, text=True, check=True, env=env)
        go_host_ops = set(json.loads(res_host.stdout))
        py_host_ops = {"vxpu::" + op.split(".")[1] for op in segments.HOST_OPS}
        self.assertEqual(go_host_ops, py_host_ops, "Go HostOps and Python segments.HOST_OPS must agree")

        # Run Go vxpu plan segments --artifact <artifact> --json
        cmd = cmd_prefix + ["plan", "segments", "--artifact", artifact, "--json"]
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=env)
        go_segments = json.loads(res.stdout)

        self.assertEqual(len(go_segments), len(py_segments))
        for go_seg, py_seg in zip(go_segments, py_segments):
            self.assertEqual(go_seg["name"], py_seg["name"])
            self.assertEqual(go_seg.get("host", False), py_seg["host"])
            go_sites = [(s["op"], s["node"], s.get("site", "")) for s in go_seg.get("call_sites", [])]
            self.assertEqual(go_sites, py_seg["sites"])

    def test_tiny_deepseek_go_plan_tree_matches_segments(self):
        """Go vxpu plan tree produces an offline HTML tree and JSON representation for tiny DeepSeek."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 fixture unavailable: {e}") from e

        try:
            artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
        except (ImportError, ModuleNotFoundError) as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd_prefix = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "auto"
        env["TINY_DEEPSEEK_ARTIFACT"] = artifact

        # Run Go test for tiny DeepSeek tree
        res_test = subprocess.run(
            ["go", "test", "-v", "./cmd/vxpu", "-run", "TestCmdPlanTree_TinyDeepSeek"],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(res_test.returncode, 0, f"go test TestCmdPlanTree_TinyDeepSeek failed:\n{res_test.stdout}\n{res_test.stderr}")

        # Run Go vxpu plan tree --artifact <artifact> --json
        cmd = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--json"]
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=env)
        tree = json.loads(res.stdout)

        self.assertIn("segments", tree)
        total_call_sites = sum(len(s.get("call_sites", [])) for s in tree["segments"])

        # Run Go vxpu plan tree --artifact <artifact> --out <outPath>
        with tempfile.TemporaryDirectory() as td:
            out_html = os.path.join(td, "tree.html")
            cmd_html = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--out", out_html]
            subprocess.run(cmd_html, cwd=cwd, capture_output=True, text=True, check=True, env=env)

            with open(out_html) as f:
                html_content = f.read()

            self.assertNotIn("http://", html_content)
            self.assertNotIn("https://", html_content)
            self.assertIn(f"<title>vXPU // {os.path.basename(artifact)}</title>", html_content)
            self.assertIn(f"{len(tree['segments'])}</span> Segments", html_content)
            self.assertIn(f"{total_call_sites}</span> CallSites", html_content)

            # Run Go vxpu plan emit to get plan for tiny DeepSeek
            fixture_dir = os.path.join(cwd, "pkg", "kernelplan", "testdata")
            plan_file = os.path.join(td, "plan.json")
            cmd_emit = cmd_prefix + [
                "plan", "emit",
                "--artifact", artifact,
                "--tree-commit", "0123456789abcdef0123456789abcdef01234567",
                "--catalog", os.path.normpath(os.path.join(fixture_dir, "..", "catalog", "kernel_catalog.json")),
                "--table", os.path.join(fixture_dir, "kernel_table.json"),
                "--out", plan_file,
                "--json",
            ]
            subprocess.run(cmd_emit, cwd=cwd, capture_output=True, text=True, check=True, env=env)

            # Test tree with plan alone (EXPLAIN)
            plan_html = os.path.join(td, "plan_tree.html")
            cmd_plan_html = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--plan", plan_file, "--out", plan_html]
            subprocess.run(cmd_plan_html, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            with open(plan_html) as f:
                plan_html_content = f.read()

            self.assertNotIn("http://", plan_html_content)
            self.assertIn(" (EXPLAIN)</title>", plan_html_content)
            self.assertIn("EXPLAIN", plan_html_content)
            self.assertIn("Configuration Knobs [plan]", plan_html_content)
            self.assertIn("body</span>", plan_html_content)
            self.assertIn("Unmatched Plan Sites", plan_html_content)
            self.assertIn("<td>default</td>", plan_html_content)

            # Test tree with run record (EXPLAIN ANALYZE)
            with open(plan_file) as f:
                rec_data = json.load(f)
            rec_data["stats"] = {
                "median_step_ms": 15.5,
                "tokens_a_stream_a_step": 1.0,
                "accepted_drafts_a_step": 0.75,
                "checks": {"bounds": False, "exactness": True},
                "per_card_peak_memory": ["524288000"],
                "knobs": {"GEMV_ROWS_MAX": 32, "FP4_NATIVE": True},
            }
            for s in rec_data.get("sites", []):
                s["stats"] = {"measured_us": 42.5}

            rec_file = os.path.join(td, "record.json")
            with open(rec_file, "w") as f:
                json.dump(rec_data, f)

            # Test tree with plan and run record (EXPLAIN ANALYZE)
            record_html = os.path.join(td, "record_tree.html")
            cmd_rec_html = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--plan", plan_file, "--record", rec_file, "--out", record_html]
            subprocess.run(cmd_rec_html, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            with open(record_html) as f:
                record_html_content = f.read()

            self.assertNotIn("http://", record_html_content)
            self.assertIn(" (EXPLAIN ANALYZE)</title>", record_html_content)
            self.assertIn("EXPLAIN ANALYZE", record_html_content)
            self.assertIn("42.5 µs [measured]", record_html_content)
            self.assertIn("15.50 ms [measured]", record_html_content)
            self.assertIn("Slowest Call Sites", record_html_content)

            # Test tree with run record alone (measurements, no plan)
            record_only_html = os.path.join(td, "record_only_tree.html")
            cmd_rec_only_html = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--record", rec_file, "--out", record_only_html]
            subprocess.run(cmd_rec_only_html, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            with open(record_only_html) as f:
                record_only_content = f.read()

            self.assertIn(" (MEASUREMENTS, NO PLAN)</title>", record_only_content)
            self.assertIn("MODEL MEASUREMENTS (NO PLAN)", record_only_content)

            # Test mismatched checksum refusal
            with open(plan_file) as f:
                bad_plan_data = json.load(f)
            bad_plan_data["artifact_md5"] = "deadbeef00000000deadbeef00000000"
            bad_plan_file = os.path.join(td, "bad_plan.json")
            with open(bad_plan_file, "w") as f:
                json.dump(bad_plan_data, f)

            bad_cmd = cmd_prefix + ["plan", "tree", "--artifact", artifact, "--plan", bad_plan_file]
            res_bad = subprocess.run(bad_cmd, cwd=cwd, capture_output=True, text=True, env=env)
            self.assertNotEqual(res_bad.returncode, 0)
            self.assertIn("checksum", res_bad.stderr + res_bad.stdout)

            # Export to CI artifacts directory if configured
            artifacts_dir = os.environ.get("ARTIFACTS")
            if artifacts_dir:
                os.makedirs(artifacts_dir, exist_ok=True)
                shutil.copyfile(out_html, os.path.join(artifacts_dir, "tiny-deepseek-tree.html"))
                shutil.copyfile(out_html, os.path.join(artifacts_dir, "tree.html"))
                shutil.copyfile(plan_html, os.path.join(artifacts_dir, "tiny-deepseek-plan-tree.html"))
                shutil.copyfile(record_html, os.path.join(artifacts_dir, "tiny-deepseek-record-tree.html"))

    def test_tiny_deepseek_go_plan_graph(self):
        """Go vxpu plan graph produces an offline HTML graph and JSON representation for tiny DeepSeek."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd_prefix = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "auto"

        # Run Go test for tiny DeepSeek graph
        res_test = subprocess.run(
            ["go", "test", "-v", "./cmd/vxpu", "-run", "TestCmdPlanGraph_TinyDeepSeek"],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(res_test.returncode, 0, f"go test TestCmdPlanGraph_TinyDeepSeek failed:\n{res_test.stdout}\n{res_test.stderr}")

        fixture_file = os.path.join(cwd, "pkg", "stepplan", "testdata", "stepgraph_chain.json")
        with tempfile.TemporaryDirectory() as td:
            plan_file = os.path.join(td, "stepplan.json")
            cmd_step = cmd_prefix + ["plan", "step", "--graph", fixture_file, "--out", plan_file, "--json"]
            subprocess.run(cmd_step, cwd=cwd, capture_output=True, text=True, check=True, env=env)

            # 1. Run vxpu plan graph --json
            cmd_json = cmd_prefix + ["plan", "graph", "--plan", plan_file, "--graph", fixture_file, "--json"]
            res = subprocess.run(cmd_json, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            graph = json.loads(res.stdout)
            self.assertEqual(len(graph["nodes"]), 11)
            self.assertIn("streams", graph)
            self.assertIn("edges", graph)

            # 2. Run vxpu plan graph --out graph.html
            out_html = os.path.join(td, "graph.html")
            cmd_html = cmd_prefix + ["plan", "graph", "--plan", plan_file, "--graph", fixture_file, "--out", out_html]
            subprocess.run(cmd_html, cwd=cwd, capture_output=True, text=True, check=True, env=env)

            with open(out_html) as f:
                html_content = f.read()

            self.assertNotIn("http://", html_content)
            self.assertNotIn("https://", html_content)
            self.assertIn("vXPU // <span>STEP PLAN GRAPH</span>", html_content)
            self.assertIn("11</span> Nodes", html_content)
            self.assertIn("Main Stream", html_content)
            self.assertIn("not measured", html_content)
            self.assertNotIn("0.000 ms", html_content)

            # Export to CI artifacts directory if configured
            artifacts_dir = os.environ.get("ARTIFACTS")
            if artifacts_dir:
                os.makedirs(artifacts_dir, exist_ok=True)
                shutil.copyfile(out_html, os.path.join(artifacts_dir, "tiny-deepseek-step-graph.html"))
                shutil.copyfile(out_html, os.path.join(artifacts_dir, "stepgraph.html"))

    def test_tiny_deepseek_packages_per_segment_renaming(self):
        """vxpu plan packages names each package from its own segment on the tiny DeepSeek model."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 fixture unavailable: {e}") from e

        try:
            artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
        except (ImportError, ModuleNotFoundError) as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

        fixture_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata")
        )
        plan_path = os.path.join(fixture_dir, "golden_call_sites_plan.json")

        class StubHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(404)
                self.end_headers()

            def do_HEAD(self) -> None:
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), StubHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]

        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd_prefix = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "auto"

        cmd = cmd_prefix + [
            "plan",
            "packages",
            "--plan",
            plan_path,
            "--artifact",
            artifact,
            "--machine",
            "g2-standard-48",
            "--model",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "--modelstore",
            f"http://127.0.0.1:{port}",
            "--json",
        ]

        try:
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            report = json.loads(res.stdout)
            required = report["required"]
            self.assertEqual(len(required), 6)
            names = [p["name"] for p in required]
            self.assertEqual(names, ["submod_0", "submod_2", "submod_4", "submod_6", "submod_8", "submod_10"])
            keys_orig = {p["name"]: p["key"] for p in required}

            # Changing the body at fp8_linear_4 (in submod_2) renames submod_2 and no other segment
            with open(plan_path) as f:
                plan_data = json.load(f)
            changed = False
            for s in plan_data["sites"]:
                if s["id"] == "fp8_linear_4":
                    s["body"] = "gpu-fp8-gemv"
                    changed = True
                    break
            self.assertTrue(changed, "fp8_linear_4 not found in plan")

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(plan_data, tf)
                mod_plan_path = tf.name

            try:
                cmd_mod = list(cmd)
                idx = cmd_mod.index("--plan")
                cmd_mod[idx + 1] = mod_plan_path
                res_mod = subprocess.run(cmd_mod, cwd=cwd, capture_output=True, text=True, check=True, env=env)
                report_mod = json.loads(res_mod.stdout)
                keys_mod = {p["name"]: p["key"] for p in report_mod["required"]}

                # Half 1: submod_2 changed its package key
                self.assertNotEqual(keys_mod["submod_2"], keys_orig["submod_2"])

                # Half 2: all other segments retain their unchanged package keys
                for name in ["submod_0", "submod_4", "submod_6", "submod_8", "submod_10"]:
                    self.assertEqual(keys_mod[name], keys_orig[name], f"segment {name} changed unexpectedly")
            finally:
                os.unlink(mod_plan_path)
        finally:
            server.shutdown()
            server.server_close()

    def test_go_plan_segments_refusal_named_versions(self):
        """The Go reader refuses archives whose format or schema version it was not written for, naming the versions."""
        if not shutil.which("vxpu") and not shutil.which("go"):
            raise unittest.SkipTest("neither vxpu nor go is available on PATH")

        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd_prefix = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "auto"

        def make_archive(td: str, format_val: str, version_val: str, major: int, minor: int) -> str:
            pt2_path = os.path.join(td, "decode.pt2")
            with zipfile.ZipFile(pt2_path, "w") as z:
                z.writestr("decode/archive_format", format_val)
                z.writestr("decode/archive_version", version_val)
                z.writestr(
                    "decode/models/model.json",
                    json.dumps({
                        "schema_version": {"major": major, "minor": minor},
                        "graph_module": {"graph": {"nodes": []}},
                    }),
                )
            return td

        # Bad format
        with tempfile.TemporaryDirectory() as td:
            art = make_archive(td, "bad_format", "0", 8, 20)
            res = subprocess.run(cmd_prefix + ["plan", "segments", "--artifact", art], cwd=cwd, capture_output=True, text=True, env=env)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("bad_format", res.stderr)
            self.assertIn("pt2", res.stderr)

        # Bad archive version
        with tempfile.TemporaryDirectory() as td:
            art = make_archive(td, "pt2", "99", 8, 20)
            res = subprocess.run(cmd_prefix + ["plan", "segments", "--artifact", art], cwd=cwd, capture_output=True, text=True, env=env)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("99", res.stderr)
            self.assertIn("0", res.stderr)

        # Bad schema version
        with tempfile.TemporaryDirectory() as td:
            art = make_archive(td, "pt2", "0", 9, 1)
            res = subprocess.run(cmd_prefix + ["plan", "segments", "--artifact", art], cwd=cwd, capture_output=True, text=True, env=env)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("9.1", res.stderr)
            self.assertIn("8.20", res.stderr)

    def test_checked_in_kernel_catalog_fixture(self):
        from . import plan
        from google.protobuf import json_format

        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
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
            "build_catalog() output differs from checked-in catalog/kernel_catalog.json; "
            "regenerate with: python -m vxpu.lower --json --catalog vxpu/pkg/kernelplan/catalog/kernel_catalog.json",
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

    def test_checked_in_golden_call_sites_plan_gpu_fixture(self):
        fixture_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "golden_call_sites_plan_gpu.json")
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"golden call sites plan gpu fixture not found at {fixture_path}")

        golden_plan = plan_pb2.PlanFile()
        with open(fixture_path, "r", encoding="utf-8") as f:
            json_format.Parse(f.read(), golden_plan)
        self.assertEqual(len(golden_plan.sites), 67)
        for s in golden_plan.sites:
            self.assertTrue(s.id)
            self.assertNotEqual(s.kind, plan_pb2.SITE_KIND_UNSPECIFIED)
            self.assertEqual(s.strategy, "")
            self.assertTrue(s.body)

    def test_tiny_deepseek_plan_sites_gpu_matches_golden(self):
        """Verifies that vxpu plan emit with --machine g2-standard-48 matches golden_call_sites_plan_gpu.json."""
        fixture_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata")
        )
        plan_gpu_path = os.path.join(fixture_dir, "golden_call_sites_plan_gpu.json")
        cat_path = os.path.normpath(
            os.path.join(fixture_dir, "..", "catalog", "kernel_catalog.json")
        )
        if not os.path.exists(plan_gpu_path) or not os.path.exists(cat_path):
            self.skipTest("fixtures not found")

        try:
            from vxpu.fixtures import tiny_deepseek_v41
        except ImportError as e:
            self.skipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}")

        artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
        with tempfile.TemporaryDirectory() as td:
            plan_file = os.path.join(td, "plan.json")
            cmd = ["vxpu"] if shutil.which("vxpu") else ["go", "run", "./cmd/vxpu"]
            cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            env = dict(os.environ, GOTOOLCHAIN="auto")
            subprocess.run(
                [*cmd, "plan", "emit", "--artifact", artifact, "--catalog", cat_path, "--machine", "g2-standard-48", "--rows", "1", "--out", plan_file, "--json", "--tree-commit", "commit123"],
                cwd=cwd if cmd[0] == "go" else None,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )

            plan = plan_pb2.PlanFile()
            with open(plan_file, "r", encoding="utf-8") as f:
                json_format.Parse(f.read(), plan)

            golden_plan = plan_pb2.PlanFile()
            with open(plan_gpu_path, "r", encoding="utf-8") as f:
                json_format.Parse(f.read(), golden_plan)

            call_site_entries = [s for s in plan.sites if s.body]
            self.assertEqual(len(call_site_entries), 67)
            self.assertEqual(len(golden_plan.sites), 67)
            for i in range(len(call_site_entries)):
                self.assertEqual(call_site_entries[i].id, golden_plan.sites[i].id)
                self.assertEqual(call_site_entries[i].body, golden_plan.sites[i].body)
                self.assertEqual(call_site_entries[i].kind, golden_plan.sites[i].kind)

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
                artifact = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
            except (ImportError, ModuleNotFoundError) as e:
                self.skipTest(f"tiny_deepseek_v41 fork unavailable: {e}")

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
