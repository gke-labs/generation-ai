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

"""Tests for out-of-band package compilation (python -m vxpu.compile_package)."""

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any
import unittest
from unittest import mock

import grpc
import torch

from . import aoti, compile_package, deepseek_v41, engine, kernels, physplan, plan as plan_mod, plan_pb2, segments, server, vxpu_pb2
from .engine import Engine, SamplingParams
from .export import export_artifact
from .fixtures import tiny_deepseek_v41


def _emit_plan(artifact_dir: str, out_path: str, machine: str, goal: str = "pass-through") -> plan_pb2.PlanFile:
    """Emits a PlanFile from artifact_dir using the Go planner (vxpu plan emit)."""
    vxpu_bin = shutil.which("vxpu")
    if vxpu_bin is not None:
        cmd = [vxpu_bin, "plan", "emit"]
        cwd = None
    elif shutil.which("go") is not None:
        cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd = ["go", "run", "./cmd/vxpu", "plan", "emit"]
    else:
        raise unittest.SkipTest("neither vxpu binary nor go is available on PATH")

    cmd.extend([
        "--artifact",
        artifact_dir,
        "--machine",
        machine,
        "--goal",
        goal,
        "--out",
        out_path,
        "--tree-commit",
        "0123456789abcdef0123456789abcdef01234567",
    ])
    env = dict(os.environ, GOTOOLCHAIN="auto")
    subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=env)
    return physplan.read_plan_file(out_path)


class _StubContext:
    """Minimal gRPC servicer context for direct ExecutorServicer unit checks."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str = ""

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def is_active(self) -> bool:
        return True


class CompilePackageTest(unittest.TestCase):
    """Verifies that compiling packages is an out-of-band task apart from an engine."""

    _td: tempfile.TemporaryDirectory[str] | None = None
    _artifact: str = ""
    _plan_file: str = ""
    _plan: Any = None
    _init_skip: str | None = None
    _init_failure: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import importlib
            importlib.import_module("PIL")
        except (ImportError, ModuleNotFoundError) as e:
            cls._init_skip = f"PIL unavailable: {e}"
            return
        try:
            deepseek_v41._fork()
        except Exception as e:
            cls._init_skip = f"deepseek_v41 model source absent: {e}"
            return
        if not kernels.available():
            return
        cls._td = tempfile.TemporaryDirectory()
        try:
            model_dir = tiny_deepseek_v41.write(os.path.join(cls._td.name, "model"))
            cls._artifact = os.path.join(cls._td.name, "artifact")
            export_artifact(model_dir, cls._artifact, max_cache_len=64)
        except (ImportError, ModuleNotFoundError) as e:
            cls._init_skip = f"tiny_deepseek_v41 dependencies unavailable: {e}"
            return

        cls._plan = plan_mod.Plan("cpu", compile_decode=True, given={"VXPU_SEGMENT_RUNTIME": "aoti"})
        cls._plan_file = os.path.join(cls._td.name, "plan.pb")
        vxpu_bin = shutil.which("vxpu")
        if vxpu_bin is not None:
            cmd_emit = [vxpu_bin, "plan", "emit"]
            cwd = None
        elif shutil.which("go") is not None:
            cwd = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            cmd_emit = ["go", "run", "./cmd/vxpu", "plan", "emit"]
        else:
            cls._init_skip = "neither vxpu binary nor go is available on PATH"
            return

        emit_machine = "g2-standard-48" if torch.cuda.is_available() else "c3d-standard-60"
        cmd_emit.extend([
            "--artifact",
            cls._artifact,
            "--machine",
            emit_machine,
            "--out",
            cls._plan_file,
            "--tree-commit",
            "0123456789abcdef0123456789abcdef01234567",
        ])
        env = dict(os.environ, GOTOOLCHAIN="auto")
        try:
            subprocess.run(
                cmd_emit,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            cls._init_failure = f"vxpu plan emit failed (exit code {e.returncode}):\n{e.stderr}"
            return

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._td is not None:
            cls._td.cleanup()

    def setUp(self) -> None:
        try:
            import importlib
            importlib.import_module("PIL")
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"PIL unavailable: {e}")
        if self._init_failure is not None:
            self.fail(self._init_failure)
        if self._init_skip is not None:
            self.skipTest(self._init_skip)
        try:
            deepseek_v41._fork()
        except Exception as e:
            self.skipTest(f"deepseek_v41 model source absent: {e}")
        if not kernels.available():
            self.skipTest(f"CPU kernels unavailable: {kernels.why_not()}")
        if not self._artifact or not self._plan_file:
            self.skipTest("artifact or plan not available")

    def test_compile_packages_and_engine_loads_all_builds_none(self) -> None:
        """The engine follows Segment.run and package_key from the emitted plan without building."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            cache_dir = os.path.join(td, "packages")

            pf = physplan.read_plan_file(self._plan_file)
            packaged_segs = [s for s in pf.segments if s.package_key]
            self.assertEqual(len(packaged_segs), 6)
            pkgs = [plan_pb2.Package(name=s.name, key=s.package_key) for s in packaged_segs]
            planner_map = {s.name: s.package_key for s in packaged_segs}
            target_machine_kind = "cuda:nvidia-l4" if device == "cuda" else "cpu:test"

            plan = plan_mod.Plan(device, compile_decode=True, given={"VXPU_SEGMENT_RUNTIME": "aoti"})
            results, failures = compile_package.compile_packages(
                self._artifact,
                cache_dir,
                pkgs,
                plan,
                machine_kind=target_machine_kind,
                software_version=pf.tree_commit,
                device=device,
            )

            if device == "cuda":
                self.assertEqual(len(failures), 0)
                self.assertEqual(len(results), 6)
                for r in results:
                    expected_pt2 = os.path.join(cache_dir, f"{r['key']}.pt2")
                    self.assertEqual(r["path"], expected_pt2)
                    self.assertTrue(os.path.exists(expected_pt2))
                    self.assertGreater(r["bytes"], 0)

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                        with mock.patch.object(aoti, "build_segment_package", side_effect=AssertionError("must not build")), \
                             mock.patch.object(torch, "compile", side_effect=AssertionError("must not torch.compile")):
                            eng = Engine(self._artifact, device=device, plan=pf)
                self.assertIn("AOTInductor segments: 6 loaded from the cache, 0 built", buf.getvalue())
                split = getattr(eng._decode_split, "split", eng._decode_split)
                for s in packaged_segs:
                    self.assertIsInstance(getattr(split, s.name), aoti.GraphedSegment)

                ref_eng = Engine(self._artifact, device=device, compile_decode=False)
                params = SamplingParams(do_sample=False)
                tokens = [t for ev in eng.generate(eng.new_session(), [1, 2, 3], 5, params) for t in ev["token_ids"]]
                ref_tokens = [t for ev in ref_eng.generate(ref_eng.new_session(), [1, 2, 3], 5, params) for t in ev["token_ids"]]
                self.assertEqual(tokens, ref_tokens)

                # Removing two required packages fails cleanly listing both segments and keys.
                removed = packaged_segs[:2]
                for s in removed:
                    os.remove(os.path.join(cache_dir, f"{s.package_key}.pt2"))
                with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                    with self.assertRaises(FileNotFoundError) as ctx:
                        Engine(self._artifact, device=device, plan=pf)
                msg = str(ctx.exception)
                for s in removed:
                    self.assertIn(s.name, msg)
                    self.assertIn(s.package_key, msg)
                self.assertIn("build the plan's packages", msg)
            else:
                self.assertGreater(len(failures), 0)
                failed_names = {f.package.name for f in failures}
                self.assertIn("submod_6", failed_names)
                for fail in failures:
                    self.assertEqual(fail.package.key, planner_map[fail.package.name])
                    self.assertEqual(fail.machine_kind, target_machine_kind)
                    self.assertEqual(fail.software_version, pf.tree_commit)
                    self.assertTrue(fail.reason)

                self.assertEqual(len(results) + len(failures), 6)
                for r in results:
                    expected_pt2 = os.path.join(cache_dir, f"{r['key']}.pt2")
                    self.assertEqual(r["path"], expected_pt2)
                    self.assertTrue(os.path.exists(expected_pt2))
                    self.assertGreater(r["bytes"], 0)

                # Because submod_6 did not compile on CPU, loading the plan must fail cleanly
                # naming the missing segment and key rather than falling back.
                with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                    with mock.patch.object(aoti, "build_segment_package", side_effect=AssertionError("must not build")), \
                         mock.patch.object(torch, "compile", side_effect=AssertionError("must not torch.compile")):
                        with self.assertRaises(FileNotFoundError) as ctx:
                            Engine(self._artifact, device="cpu", plan=pf)
                msg = str(ctx.exception)
                self.assertIn("submod_6", msg)
                self.assertIn(planner_map["submod_6"], msg)
                self.assertIn("build the plan's packages", msg)

    def test_plan_driven_segments_tiny_llama(self) -> None:
        """Pass-through runs plain torch; speed opens pre-staged packages and matches tokens."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        machine = "g2-standard-48" if device == "cuda" else "c3d-standard-60"
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-llama")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)

            pass_pf = _emit_plan(artifact, os.path.join(td, "pass.pb"), machine=machine, goal="pass-through")
            speed_pf = _emit_plan(artifact, os.path.join(td, "speed.pb"), machine=machine, goal="speed")
            self.assertGreater(len(pass_pf.segments), 0)
            self.assertTrue(all(s.run == plan_pb2.SEGMENT_RUN_PLAIN_TORCH for s in pass_pf.segments))
            packaged_segs = [s for s in speed_pf.segments if s.package_key]
            self.assertGreater(len(packaged_segs), 0)

            cache_dir = os.path.join(td, "packages")
            os.makedirs(cache_dir, exist_ok=True)

            # 1. Pass-through plan runs plain torch without any staged packages.
            with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                with mock.patch.object(aoti, "build_segment_package", side_effect=AssertionError("must not build")), \
                     mock.patch.object(torch, "compile", side_effect=AssertionError("must not torch.compile")):
                    pass_eng = Engine(artifact, device=device, plan=pass_pf)
            pass_split = getattr(pass_eng._decode_split, "split", pass_eng._decode_split)
            for s in pass_pf.segments:
                self.assertNotIsInstance(getattr(pass_split, s.name), aoti.Segment)

            params = SamplingParams(do_sample=False)
            pass_tokens = [t for ev in pass_eng.generate(pass_eng.new_session(), [1, 2, 3], 5, params) for t in ev["token_ids"]]

            # 2. Speed plan with empty cache fails cleanly in both Engine and ExecutorServicer.LoadModel.
            with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                with self.assertRaises(FileNotFoundError) as ctx:
                    Engine(artifact, device=device, plan=speed_pf)
            for s in packaged_segs:
                self.assertIn(s.name, str(ctx.exception))
                self.assertIn(s.package_key, str(ctx.exception))
            self.assertIn("build the plan's packages", str(ctx.exception))

            with open(os.path.join(artifact, "manifest.json"), encoding="utf-8") as f:
                manifest_json = f.read()
            with open(os.path.join(artifact, "binding.json"), encoding="utf-8") as f:
                binding_json = f.read()
            with open(os.path.join(artifact, "prefill.pt2"), "rb") as f:
                prefill_graph = f.read()
            with open(os.path.join(artifact, "decode.pt2"), "rb") as f:
                decode_graph = f.read()
            load_req = vxpu_pb2.LoadModelRequest(
                manifest_json=manifest_json,
                binding_json=binding_json,
                prefill_graph=prefill_graph,
                decode_graph=decode_graph,
                plan_file=speed_pf,
            )
            servicer = server.ExecutorServicer(device=device, work_dir=os.path.join(td, "work"))
            with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                grpc_ctx = _StubContext()
                resp = servicer.LoadModel(load_req, grpc_ctx)
                self.assertFalse(resp.success)
                self.assertEqual(grpc_ctx.code, grpc.StatusCode.FAILED_PRECONDITION)
                for s in packaged_segs:
                    self.assertIn(s.name, grpc_ctx.details)
                    self.assertIn(s.package_key, grpc_ctx.details)

            # 3. On CPU, a GPU-emitted captured plan is refused with ValueError / INVALID_ARGUMENT.
            if device == "cpu":
                gpu_pf = _emit_plan(artifact, os.path.join(td, "gpu.pb"), machine="g2-standard-48", goal="speed")
                self.assertTrue(any(s.run == plan_pb2.SEGMENT_RUN_CAPTURED for s in gpu_pf.segments))
                with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                    with self.assertRaises(ValueError) as ctx:
                        Engine(artifact, device="cpu", plan=gpu_pf)
                    self.assertIn("SEGMENT_RUN_CAPTURED", str(ctx.exception))

                    gpu_req = vxpu_pb2.LoadModelRequest()
                    gpu_req.CopyFrom(load_req)
                    gpu_req.plan_file.CopyFrom(gpu_pf)
                    grpc_ctx = _StubContext()
                    resp = servicer.LoadModel(gpu_req, grpc_ctx)
                    self.assertFalse(resp.success)
                    self.assertEqual(grpc_ctx.code, grpc.StatusCode.INVALID_ARGUMENT)
                    self.assertIn("SEGMENT_RUN_CAPTURED", grpc_ctx.details)

            # 4. Compile the speed plan's packages, then load without compiling and verify tokens match.
            pkgs = [plan_pb2.Package(name=s.name, key=s.package_key) for s in packaged_segs]
            compile_plan = plan_mod.Plan(device, compile_decode=True, given={"VXPU_SEGMENT_RUNTIME": "aoti"})
            results, failures = compile_package.compile_packages(
                artifact,
                cache_dir,
                pkgs,
                compile_plan,
                machine_kind="cuda:nvidia-l4" if device == "cuda" else "cpu:test",
                software_version=speed_pf.tree_commit,
                device=device,
            )
            self.assertEqual(failures, [])
            self.assertEqual(len(results), len(packaged_segs))

            with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                with mock.patch.object(aoti, "build_segment_package", side_effect=AssertionError("must not build")), \
                     mock.patch.object(torch, "compile", side_effect=AssertionError("must not torch.compile")):
                    speed_eng = Engine(artifact, device=device, plan=speed_pf)
            speed_split = getattr(speed_eng._decode_split, "split", speed_eng._decode_split)
            expected_cls = aoti.GraphedSegment if device == "cuda" else aoti.Segment
            for s in packaged_segs:
                seg_mod = getattr(speed_split, s.name)
                self.assertIsInstance(seg_mod, expected_cls)
                if device == "cpu":
                    self.assertNotIsInstance(seg_mod, aoti.GraphedSegment)
            speed_tokens = [t for ev in speed_eng.generate(speed_eng.new_session(), [1, 2, 3], 5, params) for t in ev["token_ids"]]
            self.assertEqual(speed_tokens, pass_tokens)

            # 5. ExecutorServicer.LoadModel loads the staged speed plan and reports its plan_hash.
            with mock.patch.dict(os.environ, {"VXPU_AOTI_CACHE_DIR": cache_dir}):
                grpc_ctx = _StubContext()
                resp = servicer.LoadModel(load_req, grpc_ctx)
                self.assertTrue(resp.success)
                self.assertIsNone(grpc_ctx.code)
                deadline = time.time() + 120.0
                while time.time() < deadline:
                    gl = servicer.GetLoad(vxpu_pb2.GetLoadRequest(), _StubContext())
                    if gl.state != vxpu_pb2.LOAD_STATE_LOADING:
                        break
                    time.sleep(0.05)
                self.assertEqual(gl.state, vxpu_pb2.LOAD_STATE_READY, gl.error)
                self.assertEqual(gl.plan_hash, speed_pf.plan_hash)

    def test_check_plan_refuses_invalid_run_modes(self) -> None:
        """check_plan refuses JIT, unspecified, invalid keys, and missing staged packages."""
        with tempfile.TemporaryDirectory() as td:
            jit_pf = plan_pb2.PlanFile(
                segments=[plan_pb2.Segment(name="submod_0", run=plan_pb2.SEGMENT_RUN_JIT)]
            )
            with self.assertRaises(ValueError) as ctx:
                segments.check_plan(jit_pf, "cpu", td)
            self.assertIn("SEGMENT_RUN_JIT", str(ctx.exception))

            unspec_pf = plan_pb2.PlanFile(
                segments=[plan_pb2.Segment(name="submod_0", run=plan_pb2.SEGMENT_RUN_UNSPECIFIED)]
            )
            with self.assertRaises(ValueError) as ctx:
                segments.check_plan(unspec_pf, "cpu", td)
            self.assertIn("SEGMENT_RUN_UNSPECIFIED", str(ctx.exception))

            missing_key_pf = plan_pb2.PlanFile(
                segments=[plan_pb2.Segment(name="submod_0", run=plan_pb2.SEGMENT_RUN_PACKAGE)]
            )
            with self.assertRaises(ValueError) as ctx:
                segments.check_plan(missing_key_pf, "cpu", td)
            self.assertIn("package_key", str(ctx.exception))

            plain_with_key_pf = plan_pb2.PlanFile(
                segments=[plan_pb2.Segment(name="submod_0", run=plan_pb2.SEGMENT_RUN_PLAIN_TORCH, package_key="k0")]
            )
            with self.assertRaises(ValueError) as ctx:
                segments.check_plan(plain_with_key_pf, "cpu", td)
            self.assertIn("package_key", str(ctx.exception))

    def test_parse_package_arg_refuses_malformed(self) -> None:
        """parse_package_arg refuses malformed inputs by naming them."""
        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("bad_format")
        self.assertIn("expected name=key", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("=missing_name")
        self.assertIn("empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("missing_key=")
        self.assertIn("empty", str(ctx.exception))

    def test_shared_build_procedure(self) -> None:
        """The engine's load and the compile task share one way a package is built."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            pt2_path = os.path.join(td, "shared.pt2")
            gm = engine.load_dense_decode_artifact(self._artifact, self._plan, device=device)
            split, _ = segments.decode_segments(gm, self._plan, device=device)
            sub = split.submod_2
            self.assertIsInstance(sub, torch.fx.GraphModule)
            assert isinstance(sub, torch.fx.GraphModule)
            args = aoti.example_args(split, "submod_2", sub, device)

            aoti.build_segment_package(sub, args, pt2_path)
            self.assertTrue(os.path.exists(pt2_path))
            self.assertGreater(os.path.getsize(pt2_path), 0)

            runner = aoti.load_package(pt2_path)
            self.assertIsNotNone(runner)

    def test_refuse_plan_without_compile_decode(self) -> None:
        """compile_packages refuses a Plan built without compile_decode=True."""
        plan_no_compile = plan_mod.Plan("cpu", compile_decode=False)
        with self.assertRaises(ValueError) as ctx:
            compile_package.compile_packages(
                self._artifact,
                "/tmp/dummy",
                [plan_pb2.Package(name="submod_0", key="k0")],
                plan_no_compile,
                machine_kind="cpu:test:avx512",
                software_version="commit123",
            )
        self.assertIn("compile_decode=True", str(ctx.exception))

    def test_prepared_segments_match_engine_segments_with_lowering_on(self) -> None:
        """The segments prepared by the task equal the segments prepared by the engine for the same plan with lowering on."""
        lowered_given = {"VXPU_LOWER_DECODE": "1"}
        plan = plan_mod.Plan("cpu", compile_decode=True, given=lowered_given)
        self.assertTrue(plan["lower_decode"])

        eng = Engine(
            self._artifact,
            device="cpu",
            compile_decode=True,
            plan_given=lowered_given,
        )
        eng_split = getattr(eng._decode_split, "split", eng._decode_split)
        self.assertIsInstance(eng_split, torch.fx.GraphModule)
        assert isinstance(eng_split, torch.fx.GraphModule)
        eng_segments = dict(segments.segments_of(eng_split))

        gm = engine.load_dense_decode_artifact(self._artifact, plan, device="cpu")
        task_split, task_host = segments.decode_segments(gm, plan, device="cpu")
        task_segments = dict(segments.segments_of(task_split))

        dense_task = {name: sub for name, sub in task_segments.items() if name not in task_host}
        dense_eng = {name: sub for name, sub in eng_segments.items() if name not in task_host}

        self.assertGreater(len(dense_task), 0)
        self.assertEqual(sorted(dense_task.keys()), sorted(dense_eng.keys()))
        for name in dense_task:
            code_task = aoti.canonical_code(dense_task[name])
            code_eng = aoti.canonical_code(dense_eng[name])
            self.assertEqual(
                code_task,
                code_eng,
                f"canonical code mismatch for dense segment {name}",
            )


if __name__ == "__main__":
    unittest.main()
