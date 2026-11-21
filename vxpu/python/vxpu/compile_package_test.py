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
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock
import zipfile

import torch

from . import aoti, compile_package, deepseek_v41, engine, kernels, physplan, plan as plan_mod, plan_pb2, segments
from .engine import Engine
from .export import export_artifact
from .fixtures import tiny_deepseek_v41


def _extract_so_bytes(pt2_path: str) -> bytes:
    with zipfile.ZipFile(pt2_path) as z:
        for name in z.namelist():
            if name.endswith(".so"):
                return z.read(name)
    raise ValueError(f"no .so found in {pt2_path}")


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

        cmd_emit.extend([
            "--artifact",
            cls._artifact,
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

    def _get_planner_packages(self, machine: str | None = None) -> plan_pb2.PackageReport:
        """Obtains required compiled packages from the Go planner for tiny DeepSeek."""
        import http.server
        import shutil
        import threading

        if machine is None:
            if torch.cuda.is_available():
                machine = "g2-standard-48"
            else:
                machine = "c3d-standard-60"

        class StubStoreHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(404)
                self.end_headers()

            def do_HEAD(self) -> None:
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), StubStoreHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]

        vxpu_bin = shutil.which("vxpu")
        if vxpu_bin is not None:
            cmd = [
                vxpu_bin,
                "plan",
                "packages",
            ]
            cwd = None
        elif shutil.which("go") is not None:
            cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cmd = [
                "go",
                "run",
                "./cmd/vxpu",
                "plan",
                "packages",
            ]
        else:
            server.shutdown()
            server.server_close()
            if torch.cuda.is_available():
                self.fail("neither vxpu binary nor go is available on PATH")
            else:
                raise unittest.SkipTest("neither vxpu binary nor go is available on PATH")

        cmd.extend([
            "--plan",
            self._plan_file,
            "--artifact",
            self._artifact,
            "--machine",
            machine,
            "--model",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "--modelstore",
            f"http://127.0.0.1:{port}",
            "--json",
        ])

        try:
            env = os.environ.copy()
            env["GOTOOLCHAIN"] = "auto"
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=env)
            from google.protobuf import json_format
            report = plan_pb2.PackageReport()
            json_format.Parse(res.stdout, report)
            return report
        except subprocess.CalledProcessError as e:
            print(f"vxpu plan packages failed (exit code {e.returncode}):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}", file=sys.stderr)
            raise
        finally:
            server.shutdown()
            server.server_close()

    def test_compile_packages_and_engine_loads_all_builds_none(self) -> None:
        """The task compiles missing packages, and an engine pointed at the result builds none."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            cache_dir = os.path.join(td, "packages")

            machine = "g2-standard-48" if device == "cuda" else "c3d-standard-60"
            planner_report = self._get_planner_packages(machine=machine)
            self.assertEqual(len(planner_report.required), 6)
            planner_map = {p.name: p.key for p in planner_report.required}
            self.assertTrue(planner_report.machine_kind, "planner report machine_kind is empty")
            target_machine_kind = planner_report.machine_kind

            pf = physplan.read_plan_file(self._plan_file)
            plan = plan_mod.Plan(device, compile_decode=True, given={"VXPU_SEGMENT_RUNTIME": "aoti"})
            results, failures = compile_package.compile_packages(
                self._artifact,
                cache_dir,
                planner_report.missing,
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

                with mock.patch.dict(os.environ, {
                    "VXPU_SEGMENT_RUNTIME": "aoti",
                    "VXPU_AOTI_CACHE_DIR": cache_dir,
                }):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        Engine(
                            self._artifact,
                            device=device,
                            compile_decode=True,
                            packages=planner_report,
                        )
                    self.assertIn("AOTInductor segments: 6 loaded from the cache, 0 built", buf.getvalue())
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

    def test_engine_accepts_packages_report_on_cpu(self) -> None:
        """The engine accepts a PackageReport proto on CPU and populates self.packages."""
        report = plan_pb2.PackageReport(
            required=[plan_pb2.Package(name="submod_0", key="custom_key_0")],
        )
        eng = Engine(
            self._artifact,
            device="cpu",
            compile_decode=False,
            packages=report,
        )
        self.assertEqual(eng.packages, {"submod_0": "custom_key_0"})

    def test_key_given_is_key_stored_computes_no_key_of_its_own(self) -> None:
        """The key given is the key the package is stored under; it computes no key of its own."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            given_key = "planner_named_key_12345678"

            results, failures = compile_package.compile_packages(
                self._artifact,
                out_dir,
                [plan_pb2.Package(name="submod_2", key=given_key)],
                self._plan,
                machine_kind="cpu:test:avx512",
                software_version="commit123",
                device=device,
            )

            self.assertEqual(len(failures), 0)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["name"], "submod_2")
            self.assertEqual(results[0]["key"], given_key)
            expected_file = os.path.join(out_dir, f"{given_key}.pt2")
            self.assertTrue(os.path.exists(expected_file))
            self.assertEqual(results[0]["path"], expected_file)

    def test_task_refuses_invalid_packages(self) -> None:
        """The task refuses a package it was not given enough to build."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                compile_package.compile_packages(
                    self._artifact,
                    td,
                    [plan_pb2.Package(name="submod_2", key="key123")],
                    self._plan,
                    machine_kind="",
                    software_version="commit123",
                )
            self.assertIn("machine_kind is required", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                compile_package.compile_packages(
                    self._artifact,
                    td,
                    [plan_pb2.Package(name="submod_2", key="key123")],
                    self._plan,
                    machine_kind="cpu:test:avx512",
                    software_version="",
                )
            self.assertIn("software_version is required", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                compile_package.compile_packages(
                    self._artifact,
                    td,
                    [plan_pb2.Package(name="unknown_submod_99", key="key123")],
                    self._plan,
                    machine_kind="cpu:test:avx512",
                    software_version="commit123",
                )
            self.assertIn("unknown segment", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                compile_package.compile_packages(
                    self._artifact,
                    td,
                    [plan_pb2.Package(name="submod_1", key="key123")],
                    self._plan,
                    machine_kind="cpu:test:avx512",
                    software_version="commit123",
                )
            self.assertIn("host op", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                compile_package.compile_packages(
                    self._artifact,
                    td,
                    [plan_pb2.Package(name="submod_2", key="")],
                    self._plan,
                    machine_kind="cpu:test:avx512",
                    software_version="commit123",
                )
            self.assertIn("empty", str(ctx.exception))

    def test_parse_packages_refuses_malformed(self) -> None:
        """parse_packages and parse_package_arg refuse malformed inputs by naming them."""
        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("bad_format")
        self.assertIn("expected name=key", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("=missing_name")
        self.assertIn("empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            aoti.parse_package_arg("missing_key=")
        self.assertIn("empty", str(ctx.exception))

        empty_report = plan_pb2.PackageReport()
        with self.assertRaises(ValueError) as ctx:
            aoti.parse_packages(empty_report)
        self.assertIn("no packages", str(ctx.exception))

        bad_name_report = plan_pb2.PackageReport(
            required=[plan_pb2.Package(name="", key="key123")]
        )
        with self.assertRaises(ValueError) as ctx:
            aoti.parse_packages(bad_name_report)
        self.assertIn("empty name", str(ctx.exception))

        bad_key_report = plan_pb2.PackageReport(
            required=[plan_pb2.Package(name="submod_0", key="")]
        )
        with self.assertRaises(ValueError) as ctx:
            aoti.parse_packages(bad_key_report)
        self.assertIn("empty key", str(ctx.exception))

        mixed_report = plan_pb2.PackageReport(
            required=[
                plan_pb2.Package(name="submod_0", key="k0"),
                plan_pb2.Package(name="submod_2", key="k2"),
            ],
            present=[plan_pb2.Package(name="submod_0", key="k0")],
            missing=[plan_pb2.Package(name="submod_2", key="k2")],
        )
        self.assertEqual(aoti.parse_packages(mixed_report, which="required"), {"submod_0": "k0", "submod_2": "k2"})
        self.assertEqual(aoti.parse_packages(mixed_report, which="missing"), {"submod_2": "k2"})
        self.assertEqual(aoti.parse_packages(mixed_report, which="present"), {"submod_0": "k0"})

        with self.assertRaises(ValueError) as ctx:
            aoti.parse_packages(mixed_report, which="unknown")
        self.assertIn("unknown", str(ctx.exception))

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

    def test_cli_compilation(self) -> None:
        """The task runs from the command line and reports compiled packages."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "cli_out")
            cmd = [
                sys.executable,
                "-m",
                "vxpu.compile_package",
                "--artifact",
                self._artifact,
                "--plan",
                self._plan_file,
                "--out",
                out_dir,
                "--device",
                device,
                "--machine-kind",
                "cpu:test:avx512",
                "--software-version",
                "test-commit",
                "submod_2=cli_test_key",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("[vxpu] compiled package submod_2 -> cli_test_key.pt2", res.stdout)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "cli_test_key.pt2")))

    def test_cli_compilation_reports_failures_to_report_file(self) -> None:
        """The CLI compiles packages and records failures into --out-report."""
        device = "cpu"
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "cli_out")
            report_out = os.path.join(td, "report.json")
            cmd = [
                sys.executable,
                "-m",
                "vxpu.compile_package",
                "--artifact",
                self._artifact,
                "--plan",
                self._plan_file,
                "--out",
                out_dir,
                "--device",
                device,
                "--machine-kind",
                "cpu:test:avx512",
                "--software-version",
                "test-commit",
                "--out-report",
                report_out,
                "submod_6=cli_fail_key",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("[vxpu] failed to compile package submod_6", res.stdout)
            self.assertTrue(os.path.exists(report_out))

            rep = compile_package.read_package_report(report_out)
            self.assertEqual(len(rep.failed), 1)
            self.assertEqual(rep.failed[0].package.name, "submod_6")
            self.assertEqual(rep.failed[0].package.key, "cli_fail_key")
            self.assertEqual(rep.failed[0].machine_kind, "cpu:test:avx512")
            self.assertEqual(rep.failed[0].software_version, "test-commit")
            self.assertTrue(rep.failed[0].reason)

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

    def test_internal_error_fails_task_and_records_no_failure(self) -> None:
        """A deliberate error in our own code during a build fails the task and records no failure."""
        import torch._dynamo.exc as dynamo_exc

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("vxpu.aoti.build_segment_package", side_effect=RuntimeError("deliberate internal error")):
                with self.assertRaises(RuntimeError) as ctx:
                    compile_package.compile_packages(
                        self._artifact,
                        td,
                        [plan_pb2.Package(name="submod_2", key="key2")],
                        self._plan,
                        machine_kind="cpu:test:avx512",
                        software_version="commit123",
                    )
                self.assertIn("deliberate internal error", str(ctx.exception))

            wrapped = dynamo_exc.BackendCompilerFailed(lambda: None, ValueError("deliberate wrapped error"), None)
            with mock.patch("vxpu.aoti.build_segment_package", side_effect=wrapped):
                with self.assertRaises(RuntimeError) as ctx:
                    compile_package.compile_packages(
                        self._artifact,
                        td,
                        [plan_pb2.Package(name="submod_2", key="key2")],
                        self._plan,
                        machine_kind="cpu:test:avx512",
                        software_version="commit123",
                    )
                self.assertIn("deliberate wrapped error", str(ctx.exception))

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

        gm, _ = engine.load_dense_decode_for_compile(self._artifact, plan, device="cpu")
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

    def test_compile_packages_without_weight_files(self) -> None:
        """The task builds a segment package from an artifact that has no weight files."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as td:
            # Create an artifact directory containing only graph and metadata files.
            art_dir = os.path.join(td, "no_weight_artifact")
            os.makedirs(art_dir)
            shutil.copy(os.path.join(self._artifact, "binding.json"), art_dir)
            shutil.copy(os.path.join(self._artifact, "decode.pt2"), art_dir)

            # Copy manifest.json but point all weight sources to non-existent paths so that
            # if any weight-reading path is triggered, it fails immediately.
            with open(os.path.join(self._artifact, "manifest.json")) as f:
                manifest = json.load(f)
            for f_info in manifest.get("files", {}).values():
                f_info["source"] = "file:///nonexistent/weights.safetensors"
            with open(os.path.join(art_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            out_dir = os.path.join(td, "packages")
            with mock.patch.dict(os.environ, {
                "VXPU_AOTI_CACHE_DIR": out_dir,
                **({"VXPU_SEGMENT_RUNTIME": "aoti"} if device == "cuda" else {}),
            }):
                plan = plan_mod.Plan(device, compile_decode=True)

                given_key = "test_pkg_submod_2"
                results, failures = compile_package.compile_packages(
                    art_dir,
                    out_dir,
                    [plan_pb2.Package(name="submod_2", key=given_key)],
                    plan,
                    machine_kind="gpu:test:l4" if device == "cuda" else "cpu:test:avx512",
                    software_version="commit123",
                    device=device,
                )

                self.assertEqual(len(failures), 0)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["name"], "submod_2")
                self.assertEqual(results[0]["key"], given_key)
                expected_pt2 = os.path.join(out_dir, f"{given_key}.pt2")
                self.assertTrue(os.path.exists(expected_pt2))
                self.assertGreater(os.path.getsize(expected_pt2), 0)

                # Confirm that the compiled package loads as an AOTInductor package.
                runner = aoti.load_package(expected_pt2)
                self.assertIsNotNone(runner)

                # On CUDA, compile all dense segments and verify that the engine loads them without compiling.
                if device == "cuda":
                    planner_report = self._get_planner_packages(machine="g2-standard-48")
                    target_machine_kind = planner_report.machine_kind
                    results_all, failures_all = compile_package.compile_packages(
                        art_dir,
                        out_dir,
                        planner_report.required,
                        plan,
                        machine_kind=target_machine_kind,
                        software_version="commit123",
                        device=device,
                    )
                    self.assertEqual(len(failures_all), 0)
                    self.assertEqual(len(results_all), len(planner_report.required))

                    eng_ref = Engine(self._artifact, device=device, compile_decode=False)
                    sid_ref = eng_ref.new_session()
                    events_ref = list(eng_ref.generate(sid_ref, [1, 2, 3], 5, engine.SamplingParams(do_sample=False)))
                    tokens_ref = [e["token_ids"][0] for e in events_ref if e.get("token_ids")]

                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        eng = Engine(
                            self._artifact,
                            device=device,
                            compile_decode=True,
                            packages=planner_report,
                        )
                    self.assertIn(f"AOTInductor segments: {len(planner_report.required)} loaded from the cache, 0 built", buf.getvalue())

                    sid = eng.new_session()
                    events = list(eng.generate(sid, [1, 2, 3], 5, engine.SamplingParams(do_sample=False)))
                    tokens = [e["token_ids"][0] for e in events if e.get("token_ids")]
                    self.assertEqual(tokens, tokens_ref)

    def test_compiled_libraries_match_real_weights(self) -> None:
        """Packages compiled with placeholders have identical compiled libraries to real weights."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        plan = plan_mod.Plan(
            device,
            compile_decode=True,
            given={"VXPU_SEGMENT_RUNTIME": "aoti"} if device == "cuda" else None,
        )
        gm_real = engine.load_dense_decode_artifact(self._artifact, plan, device=device)
        split_real, host = segments.decode_segments(gm_real, plan, device=device)

        with tempfile.TemporaryDirectory() as td:
            target_machine = "g2-standard-48" if device == "cuda" else "c3d-standard-60"
            planner_report = self._get_planner_packages(machine=target_machine)
            out_ph = os.path.join(td, "packages_ph")
            results_ph, failures_ph = compile_package.compile_packages(
                self._artifact,
                out_ph,
                planner_report.required,
                plan,
                machine_kind=planner_report.machine_kind,
                software_version="commit123",
                device=device,
            )

            if device == "cuda":
                expected_compared = {"submod_0", "submod_2", "submod_4", "submod_6", "submod_8", "submod_10"}
                expected_failures = set()
            else:
                expected_compared = {"submod_2", "submod_10"}
                expected_failures = {"submod_0", "submod_4", "submod_6", "submod_8"}

            failed_names = {f.package.name for f in failures_ph}
            self.assertEqual(failed_names, expected_failures)

            compared_names = {res["name"] for res in results_ph}
            self.assertEqual(compared_names, expected_compared)
            for res in results_ph:
                name = res["name"]
                sub_real = getattr(split_real, name)
                args_real = aoti.example_args(split_real, name, sub_real, device)
                pt2_real = os.path.join(td, f"{name}_real.pt2")
                aoti.build_segment_package(sub_real, args_real, pt2_real)

                so_ph = _extract_so_bytes(res["path"])
                so_real = _extract_so_bytes(pt2_real)
                self.assertEqual(
                    so_ph,
                    so_real,
                    f"compiled library mismatch for segment {name}",
                )


if __name__ == "__main__":
    unittest.main()
