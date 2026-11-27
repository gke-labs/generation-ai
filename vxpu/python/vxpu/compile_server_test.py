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

"""Tests for the vxpu gRPC compile server."""

from concurrent import futures
import os
import tempfile
import time
from unittest import mock
import unittest

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
import torch

from . import aoti, compile_package, compile_server, engine, plan_pb2, segments, vxpu_pb2, vxpu_pb2_grpc


class CompileServerTest(unittest.TestCase):
    """Verifies that CompilerServicer serves compilation over gRPC."""

    def setUp(self) -> None:
        self.staged_dir = "/staged/artifact"
        self.artifact_ref = "vxpu-artifacts/test-model:v1"
        self.software_version = "commit1"
        self.server = compile_server.create_server(
            max_workers=4,
            modelstore="http://test-modelstore",
            artifact_dir=self.staged_dir,
            artifact_ref=self.artifact_ref,
            software_version=self.software_version,
        )
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port}")
        self.stub = vxpu_pb2_grpc.CompilerStub(self.channel)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)

    def tearDown(self) -> None:
        self.channel.close()
        self.server.stop(grace=None)

    def test_health_check_serving(self) -> None:
        """The standard gRPC health service reports SERVING."""
        resp = self.health_stub.Check(health_pb2.HealthCheckRequest(service=""))  # pyright: ignore[reportAttributeAccessIssue]
        self.assertEqual(resp.status, health_pb2.HealthCheckResponse.SERVING)

    def test_validation_errors(self) -> None:
        """Invalid requests omitting required fields or naming another artifact return INVALID_ARGUMENT."""
        ref = self.artifact_ref
        ver = self.software_version
        test_cases = [
            vxpu_pb2.CompileRequest(),
            vxpu_pb2.CompileRequest(model="m"),
            vxpu_pb2.CompileRequest(model="m", artifact="vxpu-artifacts/other:v2", plan=plan_pb2.PlanFile()),
            vxpu_pb2.CompileRequest(model="m", artifact=ref),
            vxpu_pb2.CompileRequest(model="m", artifact=ref, plan=plan_pb2.PlanFile()),
            vxpu_pb2.CompileRequest(model="m", artifact=ref, plan=plan_pb2.PlanFile(), machine_kind="m"),
            vxpu_pb2.CompileRequest(model="m", artifact=ref, plan=plan_pb2.PlanFile(), machine_kind="m", software_version=ver),
            vxpu_pb2.CompileRequest(model="m", artifact=ref, plan=plan_pb2.PlanFile(), machine_kind="m", software_version=ver, device="cpu"),
        ]
        for req in test_cases:
            with self.subTest(req=req):
                with self.assertRaises(grpc.RpcError) as ctx:
                    self.stub.Compile(req)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    def test_mismatched_software_version_refused_naming_both(self) -> None:
        """A request whose software_version differs from the executor's is refused naming both."""
        req = vxpu_pb2.CompileRequest(
            model="test-model",
            artifact=self.artifact_ref,
            plan=plan_pb2.PlanFile(),
            machine_kind="cpu:test:avx512",
            software_version="other-commit-999",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_0", key="key0")],
        )
        with self.assertRaises(grpc.RpcError) as ctx:
            self.stub.Compile(req)
        self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        details = ctx.exception.details() or ""
        self.assertIn("other-commit-999", details)
        self.assertIn(self.software_version, details)

    def test_startup_requires_modelstore_and_staged_artifact(self) -> None:
        """Creating the compiler server without modelstore, artifact_dir, artifact_ref, or software_version raises ValueError."""
        with self.assertRaises(ValueError):
            compile_server.create_server(max_workers=1, modelstore="", artifact_dir=self.staged_dir, artifact_ref=self.artifact_ref, software_version=self.software_version)
        with self.assertRaises(ValueError):
            compile_server.create_server(max_workers=1, modelstore="http://test-modelstore", artifact_dir="", artifact_ref=self.artifact_ref, software_version=self.software_version)
        with self.assertRaises(ValueError):
            compile_server.create_server(max_workers=1, modelstore="http://test-modelstore", artifact_dir=self.staged_dir, artifact_ref="", software_version=self.software_version)
        with self.assertRaises(ValueError):
            compile_server.create_server(max_workers=1, modelstore="http://test-modelstore", artifact_dir=self.staged_dir, artifact_ref=self.artifact_ref, software_version="")

    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_compile_success_and_second_request_on_same_connection(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
    ) -> None:
        """A package compiles successfully and a second request on the same connection works."""
        with tempfile.TemporaryDirectory() as td:
            pt2_0 = os.path.join(td, "key0.pt2")
            with open(pt2_0, "wb") as f:
                f.write(b"data0")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path=pt2_0, bytes=5)],
                [],
            )
            req1 = vxpu_pb2.CompileRequest(
                artifact=self.artifact_ref,
                plan=plan_pb2.PlanFile(),
                machine_kind="cpu:test:avx512",
                software_version="commit1",
                device="cpu",
                packages=[plan_pb2.Package(name="submod_0", key="key0")],
                model="test-model",
            )
            resp1 = self.stub.Compile(req1)
            self.assertEqual(len(resp1.report.present), 1)
            self.assertEqual(resp1.report.present[0].name, "submod_0")
            self.assertEqual(resp1.report.present[0].key, "key0")
            self.assertEqual(len(resp1.report.failed), 0)
            _, kwargs = mock_compile.call_args
            self.assertEqual(kwargs.get("artifact_dir"), self.staged_dir)

            # Second request on the same connection
            pt2_1 = os.path.join(td, "key1.pt2")
            with open(pt2_1, "wb") as f:
                f.write(b"data1")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_1", key="key1", path=pt2_1, bytes=5)],
                [],
            )
            req2 = vxpu_pb2.CompileRequest(
                artifact=self.artifact_ref,
                plan=plan_pb2.PlanFile(),
                machine_kind="cpu:test:avx512",
                software_version="commit1",
                device="cpu",
                packages=[plan_pb2.Package(name="submod_1", key="key1")],
                model="test-model",
            )
            resp2 = self.stub.Compile(req2)
            self.assertEqual(len(resp2.report.present), 1)
            self.assertEqual(resp2.report.present[0].name, "submod_1")
            self.assertEqual(resp2.report.present[0].key, "key1")
            self.assertEqual(len(resp2.report.failed), 0)

    @mock.patch.object(compile_package, "compile_packages")
    def test_compile_failure_returned_as_result(
        self, mock_compile: mock.MagicMock
    ) -> None:
        """A compile failure comes back as a result naming the package rather than an RPC error."""
        failure = plan_pb2.CompileFailure(
            package=plan_pb2.Package(name="submod_bad", key="key_bad"),
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            reason="compiler crash in lowering",
        )
        mock_compile.return_value = ([], [failure])

        req = vxpu_pb2.CompileRequest(
            artifact=self.artifact_ref,
            plan=plan_pb2.PlanFile(),
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_bad", key="key_bad")],
            model="test-model",
        )
        resp = self.stub.Compile(req)
        self.assertEqual(len(resp.report.present), 0)
        self.assertEqual(len(resp.report.failed), 1)
        self.assertEqual(resp.report.failed[0].package.name, "submod_bad")
        self.assertEqual(resp.report.failed[0].reason, "compiler crash in lowering")

    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_simultaneous_compiles_are_serialized(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
    ) -> None:
        """Simultaneous Compile requests are serialized by the server lock."""
        active = 0
        max_active = 0
        count = 0

        with tempfile.TemporaryDirectory() as td:
            def slow_compile(*args: object, **kwargs: object):
                nonlocal active, max_active, count
                active += 1
                max_active = max(max_active, active)
                time.sleep(0.05)
                active -= 1
                count += 1
                fpath = os.path.join(td, f"key_{count}.pt2")
                with open(fpath, "wb") as f:
                    f.write(b"data")
                return (
                    [compile_package.CompiledPackage(name="submod_0", key=f"key{count}", path=fpath, bytes=4)],
                    [],
                )

            mock_compile.side_effect = slow_compile

            req = vxpu_pb2.CompileRequest(
                artifact=self.artifact_ref,
                plan=plan_pb2.PlanFile(),
                machine_kind="cpu:test:avx512",
                software_version="commit1",
                device="cpu",
                packages=[plan_pb2.Package(name="submod_0", key="key0")],
                model="test-model",
            )

            with futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(self.stub.Compile, req)
                f2 = pool.submit(self.stub.Compile, req)
                resp1 = f1.result(timeout=5)
                resp2 = f2.result(timeout=5)

            self.assertEqual(len(resp1.report.present), 1)
            self.assertEqual(len(resp2.report.present), 1)
            self.assertEqual(max_active, 1)

    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_compile_uploads_via_push_directory(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
    ) -> None:
        """Compiler builds into a temporary directory and uploads via push_directory."""
        with tempfile.TemporaryDirectory() as td:
            pt2_file = os.path.join(td, "key0.pt2")
            with open(pt2_file, "wb") as f:
                f.write(b"data")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path=pt2_file, bytes=4)],
                [],
            )
            req = vxpu_pb2.CompileRequest(
                artifact=self.artifact_ref,
                plan=plan_pb2.PlanFile(),
                machine_kind="cpu",
                software_version=self.software_version,
                device="cpu",
                packages=[plan_pb2.Package(name="submod_0", key="key0")],
                model="test-model",
            )
            resp = self.stub.Compile(req)
            self.assertEqual(len(resp.report.present), 1)
            self.assertEqual(resp.report.present[0].name, "submod_0")
            mock_push_dir.assert_called_once()
            args, kwargs = mock_push_dir.call_args
            self.assertEqual(kwargs.get("base_url") or args[0], "http://test-modelstore")
            self.assertEqual(kwargs.get("name") or args[1], "vxpu-packages/test-model")
            self.assertEqual(kwargs.get("tag") or args[2], "key0")

    @mock.patch.object(compile_package, "compile_packages")
    def test_unproduced_package_recorded_as_failure(
        self,
        mock_compile: mock.MagicMock,
    ) -> None:
        """A package that was not produced on disk is reported as a failure with the reason."""
        mock_compile.return_value = (
            [compile_package.CompiledPackage(name="submod_missing", key="key_missing", path="/nonexistent/missing.pt2", bytes=0)],
            [],
        )
        req = vxpu_pb2.CompileRequest(
            artifact=self.artifact_ref,
            plan=plan_pb2.PlanFile(),
            machine_kind="cpu",
            software_version=self.software_version,
            device="cpu",
            packages=[plan_pb2.Package(name="submod_missing", key="key_missing")],
            model="test-model",
        )
        resp = self.stub.Compile(req)
        self.assertEqual(len(resp.report.present), 0)
        self.assertEqual(len(resp.report.failed), 1)
        self.assertEqual(resp.report.failed[0].package.name, "submod_missing")
        self.assertIn("not produced", resp.report.failed[0].reason)

    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_store_failure_raises_rpc_internal_error(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
    ) -> None:
        """A failure to store is returned as an RPC error (INTERNAL), not a compile failure result."""
        mock_push_dir.side_effect = RuntimeError("network connection refused by modelstore")
        with tempfile.TemporaryDirectory() as td:
            pt2_file = os.path.join(td, "key0.pt2")
            with open(pt2_file, "wb") as f:
                f.write(b"data")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path=pt2_file, bytes=4)],
                [],
            )
            req = vxpu_pb2.CompileRequest(
                artifact=self.artifact_ref,
                plan=plan_pb2.PlanFile(),
                machine_kind="cpu",
                software_version=self.software_version,
                device="cpu",
                packages=[plan_pb2.Package(name="submod_0", key="key0")],
                model="test-model",
            )
            with self.assertRaises(grpc.RpcError) as ctx:
                self.stub.Compile(req)
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INTERNAL)
            self.assertIn("network connection refused", ctx.exception.details() or "")

    def test_internal_error_fails_task_and_records_no_failure(self) -> None:
        """A deliberate internal error during a build fails the compile RPC with INTERNAL rather than recording a failure."""
        import torch._dynamo.exc as dynamo_exc

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        sub = torch.fx.GraphModule(torch.nn.Module(), graph)
        split = torch.nn.Module()
        split.submod_2 = sub

        req = vxpu_pb2.CompileRequest(
            artifact=self.artifact_ref,
            plan=plan_pb2.PlanFile(),
            machine_kind="cpu:test:avx512",
            software_version=self.software_version,
            device="cpu",
            packages=[plan_pb2.Package(name="submod_2", key="key2")],
            model="test-model",
        )

        cases: list[tuple[BaseException, str]] = [
            (RuntimeError("deliberate internal error"), "deliberate internal error"),
            (
                dynamo_exc.BackendCompilerFailed(lambda: None, ValueError("deliberate wrapped error"), None),
                "deliberate wrapped error",
            ),
        ]

        with mock.patch.object(engine, "load_dense_decode_artifact", return_value=sub), \
             mock.patch.object(segments, "decode_segments", return_value=(split, [])), \
             mock.patch.object(aoti, "example_args", return_value=(torch.zeros(1),)):
            for err, expected_msg in cases:
                with self.subTest(expected_msg=expected_msg):
                    with mock.patch.object(aoti, "build_segment_package", side_effect=err):
                        with self.assertRaises(grpc.RpcError) as ctx:
                            self.stub.Compile(req)
                        self.assertEqual(ctx.exception.code(), grpc.StatusCode.INTERNAL)
                        self.assertIn(expected_msg, ctx.exception.details() or "")


if __name__ == "__main__":
    unittest.main()
