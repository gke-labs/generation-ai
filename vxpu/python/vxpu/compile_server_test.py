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

from . import compile_package, compile_server, physplan, plan, plan_pb2, vxpu_pb2, vxpu_pb2_grpc


class CompileServerTest(unittest.TestCase):
    """Verifies that CompilerServicer serves compilation over gRPC."""

    def setUp(self) -> None:
        self.server = compile_server.create_server(max_workers=4, modelstore="http://test-modelstore")
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
        """Invalid requests omitting required fields return INVALID_ARGUMENT."""
        test_cases = [
            vxpu_pb2.CompileRequest(),
            vxpu_pb2.CompileRequest(model="m"),
            vxpu_pb2.CompileRequest(model="m", artifact="/a"),
            vxpu_pb2.CompileRequest(model="m", artifact="/a", plan_file="/p"),
            vxpu_pb2.CompileRequest(model="m", artifact="/a", plan_file="/p", machine_kind="m"),
            vxpu_pb2.CompileRequest(model="m", artifact="/a", plan_file="/p", machine_kind="m", software_version="s"),
            vxpu_pb2.CompileRequest(model="m", artifact="/a", plan_file="/p", machine_kind="m", software_version="s", device="cpu"),
        ]
        for req in test_cases:
            with self.subTest(req=req):
                with self.assertRaises(grpc.RpcError) as ctx:
                    self.stub.Compile(req)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    def test_compile_fails_without_modelstore(self) -> None:
        """When the compiler has no modelstore configured, Compile returns FAILED_PRECONDITION."""
        server = compile_server.create_server(max_workers=1, modelstore=None)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
                stub = vxpu_pb2_grpc.CompilerStub(ch)
                req = vxpu_pb2.CompileRequest(
                    model="test-model",
                    artifact="/fake/a",
                    plan_file="/fake/p",
                    machine_kind="m",
                    software_version="s",
                    device="cpu",
                    packages=[plan_pb2.Package(name="submod_0", key="k0")],
                )
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.Compile(req)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
        finally:
            server.stop(grace=None)

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_compile_success_and_second_request_on_same_connection(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
        mock_read_plan: mock.MagicMock,
    ) -> None:
        """A package compiles successfully and a second request on the same connection works."""
        mock_read_plan.return_value = plan_pb2.PlanFile()

        with tempfile.TemporaryDirectory() as td:
            pt2_0 = os.path.join(td, "key0.pt2")
            with open(pt2_0, "wb") as f:
                f.write(b"data0")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path=pt2_0, bytes=5)],
                [],
            )
            out_dir = os.path.join(td, "out")
            req1 = vxpu_pb2.CompileRequest(
                artifact="/fake/artifact",
                plan_file="/fake/plan.pb",
                out_dir=out_dir,
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

            # Second request on the same connection
            pt2_1 = os.path.join(td, "key1.pt2")
            with open(pt2_1, "wb") as f:
                f.write(b"data1")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_1", key="key1", path=pt2_1, bytes=5)],
                [],
            )
            req2 = vxpu_pb2.CompileRequest(
                artifact="/fake/artifact",
                plan_file="/fake/plan.pb",
                out_dir=out_dir,
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

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    def test_compile_failure_returned_as_result(
        self, mock_compile: mock.MagicMock, mock_read_plan: mock.MagicMock
    ) -> None:
        """A compile failure comes back as a result naming the package rather than an RPC error."""
        mock_read_plan.return_value = plan_pb2.PlanFile()
        failure = plan_pb2.CompileFailure(
            package=plan_pb2.Package(name="submod_bad", key="key_bad"),
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            reason="compiler crash in lowering",
        )
        mock_compile.return_value = ([], [failure])

        with tempfile.TemporaryDirectory() as td:
            req = vxpu_pb2.CompileRequest(
                artifact="/fake/artifact",
                plan_file="/fake/plan.pb",
                out_dir=os.path.join(td, "out"),
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

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_simultaneous_compiles_are_serialized(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
        mock_read_plan: mock.MagicMock,
    ) -> None:
        """Simultaneous Compile requests are serialized by the server lock."""
        mock_read_plan.return_value = plan_pb2.PlanFile()

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
                artifact="/fake/artifact",
                plan_file="/fake/plan.pb",
                out_dir=os.path.join(td, "out"),
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
            # Verify that slow_compile was never executed concurrently
            self.assertEqual(max_active, 1)

    @mock.patch("vxpu.modelstore.push_directory")
    def test_compile_real_tiny_deepseek_segment(self, mock_push_dir: mock.MagicMock) -> None:
        """A package compiles for a segment of the tiny model through the gRPC server."""
        try:
            from .fixtures import tiny_deepseek_v41
            art_dir = tiny_deepseek_v41.get_artifact_dir(max_cache_len=64)
        except (ImportError, ModuleNotFoundError) as e:
            if plan.require_model_source():
                raise
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            plan_file = os.path.join(td, "plan.pb")
            physplan.write_plan_file(plan_file, plan_pb2.PlanFile())
            out_dir = os.path.join(td, "packages")
            req = vxpu_pb2.CompileRequest(
                artifact=art_dir,
                plan_file=plan_file,
                out_dir=out_dir,
                machine_kind="cpu:test:avx512",
                software_version="test-commit-1",
                device="cpu",
                packages=[plan_pb2.Package(name="submod_2", key="key_submod_2_py")],
                model="test-model",
            )
            resp = self.stub.Compile(req)
            self.assertEqual(len(resp.report.present), 1)
            self.assertEqual(resp.report.present[0].name, "submod_2")
            self.assertEqual(resp.report.present[0].key, "key_submod_2_py")
            self.assertEqual(len(resp.report.failed), 0)
            mock_push_dir.assert_called_once()

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_compile_with_optional_out_dir_and_modelstore_upload(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
        mock_read_plan: mock.MagicMock,
    ) -> None:
        """When out_dir is omitted, compiler uses a temporary directory and uploads via push_directory."""
        mock_read_plan.return_value = plan_pb2.PlanFile()
        with tempfile.TemporaryDirectory() as td:
            pt2_file = os.path.join(td, "key0.pt2")
            with open(pt2_file, "wb") as f:
                f.write(b"data")
            mock_compile.return_value = (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path=pt2_file, bytes=4)],
                [],
            )
            req = vxpu_pb2.CompileRequest(
                artifact="/fake/art",
                plan_file="/fake/plan.pb",
                machine_kind="cpu",
                software_version="commit-1",
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

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    def test_unproduced_package_recorded_as_failure(
        self,
        mock_compile: mock.MagicMock,
        mock_read_plan: mock.MagicMock,
    ) -> None:
        """A package that was not produced on disk is reported as a failure with the reason."""
        mock_read_plan.return_value = plan_pb2.PlanFile()
        mock_compile.return_value = (
            [compile_package.CompiledPackage(name="submod_missing", key="key_missing", path="/nonexistent/missing.pt2", bytes=0)],
            [],
        )
        req = vxpu_pb2.CompileRequest(
            artifact="/fake/art",
            plan_file="/fake/plan.pb",
            machine_kind="cpu",
            software_version="commit-1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_missing", key="key_missing")],
            model="test-model",
        )
        resp = self.stub.Compile(req)
        self.assertEqual(len(resp.report.present), 0)
        self.assertEqual(len(resp.report.failed), 1)
        self.assertEqual(resp.report.failed[0].package.name, "submod_missing")
        self.assertIn("not produced", resp.report.failed[0].reason)

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    @mock.patch("vxpu.modelstore.push_directory")
    def test_store_failure_raises_rpc_internal_error(
        self,
        mock_push_dir: mock.MagicMock,
        mock_compile: mock.MagicMock,
        mock_read_plan: mock.MagicMock,
    ) -> None:
        """A failure to store is returned as an RPC error (INTERNAL), not a compile failure result."""
        mock_read_plan.return_value = plan_pb2.PlanFile()
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
                artifact="/fake/art",
                plan_file="/fake/plan.pb",
                machine_kind="cpu",
                software_version="commit-1",
                device="cpu",
                packages=[plan_pb2.Package(name="submod_0", key="key0")],
                model="test-model",
            )
            with self.assertRaises(grpc.RpcError) as ctx:
                self.stub.Compile(req)
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INTERNAL)
            self.assertIn("network connection refused", ctx.exception.details() or "")


if __name__ == "__main__":
    unittest.main()
