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
        self.server = compile_server.create_server(max_workers=4)
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
            vxpu_pb2.CompileRequest(artifact="/a"),
            vxpu_pb2.CompileRequest(artifact="/a", plan_file="/p"),
            vxpu_pb2.CompileRequest(artifact="/a", plan_file="/p", out_dir="/o"),
            vxpu_pb2.CompileRequest(artifact="/a", plan_file="/p", out_dir="/o", machine_kind="m"),
            vxpu_pb2.CompileRequest(artifact="/a", plan_file="/p", out_dir="/o", machine_kind="m", software_version="s"),
            vxpu_pb2.CompileRequest(artifact="/a", plan_file="/p", out_dir="/o", machine_kind="m", software_version="s", device="cpu"),
        ]
        for req in test_cases:
            with self.subTest(req=req):
                with self.assertRaises(grpc.RpcError) as ctx:
                    self.stub.Compile(req)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    def test_compile_success_and_second_request_on_same_connection(
        self, mock_compile: mock.MagicMock, mock_read_plan: mock.MagicMock
    ) -> None:
        """A package compiles successfully and a second request on the same connection works."""
        mock_read_plan.return_value = plan_pb2.PlanFile()
        mock_compile.return_value = (
            [compile_package.CompiledPackage(name="submod_0", key="key0", path="/tmp/key0.pt2", bytes=1024)],
            [],
        )

        req1 = vxpu_pb2.CompileRequest(
            artifact="/fake/artifact",
            plan_file="/fake/plan.pb",
            out_dir="/fake/out",
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_0", key="key0")],
        )
        resp1 = self.stub.Compile(req1)
        self.assertEqual(len(resp1.report.present), 1)
        self.assertEqual(resp1.report.present[0].name, "submod_0")
        self.assertEqual(resp1.report.present[0].key, "key0")
        self.assertEqual(len(resp1.report.failed), 0)

        # Second request on the same connection
        mock_compile.return_value = (
            [compile_package.CompiledPackage(name="submod_1", key="key1", path="/tmp/key1.pt2", bytes=2048)],
            [],
        )
        req2 = vxpu_pb2.CompileRequest(
            artifact="/fake/artifact",
            plan_file="/fake/plan.pb",
            out_dir="/fake/out",
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_1", key="key1")],
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

        req = vxpu_pb2.CompileRequest(
            artifact="/fake/artifact",
            plan_file="/fake/plan.pb",
            out_dir="/fake/out",
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_bad", key="key_bad")],
        )
        resp = self.stub.Compile(req)
        self.assertEqual(len(resp.report.present), 0)
        self.assertEqual(len(resp.report.failed), 1)
        self.assertEqual(resp.report.failed[0].package.name, "submod_bad")
        self.assertEqual(resp.report.failed[0].reason, "compiler crash in lowering")

    @mock.patch.object(physplan, "read_plan_file")
    @mock.patch.object(compile_package, "compile_packages")
    def test_simultaneous_compiles_are_serialized(
        self, mock_compile: mock.MagicMock, mock_read_plan: mock.MagicMock
    ) -> None:
        """Simultaneous Compile requests are serialized by the server lock."""
        mock_read_plan.return_value = plan_pb2.PlanFile()

        active = 0
        max_active = 0

        def slow_compile(*args: object, **kwargs: object):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            time.sleep(0.05)
            active -= 1
            return (
                [compile_package.CompiledPackage(name="submod_0", key="key0", path="/tmp/key0.pt2", bytes=1024)],
                [],
            )

        mock_compile.side_effect = slow_compile

        req = vxpu_pb2.CompileRequest(
            artifact="/fake/artifact",
            plan_file="/fake/plan.pb",
            out_dir="/fake/out",
            machine_kind="cpu:test:avx512",
            software_version="commit1",
            device="cpu",
            packages=[plan_pb2.Package(name="submod_0", key="key0")],
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

    def test_compile_real_tiny_deepseek_segment(self) -> None:
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
            )
            resp = self.stub.Compile(req)
            self.assertEqual(len(resp.report.present), 1)
            self.assertEqual(resp.report.present[0].name, "submod_2")
            self.assertEqual(resp.report.present[0].key, "key_submod_2_py")
            self.assertEqual(len(resp.report.failed), 0)
            pt2 = os.path.join(out_dir, "key_submod_2_py.pt2")
            self.assertTrue(os.path.isfile(pt2))


if __name__ == "__main__":
    unittest.main()
