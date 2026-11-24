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

# pyright: strict

"""gRPC compilation server for vxpu packages.

The compiler process runs when requested through the agent, serving
package compilation over gRPC on the port assigned by the agent. It executes
compilation tasks via compile_packages and reports results or failures as
structured PackageReport messages.

The server executes one compilation build at a time to prevent concurrent
requests from racing on process-global Inductor configuration, environment
variables, and kernel knobs.
"""

from concurrent import futures
import argparse
import os
import tempfile
import threading

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
import requests

from . import compile_package, physplan, plan as plan_mod, plan_pb2, vxpu_pb2, vxpu_pb2_grpc

MAX_MESSAGE_BYTES: int = 128 * 1024 * 1024


class CompilerServicer(vxpu_pb2_grpc.CompilerServicer):
    """Serves package compilation requests over gRPC.

    Executes one compilation build at a time to prevent concurrent requests
    from racing on process-global Inductor configuration, environment
    variables, and kernel knobs.
    """

    def __init__(self, modelstore: str, artifact_dir: str, artifact_ref: str) -> None:
        if not modelstore:
            raise ValueError("compiler requires modelstore")
        if not artifact_dir or not artifact_ref:
            raise ValueError("compiler requires artifact_dir and artifact_ref")
        self._lock = threading.Lock()
        self._modelstore = modelstore
        self._artifact_dir = artifact_dir
        self._artifact_ref = artifact_ref

    def Compile(
        self,
        request: vxpu_pb2.CompileRequest,
        context: grpc.ServicerContext,
    ) -> vxpu_pb2.CompileResponse:
        """Compiles requested packages for an artifact under a plan."""
        if not request.model:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model is required")
        if not request.artifact:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "artifact is required")
        if request.artifact != self._artifact_ref:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"artifact {request.artifact!r} does not match staged {self._artifact_ref!r}")
        if not request.HasField("plan"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "plan is required")
        if not request.machine_kind:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "machine_kind is required")
        if not request.software_version:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "software_version is required")
        if not request.device:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "device is required")
        if not request.packages:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no packages specified to compile")

        try:
            plan_given = physplan.apply_env(request.plan)
            plan = plan_mod.Plan(request.device, compile_decode=True, given=plan_given)
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid plan: {e}")

        parallel_builds = request.parallel_builds if request.parallel_builds > 0 else None

        with tempfile.TemporaryDirectory(prefix="vxpu-compile-out-") as out_dir:
            with self._lock:
                try:
                    results, failures = compile_package.compile_packages(
                        artifact_dir=self._artifact_dir,
                        out_dir=out_dir,
                        packages=list(request.packages),
                        plan=plan,
                        machine_kind=request.machine_kind,
                        software_version=request.software_version,
                        device=request.device,
                        parallel_builds=parallel_builds,
                    )
                except ValueError as e:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
                except RuntimeError as e:
                    context.abort(grpc.StatusCode.INTERNAL, str(e))

            from . import modelstore

            present: list[plan_pb2.Package] = []
            repo = modelstore.package_repository(request.model)
            for r in results:
                if not os.path.isfile(r["path"]):
                    failures.append(
                        plan_pb2.CompileFailure(
                            package=plan_pb2.Package(name=r["name"], key=r["key"]),
                            machine_kind=request.machine_kind,
                            software_version=request.software_version,
                            reason=f"compiled package file was not produced at {r['path']}",
                        )
                    )
                    continue
                pkg_dir = os.path.join(out_dir, r["key"])
                os.makedirs(pkg_dir, exist_ok=True)
                os.replace(r["path"], os.path.join(pkg_dir, f"{r['key']}.pt2"))
                try:
                    modelstore.push_directory(
                        base_url=self._modelstore,
                        name=repo,
                        tag=r["key"],
                        directory=pkg_dir,
                        annotations={
                            "vnd.gke-labs.vxpu.package.name": r["name"],
                            "vnd.gke-labs.vxpu.package.key": r["key"],
                        },
                        artifact_type=modelstore.PACKAGE_ARTIFACT_TYPE,
                    )
                except (requests.RequestException, RuntimeError, OSError) as e:
                    context.abort(grpc.StatusCode.INTERNAL, f"storing package {r['name']}: {e}")
                present.append(plan_pb2.Package(name=r["name"], key=r["key"]))

        report = plan_pb2.PackageReport(
            present=present,
            failed=failures,
            machine_kind=request.machine_kind,
        )
        return vxpu_pb2.CompileResponse(report=report)


def create_server(max_workers: int = 10, modelstore: str = "", artifact_dir: str = "", artifact_ref: str = "") -> grpc.Server:
    """Creates a configured gRPC server with the Compiler and Health services."""
    servicer = CompilerServicer(modelstore, artifact_dir, artifact_ref)
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    vxpu_pb2_grpc.add_CompilerServicer_to_server(servicer, server)

    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)  # pyright: ignore[reportUnknownMemberType]

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="vXPU gRPC compiler server")
    parser.add_argument("--host", required=True, help="host address to bind")
    parser.add_argument("--port", type=int, required=True, help="port to listen on")
    parser.add_argument("--modelstore", required=True, help="modelstore base URL")
    parser.add_argument("--artifact", required=True, help="path to staged artifact directory")
    parser.add_argument("--artifact-ref", required=True, help="store reference of the staged artifact")
    args = parser.parse_args()

    server = create_server(modelstore=args.modelstore, artifact_dir=args.artifact, artifact_ref=args.artifact_ref)
    # Connections between coordinator and executor will be authenticated with
    # mTLS under #662. For now, bind insecurely; server credentials will plug into
    # server.add_secure_port(...) once #662 lands.
    bind_addr = f"{args.host}:{args.port}"
    port = server.add_insecure_port(bind_addr)
    if port == 0:
        raise RuntimeError(f"failed to bind compiler server on {bind_addr}")
    server.start()
    print(f"[vxpu] compiler serving on {bind_addr}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
