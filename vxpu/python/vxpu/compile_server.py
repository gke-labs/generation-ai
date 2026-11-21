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
import threading

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from . import compile_package, physplan, plan as plan_mod, plan_pb2, vxpu_pb2, vxpu_pb2_grpc

MAX_MESSAGE_BYTES: int = 128 * 1024 * 1024


class CompilerServicer(vxpu_pb2_grpc.CompilerServicer):
    """Serves package compilation requests over gRPC.

    Executes one compilation build at a time to prevent concurrent requests
    from racing on process-global Inductor configuration, environment
    variables, and kernel knobs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def Compile(
        self,
        request: vxpu_pb2.CompileRequest,
        context: grpc.ServicerContext,
    ) -> vxpu_pb2.CompileResponse:
        """Compiles requested packages for an artifact under a plan."""
        if not request.artifact:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "artifact is required")
        if not request.plan_file:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "plan_file is required")
        if not request.out_dir:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "out_dir is required")
        if not request.machine_kind:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "machine_kind is required")
        if not request.software_version:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "software_version is required")
        if not request.device:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "device is required")
        if not request.packages:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no packages specified to compile")

        try:
            pf = physplan.read_plan_file(request.plan_file)
            plan_given = physplan.apply_env(pf)
            plan = plan_mod.Plan(request.device, compile_decode=True, given=plan_given)
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid plan file: {e}")

        parallel_builds = request.parallel_builds if request.parallel_builds > 0 else None

        with self._lock:
            try:
                results, failures = compile_package.compile_packages(
                    artifact_dir=request.artifact,
                    out_dir=request.out_dir,
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

        report = plan_pb2.PackageReport(
            present=[plan_pb2.Package(name=r["name"], key=r["key"]) for r in results],
            failed=failures,
            machine_kind=request.machine_kind,
        )
        return vxpu_pb2.CompileResponse(report=report)


def create_server(max_workers: int = 10) -> grpc.Server:
    """Creates a configured gRPC server with the Compiler and Health services."""
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    vxpu_pb2_grpc.add_CompilerServicer_to_server(CompilerServicer(), server)

    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)  # pyright: ignore[reportUnknownMemberType]

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="vXPU gRPC compiler server")
    parser.add_argument("--host", required=True, help="host address to bind")
    parser.add_argument("--port", type=int, required=True, help="port to listen on")
    args = parser.parse_args()

    server = create_server()
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
