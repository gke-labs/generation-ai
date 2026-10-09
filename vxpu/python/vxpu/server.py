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

"""vXPU executor server: receives artifacts, serves inference.

LoadModel is idempotent — the artifact's content digest identifies the
model, so a matching request reuses the loaded engine (sub-second)
instead of reloading. An idle reaper evicts the engine after
--keep-alive seconds so bursts amortize one load but idle accelerators
come back.

Usage:
    python -m vxpu.server --device cuda --compile
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from concurrent import futures

import grpc
import torch

from . import vxpu_pb2, vxpu_pb2_grpc
from .engine import Engine

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


class ExecutorServicer(vxpu_pb2_grpc.ExecutorServicer):
    def __init__(self, device="cpu", compile_decode=False,
                 keep_alive_s=300, work_dir="/tmp/vxpu"):
        self.device = device
        self.compile_decode = compile_decode
        self.work_dir = work_dir
        self.engine = None
        self.digest = None
        self.loading = None  # digest currently loading, if any
        self.load_error = None
        self.last_used = time.time()
        self.keep_alive_s = keep_alive_s
        self.lock = threading.Lock()
        threading.Thread(target=self._reap_idle, daemon=True).start()

    def _touch(self):
        self.last_used = time.time()

    def _reap_idle(self):
        while True:
            time.sleep(5)
            with self.lock:
                idle = time.time() - self.last_used
                if self.engine is not None and idle > self.keep_alive_s:
                    print(f"[vxpu] evicting model after {idle:.0f}s idle",
                          flush=True)
                    self.engine = None
                    self.digest = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _load(self, digest, artifact_dir, total_bytes):
        try:
            print(f"[vxpu] loading {digest[:12]} "
                  f"({total_bytes / 1e9:.1f} GB of weights to rehydrate)",
                  flush=True)
            started = time.time()
            engine = Engine(artifact_dir, device=self.device,
                            compile_decode=self.compile_decode)
            with self.lock:
                if self.loading == digest:
                    self.engine = engine
                    self.digest = digest
                    self.loading = None
                    self._touch()  # idle clock starts after the work
            print(f"[vxpu] ready in {time.time() - started:.0f}s "
                  f"({engine.bytes_fetched / 1e9:.1f} GB downloaded)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] load failed: {e}", file=sys.stderr, flush=True)
            with self.lock:
                if self.loading == digest:
                    self.load_error = str(e)
                    self.loading = None

    def LoadModel(self, request, context):
        """Accepts the artifact and loads asynchronously: clients poll
        NewSession, which reports FAILED_PRECONDITION until ready. No
        long-held RPC, so tunnels (port-forward) never sit idle."""
        try:
            digest = hashlib.sha256(
                request.manifest_json.encode()
                + request.binding_json.encode()
                + request.prefill_graph
                + request.decode_graph).hexdigest()
            with self.lock:
                self._touch()
                if self.engine is not None and digest == self.digest:
                    print(f"[vxpu] {digest[:12]} already loaded; reusing",
                          flush=True)
                    return vxpu_pb2.LoadModelResponse(success=True)
                if self.loading == digest:
                    return vxpu_pb2.LoadModelResponse(success=True)

                # Reset currently resident engine so NewSession will wait for this load.
                self.engine = None
                self.digest = None
                self.loading = digest
                self.load_error = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                artifact_dir = os.path.join(self.work_dir, digest[:16])
                os.makedirs(artifact_dir, exist_ok=True)
                for name, data in (
                        ("manifest.json", request.manifest_json.encode()),
                        ("binding.json", request.binding_json.encode()),
                        ("prefill.pt2", request.prefill_graph),
                        ("decode.pt2", request.decode_graph)):
                    with open(os.path.join(artifact_dir, name), "wb") as f:
                        f.write(data)

                manifest = json.loads(request.manifest_json)
                total = sum(f["size"]
                            for f in manifest["files"].values())
                threading.Thread(
                    target=self._load,
                    args=(digest, artifact_dir, total),
                    daemon=True).start()
                return vxpu_pb2.LoadModelResponse(success=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] LoadModel error: {e}", file=sys.stderr,
                  flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.LoadModelResponse(success=False)

    def NewSession(self, request, context):
        with self.lock:
            if self.loading:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("model still loading")
                return vxpu_pb2.NewSessionResponse()
            if self.load_error:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"model load failed: {self.load_error}")
                return vxpu_pb2.NewSessionResponse()
            if self.engine is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.NewSessionResponse()
            try:
                self._touch()
                session_id = self.engine.new_session()
            except Exception as e:  # noqa: BLE001
                print(f"[vxpu] NewSession error: {e}", file=sys.stderr,
                      flush=True)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return vxpu_pb2.NewSessionResponse()

        print(f"[vxpu] new session {session_id}", flush=True)
        return vxpu_pb2.NewSessionResponse(session_id=session_id)

    def Chat(self, request, context):
        with self.lock:
            if self.engine is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.ChatResponse()
            self._touch()
            engine = self.engine

        if request.max_new_tokens <= 0:
            max_new_tokens = 96
        elif request.max_new_tokens > 4096:
            max_new_tokens = 4096
        else:
            max_new_tokens = request.max_new_tokens

        try:
            reply = engine.chat(
                request.session_id, request.text,
                max_new_tokens=max_new_tokens,
                raw_prompt=request.raw_prompt)
            return vxpu_pb2.ChatResponse(
                text=reply["text"],
                session_tokens=reply["session_tokens"],
                new_prompt_tokens=reply["new_prompt_tokens"],
                generated=reply["generated"],
                prefill_ms=float(reply["prefill_ms"]),
                ms_per_token=float(reply["ms_per_token"]))
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return vxpu_pb2.ChatResponse()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] Chat error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.ChatResponse()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compile", action="store_true",
        default=torch.cuda.is_available(),
        help="torch.compile the decode graph per session")
    parser.add_argument("--keep-alive", type=int, default=300,
                        help="seconds to keep an idle model loaded")
    args = parser.parse_args()

    print(f"[vxpu] executor on :{args.port} (device={args.device}, "
          f"compile={args.compile}, keep-alive={args.keep_alive}s, "
          f"threads={torch.get_num_threads()})",
          flush=True)
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        # Clients ping to keep port-forward tunnels alive through
        # long LoadModel calls; accept them.
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10), options=options)
    vxpu_pb2_grpc.add_ExecutorServicer_to_server(
        ExecutorServicer(device=args.device,
                         compile_decode=args.compile,
                         keep_alive_s=args.keep_alive), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
