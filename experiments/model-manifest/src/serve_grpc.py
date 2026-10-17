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

"""gRPC model executor server.

Receives model manifest, bindings, and weightless graphs, writes them to disk,
rehydrates them, and serves chat execution requests.
"""

import argparse
import gc
import hashlib
import sys
import threading
import time
from concurrent import futures

import grpc
import torch

import router_pb2
import router_pb2_grpc
from engine import Engine


class ExecutorServicer(router_pb2_grpc.ModelExecutorServicer):
    def __init__(self, device="cpu", compile_decode=False, keep_alive_s=60):
        self.device = device
        self.compile_decode = compile_decode
        self.engine = None
        self.digest = None
        self.last_used = time.time()
        self.keep_alive_s = keep_alive_s
        self.lock = threading.Lock()
        reaper = threading.Thread(target=self._reap_idle, daemon=True)
        reaper.start()

    def _touch(self):
        self.last_used = time.time()

    def _reap_idle(self):
        """Evict the engine after keep_alive_s without a request, so a
        burst of asks amortizes one load but idle GPUs come back."""
        while True:
            time.sleep(5)
            with self.lock:
                idle = time.time() - self.last_used
                if self.engine is not None and idle > self.keep_alive_s:
                    print(f"[executor] Evicting model after {idle:.0f}s idle",
                          flush=True)
                    self.engine = None
                    self.digest = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def LoadModel(self, request, context):
        try:
            digest = hashlib.sha256(
                request.manifest_json.encode()
                + request.binding_json.encode()
                + request.prefill_graph
                + request.decode_graph
            ).hexdigest()
            with self.lock:
                self._touch()
                if self.engine is not None and digest == self.digest:
                    print(f"[executor] Model {digest[:12]} already loaded "
                          "(content match); reusing", flush=True)
                    return router_pb2.LoadModelResponse(success=True)

                print("[executor] Loading model...", flush=True)
                with open("manifest.json", "w") as f:
                    f.write(request.manifest_json)
                with open("cached.binding.json", "w") as f:
                    f.write(request.binding_json)
                with open("prefill.pt2", "wb") as f:
                    f.write(request.prefill_graph)
                with open("decode.pt2", "wb") as f:
                    f.write(request.decode_graph)

                print("[executor] Initializing engine...", flush=True)
                self.engine = Engine(
                    "manifest.json",
                    binding_path="cached.binding.json",
                    device=self.device,
                    compile_decode=self.compile_decode
                )
                self.digest = digest
                # Touch again now: loading may exceed keep_alive_s, and
                # the idle clock must start after the work, not before.
                self._touch()
                print("[executor] Engine successfully initialized!", flush=True)
                return router_pb2.LoadModelResponse(success=True)
        except Exception as e:
            print(f"[executor] Error loading model: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return router_pb2.LoadModelResponse(success=False)

    def NewSession(self, request, context):
        if self.engine is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Model has not been loaded yet. Call LoadModel first.")
            return router_pb2.NewSessionResponse()
        try:
            self._touch()
            session_id = self.engine.new_session()
            print(f"[executor] Created new session: {session_id}", flush=True)
            return router_pb2.NewSessionResponse(session_id=session_id)
        except Exception as e:
            print(f"[executor] Error creating session: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return router_pb2.NewSessionResponse()

    def Chat(self, request, context):
        if self.engine is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Model has not been loaded yet. Call LoadModel first.")
            return router_pb2.ChatResponse()
        try:
            self._touch()
            print(f"[executor] Chat for session: {request.session_id}", flush=True)
            reply = self.engine.chat(
                request.session_id,
                request.text,
                max_new_tokens=request.max_new_tokens or 96
            )
            return router_pb2.ChatResponse(
                text=reply["text"],
                session_tokens=reply["session_tokens"],
                new_prompt_tokens=reply["new_prompt_tokens"],
                generated=reply["generated"],
                prefill_ms=float(reply["prefill_ms"]),
                ms_per_token=float(reply["ms_per_token"])
            )
        except Exception as e:
            print(f"[executor] Error in chat: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return router_pb2.ChatResponse()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--keep-alive", type=int, default=60,
                        help="seconds to keep an idle model loaded")
    args = parser.parse_args()

    print(f"[executor] Starting gRPC server on port {args.port} (device: {args.device}, compile: {args.compile})", flush=True)

    max_msg_size = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", max_msg_size),
        ("grpc.max_send_message_length", max_msg_size)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    router_pb2_grpc.add_ModelExecutorServicer_to_server(
        ExecutorServicer(device=args.device, compile_decode=args.compile,
                         keep_alive_s=args.keep_alive),
        server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
