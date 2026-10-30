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
from .engine import Engine, SamplingParams
from .plan import Plan, probe_hardware

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


class ExecutorServicer(vxpu_pb2_grpc.ExecutorServicer):
    def __init__(self, device="cpu", compile_decode=False,
                 keep_alive_s=3600, work_dir="/tmp/vxpu"):
        self.device = device
        self.compile_decode = compile_decode
        self.work_dir = work_dir
        self.engine = None
        self.digest = None
        self.loading = None  # digest currently loading, if any
        self.load_error = None
        self.bytes_rehydrated = 0
        self.last_used = time.time()
        self.keep_alive_s = keep_alive_s
        self.lock = threading.Lock()
        self.node_server = None  # ((artifact digest, nodes), the NodeServer) this executor serves (ServeNodes)
        self.serve_lock = threading.Lock()  # one ServeNodes at a time, start to finish
        self.range_server = None  # (the engine, its RangeServer): the loaded engine's range of layers (ServeRange)
        threading.Thread(target=self._reap_idle, daemon=True).start()

    def _touch(self):
        self.last_used = time.time()

    def _load_state(self):
        """The current load state of the executor. The caller holds self.lock."""
        if self.engine is not None:
            return vxpu_pb2.LOAD_STATE_READY
        if self.loading:
            return vxpu_pb2.LOAD_STATE_LOADING
        if self.load_error:
            return vxpu_pb2.LOAD_STATE_FAILED
        return vxpu_pb2.LOAD_STATE_NONE

    def _drop_engine(self):
        """Drop the loaded engine, and stop the range server serving it
        (ServeRange), if any. The caller holds self.lock."""
        if self.range_server is not None:
            self.range_server[1].stop()
            self.range_server = None
        self.engine = None
        self.digest = None
        self.load_error = None
        self.bytes_rehydrated = 0

    def _reap_idle(self):
        range_messages = 0
        while True:
            time.sleep(5)
            with self.lock:
                if self.range_server is not None:  # steps another engine sends this one's range count as use
                    n = self.range_server[1].messages()
                    if n != range_messages:
                        range_messages = n
                        self._touch()
                idle = time.time() - self.last_used
                if self.engine is not None and idle > self.keep_alive_s:
                    print(f"[vxpu] evicting model after {idle:.0f}s idle",
                          flush=True)
                    self._drop_engine()
                    range_messages = 0
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _load(self, digest, artifact_dir, total_bytes, plan_given):
        try:
            print(f"[vxpu] loading {digest[:12]} "
                  f"({total_bytes / 1e9:.1f} GB of weights to rehydrate)",
                  flush=True)
            started = time.time()

            def on_progress(bytes_rehydrated):
                with self.lock:
                    if self.loading == digest:
                        self.bytes_rehydrated = bytes_rehydrated

            engine = Engine(artifact_dir, device=self.device,
                            compile_decode=self.compile_decode,
                            plan_given=plan_given,
                            progress_callback=on_progress)
            with self.lock:
                if self.loading == digest:
                    self.engine = engine
                    self.digest = digest
                    self.loading = None
                    self.bytes_rehydrated = engine.bytes_fetched
                    self._touch()  # idle clock starts after the work
            print(f"[vxpu] ready in {time.time() - started:.0f}s "
                  f"({engine.load_breakdown}; "
                  f"{engine.bytes_fetched / 1e9:.1f} GB downloaded)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] load failed: {e}", file=sys.stderr, flush=True)
            with self.lock:
                if self.loading == digest:
                    self.load_error = str(e)
                    self.loading = None

    def LoadModel(self, request, context):
        """Accepts the artifact and loads asynchronously. Callers check
        GetLoad or poll NewSession until the model is ready. No long-held RPC,
        so tunnels (port-forward) never sit idle."""
        try:
            # the coordinator's plan is part of what is loaded: the same
            # artifact under another plan is another engine
            plan_given = dict(request.plan)
            try:  # a plan this executor refuses fails the call, not the load behind it
                Plan(self.device, compile_decode=self.compile_decode, given=plan_given)
            except ValueError as e:  # ConfigError, and the plain ValueErrors of a switch's own checks
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"plan refused: {e}")
                return vxpu_pb2.LoadModelResponse(success=False)
            # the files are the artifact's alone; the engine is the artifact under its plan
            artifact_digest = hashlib.sha256(
                request.manifest_json.encode()
                + request.binding_json.encode()
                + request.prefill_graph
                + request.decode_graph).hexdigest()
            digest = hashlib.sha256((artifact_digest + json.dumps(plan_given, sort_keys=True)).encode()).hexdigest()
            with self.lock:
                self._touch()
                if self.engine is not None and digest == self.digest:
                    print(f"[vxpu] {digest[:12]} already loaded; reusing",
                          flush=True)
                    return vxpu_pb2.LoadModelResponse(success=True)
                if self.loading == digest:
                    return vxpu_pb2.LoadModelResponse(success=True)

                # Reset currently resident engine so NewSession will wait for this load.
                self._drop_engine()
                self.loading = digest
                self.load_error = None
                self.bytes_rehydrated = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                artifact_dir = os.path.join(self.work_dir, artifact_digest[:16])
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
                    args=(digest, artifact_dir, total, plan_given),
                    daemon=True).start()
                return vxpu_pb2.LoadModelResponse(success=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] LoadModel error: {e}", file=sys.stderr,
                  flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.LoadModelResponse(success=False)

    def _artifact_dir(self, request):
        """The artifact a ServeNodes or DescribeNodes request carries, in
        the work directory (its manifest, binding and decode graph)."""
        digest = hashlib.sha256(request.manifest_json.encode() + request.binding_json.encode()
                                + request.decode_graph).hexdigest()
        artifact_dir = os.path.join(self.work_dir, "nodes-" + digest[:16])
        os.makedirs(artifact_dir, exist_ok=True)
        for name, data in (("manifest.json", request.manifest_json.encode()),
                           ("binding.json", request.binding_json.encode()), ("decode.pt2", request.decode_graph)):
            with open(os.path.join(artifact_dir, name), "wb") as f:
                f.write(data)
        return artifact_dir, digest

    def DescribeNodes(self, request, context):
        """Describe the artifact's decode step as nodes, for the coordinator
        (#394). If the request includes the prefill graph, also say where the
        step can be cut into ranges of whole layers (cuts.py, #387)."""
        from . import cuts, nodes
        try:
            artifact_dir, _ = self._artifact_dir(request)
            described = nodes.describe(nodes.decode_split(artifact_dir), json.loads(request.manifest_json),
                                       json.loads(request.binding_json))
            cuts_json = ""
            if request.prefill_graph:
                with open(os.path.join(artifact_dir, "prefill.pt2"), "wb") as f:
                    f.write(request.prefill_graph)
                cuts_json = json.dumps(cuts.describe(artifact_dir))
            return vxpu_pb2.DescribeNodesResponse(nodes_json=json.dumps(described), cuts_json=cuts_json)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] DescribeNodes error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.DescribeNodesResponse()

    def ServeNodes(self, request, context):
        """Serve some of an artifact's decode-step nodes natively on a port
        of this machine (nodeserver.py): how the coordinator spreads a
        model over executors (#394). The nodes' residents are loaded and
        their packages built (or found cached) before it answers. One set
        at a time: a new one replaces the old, which stops first, so the
        two sets' residents never need to fit together. A second call for
        the same set waits for the first and gets its port. Like the
        executor itself, no authentication: the executors' network is
        private."""
        from . import nodeserver
        try:
            artifact_dir, digest = self._artifact_dir(request)
            key = (digest, tuple(request.nodes))
            with self.serve_lock:
                with self.lock:
                    current = self.node_server
                    if current is not None and current[0] == key:
                        return vxpu_pb2.ServeNodesResponse(port=current[1].port())
                    self.node_server = None
                if current is not None:
                    current[1].stop()
                server = nodeserver.start(artifact_dir, list(request.nodes))
                if not context.is_active():  # the caller gave up: no engine will be wired to these
                    server.stop()
                    context.set_code(grpc.StatusCode.CANCELLED)
                    context.set_details("the caller went away while the nodes loaded")
                    return vxpu_pb2.ServeNodesResponse()
                with self.lock:
                    self.node_server = (key, server)
            return vxpu_pb2.ServeNodesResponse(port=server.port())
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return vxpu_pb2.ServeNodesResponse()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] ServeNodes error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.ServeNodesResponse()

    def ServeRange(self, request, context):
        """Serve the loaded engine's range of whole layers (set by its plan's
        VXPU_LAYER_RANGE) on a port of this machine, for the engine that runs
        the layers before it (rangeserver.py, #387). There is one server per
        loaded engine. Answers with the load state in its response while the
        engine is still loading."""
        from . import rangeserver
        with self.lock:
            state = self._load_state()
            if state == vxpu_pb2.LOAD_STATE_LOADING:
                return vxpu_pb2.ServeRangeResponse(state=vxpu_pb2.LOAD_STATE_LOADING)
            if state == vxpu_pb2.LOAD_STATE_FAILED:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"model load failed: {self.load_error}")
                return vxpu_pb2.ServeRangeResponse()
            if state == vxpu_pb2.LOAD_STATE_NONE:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.ServeRangeResponse()
            engine = self.engine
            rng = engine.plan["layer_range"]
            if rng is None or rng[0] == 0:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("the loaded engine does not run a range after the first (VXPU_LAYER_RANGE=START:END, START > 0)")
                return vxpu_pb2.ServeRangeResponse()
            if self.range_server is None or self.range_server[0] is not engine:
                if self.range_server is not None:
                    self.range_server[1].stop()
                self.range_server = (engine, rangeserver.serve(engine))
            text = f"{rng[0]}:" + ("" if rng[1] is None else str(rng[1]))
            return vxpu_pb2.ServeRangeResponse(port=self.range_server[1].port(), layer_range=text,
                                                state=vxpu_pb2.LOAD_STATE_READY)

    def GetLoad(self, request, context):
        """Report the state of the model load on this executor: whether a
        model is loading, ready to serve, or failed to load."""
        with self.lock:
            return vxpu_pb2.GetLoadResponse(
                state=self._load_state(),
                error=self.load_error or "",
                bytes_rehydrated=self.bytes_rehydrated,
            )

    def Describe(self, request, context):
        """What this executor runs on, for the coordinator's machine
        database (#394): the same probe its plans record."""
        return vxpu_pb2.DescribeResponse(hardware_json=json.dumps(probe_hardware(self.device)))

    def NewSession(self, request, context):
        with self.lock:
            state = self._load_state()
            if state == vxpu_pb2.LOAD_STATE_LOADING:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("model still loading")
                return vxpu_pb2.NewSessionResponse()
            if state == vxpu_pb2.LOAD_STATE_FAILED:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"model load failed: {self.load_error}")
                return vxpu_pb2.NewSessionResponse()
            if state == vxpu_pb2.LOAD_STATE_NONE:
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

    def ExecuteChat(self, request, context):
        with self.lock:
            if self.engine is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("call LoadModel first")
                return vxpu_pb2.ExecuteChatResponse()
            self._touch()
            engine = self.engine

        if not request.HasField("request"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("missing request in ExecuteChatRequest")
            return vxpu_pb2.ExecuteChatResponse()
        req = request.request

        if req.max_new_tokens <= 0:
            max_new_tokens = 96
        elif req.max_new_tokens > 4096:
            max_new_tokens = 4096
        else:
            max_new_tokens = req.max_new_tokens

        try:
            reply = engine.chat(
                req.session_id, req.text,
                max_new_tokens=max_new_tokens,
                raw_prompt=req.raw_prompt,
                telemetry=request.telemetry)
            chat_resp = vxpu_pb2.ChatResponse(
                text=reply["text"],
                session_tokens=reply["session_tokens"],
                new_prompt_tokens=reply["new_prompt_tokens"],
                generated=reply["generated"],
                prefill_ms=float(reply["prefill_ms"]),
                ms_per_token=float(reply["ms_per_token"]))
            return vxpu_pb2.ExecuteChatResponse(
                response=chat_resp,
                telemetry_json=json.dumps(reply["telemetry"]) if "telemetry" in reply else "")
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return vxpu_pb2.ExecuteChatResponse()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] ExecuteChat error: {e}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vxpu_pb2.ExecuteChatResponse()

    def Execute(self, request, context):
        with self.lock:
            if self.engine is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              "call LoadModel first")
            self._touch()
            engine = self.engine

        if not request.HasField("request"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "missing request in ExecuteRequest")
        req = request.request

        if req.HasField("max_new_tokens"):
            max_new_tokens = max(0, req.max_new_tokens)
        else:
            max_new_tokens = 20  # transformers' max_length default
        max_new_tokens = min(max_new_tokens, 4096)
        params = SamplingParams(
            do_sample=(req.do_sample
                       if req.HasField("do_sample") else None),
            temperature=(req.temperature
                         if req.HasField("temperature") else None),
            top_k=req.top_k if req.HasField("top_k") else None,
            top_p=req.top_p if req.HasField("top_p") else None,
            repetition_penalty=(req.repetition_penalty
                                if req.HasField("repetition_penalty")
                                else None),
            seed=req.seed if req.HasField("seed") else None)
        try:
            for chunk in engine.generate(
                    req.session_id, list(req.input_ids),
                    max_new_tokens, params, list(req.eos_token_id),
                    telemetry=request.telemetry):
                if not context.is_active():
                    return
                self._touch()  # a long stream is not idle
                if chunk["done"]:
                    gen_resp = vxpu_pb2.GenerateResponse(
                        done=True,
                        finish_reason=chunk["finish_reason"],
                        prompt_tokens=chunk["prompt_tokens"],
                        prefilled_tokens=chunk["prefilled_tokens"],
                        generated=chunk["generated"],
                        prefill_ms=float(chunk["prefill_ms"]),
                        ms_per_token=float(chunk["ms_per_token"]))
                    yield vxpu_pb2.ExecuteResponse(
                        response=gen_resp,
                        telemetry_json=(json.dumps(chunk["telemetry"])
                                        if chunk.get("telemetry") else ""))
                else:
                    gen_resp = vxpu_pb2.GenerateResponse(
                        token_ids=chunk["token_ids"])
                    yield vxpu_pb2.ExecuteResponse(response=gen_resp)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] Execute error: {e}", file=sys.stderr, flush=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))


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
    parser.add_argument("--keep-alive", type=int, default=3600,
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
