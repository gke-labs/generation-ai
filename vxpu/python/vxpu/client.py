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

"""Thin Python client for a vXPU router/executor.

No torch, no CUDA: ships an exported artifact directory over gRPC and
chats with the model wherever the router placed it. Mirrors the Go CLI
(``vxpu ask``) for use from notebooks and scripts.

    from vxpu.client import Client
    client = Client("localhost:50051")
    session = client.load_artifact("gemma-4-31b-it/")
    print(session.chat("Is the sky blue?").text)
"""

import os
import re
import subprocess
import time

import grpc

from . import vxpu_pb2, vxpu_pb2_grpc

MAX_MESSAGE_BYTES = 128 * 1024 * 1024
ARTIFACT_FILES = ("manifest.json", "binding.json", "prefill.pt2",
                  "decode.pt2")


class Reply:
    """One turn's result: the text plus the executor's own timings."""

    def __init__(self, response):
        self.text = response.text
        self.session_tokens = response.session_tokens
        self.new_prompt_tokens = response.new_prompt_tokens
        self.generated = response.generated
        self.prefill_ms = response.prefill_ms
        self.ms_per_token = response.ms_per_token

    def __repr__(self):
        return (f"Reply(generated={self.generated}, "
                f"ms_per_token={self.ms_per_token:.1f}, "
                f"text={self.text!r})")

    def _repr_markdown_(self):  # notebooks render this
        return (f"{self.text}\n\n*({self.generated} tokens, "
                f"{self.ms_per_token:.0f} ms/token on the executor, "
                f"prefill {self.prefill_ms:.0f} ms, "
                f"{self.session_tokens} tokens in session)*")


class Session:
    """A conversation: the executor keeps its KV cache between turns."""

    def __init__(self, client, session_id):
        self.client = client
        self.session_id = session_id
        self.last = None  # GenerateStats of the latest generate_ids

    def chat(self, text, max_new_tokens=96, raw_prompt=False, timeout=None):
        """Run one turn.

        With ``raw_prompt=False`` (default) ``text`` is the user's
        message; the executor applies the model's chat template and
        remembers the transcript. With ``raw_prompt=True`` ``text`` is
        the complete rendered prompt (you applied the chat template,
        perhaps with tools); send the whole rendered conversation each
        turn and the executor prefills only what is new.
        """
        try:
            response = self.client.stub.Chat(
                vxpu_pb2.ChatRequest(session_id=self.session_id, text=text,
                                     max_new_tokens=max_new_tokens,
                                     raw_prompt=raw_prompt),
                timeout=timeout or self.client.timeout)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError(
                    "the executor no longer has the model loaded (idle "
                    "eviction or restart); call client.load_artifact(...) "
                    f"again for a fresh session: {e.details()}") from e
            raise
        return Reply(response)

    def generate(self, prompt, max_new_tokens=96, timeout=None):
        """Raw text in, text out: ``prompt`` is already templated."""
        return self.chat(prompt, max_new_tokens=max_new_tokens,
                         raw_prompt=True, timeout=timeout).text

    def generate_ids(self, input_ids, max_new_tokens=None, do_sample=None,
                     temperature=None, top_k=None, top_p=None,
                     repetition_penalty=None, eos_token_id=(), seed=None,
                     timeout=None):
        """transformers' generate() over the wire, streamed.

        ``input_ids`` is the complete prompt as a list of ints (the
        client owns the tokenizer). Yields newly generated ids one at a
        time; after the stream ends, ``self.last`` holds the final
        statistics (finish_reason, prefilled_tokens, timings).
        Parameters left as None follow the model's generation_config,
        exactly as model.generate(input_ids) would.
        """
        request = vxpu_pb2.GenerateRequest(
            session_id=self.session_id,
            input_ids=[int(i) for i in input_ids],
            eos_token_id=[int(i) for i in eos_token_id])
        # Only set what the caller gave: unset fields follow the
        # model's generation_config on the executor (0 is a valid
        # max_new_tokens and a valid seed).
        for name, value in (("max_new_tokens", max_new_tokens),
                            ("do_sample", do_sample),
                            ("temperature", temperature),
                            ("top_k", top_k), ("top_p", top_p),
                            ("repetition_penalty", repetition_penalty),
                            ("seed", seed)):
            if value is not None:
                setattr(request, name, value)
        self.last = None
        try:
            for response in self.client.stub.Generate(
                    request, timeout=timeout or self.client.timeout):
                if response.done:
                    self.last = GenerateStats(response)
                else:
                    for token_id in response.token_ids:
                        yield int(token_id)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError(
                    "the executor no longer has the model loaded (idle "
                    "eviction or restart); call client.load_artifact(...) "
                    f"again for a fresh session: {e.details()}") from e
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise RuntimeError(
                    "the router or executor does not implement Generate; "
                    "rebuild both images from this version of vxpu") from e
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                # Bad sampling parameters or an over-long prompt: the
                # executor's message is the useful part.
                raise ValueError(e.details()) from e
            raise


class GenerateStats:
    """Statistics of one Generate call, from the final stream message."""

    def __init__(self, response):
        self.finish_reason = response.finish_reason
        self.prompt_tokens = response.prompt_tokens
        self.prefilled_tokens = response.prefilled_tokens
        self.generated = response.generated
        self.prefill_ms = response.prefill_ms
        self.ms_per_token = response.ms_per_token

    def __repr__(self):
        return (f"GenerateStats(finish_reason={self.finish_reason!r}, "
                f"prompt_tokens={self.prompt_tokens}, "
                f"prefilled_tokens={self.prefilled_tokens}, "
                f"generated={self.generated}, "
                f"prefill_ms={self.prefill_ms:.0f}, "
                f"ms_per_token={self.ms_per_token:.1f})")


class Client:
    def __init__(self, address="localhost:50051", timeout=1800):
        self.address = address
        self.timeout = timeout
        self.channel = grpc.insecure_channel(address, options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            # Keep port-forward tunnels alive through long LoadModel
            # calls; both router and executor permit this cadence.
            ("grpc.keepalive_time_ms", 60000),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", 1),
        ])
        self.stub = vxpu_pb2_grpc.ExecutorStub(self.channel)

    def load_artifact(self, artifact_dir, poll_interval=15, progress=print):
        """Ship an artifact and wait until the model is ready.

        LoadModel returns once the router has placed the executor and
        handed it the artifact; the executor then rehydrates weights
        asynchronously. NewSession is the readiness poll (it reports
        FAILED_PRECONDITION while loading). Returns a ready Session.
        """
        files = {}
        for name in ARTIFACT_FILES:
            with open(os.path.join(artifact_dir, name), "rb") as f:
                files[name] = f.read()
        graph_mb = (len(files["prefill.pt2"]) + len(files["decode.pt2"])) / 1e6
        progress(f"shipping artifact {artifact_dir} ({graph_mb:.0f} MB of "
                 "graphs; weights stay remote)")
        started = time.time()
        self.stub.LoadModel(vxpu_pb2.LoadModelRequest(
            manifest_json=files["manifest.json"].decode(),
            binding_json=files["binding.json"].decode(),
            prefill_graph=files["prefill.pt2"],
            decode_graph=files["decode.pt2"],
        ), timeout=self.timeout, wait_for_ready=True)
        progress(f"executor placed and artifact delivered "
                 f"({time.time() - started:.0f}s); rehydrating weights...")

        deadline = started + self.timeout
        while True:
            try:
                session = self.new_session()
                break
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.FAILED_PRECONDITION:
                    raise
                if time.time() > deadline:
                    raise TimeoutError(
                        f"model not ready after {self.timeout}s: "
                        f"{e.details()}") from e
            progress(f"  loading... ({time.time() - started:.0f}s)")
            time.sleep(poll_interval)
        progress(f"model ready in {time.time() - started:.0f}s")
        return session

    def new_session(self):
        response = self.stub.NewSession(vxpu_pb2.NewSessionRequest(),
                                        timeout=60)
        return Session(self, response.session_id)

    def close(self):
        self.channel.close()


class PortForward:
    """``kubectl port-forward`` to the router, as a context manager.

    For notebooks outside the cluster. Inside the cluster, connect to
    the router Service directly (``vxpu-router:50051``).
    """

    def __init__(self, target="pod/vxpu-router", remote_port=50051,
                 namespace=None):
        self.target = target
        self.remote_port = remote_port
        self.namespace = namespace
        self.process = None
        self.address = None

    def start(self):
        cmd = ["kubectl", "port-forward", self.target,
               f":{self.remote_port}"]
        if self.namespace:
            cmd += ["-n", self.namespace]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        pattern = re.compile(
            r"Forwarding from (?:127\.0\.0\.1|\[::1\]):(\d+)")
        for line in self.process.stdout:
            m = pattern.search(line)
            if m:
                self.address = f"127.0.0.1:{m.group(1)}"
                return self.address
            if self.process.poll() is not None:
                break
        raise RuntimeError(f"port-forward to {self.target} did not start")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
