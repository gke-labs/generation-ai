# pyright: strict
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

"""Tests for prompt session release on the vXPU executor."""

from collections.abc import Iterator
from concurrent import futures
import shutil
import tempfile
import threading
import time
from typing import Protocol, cast
import unittest
import weakref

import grpc
import torch

from . import engine
from . import vxpu_pb2
from . import vxpu_pb2_grpc
from .server import ExecutorServicer


class _ExecutorStub(Protocol):
    def NewSession(
        self, request: vxpu_pb2.NewSessionRequest
    ) -> vxpu_pb2.NewSessionResponse: ...

    def ReleaseSession(
        self, request: vxpu_pb2.ReleaseSessionRequest
    ) -> vxpu_pb2.ReleaseSessionResponse: ...

    def Execute(
        self, request: vxpu_pb2.ExecuteRequest
    ) -> Iterator[vxpu_pb2.ExecuteResponse]: ...


def _first_state_tensor_ref(eng: engine.Engine, session_id: str) -> weakref.ref[torch.Tensor]:
    state_map = cast(dict[str, torch.Tensor], eng.sessions[session_id]["state"])
    first_tensor = next(iter(state_map.values()))
    return weakref.ref(first_tensor)


class SessionReleaseTest(unittest.TestCase):
    _artifact_dir: str = ""
    _engine: engine.Engine | None = None
    _init_skip: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from .fixtures import tiny_deepseek_v41

            cls._artifact_dir = tiny_deepseek_v41.get_artifact_dir(max_cache_len=128)
            cls._engine = engine.Engine(cls._artifact_dir, device="cpu", compile_decode=False)
        except Exception as e:
            cls._init_skip = f"tiny_deepseek_v41 fixture unavailable: {e}"

    def setUp(self) -> None:
        if self._init_skip is not None:
            self.skipTest(self._init_skip)

    def _serve(self, eng: engine.Engine) -> _ExecutorStub:
        td = tempfile.mkdtemp(prefix="vxpu-release-test-")
        self.addCleanup(shutil.rmtree, td, True)

        servicer = ExecutorServicer(work_dir=td, device="cpu", compile_decode=False)
        servicer.engine = eng

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        vxpu_pb2_grpc.add_ExecutorServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(server.stop, 1)

        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(channel.close)
        return cast(_ExecutorStub, vxpu_pb2_grpc.ExecutorStub(channel))

    def _shared_executor(self) -> tuple[engine.Engine, _ExecutorStub]:
        eng = self._engine
        assert eng is not None
        for sid in list(eng.sessions):
            eng.release_session(sid)
        return eng, self._serve(eng)

    def test_release_prompt_during_concurrent_generation(self) -> None:
        """Releasing a finished session returns promptly while other streams generate."""
        eng, stub = self._shared_executor()
        num_streams = 4
        tokens_per_stream = 50
        prompt_ids = [1, 2, 3, 4]

        session_ids = [
            stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
            for _ in range(num_streams)
        ]
        self.assertEqual(len(eng.sessions), num_streams)

        all_started = threading.Barrier(num_streams)
        state_lock = threading.Lock()
        active_streams = num_streams
        results: list[tuple[int, float, int, int]] = []
        errors: list[BaseException] = []

        def run_stream(idx: int, sid: str) -> None:
            nonlocal active_streams
            try:
                req = vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=sid,
                        input_ids=prompt_ids,
                        max_new_tokens=tokens_per_stream,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
                all_started.wait(timeout=10.0)
                stream = stub.Execute(req)
                generated = 0
                for msg in stream:
                    resp = msg.response
                    if not resp.done:
                        generated += len(resp.token_ids)

                with state_lock:
                    active_streams -= 1
                    still_active_before = active_streams

                t0 = time.perf_counter()
                stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=sid))
                release_s = time.perf_counter() - t0

                with state_lock:
                    still_active_after = active_streams
                    results.append((idx, release_s, generated, min(still_active_before, still_active_after)))
            except BaseException as exc:  # noqa: BLE001
                with state_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=run_stream, args=(i, sid))
            for i, sid in enumerate(session_ids)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120.0)
            self.assertFalse(t.is_alive(), "stream thread did not finish in time")

        self.assertEqual(errors, [], f"unexpected errors: {errors}")
        self.assertEqual(len(results), num_streams)

        overlapping_releases = [r for r in results if r[3] > 0]
        self.assertGreater(
            len(overlapping_releases),
            0,
            "expected at least one finished session to release while other streams remained active",
        )
        for idx, release_s, generated, others_active in results:
            self.assertEqual(generated, tokens_per_stream, f"stream {idx} generated wrong token count")
            if others_active > 0:
                self.assertLess(
                    release_s,
                    0.5,
                    f"stream {idx} release waited {release_s:.3f}s while {others_active} stream(s) were active",
                )

        self.assertEqual(len(eng.sessions), 0)
        self.assertEqual(eng.session_bytes(), 0)

    def test_release_frees_memory_and_prevents_reuse(self) -> None:
        """Releasing a session frees its tensors, lowers reported memory, and prevents reuse."""
        eng, stub = self._shared_executor()

        s0 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        s1 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        self.assertEqual(len(eng.sessions), 2)
        total_bytes = eng.session_bytes()
        self.assertGreater(total_bytes, 0)
        per_session_bytes = total_bytes // 2

        tensor_ref = _first_state_tensor_ref(eng, s0)

        # Generate on s0 so the engine binds s0 to its graphs and _current.
        chunks = list(
            stub.Execute(
                vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=s0,
                        input_ids=[1, 2, 3],
                        max_new_tokens=3,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
            )
        )
        self.assertTrue(any(c.response.done for c in chunks))

        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s0))
        self.assertEqual(len(eng.sessions), 1)
        self.assertEqual(eng.session_bytes(), per_session_bytes)
        self.assertIsNone(tensor_ref(), "released session state tensor is still strongly referenced")

        # Releasing twice or releasing an unknown session is not an error.
        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s0))
        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id="unknown-session"))

        # Generating on the released session must fail.
        with self.assertRaises(grpc.RpcError) as cm:
            list(
                stub.Execute(
                    vxpu_pb2.ExecuteRequest(
                        request=vxpu_pb2.GenerateRequest(
                            session_id=s0,
                            input_ids=[1, 2, 3],
                            max_new_tokens=2,
                            do_sample=False,
                        )
                    )
                )
            )
        self.assertEqual(cm.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

        # The remaining session continues to generate normally.
        s1_chunks = list(
            stub.Execute(
                vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=s1,
                        input_ids=[1, 2, 3],
                        max_new_tokens=3,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
            )
        )
        self.assertTrue(any(c.response.done for c in s1_chunks))

        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s1))
        self.assertEqual(len(eng.sessions), 0)
        self.assertEqual(eng.session_bytes(), 0)

    def test_release_mid_generation_stops_and_frees(self) -> None:
        """Releasing an actively generating session stops its stream and frees its state."""
        eng, stub = self._shared_executor()

        s0 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        tensor_ref = _first_state_tensor_ref(eng, s0)

        first_token = threading.Event()
        max_tokens = 100
        received: list[int] = []
        stream_errors: list[BaseException] = []

        def run_long_stream() -> None:
            try:
                stream = stub.Execute(
                    vxpu_pb2.ExecuteRequest(
                        request=vxpu_pb2.GenerateRequest(
                            session_id=s0,
                            input_ids=[1, 2, 3, 4],
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            ignore_eos=True,
                        )
                    )
                )
                for msg in stream:
                    resp = msg.response
                    if not resp.done:
                        received.extend(int(t) for t in resp.token_ids)
                        first_token.set()
            except BaseException as exc:  # noqa: BLE001
                stream_errors.append(exc)

        t = threading.Thread(target=run_long_stream)
        t.start()
        self.assertTrue(first_token.wait(timeout=30.0), "timed out waiting for first token")

        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s0))
        t.join(timeout=30.0)
        self.assertFalse(t.is_alive(), "in-flight generation did not stop after release")
        self.assertEqual(stream_errors, [], f"unexpected stream error: {stream_errors}")
        self.assertGreater(len(received), 0)
        self.assertLess(len(received), max_tokens, "generation should have stopped before max_tokens")
        self.assertEqual(len(eng.sessions), 0)
        self.assertEqual(eng.session_bytes(), 0)
        self.assertIsNone(tensor_ref(), "state tensor still referenced after mid-generation release")

        # Subsequent sessions generate normally after an interrupted session.
        s1 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        s1_chunks = list(
            stub.Execute(
                vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=s1,
                        input_ids=[1, 2, 3],
                        max_new_tokens=4,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
            )
        )
        self.assertTrue(any(c.response.done for c in s1_chunks))
        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s1))
        self.assertEqual(len(eng.sessions), 0)

    def test_release_resident_mode_frees_tensors(self) -> None:
        """In resident-state mode, releasing the last active session frees its tensors without copy-back."""
        eng = engine.Engine(
            self._artifact_dir,
            device="cpu",
            compile_decode=False,
            plan_given={"VXPU_DATAFLOW": "1"},
        )
        self.assertIsNotNone(eng._resident)  # pyright: ignore[reportPrivateUsage]
        stub = self._serve(eng)

        s0 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        s1 = stub.NewSession(vxpu_pb2.NewSessionRequest()).session_id
        tensor_ref = _first_state_tensor_ref(eng, s0)

        list(
            stub.Execute(
                vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=s0,
                        input_ids=[1, 2, 3],
                        max_new_tokens=3,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
            )
        )
        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s0))
        self.assertEqual(len(eng.sessions), 1)
        self.assertIsNone(tensor_ref(), "resident owner still retained released session state tensor")

        s1_chunks = list(
            stub.Execute(
                vxpu_pb2.ExecuteRequest(
                    request=vxpu_pb2.GenerateRequest(
                        session_id=s1,
                        input_ids=[1, 2, 3],
                        max_new_tokens=3,
                        do_sample=False,
                        ignore_eos=True,
                    )
                )
            )
        )
        self.assertTrue(any(c.response.done for c in s1_chunks))
        stub.ReleaseSession(vxpu_pb2.ReleaseSessionRequest(session_id=s1))
        self.assertEqual(len(eng.sessions), 0)
        self.assertEqual(eng.session_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
