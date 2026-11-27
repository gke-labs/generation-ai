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

"""Unit tests for memory-mapped weight helpers and HTTP range retries in rehydrate.py."""

import http.server
import mmap
import os
import sys
import tempfile
import threading
import time
from typing import Any
import unittest

import torch

from . import rehydrate


class RehydrateTest(unittest.TestCase):
    """Verifies memory-mapping advice, page-cache release, and HTTP range retry behavior."""

    def test_advise_random(self) -> None:
        """advise_random applies MADV_RANDOM to a memory-mapped tensor when supported."""
        if getattr(mmap, "MADV_RANDOM", None) is None:
            self.skipTest("MADV_RANDOM unavailable on this platform")
        file_backed_fn: Any = getattr(rehydrate, "file_backed")
        advise_random_fn: Any = getattr(rehydrate, "advise_random")
        with tempfile.TemporaryDirectory(prefix="vxpu-advise-") as workdir:
            path = os.path.join(workdir, "advise-random.bin")
            mapped, *_ = file_backed_fn(path, 1 << 16)
            self.assertTrue(advise_random_fn(mapped))

    def test_page_cache_release(self) -> None:
        """release_mapping drops resident pages of a memory-mapped Parameter from page cache."""
        madv_pageout = getattr(mmap, "MADV_PAGEOUT", 21 if sys.platform.startswith("linux") else None)
        if madv_pageout is None:
            self.skipTest("MADV_PAGEOUT unavailable on this platform")

        file_backed_fn: Any = getattr(rehydrate, "file_backed")
        release_mapping_fn: Any = getattr(rehydrate, "release_mapping")
        resident_bytes_fn: Any = getattr(rehydrate, "resident_bytes")

        with tempfile.TemporaryDirectory(prefix="vxpu-pagecache-") as workdir:
            path = os.path.join(workdir, "page-cache-release.bin")
            length = 64 * 1024
            with open(path, "wb") as f:
                f.write(b"\x01" * length)
                f.flush()
                os.fsync(f.fileno())

            mapped, _fresh, commit = file_backed_fn(path, length)
            commit()
            # Read every byte to fault all pages into memory.
            _ = bytes(mapped.numpy())

            before = resident_bytes_fn(mapped)
            if before is None:
                self.skipTest("mincore unavailable")
            self.assertEqual(before, length)

            wrapped = mapped.view(torch.uint8)
            param = torch.nn.Parameter(wrapped, requires_grad=False)
            self.assertTrue(release_mapping_fn(param))

            after = resident_bytes_fn(param)
            self.assertEqual(after, 0)

    def test_read_range_retry(self) -> None:
        """_read_range retries transient 503 responses, bounds total wait, skips 404, and honors stop_event."""
        read_range_fn: Any = getattr(rehydrate, "_read_range")
        payload = b"0123456789abcdef"
        state = {"count_503": 0, "count_always_503": 0, "count_404": 0, "count_zero_retry_after": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/twice-503":
                    state["count_503"] += 1
                    if state["count_503"] <= 2:
                        self.send_response(503)
                        if state["count_503"] == 1:
                            self.send_header("Retry-After", "0.01")
                        self.end_headers()
                        return
                elif self.path == "/always-503":
                    state["count_always_503"] += 1
                    self.send_response(503)
                    self.end_headers()
                    return
                elif self.path == "/zero-retry-after":
                    state["count_zero_retry_after"] += 1
                    self.send_response(503)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
                elif self.path == "/not-found":
                    state["count_404"] += 1
                    self.send_response(404)
                    self.end_headers()
                    return
                range_hdr = self.headers.get("Range") or ""
                start, end = map(int, range_hdr.removeprefix("bytes=").split("-"))
                chunk = payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        def _noop_log(*_args: Any) -> None:
            pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            # Case 1: Server returns 503 twice then 206 for a range.
            logged: list[str] = []
            buf = bytearray(8)
            read_range_fn(
                f"http://127.0.0.1:{port}/twice-503",
                2,
                8,
                memoryview(buf),
                log=logged.append,
                max_wait=1.0,
                backoff_base=0.01,
            )
            self.assertEqual(bytes(buf), payload[2:10])
            self.assertEqual(len(logged), 2)
            self.assertIn("0.01s", logged[0])
            for msg in logged:
                self.assertIn("503", msg)
                self.assertIn("offset 2", msg)
                self.assertIn("twice-503", msg)
                self.assertIn("wait", msg.lower())

            # Case 2: Server always returns 503: fails after bounded wait with attempt count.
            with self.assertRaises(Exception) as always_ctx:
                read_range_fn(
                    f"http://127.0.0.1:{port}/always-503",
                    0,
                    4,
                    memoryview(bytearray(4)),
                    log=_noop_log,
                    max_wait=0.03,
                    backoff_base=0.005,
                )
            self.assertIn("503", str(always_ctx.exception))
            self.assertIn("attempt", str(always_ctx.exception).lower())
            self.assertGreater(state["count_always_503"], 1)

            # Case 3: Server returns 404: not retried.
            logged_404: list[str] = []
            with self.assertRaises(Exception) as not_found_ctx:
                read_range_fn(
                    f"http://127.0.0.1:{port}/not-found",
                    0,
                    4,
                    memoryview(bytearray(4)),
                    log=logged_404.append,
                    max_wait=1.0,
                    backoff_base=0.01,
                )
            self.assertIn("404", str(not_found_ctx.exception))
            self.assertEqual(logged_404, [])
            self.assertEqual(state["count_404"], 1)

            # Case 4: Server returns 503 with Retry-After: 0: fails within budget with attempt count.
            with self.assertRaises(Exception) as zero_ctx:
                read_range_fn(
                    f"http://127.0.0.1:{port}/zero-retry-after",
                    0,
                    4,
                    memoryview(bytearray(4)),
                    log=_noop_log,
                    max_wait=0.03,
                    backoff_base=0.005,
                )
            self.assertIn("503", str(zero_ctx.exception))
            self.assertIn("attempt", str(zero_ctx.exception).lower())
            self.assertGreater(state["count_zero_retry_after"], 1)

            # Case 5: stop_event aborts retry wait promptly without exhausting max_wait.
            stop = threading.Event()
            timer = threading.Timer(0.02, stop.set)
            timer.start()
            t0 = time.monotonic()
            try:
                with self.assertRaises(RuntimeError) as stop_ctx:
                    read_range_fn(
                        f"http://127.0.0.1:{port}/always-503",
                        0,
                        4,
                        memoryview(bytearray(4)),
                        stop_event=stop,
                        log=_noop_log,
                        max_wait=10.0,
                        backoff_base=1.0,
                    )
            finally:
                timer.cancel()
            self.assertIn("load stopped", str(stop_ctx.exception))
            self.assertLess(time.monotonic() - t0, 2.0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
