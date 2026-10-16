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

import http.server
import socketserver
import tempfile
import threading
import time
import unittest

import torch
from vxpu import kernels, rehydrate


class RangeHTTPHandler(http.server.BaseHTTPRequestHandler):
    content = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    retry_counts = {}

    def log_message(self, format, *args):
        pass  # suppress request logging in tests

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/data")
            self.end_headers()
            return

        if self.path == "/retry_then_206":
            count = self.retry_counts.get("retry_then_206", 0)
            self.retry_counts["retry_then_206"] = count + 1
            if count < 2:
                self.send_response(503)
                self.end_headers()
                return
            self.path = "/data"

        if self.path == "/always_500":
            self.send_response(500)
            self.end_headers()
            return

        if self.path == "/not_found":
            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/ignore_range_200":
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.content)))
            self.end_headers()
            self.wfile.write(self.content)
            return

        if self.path == "/stall":
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-9/{len(self.content)}")
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.write(b"123")
            self.wfile.flush()
            time.sleep(3)  # stall
            return

        if self.path == "/data":
            range_hdr = self.headers.get("Range")
            if range_hdr:
                parts = range_hdr.replace("bytes=", "").split("-")
                start = int(parts[0])
                end = int(parts[1])
                sub = self.content[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.content)}")
                self.send_header("Content-Length", str(len(sub)))
                self.end_headers()
                self.wfile.write(sub)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(self.content)))
                self.end_headers()
                self.wfile.write(self.content)


class TestNativeFetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = socketserver.TCPServer(("127.0.0.1", 0), RangeHTTPHandler)
        cls.port = cls.httpd.server_address[1]
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        RangeHTTPHandler.retry_counts.clear()

    def test_extension_available_and_has_fetch(self):
        self.assertTrue(kernels.available(), kernels.why_not())
        self.assertTrue(kernels.has_fetch(), "libcurl fetcher should be compiled in")

    def test_file_range_fetch(self):
        mod = kernels.load()
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            temp_path = f.name

        dest = torch.zeros(20, dtype=torch.uint8)
        # Fetch 10 bytes starting from offset 5 into dest at offset 2
        mod.fetch_ranges([(f"file://{temp_path}", 5, 10, dest, 2)])
        val = bytes(dest.numpy())
        self.assertEqual(val[:2], b"\x00\x00")
        self.assertEqual(val[2:12], b"FGHIJKLMNO")
        self.assertEqual(val[12:], b"\x00" * 8)

    def test_http_range_fetch(self):
        mod = kernels.load()
        dest = torch.zeros(16, dtype=torch.uint8)
        # Fetch 'abcdefgh' (8 bytes from offset 10)
        mod.fetch_ranges([(f"{self.base_url}/data", 10, 8, dest, 4)])
        val = bytes(dest.numpy())
        self.assertEqual(val[:4], b"\x00\x00\x00\x00")
        self.assertEqual(val[4:12], b"abcdefgh")
        self.assertEqual(val[12:], b"\x00" * 4)

    def test_redirect(self):
        mod = kernels.load()
        dest = torch.zeros(10, dtype=torch.uint8)
        mod.fetch_ranges([(f"{self.base_url}/redirect", 0, 10, dest, 0)])
        self.assertEqual(bytes(dest.numpy()), b"0123456789")

    def test_retry_on_5xx(self):
        mod = kernels.load()
        dest = torch.zeros(6, dtype=torch.uint8)
        mod.fetch_ranges([(f"{self.base_url}/retry_then_206", 0, 6, dest, 0)], max_retries=3)
        self.assertEqual(bytes(dest.numpy()), b"012345")
        self.assertEqual(RangeHTTPHandler.retry_counts.get("retry_then_206"), 3)

    def test_fail_on_continuous_500(self):
        mod = kernels.load()
        dest = torch.zeros(6, dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            mod.fetch_ranges([(f"{self.base_url}/always_500", 0, 6, dest, 0)], max_retries=2)
        self.assertIn("500", str(ctx.exception))

    def test_fail_on_404(self):
        mod = kernels.load()
        dest = torch.zeros(6, dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            mod.fetch_ranges([(f"{self.base_url}/not_found", 0, 6, dest, 0)], max_retries=2)
        self.assertIn("404", str(ctx.exception))

    def test_fail_when_server_ignores_range_on_partial(self):
        mod = kernels.load()
        dest = torch.zeros(6, dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            # offset 10 requested, but server returned HTTP 200 full content (> 6 bytes)
            mod.fetch_ranges([(f"{self.base_url}/ignore_range_200", 10, 6, dest, 0)], max_retries=1)
        self.assertTrue("server sent more than" in str(ctx.exception) or "expected HTTP 206" in str(ctx.exception))

    def test_stall_timeout(self):
        mod = kernels.load()
        dest = torch.zeros(10, dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            mod.fetch_ranges([(f"{self.base_url}/stall", 0, 10, dest, 0)], max_retries=1, stall_timeout=1)
        self.assertIn("fetching", str(ctx.exception).lower())

    def test_dest_tensor_bounds_validation(self):
        mod = kernels.load()
        dest = torch.zeros(10, dtype=torch.uint8)
        # dest_offset (5) + length (10) > 10
        with self.assertRaises(RuntimeError) as ctx:
            mod.fetch_ranges([(f"{self.base_url}/data", 0, 10, dest, 5)])
        self.assertIn("exceeds tensor size", str(ctx.exception))

    def test_dest_tensor_contiguous_validation(self):
        mod = kernels.load()
        # non-contiguous slice
        t = torch.zeros((4, 4), dtype=torch.uint8).t()
        self.assertFalse(t.is_contiguous())
        with self.assertRaises(RuntimeError) as ctx:
            mod.fetch_ranges([(f"{self.base_url}/data", 0, 4, t, 0)])
        self.assertIn("contiguous", str(ctx.exception))

    def test_multiple_concurrent_ranges(self):
        mod = kernels.load()
        dest = torch.zeros(36, dtype=torch.uint8)
        ranges = [
            (f"{self.base_url}/data", 0, 10, dest, 0),
            (f"{self.base_url}/data", 10, 10, dest, 10),
            (f"{self.base_url}/data", 20, 10, dest, 20),
            (f"{self.base_url}/data", 30, 6, dest, 30),
        ]
        mod.fetch_ranges(ranges, max_concurrent=2)
        expected = RangeHTTPHandler.content[:36]
        self.assertEqual(bytes(dest.numpy()), expected)

    def test_rehydrate_fetch_tensor_native_and_fallback(self):
        url = f"{self.base_url}/data"
        files = {"hash1": {"source": url}}
        ref = {"file_sha256": "hash1", "offset": 10, "length": 12, "dtype": "U8", "shape": [12]}

        # Native path
        self.assertTrue(kernels.has_fetch())
        t_native, dl_native = rehydrate.fetch_tensor(ref, files)
        self.assertEqual(dl_native, 12)
        self.assertEqual(bytes(t_native.numpy()), RangeHTTPHandler.content[10:22])

        # Fallback path (mock has_fetch to False)
        orig = kernels.has_fetch
        kernels.has_fetch = lambda: False
        try:
            t_fallback, dl_fallback = rehydrate.fetch_tensor(ref, files)
            self.assertEqual(dl_fallback, 12)
            self.assertEqual(bytes(t_fallback.numpy()), RangeHTTPHandler.content[10:22])
        finally:
            kernels.has_fetch = orig

    def test_rehydrate_fetch_stacked_native_and_fallback(self):
        url = f"{self.base_url}/data"
        files = {"hash1": {"source": url}}
        manifest_tensors = {
            "t0": {"file_sha256": "hash1", "offset": 0, "length": 4, "dtype": "U8", "shape": [4]},
            "t1": {"file_sha256": "hash1", "offset": 4, "length": 4, "dtype": "U8", "shape": [4]},
            "t2": {"file_sha256": "hash1", "offset": 8, "length": 4, "dtype": "U8", "shape": [4]},
            "t3": {"file_sha256": "hash1", "offset": 12, "length": 4, "dtype": "U8", "shape": [4]},
        }
        spec = {
            "parts": [["t0", "t1"], ["t2", "t3"]],
            "cat_dim": 0,
            "stack": True,
        }

        # Native path
        self.assertTrue(kernels.has_fetch())
        t_native, dl_native = rehydrate.fetch_stacked(spec, manifest_tensors, files)
        self.assertEqual(dl_native, 16)
        self.assertEqual(t_native.shape, torch.Size([2, 8]))
        self.assertEqual(bytes(t_native[0].numpy()), RangeHTTPHandler.content[0:8])
        self.assertEqual(bytes(t_native[1].numpy()), RangeHTTPHandler.content[8:16])

        # Fallback path
        orig = kernels.has_fetch
        kernels.has_fetch = lambda: False
        try:
            t_fallback, dl_fallback = rehydrate.fetch_stacked(spec, manifest_tensors, files)
            self.assertEqual(dl_fallback, 16)
            self.assertEqual(t_fallback.shape, torch.Size([2, 8]))
            self.assertEqual(bytes(t_fallback[0].numpy()), RangeHTTPHandler.content[0:8])
            self.assertEqual(bytes(t_fallback[1].numpy()), RangeHTTPHandler.content[8:16])
        finally:
            kernels.has_fetch = orig


if __name__ == "__main__":
    unittest.main()
