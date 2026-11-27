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

"""Unit tests for native C++ CPU kernel helpers (gather_rows, gather_table_rows, NodeClient)."""

import os
import tempfile
from typing import Any
import unittest

import torch

from . import kernels


class KernelsTest(unittest.TestCase):
    """Verifies C++ host-read kernels and NodeClient reconnection without building an Engine."""

    def _require_kernels(self) -> Any:
        k = kernels.load()
        if k is None:
            self.skipTest("native CPU kernels unavailable")
        return k

    def test_host_reads(self) -> None:
        """gather_rows and gather_table_rows match index_select and reject bad or device indices."""
        k = self._require_kernels()
        if not hasattr(k, "gather_rows"):
            self.skipTest("gather_rows unavailable in native CPU kernels")

        table = torch.arange(64 * 24, dtype=torch.int16).view(64, 24)
        idx = torch.tensor([[3, 63, 0], [17, 17, 5]])
        self.assertTrue(torch.equal(k.gather_rows(table, idx), table[idx]))

        for bad in (torch.tensor([64]), torch.tensor([-1])):
            with self.assertRaisesRegex(RuntimeError, "outside"):
                k.gather_rows(table, bad)

        device = "cuda" if torch.cuda.is_available() else "meta"
        with self.assertRaisesRegex(RuntimeError, "indices on the CPU"):
            k.gather_rows(table, idx.to(device))

        # File-backed n-gram table reads: compare pread and io_uring paths against gather_rows.
        with tempfile.TemporaryDirectory(prefix="vxpu-host-reads-") as workdir:
            tbl_path = os.path.join(workdir, "gather-rows-file.bin")
            w_offset = 128
            rows = table.shape[0]
            scale_table = table[:, : table.shape[1] // 2].contiguous()
            w_row_bytes = table.shape[1] * table.element_size()
            s_row_bytes = scale_table.shape[1] * scale_table.element_size()
            s_offset = w_offset + rows * w_row_bytes + 64
            with open(tbl_path, "wb") as f:
                f.write(b"\0" * w_offset)
                f.write(table.numpy().tobytes())
                f.write(b"\0" * (s_offset - (w_offset + rows * w_row_bytes)))
                f.write(scale_table.numpy().tobytes())

            fd = os.open(tbl_path, os.O_RDONLY)
            try:
                g = torch.Generator().manual_seed(42)
                rand_idx = torch.randint(0, rows, (4, 6), generator=g)
                want_w = k.gather_rows(table, rand_idx)
                want_s = k.gather_rows(scale_table, rand_idx)

                # use_uring=False: pread fallback.
                w_got, s_got = k.gather_table_rows(
                    fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx, use_uring=False
                )
                self.assertEqual(w_got.numpy().tobytes(), want_w.numpy().tobytes())
                self.assertEqual(s_got.numpy().tobytes(), want_s.numpy().tobytes())

                # use_uring=True: io_uring when available, pread fallback otherwise.
                w_got_u, s_got_u = k.gather_table_rows(
                    fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx, use_uring=True
                )
                self.assertEqual(w_got_u.numpy().tobytes(), want_w.numpy().tobytes())
                self.assertEqual(s_got_u.numpy().tobytes(), want_s.numpy().tobytes())

                for bad in (torch.tensor([64]), torch.tensor([-1])):
                    with self.assertRaisesRegex(RuntimeError, "outside"):
                        k.gather_table_rows(fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, bad)

                with self.assertRaisesRegex(RuntimeError, "indices on the CPU"):
                    k.gather_table_rows(
                        fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx.to(device)
                    )
            finally:
                os.close(fd)

    def test_node_client_reconnect(self) -> None:
        """NodeClient fails immediately when its NodeServer stops and reconnects when it returns."""
        k = self._require_kernels()

        def call(client: Any) -> str:
            try:
                client.call(0, [])
            except RuntimeError as e:
                return str(e).splitlines()[0]
            return "answered"

        server = k.NodeServer(0, [], [])  # No nodes: every call answers with an error.
        port = server.port()
        try:
            client = k.NodeClient("127.0.0.1", port)
            first = call(client)
            server.stop()
            gone = call(client)
            server = k.NodeServer(port, [], [])
            back = call(client)
        finally:
            server.stop()

        self.assertIn("no such node", first)
        self.assertIn("cannot connect", gone)
        self.assertIn("no such node", back)


if __name__ == "__main__":
    unittest.main()
