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

"""Unit tests for range coalescing and rehydration in vxpu."""

import hashlib
import os
import tempfile
import unittest

import torch

from .rehydrate import (
    GAP_BYTES,
    RUN_PART_BYTES,
    ReadItem,
    _Run,
    _can_map_straight,
    plan_and_fetch_ranges,
)


class TestRehydrate(unittest.TestCase):
    def test_can_map_straight(self):
        t = torch.empty(10, dtype=torch.uint8)
        item = ReadItem("sha", 100, 10, t)
        run = _Run("sha", 100, 10, [item])
        self.assertTrue(_can_map_straight(run))

        # Multiple items cannot map straight
        item2 = ReadItem("sha", 110, 5, t)
        run_multi = _Run("sha", 100, 15, [item, item2])
        self.assertFalse(_can_map_straight(run_multi))

        # Item with cat_dim cannot map straight
        item_cat = ReadItem("sha", 100, 10, t, cat_dim=0, slot_offset=0)
        run_cat = _Run("sha", 100, 10, [item_cat])
        self.assertFalse(_can_map_straight(run_cat))

        # Run with gap cannot map straight
        run_gap = _Run("sha", 100, 20, [item])
        self.assertFalse(_can_map_straight(run_gap))

    def test_plan_and_fetch_coalesce_and_scatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            # Create a 2 MB file with sequential byte data
            content = bytes(i % 256 for i in range(2 * 1024 * 1024))
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            # Range 1: 1000..2000 (1000 bytes)
            # Range 2: 2500..3500 (1000 bytes, gap of 500 bytes < GAP_BYTES)
            # Range 3: 1,500,000..1,501,000 (gap > GAP_BYTES -> new run)
            dst1 = torch.empty(250, dtype=torch.int32)
            dst2 = torch.empty((2, 500), dtype=torch.uint8)
            dst3 = torch.empty(1000, dtype=torch.uint8)

            # Stacked target for dst2: row 0 gets 500 bytes, row 1 gets 500 bytes
            item1 = ReadItem(file_sha, 1000, 1000, dst1)
            item2_row0 = ReadItem(file_sha, 2500, 500, dst2[0], cat_dim=0, slot_offset=0, shape=[500], dtype=torch.uint8)
            item2_row1 = ReadItem(file_sha, 3000, 500, dst2[1], cat_dim=0, slot_offset=0, shape=[500], dtype=torch.uint8)
            item3 = ReadItem(file_sha, 1500000, 1000, dst3)

            downloaded, requests = plan_and_fetch_ranges(
                [item1, item2_row0, item2_row1, item3], files, gap_bytes=GAP_BYTES
            )

            # item1 and item2 coalesce into 1 run of 2500 bytes (from 1000 to 3500)
            # item3 is 1 run of 1000 bytes
            # Total requests should be 2!
            self.assertEqual(requests, 2)
            self.assertEqual(downloaded, 2500 + 1000)

            # Verify contents
            expected1 = memoryview(content[1000:2000])
            self.assertEqual(bytes(dst1.view(torch.uint8).numpy()), bytes(expected1))

            expected2_0 = memoryview(content[2500:3000])
            self.assertEqual(bytes(dst2[0].numpy()), bytes(expected2_0))

            expected2_1 = memoryview(content[3000:3500])
            self.assertEqual(bytes(dst2[1].numpy()), bytes(expected2_1))

            expected3 = memoryview(content[1500000:1501000])
            self.assertEqual(bytes(dst3.numpy()), bytes(expected3))

    def test_cas_cache_hit_and_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            cas_dir = os.path.join(tmpdir, "cas")
            content = b"0123456789ABCDEF" * 100
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            dst1 = torch.empty(16, dtype=torch.uint8)
            item1 = ReadItem(file_sha, 0, 16, dst1)

            # First load: cache miss, downloads and writes to CAS
            dl1, req1 = plan_and_fetch_ranges([item1], files, cas_dir=cas_dir)
            self.assertEqual(req1, 1)
            self.assertEqual(dl1, 16)
            self.assertEqual(bytes(dst1.numpy()), content[:16])

            # Check that CAS file exists
            key = hashlib.sha256(f"{file_sha}:0:16".encode()).hexdigest()
            cas_file = os.path.join(cas_dir, key)
            self.assertTrue(os.path.exists(cas_file))
            with open(cas_file, "rb") as f:
                self.assertEqual(f.read(), content[:16])

            # Second load: cache hit, 0 network requests
            dst2 = torch.empty(16, dtype=torch.uint8)
            item2 = ReadItem(file_sha, 0, 16, dst2)
            dl2, req2 = plan_and_fetch_ranges([item2], files, cas_dir=cas_dir)
            self.assertEqual(req2, 0)
            self.assertEqual(dl2, 0)
            self.assertEqual(bytes(dst2.numpy()), content[:16])

    def test_run_part_splitting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "weights.bin")
            content = b"X" * 1000
            with open(file_path, "wb") as f:
                f.write(content)

            file_sha = hashlib.sha256(content).hexdigest()
            files = {file_sha: {"source": f"file://{file_path}", "size": len(content)}}

            dst = torch.empty(1000, dtype=torch.uint8)
            item = ReadItem(file_sha, 0, 1000, dst)

            # Force part_bytes to 200, so a 1000 byte run splits into 5 parts
            dl, req = plan_and_fetch_ranges([item], files, part_bytes=200)
            self.assertEqual(req, 5)
            self.assertEqual(dl, 1000)
            self.assertEqual(bytes(dst.numpy()), content)


if __name__ == "__main__":
    unittest.main()
