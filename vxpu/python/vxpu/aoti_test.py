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

"""Unit tests for AOTInductor package extraction and cache stray cleanup in aoti.py."""

import os
import subprocess
import tempfile
from typing import Any
import unittest
from unittest import mock
import zipfile

from . import aoti


class AOTIExtractionTest(unittest.TestCase):
    """Verifies stray temporary sweeping and keyed archive extraction in aoti.py."""

    def test_clean_cache_strays(self) -> None:
        """clean_cache_strays removes dead-PID temporaries and preserves live, overflow, and non-PID entries."""
        clean_fn: Any = getattr(aoti, "clean_cache_strays")
        with tempfile.TemporaryDirectory(prefix="vxpu-aoti-strays-") as cache_dir:
            process = subprocess.Popen(["true"])
            process.wait()
            dead_pid = process.pid

            dead_tmp_dir = os.path.join(cache_dir, f"segment_stray.d.{dead_pid}.tmp")
            dead_tmp_file = os.path.join(cache_dir, f"segment_stray.{dead_pid}.tmp.pt2")
            alive_tmp_dir = os.path.join(cache_dir, f"segment_stray.d.{os.getpid()}.tmp")
            alive_tmp_file = os.path.join(cache_dir, f"segment_stray.{os.getpid()}.tmp.pt2")
            overflow_tmp_dir = os.path.join(cache_dir, "segment_stray.d.1762000000000.tmp")
            non_pid_tmp_file = os.path.join(cache_dir, "segment_stray.other.tmp.pt2")

            os.makedirs(dead_tmp_dir, exist_ok=True)
            os.makedirs(alive_tmp_dir, exist_ok=True)
            os.makedirs(overflow_tmp_dir, exist_ok=True)
            for path, content in (
                (dead_tmp_file, "dead"),
                (alive_tmp_file, "alive"),
                (non_pid_tmp_file, "other"),
            ):
                with open(path, "w") as f:
                    f.write(content)

            clean_fn(cache_dir)

            self.assertFalse(os.path.exists(dead_tmp_dir))
            self.assertFalse(os.path.exists(dead_tmp_file))
            self.assertTrue(os.path.exists(alive_tmp_dir))
            self.assertTrue(os.path.exists(alive_tmp_file))
            self.assertTrue(os.path.exists(overflow_tmp_dir))
            self.assertTrue(os.path.exists(non_pid_tmp_file))

    def test_extract_creates_reuses_and_propagates_rename_error(self) -> None:
        """extract unpacks once into a keyed directory, reuses it on repeat calls, and raises on rename errors."""
        extract_fn: Any = getattr(aoti, "extract")
        with tempfile.TemporaryDirectory(prefix="vxpu-aoti-extract-") as cache_dir:
            fake_pt2 = os.path.join(cache_dir, "segment_test_key.pt2")
            payload = b"fake binary for testing aoti extraction path arithmetic"
            with zipfile.ZipFile(fake_pt2, "w") as archive:
                archive.writestr("model.so", payload)
                archive.writestr("_metadata.json", b"{}")

            want_package_dir = os.path.join(cache_dir, "segment_test_key.d")
            got_dir1, bytes1 = extract_fn(fake_pt2)
            self.assertEqual(got_dir1, want_package_dir)
            self.assertTrue(os.path.isdir(got_dir1))
            self.assertGreaterEqual(bytes1, len(payload))
            with open(os.path.join(got_dir1, "model.so"), "rb") as f:
                self.assertEqual(f.read(), payload)

            # Second call reuses the existing directory and reports 0 extracted bytes.
            got_dir2, bytes2 = extract_fn(fake_pt2)
            self.assertEqual(got_dir2, want_package_dir)
            self.assertEqual(bytes2, 0)

            # Failed rename propagates OSError.
            broken_pt2 = os.path.join(cache_dir, "broken.pt2")
            with zipfile.ZipFile(broken_pt2, "w") as archive:
                archive.writestr("model.so", b"data")
            with mock.patch("os.replace", side_effect=OSError("simulated rename failure")):
                with self.assertRaises(OSError):
                    extract_fn(broken_pt2)


if __name__ == "__main__":
    unittest.main()
