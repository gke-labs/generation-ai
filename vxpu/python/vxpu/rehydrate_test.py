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

# pyright: strict

"""Tests for page-cache release verification in the vectors self-test."""

import contextlib
import ctypes
import io
import mmap
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import rehydrate
from .vectors_test import check_page_cache_release


def _has_linux_madv_pageout() -> bool:
    return getattr(mmap, "MADV_PAGEOUT", 21 if sys.platform.startswith("linux") else None) is not None


class PageCacheReleaseSelfTest(unittest.TestCase):
    """Checks that check_page_cache_release is deterministic and catches bugs."""

    def setUp(self) -> None:
        if not _has_linux_madv_pageout():
            self.skipTest("MADV_PAGEOUT is unavailable on this platform")

    def test_passes_repeatedly(self) -> None:
        """Runs check_page_cache_release repeatedly in a temporary directory."""
        outputs: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            for _ in range(25):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    check_page_cache_release(td)
                outputs.append(buf.getvalue().strip())
        self.assertEqual(len(outputs), 25)
        for line in outputs:
            self.assertIn("madvise(MADV_PAGEOUT) verified for exact ranges", line)
        # Print the distinct kernel observations once for CI logs.
        for line in sorted(set(outputs)):
            print(f"  [25 runs] {line}", flush=True)

    def test_fails_when_madvise_not_called(self) -> None:
        """Fails when release_ranges succeeds without issuing madvise."""
        def _noop_release(_ranges: list[tuple[int, int]]) -> tuple[bool, int]:
            return True, 0

        with tempfile.TemporaryDirectory() as td:
            with patch.object(rehydrate, "release_ranges", side_effect=_noop_release):
                with self.assertRaises(SystemExit) as ctx:
                    check_page_cache_release(td)
            self.assertIn("made no madvise call", str(ctx.exception))

    def test_fails_when_range_is_wrong(self) -> None:
        """Fails when release_mapping asks the kernel for a shifted or resized range."""
        page_size = mmap.PAGESIZE
        real_release_ranges = rehydrate.release_ranges

        def _shifted_release(ranges: list[tuple[int, int]]) -> tuple[bool, int]:
            return real_release_ranges([(s + page_size, e) for s, e in ranges])

        def _oversized_release(ranges: list[tuple[int, int]]) -> tuple[bool, int]:
            return real_release_ranges([(s, e + page_size) for s, e in ranges])

        def _shrunk_release(ranges: list[tuple[int, int]]) -> tuple[bool, int]:
            return real_release_ranges([(s, e - page_size) for s, e in ranges])

        for bad_release in (_shifted_release, _oversized_release, _shrunk_release):
            with tempfile.TemporaryDirectory() as td:
                with patch.object(rehydrate, "release_ranges", side_effect=bad_release):
                    with self.assertRaises(SystemExit) as ctx:
                        check_page_cache_release(td)
                self.assertIn("expected madvise call", str(ctx.exception))

    def test_fails_when_advice_is_wrong(self) -> None:
        """Fails when release_ranges passes an advice flag other than MADV_PAGEOUT."""
        wrong_advice = getattr(mmap, "MADV_RANDOM", 1)

        def _wrong_advice_release(ranges: list[tuple[int, int]]) -> tuple[bool, int]:
            libc = ctypes.CDLL(None, use_errno=True)
            madvise = libc.madvise
            madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            for s, e in ranges:
                if madvise(ctypes.c_void_p(s), e - s, wrong_advice) != 0:
                    return False, ctypes.get_errno()
            return True, 0

        with tempfile.TemporaryDirectory() as td:
            with patch.object(rehydrate, "release_ranges", side_effect=_wrong_advice_release):
                with self.assertRaises(SystemExit) as ctx:
                    check_page_cache_release(td)
            self.assertIn("expected madvise call", str(ctx.exception))

    def test_fails_when_neighbor_loses_pages(self) -> None:
        """Fails when resident_bytes reports pages missing outside the released range."""
        real_resident_ranges = rehydrate.resident_bytes_ranges
        call_count = 0

        def _dropping_neighbor(ranges: list[tuple[int, int]]) -> int | None:
            nonlocal call_count
            call_count += 1
            res = real_resident_ranges(ranges)
            # Call 4 measures the neighbor immediately after releasing the target half.
            if call_count == 4 and res is not None:
                return max(0, res - mmap.PAGESIZE)
            return res

        with tempfile.TemporaryDirectory() as td:
            with patch.object(rehydrate, "resident_bytes_ranges", side_effect=_dropping_neighbor):
                with self.assertRaises(SystemExit) as ctx:
                    check_page_cache_release(td)
            self.assertIn("neighbor outside released range", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
