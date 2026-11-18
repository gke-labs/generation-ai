#!/usr/bin/env python3
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

"""Test runner that prints each test as it starts and summarizes skips."""

import sys
from typing import TYPE_CHECKING, Any
import unittest

if TYPE_CHECKING:
    _BaseResult = unittest.TextTestResult[Any]
else:
    _BaseResult = unittest.TextTestResult


class SummaryTextTestResult(_BaseResult):
    """Test result class that flushes test starts and reports skipped tests."""

    def startTest(self, test: unittest.case.TestCase) -> None:
        super().startTest(test)
        self.stream.flush()
        sys.stdout.flush()
        sys.stderr.flush()

    def printErrors(self) -> None:
        super().printErrors()
        if self.skipped:
            self.stream.writeln()
            self.stream.writeln("=" * 70)
            self.stream.writeln(f"Summary of skipped tests ({len(self.skipped)}):")
            self.stream.writeln("=" * 70)
            for test, reason in self.skipped:
                self.stream.writeln(f"  {test.id()}: {reason}")
            self.stream.writeln()
            self.stream.flush()


class SummaryTextTestRunner(unittest.TextTestRunner):
    """Text test runner that uses SummaryTextTestResult."""

    resultclass = SummaryTextTestResult


def main(argv: list[str] | None = None) -> None:
    """Discovers and runs tests, printing a summary of skipped tests at the end."""
    if argv is None:
        argv = sys.argv[1:]

    args: list[str]
    if not argv:
        args = ["discover", "-s", "vxpu/python", "-p", "*_test.py", "-v"]
    elif argv[0] not in ("discover", "-h", "--help") and any(a in ("-s", "-p", "-t") for a in argv):
        args = ["discover", "-s", "vxpu/python"] + list(argv)
    else:
        args = list(argv)

    prog = unittest.TestProgram(
        module=None,
        argv=[sys.argv[0]] + args,
        testRunner=SummaryTextTestRunner,
        exit=False,
    )
    if not prog.result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
