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

"""Unit tests for test-vector comparison rules in vectors_test.compare."""

from typing import Any
import unittest

from . import vectors_test


def _make_doc(
    kind: str,
    ids: list[int],
    tops: list[tuple[tuple[int, ...], tuple[float, ...]]],
) -> dict[str, Any]:
    return {
        "producer": {"kind": kind},
        "prompts": [
            {
                "name": "p",
                "input_ids": [1, 2],
                "generated_ids": ids,
                "top_logits": [
                    {"ids": list(t), "values": list(v)} for t, v in tops
                ],
            }
        ],
    }


class VectorsCompareTest(unittest.TestCase):
    """Verifies compare() tie-handling and strict token rules on synthetic vectors."""

    def test_compare_tie_and_strict_rules(self) -> None:
        """An exact tie passes strict mode within measured logit error, while wider gaps fail strict mode."""
        compare_fn: Any = getattr(vectors_test, "compare")

        ref = _make_doc(
            "a",
            [7, 3],
            [((7, 3, 5), (10.0, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))],
        )
        # Same logits to within 0.001 at the tie, other token picked: rounding.
        got = _make_doc(
            "b",
            [3, 3],
            [((3, 7, 5), (10.0005, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))],
        )
        _, ok = compare_fn(ref, got, strict_tokens=True)
        self.assertTrue(ok, "a flip at an exact tie within the measured error must pass strict mode")

        # The gap (0.2) is wider than any logit disagreement (0.001): not rounding.
        ref2 = _make_doc(
            "a",
            [7, 3],
            [((7, 3, 5), (10.0, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))],
        )
        got2 = _make_doc(
            "b",
            [3, 3],
            [((7, 3, 5), (10.001, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))],
        )
        rows, ok = compare_fn(ref2, got2, strict_tokens=True)
        self.assertFalse(ok)
        self.assertEqual(rows[0]["flip"]["kind"], "tie")
        self.assertIs(rows[0]["flip"]["explained"], False)

        # Cluster tier (non-strict): within the tie tolerance, passes.
        _, ok = compare_fn(ref2, got2)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
