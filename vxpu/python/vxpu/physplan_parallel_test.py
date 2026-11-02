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

"""The plan file's parallel section (hero run 14): read as pkg/kernelplan validates it and rendered
as the engine's switches."""

import json
import os
import tempfile
import unittest

from vxpu import physplan


def _plan(parallel, ranges=None):
    return {"format": physplan.FORMAT, "ranges": ranges or [{"gpu": 0, "layers": "0:43"}],
            "pipeline": {"batches": 1, "schedule": "range-order"}, "parallel": parallel}


class ParallelSection(unittest.TestCase):
    def test_loads_and_renders(self):
        section = {"dense_tp": 4, "expert_ep": 4, "gpus": [0, 1, 2, 3], "reduce": "one-shot"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plan.json")
            with open(path, "w") as f:
                json.dump(_plan(section), f)
            plan = physplan.load(path)
        self.assertEqual(plan["parallel"], section)
        self.assertEqual(physplan.parallel_env(section),
                         {"VXPU_DENSE_TP": "4", "VXPU_EXPERT_EP": "4", "VXPU_TP_GPUS": "0,1,2,3",
                          "VXPU_EP_EXPERTS": "by-id", "VXPU_REDUCE": "one-shot"})
        self.assertEqual(physplan.parallel_env(None), {})
        self.assertEqual(physplan.parallel_env({"dense_tp": 1, "expert_ep": 1}), {})

    def test_refusals(self):
        ranges4 = [{"gpu": g, "layers": l} for g, l in enumerate(("0:8", "8:20", "20:32", "32:43"))]
        for section, ranges, want in [
            ({"dense_tp": 3, "expert_ep": 1}, None, "1, 2 or 4"),
            ({"dense_tp": 4, "expert_ep": 2, "gpus": [0, 1, 2, 3]}, None, "expert_ep"),
            ({"dense_tp": 4, "expert_ep": 4, "gpus": [0, 1]}, None, "gpus"),
            ({"dense_tp": 4, "expert_ep": 4, "gpus": [0, 1, 2, 3]}, ranges4, "one range"),
            ({"dense_tp": 4, "expert_ep": 4, "gpus": [1, 0, 2, 3]}, None, "first card"),
            ({"dense_tp": 4, "expert_ep": 4, "gpus": [0, 1, 2, 3], "reduce": "tree"}, None, "reduce"),
            ({"dense_tp": 4, "expert_ep": 4, "gpus": [0, 1, 2, 3], "experts": "somewhere"}, None, "experts"),
        ]:
            with self.assertRaises(ValueError, msg=str(section)) as cm:
                physplan.check_parallel(section, ranges or [{"gpu": 0, "layers": "0:43"}])
            self.assertIn(want, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
