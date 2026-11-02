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

"""The plan file's drafter field (hero run 13): the GPU that runs the drafter's stages."""

import json
import os
import tempfile
import unittest

from vxpu import experts_gpu, physplan


def _write(plan):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(plan, f)
    f.close()
    return f.name


class DrafterField(unittest.TestCase):
    base = {"format": physplan.FORMAT, "ranges": [{"gpu": 0, "layers": "0:8"}, {"gpu": 3, "layers": "8:43"}],
            "pipeline": {"batches": 1}}

    def test_a_drafter_gpu_must_hold_a_range(self):
        ok = _write(dict(self.base, drafter={"gpu": 0}))
        self.assertEqual(physplan.load(ok)["drafter"], {"gpu": 0})
        for bad in ({"gpu": 2}, {"gpu": -1}, {"gpu": "0"}, 5):
            with self.assertRaises(ValueError):
                physplan.load(_write(dict(self.base, drafter=bad)))

    def test_apply_env_renders_the_drafter_device(self):
        path = _write(dict(self.base, drafter={"gpu": 0}))
        saved = {k: os.environ.get(k) for k in ("VXPU_PLAN", "VXPU_DRAFT_DEVICE", "VXPU_LAYER_DEVICES")}
        physplan._loaded = None
        try:
            for k in saved:
                os.environ.pop(k, None)
            os.environ["VXPU_PLAN"] = path
            physplan.apply_env()
            self.assertEqual(os.environ["VXPU_DRAFT_DEVICE"], "cuda:0")
            self.assertEqual(os.environ["VXPU_LAYER_DEVICES"], "0:8=cuda:0,8:43=cuda:3")
        finally:
            physplan._loaded = None
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class DrafterExperts(unittest.TestCase):
    def test_the_stages_experts_follow_the_drafter(self):
        gpus = [{0: [1, 2], 40: [7]}, {8: [3]}, {}, {39: [4], 40: [5, 6], 41: [8], 42: [9]}]
        out, moved = experts_gpu.with_layers_on(gpus, [40, 41, 42], 0)
        self.assertEqual(out[0], {0: [1, 2], 40: [5, 6, 7], 41: [8], 42: [9]})
        self.assertEqual(out[3], {39: [4]})
        self.assertEqual(out[1], {8: [3]})
        self.assertEqual(moved, [(40, 3), (41, 3), (42, 3)])
        self.assertEqual(gpus[3][40], [5, 6], "the input is not changed")
        same, none = experts_gpu.with_layers_on(out, [40, 41, 42], 0)
        self.assertEqual((same, none), (out, []))


if __name__ == "__main__":
    unittest.main()
