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

"""The engine's expert statistics over the step graphs of every batch in flight (hero run 12)."""

import unittest

from . import engine


class _Graph:
    def __init__(self, counts):
        self._counts = counts

    def expert_stats(self):
        return self._counts


class ExpertStats(unittest.TestCase):
    def test_sums_every_batch_graph_and_skips_graphs_that_do_not_count(self):
        own = _Graph({0: 6, 1: 6})
        batch1 = _Graph({0: 30, 1: 36})
        drafter = _Graph(None)  # the drafter's stages have no experts
        self.assertEqual(engine.sum_expert_stats([own, batch1, drafter, None]), {0: 36, 1: 42})
        self.assertIsNone(engine.sum_expert_stats([drafter, None]))

    def test_the_engine_reads_its_own_graph_and_the_pipeline_graphs(self):
        e = engine.Engine.__new__(engine.Engine)
        e._decode_split = _Graph({0: 6})
        e._pipeline_graphs = [(None, _Graph({0: 30}), _Graph(None)), (None, _Graph({0: 37}), None)]
        self.assertEqual(e.expert_stats(), {0: 73})
        del e._pipeline_graphs
        self.assertEqual(e.expert_stats(), {0: 6})


if __name__ == "__main__":
    unittest.main()
