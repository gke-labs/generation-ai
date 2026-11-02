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

"""The parallel branch's rule: which node goes on it and what it waits for (stepgraph.branch_choice),
held without a GPU on the shapes the step has."""
import unittest

from . import stepgraph


class BranchChoiceTest(unittest.TestCase):
    def test_island_after_dense_overlaps_it(self):
        # the attention's key/value projection beside the query projection: both read the node before
        on_par, wait_last, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", False, enabled=True)
        self.assertTrue(on_par)
        self.assertFalse(wait_last)

    def test_reader_of_main_last_stays_on_main(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"q"}, "q", "dense", False, enabled=True)
        self.assertFalse(on_par)

    def test_island_after_host_node_waits_for_it(self):
        # the shared expert's front beside the routed experts: on the branch, but after the host node
        on_par, wait_last, _ = stepgraph.branch_choice("dense", {"front"}, "experts", "experts", False, enabled=True)
        self.assertTrue(on_par)
        self.assertTrue(wait_last)

    def test_host_node_waits_for_the_branch(self):
        on_par, _, host_waits = stepgraph.branch_choice("experts", {"front"}, "front", "dense", False, enabled=True)
        self.assertFalse(on_par)
        self.assertTrue(host_waits)

    def test_a_wave_from_another_gpu_keeps_a_node_on_main(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", True, enabled=True)
        self.assertFalse(on_par)

    def test_disabled(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", False, enabled=False)
        self.assertFalse(on_par)




class PositionalCacheTest(unittest.TestCase):
    """The window refresh applies to [rows, positions, ...] caches, not to a small running state."""

    def test_caches_and_states(self):
        import torch
        from vxpu.stepgraph import positional_cache
        self.assertTrue(positional_cache(torch.empty(33, 2048, 512), 33, 1))
        self.assertTrue(positional_cache(torch.empty(33, 128, 512), 33, 5))
        self.assertFalse(positional_cache(torch.empty(33, 2, 512), 33, 1), "a compressor's two-slot state")
        self.assertFalse(positional_cache(torch.empty(9, 2048, 512), 33, 1), "another batch's rows")
        self.assertFalse(positional_cache(torch.empty(33, 4), 33, 1), "no position axis")
        self.assertFalse(positional_cache(torch.empty(33, 5, 512), 33, 5), "no room past the step's tokens")

    def test_compressed_caches_take_no_window(self):
        from vxpu.stepgraph import compressed_cache
        state = {"model.layers.8.attn.compress_kv_cache": 1, "model.layers.8.attn.compressor.kv_state": 1,
                 "model.layers.20.attn.compress_kv_cache": 1}
        self.assertTrue(compressed_cache("model.layers.8.attn.compress_kv_cache", state))
        self.assertFalse(compressed_cache("model.layers.20.attn.compress_kv_cache", state))
        self.assertFalse(compressed_cache("model.engram_hash.cache", state))


if __name__ == "__main__":
    unittest.main()
