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

"""parallel.Decode: the ranks' inputs are moved before any rank's graph runs.

A rank program may be handed tensors that live on another rank's GPU (the sharded drafter hands
ranks 1 to 3 tensors on rank 0's device); a rank moving them on its own thread does a cross-device
copy that waits on the source GPU's stream, and when that stream is already spinning in a sum
launched by another rank, the step hangs (hero run 15: the drafted TP step with every rank
program sharing one Comm, two ranks at 100% and two idle). Decode moves every rank's inputs on
the main thread first; these tests hold it to that, on the CPU by order and on two GPUs by the
deadlock itself."""
import ctypes
import threading
import unittest

import torch

from . import parallel


class _Graph:
    """A rank's step: records whether every rank's inputs had been moved when it was called."""

    def __init__(self, moved, size):
        self.moved, self.size = moved, size
        self.seen = []

    def __call__(self, input_ids=None, cache_position=None):
        with self.moved["lock"]:
            self.seen.append(self.moved["n"])
        return input_ids


class DecodeMovesInputsFirstTest(unittest.TestCase):
    def test_every_rank_moved_before_any_graph(self):
        size = 3
        moved = {"n": 0, "lock": threading.Lock()}
        world = parallel.World(["cpu"] * size)
        graphs = [_Graph(moved, size) for _ in range(size)]
        decode = parallel.Decode(world, graphs)

        class _Counting(torch.Tensor):  # counts .to() calls: the moves Decode makes
            pass
        original = torch.Tensor.to

        def counting_to(self, *a, **k):
            if a and isinstance(a[0], torch.device):
                with moved["lock"]:
                    moved["n"] += 1
            return original(self, *a, **k)
        torch.Tensor.to = counting_to
        try:
            for _ in range(3):
                decode(input_ids=torch.tensor([[1]]), cache_position=torch.tensor([0]))
        finally:
            torch.Tensor.to = original
        for g in graphs:
            # two tensors a rank, size ranks: all moved before the first graph of a step ran
            self.assertTrue(all(n % (2 * size) == 0 for n in g.seen), g.seen)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "needs two GPUs")
class DecodeCrossDeviceInputsTest(unittest.TestCase):
    def test_inputs_on_another_ranks_device_do_not_deadlock(self):
        """Rank 0's step waits (a stream memory wait) for a flag rank 1 writes after consuming its
        inputs, which live on rank 0's device: with the inputs moved on the ranks' threads this
        deadlocked (rank 1's copy waited on rank 0's stream, held by the wait); moved first, it
        completes."""
        from .stepgraph import _Driver
        driver = _Driver()
        d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
        flag = torch.zeros(1, dtype=torch.int32, device=d0)
        world = parallel.World([d0, d1])

        def rank0(input_ids=None, cache_position=None):
            # a stream memory wait needs an explicit stream; rank 0's default stream (the one a
            # cross-device copy from this GPU would queue behind) then waits on it, as a captured
            # graph spinning in a sum would hold it
            waiting = torch.cuda.Stream(d0)
            driver.wait_value(waiting, flag.data_ptr(), 1)  # until rank 1 has its inputs
            torch.cuda.current_stream(d0).wait_stream(waiting)
            torch.cuda.synchronize(d0)
            return input_ids

        def rank1(input_ids=None, cache_position=None):
            assert input_ids.device == d1, input_ids.device
            with torch.cuda.device(d0):
                driver.write_value(torch.cuda.Stream(d0), flag.data_ptr(), 1)
            return input_ids
        decode = parallel.Decode(world, [rank0, rank1])
        done = []

        def run():
            # the step's inputs on rank 0's device, as the sharded drafter hands them out
            decode(input_ids=torch.tensor([[7]], device=d0), cache_position=torch.tensor([0], device=d0))
            done.append(True)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(20)
        self.assertTrue(done, "the step did not finish: a rank's input copy waited on a stream held by another rank")


if __name__ == "__main__":
    unittest.main()
