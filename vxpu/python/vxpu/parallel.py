# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ranks of a tensor-parallel step in one process (hero run 14).

Under dense_tp = N the engine loads N programs of the same step, one per
rank, each holding its slice of every dense weight (the fork's column- and
row-parallel linears, its vocabulary-sharded embedding and head), and runs
them together: where the fork called torch.distributed, the programs
carry vxpu::tp_allreduce and vxpu::tp_allgather (ops.py). This module is the
world those calls are answered by.

Eagerly (the dataflow runner, the first step before a capture, the CPU
selftest) the ranks run in threads, one per rank, and a collective is a
barrier: each rank leaves its partial in a slot, the last to arrive is
the signal, every rank sums the slots in rank order (so the totals are
bit-identical across ranks) and leaves again. In the captured step the
same exchange is peer stores, flags and a local sum inside each rank's
CUDA graph (stepgraph.py); the inboxes those stores land in are allocated
here, once, at fixed addresses.
"""

import threading

import torch


class World:
    """N ranks of one step. `devices` is the rank's device per rank."""

    def __init__(self, devices, scope=None):
        self.size = len(devices)
        self.devices = [torch.device(d) for d in devices]
        self.scope = scope  # a context manager factory every rank's thread enters (the engine's external scope)
        self._local = threading.local()
        self._barrier = threading.Barrier(self.size)
        self._slots = [None] * self.size
        self._lock = threading.Lock()
        self._failed = None
        # the captured step's exchange buffers: per collective index k and destination rank j, the
        # partials every other rank writes there ({src rank: tensor on j's device}) and the flags
        # j waits on ({src rank: int32 [1] on j's device}); made the first time a rank asks
        self.inboxes = {}
        self.flags = {}
        self.counters = {}  # (k, src rank, dst rank): the ring write's program counter on src's device

    # ---- eager: threads and a barrier ----

    def rank(self):
        r = getattr(self._local, "rank", None)
        if r is None:
            raise RuntimeError("parallel: a collective ran on a thread that is no rank of the world")
        return r

    def run(self, fns, timeout=600.0):
        """Run fns[r] on rank r's thread, all at once; return their results in rank order. An
        exception on one rank breaks the barrier so the others do not wait for it forever."""
        results, errors = [None] * self.size, [None] * self.size

        def work(r):
            self._local.rank = r
            try:
                with torch.cuda.device(self.devices[r]) if self.devices[r].type == "cuda" else _nullcontext():
                    with self.scope() if self.scope is not None else _nullcontext():
                        results[r] = fns[r]()
            except BaseException as e:  # noqa: BLE001
                errors[r] = e
                self._barrier.abort()
            finally:
                self._local.rank = None

        threads = [threading.Thread(target=work, args=(r,), name=f"vxpu-rank-{r}") for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout)
        if any(t.is_alive() for t in threads):
            self._barrier.abort()
            raise RuntimeError("parallel: a rank did not finish its step")
        if self._barrier.broken:
            self._barrier.reset()
        failed = [(r, e) for r, e in enumerate(errors) if e is not None]
        if failed:
            # the rank that failed first is the cause; the others only saw the barrier break
            root = [(r, e) for r, e in failed if "collective is abandoned" not in str(e)
                    and "failed at a barrier" not in str(e)] or failed
            for r, e in failed:
                print(f"[vxpu] parallel: rank {r} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            raise root[0][1]
        return results

    def barrier(self):
        """Every rank arrives before any leaves (the captures' preparation)."""
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            raise RuntimeError("parallel: a rank failed at a barrier") from None

    def _exchange(self, x):
        """Every rank's x, in rank order, after all have arrived (the eager collective's core)."""
        r = self.rank()
        # a copy, not x itself: the caller overwrites x with the result (the fork's all_reduce is in
        # place) as soon as it returns, while the other ranks may still be reading its part
        self._slots[r] = x.detach().clone()
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            raise RuntimeError("parallel: a rank failed; the collective is abandoned") from None
        gathered = list(self._slots)
        try:
            self._barrier.wait()  # every rank has taken the slots before any writes the next ones
        except threading.BrokenBarrierError:
            raise RuntimeError("parallel: a rank failed; the collective is abandoned") from None
        return gathered

    def allreduce(self, x):
        parts = self._exchange(x)
        total = parts[0].to(x.device, torch.float32).clone()
        for p in parts[1:]:  # rank order on every rank: the totals agree bit for bit
            total += p.to(x.device, torch.float32)
        return total.to(x.dtype)

    def allgather(self, x, world_size):
        if world_size != self.size:
            raise ValueError(f"parallel: allgather of {world_size} shards in a world of {self.size}")
        parts = self._exchange(x)
        return torch.cat([p.to(x.device) for p in parts], dim=-1)

    # ---- captured: fixed exchange buffers per collective ----

    def buffers(self, k, like):
        """The exchange buffers of collective k, made once for every rank: inboxes {(src, dst): a
        tensor shaped like `like` on dst's device, where src's partial (or shard) lands}, flags
        {(src, dst): int32 [1] on dst's device, raised by src's ring write and waited on by dst} and
        counters {(src, dst): int32 [1] on src's device, the ring write's program counter}. The
        local sum (or concatenation) reads the inboxes in rank order, so every rank's total is
        bit-identical."""
        with self._lock:
            if k not in self.inboxes:
                pairs = [(i, j) for i in range(self.size) for j in range(self.size) if i != j]
                self.inboxes[k] = {(i, j): torch.zeros(like.shape, dtype=like.dtype, device=self.devices[j])
                                   for i, j in pairs}
                self.flags[k] = {(i, j): torch.zeros(1, dtype=torch.int32, device=self.devices[j]) for i, j in pairs}
                self.counters[k] = {(i, j): torch.zeros(1, dtype=torch.int32, device=self.devices[i]) for i, j in pairs}
            return self.inboxes[k], self.flags[k], self.counters[k]


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Decode(torch.nn.Module):
    """The tensor-parallel decode step: the ranks' step graphs (or dataflow runners) run together,
    each on its own thread and device, on the same inputs; the logits come from rank 0 (every rank
    holds the gathered logits after the head). Called as the engine calls a decode program:
    forward(input_ids=..., cache_position=...)."""

    def __init__(self, world, graphs):
        super().__init__()
        self.world = world
        self.graphs = graphs
        self.steps = 0

    def forward(self, *args, **kwargs):
        self.steps += 1

        def on(r):
            dev = self.world.devices[r]
            a = [x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x for x in args]
            k = {n: (x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x) for n, x in kwargs.items()}
            return lambda: self.graphs[r](*a, **k)

        outs = self.world.run([on(r) for r in range(self.world.size)])
        return outs[0]
