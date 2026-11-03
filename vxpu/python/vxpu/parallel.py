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

import faulthandler
import json
import os
import queue
import time
import threading

import torch


SINGLE_LAUNCHER = os.environ.get("VXPU_SINGLE_LAUNCHER") == "1"  # the A/B of one launcher thread (hero run 15)


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

    def run(self, fns, timeout=None):
        """Run fns[r] on rank r's thread, all at once; return their results in rank order. The ranks'
        threads are made once and fed a function a step (starting and joining four threads cost a
        step about 0.5 ms of host time, hero run 15); an exception on one rank breaks the barrier so
        the others do not wait for it forever."""
        if timeout is None:
            timeout = float(os.environ.get("VXPU_STEP_TIMEOUT", "600"))
        if not getattr(self, "_ran", False):
            # The first step builds the AOTInductor packages its segments lack (aoti.Packages, one
            # build at a time across the ranks): a new artifact, tree or plan rebuilds some three
            # hundred a rank, many minutes in all. That is not a hung step (hero run 15: every
            # "rank did not finish its step" was a first step still building).
            timeout = max(timeout, float(os.environ.get("VXPU_FIRST_STEP_TIMEOUT", "7200")))
            self._ran = True
        self._start_workers()
        for r, fn in enumerate(fns):
            self._inbox[r].put(fn)
        results, errors = [None] * self.size, [None] * self.size
        pending = set(range(self.size))
        deadline = time.perf_counter() + timeout
        while pending:
            try:
                r, ok, value = self._outbox.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                # every thread's stack, so a hang says where each rank sits (a step that never
                # finished during hero run 15's warm-ups)
                print(f"[vxpu] parallel: ranks {sorted(pending)} did not finish the step in {timeout:.0f} s; "
                      "all threads' stacks follow", flush=True)
                faulthandler.dump_traceback(all_threads=True)
                self._dump_flags()
                self._barrier.abort()
                raise RuntimeError("parallel: a rank did not finish its step") from None
            pending.discard(r)
            if ok:
                results[r] = value
            else:
                errors[r] = value
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

    def _dump_flags(self):
        """The one-shot sums' flags and arrival counters of every card, read over a stream of their
        own (the cards' default streams may be spinning in a sum): which site a stuck rank waits at,
        and whether its peers' pushes arrived (hero run 15's hung warm-ups)."""
        try:
            from . import allreduce as ar
            c = ar.comm()
            if c is None:
                print("[vxpu] parallel: no Comm (the sums do not use the one-shot flags)", flush=True)
                return
            for r, dev in enumerate(c.devices):
                with torch.cuda.device(dev), torch.cuda.stream(torch.cuda.Stream(dev)):
                    flags = c.flags[r].to("cpu", non_blocking=False)
                    counters = c.counters[r].to("cpu", non_blocking=False)
                raised = [(int(site), int(rnd), int(slot)) for site, rnd, slot in flags.nonzero().tolist()]
                arrived = [(int(site), int(k), int(counters[site, k])) for site, k in counters.nonzero().tolist()]
                print(f"[vxpu] parallel: card {r} flags raised (site, round, slot) {raised[:24]}; "
                      f"counters (site, kind, n) {arrived[:24]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] parallel: the flag dump failed: {type(e).__name__}: {e}", flush=True)

    def _start_workers(self):
        if getattr(self, "_workers", None):
            return
        self._inbox = [queue.SimpleQueue() for _ in range(self.size)]
        self._outbox = queue.SimpleQueue()

        def work(r):
            self._local.rank = r
            while True:
                fn = self._inbox[r].get()
                if fn is None:
                    return
                try:
                    with torch.cuda.device(self.devices[r]) if self.devices[r].type == "cuda" else _nullcontext():
                        with self.scope() if self.scope is not None else _nullcontext():
                            self._outbox.put((r, True, fn()))
                except BaseException as e:  # noqa: BLE001
                    self._barrier.abort()
                    self._outbox.put((r, False, e))

        self._workers = [threading.Thread(target=work, args=(r,), name=f"vxpu-rank-{r}", daemon=True)
                         for r in range(self.size)]
        for t in self._workers:
            t.start()

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
        # the step trace (VXPU_STEP_TRACE=path): the host's phases of every step, written by write_trace
        self.trace_path = os.environ.get("VXPU_STEP_TRACE") or None
        self.trace = [] if self.trace_path else None

    def forward(self, *args, **kwargs):
        self.steps += 1
        t0 = time.perf_counter()

        def on(r):
            # the inputs are moved here, on the main thread, before any rank launches: a rank moving
            # its own on its thread (tried in hero run 15, worth 0 ms) deadlocks when an input lives
            # on a GPU whose stream is already spinning in a sum launched by another rank program
            # (a cross-device copy waits on the source's stream)
            dev = self.world.devices[r]
            a = [x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x for x in args]
            k = {n: (x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x) for n, x in kwargs.items()}
            return lambda: self.graphs[r](*a, **k)

        fns = [on(r) for r in range(self.world.size)]
        t1 = time.perf_counter()
        if SINGLE_LAUNCHER and all(hasattr(g, "launch") for g in self.graphs):
            # one thread launches every rank's graph, then waits for all: no hand-off to four
            # worker threads and back through the GIL (0.2 ms to start them and 0.24 to join at one
            # row, hero run 15). The first step and the capturing step run eagerly through the
            # ranks' threads (their collectives need every rank on a thread of its own).
            if self.steps > 2 and getattr(self, "_captured", False):
                outs = self._launch_all(args, kwargs)
            else:
                outs = self.world.run(fns)
                self._captured = all(getattr(g, "graph", None) is not None for g in self.graphs)
        else:
            outs = self.world.run(fns)
        if self.trace is not None:
            self.trace.append({"start": t0, "moved": t1, "done": time.perf_counter()})
        return outs[0]

    def _launch_all(self, args, kwargs):
        finishers = []
        for r, dev in enumerate(self.world.devices):
            with torch.cuda.device(dev):
                a = [x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x for x in args]
                k = {n: (x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x) for n, x in kwargs.items()}
                finishers.append((dev, self.graphs[r].launch(*a, **k)))
        outs = []
        for dev, f in finishers:
            with torch.cuda.device(dev):
                outs.append(f() if callable(f) else f)
        return outs

    def write_trace(self, last=None):
        """The step trace as a file (VXPU_STEP_TRACE) and a line: over the last `last` steps (all when
        None), the mean of each host phase in ms. Per step: the inputs moved to the ranks' devices,
        then world.run from the threads' start to their join; per rank: when its thread started after
        that, the input buffers copied, the graph launched (host.go and replay), and the graph done
        (the GPU's time, with its waits on the peers' flags). The ranks' graphs are the step; what
        the trace adds is the host around them."""
        if self.trace is None or not self.trace:
            return None
        steps = self.trace[-last:] if last else self.trace
        n = len(steps)

        def mean(xs):
            return sum(xs) / max(len(xs), 1) * 1e3

        summary = {"steps": n,
                   "step_ms": mean([s["done"] - s["start"] for s in steps]),
                   "inputs_moved_ms": mean([s["moved"] - s["start"] for s in steps]),
                   "world_run_ms": mean([s["done"] - s["moved"] for s in steps]),
                   "ranks": []}
        for r, g in enumerate(self.graphs):
            tr = getattr(g, "trace", None)
            if not tr:
                continue
            rows = tr[-n:]
            pairs = list(zip(rows, steps[-len(rows):]))
            hops = [h for a in rows for h in a.get("hops", [])]
            summary["ranks"].append({
                "rank": r,
                "host_nodes": len(hops) / max(len(rows), 1),
                "host_round_trip_ms": sum(h[3] for h in hops) / max(len(rows), 1),
                "host_work_ms": sum(h[2] for h in hops) / max(len(rows), 1),
                "thread_start_ms": mean([a["in"] - b["moved"] for a, b in pairs]),
                "copy_inputs_ms": mean([a["copied"] - a["in"] for a in rows]),
                "launch_ms": mean([a["launched"] - a["copied"] for a in rows]),
                "graph_ms": mean([a["done"] - a["launched"] for a in rows]),
                "done_after_run_ms": mean([a["done"] - b["moved"] for a, b in pairs]),
                "join_wait_ms": mean([b["done"] - a["done"] for a, b in pairs])})
        with open(self.trace_path, "w") as f:
            json.dump({"summary": summary, "steps": steps,
                       "ranks": [getattr(g, "trace", None) for g in self.graphs]}, f)
        ranks = "; ".join(f"rank {x['rank']}: thread +{x['thread_start_ms']:.2f}, copy {x['copy_inputs_ms']:.2f}, "
                          f"launch {x['launch_ms']:.2f}, graph {x['graph_ms']:.2f} (of which {x['host_nodes']:.0f} host "
                          f"nodes' round trips {x['host_round_trip_ms']:.2f}, the host's work in them "
                          f"{x['host_work_ms']:.2f}), done +{x['done_after_run_ms']:.2f}"
                          for x in summary["ranks"])
        print(f"[vxpu] step trace ({n} steps, ms): step {summary['step_ms']:.2f} = inputs moved "
              f"{summary['inputs_moved_ms']:.2f} + world.run {summary['world_run_ms']:.2f}; {ranks}", flush=True)
        return summary
