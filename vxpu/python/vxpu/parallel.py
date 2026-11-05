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
import functools
import os
import queue
import time
import threading

import torch


# one thread launches the four ranks' captured graphs (hero run 15: 9.7 -> 9.4 ms at one row on pod A, 9.9 -> 9.6
# on pod B, 12.0 -> 11.8 at three rows); VXPU_SINGLE_LAUNCHER=0 is the A/B back to a thread per rank
SINGLE_LAUNCHER = os.environ.get("VXPU_SINGLE_LAUNCHER", "1") == "1"
# VXPU_LAUNCH_THREADS=1 launches the captured graphs from the rank threads at once instead of the one thread
# launching them in turn. Tried for hero run 15's many-row account, whose profiles showed the one thread's four
# cudaGraphLaunch calls at 1.4 to 1.8 ms each and card 0 waiting 4.6 ms at the step's first sum site: the threads
# changed nothing at 33 and 40 rows (33.9 / 36.2 against 33.6 / 35.9) and cost 0.2 to 0.3 ms at one to five rows
# (9.5 / 10.8 / 12.0 / 13.1 / 14.0 against 9.3 / 10.5 / 11.7 / 12.7 / 13.7), the thread hand-off the single launcher
# had removed; so the one thread stays the default and the launch calls' host time goes into the step trace.
LAUNCH_THREADS = os.environ.get("VXPU_LAUNCH_THREADS", "0") == "1"


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
        self._shared = {}  # share(): what every rank holds for a key, in rank order

    def reset_rings(self):
        """Every ring write's flag and counter back to zero (after steps nobody consumed: the prime)."""
        for d in (self.flags, self.counters):
            for pairs in d.values():
                for t in pairs.values():
                    t.zero_()
        for dev in self.devices:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

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
        if getattr(self, "solo", False):  # one rank stepping alone (engine._tp_prime): nothing to wait for
            return
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

    def register_shard(self, key, rank, x):
        """Rank `rank`'s tensor for `key` (a tensor the captured step reads across the ranks for good,
        such as the embedding's shard for vxpu::tp_embed_rows), registered when the rank's graph is
        built, before any prime, so a capture of any rank, alone or in step, finds every rank's."""
        with self._lock:
            self._shared.setdefault(key, [None] * self.size)[rank] = x

    def shards_of(self, key):
        """Every rank's tensor for `key`, in rank order; None where a rank has not registered."""
        with self._lock:
            return list(self._shared.get(key, [None] * self.size))

    def embed_rows(self, ids, shard, start):
        """vxpu::tp_embed_rows eagerly: this rank's rows of the table, zero for the ids it does not
        own, summed across the ranks (the fork's ParallelEmbedding); a rank stepping alone keeps
        its partial (wrong rows, a warm-up), as allreduce does."""
        import torch.nn.functional as F
        n = shard.shape[0]
        local = ids - start
        mask = (local < 0) | (local >= n)
        y = F.embedding(local.masked_fill(mask, 0), shard)
        y = y.masked_fill(mask.unsqueeze(-1), 0)
        return self.allreduce(y).to(shard.dtype)

    def allreduce(self, x):
        if getattr(self, "solo", False):  # one rank stepping alone: its partial stands for the sum (wrong values, a warm-up)
            return x.clone()
        parts = self._exchange(x)
        total = parts[0].to(x.device, torch.float32).clone()
        for p in parts[1:]:  # rank order on every rank: the totals agree bit for bit
            total += p.to(x.device, torch.float32)
        return total.to(x.dtype)

    def allgather(self, x, world_size):
        if world_size != self.size:
            raise ValueError(f"parallel: allgather of {world_size} shards in a world of {self.size}")
        if getattr(self, "solo", False):
            return torch.cat([x] * self.size, dim=-1)
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
        from . import triton_ops
        triton_ops.mx_memo_clear()  # the producer-side MXFP8 fold's memo is a step's

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
            entry = {"start": t0, "moved": t1, "done": time.perf_counter()}
            stamps = getattr(self, "_launch_stamps", None)
            if stamps and all(stamps):  # the captured step: each rank's launch call, start and end, on the host
                entry["launches"] = [[a - t0, b - t0] for a, b in stamps]
                self._launch_stamps = None
            self.trace.append(entry)
        return outs[0]

    def graph_token_ready(self):
        """Whether every rank's captured graph feeds itself (stepgraph.GRAPH_TOKEN), so a step may be
        relaunched with no inputs from the host."""
        return (SINGLE_LAUNCHER and self.steps > 2 and getattr(self, "_captured", False)
                and all(getattr(g, "graph", None) is not None and getattr(g, "_graph_token", False) for g in self.graphs))

    def captured_ready(self):
        """Whether every rank's graph is captured and the single launcher is on, so the step may be
        relaunched on input buffers the caller writes (relaunch_raw)."""
        return (SINGLE_LAUNCHER and self.steps > 2 and getattr(self, "_captured", False)
                and all(getattr(g, "graph", None) is not None for g in self.graphs))

    def relaunch_raw(self):
        """Every rank's graph replayed on its own input buffers (StepGraph.input_buffers, written by
        the caller on the stream), nothing waited for: returns the ranks' finish halves with their
        devices."""
        self.steps += 1
        from . import triton_ops
        triton_ops.mx_memo_clear()
        return list(zip(self.world.devices, self._relaunch_all(force=True)))

    def _relaunch_all(self, force=False):
        """Every rank's graph replayed, in the ranks' threads when LAUNCH_THREADS (hero run 15: a
        cudaGraphLaunch of the step graph costs 1.3 to 3.6 ms of host time at 8 to 40 rows, so four
        launches from one thread start rank 3's step 4 to 12 ms after rank 0's, which the first sum
        site of the step turns into rank 0's wait: 6 ms of the 26.4 at 33 rows; from four threads
        the stagger is one launch's skew), else one after another from this thread."""
        def launch(r, dev):
            with torch.cuda.device(dev):
                return self.graphs[r].relaunch(force=force)
        if LAUNCH_THREADS:
            return self.world.run([functools.partial(launch, r, dev) for r, dev in enumerate(self.world.devices)])
        return [launch(r, dev) for r, dev in enumerate(self.world.devices)]

    def relaunch(self):
        """Every rank's graph replayed on its own inputs, nothing waited for: returns the ranks'
        finish halves, each with its device, and rank 0's chosen token ids (on rank 0's device,
        written by the graph; the caller copies them out behind the step)."""
        self.steps += 1
        from . import triton_ops
        triton_ops.mx_memo_clear()
        return list(zip(self.world.devices, self._relaunch_all())), self.graphs[0]._chosen

    def _launch_all(self, args, kwargs):
        # every rank's inputs moved before any rank's graph is launched: a rank's inputs may live on
        # another rank's GPU (the sharded drafter's, on rank 0's), and a cross-device copy after
        # that GPU's graph has started waits on a stream spinning in a sum (parallel_test)
        moved = []
        for dev in self.world.devices:
            a = [x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x for x in args]
            k = {n: (x.to(dev, non_blocking=False) if isinstance(x, torch.Tensor) else x) for n, x in kwargs.items()}
            moved.append((a, k))
        stamps = [None] * self.world.size  # per rank: the launch call's start and end on the host (the trace)

        def launch(r, dev):
            t0 = time.perf_counter()
            with torch.cuda.device(dev):
                a, k = moved[r]
                f = self.graphs[r].launch(*a, **k)
            stamps[r] = (t0, time.perf_counter())
            return f

        if LAUNCH_THREADS:
            launched = self.world.run([functools.partial(launch, r, dev) for r, dev in enumerate(self.world.devices)])
            finishers = list(zip(self.world.devices, launched))
        else:
            finishers = [(dev, launch(r, dev)) for r, dev in enumerate(self.world.devices)]
        outs = []
        for dev, f in finishers:
            with torch.cuda.device(dev):
                outs.append(f() if callable(f) else f)
        if self.trace is not None:
            self._launch_stamps = stamps
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
