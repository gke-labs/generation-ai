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


"""Hot experts on the GPUs, computed alongside the cold ones on the CPU.

A decode step's dense work (GPU) and routed experts (CPU) cannot overlap
for one sequence: each layer's experts need its attention output, and the
next layer needs theirs. But a layer's experts can be split. Expert use is
skewed (#317: on real traffic the hottest quarter of each layer's experts
carry ~80% of decode selections), so the hottest experts are kept on the
GPUs, in the checkpoint's packed fp4 with their scales, and computed there
(triton_ops.fp4_experts_decode) while the CPU computes the rest with its
kernels; the partial sums are added. The CPU's share of the step falls to
its share of the selections.

Where each expert lives is a layer's Residency: a pool of slots on each
GPU that holds experts (the engine's own device first, so the hottest
need no copy between devices), and the host banks for everything else.
Every expert is in at most one pool; a pool's slot map says, per expert,
its slot there or -1. The kernels take that map as it is, so a pool on
another GPU runs the same code: the token's activations go there (a few
KB), its partial sum comes back.

Which experts start hot is a placement, from the engine's own statistics
(`vxpu stats --records` into `python -m vxpu.expert_stats
--placement-gb`), named by the plan's gpu_experts choice
(VXPU_GPU_EXPERTS). This is the first placement chosen from the
statistics the engine keeps (#316, #348).

With gpu_experts_moves, the residency then follows the traffic. Every
layer scores its experts as the offline simulator found best
(expert_sim.py, #369): their selections over all the traffic so far
(seeded with the placement's counts), mixed with the current request's
own, prefill included, each normalized; a request is concentrated, and
its prompt says much about its decode. When a cold expert outscores the
coldest one held, they swap: the held one is unmapped (the CPU computes
it from the next call), a worker thread copies the newcomer's rows from
the host banks into the freed slot on a copy stream, and once that copy
is done the newcomer is mapped. Unmapping, copying and mapping are
ordered on the pool's stream, so a kernel never reads a slot mid-copy;
while an expert is in flight, the CPU computes it. A layer plans its
swaps once per request, on its prefill call: the copies run while the
prefill computes on the CPU (seconds) and are done before decode starts.
Moving during decode as well cost DeepSeek more than it gained (copies
competing for host memory and PCIe with the step, ~2 swaps a step), and
the simulator credits the gain to the prompt. The number in flight is
bounded (an expert is ~19 MB on DeepSeek). Off,
the results depend only on the placement; on, they also depend on the
traffic before (the split between GPU and CPU changes the summation
order: the rounding class).
"""

from concurrent.futures import ThreadPoolExecutor
import json
import queue
import threading
import weakref

import torch


class Pool:
    """One GPU's slots for one layer: the held experts' banks (packed fp4
    and e8m0 scales, rows gathered from the host banks), which expert each
    slot holds, and the slot of every expert (-1 where it is not held
    here), on the host and on this GPU."""

    def __init__(self, moe, expert_ids, device):
        ids = sorted({int(e) for e in expert_ids})
        self.device = torch.device(device)
        table = getattr(moe, "expert_table", None)
        if table is not None:
            P = len(ids)
            I, K = table.intermediate_size, table.dim
            s1_dtype = table.w1_scale_dtype
            s2_dtype = table.w2_scale_dtype
            s3_dtype = table.w3_scale_dtype
            self.w1 = torch.empty((P, I, K // 2), dtype=torch.uint8, device=self.device)
            self.s1 = torch.empty((P, I, K // 32), dtype=s1_dtype, device=self.device)
            self.w3 = torch.empty((P, I, K // 2), dtype=torch.uint8, device=self.device)
            self.s3 = torch.empty((P, I, K // 32), dtype=s3_dtype, device=self.device)
            self.w2 = torch.empty((P, K, I // 2), dtype=torch.uint8, device=self.device)
            self.s2 = torch.empty((P, K, I // 32), dtype=s2_dtype, device=self.device)

            def copy_one(slot_e):
                slot, e = slot_e
                local = e - moe.experts_start_idx
                self.w1[slot].copy_(table.tensor(local, "w1"), non_blocking=True)
                self.s1[slot].copy_(table.tensor(local, "w1_scale"), non_blocking=True)
                self.w3[slot].copy_(table.tensor(local, "w3"), non_blocking=True)
                self.s3[slot].copy_(table.tensor(local, "w3_scale"), non_blocking=True)
                self.w2[slot].copy_(table.tensor(local, "w2"), non_blocking=True)
                self.s2[slot].copy_(table.tensor(local, "w2_scale"), non_blocking=True)

            if len(ids) > 1:
                with ThreadPoolExecutor(max_workers=min(16, len(ids))) as executor:
                    list(executor.map(copy_one, enumerate(ids)))
            elif len(ids) == 1:
                copy_one((0, ids[0]))
            torch.cuda.synchronize(self.device)
        else:
            idx = torch.tensor(ids, dtype=torch.int64)
            def held(t):
                return t.index_select(0, idx - moe.experts_start_idx).contiguous().to(device)
            self.w1, self.s1 = held(moe.w1), held(moe.w1_scale)
            self.w3, self.s3 = held(moe.w3), held(moe.w3_scale)
            self.w2, self.s2 = held(moe.w2), held(moe.w2_scale)
        slot = torch.full((moe.n_local_experts,), -1, dtype=torch.int64)
        if ids:
            slot[torch.tensor(ids, dtype=torch.int64) - moe.experts_start_idx] = torch.arange(len(ids))
        self.experts = ids  # slot -> expert
        self.slot_cpu = slot
        self.slot_gpu = slot.to(device)
        self.bytes = sum(t.numel() * t.element_size() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))
        self.stream = _side_stream(self.device)

    @property
    def count(self):
        return len(self.experts)


class Residency:
    """Where one layer's experts live: pools on GPUs, the host banks for
    the rest. held_cpu[e] is True where expert e (local index) is in a
    pool, so the CPU skips it."""

    def __init__(self, moe, by_device):
        self.pools = [Pool(moe, ids, dev) for dev, ids in by_device if ids]
        held = torch.zeros(moe.n_local_experts, dtype=torch.bool)
        for pool in self.pools:
            seen = held & (pool.slot_cpu >= 0)
            if seen.any():
                raise ValueError(f"experts {(seen.nonzero().view(-1) + moe.experts_start_idx).tolist()} "
                                 "are placed on more than one GPU")
            held |= pool.slot_cpu >= 0
        self.held_cpu = held

        self.start = moe.experts_start_idx
        self.moe = moe
        self.mover = None  # set by Mover.attach
        self.held = {e for pool in self.pools for e in pool.experts}  # global ids, for the scoring's lookups

    @property
    def count(self):
        return sum(p.count for p in self.pools)

    @property
    def bytes(self):
        return sum(p.bytes for p in self.pools)


# A cold expert replaces the coldest held one when its score is above
# RATIO times the held one's and the current request selected it at least
# EVIDENCE times: hysteresis against swapping back and forth. Without the
# placement's counts, held experts start with PRIOR selections.
RATIO, EVIDENCE, PRIOR = 1.25, 2, 8.0
MAX_PLANNED = 32  # swaps a layer queues per plan


class Mover:
    """Moves experts between the host banks and the GPU pools as the hot
    set drifts (the module docstring): every attached layer's scores and
    queued swaps, the bound on moves in flight, and the worker thread that
    copies."""

    def __init__(self, in_flight, request_weight):
        self.in_flight = in_flight
        self.request_weight = request_weight
        self.layers = []
        self.pending = []  # (res, pool, slot, expert, copied, done): every layer's moves in flight
        self.queued = []  # layers with swaps planned and not yet started
        self.active = 0
        self.started = 0
        self.completed = 0
        self.work = queue.Queue()
        # the worker holds the queue and a weak reference only: the mover
        # (and through it the pools and banks) is freed with its engine,
        # and the worker then exits
        threading.Thread(target=_copy_loop, args=(weakref.ref(self), self.work), name="vxpu-expert-moves",
                         daemon=True).start()

    def attach(self, res, counts=None):
        """Score res's experts from counts ({expert: selections}, the
        placement's history), or PRIOR for each held expert."""
        n = res.moe.n_local_experts
        res.mover = self
        res.long = [0.0] * n  # selections over all the traffic so far
        if counts:
            for e, c in counts.items():
                res.long[int(e) - res.start] = float(c)
        else:
            for e in res.held:
                res.long[e - res.start] = PRIOR
        res.long_total = sum(res.long)
        res.incoming = set()  # experts on their way to a slot
        self._reset(res)
        self.layers.append(res)

    def begin_request(self):
        """A new request: its own selections start from nothing."""
        for res in self.layers:
            self._reset(res)

    def _reset(self, res):
        res.req = [0] * res.moe.n_local_experts
        res.req_total = 0
        res.calls = 0
        res.queued = []  # (pool, slot, victim, newcomer) planned

    def observe(self, res, experts):
        """After a layer's call (prefill or decode): count its selections
        (experts, global ids), finish every layer's moves whose copies are
        done, plan this layer's on its first call of the request (the
        prefill), start what the bound allows from any layer's (a prefill
        takes each layer once: the moves it plans run while the layers
        after it compute). Returns (started, completed)."""
        res.calls += 1
        for e in experts:
            res.long[e - res.start] += 1.0
            res.req[e - res.start] += 1
        res.long_total += len(experts)
        res.req_total += len(experts)
        completed = self._finish()
        if res.calls == 1:
            self._plan(res)
        return self._start(), completed

    def _plan(self, res):
        w, lt, rt = self.request_weight, res.long_total or 1.0, res.req_total or 1
        score = [(1 - w) * a / lt + w * b / rt for a, b in zip(res.long, res.req)]
        cold = sorted((l for l in range(len(score)) if res.req[l] >= EVIDENCE
                       and l + res.start not in res.held and l + res.start not in res.incoming),
                      key=lambda l: -score[l])
        held = sorted(((score[e - res.start], slot, pool, e)
                       for pool in res.pools for slot, e in enumerate(pool.experts) if e >= 0), key=lambda h: h[0])
        res.queued = []
        for l, (s_held, slot, pool, victim) in zip(cold[:MAX_PLANNED], held):
            if score[l] <= RATIO * s_held:
                break
            res.queued.append((pool, slot, victim, l + res.start))
        if res.queued and res not in self.queued:
            self.queued.append(res)

    def _start(self):
        """Start queued swaps, oldest plans first, up to the bound. Every
        layer's pools on a device share its side stream, so another
        layer's unmapping is ordered after that layer's last kernels."""
        started = 0
        while self.queued and self.active < self.in_flight:
            res = self.queued[0]
            if not res.queued:
                self.queued.pop(0)
                continue
            pool, slot, victim, new = res.queued.pop(0)
            if pool.experts[slot] != victim or new in res.held or new in res.incoming:
                continue  # stale: an earlier swap changed it
            # unmap: the CPU computes the victim from the next call on; the
            # pool's stream orders the unmapping after this call's kernels
            lv = victim - res.start
            pool.experts[slot] = -1
            pool.slot_cpu[lv] = -1
            res.held_cpu[lv] = False
            res.held.discard(victim)
            with torch.cuda.stream(pool.stream):
                pool.slot_gpu[lv] = -1
                unmapped = torch.cuda.Event()
                unmapped.record(pool.stream)
            copied, done = threading.Event(), torch.cuda.Event()
            self.pending.append((res, pool, slot, new, copied, done))
            res.incoming.add(new)
            self.active += 1
            self.started += 1
            started += 1
            self.work.put((res.moe, pool, slot, new - res.start, unmapped, copied, done))
        return started

    def _finish(self):
        completed = 0
        for move in list(self.pending):
            res, pool, slot, expert, copied, done = move
            if not (copied.is_set() and done.query()):  # an event not yet recorded reads as done
                continue
            le = expert - res.start
            with torch.cuda.stream(pool.stream):
                pool.stream.wait_event(done)
                pool.slot_gpu[le] = slot
            pool.slot_cpu[le] = slot
            pool.experts[slot] = expert
            res.held_cpu[le] = True
            res.held.add(expert)
            res.incoming.discard(expert)
            self.pending.remove(move)
            self.active -= 1
            self.completed += 1
            completed += 1
        return completed


def _copy_loop(mover, work):
    """The mover's worker: copies queued swaps' rows from the host banks
    into their slots, on a copy stream per device. Exits once the mover is
    gone (it is only weakly referenced) and nothing is queued."""
    streams = {}
    while True:
        try:
            item = work.get(timeout=1.0)
        except queue.Empty:
            if mover() is None:
                return
            continue
        moe, pool, slot, local, unmapped, copied, done = item
        key = str(pool.device)
        if key not in streams:
            streams[key] = torch.cuda.Stream(device=pool.device)
        stream = streams[key]
        table = getattr(moe, "expert_table", None)
        with torch.cuda.stream(stream):
            stream.wait_event(unmapped)  # no kernel reads the slot after this
            if table is not None:
                pool.w1[slot].copy_(table.tensor(local, "w1"))
                pool.s1[slot].copy_(table.tensor(local, "w1_scale"))
                pool.w3[slot].copy_(table.tensor(local, "w3"))
                pool.s3[slot].copy_(table.tensor(local, "w3_scale"))
                pool.w2[slot].copy_(table.tensor(local, "w2"))
                pool.s2[slot].copy_(table.tensor(local, "w2_scale"))
            else:
                for dst, src in ((pool.w1, moe.w1), (pool.s1, moe.w1_scale), (pool.w3, moe.w3),
                                 (pool.s3, moe.w3_scale), (pool.w2, moe.w2), (pool.s2, moe.w2_scale)):
                    dst[slot].copy_(src[local])  # pageable host rows: the driver stages them, on this thread
            done.record(stream)
        copied.set()
        item = moe = pool = None  # hold nothing of the engine's while idle


def verify(res):
    """Every held expert's rows on its GPU equal the host banks' rows, and
    the slot maps agree with each other: True, or raises. For tests."""
    moe = res.moe
    table = getattr(moe, "expert_table", None)
    for pool in res.pools:
        for slot, e in enumerate(pool.experts):
            if e < 0:
                continue
            le = e - res.start
            assert int(pool.slot_cpu[le]) == slot and int(pool.slot_gpu[le]) == slot and bool(res.held_cpu[le]), \
                f"expert {e}: slot maps disagree"
            if table is not None:
                for dst, name in ((pool.w1, "w1"), (pool.s1, "w1_scale"), (pool.w3, "w3"),
                                  (pool.s3, "w3_scale"), (pool.w2, "w2"), (pool.s2, "w2_scale")):
                    assert torch.equal(dst[slot].cpu().view(torch.uint8), table.tensor(le, name).view(torch.uint8)), \
                        f"expert {e}: rows on {pool.device} differ from the host's"
            else:
                for dst, src in ((pool.w1, moe.w1), (pool.s1, moe.w1_scale), (pool.w3, moe.w3),
                                 (pool.s3, moe.w3_scale), (pool.w2, moe.w2), (pool.s2, moe.w2_scale)):
                    assert torch.equal(dst[slot].cpu().view(torch.uint8), src[le].view(torch.uint8)), \
                        f"expert {e}: rows on {pool.device} differ from the host's"
    held = sum(p.slot_cpu >= 0 for p in res.pools)
    assert torch.equal(held.bool(), res.held_cpu), "held_cpu disagrees with the pools"
    return True


_streams = {}


def _side_stream(device):
    """One side stream per device for every layer's pool there (layers run
    one after another, so they never contend)."""
    key = str(device)
    if key not in _streams:
        _streams[key] = torch.cuda.Stream(device=device)
    return _streams[key]


def devices_for(engine_device):
    """The GPUs a placement's pools go to, in order: the engine's own
    device, then every other visible GPU by index."""
    first = torch.device(engine_device)
    if first.index is None:
        first = torch.device("cuda", torch.cuda.current_device())
    return [first] + [torch.device("cuda", i) for i in range(torch.cuda.device_count()) if i != first.index]


def _placement_doc(spec):
    """A placement: inline JSON (as the coordinator sends it) or a file."""
    if spec.lstrip().startswith("{"):
        return json.loads(spec)
    with open(spec) as f:
        return json.load(f)


def load_placement(spec):
    """[{layer: [expert ids]}] per GPU, in devices_for's order, from a
    placement (expert_stats --placement-gb), inline or a file: format
    vxpu-placement/2 lists "gpus"; /1 is one GPU's "layers"."""
    doc = _placement_doc(spec)
    gpus = doc["gpus"] if "gpus" in doc else [doc]
    return [{int(layer): ids for layer, ids in gpu["layers"].items()} for gpu in gpus]


def load_counts(spec):
    """{layer: {expert: selections}}, the history a placement was chosen
    from (its "counts"), or {} when it carries none."""
    doc = _placement_doc(spec)
    return {int(layer): c for layer, c in (doc.get("counts") or {}).items()}
