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


"""What a lowered call costs, measured: the plan's choice between our
hand-written kernel and a decomposed body the compiler fuses (lower.py).

Why. Fusing a body into its neighbours saves kernel launches and the
round trips of its inputs and outputs through memory, which is all an op
on small tensors costs (rotary, the sampling epilogue, the tiny model's
every op). But inductor turns a decomposed matrix-vector product into
one reduction per output element, which reads the weight again for every
row: at the real model's shapes on an L4 the fused fp8 linear took
290 us where our GEMV took 32 us (wo_b, 5120x8192, one row), and 6.9 ms
at 32 rows. So neither choice is right everywhere, and which one wins
depends on the shapes and the machine: a database's optimizer meets the
same problem and costs its alternatives. Here the costs are measured,
per call site's shapes, on the engine's GPU.

How. Each candidate runs on the call's shapes (the graph's own weights,
random activations), as it would run in the step graph: the fused body
compiled by inductor, the kernel as it launches. It runs in a CUDA graph,
with every tensor argument rotated through more copies than the L2 cache
holds, since in a decode step the weights come from memory. The fused
body is credited one kernel launch (its first operation fuses into what
produces its input). The cheaper one is chosen. Decisions are kept on
disk per GPU, op, candidates and shapes, beside inductor's caches (whose
directory is keyed on our kernels' source, lower.key_cache_dir), so an
engine measures each shape once per machine.
"""

import json
import math
import os
import threading

import torch

_lock = threading.Lock()
_decisions = None  # key -> {"times_us": {name: us}, "chosen": name}
_launch_us = {}  # device name -> one kernel launch in a CUDA graph, us


def _path():
    from .lower import key_cache_dir
    return os.path.join(key_cache_dir(), "vxpu-costs.json")


def _load():
    global _decisions
    if _decisions is None:
        try:
            with open(_path()) as f:
                _decisions = json.load(f)
        except (OSError, ValueError):
            _decisions = {}
    return _decisions


def _save():
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_decisions, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def shapes(values):
    """A call's tensor arguments as a short key: shape and dtype each."""
    out = []
    for v in values:
        if isinstance(v, torch.Tensor):
            out.append("x".join(map(str, v.shape)) + ":" + str(v.dtype).removeprefix("torch."))
        elif isinstance(v, (list, tuple)):
            out.append("[" + ",".join(shapes(v)) + "]")
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out.append(repr(v))
    return out


def choose(op, candidates, values, device):
    """The name of the cheapest of candidates {name: (fn, fused)} for a call with these
    values (fake tensors stand for activations, real ones are the graph's weights), and
    the measured times. fn(*values) computes the call; fused ones are compiled first.
    Exactly one candidate is not fused: the choice when nothing can be measured."""
    name = torch.cuda.get_device_name(device)
    key = "|".join([name, op, ",".join(sorted(candidates))] + shapes(values))
    with _lock:
        known = _load().get(key)
    if known is not None and set(known["times_us"]) == set(candidates):
        return known["chosen"], known["times_us"]
    try:
        times = {n: _measure(fn, values, device, compile_=fused) for n, (fn, fused) in candidates.items()}
    except torch.OutOfMemoryError:  # no room for the copies (a GPU full of weights): no fusion, not recorded
        torch.cuda.empty_cache()
        return next(n for n, (_, fused) in candidates.items() if not fused), {}
    credit = {n: (_launch(device) if fused else 0.0) for n, (_, fused) in candidates.items()}
    chosen = min(times, key=lambda n: times[n] - credit[n])
    with _lock:
        _load()[key] = {"times_us": times, "chosen": chosen}
        _save()
    return chosen, times


def _materialize(v, device):
    """A real tensor for a fake one (random values; integer tensors are zeros)."""
    if not isinstance(v, torch.Tensor):
        return v
    if not isinstance(v, torch._subclasses.fake_tensor.FakeTensor):
        return v
    t = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=device)
    if t.is_floating_point() or t.is_complex():
        t.normal_()
    else:
        t.zero_()
    return t


def _flat(fn, values, device):
    """fn as a function of the call's tensor arguments only, and argument sets for it,
    each tensor its own copy, enough that together they exceed twice the L2 cache: each
    call reads its tensors from memory, as a decode step does."""
    import torch.utils._pytree as pytree
    leaves, spec = pytree.tree_flatten(values)
    leaves = [_materialize(v, device) for v in leaves]
    at = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]

    def flat(*tensors):
        vals = list(leaves)
        for i, t in zip(at, tensors):
            vals[i] = t
        return fn(*pytree.tree_unflatten(vals, spec))

    first = [leaves[i] for i in at]
    size = sum(t.numel() * t.element_size() for t in first) or 1
    l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 64 << 20)
    n = max(2, min(64, math.ceil(2 * l2 / size)))
    return flat, [first] + [[t.clone() for t in first] for _ in range(n - 1)]


def _replay_us(fn, sets, reps=5):
    """Microseconds per call of fn, its calls over every argument set captured in one
    CUDA graph and replayed."""
    for s in sets[:2]:
        fn(*s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for s in sets:
            fn(*s)
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1000 / (reps * len(sets)), 2)


def _measure(fn, values, device, compile_):
    """fn's time per call on values, compiled by inductor when compile_ (traced first, so
    each candidate is its own graph to dynamo, under no limit on recompiles)."""
    from torch.fx.experimental.proxy_tensor import make_fx
    with torch.cuda.device(device):
        flat, sets = _flat(fn, values, device)
        if compile_:
            flat = torch.compile(make_fx(flat)(*sets[0]), dynamic=False)
        return _replay_us(flat, sets)


def _launch(device):
    """One kernel launch in a CUDA graph, in microseconds (a one-element add)."""
    name = torch.cuda.get_device_name(device)
    if name not in _launch_us:
        with torch.cuda.device(device):
            x = torch.zeros(1, device=device)
            _launch_us[name] = _replay_us(lambda t: t.add_(1), [(x,)] * 100)
    return _launch_us[name]
