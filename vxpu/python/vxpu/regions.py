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


"""Regions: the decode graph cut where a single value carries the state,
and each region's measured time against its bandwidth roofline.

Why. The planner chooses implementations for pieces of the graph: our
kernels, a compiler's fusion, or a kernel that replaces a whole piece
(costs.py). Choosing over every way to cut the graph is a combinatorial
search. But a model's step has natural cut points, where its state is one
tensor: in a transformer the residual stream between blocks (for DeepSeek,
the hyper-connection stream [rows, 1, 4, dim]). Inside a block the stream
stays live beside the block's own intermediates (q, the KV window, the
attention scores), so live state there is two to six times larger. Pieces
between those cuts can be costed independently, and the step's cost is
the sum. And since decode is memory-bound, each region has a floor: the
bytes it must read over the GPU's bandwidth. How far each region is from
that floor says where a better kernel would pay.

What. A boundary between two nodes is a cut when the activations live
across it total at most CUT_RATIO times the largest of them (one value
carries the state, and everything else live is small: positions, the
indexer's top-k rows), and no boundary within WINDOW nodes of it has less
live (so a large intermediate alone, such as a copy of a whole cache, is
not a cut, and neither is the tail of a block where only its output and
the stream are live). Boundaries with nearly nothing live (the step's
start and end: token ids) are not cuts and do not veto one. Values
live across more than LONG_LIVED nodes do not count: they
cross every cut in their range, so they say nothing about where to cut
(the drafter keeps each target layer's hidden state, a quarter of the
stream, until the end of the step). Host ops are always cuts: they are
where the step crosses to another device. Regions of fewer than MIN_NODES
nodes join the region after them, or the one before a host op. On the
shaped DeepSeek (fixtures.tiny_deepseek_v41 --shaped) the cuts fall after
each block's hc_post and around the engram layer, and nowhere inside an
attention block; the last block's is missed (the drafter's stages after
it hold less live than the stream), so the head's region holds that
block's shared expert too. Each region is named for the deepest module its nodes
share (the export's nn_module_stack), such as layers.2.attn.

Measured (profile). Each dense region is compiled by inductor as the
lowering left it, captured in a CUDA graph, and the regions replay in
step order with events between them, so every weight comes from memory
as in a step. The roofline is the bytes a region reads (its weights and
caches; a table only gathered from counts its rows; an fp8 weight counts
twice where the plan reads its bf16 copy) over the GPU's measured copy
bandwidth. The shaped DeepSeek, five layers, on an L4 (231 GB/s copy):

    rows   dense regions   the head           attention blocks
    1      9.8 ms          5.2 ms (114%)      89-98% of roofline
    8      50.1 ms         40.6 ms (15%)      89-96%, of twice the fp8 bytes
    32     25.1 ms         12.1 ms (50%)      74-87%, of twice the fp8 bytes

At one row every region that reads weights is at the roofline. At 8 rows
the head's kernel reads its 1.3 GB weight once per row, and the fp8
linears read bf16 copies: the step is bound by two kernel shapes, a
matrix-vector product over several rows that reads the weight once, for
bf16 and for block-scaled fp8 weights. The small regions between (the
hyper-connection mixes, the gate: 2 MB of weights, 13-19 kernels) are
launch-bound at 11-23%, about 80 us a layer.

    python -m vxpu.regions ARTIFACT_DIR [--cuts]
"""

import collections
import sys
import time

import torch
import torch.utils._pytree as pytree
from torch.fx.passes.split_module import split_module

from .segments import HOST_OPS

CUT_RATIO = 1.5
WINDOW = 32
LONG_LIVED = 500
MIN_NODES = 8
TRIVIAL = 4096  # bytes

GATHERS = ("aten.index.Tensor", "aten.embedding.default", "aten.index_select.default")


def _nbytes(node):
    val = node.meta.get("val")
    return sum(t.numel() * t.element_size() for t in pytree.tree_leaves(val) if isinstance(t, torch.Tensor))


def _is_host(node):
    return node.op == "call_function" and str(node.target) in HOST_OPS


def _source(node):
    lines = node.meta.get("stack_trace", "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def live(gm):
    """[(node, live bytes, largest live value's bytes)] for the boundary after each compute node:
    the activations defined at or before it and used after it, those live across more than
    LONG_LIVED nodes left out."""
    nodes = list(gm.graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    last = {n: max((index[u] for u in n.users), default=index[n]) for n in nodes if n.op == "call_function"}
    ends = collections.defaultdict(list)
    for n, i in last.items():
        ends[i].append(n)
    alive, out = {}, []
    for i, n in enumerate(nodes):
        if i < last.get(n, i) <= i + LONG_LIVED:
            alive[n] = _nbytes(n)
        for d in ends[i]:
            alive.pop(d, None)
        if n.op == "call_function":
            out.append((n, sum(alive.values()), max(alive.values(), default=0)))
    return out


def cuts(gm):
    """The nodes after which the graph is cut (host ops aside)."""
    boundaries = live(gm)
    out = []
    for i, (node, total, largest) in enumerate(boundaries):
        if _is_host(node) or not TRIVIAL < total <= CUT_RATIO * largest:
            continue
        window = boundaries[max(0, i - WINDOW):i + WINDOW + 1]
        if total <= min((t for _, t, _ in window if t > TRIVIAL), default=total):
            out.append(node)
    return out


def assign(gm):
    """{node: region index} for gm's nodes, and [region name] by index. A host op is a region of
    its own."""
    cut_after = set(cuts(gm))
    region, part, size, hosts = 0, {}, collections.Counter(), set()
    pending = False
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_host(node):
            if 0 < size[region] < MIN_NODES and region - 1 not in hosts and size[region - 1]:
                for n in [n for n, r in part.items() if r == region]:  # a few nodes before a host op
                    part[n] = region - 1  # join the dense region before them
                size[region - 1] += size.pop(region)
            region += 1 if size[region] else 0
            part[node] = region
            size[region] += 1
            hosts.add(region)
            region += 1
            pending = False
            continue
        if pending and size[region] >= MIN_NODES:
            region += 1
        pending = False
        part[node] = region
        size[region] += 1
        if node in cut_after:
            pending = True  # the cut falls after this node
    for node in gm.graph.nodes:  # a parameter or buffer goes with its first user
        if node.op == "get_attr":
            users = [u for u in node.users if u in part]
            part[node] = part[users[0]] if users else 0
    renumber = {r: i for i, r in enumerate(sorted(set(part.values())))}
    part = {n: renumber[r] for n, r in part.items()}
    members = collections.defaultdict(list)
    for n, r in part.items():
        if n.op == "call_function":
            members[r].append(n)
    return part, [_name(members[r]) for r in range(len(renumber))]


def _name(nodes):
    """The deepest module path all of nodes were traced in, e.g. layers.2.attn ("model" if none)."""
    paths = []
    for n in nodes:
        stack = n.meta.get("nn_module_stack")
        if stack:
            paths.append([v[0] if isinstance(v, tuple) else str(v) for v in stack.values()])
    common = paths[0] if paths else []
    for p in paths[1:]:
        k = 0
        while k < min(len(common), len(p)) and common[k] == p[k]:
            k += 1
        common = common[:k]
    name = common[-1] if common and common[-1] else "model"
    for prefix in ("L['self'].", "self."):
        name = name.removeprefix(prefix)
    return name + (" (host)" if any(_is_host(n) for n in nodes) else "")


def split(gm):
    """gm split into its regions: (split GraphModule, [(submodule name, region name, host)])."""
    part, names = assign(gm)
    out = split_module(gm, gm, lambda n: part.get(n, 0), keep_original_order=True)
    info = []
    for sub_name, sub in out.named_children():
        if sub_name.startswith("submod_"):
            r = int(sub_name.removeprefix("submod_"))
            info.append((sub_name, names[r], any(_is_host(n) for n in sub.graph.nodes)))
    return out, info


# ---------------------------------------------------------------------------
# measured


def copy_bandwidth(device):
    """GB/s of a device-to-device copy of 1 GB, read and write counted (measured)."""
    src = torch.empty(1 << 30, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    return 2 * 10 * src.numel() / (time.perf_counter() - t) / 1e9


def _bytes(sub, args, params, rows):
    """(weights, state, activations) bytes a region's call reads, by its arguments: the graph's
    parameters and buffers are weights (a buffer the region writes is state: a cache); a table
    only gathered from counts its rows; an fp8 weight counts twice where the plan reads its bf16
    copy (ops.FP8_GEMV_MAX_ROWS). Also the three largest weights."""
    from . import ops
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    gathered, mutated = {}, set()
    for n in placeholders:
        if n.users and all(str(u.target) in GATHERS and u.args and u.args[0] is n for u in n.users):
            gathered[n] = sum(_nbytes(u) for u in n.users)
    for n in sub.graph.nodes:
        t = str(n.target).split(".")
        if n.op == "call_function" and len(t) == 3 and t[1].endswith("_") and n.args and \
                isinstance(n.args[0], torch.fx.Node) and n.args[0].op == "placeholder":
            mutated.add(n.args[0])
    weights = state = act = 0
    top = []
    for n, a in zip(placeholders, args):
        if not isinstance(a, torch.Tensor):
            continue
        b = gathered.get(n, a.numel() * a.element_size())
        if a.dtype == torch.float8_e4m3fn and rows > ops.FP8_GEMV_MAX_ROWS:
            b *= 2
        if n in mutated:
            state += b
        elif a.untyped_storage().data_ptr() in params:
            weights += b
            top.append((b, tuple(a.shape), str(a.dtype).removeprefix("torch.")))
        else:
            act += b
    return weights, state, act, sorted(top, reverse=True)[:3]


def profile(engine, plan=None, reps=12):
    """Each dense region of engine's decode graph, lowered under plan (default: a compiling plan
    for the engine's device) and compiled by inductor: its kernels, its median time replayed in
    step order, the bytes it reads and its roofline. Returns (records, copy GB/s). The engine
    must not have compiled its decode graph (compile_decode=False): the graph is lowered here."""
    from . import lower, ops
    from .engine import strip_asserts
    from .plan import Plan
    from .stepgraph import _Driver
    gm, device = engine._decode, torch.device(engine.device)
    strip_asserts(gm)
    lower.lower(gm, plan or Plan(device.type, compile_decode=True))
    split_gm, info = split(gm)
    inputs = {n.target if n.op == "placeholder" else n.name: n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"}
    ids = next(v for k, v in inputs.items() if "input_ids" in k)
    rows = ids.shape[0]
    params = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            obj = gm
            for part in node.target.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                params.add(obj.untyped_storage().data_ptr())
    params |= {t.untyped_storage().data_ptr() for t in list(gm.parameters()) + list(gm.buffers())}
    # one eager step through the split records each region's arguments (activations copied)
    recorded = {}
    for name, _, _ in info:
        sub = getattr(split_gm, name)

        def record(*args, _name=name, _fwd=sub.forward):
            recorded[_name] = [a.clone() if isinstance(a, torch.Tensor) and a.untyped_storage().data_ptr()
                               not in params else a for a in args]
            return _fwd(*args)
        sub.forward = record
    with torch.inference_mode(), engine.external_scope():
        split_gm(torch.randint(100, 30000, tuple(ids.shape), device=device),
                 torch.full(tuple(inputs[next(k for k in inputs if "cache_position" in k)].shape), 300,
                            device=device, dtype=torch.long))
    for name, _, _ in info:
        del getattr(split_gm, name).forward
    torch.cuda.synchronize(device)
    bw = copy_bandwidth(device)
    graphs, records = [], []
    with torch.inference_mode(), engine.external_scope():
        for name, region, host in info:
            sub = getattr(split_gm, name)
            nodes = sum(1 for n in sub.graph.nodes if n.op == "call_function")
            if host:
                records.append({"region": region, "host": True, "nodes": nodes})
                continue
            args = recorded[name]
            compiled = torch.compile(sub, dynamic=False)
            compiled(*args)
            torch.cuda.synchronize(device)
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g):
                compiled(*args)
            names = {}
            try:
                kernels = _Driver().node_counts(g.raw_cuda_graph(), names).get("kernels", 0)
            except RuntimeError:
                kernels = 0  # an empty graph
            g.instantiate()
            weights, state, act, top = _bytes(sub, args, params, rows)
            records.append({"region": region, "host": False, "nodes": nodes, "kernels": kernels,
                            "weight_bytes": weights, "state_bytes": state, "activation_bytes": act,
                            "largest": top, "roofline_us": (weights + state) / (bw * 1e3), "kernel_names": names})
            graphs.append((len(records) - 1, g))
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(graphs) + 1)]
        times = collections.defaultdict(list)
        for rep in range(reps):
            events[0].record()
            for k, (_, g) in enumerate(graphs):
                g.replay()
                events[k + 1].record()
            torch.cuda.synchronize(device)
            if rep >= 2:
                for k in range(len(graphs)):
                    times[k].append(events[k].elapsed_time(events[k + 1]) * 1000)
    for k, (i, _) in enumerate(graphs):
        records[i]["us"] = sorted(times[k])[len(times[k]) // 2]
    return records, bw


def describe(records, bw):
    lines = [f"copy bandwidth {bw:.0f} GB/s (read + write); {sum(1 for r in records if not r['host'])} dense regions"]
    total, by_name = 0.0, collections.defaultdict(float)
    for r in records:
        if r["host"]:
            lines.append(f"  {r['region']:32s} host op")
            continue
        us, ideal = r["us"], r["roofline_us"]
        total += us
        by_name[r["region"]] += us
        top = " ".join(f"{b / 1e6:.0f}MB{shape}{dt}" for b, shape, dt in r["largest"])
        lines.append(f"  {r['region']:32s} {r['nodes']:4d} nodes {r['kernels']:4d} kernels {us:8.1f} us  "
                     f"reads {(r['weight_bytes'] + r['state_bytes']) / 1e6:7.1f} MB  activations "
                     f"{r['activation_bytes'] / 1e6:5.2f} MB  roofline {ideal:7.1f} us ({ideal / us * 100 if us else 0:3.0f}%)  {top}")
    lines.append(f"dense regions in step order: {total / 1e3:.2f} ms")
    for name, us in sorted(by_name.items(), key=lambda x: -x[1])[:8]:
        lines.append(f"  {us / 1e3:7.2f} ms  {name}")
    return "\n".join(lines)


def main(argv):
    from .engine import Engine
    artifact = [a for a in argv if not a.startswith("--")][0]
    engine = Engine(artifact, device="cuda", compile_decode=False)
    if "--cuts" in argv:
        from .engine import strip_asserts
        strip_asserts(engine._decode)
        for node in cuts(engine._decode):
            print(f"cut after {node.name}: {_source(node)[:100]}")
        return
    records, bw = profile(engine)
    print(describe(records, bw))
    if "--names" in argv:  # the kernels of the regions furthest from their floor, by name
        worst = sorted((r for r in records if not r["host"]), key=lambda r: r["roofline_us"] - r["us"])[:4]
        for r in worst:
            print(f"\n{r['region']}: {r['kernels']} kernels, {r['us']:.0f} us, floor {r['roofline_us']:.0f} us")
            for name, n in sorted(r["kernel_names"].items(), key=lambda x: -x[1]):
                print(f"  {n:3d}  {name[:140]}")


if __name__ == "__main__":
    main(sys.argv[1:])
