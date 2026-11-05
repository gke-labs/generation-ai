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

"""A graph split at its host ops, each dense segment compiled on its own.

A decode step of DeepSeek-V4.1-Flash is 26,000 small nodes with 42
places where the GPU must stop and wait for the host: the experts on
the CPU and the n-gram rows from a file (ops.py). Issued one node at a
time from Python the step is launch-bound; a CUDA graph would replay
the whole sequence at once, but a CUDA graph cannot contain host work,
and inductor's own partitioning did not keep these ops out of capture.

So the split is made here, explicitly: the exported GraphModule is cut
at every host-op node (torch.fx's split_module), the host ops stay
eager, and each dense segment between them is compiled separately (on
CUDA with CUDA graphs, one per segment). The segments are the same
boundaries an ahead-of-time compile (AOTInductor) or a runtime in
another language would use: segment, mailbox, segment.

The split is made from a graph already bound to the engine's caches:
split_module keeps every attribute read in the top-level graph, under
its original name, and passes the tensors to the segments as arguments.
The compiled segments need those addresses fixed, so the engine keeps
that one state bound and copies sessions in and out of it (Engine._bind)
rather than re-pointing the graph.
"""


import operator
import os

import torch
from torch.fx.passes.split_module import split_module

HOST_OPS = ("vxpu.moe_experts.default", "vxpu.engram_rows.default",
            # a tensor-parallel step's collectives (hero run 14): the sum and the gather across the ranks cut the
            # dense segments too, so the executor runs them between the segments (parallel.py, stepgraph.py)
            "vxpu.tp_allreduce.default", "vxpu.tp_allreduce_hc.default", "vxpu.tp_gemv_allreduce_hc.default",
            "vxpu.tp_allgather.default",
            # the embedding's rows read from the owning rank's shard (hero run 15): a step-graph node too
            "vxpu.tp_embed_rows.default")
COLLECTIVE_OPS = ("vxpu.tp_allreduce.default", "vxpu.tp_allreduce_hc.default", "vxpu.tp_gemv_allreduce_hc.default",
                  "vxpu.tp_allgather.default")

# Ops cut into segments of their own, so that the step graph can run them beside the work they
# are independent of (stepgraph: a node that does not read the node before it goes on a parallel
# branch). The attention's query projection and its key/value projection read the same input and
# neither reads the other: the query path is 49 us at one row and the key/value one 9 us, and in
# one compiled segment they ran one after the other. The shared expert's front (its three GEMVs,
# about 47 us at one row) reads the FFN front's quantized input and nothing of the routed experts,
# so it runs beside the experts node; its combine with the experts' sum follows. VXPU_ISLANDS
# overrides the list for an A/B (empty: none).
# (Hero run 10 kept the shared expert's front off the branch after it corrupted rows at nine rows; the cause,
# the expert scratch's reuse by the capture's pool, is closed in stepgraph.py and the front is an island again.)
# The shared expert's front is an island by default since hero run 15's read/write rule (stepgraph.branch_choice): the
# fused-push form (shared_ffn_mid, to 16 rows) and the whole front (shared_ffn_front, above) are both named so that
# whichever the lowering leaves in the graph runs beside the routed experts. Measured against the tokens at 1 to 40
# rows: -0.4 ms a step at 1 to 8 rows, -0.7 at 33 and 40, tokens identical (A86, A87b, A88).
ISLAND_OPS = ("vxpu.attn_q.default", "vxpu.window_kv.default", "vxpu.shared_ffn_mid.default", "vxpu.shared_ffn_front.default")
_islands_env = os.environ.get("VXPU_ISLANDS")
if _islands_env is not None:
    ISLAND_OPS = tuple(f"vxpu.{name}.default" for name in _islands_env.split(",") if name)



def functionalize_host_outputs(gm):
    """Make in-place ops on a host op's output out of place.

    The fork adds the shared expert into the routed experts' result in
    place (y += shared(x)); after a split that result is an input of the
    next segment, and a CUDA graph will not capture a graph that mutates
    its inputs, so every segment after an expert call ran uncaptured.
    The rewrite is exact: the out-of-place op computes the same values,
    and later readers of the original tensor read the new one, as they
    would have read the mutated one. That holds only for readers of the
    tensor itself: a view of it taken before the in-place op would keep
    the old values. So an output with such a view is left as it is (and
    reported): the segment that mutates it then runs uncaptured, which
    costs speed, never correctness. Returns the number of nodes rewritten."""
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    host = {n for n in gm.graph.nodes if n.op == "call_function" and str(n.target) in HOST_OPS}
    rewritten = 0
    for node in list(gm.graph.nodes):
        target = str(node.target)
        if not (node.op == "call_function" and target.startswith("aten.") and node.args
                and isinstance(node.args[0], torch.fx.Node) and node.args[0] in host):
            continue
        packet, overload = target.split(".")[1], target.split(".")[2]
        if not packet.endswith("_") or packet.endswith("__"):
            continue
        functional = getattr(getattr(torch.ops.aten, packet[:-1]), overload)
        base = node.args[0]
        early_views = [u for u in base.users if u is not node and order.get(u, -1) < order[node] and _aliases(u)]
        if early_views:
            print(f"[vxpu] left {node.name} in place: {base.name} has a view taken before it "
                  f"({early_views[0].name}); that segment will not be a CUDA graph", flush=True)
            continue
        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(functional, node.args, node.kwargs)
        replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        for user in list(base.users):
            if user is not replacement and order.get(user, -1) > order[node]:
                user.replace_input_with(base, replacement)
        gm.graph.erase_node(node)
        rewritten += 1
    if rewritten:
        gm.graph.lint()
        gm.recompile()
    return rewritten


def _aliases(node):
    """Whether node's result aliases an input (a view), per its schema."""
    schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
    return schema is not None and any(r.alias_info is not None for r in schema.returns)


def mark_islands(gm):
    """Fence every island op (ISLAND_OPS) between two alias nodes tagged in their meta, before
    the lowering runs: the lowering replaces the op's node with its kernels' nodes, spliced in
    right before it, so after it nothing names the op, but the fences still stand around what
    it became and split_at_host_ops cuts at them. An alias launches nothing. Returns how many
    ops were fenced."""
    fenced = 0
    for node in list(gm.graph.nodes):
        if not (node.op == "call_function" and str(node.target) in ISLAND_OPS):
            continue
        name = f"{str(node.target).split('.')[1]}_{fenced}"
        first = next((a for a in node.args if isinstance(a, torch.fx.Node)), None)
        if first is None:
            continue
        with gm.graph.inserting_before(node):
            start = gm.graph.call_function(torch.ops.aten.alias.default, (first,))
        start.meta = dict(first.meta)
        start.meta["vxpu_island"] = ("start", name)
        node.replace_input_with(first, start)
        outs = [u for u in node.users if u.op == "call_function" and u.target is operator.getitem] or [node]
        for out in outs:
            with gm.graph.inserting_after(out):
                end = gm.graph.call_function(torch.ops.aten.alias.default, (out,))
            end.meta = dict(out.meta)
            end.meta["vxpu_island"] = ("end", name)
            for user in list(out.users):
                if user is not end:
                    user.replace_input_with(out, end)
        fenced += 1
    if fenced:
        gm.graph.lint()
        gm.recompile()
    return fenced


def split_at_host_ops(gm):
    """Cut gm at every host-op node, and at the fences around island ops
    (mark_islands; an unlowered island op counts as well). Returns (split
    GraphModule, names of the submodules that hold a host op and must
    stay eager)."""
    functionalize_host_outputs(gm)
    # an island spans its start fence to its last end fence (the op's nodes and its outputs'
    # getitems between them), one partition; what follows is a new dense partition
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    span = {}  # island name -> (first index, last index)
    for node in gm.graph.nodes:
        tag = node.meta.get("vxpu_island") if node.op == "call_function" else None
        if tag is not None:
            first, last = span.get(tag[1], (order[node], order[node]))
            span[tag[1]] = (min(first, order[node]), max(last, order[node]))
    island_of = {}
    for name, (first, last) in span.items():
        for i in range(first, last + 1):
            island_of[i] = name
    current = 0  # the dense segment being filled; a host op or an island takes the next number alone
    part = {}
    inside = None
    alone_islands = []  # unfenced island ops cut into a segment of their own
    for node in gm.graph.nodes:
        here = island_of.get(order[node])
        if here != inside:  # entering or leaving an island: a new number either way
            current += 1
            inside = here
        if node.op == "call_function" and str(node.target) in HOST_OPS + (ISLAND_OPS if here is None else ()):
            part[node] = current + 1  # a segment of its own
            current += 2
            if str(node.target) in ISLAND_OPS:
                alone_islands.append(node)
        elif node.op in ("call_function", "call_method", "call_module"):
            part[node] = current
    # an unfenced island op that returns several tensors keeps the getitems that unpack them in its
    # segment: the segment then returns the tensors and the split hands each to its consumers through
    # a top-level getitem, which the dataflow runner routes through an inbox per output. Left in the
    # consumer's segment, the getitems made the island segment return its whole tuple as one value,
    # and the runner kept a tuple in one inbox (no shape to plan, no copy to make).
    for node in alone_islands:
        for user in node.users:
            if user.op == "call_function" and user.target is operator.getitem:
                part[user] = part[node]
    # a lifted parameter or buffer goes with its first user
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            users = [u for u in node.users if u in part]
            part[node] = part[users[0]] if users else 0
    split = split_module(gm, gm, lambda n: part.get(n, 0), keep_original_order=True)
    host = []
    for name, sub in segments_of(split):
        if any(n.op == "call_function" and str(n.target) in HOST_OPS for n in sub.graph.nodes):
            host.append(name)
    return split, host


def segments_of(split):
    """The (name, GraphModule) segments of a split module, in call order
    (its other children are the attribute holders split_module creates)."""
    return [(name, sub) for name, sub in split.named_children()
            if name.startswith("submod_") and isinstance(getattr(sub, "_orig_mod", sub), torch.fx.GraphModule)]


class Split(torch.nn.Module):
    """The split module behind the original's calling convention (the
    original returned a tensor; split_module returns a one-tuple)."""

    def __init__(self, split, total, compiled):
        super().__init__()
        self.split = split
        self.vxpu_segments = (total, compiled)
        self.recorder = None
        self._unwrapped = None

    def set_recorder(self, recorder):
        """Time every segment into recorder (telemetry.Recorder) until
        called with None. Each segment is wrapped while a recorder is set
        and unwrapped after, so a step without telemetry runs exactly the
        unwrapped module."""
        if recorder is not None and self._unwrapped is None:
            names = [name for name, _ in segments_of(self.split)]
            self._unwrapped = {name: self.split._modules[name] for name in names}
            for index, name in enumerate(names):
                self.split._modules[name] = _Recorded(self._unwrapped[name], index, recorder)
        elif recorder is None and self._unwrapped is not None:
            for name, sub in self._unwrapped.items():
                self.split._modules[name] = sub
            self._unwrapped = None
        self.recorder = recorder

    def forward(self, *args, **kwargs):
        # One call is one decode step: a CUDA-graph replay reuses its output
        # buffers, so mark the step boundary (earlier steps' outputs may be
        # recycled from here) and hand back a copy of what the caller keeps.
        torch.compiler.cudagraph_mark_step_begin()
        recorder = self.recorder
        if recorder is not None:
            recorder.begin_step()
        out = self.split(*args, **kwargs)
        if recorder is not None:
            recorder.end_step()
        out = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        return out.clone() if isinstance(out, torch.Tensor) else out


class _Recorded(torch.nn.Module):
    """A segment that tells the recorder when it has been issued."""

    def __init__(self, sub, index, recorder):
        super().__init__()
        self.sub, self.index, self.recorder = sub, index, recorder

    def forward(self, *args, **kwargs):
        out = self.sub(*args, **kwargs)
        self.recorder.after_segment(self.index)
        return out


def call_host_ops_directly(sub):
    """In an eager host-op segment, call each host op's Python
    implementation instead of going through the dispatcher: the segment
    is never traced again, and torch.library's custom-op wrapper costs
    tens of microseconds a call, forty-odd calls a decode step (the
    plan's direct_host_ops; VXPU_DIRECT_HOST_OPS=0 keeps the dispatcher)."""
    from . import ops
    direct = {"vxpu.moe_experts.default": ops.moe_experts._init_fn,
              "vxpu.engram_rows.default": ops.engram_rows._init_fn}
    changed = False
    for node in sub.graph.nodes:
        if node.op == "call_function" and str(node.target) in direct:
            node.target = direct[str(node.target)]
            changed = True
    if changed:
        sub.recompile()


def compile_segments(gm, device, mode, split_compile=True, direct=True, runtime="jit",
                     aoti_cache_dir=None):
    """The module split at its host ops with every dense segment compiled
    (split_compile=False: the split alone, segments eager, a check of the
    cut) and the host-op segments calling their implementations directly
    (direct=False: through torch's dispatcher). runtime="aoti" compiles the
    dense segments with AOTInductor instead and runs them in the engine's
    own CUDA graphs (aoti.py; the graph must be lowered). The engine passes
    its plan's choices."""
    split, host = split_at_host_ops(gm)
    compiled = 0
    packages = pool = None
    if runtime == "aoti" and split_compile:
        from . import aoti
        packages = aoti.Packages(aoti_cache_dir, expected=len(segments_of(split)) - len(host))
        pool = torch.cuda.graph_pool_handle()  # one memory pool for every segment's graph
    for name, sub in segments_of(split):
        if name in host:
            if direct:
                call_host_ops_directly(sub)
            continue
        if not split_compile:
            continue
        if packages is not None:
            setattr(split, name, aoti.GraphedSegment(name, sub, aoti.static_arguments(split, name), packages, pool))
            compiled += 1
            continue
        setattr(split, name, torch.compile(sub, mode=mode))
        compiled += 1
    return Split(split, len(segments_of(split)), compiled)  # every segment, compiled or not
