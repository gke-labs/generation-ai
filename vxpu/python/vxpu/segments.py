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

The engine swaps per-session cache tensors into its graphs by name;
after a split the top-level module reads them (under their original
names) and hands them to the segments, so bind_state swaps them there.
"""

import os

import torch
from torch.fx.passes.split_module import split_module

HOST_OPS = ("vxpu.moe_experts.default", "vxpu.engram_rows.default")


def enabled():
    v = os.environ.get("VXPU_SPLIT_DECODE", "1").lower()
    return v not in ("0", "off", "false")


def functionalize_host_outputs(gm):
    """Make in-place ops on a host op's output out of place.

    The fork adds the shared expert into the routed experts' result in
    place (y += shared(x)); after a split that result is an input of the
    next segment, and a CUDA graph will not capture a graph that mutates
    its inputs, so every segment after an expert call ran uncaptured.
    The rewrite is exact: the out-of-place op computes the same values,
    and later readers of the original tensor read the new one, as they
    would have read the mutated one. Returns the number of nodes rewritten."""
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


def split_at_host_ops(gm):
    """Cut gm at every host-op node. Returns (split GraphModule, names of
    the submodules that hold a host op and must stay eager)."""
    functionalize_host_outputs(gm)
    segment = 0
    part = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in HOST_OPS:
            part[node] = 2 * segment + 1  # a segment of its own
            segment += 1
        elif node.op in ("call_function", "call_method", "call_module"):
            part[node] = 2 * segment
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

    def forward(self, *args, **kwargs):
        # One call is one decode step: a CUDA-graph replay reuses its output
        # buffers, so mark the step boundary (earlier steps' outputs may be
        # recycled from here) and hand back a copy of what the caller keeps.
        torch.compiler.cudagraph_mark_step_begin()
        out = self.split(*args, **kwargs)
        out = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        return out.clone() if isinstance(out, torch.Tensor) else out


def compile_segments(gm, device, mode, options=None):
    """The module split at its host ops with every dense segment compiled."""
    split, host = split_at_host_ops(gm)
    compiled = 0
    for name, sub in segments_of(split):
        if name in host or os.environ.get("VXPU_SPLIT_COMPILE", "1") in ("0", "off"):
            continue  # VXPU_SPLIT_COMPILE=0: the split alone, segments eager (a check of the cut)
        kwargs = {"mode": mode}
        if options:
            kwargs["options"] = options
        setattr(split, name, torch.compile(sub, **kwargs))
        compiled += 1
    return Split(split, len(host) + compiled, compiled)


def _bind_one(module, fqn, tensor):
    """Point module's attribute fqn (a registered buffer or a plain tensor
    attribute, as export leaves lifted constants) at tensor, if it has one."""
    parent, name = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
    try:
        target = module.get_submodule(parent) if parent else module
    except AttributeError:
        return False
    if name in target._buffers:
        target._buffers[name] = tensor
        return True
    if isinstance(target.__dict__.get(name), torch.Tensor):
        setattr(target, name, tensor)
        return True
    return False


def bind_state(split, state):
    """Bind each (fqn -> tensor) where the split module reads it.

    split_module keeps every attribute read (get_attr) in the top-level
    graph, under its original name, and passes the tensors into the
    segments as arguments; so the top level is where a session's caches
    must be swapped in. Segments are covered too, in case a future cut
    moves an attribute inside one."""
    root = getattr(split, "split", split)
    for fqn, tensor in state.items():
        _bind_one(root, fqn, tensor)
    for name, sub in segments_of(root):
        inner = getattr(sub, "_orig_mod", sub)  # a compiled segment wraps the module
        for fqn, tensor in state.items():
            _bind_one(inner, fqn, tensor)
