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


import os
from typing import Any

import torch
from torch.fx.passes.split_module import split_module

HOST_OPS = ("vxpu.moe_experts.default", "vxpu.engram_rows.default")

from . import lower, plan_pb2, rehydrate


def decode_segments(gm: torch.fx.GraphModule, plan, device="cpu"):
    """Splits a decode module at host ops with lowering applied for package compilation."""
    rehydrate.strip_asserts(gm)
    if str(device).startswith("cuda"):
        import torch._inductor.config as inductor_config
        inductor_config.prologue_fusion = True
    report = lower.lower(gm, plan)
    msg = lower.describe(report)
    print(f"[vxpu] {msg}", flush=True)
    plan.note(msg)
    rehydrate.strip_asserts(gm)
    return split_at_host_ops(gm)



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


def _segment_run_name(run: int) -> str:
    try:
        return plan_pb2.SegmentRun.Name(run)
    except ValueError:
        return str(run)


def check_plan(plan_file: plan_pb2.PlanFile, device: str, aoti_cache_dir: str) -> None:
    """Validates each segment's run mode and verifies every required package is staged."""
    seen: set[str] = set()
    required: list[tuple[str, str]] = []
    is_cuda = str(device).startswith("cuda")
    for seg in plan_file.segments:
        name = seg.name
        if not name or name != name.strip():
            raise ValueError(f"plan segment has invalid name {name!r}")
        if name in seen:
            raise ValueError(f"plan names segment {name!r} more than once")
        seen.add(name)
        run = seg.run
        if run == plan_pb2.SEGMENT_RUN_PLAIN_TORCH:
            if seg.package_key:
                raise ValueError(
                    f"segment {name!r} runs as SEGMENT_RUN_PLAIN_TORCH but sets package_key {seg.package_key!r}"
                )
            continue
        if run in (plan_pb2.SEGMENT_RUN_PACKAGE, plan_pb2.SEGMENT_RUN_CAPTURED):
            if run == plan_pb2.SEGMENT_RUN_CAPTURED and not is_cuda:
                raise ValueError(
                    f"segment {name!r} requires CUDA graph capture (SEGMENT_RUN_CAPTURED) on non-CUDA device {device!r}"
                )
            key = seg.package_key
            if not key or key != key.strip() or "/" in key or "\\" in key or key in (".", ".."):
                raise ValueError(f"segment {name!r} requires a valid package_key, got {key!r}")
            required.append((name, key))
            continue
        if run == plan_pb2.SEGMENT_RUN_JIT:
            raise ValueError(f"segment {name!r} requests SEGMENT_RUN_JIT, which the executor refuses")
        raise ValueError(f"segment {name!r} has unsupported run mode {_segment_run_name(run)}")

    if not required:
        return
    cache_dir = os.path.abspath(aoti_cache_dir)
    missing = [
        (name, key)
        for name, key in required
        if not os.path.isfile(os.path.join(cache_dir, f"{key}.pt2"))
    ]
    if missing:
        details = ", ".join(f"{name} ({key}.pt2)" for name, key in missing)
        raise FileNotFoundError(
            f"missing compiled package(s) in {cache_dir}: {details}; "
            "build the plan's packages first (for example, with 'vxpu plan packages --build')"
        )


def apply_plan(
    gm: torch.fx.GraphModule,
    plan_file: plan_pb2.PlanFile,
    device: str,
    aoti_cache_dir: str,
    direct: bool = True,
) -> Split:
    """Splits gm at host ops and configures each segment strictly from plan_file.segments."""
    check_plan(plan_file, device, aoti_cache_dir)
    rehydrate.strip_asserts(gm)
    split, host = split_at_host_ops(gm)
    host_set = set(host)
    actual = segments_of(split)
    actual_names = {name for name, _ in actual}
    planned = {seg.name: seg for seg in plan_file.segments}

    omitted = sorted(actual_names - set(planned))
    if omitted:
        raise ValueError(f"plan omits segment(s): {', '.join(omitted)}")
    extra = sorted(set(planned) - actual_names)
    if extra:
        raise ValueError(f"plan names unknown segment(s): {', '.join(extra)}")
    for name in host:
        if planned[name].run != plan_pb2.SEGMENT_RUN_PLAIN_TORCH:
            raise ValueError(
                f"host segment {name!r} must run as SEGMENT_RUN_PLAIN_TORCH, got {_segment_run_name(planned[name].run)}"
            )

    has_captured = any(seg.run == plan_pb2.SEGMENT_RUN_CAPTURED for seg in plan_file.segments)
    has_packaged = any(
        seg.run in (plan_pb2.SEGMENT_RUN_PACKAGE, plan_pb2.SEGMENT_RUN_CAPTURED)
        for seg in plan_file.segments
    )
    pool: Any = torch.cuda.graph_pool_handle() if has_captured else None
    loaded = 0
    from . import aoti

    if has_packaged:
        aoti.clean_cache_strays(os.path.abspath(aoti_cache_dir))

    for name, sub in actual:
        seg = planned[name]
        if seg.run == plan_pb2.SEGMENT_RUN_PLAIN_TORCH:
            if name in host_set and direct:
                call_host_ops_directly(sub)
            continue

        args = aoti.example_args(split, name, sub, device)
        runner = aoti.open_package(aoti_cache_dir, seg.package_key)
        static = aoti.static_arguments(split, name)
        if seg.run == plan_pb2.SEGMENT_RUN_CAPTURED:
            wrapped: torch.nn.Module = aoti.GraphedSegment(name, sub, static, runner, pool, example_args=args)
        else:
            wrapped = aoti.Segment(name, sub, static, runner, example_args=args)
        setattr(split, name, wrapped)
        loaded += 1

    if loaded:
        print(
            f"[vxpu] AOTInductor segments: {loaded} loaded from the cache, 0 built",
            flush=True,
        )
    return Split(split, len(actual), loaded)
