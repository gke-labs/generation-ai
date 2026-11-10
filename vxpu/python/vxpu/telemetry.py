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

"""Internal telemetry for tuning execution plans (#326).

A Recorder is created for one generate call when the caller asks for
telemetry, and its result rides back with the token stream, on the final
event, as a versioned JSON-able dict. Nothing is collected otherwise: the
hooks are installed for the call and removed after it.

What it records:

  segments    with the split decode (segments.py): each segment's time
              per decode step. On a GPU a CUDA event is recorded after
              every segment and read back in batches, so timing adds no
              synchronization; a segment's time is the GPU timeline
              between its neighbours' events (for a host op, the GPU
              waiting while the CPU works). On the CPU, wall time.
  host ops    the expert op's phases and the n-gram lookup, per decode
              step: wait_to_host (waiting for the GPU's preceding work,
              then the inputs' copy to the host: on the hybrid placement
              mostly the CPU idle while the GPU works), impl (the selected
              implementation's own time), to_device (the result back)
  experts     the experts each layer selected, per phase (prefill,
              decode): the statistics for placement and caching (#317)
  steps       each decode step's wall time; the prefill's time and tokens
  plan        the engine's plan (choices and selected implementations)

It is not a profiler and not exported anywhere: it is the executor
describing its own execution, for the planner (#316).
"""

import collections
import time
from typing import Any

import torch

FORMAT = "vxpu-telemetry/1"
DRAIN_EVERY = 8  # decode steps of CUDA events held before reading them back


class Recorder:
    """Telemetry for one generate call."""

    def __init__(self, device, segments=None, trace=False):
        self.cuda = str(device).startswith("cuda")
        self.device = device
        self.segments = segments or []  # [{name, kind, layers, nodes}] in call order
        self.trace = trace
        self.phase = "prefill"
        self.seg_ms = collections.Counter()
        self.seg_steps = 0
        self.host = collections.defaultdict(collections.Counter)
        self.experts = {"prefill": collections.defaultdict(collections.Counter),
                        "decode": collections.defaultdict(collections.Counter)}
        self.expert_trace = collections.defaultdict(list)
        self.step_ms = []
        self.prefill_ms = None
        self.prefill_tokens = None  # prompt tokens this call prefilled (the cached prefix excluded)
        self._pending = []
        self._step = None
        self._t = None

    # ---- segments (called by segments.Split while installed)

    def begin_step(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(self.device))  # the engine's device, not the process's current one
            self._step = [e]
        else:
            self._t = time.perf_counter()
            self._step = []

    def after_segment(self, index):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(self.device))  # the engine's device, not the process's current one
            self._step.append(e)
        else:
            now = time.perf_counter()
            self._step.append((index, (now - self._t) * 1e3))  # counted at end_step, if it was a decode step
            self._t = now

    def end_step(self):
        if self.phase != "decode":
            self._step = None  # a one-token prefill through the decode graph is not a decode step
            return
        if self.cuda:
            self._pending.append(self._step)
            if len(self._pending) >= DRAIN_EVERY:
                self.drain()
        else:
            for index, ms in self._step:
                self.seg_ms[index] += ms
            self.seg_steps += 1
        self._step = None

    def drain(self):
        """Read back the recorded CUDA events (waits for the last one)."""
        if not self._pending:
            return
        self._pending[-1][-1].synchronize()
        for events in self._pending:
            for i in range(1, len(events)):
                self.seg_ms[i - 1] += events[i - 1].elapsed_time(events[i])
            self.seg_steps += 1
        self._pending = []

    # ---- host ops (called by ops.py while bound)

    def host_op(self, op, **ms):
        counters = self.host[(self.phase, op)]
        counters["calls"] += 1
        for k, v in ms.items():
            counters[k] += v

    def selected_experts(self, layer, indices):
        counts = self.experts[self.phase][layer]
        for e in indices:
            counts[e] += 1
        if self.trace and self.phase == "decode":
            self.expert_trace[layer].append([int(e) for e in indices])

    # ---- result

    def result(self, plan_record=None):
        self.drain()
        steps = max(self.seg_steps, 1)
        out = {"format": FORMAT, "device": str(self.device), "decode_steps": len(self.step_ms)}
        if self.prefill_ms is not None:
            out["prefill_ms"] = round(self.prefill_ms, 2)
        if self.prefill_tokens is not None:
            out["prefill_tokens"] = self.prefill_tokens
        if self.step_ms:
            s = sorted(self.step_ms)
            out["step_ms"] = {"mean": round(sum(s) / len(s), 3), "p50": round(s[len(s) // 2], 3),
                              "p90": round(s[min(len(s) - 1, int(len(s) * 0.9))], 3),
                              "values": [round(v, 3) for v in self.step_ms]}
        if self.segments and self.seg_steps:
            rows, kinds = [], collections.Counter()
            for i, seg in enumerate(self.segments):
                ms = self.seg_ms[i] / steps
                kinds[seg["kind"]] += ms
                rows.append({**seg, "ms_per_step": round(ms, 4)})
            out["segments"] = rows
            out["by_kind_ms_per_step"] = {k: round(v, 3) for k, v in kinds.items()}
            out["segment_steps"] = self.seg_steps
        host = {}
        for (phase, op), c in sorted(self.host.items()):
            per = len(self.step_ms) if phase == "decode" else 1
            host.setdefault(phase, {})[op] = {k: round(v / max(per, 1), 4) for k, v in c.items()}
        out["host_ops"] = host  # decode: per step; prefill: totals
        out["experts"] = {phase: {str(layer): dict(sorted(c.items())) for layer, c in sorted(layers.items())}
                          for phase, layers in self.experts.items() if layers}
        if self.trace:
            out["expert_trace"] = {str(layer): steps for layer, steps in sorted(self.expert_trace.items())}
        if plan_record is not None:
            out["plan"] = plan_record
        return out


def segment_info(split: torch.nn.Module) -> list[dict[str, Any]]:
    """[{name, kind, layers, nodes}] for a split module's segments, in call
    order: kind is experts / engram / dense, layers the decoder layers its
    nodes come from (from the export's module stacks)."""
    import re
    from . import segments
    layer_re = re.compile(r"layers\.(\d+)")
    info: list[dict[str, Any]] = []
    for name, sub in segments.segments_of(split):
        graph = getattr(sub, "_orig_mod", sub).graph
        targets = {getattr(n.target, "__name__", str(n.target)) for n in graph.nodes if n.op == "call_function"}
        targets |= {str(n.target) for n in graph.nodes if n.op == "call_function"}
        kind = ("experts" if targets & {"vxpu.moe_experts.default", "moe_experts"}
                else "engram" if targets & {"vxpu.engram_rows.default", "engram_rows"} else "dense")
        layers = set()
        for n in graph.nodes:
            for v in (n.meta.get("nn_module_stack") or {}).values():
                path = v[0] if isinstance(v, (tuple, list)) else str(v)
                m = layer_re.search(path)
                if m:
                    layers.add(int(m.group(1)))
        info.append({"name": name, "kind": kind, "layers": sorted(layers),
                     "nodes": sum(1 for n in graph.nodes if n.op == "call_function"),
                     "writes": segment_writes(graph)})
    return info


def segment_writes(graph: torch.fx.Graph | torch.fx.GraphModule) -> list[int]:
    """The placeholders an in-place aten op (index_put_, copy_, ...) writes,
    reached through views via segments._aliases, as sorted positions."""
    import torch
    from . import segments
    g = getattr(graph, "graph", graph)
    placeholders = [n for n in g.nodes if n.op == "placeholder"]
    writes = set()
    for node in g.nodes:
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            continue
        is_inplace = False
        if node.op == "call_function":
            target = str(node.target)
            if target.startswith("aten."):
                parts = target.split(".")
                if len(parts) > 1 and parts[1].endswith("_") and not parts[1].endswith("__"):
                    is_inplace = True
            schema = getattr(node.target, "_schema", None)
            if schema and schema.arguments and schema.arguments[0].alias_info and schema.arguments[0].alias_info.is_write:
                is_inplace = True
        elif node.op == "call_method":
            if isinstance(node.target, str) and node.target.endswith("_") and not node.target.endswith("__"):
                is_inplace = True
        if not is_inplace:
            continue
        curr = node.args[0]
        while isinstance(curr, torch.fx.Node):
            if curr.op == "placeholder":
                if curr in placeholders:
                    writes.add(placeholders.index(curr))
                break
            if segments._aliases(curr) and curr.args and isinstance(curr.args[0], torch.fx.Node):
                curr = curr.args[0]
            else:
                break
    return sorted(writes)
