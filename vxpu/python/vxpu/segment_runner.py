# pyright: strict
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

"""Sequential execution of plan segments through one run_segment function.

The Go planner emits PlanFile.segments in execution order, with each segment's
reads and writes declared by graph value name and lifecycle (step input,
resident, intermediate, or step output). This module wires a pre-allocated slot
table strictly from those declarations and runs a decode step by iterating
through the plan's segments in order through run_segment.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import operator
from typing import Protocol

import torch
from torch.fx.node import map_arg

from . import plan_pb2

_UNSET: torch.Tensor = torch.empty(0)


class SegmentCallable(Protocol):
    """Callable that executes one segment from positional tensor inputs."""

    def __call__(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]: ...


class SegmentRecorder(Protocol):
    """Telemetry recorder notified around each segment execution."""

    def begin_step(self) -> None: ...
    def after_segment(self, index: int) -> None: ...
    def end_step(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SegmentExecution:
    """One executable plan segment bound to pre-allocated slot indices."""

    name: str
    host: bool
    callable: SegmentCallable
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]


def run_segment(
    segment: SegmentExecution,
    env: list[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Runs one segment, reading inputs from env and writing outputs into env."""
    out = segment.callable(*[env[i] for i in segment.input_slots])
    out_slots = segment.output_slots
    if len(out_slots) == 1:
        if isinstance(out, tuple):
            raise RuntimeError(
                f"segment {segment.name!r}: expected 1 tensor output, got tuple of {len(out)}"
            )
        env[out_slots[0]] = out
    elif len(out_slots) > 1:
        if isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"segment {segment.name!r}: expected {len(out_slots)} outputs, got single tensor"
            )
        if len(out) != len(out_slots):
            raise RuntimeError(
                f"segment {segment.name!r}: expected {len(out_slots)} outputs, got {len(out)}"
            )
        for slot, val in zip(out_slots, out):
            env[slot] = val
    return out


class SegmentRunner(torch.nn.Module):
    """Executes PlanFile.segments sequentially through run_segment."""

    def __init__(
        self,
        split: torch.fx.GraphModule,
        segments: tuple[SegmentExecution, ...],
        env: list[torch.Tensor],
        step_input_slots: tuple[int, ...],
        step_input_by_name: dict[str, int],
        step_output_slots: tuple[int, ...],
        resident_commits: tuple[tuple[int, torch.Tensor], ...],
        transient_slots: tuple[int, ...],
        loaded: int,
    ) -> None:
        super().__init__()
        self.split: torch.fx.GraphModule = split
        self.segments: tuple[SegmentExecution, ...] = segments
        self.vxpu_segments: tuple[int, int] = (len(segments), loaded)
        self._env: list[torch.Tensor] = env
        self._step_input_slots: tuple[int, ...] = step_input_slots
        self._step_input_by_name: dict[str, int] = step_input_by_name
        self._step_output_slots: tuple[int, ...] = step_output_slots
        self._resident_commits: tuple[tuple[int, torch.Tensor], ...] = resident_commits
        self._transient_slots: tuple[int, ...] = transient_slots
        self._recorder: SegmentRecorder | None = None

    def set_recorder(self, recorder: SegmentRecorder | None) -> None:
        """Installs or clears a per-segment telemetry recorder."""
        self._recorder = recorder

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Runs one decode step by executing each plan segment in order."""
        # One call is one decode step: a CUDA-graph replay reuses its output
        # buffers, so mark the step boundary and return a cloned output.
        torch.compiler.cudagraph_mark_step_begin()
        if len(args) + len(kwargs) != len(self._step_input_slots):
            raise TypeError(
                f"expected {len(self._step_input_slots)} step inputs, "
                f"got {len(args)} positional and {len(kwargs)} keyword"
            )
        env = self._env
        try:
            if args:
                for i, arg in enumerate(args):
                    env[self._step_input_slots[i]] = arg
            if kwargs:
                bound_positional = self._step_input_slots[: len(args)] if args else ()
                for name, arg in kwargs.items():
                    if name not in self._step_input_by_name:
                        raise TypeError(f"unexpected step input keyword argument {name!r}")
                    slot = self._step_input_by_name[name]
                    if bound_positional and slot in bound_positional:
                        raise TypeError(f"multiple values for step input {name!r}")
                    env[slot] = arg

            recorder = self._recorder
            if recorder is not None:
                recorder.begin_step()
            for index, segment in enumerate(self.segments):
                run_segment(segment, env)
                if recorder is not None:
                    recorder.after_segment(index)
            if recorder is not None:
                recorder.end_step()

            for slot, dst in self._resident_commits:
                dst.copy_(env[slot])

            if len(self._step_output_slots) == 1:
                return env[self._step_output_slots[0]].clone()
            return tuple(env[slot].clone() for slot in self._step_output_slots)
        finally:
            for slot in self._transient_slots:
                env[slot] = _UNSET


def _resolve_tensor(module: torch.nn.Module, fqn: str) -> torch.Tensor:
    """Resolves a dotted resident attribute path on module to its tensor."""
    if not fqn:
        raise ValueError("resident value has empty resident path")
    value: object = operator.attrgetter(fqn)(module)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"resident path {fqn!r} resolved to {type(value).__name__}, expected Tensor")
    return value


def _submodule_output_names(sub: torch.fx.GraphModule, seg_name: str) -> list[str]:
    """Returns the original FX node names emitted by sub's output node in positional order."""
    for node in reversed(list(sub.graph.nodes)):
        if node.op != "output":
            continue
        names: list[str] = []

        def _collect(n: torch.fx.Node) -> torch.fx.Node:
            names.append(n.name)
            return n

        map_arg(node.args[0], _collect)
        return names
    raise ValueError(f"segment {seg_name!r}: submodule graph has no output node")


class _SlotTable:
    """Mutable slot allocation state used while wiring PlanFile.segments."""

    __slots__ = (
        "slot_by_name",
        "env",
        "written_names",
        "first_seen_step_input_slots",
        "step_input_by_name",
        "step_output_slots",
        "resident_commits",
        "transient_slots",
    )

    def __init__(self) -> None:
        self.slot_by_name: dict[str, int] = {}
        self.env: list[torch.Tensor] = []
        self.written_names: set[str] = set()
        self.first_seen_step_input_slots: list[int] = []
        self.step_input_by_name: dict[str, int] = {}
        self.step_output_slots: list[int] = []
        self.resident_commits: list[tuple[int, torch.Tensor]] = []
        self.transient_slots: list[int] = []


def _wire_segment_reads(
    table: _SlotTable,
    split: torch.fx.GraphModule,
    seg: plan_pb2.Segment,
    sub: torch.fx.GraphModule,
    attr_targets: Mapping[str, str],
) -> tuple[int, ...]:
    """Resolves seg.reads into table slots and aligns them to sub's placeholder order."""
    read_slots_by_key: dict[str, int] = {}
    for val in seg.reads:
        if val.kind == plan_pb2.VALUE_KIND_STEP_INPUT:
            if val.name not in table.slot_by_name:
                slot = len(table.env)
                table.slot_by_name[val.name] = slot
                table.env.append(_UNSET)
                table.first_seen_step_input_slots.append(slot)
                table.step_input_by_name[val.name] = slot
                table.transient_slots.append(slot)
            read_slots_by_key[val.name] = table.slot_by_name[val.name]
        elif val.kind == plan_pb2.VALUE_KIND_RESIDENT:
            if val.name in table.written_names:
                read_slots_by_key[val.name] = table.slot_by_name[val.name]
            else:
                if val.name not in table.slot_by_name:
                    slot = len(table.env)
                    table.slot_by_name[val.name] = slot
                    table.env.append(_resolve_tensor(split, val.resident))
                read_slots_by_key[val.resident] = table.slot_by_name[val.name]
        elif val.kind == plan_pb2.VALUE_KIND_INTERMEDIATE:
            if val.name not in table.written_names:
                raise ValueError(
                    f"segment {seg.name!r}: reads intermediate {val.name!r} before any segment writes it"
                )
            read_slots_by_key[val.name] = table.slot_by_name[val.name]
        else:
            raise ValueError(
                f"segment {seg.name!r}: unsupported read value kind {val.kind} for {val.name!r}"
            )

    placeholders = [n.name for n in sub.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(seg.reads):
        raise ValueError(
            f"segment {seg.name!r}: plan declares {len(seg.reads)} reads, "
            f"submodule expects {len(placeholders)} inputs"
        )
    input_slots: list[int] = []
    for ph in placeholders:
        key = attr_targets.get(ph, ph)
        if key not in read_slots_by_key:
            raise ValueError(
                f"segment {seg.name!r}: submodule input {ph!r} (key {key!r}) not found in plan reads"
            )
        input_slots.append(read_slots_by_key[key])
    return tuple(input_slots)


def _wire_segment_writes(
    table: _SlotTable,
    split: torch.fx.GraphModule,
    seg: plan_pb2.Segment,
    sub: torch.fx.GraphModule,
) -> tuple[int, ...]:
    """Allocates table slots for seg.writes and aligns them to sub's output order."""
    write_slots_by_name: dict[str, int] = {}
    for val in seg.writes:
        if val.name in table.slot_by_name:
            raise ValueError(f"segment {seg.name!r}: duplicate write {val.name!r}")
        slot = len(table.env)
        table.slot_by_name[val.name] = slot
        table.env.append(_UNSET)
        table.written_names.add(val.name)
        table.transient_slots.append(slot)
        write_slots_by_name[val.name] = slot
        if val.kind == plan_pb2.VALUE_KIND_STEP_OUTPUT:
            table.step_output_slots.append(slot)
        elif val.kind == plan_pb2.VALUE_KIND_RESIDENT:
            table.resident_commits.append((slot, _resolve_tensor(split, val.resident)))
        elif val.kind != plan_pb2.VALUE_KIND_INTERMEDIATE:
            raise ValueError(
                f"segment {seg.name!r}: unsupported write value kind {val.kind} for {val.name!r}"
            )

    out_names = _submodule_output_names(sub, seg.name)
    if len(out_names) != len(seg.writes):
        raise ValueError(
            f"segment {seg.name!r}: plan declares {len(seg.writes)} writes, "
            f"submodule produces {len(out_names)} outputs"
        )
    output_slots: list[int] = []
    for out_name in out_names:
        if out_name not in write_slots_by_name:
            raise ValueError(
                f"segment {seg.name!r}: submodule output {out_name!r} not found in plan writes"
            )
        output_slots.append(write_slots_by_name[out_name])
    return tuple(output_slots)


def _order_step_input_slots(
    split: torch.fx.GraphModule,
    step_input_by_name: Mapping[str, int],
    first_seen_slots: list[int],
) -> tuple[int, ...]:
    """Orders positional step-input slots to match split.graph's placeholder signature."""
    split_placeholders = [n.name for n in split.graph.nodes if n.op == "placeholder"]
    if not split_placeholders:
        return tuple(first_seen_slots)
    if set(split_placeholders) != set(step_input_by_name):
        raise ValueError(
            f"plan step inputs {sorted(step_input_by_name)} do not match "
            f"split graph placeholders {split_placeholders}"
        )
    return tuple(step_input_by_name[name] for name in split_placeholders)


def build_segment_runner(
    split: torch.fx.GraphModule,
    plan_file: plan_pb2.PlanFile,
    callables: Mapping[str, SegmentCallable],
    submodules: Mapping[str, torch.fx.GraphModule],
    attr_targets: Mapping[str, str],
    loaded: int,
) -> SegmentRunner:
    """Builds a SegmentRunner wired strictly from PlanFile.segments reads and writes."""
    if not plan_file.segments:
        raise ValueError("plan has no segments")

    table = _SlotTable()
    executions: list[SegmentExecution] = []

    for seg in plan_file.segments:
        if seg.name not in callables or seg.name not in submodules:
            raise ValueError(f"plan segment {seg.name!r} has no loaded callable")
        sub = submodules[seg.name]
        input_slots = _wire_segment_reads(table, split, seg, sub, attr_targets)
        output_slots = _wire_segment_writes(table, split, seg, sub)
        executions.append(
            SegmentExecution(
                name=seg.name,
                host=seg.host,
                callable=callables[seg.name],
                input_slots=input_slots,
                output_slots=output_slots,
            )
        )

    if not table.step_output_slots:
        raise ValueError("plan declares no step outputs")

    step_input_slots = _order_step_input_slots(
        split, table.step_input_by_name, table.first_seen_step_input_slots
    )
    return SegmentRunner(
        split=split,
        segments=tuple(executions),
        env=table.env,
        step_input_slots=step_input_slots,
        step_input_by_name=table.step_input_by_name,
        step_output_slots=tuple(table.step_output_slots),
        resident_commits=tuple(table.resident_commits),
        transient_slots=tuple(table.transient_slots),
        loaded=loaded,
    )
