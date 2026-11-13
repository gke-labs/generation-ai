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

"""The decode step as a dataflow graph of nodes connected by mailboxes (#372).

A decode step split at its host ops (segments.py) is already a graph:
the dense segments and the host ops (the routed experts, the n-gram
lookup) are its nodes, the tensors one passes to the next are its edges.
Here that graph is made explicit, so that where each node runs, and how
each edge travels, become the executor's choices:

  node       a segment: dense (compiled or eager), experts or engram,
             with the decoder layers it covers
  edge       one output of a node read by another node: shape, dtype,
             bytes per step
  inbox      a buffer at the consumer, allocated once (fixed address):
             the producer writes its output into it, and the consumer
             fires when all of its inboxes are full
  resident   what a node reads that does not travel: the weights and the
             session state (caches), bound where the node runs
  placement  the device each node runs on

The Runner here is the reference: one thread, firing nodes in the order
their inboxes fill, every edge copied into its inbox. It computes what
the split module computes (the same segments, the same arguments), so
the engine can run its decode through it (plan choice dataflow) and the
selftest checks it token for token. The native runner (no Python inside
a step), the experts as nodes of their own, and transports between
devices and machines build on the same graph and inbox plan (#372).
"""

import collections
import operator
import os
from typing import TYPE_CHECKING, Any, Sequence

import torch

from . import segments

if TYPE_CHECKING:
    from . import stepgraph_pb2


def segment_writes(module: Any) -> list[int]:
    """Returns the positions of the arguments the segment writes in place,
    reached through views via segments._aliases, as sorted positions."""
    g: Any = getattr(module, "graph", module)
    placeholders: list[torch.fx.Node] = [n for n in g.nodes if n.op == "placeholder"]
    writes: set[int] = set()

    def trace_placeholder(start: torch.fx.Node) -> None:
        curr: Any = start
        while isinstance(curr, torch.fx.Node):
            if curr.op == "placeholder":
                if curr in placeholders:
                    writes.add(placeholders.index(curr))
                break
            if segments._aliases(curr) and curr.args and isinstance(curr.args[0], torch.fx.Node):
                curr = curr.args[0]
            else:
                break

    for node in g.nodes:
        if node.op == "call_function":
            written_nodes: list[torch.fx.Node] = []
            schema = getattr(node.target, "_schema", None)
            if schema and schema.arguments:
                for i, arg_schema in enumerate(schema.arguments):
                    if arg_schema.alias_info and arg_schema.alias_info.is_write:
                        if i < len(node.args) and isinstance(node.args[i], torch.fx.Node):
                            written_nodes.append(node.args[i])
            target_str = str(node.target)
            if not written_nodes and target_str.startswith("aten."):
                parts = target_str.split(".")
                if len(parts) > 1 and parts[1].endswith("_") and not parts[1].endswith("__"):
                    if node.args and isinstance(node.args[0], torch.fx.Node):
                        written_nodes.append(node.args[0])
            for arg_node in written_nodes:
                trace_placeholder(arg_node)
        elif node.op == "call_method":
            if isinstance(node.target, str) and node.target.endswith("_") and not node.target.endswith("__"):
                if node.args and isinstance(node.args[0], torch.fx.Node):
                    trace_placeholder(node.args[0])

    return sorted(writes)


def write_step_graph(
    path: str | os.PathLike[str],
    description: "stepgraph_pb2.StepGraphDescription",
) -> None:
    """Writes the StepGraphDescription as formatted JSON atomically.

    Writes through a temporary file created in the target directory and renames
    it so readers never see a partial document. Cleans up the temporary file on
    failure.
    """
    import tempfile
    from google.protobuf import json_format

    str_path = os.fspath(path)
    parent = os.path.dirname(str_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = json_format.MessageToJson(
        description, indent=2, preserving_proto_field_name=True
    )
    fd, tmp_path = tempfile.mkstemp(prefix="stepgraph-", suffix=".tmp", dir=parent or ".")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, str_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class Ref:
    """An argument of a node: a step input (a placeholder of the split),
    a resident tensor (an attribute of the split), or an edge (output
    `index` of node `node`, index None for a node's only output)."""

    __slots__ = ("kind", "name", "node", "index")

    def __init__(self, kind, name=None, node=None, index=None):
        self.kind, self.name, self.node, self.index = kind, name, node, index

    def __repr__(self):
        if self.kind == "edge":
            return f"{self.node}" + ("" if self.index is None else f"[{self.index}]")
        return f"{self.kind}:{self.name}"


class Node:
    """One segment of the step."""

    def __init__(
        self,
        name: str,
        kind: str,
        layers: Sequence[int],
        args: list[Any],
        kwargs: dict[str, Any],
        host: tuple[str, int] | None = None,
        writes: Sequence[int] = (),
        execution: str | None = None,
        answer: int | None = None,
    ) -> None:
        self.name = name
        self.kind = kind  # "dense", "experts" or "engram"
        self.layers = list(layers)
        self.args = args  # [Ref]
        self.kwargs = kwargs  # {name: Ref}
        self.host = host  # a host node's (collection, index): ("experts", 3) for layer 3's experts
        self.writes = tuple(writes)
        self.answer = answer
        if execution is None:
            self.execution = "exchange" if host is not None else "graph"
        elif execution in ("graph", "exchange"):
            self.execution = execution
        else:
            raise ValueError(f"node {name}: unknown execution {execution!r}, must be 'graph' or 'exchange'")
        if self.execution == "graph" and host is not None:
            raise ValueError(f"node {name}: graph node cannot have a host call")
        if self.execution == "exchange" and host is None:
            raise ValueError(f"node {name}: exchange node must have a host call")

    def refs(self) -> list[Any]:
        return list(self.args) + list(self.kwargs.values())

    def __repr__(self) -> str:
        return f"{self.name}({self.kind}, layers {self.layers})"


class Graph:
    """The nodes of a split decode step in call order, its step inputs,
    and the reference of what it returns."""

    def __init__(
        self,
        nodes: Sequence[Node],
        inputs: Sequence[str],
        output: Any,
        output_names: Sequence[str] | None = None,
    ) -> None:
        self.nodes: list[Node] = list(nodes)
        self.by_name: dict[str, Node] = {n.name: n for n in self.nodes}
        self.inputs: list[str] = list(inputs)
        self.output: Any = output  # Ref, or a tuple of Refs
        # a slice's (slice()) outputs by the name the next slice takes them as; None: the step's output
        self.output_names: list[str] | None = list(output_names) if output_names is not None else None

    @classmethod
    def from_split(cls, split: torch.nn.Module) -> "Graph":
        """The graph of a module split at its host ops (split_module's
        output: placeholders, get_attr, call_module per segment, getitem
        to unpack a segment's outputs, output)."""
        from . import nodes as native_nodes
        from .telemetry import segment_info
        info = {seg["name"]: seg for seg in segment_info(split)}
        refs: dict[torch.fx.Node, Ref] = {}
        nodes: list[Node] = []
        inputs: list[str] = []
        output = None

        def ref(arg: Any) -> Any:
            if isinstance(arg, torch.fx.Node):
                return refs[arg]
            if isinstance(arg, (tuple, list)):
                return type(arg)(ref(a) for a in arg)
            return arg  # a constant argument, passed as it is

        for fx in split.graph.nodes:
            if fx.op == "placeholder":
                inputs.append(fx.target)
                refs[fx] = Ref("input", name=fx.target)
            elif fx.op == "get_attr":
                refs[fx] = Ref("resident", name=fx.target)
            elif fx.op == "call_module":
                seg = info[fx.target]
                sub = getattr(split, fx.target)
                sub = getattr(sub, "_orig_mod", sub)
                host = native_nodes.host_call(sub) if native_nodes.is_native(sub) else None
                writes = segment_writes(sub)
                nodes.append(Node(fx.target, seg["kind"], seg["layers"],
                                  [ref(a) for a in fx.args], {k: ref(v) for k, v in fx.kwargs.items()}, host,
                                  writes=writes))
                refs[fx] = Ref("edge", node=fx.target)
            elif fx.op == "call_function" and fx.target is operator.getitem:
                base = refs[fx.args[0]]
                if base.kind != "edge" or base.index is not None:
                    raise ValueError(f"dataflow: {fx.name} unpacks something other than a segment's outputs")
                refs[fx] = Ref("edge", node=base.node, index=fx.args[1])
            elif fx.op == "output":
                output = ref(fx.args[0])
            else:
                raise ValueError(f"dataflow: unexpected top-level node {fx.op} {fx.target} in the split")
        return cls(nodes, inputs, output)

    def edges(self):
        """{(producer, index): [consumer names]}, in call order."""
        out = collections.OrderedDict()
        for node in self.nodes:
            for r in _flat(node.refs()):
                if isinstance(r, Ref) and r.kind == "edge":
                    out.setdefault((r.node, r.index), []).append(node.name)
        return out

    def cut_position(self, layer):
        """Where a range starting at experts layer `layer` begins: right
        after layer-1's experts node (cuts.py says where that is legal)."""
        for i, n in enumerate(self.nodes):
            if n.host == ("experts", layer - 1):
                return i + 1
        raise KeyError(f"dataflow: no experts node for layer {layer - 1}")

    def slice(self, start, end):
        """Nodes start..end-1 as a graph of their own (#387): an edge from
        a node before start becomes a step input, named after the value
        (cross:NODE or cross:NODE:INDEX); the outputs are every value made
        before end that the rest of the step reads (passing through values
        made before start, so slices chain), named the same way, or the
        step's output for the last slice."""
        inside = {n.name for n in self.nodes[start:end]}
        cross = {}

        def name(r):
            return f"cross:{r.node}" + ("" if r.index is None else f":{r.index}")

        def rewrite(r):
            if isinstance(r, (tuple, list)):
                return type(r)(rewrite(x) for x in r)
            if isinstance(r, Ref) and r.kind == "edge" and r.node not in inside:
                cross[(r.node, r.index)] = name(r)
                return Ref("input", name=name(r))
            return r

        nodes = [Node(n.name, n.kind, n.layers, [rewrite(a) for a in n.args],
                      {k: rewrite(v) for k, v in n.kwargs.items()}, n.host, writes=n.writes,
                      execution=n.execution) for n in self.nodes[start:end]]
        if end >= len(self.nodes):
            output, names = rewrite(self.output), None
        else:
            made_before = {n.name for n in self.nodes[:end]}
            needed = []
            for r in _flat([n.refs() for n in self.nodes[end:]] + [self.output]):
                if isinstance(r, Ref) and r.kind == "edge" and r.node in made_before and (r.node, r.index) not in needed:
                    needed.append((r.node, r.index))
            output = tuple(rewrite(Ref("edge", node=p, index=i)) for p, i in needed)
            names = [name(Ref("edge", node=p, index=i)) for p, i in needed]
        used = {r.name for n in nodes for r in _flat(n.refs()) if isinstance(r, Ref) and r.kind == "input"}
        inputs = [i for i in self.inputs if i in used] + sorted(set(cross.values()))
        return Graph(nodes, inputs, output, names)

    def residents(self):
        """The resident tensors (weights, state) this graph's nodes read."""
        return {r.name for n in self.nodes for r in _flat(n.refs()) if isinstance(r, Ref) and r.kind == "resident"}

    def description(
        self,
        host_mappings: Sequence["stepgraph_pb2.HostMapping"] | None = None,
    ) -> "stepgraph_pb2.StepGraphDescription":
        """Returns the StepGraphDescription protobuf message for this graph."""
        from . import stepgraph_pb2

        def to_value_ref(r: Ref) -> stepgraph_pb2.ValueRef:
            if r.kind == "input":
                return stepgraph_pb2.ValueRef(input=r.name)
            if r.kind == "resident":
                return stepgraph_pb2.ValueRef(resident=r.name)
            if r.kind == "host_mapping":
                return stepgraph_pb2.ValueRef(host_mapping=r.name)
            if r.kind == "edge":
                edge = stepgraph_pb2.EdgeRef(node=r.node)
                if r.index is not None:
                    edge.index = int(r.index)
                return stepgraph_pb2.ValueRef(edge=edge)
            raise ValueError(f"dataflow: unexpected ref kind {r.kind!r}")

        nodes: list[stepgraph_pb2.StepNode] = []
        for n in self.nodes:
            reads = [to_value_ref(r) for r in _flat(n.refs()) if isinstance(r, Ref)]
            if n.host is not None:
                from . import ops
                reads.append(stepgraph_pb2.ValueRef(host_mapping=ops.External.host_mapping_name(n.host[0], n.host[1])))
            writes: list[str] = []
            for pos in n.writes:
                if pos >= len(n.args):
                    raise ValueError(f"node {n.name!r} writes argument {pos}, but has only {len(n.args)} arguments")
                arg = n.args[pos]
                arg_refs = list(_flat([arg]))
                for r in arg_refs:
                    if not isinstance(r, Ref) or r.kind != "resident":
                        raise ValueError(f"node {n.name!r} writes {r!r} at position {pos}, which is not a resident tensor")
                    if r.name is None:
                        raise ValueError(f"node {n.name!r} writes unnamed resident at position {pos}")
                    if r.name not in writes:
                        writes.append(r.name)
            host_call = None
            if n.host is not None:
                host_call = stepgraph_pb2.HostCall(collection=str(n.host[0]), index=int(n.host[1]))
            if n.execution == "exchange":
                exec_enum = stepgraph_pb2.Execution.EXECUTION_EXCHANGE
                answer = n.answer if n.answer is not None else stepgraph_pb2.Answer.ANSWER_UNSPECIFIED
            elif n.execution == "graph":
                exec_enum = stepgraph_pb2.Execution.EXECUTION_GRAPH
                answer = stepgraph_pb2.Answer.ANSWER_UNSPECIFIED
            else:
                raise ValueError(f"node {n.name}: invalid execution {n.execution!r}")
            nodes.append(stepgraph_pb2.StepNode(
                name=n.name,
                kind=n.kind,
                execution=exec_enum,
                layers=list(n.layers),
                reads=reads,
                writes=writes,
                host=host_call,
                answer=answer,
            ))

        outputs = [to_value_ref(r) for r in _flat([self.output]) if isinstance(r, Ref)]
        desc = stepgraph_pb2.StepGraphDescription(
            nodes=nodes,
            inputs=list(self.inputs),
            outputs=outputs,
        )
        if host_mappings:
            desc.host_mappings.extend(host_mappings)
        return desc

    def describe(self):
        kinds = collections.Counter(n.kind for n in self.nodes)
        return (f"dataflow: {len(self.nodes)} nodes ("
                + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + f"), {len(self.edges())} edges")


class Runner(torch.nn.Module):
    """The reference dataflow runner for one split decode step: inboxes
    allocated on the first step (at the consumer's device), then every
    step each node's outputs are written into its consumers' inboxes, and
    a node fires once all of them are full.

    Stands in for segments.Split in the engine (the same split, the same
    recorder hooks and calling convention)."""

    def __init__(self, split, total, compiled, placement=None, device=None, graph=None):
        super().__init__()
        self.split = split  # the modules and the resident tensors live here
        self.vxpu_segments = (total, compiled)
        self.graph = graph if graph is not None else Graph.from_split(split)  # or a slice of it
        self.placement = dict(placement or {})  # node name -> device; default: where its resident tensors are (place)
        self.device = torch.device(device) if device is not None else None
        self.recorder = None
        self.inboxes = {}  # {(producer, index, device): tensor}, each allocated the first time it is written
        self.returned = {}  # the nodes that fed the last step's output, and what they made
        self._index = {name: i for i, (name, _) in enumerate(segments.segments_of(split))}
        self.place(self.placement)

    def place(self, placement):
        """Run each node of placement ({node name: device}) there, the rest
        where their resident tensors are. The inboxes are allocated again
        by the next step, at the new consumers' devices. A node's resident
        tensors are not moved: placing a node away from its weights is for
        nodes that hold none of their own (the host ops)."""
        self.placement = dict(placement)
        self.inboxes = {}
        self._devices = {n.name: self._device_of(n) for n in self.graph.nodes}
        # per producer: [(index, consumer devices, [(consumer, uses)])]
        uses = collections.Counter()
        for node in self.graph.nodes:
            for r in _flat(node.refs()):
                if isinstance(r, Ref) and r.kind == "edge":
                    uses[(r.node, r.index, node.name)] += 1
        self._waiting = collections.Counter()
        self._fanout = collections.defaultdict(list)
        for (producer, index), consumers in self.graph.edges().items():
            consumers = list(dict.fromkeys(consumers))
            devices = list(dict.fromkeys(self._devices[c] for c in consumers))
            self._fanout[producer].append((index, devices, [(c, uses[(producer, index, c)]) for c in consumers]))
            for c in consumers:
                self._waiting[c] += uses[(producer, index, c)]
        self._returns = {r.node for r in _flat([self.graph.output]) if isinstance(r, Ref) and r.kind == "edge"}

    def set_recorder(self, recorder):
        """Time every node into recorder (telemetry.Recorder), as Split does."""
        self.recorder = recorder

    def _device_of(self, node):
        if node.name in self.placement:
            return torch.device(self.placement[node.name])
        for r in _flat(node.refs()):
            if isinstance(r, Ref) and r.kind == "resident":
                t = _attr(self.split, r.name)
                if isinstance(t, torch.Tensor):
                    return t.device
        return self.device or torch.device("cpu")

    def inbox_plan(self):
        """[{producer, index, device, shape, dtype, bytes}] for every
        inbox allocated so far (all of them after a step)."""
        return [{"producer": p, "index": i, "device": str(d), "shape": list(t.shape), "dtype": str(t.dtype),
                 "bytes": t.numel() * t.element_size()} for (p, i, d), t in self.inboxes.items()]

    def forward(self, *args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()  # as Split: a CUDA-graph replay may reuse its outputs
        recorder = self.recorder
        if recorder is not None:
            recorder.begin_step()
        inputs = dict(zip(self.graph.inputs, args), **kwargs)
        waiting = collections.Counter(self._waiting)
        ready = collections.deque(n.name for n in self.graph.nodes if not waiting[n.name])
        returned = {}
        fired = 0
        while ready:
            node = self.graph.by_name[ready.popleft()]
            dev = self._devices[node.name]
            out = getattr(self.split, node.name)(*[self._arg(r, dev, inputs) for r in node.args],
                                                 **{k: self._arg(r, dev, inputs) for k, r in node.kwargs.items()})
            fired += 1
            if recorder is not None:
                recorder.after_segment(self._index[node.name])
            if node.name in self._returns:
                returned[node.name] = out
            for index, devices, consumers in self._fanout[node.name]:
                value = out if index is None else out[index]
                for d in devices:  # one inbox per consumer device
                    key = (node.name, index, d)
                    if not isinstance(value, torch.Tensor):  # a number a segment computed (a prefill's length)
                        self.inboxes[key] = value
                        continue
                    # allocated once, then the same address every step (a decode step's shapes never
                    # change; a prefill's, its length, do: then again)
                    if key not in self.inboxes or self.inboxes[key].shape != value.shape:
                        self.inboxes[key] = torch.empty_like(value, device=d)
                    self.inboxes[key].copy_(value)
                for consumer, n in consumers:
                    waiting[consumer] -= n
                    if not waiting[consumer]:
                        ready.append(consumer)
        if fired != len(self.graph.nodes):
            raise RuntimeError(f"dataflow: {len(self.graph.nodes) - fired} nodes never had all their inputs")
        if recorder is not None:
            recorder.end_step()
        self.returned = returned  # what fed the step's output: shapes for a node with no inbox (a slice's last)
        result = self._resolve(self.graph.output, returned, inputs)
        if self.graph.output_names is not None:  # a slice: its outputs by name, for the next
            return dict(zip(self.graph.output_names, result))
        result = result[0] if isinstance(result, (tuple, list)) and len(result) == 1 else result
        return result.clone() if isinstance(result, torch.Tensor) else result

    def _arg(self, r, device, inputs):
        if isinstance(r, (tuple, list)):
            return type(r)(self._arg(x, device, inputs) for x in r)
        if not isinstance(r, Ref):
            return r
        if r.kind == "input":
            return inputs[r.name]
        if r.kind == "resident":
            return _attr(self.split, r.name)
        return self.inboxes[(r.node, r.index, device)]

    def _resolve(self, out, returned, inputs):
        if isinstance(out, (tuple, list)):
            return type(out)(self._resolve(o, returned, inputs) for o in out)
        if isinstance(out, Ref) and out.kind == "edge":
            value = returned[out.node]
            return value if out.index is None else value[out.index]
        if isinstance(out, Ref) and out.kind == "input":  # a value passing through a slice
            return inputs[out.name]
        return out


class Chain(torch.nn.Module):
    """A step run as ranges of whole layers in turn (#387), each a slice of
    one split graph: the first range takes the step's inputs, each next one
    the step's inputs and what the ranges before it made for it, the last
    returns the step's output. In one process here; a range on another
    machine takes the same inputs and returns the same outputs. starts: the
    layers the ranges after the first begin at (where cuts.py allows)."""

    def __init__(self, split, starts, device=None):
        super().__init__()
        graph = Graph.from_split(split)
        bounds = [0] + [graph.cut_position(layer) for layer in starts] + [len(graph.nodes)]
        self.split, self.inputs = split, graph.inputs
        self.runners = torch.nn.ModuleList(Runner(split, len(graph.nodes), 0, device=device, graph=graph.slice(a, b))
                                           for a, b in zip(bounds, bounds[1:]))

    def forward(self, *args, **kwargs):
        step = dict(zip(self.inputs, args), **kwargs)
        made = {}
        for runner in self.runners:
            out = runner(**{name: step[name] if name in step else made[name] for name in runner.graph.inputs})
            if isinstance(out, dict):
                made.update(out)
        return out


def _flat(refs):
    for r in refs:
        if isinstance(r, (tuple, list)):
            yield from _flat(r)
        else:
            yield r


def _attr(module, target):
    """The attribute a get_attr node reads (a dotted path, read each step:
    the engine may re-point it)."""
    obj = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj
