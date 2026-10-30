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
import contextlib
import operator

import torch

from . import segments


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

    def __init__(self, name, kind, layers, args, kwargs, host=None):
        self.name = name
        self.kind = kind  # "dense", "experts" or "engram"
        self.layers = layers
        self.args = args  # [Ref]
        self.kwargs = kwargs  # {name: Ref}
        self.host = host  # a host node's (collection, index): ("experts", 3) for layer 3's experts

    def refs(self):
        return list(self.args) + list(self.kwargs.values())

    def __repr__(self):
        return f"{self.name}({self.kind}, layers {self.layers})"


class Graph:
    """The nodes of a split decode step in call order, its step inputs,
    and the reference of what it returns."""

    def __init__(self, nodes, inputs, output, output_names=None):
        self.nodes = nodes
        self.by_name = {n.name: n for n in nodes}
        self.inputs = inputs
        self.output = output  # Ref, or a tuple of Refs
        # a slice's (slice()) outputs by the name the next slice takes them as; None: the step's output
        self.output_names = output_names

    @classmethod
    def from_split(cls, split):
        """The graph of a module split at its host ops (split_module's
        output: placeholders, get_attr, call_module per segment, getitem
        to unpack a segment's outputs, output)."""
        from . import nodes as native_nodes
        from .telemetry import segment_info
        info = {seg["name"]: seg for seg in segment_info(split)}
        refs, nodes, inputs, output = {}, [], [], None

        def ref(arg):
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
                host = native_nodes.host_call(sub) if seg["kind"] != "dense" and native_nodes.is_native(sub) else None
                nodes.append(Node(fx.target, seg["kind"], seg["layers"],
                                  [ref(a) for a in fx.args], {k: ref(v) for k, v in fx.kwargs.items()}, host))
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
                      {k: rewrite(v) for k, v in n.kwargs.items()}, n.host) for n in self.nodes[start:end]]
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
        self._copies = {}  # {(resident name, device): its copy there}, for a resident a node on another GPU reads
        self._input_copies = {}  # {(step input name, device): its copy there}, filled once a step
        self._step_inputs = {}
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
        """The device a node runs on, a CUDA device always with its index
        (the inboxes are keyed by it, and "cuda" and "cuda:0" are not equal)."""
        d = None
        if node.name in self.placement:
            d = torch.device(self.placement[node.name])
        else:
            for r in _flat(node.refs()):
                if isinstance(r, Ref) and r.kind == "resident":
                    t = _attr(self.split, r.name)
                    if isinstance(t, torch.Tensor):
                        d = t.device
                        break
        d = d or self.device or torch.device("cpu")
        if d.type == "cuda" and d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        return d

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
        self._step_inputs = {}
        waiting = collections.Counter(self._waiting)
        ready = collections.deque(n.name for n in self.graph.nodes if not waiting[n.name])
        returned = {}
        fired = 0
        while ready:
            node = self.graph.by_name[ready.popleft()]
            dev = self._devices[node.name]
            args_ = [self._arg(r, dev, inputs) for r in node.args]
            kwargs_ = {k: self._arg(r, dev, inputs) for k, r in node.kwargs.items()}
            try:
                # on the node's GPU as the current device: the graphs were exported on the meta
                # device and retargeted to "cuda", so their tensor factories (arange, ones) land on
                # the current device
                with torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext():
                    out = getattr(self.split, node.name)(*args_, **kwargs_)
            except RuntimeError as e:
                where = [f"{r.kind}:{r.name or r.node}@{a.device}" for r, a in
                         list(zip(node.args, args_)) + list(zip(node.kwargs.values(), kwargs_.values()))
                         if isinstance(a, torch.Tensor)]
                raise RuntimeError(f"dataflow: node {node.name} ({node.kind}, layers {node.layers}) on {dev} "
                                   f"failed with its arguments {where}: {e}") from e
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
            v = inputs[r.name]
            if isinstance(v, torch.Tensor) and v.is_cuda and device.type == "cuda" and v.device != device:
                key = (r.name, device)  # a step input a node on another GPU reads: its copy there, once a step
                if key not in self._step_inputs:
                    buf = self._input_copies.get(key)
                    if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                        buf = self._input_copies[key] = torch.empty_like(v, device=device)
                    buf.copy_(v)
                    self._step_inputs[key] = buf
                return self._step_inputs[key]
            return v
        if r.kind == "resident":
            t = _attr(self.split, r.name)
            if isinstance(t, torch.Tensor) and t.is_cuda and device.type == "cuda" and t.device != device:
                key = (r.name, device)  # a table or norm weight shared by layers on different GPUs
                if key not in self._copies:
                    self._copies[key] = t.to(device)
                return self._copies[key]
            return t
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
