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

"""The whole decode step in one CUDA graph, host ops through mailboxes (#372).

With the split decode run through the dataflow runner (dataflow.py) and
its dense segments as AOTInductor packages (aoti.py, segment_runtime
aoti), every dense node is native code that a CUDA graph can capture.
What still breaks a step into pieces is the host ops: today Python waits
for the GPU before each, runs it, and launches the next segment after,
and the GPU waits for Python in between (on DeepSeek-V4.1-Flash, 43 dense
segments take 20.7 ms a step in the engine and 11.3 ms replayed back to
back, #277).

Here a host op is a mailbox inside the graph instead:

  GPU   copy the node's inputs into pinned host inboxes, then write 1 to
        the node's to-CPU flag (cuStreamWriteValue32, a system-wide fence
        first), then wait for its from-CPU flag to be 1
        (cuStreamWaitValue32), clear it, and copy the reply from the
        pinned outbox into the consumers' inboxes on the GPU
  CPU   the step host, a native thread (csrc/node_server.cpp, #409),
        waits for each host node's to-CPU flag in turn, clears it, runs
        the node on the pinned inputs, writes the reply into the pinned
        outbox, then sets the from-CPU flag. A node the engine keeps runs
        as its native package (nodes.py: the host op lowered to the
        native kernels, its residents passed by address), one a node
        server holds through the native client (nodeserver.py): no
        Python between the GPU's flag and the reply

So a decode step is one graph replay: the GPU runs the dense segments
and blocks on the host ops' flags itself; Python launches the step and
waits for its end. The flags and buffers have fixed addresses and fixed
values, so the same graph replays every step.

Hot experts (a placement's GPU pools, experts_gpu.py) take part too. An
experts node whose layer has a pool on the engine's GPU runs the pool's
kernels inside the graph, on a forked stream, while the CPU worker
answers the pairs no GPU holds; a pool on another GPU runs in that GPU's
own graph, replayed alongside, with nothing in host memory: the engine's
GPU copies the node's inputs straight into an inbox on the other GPU and
writes a flag in that GPU's memory; the other GPU, waiting on the flag,
computes, copies its per-pair replies straight back and writes a flag in
the engine's GPU's memory. The engine's graph then takes each
pair's reply from whoever holds the expert and sums them in k order, as
the mailboxes do (mailboxes.py). The step host answers such a node's
other pairs with nodes.py's cold lowering, reading the pools' slot maps
in place.

The first decode step after a load runs through the dataflow runner
(loading the packages, allocating the inboxes); the second builds the
host nodes' packages, captures the step and replays it; later steps
replay. Telemetry records the host ops from the step host's clock (when
it saw each flag, when the reply was ready, when it raised the flag back)
after the step, and the steps; the segments' own times, which only a
node-by-node run can take, are left out.
"""

import ctypes
import gc
import weakref

import torch

from . import dataflow, kernels, nodes, triton_ops

CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0  # with a system-wide memory fence before the write


class _Driver:
    """The CUDA driver's stream memory operations, through ctypes."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.write = self.lib.cuStreamWriteValue32_v2
        self.wait = self.lib.cuStreamWaitValue32_v2
        for fn in (self.write, self.wait):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            fn.restype = ctypes.c_int
        self.copy = self.lib.cuMemcpyAsync
        self.copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        self.copy.restype = ctypes.c_int
        self.dptr = self.lib.cuMemHostGetDevicePointer_v2
        self.dptr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        self.dptr.restype = ctypes.c_int

    def device_address(self, pinned):
        out = ctypes.c_uint64()
        _check(self.dptr(ctypes.byref(out), ctypes.c_void_p(pinned.data_ptr()), 0), "cuMemHostGetDevicePointer")
        return out.value

    def write_value(self, stream, address, value):
        _check(self.write(ctypes.c_void_p(stream.cuda_stream), address, value, CU_STREAM_WRITE_VALUE_DEFAULT),
               "cuStreamWriteValue32")

    def memcpy(self, stream, dst, src):
        """dst <- src on stream, whichever devices they are on (unified
        addressing; peer access enabled): a copy node in a capture."""
        _check(self.copy(dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size(),
                         ctypes.c_void_p(stream.cuda_stream)), "cuMemcpyAsync")

    def wait_value(self, stream, address, value):
        _check(self.wait(ctypes.c_void_p(stream.cuda_stream), address, value, CU_STREAM_WAIT_VALUE_EQ),
               "cuStreamWaitValue32")

    def node_counts(self, graph):
        """Count the nodes in a CUDA graph by type (kernel, memcpy, memset, etc.)."""
        h_graph = graph.raw_cuda_graph() if hasattr(graph, "raw_cuda_graph") else graph
        h_graph = ctypes.c_void_p(h_graph)
        num_nodes = ctypes.c_size_t()
        if not hasattr(self, "graph_nodes"):
            self.graph_nodes = self.lib.cuGraphGetNodes
            self.graph_nodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            self.graph_nodes.restype = ctypes.c_int
            self.node_type = self.lib.cuGraphNodeGetType
            self.node_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.node_type.restype = ctypes.c_int
        _check(self.graph_nodes(h_graph, None, ctypes.byref(num_nodes)), "cuGraphGetNodes")
        if num_nodes.value == 0:
            return {"total": 0, "kernel": 0}
        nodes = (ctypes.c_void_p * num_nodes.value)()
        _check(self.graph_nodes(h_graph, nodes, ctypes.byref(num_nodes)), "cuGraphGetNodes")
        types_map = {
            0: "kernel",
            1: "memcpy",
            2: "memset",
            3: "host",
            4: "graph",
            5: "empty",
            6: "wait_event",
            7: "event_record",
            8: "ext_semas_signal",
            9: "ext_semas_wait",
            10: "mem_alloc",
            11: "mem_free",
            12: "batch_mem_op",
        }
        counts = {"total": num_nodes.value}
        nt = ctypes.c_int()
        for node in nodes:
            _check(self.node_type(node, ctypes.byref(nt)), "cuGraphNodeGetType")
            name = types_map.get(nt.value, f"type_{nt.value}")
            counts[name] = counts.get(name, 0) + 1
        return counts


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"step graph: {what} failed (CUresult {status})")


def _experts_call(sub):
    """For a host segment that calls the routed experts: (layer, the
    positions of x, weights and indices among the segment's inputs)."""
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    for n in sub.graph.nodes:
        if n.op == "call_function" and "moe_experts" in str(getattr(n.target, "__name__", n.target)):
            x, w, i, layer = n.args[:4]
            return int(layer), [placeholders.index(a) for a in (x, w, i)]
    return None, None


def _slot_maps(entry):
    """An experts node's GPU pools' slot maps (expert - start -> slot, -1
    where the pool does not hold it), from its pooled entry; None without
    pools."""
    if entry is None:
        return None
    _, here, peer_boxes, _, _ = entry
    return [pool.slot_cpu for pool in list(here) + [box[0] for box in peer_boxes.values()]]


class StepGraph(torch.nn.Module):
    """A dataflow runner's step as one CUDA graph (module docstring)."""

    def __init__(self, runner):
        super().__init__()
        self.runner = runner  # dataflow.Runner over a split whose dense segments are aoti.GraphedSegment
        self.split = runner.split
        self.vxpu_segments = runner.vxpu_segments
        self.graph = None
        self._peer_graphs = []
        self.steps = 0
        self.recorder = None
        self._host = None

    def set_recorder(self, recorder):
        self.recorder = recorder
        self.runner.set_recorder(recorder)

    def forward(self, *args, **kwargs):
        self.steps += 1
        if self.steps == 1:
            return self.runner(*args, **kwargs)  # the first step loads the packages and allocates the inboxes
        if self.graph is None:
            gc.disable()  # a collection inside a capture can unload a kernel's module, which a capture forbids
            try:
                self._capture(args, kwargs)
            finally:
                gc.enable()
        for buf, a in zip(self._inputs, self._input_values(args, kwargs)):
            buf.copy_(a)
        self._host.go()
        self.graph.replay()
        # the other GPUs' graphs after this one's: launched first, a graph
        # that opens by waiting on a flag held the launching thread ~5 ms
        for pdev, g in self._peer_graphs:
            with torch.cuda.device(pdev):
                g.replay()
        torch.cuda.current_stream().synchronize()
        for pdev, _ in self._peer_graphs:
            torch.cuda.current_stream(pdev).synchronize()
        error = self._host.take_error()
        if error:
            # the host raised every from-CPU flag to release the GPU, and
            # the to-CPU flags after the failed node are still up: the next
            # step starts clean (the GPU is idle and the host done)
            self._flags.zero_()
            raise RuntimeError(f"step graph: {error}")
        self._record()
        out = self._output
        if self.runner.graph.output_names is not None:  # a range's slice (#387): its outputs by name
            return {name: o.clone() for name, o in zip(self.runner.graph.output_names, out)}
        out = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        return out.clone() if isinstance(out, torch.Tensor) else out

    def _input_values(self, args, kwargs):
        values = dict(zip(self.runner.graph.inputs, args), **kwargs)
        return [values[name] for name in self.runner.graph.inputs]

    def _capture(self, args, kwargs):
        from . import ops
        graph, runner = self.runner.graph, self.runner
        driver = _Driver()
        self._inputs = [v.clone() for v in self._input_values(args, kwargs)]
        inputs = dict(zip(graph.inputs, self._inputs))
        host_nodes = [n for n in graph.nodes if n.kind != "dense"]
        produced = {}  # node -> its output tensors on the GPU, as captured
        dev = runner.device
        if dev.index is None:  # the pools name their GPU by index
            dev = torch.device(dev.type, torch.cuda.current_device())
        external = ops._active.external

        def resolve(r):
            if isinstance(r, (tuple, list)):
                return type(r)(resolve(x) for x in r)
            if not isinstance(r, dataflow.Ref):
                return r
            if r.kind == "input":
                return inputs[r.name]
            if r.kind == "resident":
                return dataflow._attr(self.split, r.name)
            out = produced[r.node]
            return out if r.index is None else out[r.index]

        # Every pinned buffer before the captures (allocating pinned memory
        # while a stream captures is not allowed), shaped like the inboxes
        # the runner's first step allocated. An experts node whose layer
        # has pools gets per-pair outboxes: one the CPU fills, one per
        # other GPU holding some of its experts.
        boxes, pooled, peers = {}, {}, []
        for node in host_nodes:
            dev_ = runner._devices[node.name]
            example = [runner._arg(r, dev_, inputs) for r in node.args]
            like = next((t for (p, _, _), t in runner.inboxes.items() if p == node.name), None)
            if like is None:  # its output only leaves the slice (#387): no inbox, shaped as returned
                like = runner.returned[node.name]
            out_like, peer_boxes = like, {}
            if node.kind == "experts":
                layer, where = _experts_call(getattr(self.split, node.name))
                res = getattr(external.experts.get(layer), "_vxpu_hot", None) if layer is not None else None
                if res is not None and res.pools:
                    x_, _, i_ = (example[k] for k in where)
                    out_like = torch.empty(i_.shape[0], i_.shape[1], x_.shape[1], dtype=torch.float32)
                    here = [p for p in res.pools if p.device == dev]
                    for pool in res.pools:
                        if pool.device != dev:
                            peers += [pool.device] if pool.device not in peers else []
                            # straight between the GPUs: the inputs into an inbox on the other GPU, its
                            # replies into one here, each followed by a flag in the receiver's memory
                            inbox = [torch.empty_like(example[k], device=pool.device) for k in where]
                            peer_boxes[pool.device] = (pool, inbox, torch.zeros(1, dtype=torch.int32, device=pool.device),
                                                       torch.empty_like(out_like, device=dev),
                                                       torch.zeros(1, dtype=torch.int32, device=dev),
                                                       pool.slot_cpu.to(dev))  # its slot map, here, for the combine
                    pooled[node.name] = (external.experts[layer], here, peer_boxes, where, layer)
            boxes[node.name] = ([torch.empty_like(a, device="cpu").pin_memory() if isinstance(a, torch.Tensor) else a
                                 for a in example],
                                torch.empty_like(out_like, device="cpu").pin_memory(), out_like)
        for pdev in peers:  # peer access, both ways, before any capture
            torch.empty(1, device=pdev).copy_(torch.empty(1, device=dev))
            torch.empty(1, device=dev).copy_(torch.empty(1, device=pdev))
        # per host node i: its to-CPU and from-CPU flags
        F = 2
        flags = torch.zeros(F * len(host_nodes), dtype=torch.int32, pin_memory=True)
        base = driver.device_address(flags)

        def flag(i, k):
            return base + 4 * (F * i + k)

        mailboxes = []  # per host node: (node, pinned inputs, kwargs, outbox, flag offset, pooled)
        side = torch.cuda.Stream(device=dev)  # this GPU's pools' kernels, forked from the step and joined back
        answers = self._answers(host_nodes, boxes, pooled, external)  # before any capture: builds packages
        self._warm(pooled, runner, inputs, dev)
        torch.cuda.synchronize()
        self._capture_all(graph, mailboxes, boxes, pooled, peers, driver, flags, F, flag, side, dev, resolve,
                          produced, host_nodes, answers)

    def _warm(self, pooled, runner, inputs, dev):
        """Every kernel the captures will launch, run once first on the
        runner's inboxes: Triton compiles (and loads) a kernel on its first
        call, which a capture forbids."""
        for name, (moe, here, peer_boxes, where, _) in pooled.items():
            node = self.runner.graph.by_name[name]
            example = [runner._arg(r, runner._devices[name], inputs) for r in node.args]
            x_, w_, i_ = (example[k] for k in where)
            start = moe.experts_start_idx
            local = None
            for pool in here:
                local = triton_ops.fp4_experts_decode(x_, w_, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3,
                                                      pool.w2, pool.s2, moe.swiglu_limit, per_pair=True,
                                                      indices=i_, start=start)
            peer = None
            for pdev, (pool, inbox, _, theirs, _, slot_here) in peer_boxes.items():
                with torch.cuda.device(pdev):
                    xs, ws, is_ = (t.to(pdev) for t in (x_, w_, i_))
                    triton_ops.fp4_experts_decode(xs, ws, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3,
                                                  pool.w2, pool.s2, moe.swiglu_limit, per_pair=True,
                                                  indices=is_, start=start)
                peer = (theirs, slot_here)
            cpu = torch.zeros(i_.shape[0], i_.shape[1], x_.shape[1], dtype=torch.float32, device=dev)
            if len(here) <= 1 and len(peer_boxes) <= 1:
                triton_ops.combine_replies(cpu, i_, start, local=local, local_map=here[0].slot_gpu if here else None,
                                           peer=peer[0] if peer else None, peer_map=peer[1] if peer else None)
        for pdev in {d for _, _, boxes_, _, _ in pooled.values() for d in boxes_}:
            torch.cuda.synchronize(pdev)

    def _answers(self, host_nodes, boxes, pooled, external):
        """How the step host answers each host node, by name: ("remote",
        RemoteNode) for a node a node server holds, else ("local",
        package path, residents), its native package built (or found in
        the package cache) here, before any capture."""
        if kernels.load() is None:
            raise RuntimeError(f"step graph: the step host needs the CPU kernels: {kernels.why_not()}")
        packages = next(m.packages for _, m in self.split.named_children() if hasattr(m, "packages"))
        remote_nodes = getattr(external, "remote_nodes", {})
        answers = {}
        for node in host_nodes:
            sub = getattr(self.split, node.name)
            pinned, _, _ = boxes[node.name]
            if node.kwargs or not all(isinstance(p, torch.Tensor) for p in pinned):
                raise RuntimeError(f"step graph: host node {node.name} takes more than tensors")
            remote = remote_nodes.get(nodes.host_call(sub))
            entry = pooled.get(node.name)
            if remote is not None:
                if entry is not None:
                    raise RuntimeError(f"step graph: {node.name} is on a node server and in GPU pools")
                answers[node.name] = ("remote", remote)
                continue
            path, residents = nodes.native_package(packages, sub, external, pinned, _slot_maps(entry))
            answers[node.name] = ("local", path, residents)
        return answers

    def _record(self):
        """The step's host ops into telemetry (#409), from the step host's
        clock and the GPU's:
          impl_ms        from the GPU's flag seen to the reply in the outbox
                         (the step host's work: the node, or the call)
          crossing_ms    the GPU's round trip (raising its flag to seeing the
                         reply's) less the step host's time between seeing
                         the flag and raising it back: both PCIe crossings
                         and the flags' notice, nothing else
        and for a node on a node server
          server_run_ms  the server running the node
          wire_ms        impl_ms less the server's run: the network both ways,
                         serialization, the server reading the request"""
        from . import ops
        rec = ops._recorder()
        if rec is None:
            return
        times = self._host.times()
        for (kind, layer, remote, i, slot_maps, start), t, (sent, back) in zip(self._box_info, times.tolist(),
                                                                             self._events):
            impl_ms = (t[1] - t[0]) / 1e6
            crossing_ms = sent.elapsed_time(back) - (t[2] - t[0]) / 1e6
            fields = {"impl_ms": impl_ms, "crossing_ms": crossing_ms}
            if remote:
                fields.update(server_run_ms=t[4] / 1e6, wire_ms=impl_ms - t[4] / 1e6)
            if kind == "engram":
                rec.host_op("engram_rows", total_ms=impl_ms, **{k: v for k, v in fields.items() if k != "impl_ms"})
                continue
            if remote:
                rec.host_op("moe_experts", remote_pairs=float(i.numel()), **fields)
            else:
                held = torch.zeros_like(i, dtype=torch.bool)
                for slots in slot_maps or ():
                    held |= slots[i - start] >= 0
                n_held = int(held.sum())
                rec.host_op("moe_experts", gpu_pairs=float(n_held), cpu_pairs=float(i.numel() - n_held), **fields)
            rec.selected_experts(layer, i.flatten().tolist())

    def _capture_all(self, graph, mailboxes, boxes, pooled, peers, driver, flags, F, flag, side, dev, resolve,
                     produced, host_nodes, answers):
        from . import aoti
        events = []  # per host node: (sent, back), timing events recorded in the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            stream = torch.cuda.current_stream()
            for node in graph.nodes:
                args_ = [resolve(r) for r in node.args]
                kwargs_ = {k: resolve(r) for k, r in node.kwargs.items()}
                if node.kind == "dense":
                    module = getattr(self.split, node.name)
                    if not isinstance(module, aoti.GraphedSegment) or module.runner is None:
                        raise RuntimeError(f"step graph: {node.name} is not a loaded AOTInductor segment")
                    produced[node.name] = module.runner(*args_)
                    continue
                i = len(mailboxes)
                pinned, outbox, like = boxes[node.name]
                moe, here, peer_boxes, where, _ = pooled.get(node.name, (None, [], {}, None, None))
                hot = []
                if here:
                    x_, w_, i_ = (args_[k] for k in where)
                    side.wait_stream(stream)
                    with torch.cuda.stream(side):
                        for pool in here:
                            hot.append((pool, triton_ops.fp4_experts_decode(
                                x_, w_, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3, pool.w2, pool.s2,
                                moe.swiglu_limit, per_pair=True, indices=i_, start=moe.experts_start_idx)))
                for pdev, (pool, inbox, flag_in, _, _, _) in peer_boxes.items():  # the other GPUs first: they
                    for dst, k in zip(inbox, where):                              # wait on copies, not on the CPU
                        driver.memcpy(stream, dst, args_[k])
                    driver.write_value(stream, flag_in.data_ptr(), 1)
                for p, a in zip(pinned, args_):
                    if isinstance(a, torch.Tensor):
                        p.copy_(a, non_blocking=True)
                sent, back = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                sent.record(stream)  # the GPU's side of the crossing: from raising its flag to seeing the reply's
                driver.write_value(stream, flag(i, 0), 1)
                driver.wait_value(stream, flag(i, 1), 1)
                driver.write_value(stream, flag(i, 1), 0)
                back.record(stream)
                events.append((sent, back))
                reply = torch.empty_like(like, device=dev)
                reply.copy_(outbox, non_blocking=True)
                if moe is not None:  # per-pair replies: each from whoever holds the expert, then the k-order sum
                    i_ = args_[where[2]]
                    start = moe.experts_start_idx
                    for pdev, (pool, _, _, theirs, flag_back, slot_here) in peer_boxes.items():
                        driver.wait_value(stream, flag_back.data_ptr(), 1)
                        driver.write_value(stream, flag_back.data_ptr(), 0)
                    if hot:
                        stream.wait_stream(side)
                    if len(hot) <= 1 and len(peer_boxes) <= 1:  # one kernel for the choice and the sum
                        peer = next(iter(peer_boxes.values()), None)
                        reply = triton_ops.combine_replies(
                            reply, i_, start, local=hot[0][1] if hot else None, local_map=hot[0][0].slot_gpu if hot else None,
                            peer=peer[3] if peer else None, peer_map=peer[5] if peer else None)
                    else:
                        for pool, _, _, theirs, _, slot_here in peer_boxes.values():
                            reply = torch.where((slot_here[i_ - start] >= 0).unsqueeze(-1), theirs, reply)
                        for pool, replies in hot:
                            reply = torch.where((pool.slot_gpu[i_ - start] >= 0).unsqueeze(-1), replies, reply)
                        y = torch.zeros(reply.shape[0], reply.shape[2], dtype=torch.float32, device=dev)
                        for k in range(reply.shape[1]):
                            y = y + reply[:, k]
                        reply = y
                produced[node.name] = reply
                mailboxes.append((node, pinned, kwargs_, outbox, F * i, pooled.get(node.name)))
            output = resolve(graph.output)
        peer_graphs = [self._capture_peer(pdev, peers.index(pdev), driver, flags, F, host_nodes, boxes, pooled)
                       for pdev in peers]
        host = kernels.load().StepHost(flags)
        info = []
        for node, pinned, _, outbox, _, entry in mailboxes:  # in step order: flags 2i and 2i+1
            answer = answers[node.name]
            kind, layer = nodes.host_call(getattr(self.split, node.name))
            if answer[0] == "remote":
                host.add_remote(pinned, outbox, answer[1].client, answer[1].index)
            else:
                host.add_local(pinned, outbox, answer[1], answer[2])
            start = entry[0].experts_start_idx if entry is not None else 0
            indices = pinned[_experts_call(getattr(self.split, node.name))[1][2]] if kind == "experts" else None
            info.append((kind, layer, answer[0] == "remote", indices, _slot_maps(entry), start))
        host.start()
        # all at once: a capture that fails leaves the step graph as it was, and the next step tries again
        self.graph, self._output, self._peer_graphs, self._host = g, output, peer_graphs, host
        self._flags, self._box_info, self._events = flags, info, events
        # what the graphs use that lives outside their memory pools (the pinned boxes, the other
        # GPUs' inboxes, flags and reply buffers, the slot maps' copies): freed, that memory would
        # be reused under the graphs. The Python worker held it through its mailboxes.
        self._keep = (boxes, pooled, mailboxes)
        weakref.finalize(self, host.stop)  # the engine dropped: the thread lets go of the model

    def _capture_peer(self, pdev, p, driver, flags, F, host_nodes, boxes, pooled):
        """Another GPU's part of the step, as its own graph: for each
        experts node with a pool there, in step order, wait for the node's
        flag in this GPU's memory, run the pool's kernels on the inbox the
        engine's GPU filled, copy the per-pair replies straight into the
        engine's GPU and raise the flag there."""
        with torch.cuda.device(pdev):
            work = []
            for node in host_nodes:
                if node.name in pooled and pdev in pooled[node.name][2]:
                    moe, _, peer_boxes, _, _ = pooled[node.name]
                    work.append((moe,) + tuple(peer_boxes[pdev][:5]))
            stream = torch.cuda.Stream(device=pdev)
            torch.cuda.synchronize(pdev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                s = torch.cuda.current_stream()
                for moe, pool, inbox, flag_in, theirs, flag_back in work:
                    driver.wait_value(s, flag_in.data_ptr(), 1)
                    driver.write_value(s, flag_in.data_ptr(), 0)
                    x_, w_, i_ = inbox
                    replies = triton_ops.fp4_experts_decode(x_, w_, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3,
                                                            pool.w2, pool.s2, moe.swiglu_limit, per_pair=True,
                                                            indices=i_, start=moe.experts_start_idx)
                    driver.memcpy(s, theirs, replies)
                    driver.write_value(s, flag_back.data_ptr(), 1)
        return pdev, g
