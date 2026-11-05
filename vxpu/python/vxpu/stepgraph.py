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

import collections
import os
import ctypes
import gc
import threading
import time
import weakref

import torch

from . import dataflow, kernels, nodes, triton_ops

CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0  # with a system-wide memory fence before the write


EXPERT_STATS = os.environ.get("VXPU_EXPERT_STATS", "") == "1"  # count the distinct experts a step (measurement)
PARALLEL_BRANCHES = os.environ.get("VXPU_PARALLEL_BRANCHES", "1") != "0"  # A/B: 0 keeps every node on one stream
# Islands and host nodes may run at once (the rule in branch_choice; hero run 15): the shared expert's front beside the
# routed experts' kernels. VXPU_HOST_BRANCH_APART=1 is the debug switch back to hero run 10's blanket serialization.
HOST_BRANCH_APART = os.environ.get("VXPU_HOST_BRANCH_APART", "0") == "1"
# The expert kernels' scratch held for the capture's lifetime (triton_ops.CAPTURE_KEEP); VXPU_CAPTURE_KEEP=0 frees it at
# each call as before, the A/B that shows the pool's reuse under the overlap.
CAPTURE_KEEP = os.environ.get("VXPU_CAPTURE_KEEP", "1") != "0"
# The two-part push sum (hero run 15): at a mixing sum whose partial is the routed experts' sum plus the shared expert's
# GEMV (the FFN's), the experts host node pushes its partial to the peers from its kernel's epilogue and the shared
# expert's GEMV, on the parallel branch, pushes its own; the sum adds both in the fused push's order. Nothing of the
# step then waits for the shared expert but the sum (the GEMV inside the collective was the 0.55 ms a step still
# exposed after the island). Off until measured; VXPU_TWO_PART_SUM=1 turns it on.
TWO_PART_SUM = os.environ.get("VXPU_TWO_PART_SUM", "0") == "1"
# The step feeds itself (hero run 15, the sampling hop): with VXPU_GRAPH_TOKEN=1 a plain decode step's graph
# writes its greedy choice into its own input_ids and advances its cache_position at the end, so the next
# replay needs no token from the host (relaunch); the host reads the choice back a step behind. On by default
# (VXPU_GRAPH_TOKEN=0 the A/B back): -0.5 ms a step at 1, 5 and 8 rows with identical tokens (hero run 15).
GRAPH_TOKEN = os.environ.get("VXPU_GRAPH_TOKEN", "1") == "1"
STEP_COPIES = os.environ.get("VXPU_STEP_COPIES") == "1"  # list every copy the captured step makes between GPUs
CACHE_WINDOWS = os.environ.get("VXPU_CACHE_WINDOWS", "1") != "0"  # A/B: 0 refreshes shared-cache replicas whole, as before
COPY_LOG = []  # (what, name, src GPU, dst GPU, shape, dtype, bytes) for the step's GPU-to-GPU copies, filled at capture
BRANCH_TRACE = os.environ.get("VXPU_BRANCH_TRACE") == "1"  # print each node's stream and the events it waits on (capture)
EXPERT_STATS_LAYERS, EXPERT_STATS_EXPERTS = 64, 1024  # slots for layers and expert ids
# VXPU_EXPERT_TRACE=1: keep every experts node's selected ids per step (which expert each token row chose,
# per layer), for the planner's offline evaluation of expert-aware speculation (hero run 13); a measurement
# switch like EXPERT_STATS, with a fixed buffer per GPU (steps x ids a step) allocated before any capture
EXPERT_TRACE = os.environ.get("VXPU_EXPERT_TRACE", "") == "1"
EXPERT_TRACE_STEPS, EXPERT_TRACE_IDS = 256, 1024  # steps kept (the last slot is overwritten past it) and ids a step


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
        self.nodes = self.lib.cuGraphGetNodes
        self.nodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        self.nodes.restype = ctypes.c_int
        self.node_type = self.lib.cuGraphNodeGetType
        self.node_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.node_type.restype = ctypes.c_int
        self.kernel_params = self.lib.cuGraphKernelNodeGetParams_v2
        self.kernel_params.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.kernel_params.restype = ctypes.c_int
        self.func_name = self.lib.cuFuncGetName
        self.func_name.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]
        self.func_name.restype = ctypes.c_int
        self.kernel_name = getattr(self.lib, "cuKernelGetName", None)
        if self.kernel_name is not None:
            self.kernel_name.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]
            self.kernel_name.restype = ctypes.c_int

    # CUgraphNodeType, the kinds a step's graph holds
    NODE_KINDS = {0: "kernels", 1: "copies", 2: "memsets", 12: "mailbox operations"}

    def node_counts(self, graph, names=None):
        """{kind: count} of a captured graph's nodes (cudaGraph_t, as an int): its kernels,
        copies, memsets and stream memory operations (the mailboxes' flags), and anything else.
        With names (a dict), each kernel's function name is counted into it too."""
        count = ctypes.c_size_t(0)
        _check(self.nodes(ctypes.c_void_p(graph), None, ctypes.byref(count)), "cuGraphGetNodes")
        nodes = (ctypes.c_void_p * count.value)()
        _check(self.nodes(ctypes.c_void_p(graph), nodes, ctypes.byref(count)), "cuGraphGetNodes")
        out = {}
        for node in nodes:
            kind = ctypes.c_int(0)
            _check(self.node_type(node, ctypes.byref(kind)), "cuGraphNodeGetType")
            name = self.NODE_KINDS.get(kind.value, "other")
            out[name] = out.get(name, 0) + 1
            if names is not None and kind.value == 0:
                fn = self._kernel_name(node)
                names[fn] = names.get(fn, 0) + 1
        return out

    def _kernel_name(self, node):
        """A kernel node's function name (CUDA_KERNEL_NODE_PARAMS_v2: the CUfunction first, the
        CUkernel after the grid, block, shared memory and parameter fields)."""
        params = (ctypes.c_byte * 256)()
        if self.kernel_params(node, ctypes.byref(params)) != 0:
            return "?"
        func = ctypes.c_void_p.from_buffer(params, 0).value
        out = ctypes.c_char_p()
        if func and self.func_name(ctypes.byref(out), ctypes.c_void_p(func)) == 0 and out.value:
            return out.value.decode()
        kern = ctypes.c_void_p.from_buffer(params, 8 + 7 * 4 + 4 + 2 * 8).value  # 8-aligned after 7 uints
        if kern and self.kernel_name is not None and self.kernel_name(ctypes.byref(out), ctypes.c_void_p(kern)) == 0:
            return out.value.decode() if out.value else "?"
        return "?"

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


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"step graph: {what} failed (CUresult {status})")


def _ring_ok(value, inbox):
    """Whether the ring write (triton_ops.ring_write) can carry value into inbox: on (RING_WRITE),
    both contiguous CUDA tensors of the same bytes, a multiple of 4."""
    return (triton_ops is not None and triton_ops.RING_WRITE and triton_ops.triton is not None and value.is_cuda
            and inbox.is_cuda and value.is_contiguous() and inbox.is_contiguous()
            and value.numel() * value.element_size() == inbox.numel() * inbox.element_size()
            and (value.numel() * value.element_size()) % 4 == 0)


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


TOP_CANDIDATES = 1024  # the tokens top-k and top-p choose among
_seeds = {}  # device -> the sampler's seed counter (ops.choose_argmax), stepped by every choose


def _seed(device):
    if device.type == "cuda" and device.index is None:  # "cuda" and "cuda:0" are one key
        device = torch.device("cuda", torch.cuda.current_device())
    if device not in _seeds:
        _seeds[device] = torch.randint(0, 2 ** 31, (1,), dtype=torch.int64, device=device)
    return _seeds[device]


def choose(logits, temperature, top_k, top_p):
    """Each position's token from logits [rows, tokens, vocab], in the graph, by each row's
    sampling settings: the argmax where its temperature is 0; else a sample of softmax(logits /
    temperature), by Gumbel-max (the argmax of the scaled logits plus Gumbel noise), over the whole
    vocabulary, or with top_k > 0 or top_p < 1 over the TOP_CANDIDATES most likely tokens kept by
    top-k (the k most likely) and then top-p (the most likely, until their mass reaches top_p), as
    transformers' processors keep them. Returns [rows, tokens] int64."""
    from . import ops
    seed = _seed(logits.device)
    seed.add_(1)  # in the graph: every replay draws fresh noise
    greedy, full = ops.choose_argmax(logits, temperature, seed)  # argmax, and Gumbel-max over the whole vocabulary
    def gumbel(like):
        return -torch.log(-torch.log(torch.rand_like(like).clamp_(min=1e-20)))
    # the candidates from the logits as stored (bf16: half the radix passes of the fp32 scaled
    # values), scaled after: a positive temperature keeps their order, and the values are the same
    vals, idx = logits.topk(min(TOP_CANDIDATES, logits.shape[-1]), dim=-1)  # most likely first
    vals = vals / temperature.clamp_min(1e-5).view(-1, 1, 1)
    ranks = torch.arange(vals.shape[-1], device=vals.device)
    k = top_k.view(-1, 1, 1)
    vals = vals.masked_fill((k > 0) & (ranks >= k), float("-inf"))
    probs = vals.softmax(-1)
    vals = vals.masked_fill(probs.cumsum(-1) - probs > top_p.view(-1, 1, 1), float("-inf"))
    kept = idx.gather(-1, (vals + gumbel(vals)).argmax(-1, keepdim=True)).squeeze(-1)
    sampled = torch.where(((top_k > 0) | (top_p < 1)).view(-1, 1), kept, full)
    return torch.where((temperature > 0).view(-1, 1), sampled, greedy)


def _covers(partial, moe, here, peer_boxes):
    """Whether the node's pairs never need the host: every expert of the layer is in a GPU pool, or the
    plan's expert_ep says this GPU answers for its held pairs alone and the step's reduce adds the
    other GPUs' partial sums (hero run 14)."""
    return bool(partial) or _all_held(moe, here, peer_boxes)


def _all_held(moe, here, peer_boxes):
    """Whether the GPU pools (this GPU's and the other GPUs') hold every expert of the node's
    layer, so no pair can fall to the host."""
    del moe
    maps = [pool.slot_cpu for pool in here] + [box[0].slot_cpu for box in peer_boxes.values()]  # CPU copies: this
    if not maps:                                                                                 # runs inside a capture
        return False
    held = torch.zeros_like(maps[0], dtype=torch.bool)  # a slot map has one entry per expert of the layer
    for m in maps:
        held |= m >= 0
    return bool(held.all())


def _check_flat(inputs, out, what):
    """The native loop copies a step's inputs and output as flat arrays (node_server.cpp's
    Graphs::run): each must be contiguous, or rows read each other's values."""
    for name, t in list(inputs.items()) + [("output", out)]:
        if not t.is_contiguous():
            raise RuntimeError(f"native loop: the {what} step's {name} is not contiguous "
                               f"(shape {tuple(t.shape)}, stride {t.stride()}); the loop copies it flat")


def compressed_cache(name, state):
    """Whether a shared cache is written by a running compressor (a KV group whose source layer
    keeps compressor state: layers 2, 8 and 14 of DeepSeek-V4.1-Flash): its slots are compressed
    positions, so the step's window at the token positions is not what the step wrote. Layer
    20's group has no compressor and its cache takes the window (hero run 9, verified token for
    token); layer 8's replicated across a cut at layer 9 took the window too and decoded 28 of
    33 rows wrong (hero run 12), and is refreshed whole."""
    prefix = name.rsplit(".attn.", 1)[0] if ".attn." in name else None
    return prefix is not None and f"{prefix}.attn.compressor.kv_state" in state


def positional_cache(src, rows, toks, min_positions=64):
    """Whether a shared state tensor is a cache indexed [row, position, ...] whose replica the
    step may refresh by the window it wrote (CACHE_WINDOWS): its first dim is the rows, its
    second the positions, with at least min_positions of them. A state shaped [rows, 2, dim]
    (a compressor's running state at a KV group's source layer) is not positional: the window
    gather at the clamped position copied one of its two slots and left the replica wrong, which
    made every row decode wrong when a range started one layer after a source layer (hero run
    12's cut at layer 9); such a state is refreshed whole, as before the windows."""
    return (src.dim() >= 3 and src.shape[0] == rows and src.shape[1] > toks
            and src.shape[1] >= min_positions)


def branch_choice(kind, producers, main_last, main_last_kind, needs_wave, enabled=None, apart=None):
    """Where a node of the step runs and what it waits for, besides the nodes it reads. The rule
    (hero run 15, the user's words): two nodes may not run at once if they touch the same memory
    with at least one writing and no atomics. A node's reads are its producers' outputs and the
    resident tensors; its writes are its own outputs, fresh for every node, and for a host node
    also its mailbox, its ring inboxes and its flags, which no other node touches (the flags by
    atomics), and the expert kernels' scratch, which the capture keeps alive for its lifetime
    (triton_ops.CAPTURE_KEEP) so that the pool never hands the same memory to another node's
    kernels while those run. So the sets of two nodes conflict only through a producer edge, and
    a dense node that does not read the node captured last on the main stream, and waits on no
    other GPU's wave, runs on the parallel branch whatever that node's kind. `apart`
    (HOST_BRANCH_APART, the debug switch) is hero run 10's blanket rule instead: an island and a
    host node never run at once (a node on the branch right after a host node waits for it, and a
    host node waits for the whole branch), which hid the scratch's reuse at nine rows before it
    was named. Returns (on_par, wait_main_last, host_waits_branch)."""
    if enabled is None:
        enabled = PARALLEL_BRANCHES
    if apart is None:
        apart = HOST_BRANCH_APART
    on_par = bool(enabled and kind == "dense" and main_last is not None and main_last not in producers
                  and not needs_wave)
    wait_main_last = on_par and main_last is not None and main_last_kind != "dense" and apart
    host_waits_branch = kind != "dense" and apart
    return on_par, wait_main_last, host_waits_branch

_CHOOSE_LOCK = threading.Lock()  # see StepGraph._compile_choose


class StepGraph(torch.nn.Module):
    """A dataflow runner's step as one CUDA graph (module docstring)."""

    def __init__(self, runner, plan=None, role="decode", world=None, rank=None, tag=None):
        super().__init__()
        self.runner = runner  # dataflow.Runner over a split whose dense segments are aoti.GraphedSegment
        self.plan = plan  # the engine's plan: the step's statistics go into its record (plan.stat)
        # This GPU's expert node computes its held pairs only and hands on the partial sum (hero run 14),
        # for either reason: the plan's expert_ep spreads a layer's experts over the GPUs, or this graph
        # is a tensor-parallel rank and every rank runs the node; the step's reduce adds the partials
        try:
            self._partial_experts = bool(plan is not None and plan["expert_ep"] and plan["expert_ep"] > 1)
        except (KeyError, TypeError):
            self._partial_experts = False
        self._partial_experts = self._partial_experts or world is not None
        self.role = role  # "decode" or "draft": the draft's statistics are prefixed
        # tensor parallelism (hero run 14): this graph is rank `rank` of `world` (parallel.World); its
        # collective nodes exchange partials with the other ranks' graphs through the world's buffers
        self.world, self.rank = world, rank
        # the key space of this graph's exchange buffers in the world: None for the step, a name for a
        # program beside it (the verifies per draft length, hero run 15), whose collectives are shaped
        # differently and must not take the step's inboxes
        self.tag = tag
        self._collectives = 0
        self._sum_sites = 0  # collective nodes captured so far, in step order (the exchange's index)
        # the step trace of a tensor-parallel rank (VXPU_STEP_TRACE; parallel.Decode writes it): per step,
        # when this thread entered, had the inputs copied, had launched the graph, and saw it done, and
        # the host nodes' round trips inside the graph
        self.trace = [] if world is not None and os.environ.get("VXPU_STEP_TRACE") else None
        self.split = runner.split
        # the embedding's shard (vxpu::tp_embed_rows) registered on the world now, at build time: the prime
        # captures each rank alone in solo mode, so the capture cannot exchange anything, and every rank's
        # graph is built before any is primed, so the registry holds all of them by the first capture
        if world is not None and rank is not None and world.size > 1:
            for node in runner.graph.nodes:
                if node.kind == "embed_rows":
                    ref = node.args[1]
                    t = runner._arg(ref, runner.device if hasattr(runner, "device") else torch.device("cpu"), {})
                    if not isinstance(t, torch.Tensor):
                        raise RuntimeError(f"step graph: {node.name} takes no shard tensor")
                    world.register_shard(("embed_rows", tag, getattr(ref, "name", node.name)), rank, t.contiguous())
        self.vxpu_segments = runner.vxpu_segments
        self.graph = None
        self._chosen = self._sampling = None
        self._graph_token = False  # the captured graph writes its choice into its own inputs (GRAPH_TOKEN)
        self._choose = choose  # compiled after the first step (_compile_choose)
        self.node_counts = {}
        self._peer_graphs = []
        self.steps = 0
        self.recorder = None
        self._host = None
        self._expert_counts = {}  # per GPU: distinct experts selected, summed per layer (VXPU_EXPERT_STATS)

    def set_recorder(self, recorder):
        self.recorder = recorder
        self.runner.set_recorder(recorder)

    def forward(self, *args, **kwargs):
        finish = self.launch(*args, **kwargs)
        return finish() if callable(finish) else finish

    def launch(self, *args, **kwargs):
        """The step's launch half: the inputs into their buffers, the step host started and the
        graphs replayed; returns the finish half (the streams' synchronization and the output), or
        the output itself for the first step, which runs eagerly. One
        thread can so launch every rank's graph of a tensor-parallel step and then wait for all
        (parallel.Decode's single launcher, hero run 15)."""
        self.steps += 1
        triton_ops.mx_memo_clear()  # the producer-side MXFP8 fold's memo is a step's (triton_ops.x_mxfp8)
        if self.steps == 1:
            out = self.runner(*args, **kwargs)  # the first step loads the packages and allocates the inboxes
            self._compile_choose(out)
            return out
        if self.graph is None:
            gc.disable()  # a collection inside a capture can unload a kernel's module, which a capture forbids
            try:
                self._capture(args, kwargs)  # then the step runs as the captured graph below
            finally:
                gc.enable()
        trace = self.trace is not None
        t_in = time.perf_counter() if trace else 0.0
        for buf, a in zip(self._inputs, self._input_values(args, kwargs)):
            buf.copy_(a)
        t_copied = time.perf_counter() if trace else 0.0
        return self._replay(trace, t_in, t_copied)

    def input_buffers(self):
        """The captured graph's input buffers by name (the tensors the graph reads; a caller writing
        them on the stream and relaunching with force feeds the step itself)."""
        return dict(zip(self.runner.graph.inputs, self._inputs))

    def output_tensor(self):
        """The captured graph's output tensor (a one-tensor tuple unwrapped), on its device."""
        out = self._output
        return out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out

    def relaunch(self, force=False):
        """The launch half on the graph's own inputs: the step before wrote its choice and position
        there (GRAPH_TOKEN), so nothing is copied in. Returns the finish half. With `force` the
        caller wrote the input buffers itself (engine._tp_spec_fixed writes the drafter's and the
        verify's on the stream), so only a captured graph is required."""
        if self.graph is None or not (self._graph_token or force):
            raise RuntimeError("relaunch: the step graph does not feed itself (VXPU_GRAPH_TOKEN, a captured one-token step)")
        self.steps += 1
        triton_ops.mx_memo_clear()
        return self._replay(self.trace is not None, 0.0, 0.0)

    def _replay(self, trace, t_in, t_copied):
        self._host.go()
        self.graph.replay()
        t_launched = time.perf_counter() if trace else 0.0
        # the other GPUs' graphs after this one's: launched first, a graph
        # that opens by waiting on a flag held the launching thread ~5 ms.
        # A second graph on one GPU (its peer graph beside its range graph, hero run 14) goes on
        # a stream of its own: on one stream the second would wait for the first to finish, and
        # the first waits on replies the second produces.
        seen = {self.graph_device_index()}
        extra = []
        for pdev, g in self._peer_graphs:
            d = torch.device(pdev).index
            with torch.cuda.device(pdev):
                if d in seen:
                    st = self._side_stream(d, len(extra))
                    extra.append(st)
                    with torch.cuda.stream(st):
                        g.replay()
                else:
                    seen.add(d)
                    g.replay()
        stream = torch.cuda.current_stream()
        return lambda: self._finish(stream, extra, trace, t_in, t_copied, t_launched)

    def _finish(self, stream, extra, trace, t_in, t_copied, t_launched):
        stream.synchronize()
        for pdev, _ in self._peer_graphs:
            torch.cuda.current_stream(pdev).synchronize()
        for st in extra:
            st.synchronize()
        if trace:
            t_done = time.perf_counter()
            # the host nodes' share of the graph's time: per node the GPU's round trip (its flag raised to
            # the reply's flag seen, the graph idle meanwhile) and the step host's work inside it
            hops = []
            try:
                for (kind, layer, remote, _, _, _), t, (sent, back) in zip(self._box_info, self._host.times().tolist(),
                                                                          self._events):
                    hops.append([kind, layer, (t[1] - t[0]) / 1e6, sent.elapsed_time(back)])
            except Exception:  # noqa: BLE001  (a graph without host nodes, or timing events not recorded)
                hops = []
            self.trace.append({"in": t_in, "copied": t_copied, "launched": t_launched, "done": t_done, "hops": hops})
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

    def native_parts(self, streams=None):
        """What a native loop needs to run this captured step: the input buffers by name, the
        output (a decode step's chosen tokens, or the step's own output when it is not logits:
        a drafter's tokens), the graphs to launch (cudaGraphExec_t), their streams and devices,
        the step host and its flags. `streams` ({GPU index: torch.cuda.Stream}) launches the
        graphs on those streams instead of each GPU's current one: several batches in flight
        (engine._PipelineRows) each launch on streams of their own, so one batch's graph waiting
        on a flag (a stream memory wait, not a kernel) holds nothing of another batch's."""
        if self.graph is None:
            raise RuntimeError("native loop: the step graph is not captured")
        inputs = dict(zip(self.runner.graph.inputs, self._inputs))
        out = self._chosen if self._chosen is not None else self._output
        out = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        any_input = next(iter(inputs.values()))
        index = any_input.device.index if any_input.device.index is not None else torch.cuda.current_device()
        execs = [self.graph.raw_cuda_graph_exec()] + [g.raw_cuda_graph_exec() for _, g in self._peer_graphs]
        devices = [index] + [torch.device(p).index for p, _ in self._peer_graphs]
        # one stream per graph: a GPU running its range graph and a peer graph (hero run 14, the
        # experts spread over the cards) must not queue the second behind the first, since the
        # first waits on replies the second produces; the k-th extra graph on a GPU gets a side
        # stream of its own, kept with this step graph
        handles, seen = [], {}
        for d in devices:
            k = seen.get(d, 0)
            seen[d] = k + 1
            if k == 0:
                handles.append((torch.cuda.current_stream(d) if streams is None else streams[d]).cuda_stream)
            else:
                handles.append(self._side_stream(d, k - 1).cuda_stream)
        return inputs, out, execs, handles, devices, self._host, self._flags

    def graph_device_index(self):
        any_input = next(iter(self._inputs)) if self._inputs else None
        if any_input is not None and any_input.device.index is not None:
            return any_input.device.index
        return torch.cuda.current_device()

    def _side_stream(self, d, k):
        """The k-th extra stream on GPU d for a second graph of this step there (see native_parts)."""
        streams = self.__dict__.setdefault("_side_streams", {})
        key = (d, k)
        if key not in streams:
            streams[key] = torch.cuda.Stream(device=torch.device("cuda", d))
        return streams[key]

    def native_loop(self, draft=None, streams=None, draft_block=None, policy=None, draft_scale=0.0, verify_graphs=()):
        """The captured step, decoded by a native thread with no Python between steps
        (node_server.cpp's DecodeLoop): it writes each row's tokens and position into the graph's
        input buffers and each row's sampling settings into the graph's (choose), replays the graphs
        and reads the chosen tokens back. With `draft` (the drafter's captured StepGraph), each step
        first runs the drafter, verifies its tokens in this step and keeps those the choices agree
        with. `streams` (see native_parts) launches this step's graphs and the drafter's on those
        streams: a batch in flight runs its drafter on its own streams as well, ahead of its
        verify on the same stream of the drafter's GPU. `policy` (vxpu.speculation.Policy, from
        the plan's speculation section) caps the drafts a row issues a step or stops them at the
        first under its confidence (`draft_scale` turns the drafter's scores into probabilities);
        the verify graph keeps its token rows, the positions past the issued drafts carry the last
        issued token (DecodeLoop.set_policy). `verify_graphs` are (token rows, StepGraph) pairs of
        the same step captured from exports with fewer token rows (hero run 15: the verify sized
        to the draft); under a policy the loop runs the shortest that fits the drafts issued
        (DecodeLoop.set_verify)."""
        own = streams
        inputs, chosen, execs, streams, devices, host, flags = self.native_parts(own)
        if self._chosen is None or sorted(inputs) != ["cache_position", "input_ids"]:
            raise RuntimeError(f"native loop: a decode step of logits from input_ids and cache_position, not {sorted(inputs)}")
        out_graph = devices.index(chosen.device.index) if chosen.device.index in devices else 0
        _check_flat(inputs, chosen, "decode")
        if draft is not None:
            _check_flat(*draft.native_parts(own)[:2], "draft")
        loop = kernels.load().DecodeLoop(inputs["input_ids"], inputs["cache_position"], chosen, execs, streams, devices,
                                         host, flags, out_graph)
        loop.set_sampling(*self._sampling)
        if draft is not None:
            d_inputs, d_out, d_execs, d_streams, d_devices, d_host, d_flags = draft.native_parts(own)
            # the drafter's tokens into this step's input ids on the GPU: a graph of one copy per
            # row (ids[i, 1:S] <- out[i, :S-1], contiguous slices, across GPUs if the drafter
            # lives on another), run by the loop on the verify's stream after an event on the
            # drafter's; the drafts reach the host only after the verify, for the acceptance.
            # Before this the loop synchronized after the drafter's graphs, copied the drafts to
            # the host, wrote the verify's ids and synchronized again: about 3 ms of a 30 ms
            # speculative step at one row sat in those two round trips.
            ids = inputs["input_ids"]
            S = ids.shape[1]
            driver = _Driver()
            compose = torch.cuda.CUDAGraph()
            with torch.cuda.device(ids.device), torch.cuda.graph(compose, stream=torch.cuda.Stream(device=ids.device)):
                stream = torch.cuda.current_stream()
                for i in range(ids.shape[0]):
                    driver.memcpy(stream, ids[i, 1:S], d_out[i, :S - 1])
            self._compose = compose  # the loop holds the exec's handle, not the graph
            # draft_block: the tokens' width when the output also carries each draft's confidence
            # after them (binding draft.block and draft.confidence); the loop records both
            loop.set_draft(d_inputs["input_ids"], d_inputs["cache_position"], d_out, d_execs, d_streams, d_devices,
                           d_host, d_flags, compose=compose.raw_cuda_graph_exec(), block=draft_block or 0)
            if policy is not None:
                from . import speculation
                confidence = policy.policy == speculation.CONFIDENCE
                if confidence and not (draft_block and draft_scale):
                    raise RuntimeError("native loop: the confidence policy needs a drafter that outputs its confidence")
                # a plan that names a policy takes the host path even when it issues every draft, so an
                # A/B of policies differs in the drafts issued and nothing else
                loop.set_policy(min(policy.draft, S - 1), int(round(policy.min_confidence * draft_scale)) if confidence else 0)
            for rows, short in verify_graphs:
                if policy is None:
                    raise RuntimeError("native loop: a verify per draft length needs a policy (the host path)")
                v_inputs, v_chosen, v_execs, v_streams, v_devices, v_host, v_flags = short.native_parts(own)
                _check_flat(v_inputs, v_chosen, f"verify of {rows} token rows")
                v_out = v_devices.index(v_chosen.device.index) if v_chosen.device.index in v_devices else 0
                loop.set_verify(rows, v_inputs["input_ids"], v_inputs["cache_position"], v_chosen, v_execs, v_streams,
                                v_devices, v_host, v_flags, v_out)
        loop.start()
        return loop

    def _compile_choose(self, out):
        """Compile choose for this step's logits, before the capture (compiling inside one would
        load modules mid-capture). Inductor fuses its elementwise work: 8 kernels instead of 38
        (an L4, 20,000 rows); only top-k stays a library kernel. Inductor's random numbers
        are drawn by the graph on every replay, as eager rand_like's are. Not fullgraph: past
        dynamo's limit on recompiles (8 shapes of choose in a process), choose runs eagerly
        instead of failing."""
        logits = out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out
        if not (isinstance(logits, torch.Tensor) and logits.dim() == 3 and logits.is_cuda):
            return
        _seed(logits.device)  # made before the compile: a first call that makes it guards on its absence
        # one compile at a time in the process: the ranks of a tensor-parallel step reach their first
        # step on threads of their own, and dynamo's tracing state is global (four ranks compiling
        # choose at once broke its guards and its mode stack, hero run 15). Each rank drains its
        # stream first: its eager step's collectives are still in flight, waiting for the peers'
        # pushes, and the warm call under the lock would wait for them while a peer waits for the
        # lock (three ranks spinning in a sum, one on the host: the first locked run hung)
        if logits.is_cuda:
            torch.cuda.synchronize(logits.device)
        with _CHOOSE_LOCK:
            self._choose = torch.compile(choose, dynamic=False)
            rows = logits.shape[0]
            self._choose(logits, torch.zeros(rows, device=logits.device),
                         torch.zeros(rows, dtype=torch.long, device=logits.device), torch.ones(rows, device=logits.device))

    def _input_values(self, args, kwargs):
        values = dict(zip(self.runner.graph.inputs, args), **kwargs)
        return [values[name] for name in self.runner.graph.inputs]

    def _capture(self, args, kwargs):
        from . import ops
        triton_ops.mx_memo_clear()
        graph, runner = self.runner.graph, self.runner
        driver = _Driver()
        self._inputs = [v.clone() for v in self._input_values(args, kwargs)]
        inputs = dict(zip(graph.inputs, self._inputs))
        # each row's sampling settings (choose), outside the graph's memory: the native loop writes them
        ids = inputs.get("input_ids")
        self._sampling = None
        if isinstance(ids, torch.Tensor) and ids.dim() == 2:
            rows = ids.shape[0]
            self._sampling = (torch.zeros(rows, device=ids.device),  # temperature: 0 is greedy
                              torch.zeros(rows, dtype=torch.long, device=ids.device),  # top_k: 0 is off
                              torch.ones(rows, device=ids.device))  # top_p: 1 is off
        host_nodes = [n for n in graph.nodes if n.kind in ("experts", "engram")]
        produced = {}  # node -> its output tensors on the GPU, as captured
        self._collectives = 0
        self._sum_sites = 0
        self._driver = driver
        runner.begin_step()  # the captured step's per-step copies (step inputs, shared state) are captured too
        dev = runner.device
        if dev.index is None:  # the pools name their GPU by index
            dev = torch.device(dev.type, torch.cuda.current_device())
        external = ops._active.external
        # the GPU of each node (layer sharding puts a layer's nodes on its GPU); a CPU node's is
        # the GPU of the step graph it answers from
        node_dev = {n.name: (d if d.type == "cuda" else dev) for n in graph.nodes for d in [runner._devices[n.name]]}
        devs = [dev] + [d for d in dict.fromkeys(node_dev.values()) if d != dev]
        # a step input a node on another GPU reads: its copy there, filled at the start of the step
        in_copies = {(name, d): inputs[name].clone().to(d) for name in inputs for d in devs[1:]
                     if isinstance(inputs[name], torch.Tensor)}

        def resolve(r, d=dev):
            """A node's argument on GPU d: a step input (its copy on d), a resident (the runner's
            copy on d if it lives elsewhere), or a value another node made (its inbox on d if that
            node ran on another GPU)."""
            if isinstance(r, (tuple, list)):
                return type(r)(resolve(x, d) for x in r)
            if not isinstance(r, dataflow.Ref):
                return r
            if r.kind == "input":
                v = inputs[r.name]
                return in_copies.get((r.name, d), v)
            if r.kind == "resident":
                return runner._arg(r, d, inputs)
            if node_dev[r.node] != d:
                return runner.inboxes[(r.node, r.index, d)]
            out = produced[r.node]
            return out if r.index is None else out[r.index]

        # Every pinned buffer before the captures (allocating pinned memory
        # while a stream captures is not allowed), shaped like the inboxes
        # the runner's first step allocated. An experts node whose layer
        # has pools gets per-pair outboxes: one the CPU fills, one per
        # other GPU holding some of its experts.
        boxes, pooled, peers = {}, {}, []
        for node in host_nodes:
            dev_ = node_dev[node.name]  # with its index: a pool on this GPU is "here", not a peer
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
                    here = [p for p in res.pools if p.device == dev_]
                    for pool in res.pools:
                        if pool.device != dev_ and not self._partial_experts:  # expert_ep: the other GPUs' pools
                            # are theirs to run; nothing crosses here
                            peers += [pool.device] if pool.device not in peers else []
                            # straight between the GPUs: the inputs into an inbox on the other GPU, its
                            # replies into one here, each followed by a flag in the receiver's memory
                            inbox = [torch.empty_like(example[k], device=pool.device) for k in where]
                            peer_boxes[pool.device] = (pool, inbox, torch.zeros(1, dtype=torch.int32, device=pool.device),
                                                       torch.empty_like(out_like, device=dev_),
                                                       torch.zeros(1, dtype=torch.int32, device=dev_),
                                                       pool.slot_cpu.to(dev_))  # its slot map, here, for the combine
                    pooled[node.name] = (external.experts[layer], here, peer_boxes, where, layer)
            boxes[node.name] = ([torch.empty_like(a, device="cpu").pin_memory() if isinstance(a, torch.Tensor) else a
                                 for a in example],
                                torch.empty_like(out_like, device="cpu").pin_memory(), out_like)
        for pdev in set(peers) | set(devs[1:]):  # peer access, both ways, before any capture
            torch.empty(1, device=pdev).copy_(torch.empty(1, device=dev))
            torch.empty(1, device=dev).copy_(torch.empty(1, device=pdev))
        # per host node i: its to-CPU and from-CPU flags
        F = 2
        flags = torch.zeros(F * len(host_nodes), dtype=torch.int32, pin_memory=True)
        base = driver.device_address(flags)

        def flag(i, k):
            return base + 4 * (F * i + k)

        mailboxes = []  # per host node: (node, pinned inputs, kwargs, outbox, flag offset, pooled)
        sides = {d: torch.cuda.Stream(device=d) for d in devs}  # a GPU's pools' kernels, forked and joined back
        self._keeps = []  # each GPU's capture's expert scratch, alive with the graphs (branch_choice's rule)
        if EXPERT_STATS:  # the distinct-expert counters, allocated before any capture (see _count_experts)
            self._expert_counts = {d: torch.zeros(EXPERT_STATS_LAYERS, dtype=torch.int64, device=d) for d in devs}
        if EXPERT_TRACE:  # the per-step expert ids per layer, allocated before any capture (see _count_experts)
            self._expert_trace = {d: torch.full((EXPERT_STATS_LAYERS, EXPERT_TRACE_STEPS, EXPERT_TRACE_IDS), -1,
                                                dtype=torch.int32, device=d) for d in devs}
            self._expert_trace_pos = {d: torch.zeros(EXPERT_STATS_LAYERS, dtype=torch.int64, device=d) for d in devs}
            self._expert_trace_shape = {}  # layer -> (token rows, top-k) of the ids a step
        answers = self._answers(host_nodes, boxes, pooled, external)  # before any capture: builds packages
        # the expert kernels' arrival counters belong to these graphs (triton_ops._fp4_counters):
        # another step graph in flight gets its own. They are made by the warm-up, outside any
        # capture: a buffer made inside one is a memset node on the capture stream, unordered
        # against the kernel on the pools' side stream that reads it (an illegal address).
        scope = []
        triton_ops.FP4_COUNTER_SCOPE = scope
        try:
            self._warm(pooled, runner, inputs, node_dev)
            if self.world is not None:
                self._prepare_collectives(graph, runner, dev)
            torch.cuda.synchronize()
            self._capture_all(graph, mailboxes, boxes, pooled, peers, driver, flags, F, flag, sides, dev, devs,
                              node_dev, in_copies, resolve, produced, host_nodes, answers)
        finally:
            triton_ops.FP4_COUNTER_SCOPE = None
            triton_ops.fp4_counter_scope_end(scope)
        self._keep = self._keep + (scope,)
        if COPY_LOG:  # what the step moves between GPUs: the bytes a crossing costs
            total = sum(c[-1] for c in COPY_LOG)
            big = sorted(COPY_LOG, key=lambda c: -c[-1])[:6]
            print(f"[vxpu] the step copies {len(COPY_LOG)} tensors between GPUs, {total / 1e6:.1f} MB; the largest: "
                  + ", ".join(f"{c[1]} {c[2]}->{c[3]} {c[4]} {c[5].replace('torch.', '')} {c[-1] / 1e6:.2f} MB" for c in big),
                  flush=True)
            if STEP_COPIES:
                for c in COPY_LOG:
                    print(f"[vxpu]   copy {c[0]} {c[1]} {c[2]}->{c[3]} {c[4]} {c[5]} {c[-1]} B", flush=True)

    def _prepare_collectives(self, graph, runner, dev):
        """Before a tensor-parallel rank's capture: the world's exchange buffers for every
        collective of the step, made outside any capture (an allocation on another rank's device
        while that rank captures is not allowed), and the ring write's kernel compiled by one
        eager launch per peer (a capture forbids a first launch), its flags cleared after."""
        # the sums through the all-reduce template (vxpu::allreduce, allreduce.py: one kernel pushes the
        # partial into the peers' inboxes and raises their flags, one waits and sums in rank order, about
        # 11 us a reduction at one row) unless the plan's reduce is "ring" (the ring write per peer,
        # the stream's flag waits and the sum as aten kernels; about 110 us a reduction, measured
        # on the real model); one Comm per process, set up by rank 0 for every all-reduce site of
        # the step before any rank captures
        from . import allreduce as ar
        likes = {}
        hcs = set()
        warmed = set()
        for node in graph.nodes:
            if node.kind in ("allreduce", "allgather"):
                r0 = node.args[0]
                if not (isinstance(r0, dataflow.Ref) and r0.kind == "edge"):
                    raise RuntimeError(f"step graph: {node.name} does not take a node's output")
                x = runner.inboxes[(r0.node, r0.index, dev)].contiguous()
                gemv = self._gemv_sum(node.name)
                if gemv:
                    # the fused push (hero run 15): the partial is the GEMV's result, bf16 [rows, N] (the
                    # attention's wo_b) or fp32 with the routed sum added (the FFN's w2); its kernel is
                    # launched once against scratch here, before any capture, so its module is loaded
                    refs = [node.args[i] if i is not None else None for i in gemv]
                    weight, scale = (runner._arg(r, dev, {}) for r in refs[1:3])
                    add = runner.inboxes[(refs[3].node, refs[3].index, dev)] if refs[3] is not None else None
                    rows = x.numel() // x.shape[-1]
                    x = torch.empty(rows, weight.shape[0], dtype=torch.float32 if add is not None else x.dtype, device=dev)
                    key = (rows, tuple(weight.shape), x.dtype)
                    if key not in warmed and dev.type == "cuda" and triton_ops.triton is not None:
                        warmed.add(key)
                        ar.warm_gemv_push(runner.inboxes[(r0.node, r0.index, dev)], weight, scale, add, self.world.size - 1)
                        if add is not None and TWO_PART_SUM:  # the two-part form's GEMV pushes bf16 without the add
                            ar.warm_gemv_push(runner.inboxes[(r0.node, r0.index, dev)], weight, scale, None, self.world.size - 1)
                likes[node.name] = x
                r1 = node.args[gemv[4]] if gemv else node.args[1] if len(node.args) >= 4 else None
                if node.kind == "allreduce" and isinstance(r1, dataflow.Ref) and r1.kind == "edge":
                    hcs.add(runner.inboxes[(r1.node, r1.index, dev)].shape[-2])  # the mixing sums' hc width
        sums = [likes[n.name] for n in graph.nodes if n.kind == "allreduce"]
        self._comm_sums = (self.plan["reduce"] not in ("ring", "skip") and bool(sums) and dev.type == "cuda"
                           and triton_ops.triton is not None)
        if self._comm_sums and self.rank == 0:
            dtypes = {t.dtype for t in sums}
            if hcs:  # the mixing sums gather in the stream's dtype, and fp32 partials may cross as bf16 (push_bf16)
                dtypes.add(torch.bfloat16)
            max_elems = max(t.numel() for t in sums)
            if TWO_PART_SUM:  # a two-part box holds the routed partial as fp32 and the shared output as bf16: 1.5 x
                max_elems = -(-max_elems * 3 // 2)
            ar.setup(self.world.devices, len(sums), max_elems, dtypes=sorted(dtypes, key=str), hcs=sorted(hcs))
        self.world.barrier()
        if self._comm_sums:  # from here this graph holds the Comm's buffers: it is never replaced again
            from . import allreduce as ar
            ar.comm().bound = True
        k = 0
        for node in graph.nodes:
            if node.kind not in ("allreduce", "allgather"):
                continue
            if node.kind == "allreduce" and self._comm_sums:
                k += 1
                continue
            like = likes[node.name]
            inboxes, flags, counters = self.world.buffers(self._xkey(k), like)
            r = self.rank
            stream = torch.cuda.current_stream(dev)
            for j in range(self.world.size):
                if j != r and _ring_ok(like, inboxes[(r, j)]):
                    triton_ops.ring_write([(like, inboxes[(r, j)])], counters[(r, j)], flags[(r, j)], stream)
            k += 1
        torch.cuda.synchronize(dev)
        # every rank has warmed its writes once the barrier passes; then the flags go back to zero
        self.world.barrier()
        for kk in range(k):
            for (i, j), f in self.world.flags.get(self._xkey(kk), {}).items():
                if j == self.rank:
                    f.zero_()
            for (i, j), c in self.world.counters.get(self._xkey(kk), {}).items():
                if i == self.rank:
                    c.zero_()
        torch.cuda.synchronize(dev)
        self.world.barrier()

    def _xkey(self, k):
        """The world's key for this graph's collective k (parallel.World.buffers)."""
        return k if self.tag is None else (self.tag, k)

    def _gemv_sum(self, name):
        """For a mixing sum with its GEMV inside (vxpu::tp_gemv_allreduce_hc, the fused push): where each
        of the op's seven arguments (x, weight, scale, add, residual, post, comb) sits among the node's
        inputs, None for an argument that is a constant (the attention sites' add). The split orders a
        segment's inputs by first use, so a position is read off the segment's graph, not assumed. None
        for any other collective."""
        sub = getattr(self.split, name)
        sub = getattr(sub, "_orig_mod", sub)
        placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
        for n in sub.graph.nodes:
            if n.op == "call_function" and str(n.target) == "vxpu.tp_gemv_allreduce_hc.default":
                return tuple(placeholders.index(a) if isinstance(a, torch.fx.Node) else None for a in n.args)
        return None

    def _capture_collective(self, node, args_, d, stream):
        """A collective node of a tensor-parallel rank's graph (hero run 14): this rank's partial
        goes into each other rank's inbox for this exchange with the ring write (one kernel per
        peer, raising a flag in the peer's memory once the copy is visible), the stream waits on
        the flags the peers raise here and clears them, and the sum (vxpu::tp_allreduce) or the
        concatenation (vxpu::tp_allgather) is taken over the ranks' pieces in rank order, so every
        rank's result is bit-identical. The buffers are the world's, made once at fixed addresses
        (parallel.World.buffers); the exchange's index is the node's position among the step's
        collectives, the same on every rank since the ranks run one program."""
        if self.world is None or self.rank is None:
            raise RuntimeError(f"step graph: {node.name} is a collective, and this graph is no rank of a world")
        from . import ops
        k = self._collectives
        self._collectives += 1
        x = args_[0]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"step graph: {node.name} takes no tensor")
        with torch.cuda.stream(stream):
            x = x.contiguous()
        # vxpu::tp_allreduce_hc (hero run 15): the sum with the block's hc_post as its epilogue;
        # vxpu::tp_gemv_allreduce_hc (the fused push): the GEMV that makes the partial inside it too
        gemv = self._gemv_sum(node.name) if node.kind == "allreduce" else None
        if gemv:
            x, weight, scale, add = (args_[i] if i is not None else None for i in gemv[:4])
            with torch.cuda.stream(stream):
                x = x.contiguous()
            hc = tuple(args_[i] for i in gemv[4:7])
        else:
            hc = tuple(args_[1:4]) if node.kind == "allreduce" and len(args_) >= 4 else None

        def mixed(total):
            from .ops import _hc_post_reference
            return total if hc is None else _hc_post_reference(total.to(hc[0].dtype), *hc)
        if self.plan["reduce"] == "skip":  # the timing floor: the step without its sums (wrong tokens)
            with torch.cuda.stream(stream):
                if gemv:
                    return mixed(ops.gemv_partial(x, weight, scale, add))
                return mixed(x.clone()) if node.kind == "allreduce" else torch.cat([x] * self.world.size, dim=-1)
        if node.kind == "allreduce" and getattr(self, "_comm_sums", False):
            site = self._sum_sites
            self._sum_sites += 1
            with torch.cuda.stream(stream):
                if gemv and getattr(self, "_two_part_sites", {}).get(node.name) == site:
                    return torch.ops.vxpu.gemv_allreduce_hc2(x, weight, scale, add, site, *hc)
                if gemv:
                    return torch.ops.vxpu.gemv_allreduce_hc(x, weight, scale, add, site, *hc)
                if hc is not None:
                    return torch.ops.vxpu.allreduce_hc(x, site, *hc)
                return torch.ops.vxpu.allreduce(x, site)
        if gemv:  # the ring write form: the partial first, then the exchange as for any sum
            with torch.cuda.stream(stream):
                x = ops.gemv_partial(x, weight, scale, add)
        inboxes, flags, counters = self.world.buffers(self._xkey(k), x)
        r, size = self.rank, self.world.size
        for j in range(size):
            if j != r:
                if _ring_ok(x, inboxes[(r, j)]):
                    self._ring_writes += triton_ops.ring_write([(x, inboxes[(r, j)])], counters[(r, j)], flags[(r, j)], stream)
                else:  # the ring write off (VXPU_RING_WRITE=0): a peer copy, then the flag, as the crossings do
                    self._driver.memcpy(stream, inboxes[(r, j)], x)
                    self._driver.write_value(stream, flags[(r, j)].data_ptr(), 1)
        for i in range(size):
            if i != r:
                self._driver.wait_value(stream, flags[(i, r)].data_ptr(), 1)
                self._driver.write_value(stream, flags[(i, r)].data_ptr(), 0)
        with torch.cuda.stream(stream):
            parts = [x if i == r else inboxes[(i, r)] for i in range(size)]
            if node.kind == "allreduce":
                total = parts[0].to(torch.float32)
                total = total.clone() if total.data_ptr() == parts[0].data_ptr() else total
                for part in parts[1:]:
                    total += part.to(torch.float32)
                return mixed(total.to(x.dtype))
            return torch.cat(parts, dim=-1)

    def _embed_start(self, name):
        """The `start` of the vxpu::tp_embed_rows call in segment `name` (an int constant of the call)."""
        sub = getattr(self.split, name)
        sub = getattr(sub, "_orig_mod", sub)
        for n in sub.graph.nodes:
            if n.op == "call_function" and str(n.target) == "vxpu.tp_embed_rows.default":
                start = n.args[2] if len(n.args) > 2 else n.kwargs.get("start")
                if isinstance(start, int):
                    return start
                raise RuntimeError(f"step graph: {name}: tp_embed_rows takes a start that is not a constant ({start!r})")
        raise RuntimeError(f"step graph: {name} has no vxpu::tp_embed_rows call")

    def _capture_embed_rows(self, node, args_, d, stream):
        """vxpu::tp_embed_rows in a tensor-parallel rank's graph (hero run 15): the ranks share their
        shards of the embedding once (parallel.World.share, the tensors themselves at their fixed
        addresses), and this rank reads each id's row from the owning rank's shard over P2P
        (triton_ops.peer_rows), no exchange and no flag, since the shards never change. A rank
        stepping alone (the prime) reads its own shard for every id, as the eager form does."""
        if self.world is None or self.rank is None:
            raise RuntimeError(f"step graph: {node.name} reads the ranks' shards, and this graph is no rank of a world")
        ids, shard = args_[0], args_[1]
        if not (isinstance(ids, torch.Tensor) and isinstance(shard, torch.Tensor)):
            raise RuntimeError(f"step graph: {node.name} takes no tensors")
        # the rank's first vocabulary index is a constant of the op's call inside the segment (the split
        # hands a node only its tensor inputs), read off the segment's graph as _gemv_sum reads its op
        start = args_[2] if len(args_) > 2 else self._embed_start(node.name)
        if self.world.size == 1:
            with torch.cuda.stream(stream):
                return self.world.embed_rows(ids, shard.contiguous(), start)
        # the shards every rank shared before the capture (_prepare_collectives): the prime captures each
        # rank alone in solo mode and that captured graph is the one replayed, so the capture must take the
        # real path here as the collectives do (the first capture took the solo lookup, zero rows for the
        # ids other ranks own, and the drafts were wrong: 723 against 1,048 tok/s at 16 streams)
        ref = node.args[1]
        shards = self.world.shards_of(("embed_rows", self.tag, getattr(ref, "name", node.name)))
        if any(sh is None for sh in shards):
            raise RuntimeError(f"step graph: {node.name}: not every rank registered its shard before the capture "
                               f"({[i for i, sh in enumerate(shards) if sh is None]} missing)")
        if shards[self.rank].data_ptr() != shard.data_ptr():
            raise RuntimeError(f"step graph: {node.name}: this rank's registered shard is not the one captured")
        if triton_ops.EMBED_ROWS_FORM != "push":
            with torch.cuda.stream(stream):
                return triton_ops.peer_rows(ids, shards, stream)
        # the push form (hero run 15): the pull form's P2P reads stalled behind the peers' own steps (129 to
        # 466 us a call for 115 to 229 KB), so each rank writes the rows it owns into every rank's output
        # buffer instead (writes are posted) and raises a flag per peer; the reader waits on three flags,
        # as the ring path does, and reads its own buffer. The buffers are the world's for this node, one
        # per destination rank (the pair inboxes from rank 0, or rank 1 for rank 0's own), at fixed
        # addresses; the first rank to capture makes them, alone in the prime, so no rank is capturing
        # while another's device gets an allocation.
        r, size = self.rank, self.world.size
        like = torch.empty(tuple(ids.shape) + (shard.shape[-1],), dtype=shard.dtype, device=ids.device)
        inboxes, _flags, _counters = self.world.buffers(("embed_out", self._xkey(node.name)), like)
        outs = [inboxes[(1 if j == 0 else 0, j)] for j in range(size)]
        # the flags are raised by the ring write's kernel over a four-byte pair per peer, as the collectives
        # raise theirs (a device store through the peer mapping, after a fence): a stream write to a peer's
        # flag address stopped the first step on one pod (rank 0's wait never saw its peers' flags there,
        # hero run 15) where the same tree ran on another, and the ring path has no such dependence
        # the payload is two ring blocks of words, not four bytes: Triton specializes integer arguments that
        # equal one or divide by sixteen, and a one-word payload gave the ring write's kernel a word count
        # and a chunk count of one, a variant the prepare step never compiled, so its first launch fell
        # inside the capture (cudaErrorStreamCaptureUnsupported); two blocks put both counts in the
        # specialization the warmed full-width writes use
        words = 2 * triton_ops.RING_BLOCK
        tiny = torch.empty(words, dtype=torch.int32, device=ids.device)
        pairs, flags, counters = self.world.buffers(("embed_flag", self._xkey(node.name)), tiny)
        if not hasattr(self, "_embed_tiny"):
            self._embed_tiny = {}
        src = self._embed_tiny.setdefault(node.name, torch.zeros(words, dtype=torch.int32, device=ids.device))
        with torch.cuda.stream(stream):
            triton_ops.peer_rows_push(ids, shard, start, outs, stream)
        for j in range(size):
            if j != r:
                if _ring_ok(src, pairs[(r, j)]):
                    self._ring_writes += triton_ops.ring_write([(src, pairs[(r, j)])], counters[(r, j)], flags[(r, j)], stream)
                else:
                    self._driver.memcpy(stream, pairs[(r, j)], src)
                    self._driver.write_value(stream, flags[(r, j)].data_ptr(), 1)
        for i in range(size):
            if i != r:
                self._driver.wait_value(stream, flags[(i, r)].data_ptr(), 1)
                self._driver.write_value(stream, flags[(i, r)].data_ptr(), 0)
        return outs[r]

    def refresh_rows(self, rows):
        """A prompt's prefill wrote these rows of the shared caches outside the captured step
        (the prefill program runs through its own runner): the replicas take those rows whole,
        once, so the step's windows have a correct base to add to."""
        windows = getattr(self, "_windows", None)
        if not windows:
            return
        for w in windows.values():
            for row in rows:
                w["replica"][row].copy_(w["src"][row])
        for d in {w["replica"].device for w in windows.values()} | {w["src"].device for w in windows.values()}:
            torch.cuda.synchronize(d)

    def _warm(self, pooled, runner, inputs, node_dev):
        """Every kernel the captures will launch, run once first on the
        runner's inboxes: Triton compiles (and loads) a kernel on its first
        call, which a capture forbids."""
        for name, (moe, here, peer_boxes, where, _) in pooled.items():
            if self.world is not None and not here:
                continue  # expert parallelism: none of the layer's experts on this rank, nothing to warm
            node = self.runner.graph.by_name[name]
            dev = node_dev[name]
            example = [runner._arg(r, runner._devices[name], inputs) for r in node.args]
            x_, w_, i_ = (example[k] for k in where)
            start = moe.experts_start_idx
            local = None
            summed = _covers(self._partial_experts, moe, here, peer_boxes) and len(here) == 1 and not peer_boxes
            for pool in here:
                local = triton_ops.fp4_experts_decode(x_, w_, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3,
                                                      pool.w2, pool.s2, moe.swiglu_limit, per_pair=not summed,
                                                      indices=i_, start=start)
            if summed:
                continue
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

    def _capture_all(self, graph, mailboxes, boxes, pooled, peers, driver, flags, F, flag, sides, dev, devs, node_dev,
                     in_copies, resolve, produced, host_nodes, answers):
        """The step as one CUDA graph per GPU (one, unless the layers are sharded): each GPU's
        graph runs its nodes in step order. A value a node on another GPU reads is copied into
        the runner's inbox there and a flag in that GPU's memory raised; the reader's graph
        waits on the flag first. The engine's GPU's graph starts by sending the step's inputs
        to the other GPUs; the chosen tokens stay on the GPU that holds the head, where the
        native loop reads them (native_loop passes it which graph that is)."""
        from . import aoti
        events = []  # per host node: (sent, back), timing events recorded in the graph
        self._gpu_only = 0  # experts nodes answered by GPU pools alone, with no host crossing
        self._branched = 0  # dense nodes captured on a parallel branch

        def reader_dev(node, d2):
            return node_dev.get(node.name) if d2.type != "cuda" else (torch.device("cuda", dev.index) if d2.index is None else d2)

        # The values crossing GPUs, grouped into waves: one flag per (producer GPU, reader GPU,
        # wave), in the reader's memory, written after the wave's last value is copied and
        # waited on once by the reader's first node that needs any of them. A wave ends when
        # the producer GPU itself waits on a flag (whatever it produces after that could depend
        # on the reader's progress, so it gets a flag of its own). The layer-sharded step is a
        # chain, so a boundary between two ranges is one flag, however many values cross it
        # (seven to eight), and the step's inputs ride the same ordering: a GPU's first wait is
        # on a flag written after the engine's GPU copied them out.
        xvalue_wave = {}  # (producer, index, reader GPU) -> wave key (producer GPU, reader GPU, wave)
        need = collections.defaultdict(list)  # node -> wave keys it waits on (in order)
        last_producer = {}  # wave key -> the node whose copies end the wave
        waits = collections.Counter()  # GPU -> flags waited on so far, in step order
        waited_keys = set()
        for node in graph.nodes:
            g = node_dev[node.name]
            for r in dataflow._flat(node.refs()):
                if isinstance(r, dataflow.Ref) and r.kind == "edge" and node_dev[r.node] != g:
                    key = xvalue_wave[(r.node, r.index, g)]
                    if key not in waited_keys:
                        waited_keys.add(key)
                        need[node.name].append(key)
            if need[node.name]:
                waits[g] += 1
            for index, devices, _ in self.runner._fanout[node.name]:
                for d2 in devices:
                    d2 = reader_dev(node, d2)
                    if d2 != g:
                        key = (g, d2, waits[g])
                        xvalue_wave[(node.name, index, d2)] = key
                        last_producer[key] = node.name
        wflags = {key: torch.zeros(1, dtype=torch.int32, device=key[1]) for key in last_producer}
        # the ring write's program counters, one per wave on the producer GPU, made before any capture
        counters = {key: torch.zeros(1, dtype=torch.int32, device=key[0]) for key in last_producer}
        # the step's inputs: a GPU whose first flag comes (transitively) after the engine's GPU
        # copied them out needs no flag of its own for them
        first_wait = {}
        for node in graph.nodes:
            g = node_dev[node.name]
            if g not in first_wait:
                first_wait[g] = need[node.name][0][0] if need[node.name] else None  # the GPU it first waits on
        covered = {dev: True}
        for d in devs[1:]:
            seen, s_ = set(), first_wait.get(d)
            while s_ is not None and s_ != dev and s_ not in seen:
                seen.add(s_)
                s_ = first_wait.get(s_)
            covered[d] = s_ == dev
        in_flags = {d: torch.zeros(1, dtype=torch.int32, device=d) for d in devs[1:] if not covered[d]}
        xflags = wflags  # what _keep holds alive
        out_node = next((r.node for r in dataflow._flat([graph.output]) if isinstance(r, dataflow.Ref) and r.kind == "edge"), None)
        out_dev = node_dev[out_node] if out_node is not None else dev
        if self._sampling is not None and out_dev != dev:  # the choice runs where the logits are
            self._sampling = tuple(t.to(out_dev) for t in self._sampling)
        graphs, output, chosen = {}, None, None
        self.kernel_names = {}
        counts = collections.Counter()
        if len(devs) > 1:  # the step spans GPUs (layer sharding): what crosses, and where the tokens come out
            per = collections.Counter(str(node_dev[n.name]) for n in graph.nodes)
            print(f"[vxpu] step graph: one CUDA graph per GPU, nodes {dict(per)}, {len(xvalue_wave)} values cross "
                  f"GPUs in {len(wflags)} flags ({len(in_copies)} step inputs copied, {len(in_flags)} input flags), "
                  f"the chosen tokens read from {out_dev}", flush=True)
        self.crossings = (len(xvalue_wave), len(wflags), len(in_flags))
        del COPY_LOG[:]
        # A cache written by a layer on one GPU and read by layers on another (layer 20's
        # compressed KV and index keys, read by layers 20 to 39 across a range boundary) was
        # refreshed whole once a step: at 33 rows 138 + 35 MB a step over PCIe, 3.4 ms of a
        # 99 ms step, for the one slot per row the step had written. The replica now takes
        # the step's window only: the writer's GPU gathers rows' slots [position, position +
        # tokens) into a staging buffer right after the writing node, copies it across, and
        # the reader's GPU scatters it into the replica before the first node that reads it,
        # 33 KB instead of 173 MB a step. The positions are the step's inputs, so the gather
        # and scatter are kernels in the graphs with fixed buffers, not copies with baked
        # addresses. Caches not shaped [rows, positions, ...] keep the whole-tensor refresh.
        windows = {}  # (cache name, reader GPU) -> what the window needs
        inputs_now = dict(zip(graph.inputs, self._inputs))
        ids_in, pos_in = inputs_now.get("input_ids"), inputs_now.get("cache_position")
        if CACHE_WINDOWS and ids_in is not None and pos_in is not None and ids_in.dim() == 2 and pos_in.dim() == 1:
            rows_n, toks = ids_in.shape
            writers = {}
            for node in graph.nodes:
                for i in node.writes:
                    a = node.args[i] if i < len(node.args) else None
                    if isinstance(a, dataflow.Ref) and a.kind == "resident":
                        writers[a.name] = node.name
            for (name, d2), replica in list(self.runner._copies.items()):
                if name not in self.runner.state or name not in writers or d2.type != "cuda":
                    continue
                src = dataflow._attr(self.split, name)
                if not (isinstance(src, torch.Tensor) and positional_cache(src, rows_n, toks) and replica.shape == src.shape):
                    continue
                if compressed_cache(name, self.runner.state):
                    continue  # written at compressed slots, not at the step's positions: refreshed whole
                owner = node_dev[writers[name]]
                if owner == d2 or src.device != owner:
                    continue
                tail = tuple(src.shape[2:])
                windows[(name, d2)] = {"writer": writers[name], "owner": owner, "src": src, "replica": replica,
                                       "stage_src": torch.empty((rows_n, toks) + tail, dtype=src.dtype, device=owner),
                                       "stage_dst": torch.empty((rows_n, toks) + tail, dtype=src.dtype, device=d2),
                                       "scattered": False}
                self.runner._fresh.add((name, d2))  # _arg's whole-cache refresh is replaced by the window
        self.windows = len(windows)
        self._windows = windows

        def positions_on(d):
            return in_copies.get(("cache_position", d), pos_in)

        def window_index(w, d):
            pos = positions_on(d).to(torch.int64)
            idx = (pos[:, None] + torch.arange(toks, device=d)).clamp_(max=w["src"].shape[1] - 1)
            return idx.reshape((rows_n, toks) + (1,) * len(w["src"].shape[2:])).expand((rows_n, toks) + tuple(w["src"].shape[2:]))

        pending = collections.defaultdict(list)  # wave key -> [(value, inbox, producer)] the ring write carries
        self._ring_writes = 0
        for d in devs:
            g = torch.cuda.CUDAGraph(keep_graph=True)  # kept, to count its nodes (node_counts)
            # the ranks of a tensor-parallel step capture at once, one thread each: a capture in the
            # default (global) error mode would fault on the other threads' calls
            mode = "thread_local" if self.world is not None else "global"
            keep = []  # the expert kernels' scratch, alive for the capture (triton_ops.CAPTURE_KEEP, the rule in branch_choice)
            self._keeps.append(keep)
            triton_ops.CAPTURE_KEEP = keep if CAPTURE_KEEP else None
            with torch.cuda.device(d), torch.cuda.graph(g, stream=torch.cuda.Stream(device=d), capture_error_mode=mode):
                stream = torch.cuda.current_stream()
                side = sides[d]
                # The parallel branch: a dense node that does not read the node captured before it
                # on this GPU runs on `par`; every node waits on the events of the nodes it reads
                # (recorded in the graph), so the order is the dataflow's and not the step's.
                # The step is a chain at the node level except where the cut made islands
                # (segments.ISLAND_OPS: the attention's two projections), which is what the branch
                # overlaps. Host nodes stay on the main stream (their own capture uses `side`).
                par = torch.cuda.Stream(device=d)
                par.wait_stream(stream)  # forked from the capture's stream before anything runs on it
                done = {}  # node name -> (its stream, the event after it)
                main_last = None  # the node captured last on the main stream
                par_used = False
                if d == dev:
                    for d2 in devs[1:]:  # the step's inputs out to the other GPUs
                        for (name, d3), copy in in_copies.items():
                            if d3 == d2:
                                driver.memcpy(stream, copy, dict(zip(graph.inputs, self._inputs))[name])
                                COPY_LOG.append(("input", name, str(d), str(d2), tuple(copy.shape), str(copy.dtype),
                                                 copy.numel() * copy.element_size()))
                        if d2 in in_flags:
                            driver.write_value(stream, in_flags[d2].data_ptr(), 1)
                elif d in in_flags:
                    driver.wait_value(stream, in_flags[d].data_ptr(), 1)
                    driver.write_value(stream, in_flags[d].data_ptr(), 0)
                for node in graph.nodes:
                    if node_dev[node.name] != d:
                        continue
                    producers = {r.node for r in dataflow._flat(node.refs())
                                 if isinstance(r, dataflow.Ref) and r.kind == "edge" and node_dev[r.node] == d}
                    on_par, wait_main_last, host_waits_branch = branch_choice(
                        node.kind, producers, main_last, graph.by_name[main_last].kind if main_last else None,
                        bool(need[node.name]))
                    s = par if on_par else stream
                    for pname in producers:  # the nodes it reads, wherever they ran
                        ps, pe = done[pname]
                        if ps is not s:
                            s.wait_event(pe)
                    if wait_main_last:  # an island never overlaps the host node before it (branch_choice)
                        s.wait_event(done[main_last][1])
                    for key in need[node.name]:  # the waves of values other GPUs made for this node
                        f = wflags[key]
                        driver.wait_value(s, f.data_ptr(), 1)
                        driver.write_value(s, f.data_ptr(), 0)
                    for (wname, wd2), w in windows.items():  # the window of a shared cache lands in the replica
                        if wd2 == d and not w["scattered"] and any(
                                isinstance(r, dataflow.Ref) and r.kind == "resident" and r.name == wname
                                for r in dataflow._flat(node.refs())):
                            with torch.cuda.stream(s):
                                w["replica"].scatter_(1, window_index(w, d), w["stage_dst"])
                            w["scattered"] = True
                    with torch.cuda.stream(s):
                        args_ = [resolve(r, d) for r in node.args]
                        kwargs_ = {k: resolve(r, d) for k, r in node.kwargs.items()}
                        if host_waits_branch and par_used:
                            stream.wait_stream(par)  # a host node starts after every island before it (branch_choice)
                        if node.kind == "dense":
                            module = getattr(self.split, node.name)
                            if not isinstance(module, aoti.GraphedSegment) or module.runner is None:
                                raise RuntimeError(f"step graph: {node.name} is not a loaded AOTInductor segment")
                            out = module.runner(*args_)
                        elif node.kind in ("allreduce", "allgather"):
                            out = self._capture_collective(node, args_, d, stream)
                        elif node.kind == "embed_rows":
                            out = self._capture_embed_rows(node, args_, d, stream)
                        else:
                            if node.kind == "experts" and (EXPERT_STATS or EXPERT_TRACE):
                                self._count_experts(node, args_, d)
                            out = self._capture_host_node(node, args_, kwargs_, d, stream, side, boxes, pooled, driver,
                                                          flags, flag, F, mailboxes, events,
                                                          push=self._two_part_push(graph, node, d))
                    produced[node.name] = out
                    for (wname, wd2), w in windows.items():  # the step's slots of a shared cache, out to its replica
                        if w["writer"] == node.name and w["owner"] == d:
                            if s is not stream:
                                ev_w = torch.cuda.Event()
                                ev_w.record(s)
                                stream.wait_event(ev_w)
                            with torch.cuda.stream(stream):
                                torch.gather(w["src"], 1, window_index(w, d), out=w["stage_src"])
                            driver.memcpy(stream, w["stage_dst"], w["stage_src"])
                            COPY_LOG.append(("window", wname, str(d), str(wd2), tuple(w["stage_src"].shape),
                                             str(w["stage_src"].dtype), w["stage_src"].numel() * w["stage_src"].element_size()))
                    ended = []
                    for index, devices, _ in self.runner._fanout[node.name]:  # out to readers on other GPUs
                        value = out if index is None else out[index]
                        for d2 in devices:
                            d2 = reader_dev(node, d2)
                            if d2 != d and isinstance(value, torch.Tensor):
                                inbox = self.runner.inboxes[(node.name, index, d2)]
                                key = xvalue_wave[(node.name, index, d2)]
                                if _ring_ok(value, inbox):
                                    # the ring write: the wave's values go across in one kernel with its flag,
                                    # launched when the wave ends (below); a value made on the branch is
                                    # waited for there
                                    pending[key].append((value, inbox, node.name))
                                else:
                                    driver.memcpy(s, inbox, value)
                                COPY_LOG.append(("value", f"{node.name}[{index}]", str(d), str(d2), tuple(value.shape),
                                                 str(value.dtype), value.numel() * value.element_size()))
                                if last_producer[key] == node.name and key not in ended:
                                    ended.append(key)
                    for key in ended:  # the wave's last value copied: its flag, after the copies
                        if pending.get(key):
                            for _, _, pname in pending[key]:
                                if pname != node.name and done[pname][0] is not s:
                                    s.wait_event(done[pname][1])
                            self._ring_writes += triton_ops.ring_write([(v, i) for v, i, _ in pending.pop(key)],
                                                                       counters[key], wflags[key], s)
                        else:
                            driver.write_value(s, wflags[key].data_ptr(), 1)
                    ev = torch.cuda.Event()
                    ev.record(s)
                    done[node.name] = (s, ev)
                    if BRANCH_TRACE and d == devs[0] and (node.layers and int(node.layers[0]) in (0, 1)):
                        waits = [pn for pn in producers if done[pn][0] is not s]
                        def _r(r):
                            if isinstance(r, dataflow.Ref):
                                return f"{r.kind}:{r.name or r.node}" + (f"[{r.index}]" if r.kind == "edge" and r.index is not None else "")
                            return repr(r)[:12]
                        shapes = [f"{tuple(a.shape)}:{str(a.dtype).replace('torch.', '')}@{a.data_ptr() % (1 << 32):x}"
                                  if isinstance(a, torch.Tensor) else repr(a)[:10] for a in args_]
                        outs = out if isinstance(out, (tuple, list)) else (out,)
                        oshapes = [f"{tuple(o.shape)}@{o.data_ptr() % (1 << 32):x}" if isinstance(o, torch.Tensor) else "-" for o in outs]
                        print(f"[branch] {d} {node.name} {node.kind} layers={node.layers} on={'par' if on_par else 'main'} "
                              f"producers={sorted(producers)} waits_on={sorted(waits)} main_last={main_last}\n"
                              f"         args={[ _r(r) for r in node.args ]}\n         shapes={shapes}\n         outs={oshapes}", flush=True)
                    if on_par:
                        par_used = True
                        self._branched += 1
                    else:
                        main_last = node.name
                if par_used:
                    stream.wait_stream(par)  # the branch joins the main stream before the step's end
                if d == out_dev:
                    output = resolve(graph.output, d)
                    # the native loop reads the output with one flat copy: a strided view (the
                    # drafter's tokens are spec_draft(...)[0][:, 1:], stride 6 for 5 columns) is made
                    # contiguous here, in the graph, or every row after the first reads its
                    # neighbour's tokens
                    if isinstance(output, torch.Tensor) and not output.is_contiguous():
                        output = output.contiguous()
                    elif (isinstance(output, (tuple, list)) and len(output) == 1 and isinstance(output[0], torch.Tensor)
                          and not output[0].is_contiguous()):
                        output = type(output)([output[0].contiguous()])
                    # the greedy choice, in the graph: a decode step's logits [rows, tokens, vocab] become each
                    # position's argmax, so a native loop (native_loop) reads token ids, not logits
                    logits = output[0] if isinstance(output, (tuple, list)) and len(output) == 1 else output
                    if isinstance(logits, torch.Tensor) and logits.dim() == 3 and self._sampling is not None:
                        chosen = self._choose(logits, *self._sampling)  # read by the loop on this GPU
                        if GRAPH_TOKEN:
                            # the step feeds itself: its choice into its own input_ids, its position one on
                            # (a one-token step whose inputs live on this GPU; the host path overwrites both
                            # when it feeds a step, so either path may follow). The host nodes are no obstacle:
                            # their inputs reach the step host through the mailboxes' captured copies, from
                            # the graph's own buffers, never from the tensors the host passed in
                            own = dict(zip(graph.inputs, self._inputs))
                            ids_buf, pos_buf = own.get("input_ids"), own.get("cache_position")
                            if (isinstance(ids_buf, torch.Tensor) and isinstance(pos_buf, torch.Tensor)
                                    and ids_buf.device == chosen.device and pos_buf.device == chosen.device
                                    and ids_buf.dim() == 2 and ids_buf.shape[1] == 1 and chosen.shape == ids_buf.shape):
                                ids_buf.copy_(chosen.to(ids_buf.dtype))
                                pos_buf.add_(1)
                                self._graph_token = True
                                print(f"[vxpu] step graph on {chosen.device}: feeds itself (the greedy choice into its "
                                      f"input_ids, cache_position + 1; VXPU_GRAPH_TOKEN)", flush=True)
                            else:
                                print(f"[vxpu] step graph: VXPU_GRAPH_TOKEN set but the step cannot feed itself: input_ids "
                                      f"{getattr(ids_buf, 'shape', None)} on {getattr(ids_buf, 'device', None)}, cache_position on "
                                      f"{getattr(pos_buf, 'device', None)}, chosen {tuple(chosen.shape)} on {chosen.device}", flush=True)
            names = {}
            for k, v in driver.node_counts(g.raw_cuda_graph(), names).items():
                counts[k] += v
            for k, v in names.items():
                self.kernel_names[k] = self.kernel_names.get(k, 0) + v
            g.instantiate()
            graphs[d] = g
        self.node_counts = dict(counts)
        # with the expert counters captured, the count includes their kernels (five a layer):
        # that run's number goes in under its own name, so "kernels" always means the plain step
        self._stat("kernels_with_expert_stats" if EXPERT_STATS else "kernels", self.node_counts.get("kernels"))
        self._stat("crossings", {"values": len(xvalue_wave), "flags": len(wflags), "input_flags": len(in_flags),
                                 "step_inputs_copied": len(in_copies), "ring_writes": self._ring_writes})
        self._ring_counters = counters  # the kernels' counters live as long as the graphs
        self._stat("branched_nodes", self._branched)
        if self._branched:
            print(f"[vxpu] step graph: {self._branched} nodes captured on a parallel branch beside the node before them",
                  flush=True)
        g = graphs[dev]
        peer_graphs = [(d2, graphs[d2]) for d2 in devs[1:]]
        peer_graphs += [self._capture_peer(pdev, peers.index(pdev), driver, flags, F, host_nodes, boxes, pooled)
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
        self._chosen = chosen
        self._flags, self._box_info, self._events = flags, info, events
        # what the graphs use that lives outside their memory pools (the pinned boxes, the other
        # GPUs' inboxes, flags and reply buffers, the slot maps' copies): freed, that memory would
        # be reused under the graphs. The Python worker held it through its mailboxes.
        triton_ops.CAPTURE_KEEP = None
        self._keep = (boxes, pooled, mailboxes, xflags, in_flags, in_copies, graphs, self._keeps)
        weakref.finalize(self, host.stop)  # the engine dropped: the thread lets go of the model

    def _count_experts(self, node, args_, d):
        """The distinct experts a layer's node selects this step, added into this GPU's
        counter at the layer's slot (captured: three small kernels a layer, so only for
        measurement runs, VXPU_EXPERT_STATS=1). The bytes a batched step reads are the
        distinct experts times the expert's size; with 33 different prompts the count is
        between the rows' pairs and the model's experts, and only this number says where."""
        layer, where = _experts_call(getattr(self.split, node.name))
        if layer is None:
            return
        raw = args_[where[2]]
        if EXPERT_TRACE:  # this step's ids into the layer's next slot (the buffers live outside the graph's memory)
            flat = raw.reshape(-1).to(torch.int32)
            n = min(flat.numel(), EXPERT_TRACE_IDS)
            self._expert_trace_shape[layer] = tuple(raw.shape[-2:]) if raw.dim() >= 2 else (raw.numel(), 1)
            pos = self._expert_trace_pos[d][layer:layer + 1]
            row = torch.nn.functional.pad(flat[:n], (0, EXPERT_TRACE_IDS - n), value=-1).unsqueeze(0)
            self._expert_trace[d][layer].index_copy_(0, pos, row)
            pos.add_(1).clamp_(max=EXPERT_TRACE_STEPS - 1)
        if not EXPERT_STATS:
            return
        # the counters live outside the graph's memory: a tensor made inside the capture is re-zeroed
        # by the graph on every replay (its zeros become a memset node), and the count never adds up
        counters = self._expert_counts[d]
        ids = raw.reshape(-1).to(torch.long)
        present = torch.zeros(EXPERT_STATS_EXPERTS, dtype=torch.int32, device=d)
        present.index_fill_(0, ids, 1)
        counters[layer] += present.sum().to(torch.int64)

    def expert_trace(self, reset=False):
        """{layer: int32 array [steps, token rows, top-k]} of the expert ids every replay since the
        capture selected (VXPU_EXPERT_TRACE=1; None without it; the last slot holds the steps past
        EXPERT_TRACE_STEPS). `reset` empties the trace after reading, for the next run."""
        if not getattr(self, "_expert_trace", None):
            return None
        out = {}
        for d, buf in self._expert_trace.items():
            pos = self._expert_trace_pos[d].cpu()
            for layer in range(EXPERT_STATS_LAYERS):
                steps = int(pos[layer])
                if steps == 0 or layer not in self._expert_trace_shape:
                    continue
                rows, k = self._expert_trace_shape[layer]
                ids = buf[layer, :steps, :rows * k].cpu().reshape(steps, rows, k).numpy()
                out[layer] = ids
            if reset:
                self._expert_trace_pos[d].zero_()
        return out or None

    def expert_stats(self):
        """{layer: distinct experts selected, summed over every replay since the capture} (None
        without VXPU_EXPERT_STATS). The caller knows how many steps it ran (the native loop
        replays the graphs without this object), divides, and reports the mean through
        plan.stat("distinct_experts", ...); hero-spec-stats.py does that per run."""
        if not self._expert_counts:
            return None
        total = sum(c.cpu() for c in self._expert_counts.values())
        return {layer: int(total[layer]) for layer in range(EXPERT_STATS_LAYERS) if total[layer] > 0}

    def _stat(self, name, value):
        if self.plan is not None and hasattr(self.plan, "stat"):
            self.plan.stat(name if self.role == "decode" else f"{self.role}_{name}", value)

    def _two_part_push(self, graph, node, dev):
        """The two-part push sum's first part for an experts host node (TWO_PART_SUM): when the next
        collective of the step is the FFN's mixing sum taking this node's output as its `add`, the
        node's kernel pushes its partial to the peers' inboxes for that sum's site, and the collective
        is captured as vxpu::gemv_allreduce_hc2. Returns the push dict (allreduce.two_part_push) or None."""
        if not (TWO_PART_SUM and node.kind == "experts" and getattr(self, "_comm_sums", False)):
            return None
        names = [n.name for n in graph.nodes]
        after = graph.nodes[names.index(node.name) + 1:]
        consumer = next((n for n in after if n.kind in ("allreduce", "allgather")), None)
        if consumer is None or consumer.kind != "allreduce":
            return None
        gemv = self._gemv_sum(consumer.name)
        if not gemv or gemv[3] is None:
            return None
        add = consumer.args[gemv[3]]
        if not (isinstance(add, dataflow.Ref) and add.kind == "edge" and add.node == node.name):
            return None
        from . import allreduce as ar
        c = ar.comm()
        rank = c.rank_of(dev)
        like = self.runner.inboxes[(add.node, add.index, dev)]
        push = ar.two_part_push(c, self._sum_sites, rank, like.numel())
        self._two_part_sites = getattr(self, "_two_part_sites", {})
        self._two_part_sites[consumer.name] = self._sum_sites
        return push

    def _two_part_send(self, push, reply, stream):
        """The two-part push sum's first part when the expert kernel did not push it (another form
        ran, or this rank holds none of the layer's experts and contributes zeros): the push kernel
        sends `reply` to the peers' routed regions and raises their first-round flags. Returns reply."""
        if push is not None and not push["done"]:
            from . import allreduce as ar
            flat = reply.contiguous().view(-1)
            dests = (list(push["dests"]) + [flat] * 3)[:3]
            raise_ = (list(push["flags"]) + [push["counter"]] * 3)[:3]
            with torch.cuda.stream(stream):
                ar._launch_push(flat, dests, flat.numel(), push["counter"], raise_, len(push["flags"]),
                                min(ar.block_elems(), ar.PUSH_BLOCK), False)
            push["done"] = True
        return reply

    def _capture_host_node(self, node, args_, kwargs_, dev, stream, side, boxes, pooled, driver, flags, flag, F,
                           mailboxes, events, push=None):
        """One host node inside a GPU's capture: its mailbox (unless its layer's experts are all
        in GPU pools), this GPU's pool, the other GPUs' pools, and the combined reply."""
        i = len(mailboxes)
        pinned, outbox, like = boxes[node.name]
        moe, here, peer_boxes, where, _ = pooled.get(node.name, (None, [], {}, None, None))
        ep = self.world is not None and node.kind == "experts"
        if ep:
            # Expert parallelism under a tensor-parallel step (hero run 14): every rank runs this node and
            # the ranks' partials are summed by the FFN's allreduce, so a rank computes only the pairs whose
            # experts its own GPU's pools hold (no peer pools, nothing to the host); a rank with none of the
            # layer's experts contributes zeros. Without GPU pools (the banks on the host) rank 0 computes
            # the layer through its mailbox and the other ranks contribute zeros.
            if moe is None:
                if self.rank != 0:
                    return self._two_part_send(push, torch.zeros(tuple(like.shape), dtype=torch.float32, device=dev), stream)
            elif not here:
                x_ = args_[where[0]]
                return self._two_part_send(push, torch.zeros(x_.shape[0], x_.shape[1], dtype=torch.float32, device=dev),
                                           stream)
            else:
                peer_boxes = {}
                if len(here) > 1:
                    raise RuntimeError(f"step graph: {node.name}: several pools of one layer on one GPU under "
                                       "tensor parallelism is not handled")
        # every expert of the layer in a GPU pool: the node never needs the host. No pinned
        # copies, no flags, no outbox; the reply is the pools' alone, and the step host does
        # not get the node (it is not in mailboxes). Decided here, at capture, from the slot
        # maps: a placement holding whole layers is static.
        on_gpus = moe is not None and _covers(self._partial_experts, moe, here, peer_boxes)
        # one pool holding every expert of the layer (a sharded placement): the kernels sum the
        # token's pairs themselves, and no combine runs
        summed = on_gpus and len(here) == 1 and not peer_boxes and _all_held(moe, here, peer_boxes)
        # Under expert parallelism this card's pool holds a quarter of the layer's experts and its reply is a
        # partial the FFN's allreduce adds: the direct kernel sums the held pairs itself (0 for the rest), so
        # the masked k-sum below (a where, a zeros and top-k adds a layer, about ten launches on every card)
        # is not needed (hero run 15, the tensor-parallel step's kernel count).
        if ep and on_gpus and len(here) == 1 and not peer_boxes:
            summed = True
        hot = []
        if here:
            x_, w_, i_ = (args_[k] for k in where)
            side.wait_stream(stream)
            with torch.cuda.stream(side):
                for pool in here:
                    hot.append((pool, triton_ops.fp4_experts_decode(
                        x_, w_, pool.slot_gpu, pool.w1, pool.s1, pool.w3, pool.s3, pool.w2, pool.s2,
                        moe.swiglu_limit, per_pair=not summed, indices=i_, start=moe.experts_start_idx,
                        push=push if summed else None)))
        for pdev, (pool, inbox, flag_in, _, _, _) in peer_boxes.items():  # the other GPUs first: they
            for dst, k in zip(inbox, where):                              # wait on copies, not on the CPU
                driver.memcpy(stream, dst, args_[k])
            driver.write_value(stream, flag_in.data_ptr(), 1)
        if on_gpus:
            reply = torch.empty_like(like, device=dev)  # the CPU's share: none; never read
        else:
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
            if summed:
                reply = hot[0][1]
            elif ep and hot and not peer_boxes:  # this card's pairs only (expert parallelism): a masked k-sum
                pool, replies = hot[0]
                reply = torch.where((pool.slot_gpu[i_ - start] >= 0).unsqueeze(-1), replies, torch.zeros_like(replies))
                y = torch.zeros(reply.shape[0], reply.shape[2], dtype=torch.float32, device=dev)
                for k in range(reply.shape[1]):
                    y = y + reply[:, k]
                reply = y
            elif len(hot) <= 1 and len(peer_boxes) <= 1:  # one kernel for the choice and the sum
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
        if on_gpus:
            self._gpu_only += 1
        else:
            mailboxes.append((node, pinned, kwargs_, outbox, F * i, pooled.get(node.name)))
        return self._two_part_send(push, reply, stream)

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
            # this GPU may also run a range of its own: the peer graph and the range graph replay at
            # once there, so the expert kernels' arrival counters are keyed apart (hero run 14)
            triton_ops.FP4_COUNTER_GRAPH = ("peer", p)
            try:
                self._capture_peer_work(g, stream, work, driver)
            finally:
                triton_ops.FP4_COUNTER_GRAPH = None
        return pdev, g

    def _capture_peer_work(self, g, stream, work, driver):
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
