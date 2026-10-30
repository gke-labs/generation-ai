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

"""Custom ops for the tensors a graph does not hold.

A model larger than its device keeps its biggest tensors outside the
graph: the mixture-of-experts banks (272 GB of fp4 for
DeepSeek-V4.1-Flash) and the n-gram tables (2 x 98 GB). The exporter
traces calls to the two ops below in their place, so the artifact's
graphs carry the dense model only, and the executor answers the calls
from wherever it placed those tensors: host memory, a memory-mapped
file on local SSD, or another device. The math is the fork's own code
on the same bytes, so the result does not depend on the placement.

Each engine owns what it placed (an External) and makes it the ops'
source only while it runs a graph; the exporter only records the shape
information tracing needs. The two ops that do host work are tagged
cudagraph_unsafe and, for torch.compile, kept out of the compiled graph
(prepare_compile), so CUDA graphs cover the dense segments between.
"""

import functools
import contextlib
import threading
import time

import torch
from torch.library import custom_op

from . import experts_cpu, kernels, triton_ops
from .plan import default_plan, implementation


class External:
    """The tensors one engine placed outside its graphs: the fork's MoE
    module per layer (holding that layer's banks) and the fork's n-gram
    table module per engram layer. Owned by the engine, so they go when
    it goes, and active only while that engine runs a graph (see bound):
    two engines in one process, or a load superseded while in flight,
    never see each other's banks."""

    def __init__(self):
        self.experts = {}
        self.engram = {}
        # host nodes a node server serves: (collection, index) -> RemoteNode
        self.remote_nodes = {}
        # fp8 dense weights dequantized to bf16 on first use on a GPU, keyed
        # by the weight's storage address (see fp8_linear): engine-owned so
        # the copies go with the engine, and shared by its two graphs
        self.bf16 = {}
        # bf16 weights widened to fp32 once where no kernel reads bf16
        # directly (the output head off the Triton path, see head_linear)
        self.f32 = {}
        # the engine's physical plan (plan.Plan): which implementation of
        # each op runs here; ops called with no engine bound use a default
        self.plan = None
        # a telemetry.Recorder while a generate call asked for telemetry
        self.telemetry = None
        # pinned host buffers for the expert op's device crossings in decode
        # (see _to_host and _to_device), reused call to call
        self.pinned = {}


# layer id -> row width, for shape inference while tracing (set by the exporter)
ENGRAM_DIM = {}

_active = threading.local()


@contextlib.contextmanager
def bound(external):
    """Make `external` what the ops answer from, in this thread, for the
    duration of the block (an engine wraps each graph call)."""
    previous = getattr(_active, "external", None)
    _active.external = external
    try:
        yield
    finally:
        _active.external = previous


def _plan(device_type):
    """The bound engine's plan, or the default plan for device_type."""
    ext = getattr(_active, "external", None)
    plan = getattr(ext, "plan", None) if ext is not None else None
    return plan if plan is not None else default_plan(device_type)


class RemoteNode:
    """A host node served by a node server (nodeserver.py): its host op
    sends the op's tensors to the node's mailbox (address, the server's
    node index) and returns the reply, on the inputs' device."""

    def __init__(self, client, index, address):
        self.client, self.index, self.address = client, index, address

    def __call__(self, *tensors):
        out = self.client.call(self.index, [t.cpu() for t in tensors])
        return out[0].to(tensors[0].device)


def _remote(what, key):
    """The node server that serves what[key], if one does."""
    ext = getattr(_active, "external", None)
    return getattr(ext, "remote_nodes", {}).get((what, key)) if ext is not None else None


def _current(what, key):
    ext = getattr(_active, "external", None)
    table = getattr(ext, what, None) if ext is not None else None
    if table is None or key not in table:
        raise RuntimeError(f"vxpu::{what}: no {what} bound for {key} (the engine binds the artifact's "
                           "external tensors and runs its graphs under ops.bound)")
    return table[key]


@custom_op("vxpu::moe_experts", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,))
def moe_experts(x_in: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                layer: int) -> torch.Tensor:
    """The routed experts of `layer` applied to x_in [n, dim] (already
    input-quantized, as the module does), with gate weights and expert
    indices [n, topk]. Returns y [n, dim] float32, what the module's own
    routings return. One token takes the batched routing (gather the
    chosen experts), more take the grouped one (sorted by expert), the
    same choice the exporter made per graph."""
    if x_in.is_meta:  # a graph walked on the meta device (loading, device passes): shape only
        return x_in.new_empty((x_in.shape[0], x_in.shape[1]), dtype=torch.float32)
    remote = _remote("experts", layer)
    if remote is not None:  # a node server holds the layer: its mailbox computes it
        rec = _recorder()
        t0 = time.perf_counter() if rec is not None else 0.0
        y = remote(x_in, weights, indices)
        if rec is not None:
            rec.host_op("moe_experts", impl_ms=(time.perf_counter() - t0) * 1e3, remote_pairs=float(indices.numel()))
            rec.selected_experts(layer, indices.flatten().tolist())
        return y
    moe = _current("experts", layer)
    host = moe.w1.device if not moe.w1.is_meta else torch.device("cpu")  # meta: the banks are elsewhere
    # decode-sized inputs cross through the engine's pinned buffers; a
    # prefill keeps plain copies (its crossing is noise against its compute,
    # and pinned buffers sized for the longest prompt would stay pinned)
    crossing = x_in.device.type == "cuda" and host.type == "cpu"
    rec = _recorder()
    t0 = time.perf_counter() if rec is not None else 0.0
    boxes = getattr(moe, "_vxpu_mailboxes", None)
    if boxes is not None:
        impl = _plan(host.type).select("vxpu::moe_experts", host.type, key=f"rows={x_in.shape[0]}", moe=moe,
                                       rows=x_in.shape[0], mailboxes=True)
        if impl.fn is _moe_mailboxes:
            return _moe_mailboxes(moe, boxes, x_in, weights, indices, layer, rec, t0, crossing)
    hot = getattr(moe, "_vxpu_hot", None) if crossing else None
    rows = x_in.shape[0]
    impl = _plan(host.type).select("vxpu::moe_experts", host.type, key=f"rows={rows}", moe=moe, rows=rows,
                                   hot=hot is not None)
    if impl.fn is _moe_gpu_hot:
        return _moe_gpu_hot(moe, hot, x_in, weights, indices, layer, rec, t0)
    if crossing:
        x, w, i = _to_host(x_in, weights, indices)
    else:
        x, w, i = x_in.to(host), weights.to(host), indices.to(host)
    res = getattr(moe, "_vxpu_hot", None)
    if rec is None:
        y = impl.fn(moe, x, w, i)
        if res is not None and res.mover is not None:  # a prefill: the request's selections, while the CPU works
            res.mover.observe(res, i.flatten().tolist())
        return _to_device(y, x_in.device) if crossing else y.to(x_in.device)
    t1 = time.perf_counter()
    y = impl.fn(moe, x, w, i)
    if res is not None and res.mover is not None:
        res.mover.observe(res, i.flatten().tolist())
    t2 = time.perf_counter()
    out = _to_device(y, x_in.device) if crossing else y.to(x_in.device)
    t3 = time.perf_counter()
    # wait_to_host includes waiting for the GPU's preceding work (the copy
    # synchronizes the stream): on the hybrid placement, the CPU idle while
    # the GPU finishes the segment before
    rec.host_op("moe_experts", wait_to_host_ms=(t1 - t0) * 1e3, impl_ms=(t2 - t1) * 1e3, to_device_ms=(t3 - t2) * 1e3)
    rec.selected_experts(layer, i.flatten().tolist())
    return out


def _recorder():
    """The telemetry recorder of the engine running a graph, if it asked for one."""
    ext = getattr(_active, "external", None)
    return getattr(ext, "telemetry", None) if ext is not None else None


def _cpu_experts(plan, moe=None, **_):
    return plan["kernels"] and experts_cpu.applicable(moe)


@implementation("vxpu::moe_experts", "mailboxes", ("cpu",), "rounding",
                "decode: each (token, k) pair answered by its expert's owner (the CPU kernels, a GPU pool) and the "
                "replies summed in k order, so the sum does not depend on where each expert ran (mailboxes.py)",
                applies=lambda plan, rows=0, mailboxes=False, **ctx: mailboxes
                and plan["fused_experts"] and _cpu_experts(plan, **ctx),
                host=True)
def _moe_mailboxes(moe, boxes, x_in, weights, indices, layer, rec, t0, crossing):
    host = _to_host(x_in, weights, indices) if crossing else None
    t1 = time.perf_counter()
    y, counts = boxes.forward(x_in, weights, indices, host_inputs=host)
    t2 = time.perf_counter()
    res = boxes.residency
    selected = indices.flatten().tolist() if (rec is not None or (res is not None and res.mover is not None)) else None
    started = completed = 0
    if res is not None and res.mover is not None:
        started, completed = res.mover.observe(res, selected)
    if rec is not None:
        t3 = time.perf_counter()
        gpu = sum(v for k, v in counts.items() if k not in ("cpu", "remote"))
        rec.host_op("moe_experts", wait_to_host_ms=(t1 - t0) * 1e3, impl_ms=(t2 - t1) * 1e3, to_device_ms=(t3 - t2) * 1e3,
                    gpu_pairs=float(gpu), cpu_pairs=float(counts.get("cpu", 0)),
                    **({"remote_pairs": float(counts["remote"])} if "remote" in counts else {}),
                    **({"moves_started": float(started), "moves_completed": float(completed)} if res and res.mover else {}))
        rec.selected_experts(layer, selected)
    return y


@implementation("vxpu::moe_experts", "gpu-hot+cpu", ("cpu",), "rounding",
                "a placement holds this layer's hot experts on the GPUs (experts_gpu.py): the token's hot pairs computed "
                "there on side streams while the CPU kernels compute the cold ones, and the sums added",
                applies=lambda plan, rows=0, hot=False, **ctx: hot and plan["triton"]
                and _cpu_experts(plan, **ctx), host=True)
def _moe_gpu_hot(moe, res, x_in, weights, indices, layer, rec, t0):
    dev = x_in.device
    current = torch.cuda.current_stream(dev)
    partial = []
    for pool in res.pools:  # every GPU's share is launched before the host waits for anything
        pool.stream.wait_stream(current)  # the dense work before produces x_in
        with torch.cuda.stream(pool.stream):
            if pool.device == dev:
                x, w, i = x_in, weights, indices
            else:  # a few KB to the other GPU
                x, w, i = (t.to(pool.device, non_blocking=True) for t in (x_in, weights, indices))
            slots = pool.slot_gpu[i - moe.experts_start_idx]
            partial.append((pool, triton_ops.fp4_experts_decode(x, w, slots, pool.w1, pool.s1, pool.w3, pool.s3,
                                                                pool.w2, pool.s2, moe.swiglu_limit)))
    # With every expert of the layer on a GPU, nothing goes to the host and nothing waits for it:
    # the step's work stays queued on the GPUs.
    i = None
    n_cold = 0
    y_cold = None
    if not bool(res.held_cpu.all()):
        x, w, i = _to_host(x_in, weights, indices)  # the GPUs compute the hot pairs meanwhile
        cold = ~res.held_cpu[i - moe.experts_start_idx]
        n_cold = int(cold.sum())
    t1 = time.perf_counter()
    if n_cold:
        fused = _plan("cpu")["fused_experts"]
        if x.shape[0] == 1:  # one token: only its cold pairs reach the CPU kernels
            y_cold = experts_cpu.forward_batched(moe, x, w[cold].view(1, -1), i[cold].view(1, -1), fused=fused)
        else:  # hot pairs weigh 0: h = w * silu(g) * u = 0, so they add exactly nothing
            y_cold = experts_cpu.forward_batched(moe, x, torch.where(cold, w, torch.zeros_like(w)), i, fused=fused)
    t2 = time.perf_counter()
    y = _to_device(y_cold, dev) if y_cold is not None else None
    for pool, y_pool in partial:
        if pool.device == dev:
            current.wait_stream(pool.stream)
            y_pool.record_stream(current)
        else:  # only now does this device's stream wait for the other GPU
            with torch.cuda.stream(pool.stream):
                y_pool = y_pool.to(dev, non_blocking=True)
            current.wait_stream(pool.stream)
            y_pool.record_stream(current)
        y = y_pool if y is None else y + y_pool
    started = completed = 0
    if res.mover is None and rec is None:
        return y
    if i is None:  # observed only: the indices come to the host for the mover and the recorder
        i = indices.cpu()
    selected = i.flatten().tolist()
    if res.mover is not None:  # after this call's kernels, so a move's unmapping is ordered after them
        started, completed = res.mover.observe(res, selected)
    if rec is not None:
        t3 = time.perf_counter()
        rec.host_op("moe_experts", wait_to_host_ms=(t1 - t0) * 1e3, impl_ms=(t2 - t1) * 1e3, to_device_ms=(t3 - t2) * 1e3,
                    gpu_pairs=float(i.numel() - n_cold), cpu_pairs=float(n_cold),
                    **({"moves_started": float(started), "moves_completed": float(completed)} if res.mover else {}))
        rec.selected_experts(layer, selected)
    return y


# The routed experts' implementations, run on the banks' device: the fused
# fp4 kernels (csrc/; same routings, no dequantized copy), or the module's
# own routings (the reference computation). The kernels group each
# expert's tokens and work in tiles, so one path serves every row count:
# a decode step, batched or speculative, and a prefill. (A dequantize-
# and-GEMM path for long prefills lost at every length measured: on the
# hero VM, 2414 tokens took 18.3 s on the kernels against 29.7 s.)
@implementation("vxpu::moe_experts", "cpu-fused", ("cpu",), "rounding",
                "kernels on, fused experts on: one C++ call per layer, any number of tokens",
                applies=lambda plan, rows=0, **ctx: rows >= 1 and plan["fused_experts"] and _cpu_experts(plan, **ctx), host=True)
def _moe_cpu_fused(moe, x, w, i):
    return experts_cpu.forward_batched(moe, x, w, i, fused=True)


@implementation("vxpu::moe_experts", "cpu-three-call", ("cpu",), "rounding",
                "kernels on, fused experts off: three kernel calls with the module's SwiGLU between",
                applies=lambda plan, rows=0, **ctx: rows >= 1 and _cpu_experts(plan, **ctx), host=True)
def _moe_cpu_three_call(moe, x, w, i):
    return experts_cpu.forward_batched(moe, x, w, i, fused=False)


@implementation("vxpu::moe_experts", "reference", ("cpu", "cuda"), "bit-identical",
                "the module's own routings (batched for one token, grouped for more)", reference=True, host=True)
def _moe_reference(moe, x, w, i):
    return moe._forward_batched(x, w, i) if x.shape[0] == 1 else moe._forward_grouped(x, w, i)


def _pinned(name, like):
    """A pinned host tensor shaped and typed like `like`, kept by the bound
    engine (the op has already required one) and reused call to call;
    regrown when a call needs more."""
    table = _active.external.pinned
    nbytes = like.numel() * like.element_size()
    buf = table.get(name)
    if buf is None or buf.numel() < nbytes:
        buf = table[name] = torch.empty(max(nbytes, 1 << 16), dtype=torch.uint8, pin_memory=True)
    return buf[:nbytes].view(like.dtype).view(like.shape)


def _to_host(x_in, weights, indices):
    """The expert op's three inputs to the host with one synchronize:
    three asynchronous copies into pinned buffers, then the stream waits
    once. Three .to("cpu") calls are three synchronizing pageable copies,
    about 28 us a layer against about 13."""
    out = []
    for name, t in (("x", x_in), ("w", weights), ("i", indices)):
        host = _pinned(name, t)
        host.copy_(t, non_blocking=True)
        out.append(host)
    torch.cuda.current_stream(x_in.device).synchronize()
    return tuple(out)


def _to_device(y, device):
    """The result back without blocking: staged in a pinned buffer and
    copied asynchronously on the current stream. The buffer is reused by
    the next call only after that call's input copy has synchronized the
    same stream, by which point this copy has completed."""
    host = _pinned("y", y)
    host.copy_(y)
    return host.to(device, non_blocking=True)


@moe_experts.register_fake
def _(x_in, weights, indices, layer):
    return x_in.new_empty((x_in.shape[0], x_in.shape[1]), dtype=torch.float32)


@custom_op("vxpu::engram_rows", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,))
def engram_rows(indices: torch.Tensor, table: int) -> torch.Tensor:
    """The rows of n-gram table `table` for hash ids `indices`, dequantized
    to bf16 [..., dim], what the module's lookup returns."""
    if indices.is_meta:
        return indices.new_empty(tuple(indices.shape) + (ENGRAM_DIM[table],), dtype=torch.bfloat16)
    remote = _remote("engram", table)
    if remote is not None:  # a node server holds the table
        rec = _recorder()
        t0 = time.perf_counter() if rec is not None else 0.0
        out = remote(indices)
        if rec is not None:
            rec.host_op("engram_rows", total_ms=(time.perf_counter() - t0) * 1e3)
        return out
    embed = _current("engram", table)
    host = embed.weight.device
    impl = _plan(host.type).select("vxpu::engram_rows", host.type, embed=embed)
    rec = _recorder()
    if rec is None:
        return impl.fn(embed, indices.to(host)).to(indices.device)
    t0 = time.perf_counter()
    out = impl.fn(embed, indices.to(host)).to(indices.device)
    rec.host_op("engram_rows", total_ms=(time.perf_counter() - t0) * 1e3)
    return out


@implementation("vxpu::engram_rows", "host-file", ("cpu",), "bit-identical",
                "kernels on: the rows read straight from the checkpoint file with io_uring (or pread fallback); "
                "dequantized as the module does",
                applies=lambda plan, embed=None, **_: plan["kernels"] and kernels.available() and getattr(embed, "file_backed", None) is not None,
                host=True)
def _engram_file(embed, indices):
    k = kernels.load()
    mask = (indices < embed.vocab_start_idx) | (indices >= embed.vocab_end_idx)
    local = (indices - embed.vocab_start_idx).masked_fill(mask, 0)
    fb = embed.file_backed
    values, scales = k.gather_table_rows(
        fb.fd, fb.weight_offset, fb.weight_row_bytes,
        fb.scale_offset, fb.scale_row_bytes, fb.rows, local)
    if fb.weight_dtype != values.dtype:
        values = values.view(fb.weight_dtype)
    if fb.scale_dtype != scales.dtype:
        scales = scales.view(fb.scale_dtype)
    values = values.float().unflatten(-1, (-1, embed.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


@implementation("vxpu::engram_rows", "host-parallel", ("cpu",), "bit-identical",
                "kernels on: the rows gathered by the kernel threads, so a table memory-mapped from SSD reads a "
                "step's rows concurrently (one page fault per thread, not one after another); dequantized as the "
                "module does",
                applies=lambda plan, embed=None, **_: plan["kernels"] and kernels.load() is not None and getattr(embed, "file_backed", None) is None,
                host=True)
def _engram_parallel(embed, indices):
    k = kernels.load()
    mask = (indices < embed.vocab_start_idx) | (indices >= embed.vocab_end_idx)
    local = (indices - embed.vocab_start_idx).masked_fill(mask, 0)
    # the module's lookup with its two F.embedding gathers made concurrent;
    # the arithmetic is the module's, so the rows are bit-identical
    values = k.gather_rows(embed.weight, local)
    scales = k.gather_rows(embed.scale, local)
    values = values.float().unflatten(-1, (-1, embed.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


@implementation("vxpu::engram_rows", "host-lookup", ("cpu", "cuda"), "bit-identical",
                "the module's own lookup, on the table's device (RAM or a memory-mapped file)", reference=True, host=True)
def _engram_lookup(embed, indices):
    if getattr(embed, "file_backed", None) is not None:
        raise RuntimeError("this n-gram table is read from its checkpoint file and needs the CPU kernels")
    return embed.lookup(indices)


@engram_rows.register_fake
def _(indices, table):
    return indices.new_empty(tuple(indices.shape) + (ENGRAM_DIM[table],), dtype=torch.bfloat16)


@custom_op("vxpu::fp8_linear", mutates_args=())  # pure tensor work on every device: capturable
def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The fork's fp8 linear: x [..., K] bf16, weight [N, K] fp8 with
    [N/32, K/32] e8m0 block scales -> [..., N] in x's dtype. The
    activation is quantized per 32-block (the fork's act_quant) as the
    reference does; on the CPU the fused kernel reads the fp8 weight once
    (one token: a GEMV per row; more: decode once and GEMM); on a GPU a
    bf16 copy of the weight, made once and kept while memory allows
    (_bf16_weight); elsewhere the fork's own fp8_gemm."""
    if x.is_meta:
        return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],))
    return select("vxpu::fp8_linear", x, weight, scale).fn(x, weight, scale)


def _fp8_linear_context(x, weight, scale):
    rows = x.numel() // x.shape[-1]
    return x.device.type, f"{rows}x{x.shape[-1]}->{weight.shape[0]}", {"rows": rows}


@implementation("vxpu::fp8_linear", "cpu-kernel", ("cpu",), "rounding",
                "kernels on: the fused kernel reads the fp8 weight once and quantizes the activation itself",
                applies=lambda plan, **_: plan["kernels"] and kernels.load() is not None)
def _fp8_linear_cpu(x, weight, scale):
    K = x.shape[-1]
    y = kernels.load().fp8_linear(x.reshape(-1, K).contiguous().to(torch.bfloat16), weight, scale)
    return y.reshape(tuple(x.shape[:-1]) + (weight.shape[0],)).to(x.dtype)


def _fusing(plan, lowering, op):
    """Whether op's decomposed body applies: only when the plan lowers a graph for a compiler to
    fuse, and the plan fuses that op (an eager call would run the body op by op)."""
    return lowering and op in plan["fuse"]


def round_e4m3(q):
    """q (fp32, within +-448) onto the fp8 e4m3 grid, ties to even, in plain tensor operations a
    compiler fuses (a cast to float8 needs sm_89 in Triton): a normal value keeps 3 of its 23
    mantissa bits, a subnormal one (below 2^-6) becomes a multiple of 2^-9."""
    bits = q.view(torch.int32)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) & ~0xFFFFF).view(torch.float32)
    return torch.where(q.abs() < 0.015625, torch.round(q * 512.0) / 512.0, normal)


def act_quant_dequant(x):
    """The fork's act_quant and its dequantization in tensor operations: per block of 32 along the
    last dimension, a power-of-two scale (ue8m0) putting the block's amax at most at 448, and the
    values rounded to fp8 e4m3, returned in fp32."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (round_e4m3((blocks / scale).clamp(-448.0, 448.0)) * scale).flatten(-2)


FUSED_GEMV_MAX_ROWS = 32  # up to here a fused linear is one reduction kernel; beyond, a GEMM


@implementation("vxpu::fp8_linear", "gpu-fused", ("cuda",), "rounding",
                "decomposed for the compiler (lowering, VXPU_FUSE; where it measures cheaper): the activation's "
                "quantization fused with what produces it, the fp8 weight dequantized inside the product's reduction "
                "(a GEMM beyond 32 rows)",
                applies=lambda plan, lowering=False, **_: _fusing(plan, lowering, "fp8_linear"),
                lowerable=True, fused=True)
def _fp8_linear_fused(x, weight, scale, tk=None):
    N, K = weight.shape
    xq = act_quant_dequant(x).reshape(-1, K)
    w = (weight.float().unflatten(1, (K // 32, 32)).unflatten(0, (N // 32, 32))
         * scale.float()[:, None, :, None]).reshape(N, K)
    if xq.shape[0] <= FUSED_GEMV_MAX_ROWS:
        y = (xq[:, None, :] * w[None, :, :]).sum(-1)
    else:
        y = torch.mm(xq.to(torch.bfloat16), w.to(torch.bfloat16).t()).float()
    return y.to(x.dtype).reshape(*x.shape[:-1], N)


# On a GPU the reference expands the fp8 weight to fp32 on every call, about
# 15x the cost of the matmul. The quantized activation is exact in bf16, and
# one Triton kernel computes it (the aten version is a dozen kernels). A
# decode step's few rows then go through the fp8 GEMV, which reads the fp8
# weight and its scales directly; longer inputs use a bf16 copy of the
# weight (fp8 values times a power-of-two block scale are exact in bf16),
# dequantized once and kept, and torch's GEMM.
@implementation("vxpu::fp8_linear", "gpu-fp8-gemv", ("cuda",), "rounding",
                f"Triton, at most {triton_ops.GEMV_ROWS_MAX} rows (decode): the activation quantized by one kernel, "
                "the fp8 weight read once for all rows",
                applies=lambda plan, rows=0, **_: plan["triton"] and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(),
                lowerable=True)
def _fp8_linear_gemv(x, weight, scale, tk=triton_ops):
    xf = tk.act_quant_dequant(x)
    y = tk.fp8_gemv(xf.reshape(-1, xf.shape[-1]), weight, scale)
    return y.reshape(tuple(x.shape[:-1]) + (weight.shape[0],)).to(x.dtype)


@implementation("vxpu::fp8_linear", "gpu-bf16-copy", ("cuda",), "rounding",
                "longer inputs (prefill), or no Triton: torch's GEMM over a bf16 copy of the weight, made once and "
                "kept while memory allows")
def _fp8_linear_bf16(x, weight, scale):
    if triton_ops.available():
        xf = triton_ops.act_quant_dequant(x)
    else:
        ds, ka = _fork_kernels()
        K = x.shape[-1]
        xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
        xf = (xq.to(torch.bfloat16).unflatten(-1, (K // 32, 32)) * xs.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)
    return torch.nn.functional.linear(xf, _bf16_weight(weight, scale)).to(x.dtype)


@implementation("vxpu::fp8_linear", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's act_quant and fp8_gemm (aten)", reference=True)
def _fp8_linear_reference(x, weight, scale):
    ds, ka = _fork_kernels()
    xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
    return ka.fp8_gemm(xq, xs, weight, scale, ds.scale_dtype, ds.fp8_block_size).to(x.dtype)


def _fork_kernels():
    from .deepseek_v41 import _fork
    ds = _fork()
    import kernel_aten as ka
    return ds, ka


@custom_op("vxpu::fp8_grouped_linear", mutates_args=())  # pure tensor work: capturable
def fp8_grouped_linear(o: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The fork's grouped output projection (wo_a): o [b, s, g, d], already
    on the fp8 grid, times a block-diagonal fp8 weight [g * r, d] with
    [g * r / 32, d / 32] e8m0 block scales -> [b, s, g, r] in o's dtype.
    The reference dequantizes the whole weight and runs an einsum on every
    call. On a GPU a decode step's rows take the fp8 GEMV (the weight read
    once, as fp8) and longer inputs the bf16 copy and an einsum; elsewhere
    the fork's own computation (bit-identical)."""
    *lead, g, d = o.shape
    if o.is_meta:
        return o.new_empty(tuple(lead) + (g, weight.shape[0] // g))
    return select("vxpu::fp8_grouped_linear", o, weight, scale).fn(o, weight, scale)


def _grouped_context(o, weight, scale):
    *lead, g, d = o.shape
    r = weight.shape[0] // g
    rows = o.numel() // (g * d)
    return o.device.type, f"{rows}x{g}x{d}->{g}x{r}", {"rows": rows}


@implementation("vxpu::fp8_grouped_linear", "gpu-fp8-gemv", ("cuda",), "rounding",
                "Triton, at most 4 rows (decode): the grouped fp8 GEMV, the weight read once as fp8",
                applies=lambda plan, rows=0, **_: plan["triton"] and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(),
                lowerable=True)
def _grouped_gemv(o, weight, scale, tk=triton_ops):
    *lead, g, d = o.shape
    r = weight.shape[0] // g
    rows = o.numel() // (g * d)
    y = tk.fp8_gemv(o.reshape(rows, g * d), weight, scale, group_rows=r, out_dtype=o.dtype)
    return y.reshape(tuple(lead) + (g, r))


@implementation("vxpu::fp8_grouped_linear", "gpu-bf16-copy", ("cuda",), "rounding",
                "Triton, longer inputs (prefill): an einsum over the cached bf16 copy",
                applies=lambda plan, **_: plan["triton"])
def _grouped_bf16(o, weight, scale):
    *lead, g, d = o.shape
    r = weight.shape[0] // g
    return torch.einsum("bsgd,grd->bsgr", o, _bf16_weight(weight, scale).view(g, r, d)).contiguous()


@implementation("vxpu::fp8_grouped_linear", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's dequantize and einsum", reference=True)
def _grouped_reference(o, weight, scale):
    *lead, g, d = o.shape
    r = weight.shape[0] // g
    from .deepseek_v41 import _fork
    w = _fork()._dequantized(weight, scale).view(g, r, d)
    return torch.einsum("bsgd,grd->bsgr", o, w).contiguous()


@fp8_grouped_linear.register_fake
def _(o, weight, scale):
    *lead, g, d = o.shape
    return o.new_empty(tuple(lead) + (g, weight.shape[0] // g))


@functools.lru_cache(1)
def _fp8_native():
    """Whether this GPU has fp8 (e4m3) in hardware, which the fp8 GEMV loads its weights as
    (sm_89 and newer; an A100 is sm_80). Without it the bf16 copy serves the fp8 linears."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


# inputs up to this many rows take the fp8 GEMV on a GPU, one pass over the
# weight for all of them (triton_ops.gemv_rows); longer ones the bf16 copy
# and a GEMM
FP8_GEMV_MAX_ROWS = triton_ops.GEMV_ROWS_MAX


def _bf16_weight(weight, scale):
    """weight [N, K] fp8 with [N/32, K/32] e8m0 block scales as bf16 (exact),
    cached in the active engine's External when there is one and the copy
    leaves enough device memory free (_may_cache_bf16). A weight that is
    not cached is dequantized again on its next call: slower, but an
    engine whose fp8 weights fit always runs."""
    ext = getattr(_active, "external", None)
    key = (weight.untyped_storage().data_ptr(), weight.storage_offset(), tuple(weight.shape))
    if ext is not None and key in ext.bf16:
        return ext.bf16[key]
    N, K = weight.shape
    w = (weight.to(torch.bfloat16).view(N // 32, 32, K // 32, 32)
         * scale.to(torch.bfloat16).view(N // 32, 1, K // 32, 1)).view(N, K)
    if ext is not None and _may_cache_bf16(ext, w):
        ext.bf16[key] = w
    return w


def _may_cache_bf16(ext, w):
    """Whether to keep w (a bf16 copy just made): the copies double the
    dense weights' footprint, which is the difference between fitting a
    card and not. The plan's bf16_weight_cache=False never keeps them;
    otherwise a copy is kept while free device memory (the driver's plus
    torch's reserved-but-unused) stays above the plan's headroom. The first
    refusal is reported once (and noted in the plan); later weights keep
    the per-call path."""
    plan = _plan(w.device.type)
    if not plan["bf16_weight_cache"]:
        return False
    if not w.is_cuda:
        return True
    headroom = int(plan["bf16_cache_headroom_gb"] * 1e9)
    free, _ = torch.cuda.mem_get_info(w.device)
    free += torch.cuda.memory_reserved(w.device) - torch.cuda.memory_allocated(w.device)
    if free >= headroom:
        return True
    if not getattr(ext, "bf16_full", False):
        ext.bf16_full = True
        held = sum(t.numel() * t.element_size() for t in ext.bf16.values())
        msg = (f"bf16 weight cache stopped at {held / 1e9:.1f} GB ({len(ext.bf16)} weights): less than "
               f"{headroom / 1e9:.1f} GB of device memory would remain; the other fp8 linears dequantize per call")
        print(f"[vxpu] {msg}", flush=True)
        plan.note(msg)
    return False


@custom_op("vxpu::shared_ffn", mutates_args=())  # pure tensor work: capturable
def shared_ffn(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor,
               w2: torch.Tensor, s2: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """An fp8 SwiGLU FFN over x [.., K]: the fork's Expert.forward (the MoE's shared expert), w1 and
    w3 [I, K] and w2 [K, I] fp8 with e8m0 block scales, the SwiGLU's clamps at swiglu_limit (0 for
    none). One region of the decode step: about nine kernels as aten, three as Triton."""
    impl = select("vxpu::shared_ffn", x, w1, s1, w3, s3, w2, s2, swiglu_limit)
    return impl.fn(x, w1, s1, w3, s3, w2, s2, swiglu_limit)


@shared_ffn.register_fake
def _(x, w1, s1, w3, s3, w2, s2, swiglu_limit):
    return x.new_empty(x.shape)


def _shared_ffn_context(x, w1, s1, w3, s3, w2, s2, swiglu_limit):
    rows = x.numel() // x.shape[-1]
    return x.device.type, f"rows={rows} dim={x.shape[-1]} inter={w1.shape[0]}", {
        "rows": rows, "dim": x.shape[-1], "inter": w1.shape[0]}


def _shared_ffn_body(x, w1, s1, w3, s3, w2, s2, swiglu_limit, tk):
    K = x.shape[-1]
    xf = tk.act_quant_dequant(x).reshape(-1, K)
    gate = tk.fp8_gemv(xf, w1, s1).float()
    up = tk.fp8_gemv(xf, w3, s3).float()
    if swiglu_limit > 0:
        up = up.clamp(-swiglu_limit, swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    y = tk.fp8_gemv(tk.act_quant_dequant(h), w2, s2)
    return y.reshape(x.shape).to(x.dtype)


@implementation("vxpu::shared_ffn", "gpu-fused", ("cuda",), "rounding",
                "the fp8 GEMVs with the SwiGLU between them as tensor operations for the compiler to fuse (lowering, "
                "VXPU_FUSE; where it measures cheaper than the kernels)",
                applies=lambda plan, lowering=False, rows=0, **_: _fusing(plan, lowering, "shared_ffn")
                and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(), lowerable=True, fused=True)
def _shared_ffn_fused(x, w1, s1, w3, s3, w2, s2, swiglu_limit, tk=triton_ops):
    return _shared_ffn_body(x, w1, s1, w3, s3, w2, s2, swiglu_limit, tk)


@implementation("vxpu::shared_ffn", "gpu-triton", ("cuda",), "rounding",
                "three Triton kernels: the input quantized, the dual GEMV over w1 and w3 with the SwiGLU and the w2 "
                "input's quantization in its epilogue, the w2 GEMV",
                applies=lambda plan, rows=0, dim=0, inter=0, **_: plan["triton"] and rows <= FP8_GEMV_MAX_ROWS
                and dim % 32 == 0 and inter % 32 == 0 and _fp8_native(), lowerable=True)
def _shared_ffn_triton(x, w1, s1, w3, s3, w2, s2, swiglu_limit, tk=triton_ops):
    return tk.shared_ffn(x, w1, s1, w3, s3, w2, s2, swiglu_limit)


@implementation("vxpu::shared_ffn", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's Expert.forward: three fp8 linears (act_quant and fp8_gemm, aten) and the SwiGLU",
                reference=True)
def _shared_ffn_reference(x, w1, s1, w3, s3, w2, s2, swiglu_limit):
    gate = _fp8_linear_reference(x, w1, s1).float()
    up = _fp8_linear_reference(x, w3, s3).float()
    if swiglu_limit > 0:
        up = up.clamp(-swiglu_limit, swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    return _fp8_linear_reference(h, w2, s2)


@custom_op("vxpu::attn_q", mutates_args=())  # pure tensor work: capturable
def attn_q(x: torch.Tensor, wq_a: torch.Tensor, sa: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
           wq_b: torch.Tensor, sb: torch.Tensor, freqs_cis: torch.Tensor, heads: int, head_dim: int, rope_dim: int
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention's query path (the fork's Attention.forward up to the attention): the fp8
    down-projection wq_a [R, K] of x [b, s, K], q_norm (RMSNorm, norm_weight [R]), the fp8
    up-projection wq_b [N, K] to heads x head_dim, and the rotary of each head's last rope_dim
    dims by freqs_cis (complex [s, rope_dim / 2] shared by the rows, or [b, s, rope_dim / 2] per
    row). Returns (qn [b, s, R], the normalized query the indexer reads; q [b, s, N] rotated).
    One region of the decode step: about nine kernels as aten, four as Triton."""
    impl = select("vxpu::attn_q", x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim)
    return impl.fn(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim)


@attn_q.register_fake
def _(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim):
    lead = tuple(x.shape[:-1])
    return x.new_empty(lead + (wq_a.shape[0],)), x.new_empty(lead + (wq_b.shape[0],))


def _attn_q_context(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim):
    rows = x.numel() // x.shape[-1]
    return x.device.type, f"rows={rows} dim={x.shape[-1]} rank={wq_a.shape[0]} heads={heads}x{head_dim}", {
        "rows": rows, "dim": x.shape[-1], "rank": wq_a.shape[0], "head_dim": head_dim, "rope_dim": rope_dim}


def _rms_norm(x, weight, eps):
    """The fork's RMSNorm.forward."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight * xf).to(x.dtype)


def _freqs_rows(freqs_cis, b, rope_dim):
    """The complex frequencies ([s, D] shared by the rows, or [b, s, D] per row) as their real view
    [b * s, D, 2] (cos and sin interleaved): a view when per row, one copy when shared."""
    f = torch.view_as_real(freqs_cis)
    if f.dim() == 3:  # [s, D, 2] shared by the rows
        f = f.unsqueeze(0).expand(b, -1, -1, -1)
    return f.reshape(-1, rope_dim // 2, 2)



@implementation("vxpu::attn_q", "gpu-fused", ("cuda",), "rounding",
                "the fp8 GEMVs with the norm, quantization and rotary between them as tensor operations for the "
                "compiler to fuse (lowering, VXPU_FUSE; where it measures cheaper than the kernels)",
                applies=lambda plan, lowering=False, rows=0, **_: _fusing(plan, lowering, "attn_q")
                and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(), lowerable=True, fused=True)
def _attn_q_fused(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim, tk=triton_ops):
    K = x.shape[-1]
    lead = tuple(x.shape[:-1])
    qr = tk.fp8_gemv(tk.act_quant_dequant(x).reshape(-1, K), wq_a, sa).reshape(lead + (wq_a.shape[0],))
    qn = _rms_norm(qr, norm_weight, norm_eps)
    q = tk.fp8_gemv(tk.act_quant_dequant(qn).reshape(-1, qn.shape[-1]), wq_b, sb).reshape(lead + (heads, head_dim))
    tail = q[..., head_dim - rope_dim:]
    rotated = tk.rotary(tail, freqs_cis.real, freqs_cis.imag)
    q = torch.cat([q[..., :head_dim - rope_dim], rotated.to(q.dtype)], dim=-1)
    return qn, q.flatten(-2)


@implementation("vxpu::attn_q", "gpu-triton", ("cuda",), "rounding",
                "four Triton kernels: the input quantized, the down-projection, one program per row for the norm "
                "and quantization, the up-projection with the rotary in its epilogue",
                applies=lambda plan, rows=0, dim=0, rank=0, head_dim=0, rope_dim=0, **_: plan["triton"]
                and rows <= FP8_GEMV_MAX_ROWS and dim % 32 == 0 and rank % 256 == 0 and head_dim % 32 == 0
                and (head_dim - rope_dim) % 32 == 0 and rope_dim % 2 == 0 and _fp8_native(), lowerable=True)
def _attn_q_triton(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim, tk=triton_ops):
    return tk.attn_q(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, _freqs_rows(freqs_cis, x.shape[0], rope_dim),
                     heads, head_dim, rope_dim)


@implementation("vxpu::attn_q", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's: two fp8 linears (act_quant and fp8_gemm, aten), RMSNorm and apply_rotary_emb",
                reference=True)
def _attn_q_reference(x, wq_a, sa, norm_weight, norm_eps, wq_b, sb, freqs_cis, heads, head_dim, rope_dim):
    qn = _rms_norm(_fp8_linear_reference(x, wq_a, sa), norm_weight, norm_eps)
    q = _fp8_linear_reference(qn, wq_b, sb).unflatten(-1, (heads, head_dim))
    tail = _rotary_reference(q[..., head_dim - rope_dim:], freqs_cis)
    q = torch.cat([q[..., :head_dim - rope_dim], tail.to(q.dtype)], dim=-1)
    return qn, q.flatten(-2)


@custom_op("vxpu::window_kv", mutates_args=())  # pure tensor work: capturable
def window_kv(x: torch.Tensor, wkv: torch.Tensor, scale: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
              freqs_cis: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The attention's window key (the fork's _window_kv up to the ring write): the fp8 projection
    wkv [D, K] of x [b, s, K], kv_norm (RMSNorm, norm_weight [D]), the rotary of the last rope_dim
    elements by freqs_cis (complex [s, rope_dim / 2] shared by the rows, or [b, s, rope_dim / 2]
    per row) and act_quant to the fp8 grid the cache holds. Returns [b, s, D] bf16. About six
    kernels as aten, three as Triton."""
    impl = select("vxpu::window_kv", x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim)
    return impl.fn(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim)


@window_kv.register_fake
def _(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim):
    return x.new_empty(tuple(x.shape[:-1]) + (wkv.shape[0],))


def _window_kv_context(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim):
    rows = x.numel() // x.shape[-1]
    return x.device.type, f"rows={rows} dim={x.shape[-1]} kv={wkv.shape[0]}", {
        "rows": rows, "dim": x.shape[-1], "kv_dim": wkv.shape[0], "rope_dim": rope_dim}


def _window_kv_body(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim, tk, act_quant_dequant):
    K = x.shape[-1]
    D = wkv.shape[0]
    kv = tk.fp8_gemv(tk.act_quant_dequant(x).reshape(-1, K), wkv, scale).reshape(tuple(x.shape[:-1]) + (D,))
    kv = _rms_norm(kv, norm_weight, norm_eps)
    tail = tk.rotary(kv[..., D - rope_dim:], freqs_cis.real, freqs_cis.imag).to(kv.dtype)
    kv = torch.cat([kv[..., :D - rope_dim], tail], dim=-1)
    return act_quant_dequant(kv)


@implementation("vxpu::window_kv", "gpu-fused", ("cuda",), "rounding",
                "the fp8 GEMV with the norm, rotary and quantization after it as tensor operations for the compiler "
                "to fuse (lowering, VXPU_FUSE; where it measures cheaper than the kernels)",
                applies=lambda plan, lowering=False, rows=0, **_: _fusing(plan, lowering, "window_kv")
                and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(), lowerable=True, fused=True)
def _window_kv_fused(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim, tk=triton_ops):
    return _window_kv_body(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim, tk, tk.act_quant_dequant)


@implementation("vxpu::window_kv", "gpu-triton", ("cuda",), "rounding",
                "three Triton kernels: the input quantized, the projection, the row finished (norm, rotary, "
                "quantization) by one program per row",
                applies=lambda plan, rows=0, dim=0, kv_dim=0, rope_dim=0, **_: plan["triton"]
                and rows <= FP8_GEMV_MAX_ROWS and dim % 32 == 0 and kv_dim % 32 == 0
                and rope_dim % 2 == 0 and _fp8_native(), lowerable=True)
def _window_kv_triton(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim, tk=triton_ops):
    return tk.window_kv(x, wkv, scale, norm_weight, norm_eps, _freqs_rows(freqs_cis, x.shape[0], rope_dim), rope_dim)


@implementation("vxpu::window_kv", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's: an fp8 linear (act_quant and fp8_gemm, aten), RMSNorm, apply_rotary_emb and act_quant",
                reference=True)
def _window_kv_reference(x, wkv, scale, norm_weight, norm_eps, freqs_cis, rope_dim):
    ds, ka = _fork_kernels()
    D = wkv.shape[0]
    kv = _rms_norm(_fp8_linear_reference(x, wkv, scale), norm_weight, norm_eps)
    tail = _rotary_reference(kv[..., D - rope_dim:], freqs_cis).to(kv.dtype)
    kv = torch.cat([kv[..., :D - rope_dim], tail], dim=-1).contiguous()
    return ka.act_quant(kv, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype, True)


@custom_op("vxpu::compress_post", mutates_args=())  # pure tensor work: capturable
def compress_post(x: torch.Tensor, freqs_cis: torch.Tensor, rope_dim: int, quant_kind: str) -> torch.Tensor:
    """The compressed KV or index key finishing (the fork's _compress_kv and Indexer.forward up to
    the cache write): the rotary of the last rope_dim elements by freqs_cis (complex [s, rope_dim / 2]
    shared by the rows, or [b, s, rope_dim / 2] per row) and fp4 quantization (quant_kind: 'e4m3' with
    e4m3 scales per 16 elements, or 'e8m0' with ue8m0 scales per 32 elements). Returns [b, s, D] bf16."""
    impl = select("vxpu::compress_post", x, freqs_cis, rope_dim, quant_kind)
    return impl.fn(x, freqs_cis, rope_dim, quant_kind)


@compress_post.register_fake
def _(x, freqs_cis, rope_dim, quant_kind):
    return torch.empty_like(x)


def _compress_post_context(x, freqs_cis, rope_dim, quant_kind):
    rows = x.numel() // x.shape[-1]
    return x.device.type, f"rows={rows} dim={x.shape[-1]} quant={quant_kind}", {
        "rows": rows, "dim": x.shape[-1], "rope_dim": rope_dim, "quant_kind": quant_kind}


@implementation("vxpu::compress_post", "gpu-fused", ("cuda",), "rounding",
                "the rotary and fp4 quantization as plain tensor operations for the compiler to fuse",
                applies=lambda plan, lowering=False, rows=0, **_: _fusing(plan, lowering, "compress_post"),
                lowerable=True, fused=True)
def _compress_post_fused(x, freqs_cis, rope_dim, quant_kind, tk=triton_ops):
    ds, ka = _fork_kernels()
    D = x.shape[-1]
    c, s = freqs_cis.real, freqs_cis.imag
    rows = x.size(0) if freqs_cis.dim() == 3 else 1
    shape = (rows, x.size(1), 1, rope_dim // 2) if x.ndim == 4 else (rows, x.size(1), rope_dim // 2)
    c, s = c.reshape(shape), s.reshape(shape)
    tail = x[..., D - rope_dim:].float()
    pairs = tail.unflatten(-1, (-1, 2))
    xe, xo = pairs[..., 0], pairs[..., 1]
    rot_tail = torch.stack((xe * c - xo * s, xe * s + xo * c), dim=-1).flatten(-2).to(x.dtype)
    rotated = torch.cat([x[..., :D - rope_dim], rot_tail], dim=-1)
    if quant_kind in ("e4m3", "16_e4m3", torch.float8_e4m3fn):
        return ka.fp4_quant_dequant(rotated, 16, scale_dtype=torch.float8_e4m3fn)
    return ka.fp4_quant_dequant(rotated, ds.fp4_block_size, scale_dtype=torch.float8_e8m0fnu)


@implementation("vxpu::compress_post", "gpu-triton", ("cuda",), "rounding",
                "one program per row: rotary of the rope tail, then fp4 quantization",
                applies=lambda plan, rows=0, dim=0, rope_dim=0, **_: plan["triton"]
                and rows <= 64 and dim % 32 == 0 and rope_dim % 2 == 0, lowerable=True)
def _compress_post_triton(x, freqs_cis, rope_dim, quant_kind, tk=triton_ops):
    return tk.compress_post(x, _freqs_rows(freqs_cis, x.shape[0], rope_dim), rope_dim, quant_kind)


@implementation("vxpu::compress_post", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's: apply_rotary_emb on the rope tail and fp4_quant_dequant",
                reference=True)
def _compress_post_reference(x, freqs_cis, rope_dim, quant_kind):
    ds, ka = _fork_kernels()
    D = x.shape[-1]
    tail = _rotary_reference(x[..., D - rope_dim:], freqs_cis).to(x.dtype)
    rotated = torch.cat([x[..., :D - rope_dim], tail], dim=-1).contiguous()
    if quant_kind in ("e4m3", "16_e4m3", torch.float8_e4m3fn):
        return ka.fp4_quant_dequant(rotated, 16, scale_dtype=torch.float8_e4m3fn)
    return ka.fp4_quant_dequant(rotated, ds.fp4_block_size, scale_dtype=torch.float8_e8m0fnu)


@custom_op("vxpu::attn_out", mutates_args=())  # pure tensor work: capturable
def attn_out(o: torch.Tensor, freqs_cis: torch.Tensor, wo_a: torch.Tensor, sa: torch.Tensor, groups: int, rank: int,
             wo_b: torch.Tensor, sb: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The attention's output path (the fork's Attention.forward after the attention): the inverse
    rotary of each head's last rope_dim dims of o [b, s, heads, head_dim] by freqs_cis (complex
    [s, rope_dim / 2] shared by the rows, or [b, s, rope_dim / 2] per row), act_quant, the grouped
    fp8 projection wo_a [groups * rank, heads_per_group * head_dim] and the fp8 projection wo_b
    [dim, groups * rank]. Returns [b, s, dim]. About five kernels as aten, four as Triton."""
    impl = select("vxpu::attn_out", o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim)
    return impl.fn(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim)


@attn_out.register_fake
def _(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim):
    return o.new_empty(tuple(o.shape[:-2]) + (wo_b.shape[0],))


def _attn_out_context(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim):
    rows = o.numel() // (o.shape[-1] * o.shape[-2])
    return o.device.type, f"rows={rows} heads={o.shape[-2]}x{o.shape[-1]} groups={groups} rank={rank}", {
        "rows": rows, "heads": o.shape[-2], "head_dim": o.shape[-1], "groups": groups, "rank": rank,
        "rope_dim": rope_dim, "dim": wo_b.shape[0]}


def _attn_out_body(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim, tk, act_quant_dequant, grouped):
    hd = o.shape[-1]
    tail = tk.rotary(o[..., hd - rope_dim:], freqs_cis.real, -freqs_cis.imag).to(o.dtype)
    o = torch.cat([o[..., :hd - rope_dim], tail], dim=-1)
    o = act_quant_dequant(o.flatten(-2).unflatten(-1, (groups, -1)))
    mid = grouped(o, wo_a, sa).flatten(-2)
    return tk.fp8_gemv(tk.act_quant_dequant(mid).reshape(-1, mid.shape[-1]), wo_b, sb).reshape(
        tuple(o.shape[:-2]) + (wo_b.shape[0],)).to(o.dtype)


@implementation("vxpu::attn_out", "gpu-fused", ("cuda",), "rounding",
                "the inverse rotary, quantization and the two fp8 GEMVs as tensor operations for the compiler to "
                "fuse (lowering, VXPU_FUSE; where it measures cheaper than the kernels)",
                applies=lambda plan, lowering=False, rows=0, **_: _fusing(plan, lowering, "attn_out")
                and rows <= FP8_GEMV_MAX_ROWS and _fp8_native(), lowerable=True, fused=True)
def _attn_out_fused(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim, tk=triton_ops):
    return _attn_out_body(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim, tk, tk.act_quant_dequant,
                          lambda o_, w, s_: _grouped_gemv(o_, w, s_, tk=tk))


@implementation("vxpu::attn_out", "gpu-triton", ("cuda",), "rounding",
                "four Triton kernels: one program per row and head for the inverse rotary and the quantization, the "
                "grouped projection, its result quantized, the output projection",
                applies=lambda plan, rows=0, head_dim=0, rope_dim=0, rank=0, heads=0, groups=0, **_: plan["triton"]
                and rows <= FP8_GEMV_MAX_ROWS and head_dim % 32 == 0 and (head_dim - rope_dim) % 2 == 0
                and rank % 32 == 0 and groups and heads % groups == 0 and _fp8_native(), lowerable=True)
def _attn_out_triton(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim, tk=triton_ops):
    return tk.attn_out(o, _freqs_rows(freqs_cis, o.shape[0], rope_dim), wo_a, sa, groups, rank, wo_b, sb,
                       o.shape[-1], rope_dim)


@implementation("vxpu::attn_out", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's: apply_rotary_emb inverse, act_quant, the dequantized grouped einsum and an fp8 linear "
                "(aten)", reference=True)
def _attn_out_reference(o, freqs_cis, wo_a, sa, groups, rank, wo_b, sb, rope_dim):
    ds, ka = _fork_kernels()
    hd = o.shape[-1]
    tail = _rotary_reference(o[..., hd - rope_dim:], freqs_cis, inverse=True).to(o.dtype)
    o = torch.cat([o[..., :hd - rope_dim], tail], dim=-1).flatten(-2).unflatten(-1, (groups, -1)).contiguous()
    o = ka.act_quant(o, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype, True)
    mid = _grouped_reference(o, wo_a, sa).flatten(-2)
    return _fp8_linear_reference(mid, wo_b, sb)


@custom_op("vxpu::ffn_front", mutates_args=())  # pure tensor work: capturable
def ffn_front(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
              hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_weight: torch.Tensor,
              norm_eps: float, gate_weight: torch.Tensor, gate_bias: torch.Tensor, topk: int, score_func: str,
              gate_temp: float, norm_topk_prob: bool, route_scale: float
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A block's FFN front half (the fork's Block.forward from the FFN's hc_mixes to the experts'
    call): the hyper-connection mix of stream [b, s, hc, d] for the FFN (pre, post, comb), the
    stream collapsed by pre_mix and normalized (x_normed [b, s, d]), the experts' quantized input
    (x_in [n, d], n = b * s), and the gate's choice (weights [n, k] fp32, indices [n, k] int64). One region of the decode step: about fifteen small kernels as aten, two as
    Triton."""
    impl = select("vxpu::ffn_front", stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                  norm_weight, norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob,
                  route_scale)
    return impl.fn(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                   norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale)


@ffn_front.register_fake
def _(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps,
      gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale):
    b, s_, hc, d = stream.shape
    f32 = dict(dtype=torch.float32)
    return (stream.new_empty((b, s_, hc), **f32), stream.new_empty((b, s_, hc), **f32),
            stream.new_empty((b, s_, hc, hc), **f32), stream.new_empty((b, s_, d)), stream.new_empty((b * s_, d)),
            stream.new_empty((b * s_, topk), **f32), stream.new_empty((b * s_, topk), dtype=torch.int64))


def _ffn_front_context(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                       norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale):
    rows = stream.shape[0] * stream.shape[1]
    return stream.device.type, f"rows={rows} experts={gate_weight.shape[0]} k={topk}", {
        "rows": rows, "experts": gate_weight.shape[0], "topk": topk, "hc_mult": hc_mult, "dim": stream.shape[-1],
        "score_func": score_func}


@implementation("vxpu::ffn_front", "gpu-fused", ("cuda",), "bit-identical",
                "the fork's computation as tensor operations for the compiler to fuse (lowering, VXPU_FUSE; where it "
                "measures cheaper than the kernels: a model with few experts)",
                applies=lambda plan, lowering=False, **_: _fusing(plan, lowering, "ffn_front"), lowerable=True,
                fused=True)
def _ffn_front_fused(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                     norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale,
                     tk=None):
    return _ffn_front_reference(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                                norm_weight, norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp,
                                norm_topk_prob, route_scale)


@implementation("vxpu::ffn_front", "gpu-triton", ("cuda",), "rounding",
                "three Triton kernels: the mix dots and norm statistics per chunk of the row, the normalized and "
                "quantized row and the gate's scores per group of experts, then the Sinkhorn iterations and the "
                "top-k choice per row (aten launches about fifteen)",
                applies=lambda plan, rows=0, experts=0, topk=0, hc_mult=0, dim=0, **_: plan["triton"] and rows <= 64
                and experts <= 1024 and topk <= 16 and hc_mult in (2, 4, 8) and dim % 32 == 0, lowerable=True)
def _ffn_front_triton(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                      norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale,
                      tk=triton_ops):
    return tk.ffn_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                        norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale)


@implementation("vxpu::ffn_front", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's computation (aten): hc_mixes, hc_pre, RMSNorm, Gate.forward and the experts' act_quant",
                reference=True)
def _ffn_front_reference(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                         norm_eps, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale):
    ds, ka = _fork_kernels()
    F = torch.nn.functional
    xf = stream.flatten(2).float()  # hc_mixes
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = _sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    x = torch.sum(pre_mix.unsqueeze(-1) * stream.float(), dim=2).to(stream.dtype)  # hc_pre
    xn = x.float()  # RMSNorm
    xn = xn * torch.rsqrt(xn.square().mean(-1, keepdim=True) + norm_eps)
    xn = (norm_weight * xn).to(stream.dtype)
    x2 = xn.view(-1, xn.shape[-1])
    scores = F.linear(x2.float(), gate_weight.float()) / gate_temp  # Gate.forward
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    indices = (scores + gate_bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if norm_topk_prob and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * route_scale
    q, sc = ka.act_quant(x2, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)  # MoE._quantized_input
    x_in = (q.to(torch.bfloat16).unflatten(-1, (-1, ds.fp8_block_size)) * sc.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)
    return pre, post, comb, xn, x_in, weights, indices


@custom_op("vxpu::sparse_attn", mutates_args=())  # pure tensor work: capturable
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float) -> torch.Tensor:
    """The fork's sparse attention: each query (q [b, m, h, d]) over the KV rows its topk_idxs
    [b, m, t] select from kv [b, n, d] (-1 for an absent slot), softmax with a sink term per head
    that only adds to the denominator. One kernel on a GPU; the fork's aten version (a gather,
    two products and a softmax, about ten kernels) elsewhere."""
    impl = select("vxpu::sparse_attn", q, kv, attn_sink, topk_idxs, softmax_scale)
    return impl.fn(q, kv, attn_sink, topk_idxs, softmax_scale)


@sparse_attn.register_fake
def _(q, kv, attn_sink, topk_idxs, softmax_scale):
    return q.new_empty(q.shape)


def _sparse_attn_context(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    return q.device.type, f"queries={b * m} heads={h} d={d} t={topk_idxs.shape[-1]}", {
        "queries": b * m, "heads": h, "d": d, "t": topk_idxs.shape[-1]}


@implementation("vxpu::sparse_attn", "gpu-triton", ("cuda",), "rounding",
                "Triton: a program per query, group of 16 heads and 32 selected rows gathers the rows and forms the "
                "scores and the softmax partials on the tensor cores (the probabilities as a bf16 pair); a second "
                "kernel combines the partials with the sink term (aten launches about ten)",
                applies=lambda plan, d=0, heads=0, **_: plan["triton"] and d in (64, 128, 256, 512) and heads % 16 == 0,
                lowerable=True)
def _sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale, tk=triton_ops):
    return tk.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


@implementation("vxpu::sparse_attn", "reference", ("cpu", "cuda"), "bit-identical", "the fork's aten sparse_attn",
                reference=True)
def _sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale):
    ds, ka = _fork_kernels()
    return ka.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


@custom_op("vxpu::hc_front", mutates_args=())  # pure tensor work: capturable
def hc_front(stream: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
             hc_base: torch.Tensor, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_weight: torch.Tensor,
             norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A block's attention front: the hyper-connection mix of stream [b, s, hc, d] for the
    attention (pre, post, comb), and the stream collapsed by pre_mix and normalized (x_normed
    [b, s, d]). ffn_front without the gate: the same kernels."""
    impl = select("vxpu::hc_front", stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                  norm_weight, norm_eps)
    return impl.fn(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps)


@hc_front.register_fake
def _(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight, norm_eps):
    b, s_, hc, d = stream.shape
    f32 = dict(dtype=torch.float32)
    return (stream.new_empty((b, s_, hc), **f32), stream.new_empty((b, s_, hc), **f32),
            stream.new_empty((b, s_, hc, hc), **f32), stream.new_empty((b, s_, d)))


def _hc_front_context(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                      norm_eps):
    rows = stream.shape[0] * stream.shape[1]
    return stream.device.type, f"rows={rows} hc={hc_mult}", {"rows": rows, "hc_mult": hc_mult, "dim": stream.shape[-1]}


@implementation("vxpu::hc_front", "gpu-fused", ("cuda",), "bit-identical",
                "the fork's computation as tensor operations for the compiler to fuse (lowering, VXPU_FUSE; where it "
                "measures cheaper than the kernels)",
                applies=lambda plan, lowering=False, **_: _fusing(plan, lowering, "hc_front"), lowerable=True,
                fused=True)
def _hc_front_fused(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                    norm_eps, tk=None):
    return _hc_front_reference(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
                               norm_weight, norm_eps)


@implementation("vxpu::hc_front", "gpu-triton", ("cuda",), "rounding",
                "ffn_front's two Triton kernels without the gate: the mix dots, the stream's collapse and the norm, "
                "then the Sinkhorn iterations (aten launches about six)",
                applies=lambda plan, rows=0, hc_mult=0, dim=0, **_: plan["triton"] and rows <= 64
                and hc_mult in (2, 4, 8) and dim % 32 == 0, lowerable=True)
def _hc_front_triton(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                     norm_eps, tk=triton_ops):
    return tk.hc_front(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                       norm_eps)


@implementation("vxpu::hc_front", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's computation (aten): hc_mixes, hc_pre and RMSNorm", reference=True)
def _hc_front_reference(stream, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_weight,
                        norm_eps):
    F = torch.nn.functional
    xf = stream.flatten(2).float()  # hc_mixes
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = _sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    x = torch.sum(pre_mix.unsqueeze(-1) * stream.float(), dim=2).to(stream.dtype)  # hc_pre
    xn = x.float()  # RMSNorm
    xn = xn * torch.rsqrt(xn.square().mean(-1, keepdim=True) + norm_eps)
    return pre, post, comb, (norm_weight * xn).to(stream.dtype)


@custom_op("vxpu::hc_sinkhorn", mutates_args=())
def hc_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int,
                sinkhorn_iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fork's hyper-connection mixing (hc_split_sinkhorn): pre, post and
    the doubly stochastic comb from the mixes. One Triton kernel on a GPU;
    the fork's own function everywhere else (so the CPU is bit-identical)."""
    impl = select("vxpu::hc_sinkhorn", mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    return impl.fn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


def _sinkhorn_context(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    return mixes.device.type, f"tokens={mixes.numel() // mixes.shape[-1]} hc={hc_mult}", {"hc_mult": hc_mult}


@implementation("vxpu::hc_sinkhorn", "gpu-fused", ("cuda",), "rounding",
                "decomposed for the compiler (lowering, VXPU_FUSE; where it measures cheaper): the fork's Sinkhorn "
                "iterations as tensor operations, fused with their neighbours",
                applies=lambda plan, lowering=False, **_: _fusing(plan, lowering, "hc_sinkhorn"),
                lowerable=True, fused=True)
def _sinkhorn_fused(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, tk=None):
    return _sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


@implementation("vxpu::hc_sinkhorn", "gpu-triton", ("cuda",), "rounding",
                "Triton, power-of-two width: one kernel per call, the matrix in registers",
                applies=lambda plan, hc_mult=0, **_: plan["triton"] and triton_ops.hc_split_sinkhorn_supported(hc_mult),
                lowerable=True)
def _sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, tk=triton_ops):
    return tk.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


@implementation("vxpu::hc_sinkhorn", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's hc_split_sinkhorn (aten)", reference=True)
def _sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    import kernel_aten as ka
    pre, post, comb = ka.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    return pre.contiguous(), post.contiguous(), comb.contiguous()  # outputs must not alias the input


@hc_sinkhorn.register_fake
def _(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    lead = tuple(mixes.shape[:-1])
    new = mixes.new_empty
    return (new(lead + (hc_mult,), dtype=torch.float32), new(lead + (hc_mult,), dtype=torch.float32),
            new(lead + (hc_mult, hc_mult), dtype=torch.float32))


@fp8_linear.register_fake
def _(x, weight, scale):
    return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],))

@custom_op("vxpu::head_linear", mutates_args=())  # pure tensor work: capturable
def head_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The output head: x [..., K] times weight [N, K] transposed, in fp32
    (the fork's F.linear(x.float(), weight) with its fp32 weight), for a
    weight held in the checkpoint's bf16. Widening bf16 to fp32 is exact,
    so the products are the reference's; on a GPU one Triton kernel reads
    the bf16 bytes and accumulates in fp32 (a different summation order:
    equal to fp32 rounding), elsewhere torch's fp32 linear over an fp32
    copy of the weight, widened once and kept by the engine, as it held
    the fork's fp32 weight before (bit-identical)."""
    if x.is_meta:
        return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],), dtype=torch.float32)
    return _call_head(select("vxpu::head_linear", x, weight).fn, x, weight)


def _call_head(fn, x, weight):
    """head_linear's implementations take the rows as one [rows, K] matrix."""
    return fn(x.reshape(-1, x.shape[-1]), weight).reshape(tuple(x.shape[:-1]) + (weight.shape[0],))


def _head_context(x, weight):
    rows = x.numel() // x.shape[-1]
    return (x.device.type, f"{rows}x{x.shape[-1]}->{weight.shape[0]} {str(weight.dtype)[6:]}",
            {"rows": rows, "dtype": weight.dtype})


@implementation("vxpu::head_linear", "gpu-triton-bf16", ("cuda",), "rounding",
                f"Triton, bf16 weight, at most {triton_ops.GEMV_ROWS_MAX} rows: reads the bf16 weight once for all "
                "rows, accumulates in fp32",
                applies=lambda plan, rows=0, dtype=None, **_: plan["triton"] and dtype == torch.bfloat16
                and rows <= triton_ops.GEMV_ROWS_MAX,
                lowerable=True)
def _head_triton(x2, weight, tk=triton_ops):
    return tk.linear_f32(x2, weight)


@implementation("vxpu::head_linear", "f32-copy", ("cpu", "cuda"), "bit-identical",
                "torch's fp32 linear over an fp32 copy of the weight, widened once and kept (the fork's computation)",
                reference=True)
def _head_f32(x2, weight):
    return torch.nn.functional.linear(x2.float(), _f32_weight(weight))


def _f32_weight(weight):
    """weight widened to fp32 once, kept in the active engine's External
    (keyed by storage address, like the bf16 copies); without an engine
    bound, widened for this call only."""
    if weight.dtype == torch.float32:
        return weight
    ext = getattr(_active, "external", None)
    key = (weight.untyped_storage().data_ptr(), weight.storage_offset(), tuple(weight.shape))
    if ext is not None and key in ext.f32:
        return ext.f32[key]
    w = weight.float()
    if ext is not None:
        ext.f32[key] = w
    return w


@head_linear.register_fake
def _(x, weight):
    return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],), dtype=torch.float32)


# ---------------------------------------------------------------------------
# rotary: rotary embedding for interleaved complex pairs

@custom_op("vxpu::rotary", mutates_args=())
def rotary(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs in x by the angles encoded in freqs_cis
    (complex64). Accepts [b, s, d] and [b, s, h, d]; inverse conjugates the
    rotation. The reference implementation uses the fork's complex arithmetic;
    on CUDA the gpu-triton implementation rotates interleaved pairs in
    registers via a Triton kernel, lowered through vxpu_triton::rotary."""
    if x.is_meta:
        return torch.empty_like(x)
    return select("vxpu::rotary", x, freqs_cis, inverse).fn(x, freqs_cis, inverse)


def _rotary_context(x, freqs_cis, inverse=False):
    shape = "x".join(str(d) for d in x.shape)
    inv = " inv" if inverse else ""
    return x.device.type, f"{shape}{inv}", {}


@implementation("vxpu::rotary", "gpu-fused", ("cuda",), "rounding",
                "decomposed for the compiler (lowering, VXPU_FUSE; where it measures cheaper): each interleaved pair "
                "rotated in real arithmetic (a compiler cannot fuse complex numbers), fused with the projection before it",
                applies=lambda plan, lowering=False, **_: _fusing(plan, lowering, "rotary"),
                lowerable=True, fused=True)
def _rotary_fused(x, freqs_cis, inverse=False, tk=None):
    c, s = freqs_cis.real, freqs_cis.imag
    if inverse:
        s = -s
    rows = x.size(0) if freqs_cis.dim() == 3 else 1
    shape = (rows, x.size(1), 1, x.size(-1) // 2) if x.ndim == 4 else (rows, x.size(1), x.size(-1) // 2)
    c, s = c.reshape(shape), s.reshape(shape)
    pairs = x.float().unflatten(-1, (-1, 2))
    xe, xo = pairs[..., 0], pairs[..., 1]
    return torch.stack((xe * c - xo * s, xe * s + xo * c), dim=-1).flatten(-2).to(x.dtype)


@implementation("vxpu::rotary", "gpu-triton", ("cuda",), "rounding",
                "Triton kernel: interleaved pairs rotated in registers",
                applies=lambda plan, **_: plan["triton"], lowerable=True)
def _rotary_triton(x, freqs_cis, inverse=False, tk=triton_ops):
    c = freqs_cis.real
    s = -freqs_cis.imag if inverse else freqs_cis.imag
    return tk.rotary(x, c, s)


@implementation("vxpu::rotary", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's complex arithmetic", reference=True)
def _rotary_reference(x, freqs_cis, inverse=False):
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    # [s, d/2] frequencies are shared by every row; [b, s, d/2] are per row (decode steps whose
    # rows sit at different positions)
    rows = x_c.size(0) if freqs_cis.dim() == 3 else 1
    if x_c.ndim == 3:
        freqs_cis = freqs_cis.view(rows, x_c.size(1), x_c.size(-1))
    else:
        freqs_cis = freqs_cis.view(rows, x_c.size(1), 1, x_c.size(-1))
    return torch.view_as_real(x_c * freqs_cis).flatten(-2).to(x.dtype)


@rotary.register_fake
def _(x, freqs_cis, inverse=False):
    return torch.empty_like(x)


# per op: its selection context from its arguments (device type, the key
# recorded with the selection, what the implementations' applies() read);
# and, where it is not fn(*args), how the op calls the implementation
SELECTION = {
    "vxpu::fp8_linear": _fp8_linear_context,
    "vxpu::fp8_grouped_linear": _grouped_context,
    "vxpu::hc_sinkhorn": _sinkhorn_context,
    "vxpu::ffn_front": _ffn_front_context,
    "vxpu::sparse_attn": _sparse_attn_context,
    "vxpu::hc_front": _hc_front_context,
    "vxpu::head_linear": _head_context,
    "vxpu::rotary": _rotary_context,
    "vxpu::shared_ffn": _shared_ffn_context,
    "vxpu::attn_q": _attn_q_context,
    "vxpu::window_kv": _window_kv_context,
    "vxpu::compress_post": _compress_post_context,
    "vxpu::attn_out": _attn_out_context,
}
CALL = {"vxpu::head_linear": _call_head}


def select(op, *args, plan=None, **extra):
    """The implementation of op the plan selects for these arguments:
    tensors when the op runs, fake tensors (shapes only) when the plan's
    lowering rewrites a graph (which passes lowering=True: a decomposed
    body only makes sense where a compiler fuses it). plan defaults to the
    bound engine's."""
    device_type, key, ctx = SELECTION[op](*args)
    return (plan or _plan(device_type)).select(op, device_type, key=key, **ctx, **extra)


def call(op, fn, *args):
    """op's result from implementation fn, as the op itself calls it."""
    return CALL[op](fn, *args) if op in CALL else fn(*args)


def prepare_compile():
    """Before torch.compile of a graph that calls these ops: the two that do
    host work (the experts on the CPU, the n-gram rows from a file) must
    break the compiled graph rather than be captured into a CUDA graph,
    which would replay without them. Inductor's partitioning did not keep
    them out on its own (the tag notwithstanding), so dynamo is told to
    leave them in eager: ~40 breaks per decode step, each segment its own
    graph. fp8_linear is pure tensor work and stays inside."""
    import torch._dynamo
    torch._dynamo.disallow_in_graph([torch.ops.vxpu.moe_experts.default, torch.ops.vxpu.engram_rows.default])
