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
    crossing = x_in.device.type == "cuda" and host.type == "cpu" and x_in.shape[0] <= PINNED_MAX_ROWS
    rec = _recorder()
    t0 = time.perf_counter() if rec is not None else 0.0
    boxes = getattr(moe, "_vxpu_mailboxes", None)
    if boxes is not None and x_in.shape[0] <= PINNED_MAX_ROWS:
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
                applies=lambda plan, rows=0, mailboxes=False, **ctx: mailboxes and rows <= PINNED_MAX_ROWS
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
                applies=lambda plan, rows=0, hot=False, **ctx: hot and rows <= PINNED_MAX_ROWS and plan["triton"]
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
    x, w, i = _to_host(x_in, weights, indices)  # the GPUs compute the hot pairs meanwhile
    t1 = time.perf_counter()
    cold = ~res.held_cpu[i - moe.experts_start_idx]
    n_cold = int(cold.sum())
    y_cold = None
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
    selected = i.flatten().tolist()
    started = completed = 0
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
# own routings (the reference computation).
@implementation("vxpu::moe_experts", "cpu-fused", ("cpu",), "rounding",
                "decode (one token), kernels on, fused experts on: one C++ call per layer",
                applies=lambda plan, rows=0, **ctx: rows == 1 and plan["fused_experts"] and _cpu_experts(plan, **ctx), host=True)
def _moe_cpu_fused(moe, x, w, i):
    return experts_cpu.forward_batched(moe, x, w, i, fused=True)


@implementation("vxpu::moe_experts", "cpu-three-call", ("cpu",), "rounding",
                "decode (one token), kernels on, fused experts off: three kernel calls with the module's SwiGLU between",
                applies=lambda plan, rows=0, **ctx: rows == 1 and _cpu_experts(plan, **ctx), host=True)
def _moe_cpu_three_call(moe, x, w, i):
    return experts_cpu.forward_batched(moe, x, w, i, fused=False)


@implementation("vxpu::moe_experts", "cpu-grouped", ("cpu",), "rounding",
                "prefill (several tokens), kernels on: each expert decoded once, torch's GEMM over its tokens",
                applies=lambda plan, rows=0, **ctx: rows > 1 and _cpu_experts(plan, **ctx), host=True)
def _moe_cpu_grouped(moe, x, w, i):
    return experts_cpu.forward_grouped(moe, x, w, i)


@implementation("vxpu::moe_experts", "reference", ("cpu", "cuda"), "bit-identical",
                "the module's own routings (batched for one token, grouped for more)", reference=True, host=True)
def _moe_reference(moe, x, w, i):
    return moe._forward_batched(x, w, i) if x.shape[0] == 1 else moe._forward_grouped(x, w, i)


# inputs of up to this many token rows cross through pinned buffers
PINNED_MAX_ROWS = 8


def _pinned(name, like):
    """A pinned host tensor shaped and typed like `like`, kept by the bound
    engine (the op has already required one) and reused call to call;
    regrown when a call needs more, at most PINNED_MAX_ROWS rows."""
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


# On a GPU the reference expands the fp8 weight to fp32 on every call, about
# 15x the cost of the matmul. The quantized activation is exact in bf16, and
# one Triton kernel computes it (the aten version is a dozen kernels). A
# decode step's few rows then go through the fp8 GEMV, which reads the fp8
# weight and its scales directly; longer inputs use a bf16 copy of the
# weight (fp8 values times a power-of-two block scale are exact in bf16),
# dequantized once and kept, and torch's GEMM.
@implementation("vxpu::fp8_linear", "gpu-fp8-gemv", ("cuda",), "rounding",
                f"Triton, at most {4} rows (decode): the activation quantized by one kernel, the fp8 weight read directly",
                applies=lambda plan, rows=0, **_: plan["triton"] and rows <= FP8_GEMV_MAX_ROWS, lowerable=True)
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
                applies=lambda plan, rows=0, **_: plan["triton"] and rows <= FP8_GEMV_MAX_ROWS, lowerable=True)
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


# inputs up to this many rows take the fp8 GEMV on a GPU; longer ones the
# bf16 copy and a GEMM (the GEMV reads the weight once per row)
FP8_GEMV_MAX_ROWS = 4


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
                "Triton, bf16 weight, at most 8 rows: reads the bf16 weight, accumulates in fp32",
                applies=lambda plan, rows=0, dtype=None, **_: plan["triton"] and dtype == torch.bfloat16 and rows <= 8,
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
    if x_c.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_c.size(1), x_c.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    return torch.view_as_real(x_c * freqs_cis).flatten(-2).to(x.dtype)


@rotary.register_fake
def _(x, freqs_cis, inverse=False):
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# index_topk: indexer score reduction and top-k selection

INDEX_TOPK_MAX_ROWS = 64


@custom_op("vxpu::index_topk", mutates_args=())
def index_topk(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
               compress_lens: torch.Tensor, candidates: torch.Tensor | None,
               index_topk: int, offset: int = 0) -> torch.Tensor:
    """Score compressed positions and select the `index_topk` best per query.
    Takes q [b, s, heads, head_dim], index_k [b, T, head_dim], weights [b, s, heads],
    compress_lens, optional candidates mask [b, s, T], index_topk, and offset.
    Returns int32 indices [b, s, index_topk] sorted ascending, with -1 for unreachable picks."""
    if q.is_meta:
        return q.new_empty((q.shape[0], q.shape[1], index_topk), dtype=torch.int32)
    return select("vxpu::index_topk", q, index_k, weights, compress_lens, candidates, index_topk, offset).fn(
        q, index_k, weights, compress_lens, candidates, index_topk, offset)


def _index_topk_context(q, index_k, weights, compress_lens, candidates, index_topk, offset=0):
    b, s, h, d = q.shape
    t = index_k.shape[1]
    rows = b * s
    cand_str = " cand" if candidates is not None else ""
    return q.device.type, f"{rows}x{h}x{d}->T={t} k={index_topk}{cand_str}", {"rows": rows, "t": t, "index_topk": index_topk}


@implementation("vxpu::index_topk", "gpu-triton", ("cuda",), "rounding",
                "Triton kernels: score reduction and top-k selection",
                applies=lambda plan, rows=0, **_: plan["triton"] and rows <= INDEX_TOPK_MAX_ROWS,
                lowerable=True)
def _index_topk_triton(q, index_k, weights, compress_lens, candidates, index_topk, offset=0, tk=triton_ops):
    return tk.index_topk(q, index_k, weights, compress_lens, candidates, index_topk, offset)


@implementation("vxpu::index_topk", "reference", ("cpu", "cuda"), "bit-identical",
                "the fork's einsum, relu, mask and top-k", reference=True)
def _index_topk_reference(q, index_k, weights, compress_lens, candidates, index_topk, offset=0):
    b, s, _, _ = q.shape
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    slots = torch.arange(index_k.size(1), device=q.device)
    if compress_lens.ndim == 3 and compress_lens.shape[-1] == 1:
        cl = compress_lens
    else:
        n = compress_lens.numel()
        if n == b * s:
            cl = compress_lens.view(b, s, 1)
        elif n == s:
            cl = compress_lens.view(1, s, 1)
        elif n == b:
            cl = compress_lens.view(b, 1, 1)
        else:
            cl = compress_lens.view(1, 1, 1)
    index_score = index_score.masked_fill(slots >= cl, -torch.inf)
    if candidates is not None:
        index_score = index_score.masked_fill(~candidates, -torch.inf)
    idxs = index_score.topk(index_topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < cl, idxs + offset, -1).int()


@index_topk.register_fake
def _(q, index_k, weights, compress_lens, candidates, index_topk, offset=0):
    b, s, _, _ = q.shape
    return q.new_empty((b, s, index_topk), dtype=torch.int32)


# per op: its selection context from its arguments (device type, the key
# recorded with the selection, what the implementations' applies() read);
# and, where it is not fn(*args), how the op calls the implementation
SELECTION = {
    "vxpu::fp8_linear": _fp8_linear_context,
    "vxpu::fp8_grouped_linear": _grouped_context,
    "vxpu::hc_sinkhorn": _sinkhorn_context,
    "vxpu::head_linear": _head_context,
    "vxpu::rotary": _rotary_context,
    "vxpu::index_topk": _index_topk_context,
}
CALL = {"vxpu::head_linear": _call_head}


def select(op, *args, plan=None):
    """The implementation of op the plan selects for these arguments:
    tensors when the op runs, fake tensors (shapes only) when the plan's
    lowering rewrites a graph. plan defaults to the bound engine's."""
    device_type, key, ctx = SELECTION[op](*args)
    return (plan or _plan(device_type)).select(op, device_type, key=key, **ctx)


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
