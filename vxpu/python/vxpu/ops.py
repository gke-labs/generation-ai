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
import os
import threading

import torch
from torch.library import custom_op

from . import experts_cpu, kernels, triton_ops


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
        # fp8 dense weights dequantized to bf16 on first use on a GPU, keyed
        # by the weight's storage address (see fp8_linear): engine-owned so
        # the copies go with the engine, and shared by its two graphs
        self.bf16 = {}
        # bf16 weights widened to fp32 once where no kernel reads bf16
        # directly (the output head off the Triton path, see head_linear)
        self.f32 = {}
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
    moe = _current("experts", layer)
    host = moe.w1.device
    # decode-sized inputs cross through the engine's pinned buffers; a
    # prefill keeps plain copies (its crossing is noise against its compute,
    # and pinned buffers sized for the longest prompt would stay pinned)
    crossing = x_in.device.type == "cuda" and host.type == "cpu" and x_in.shape[0] <= PINNED_MAX_ROWS
    if crossing:
        x, w, i = _to_host(x_in, weights, indices)
    else:
        x, w, i = x_in.to(host), weights.to(host), indices.to(host)
    if experts_cpu.applicable(moe):
        # the fused fp4 kernels (csrc/): same routings, no dequantized copy
        y = experts_cpu.forward_batched(moe, x, w, i) if x.shape[0] == 1 else experts_cpu.forward_grouped(moe, x, w, i)
    elif x.shape[0] == 1:
        y = moe._forward_batched(x, w, i)
    else:
        y = moe._forward_grouped(x, w, i)
    return _to_device(y, x_in.device) if crossing else y.to(x_in.device)


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
    embed = _current("engram", table)
    host = embed.weight.device
    return embed.lookup(indices.to(host)).to(indices.device)


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
    k = kernels.load() if x.device.type == "cpu" else None
    if k is not None:
        # the kernel quantizes the activation itself (the fork's act_quant, fused)
        K = x.shape[-1]
        y = k.fp8_linear(x.reshape(-1, K).contiguous().to(torch.bfloat16), weight, scale)
        return y.reshape(tuple(x.shape[:-1]) + (weight.shape[0],)).to(x.dtype)
    from .deepseek_v41 import _fork
    ds = _fork()
    import kernel_aten as ka
    if x.is_cuda:
        # The reference expands the fp8 weight to fp32 on every call, about
        # 15x the cost of the matmul on a GPU. The quantized activation is
        # exact in bf16, and one Triton kernel computes it (the aten version
        # is a dozen kernels). A decode step's few rows then go through the
        # fp8 GEMV, which reads the fp8 weight and its scales directly;
        # longer inputs use a bf16 copy of the weight (fp8 values times a
        # power-of-two block scale are exact in bf16), dequantized once and
        # kept, and torch's GEMM.
        if triton_ops.available():
            xf = triton_ops.act_quant_dequant(x)
            if xf.numel() // xf.shape[-1] <= FP8_GEMV_MAX_ROWS:
                # a decode step: read the fp8 weight itself, half the bytes
                # of the bf16 copy (the copy is then made only if a longer
                # input, a prefill, ever comes)
                y = triton_ops.fp8_gemv(xf.reshape(-1, xf.shape[-1]), weight, scale)
                return y.reshape(tuple(x.shape[:-1]) + (weight.shape[0],)).to(x.dtype)
        else:
            K = x.shape[-1]
            xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
            xf = (xq.to(torch.bfloat16).unflatten(-1, (K // 32, 32)) * xs.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)
        return torch.nn.functional.linear(xf, _bf16_weight(weight, scale)).to(x.dtype)
    xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
    return ka.fp8_gemm(xq, xs, weight, scale, ds.scale_dtype, ds.fp8_block_size).to(x.dtype)


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
    r = weight.shape[0] // g
    if o.is_meta:
        return o.new_empty(tuple(lead) + (g, r))
    rows = o.numel() // (g * d)
    if o.is_cuda and triton_ops.available():
        if rows <= FP8_GEMV_MAX_ROWS:
            y = triton_ops.fp8_gemv(o.reshape(rows, g * d), weight, scale, group_rows=r, out_dtype=o.dtype)
            return y.reshape(tuple(lead) + (g, r))
        return torch.einsum("bsgd,grd->bsgr", o, _bf16_weight(weight, scale).view(g, r, d)).contiguous()
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


# device memory the bf16 copies must leave free for activations, caches,
# CUDA graph pools and the compiler's workspace
BF16_CACHE_HEADROOM_BYTES = int(float(os.environ.get("VXPU_BF16_CACHE_HEADROOM_GB", "4")) * 1e9)


def _may_cache_bf16(ext, w):
    """Whether to keep w (a bf16 copy just made): the copies double the
    dense weights' footprint, which is the difference between fitting a
    card and not. VXPU_BF16_WEIGHT_CACHE=0 never keeps them; otherwise a
    copy is kept while free device memory (the driver's plus torch's
    reserved-but-unused) stays above the headroom. The first refusal is
    reported once; later weights keep the per-call path."""
    if os.environ.get("VXPU_BF16_WEIGHT_CACHE", "1").lower() in ("0", "off", "false"):
        return False
    if not w.is_cuda:
        return True
    free, _ = torch.cuda.mem_get_info(w.device)
    free += torch.cuda.memory_reserved(w.device) - torch.cuda.memory_allocated(w.device)
    if free >= BF16_CACHE_HEADROOM_BYTES:
        return True
    if not getattr(ext, "bf16_full", False):
        ext.bf16_full = True
        held = sum(t.numel() * t.element_size() for t in ext.bf16.values())
        print(f"[vxpu] bf16 weight cache stopped at {held / 1e9:.1f} GB ({len(ext.bf16)} weights): less than "
              f"{BF16_CACHE_HEADROOM_BYTES / 1e9:.1f} GB of device memory would remain; the other fp8 linears "
              f"dequantize per call", flush=True)
    return False


@custom_op("vxpu::hc_sinkhorn", mutates_args=())
def hc_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int,
                sinkhorn_iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fork's hyper-connection mixing (hc_split_sinkhorn): pre, post and
    the doubly stochastic comb from the mixes. One Triton kernel on a GPU;
    the fork's own function everywhere else (so the CPU is bit-identical)."""
    if mixes.is_cuda and triton_ops.available() and triton_ops.hc_split_sinkhorn_supported(hc_mult):
        return triton_ops.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
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
    lead = tuple(x.shape[:-1])
    x2 = x.reshape(-1, x.shape[-1])
    if x.is_meta:
        return x.new_empty(lead + (weight.shape[0],), dtype=torch.float32)
    if x.is_cuda and triton_ops.available() and weight.dtype == torch.bfloat16 and x2.shape[0] <= 8:
        y = triton_ops.linear_f32(x2, weight)
    else:
        y = torch.nn.functional.linear(x2.float(), _f32_weight(weight))
    return y.reshape(lead + (weight.shape[0],))


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
