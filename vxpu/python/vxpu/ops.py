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

import torch
from torch.library import custom_op

from . import experts_cpu, kernels


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
    x, w, i = x_in.to(host), weights.to(host), indices.to(host)
    if experts_cpu.applicable(moe):
        # the fused fp4 kernels (csrc/): same routings, no dequantized copy
        y = experts_cpu.forward_batched(moe, x, w, i) if x.shape[0] == 1 else experts_cpu.forward_grouped(moe, x, w, i)
    elif x.shape[0] == 1:
        y = moe._forward_batched(x, w, i)
    else:
        y = moe._forward_grouped(x, w, i)
    return y.to(x_in.device)


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
    (one token: a GEMV per row; more: decode once and GEMM), elsewhere
    the fork's own fp8_gemm."""
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
    xq, xs = ka.act_quant(x, ds.fp8_block_size, ds.scale_fmt, ds.scale_dtype)
    return ka.fp8_gemm(xq, xs, weight, scale, ds.scale_dtype, ds.fp8_block_size).to(x.dtype)


@fp8_linear.register_fake
def _(x, weight, scale):
    return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],))

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
