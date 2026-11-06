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

"""Their expert kernel behind our expert op (hero run 15, the user's order of Nov 6): the vLLM
fork's W4A8 compact micro kernel (flashinfer's experimental b12x package: the activations
quantized to MXFP8 in the launch, the fp4 experts in its n64 tile layout with e8m0 scales, the
SwiGLU and the down projection per route, then a top-k sum) runs in place of
triton_ops.fp4_experts_decode for the tensor-parallel slice pools, as the plan option
VXPU_EXPERT_KERNEL=b12x. The pools' banks are repacked once at load with their in-place Triton
repack: w13 is built from our w3 and w1 (up rows then gate rows per expert, their order) and those are freed,
w2 is repacked in place; the load time is reported. A measurement of the ceiling with their
kernel, not a shipped form; the gates run and record it as a class (W4A8: MXFP8 activations and
their rounding points). VXPU_B12X_PATH names the directories holding the b12x package and the
CuTe DSL (colon-separated); VXPU_B12X_FAST_MATH=1 is their fast-math arm."""
import os
import sys
import time

import torch

EXPERT_KERNEL = os.environ.get("VXPU_EXPERT_KERNEL", "vxpu")
B12X_PATH = os.environ.get("VXPU_B12X_PATH", "")
FAST_MATH = os.environ.get("VXPU_B12X_FAST_MATH", "0") == "1"
_packs = {}  # the w2 bank's data_ptr -> _Pack (w2 is repacked in place, so its storage names the pool)
_mods = None


def on():
    """Whether the plan routes the expert op to their kernel."""
    return EXPERT_KERNEL == "b12x"


def _modules():
    global _mods
    if _mods is None:
        for p in reversed([p for p in B12X_PATH.split(":") if p]):
            if p not in sys.path:
                sys.path.insert(0, p)
        from b12x.moe.fused_moe import _impl
        from b12x.moe._shared.kernels.w4a8_compact_micro import launch_w4a8_compact_micro, micro_scratch_nbytes
        from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat
        _mods = (_impl, launch_w4a8_compact_micro, micro_scratch_nbytes, _w4a16_topk_sum_launch_flat)
    return _mods


class _Pack:
    """One pool's banks in their layout: w13 [E, 2I, K/2] and w2 [E, K, I/2] repacked (int32 views of
    the same storage), the scales as their sfb tiles, the per-expert alphas (1: the e8m0 scales
    carry everything), the activation scales (1: MXFP8 per 32 in the launch), a scratch per shape."""

    def __init__(self, pool):
        impl = _modules()[0]
        dev = pool.device
        E, I, Kh = pool.w1.shape[0], pool.w1.shape[1], pool.w1.shape[2] * 2
        started = time.perf_counter()
        with torch.cuda.device(dev):
            # [E, 2I, K/2]: up rows then gate rows. Measured on one expert against the dequantized reference: this order
            # reads 2.25% (our kernel's own figure), gate-then-up reads 88%, interleaved rows and swapped nibbles 120%
            w13 = torch.cat([pool.w3, pool.w1], dim=1).contiguous()
            s13 = torch.cat([pool.s3.view(torch.uint8), pool.s1.view(torch.uint8)], dim=1).contiguous()  # [E, 2I, K/32]
            pool.w1 = pool.w3 = pool.s1 = pool.s3 = None  # their layout holds the same bytes
            torch.cuda.empty_cache()
            self.w13_storage, self.s13_storage = w13, s13
            self.w13 = impl._logical_weight_to_w4a8_n64_inplace(w13, size_k=Kh, size_n=2 * I, group_rows=I)
            self.s13 = impl._e8m0_scale_to_w4a8_n64_sfb_inplace(s13, weight_E=E, rows=2 * I, k_dim=Kh, group_rows=I)
            self.w2 = impl._logical_weight_to_w4a8_n64_inplace(pool.w2, size_k=I, size_n=Kh)
            s2 = pool.s2.view(torch.uint8)
            self.s2 = impl._e8m0_scale_to_w4a8_n64_sfb_inplace(s2, weight_E=E, rows=Kh, k_dim=I)
            self.alpha = torch.ones(E, dtype=torch.float32, device=dev)
            self.one = torch.ones(1, dtype=torch.float32, device=dev)
            torch.cuda.synchronize(dev)
        self.E, self.I, self.K, self.device = E, I, Kh, dev
        self.scratches = {}
        self.seconds = time.perf_counter() - started

    def scratch(self, n, topk):
        key = (n, topk)
        if key not in self.scratches:
            nbytes = _modules()[2](n, self.K, self.I, topk)
            self.scratches[key] = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        return self.scratches[key]


def prepare(pool):
    """Repack one pool's banks into their layout (once, at load) and register it by its w2 storage."""
    pack = _Pack(pool)
    _packs[pool.w2.data_ptr()] = pack
    return pack


def prepared(w2):
    return _packs.get(w2.data_ptr())


def run(x_in, weights, slots, indices, start, w2, swiglu_limit, per_pair=False, push=None):
    """fp4_experts_decode's contract through their kernel: x_in [n, K] bf16, weights [n, topk] fp32,
    the pairs' slots (slots [n, topk], or the slot map with indices [n, topk] expert ids from
    `start`), w2 the pool's down bank (names the pack). Returns y [n, K] fp32 (the sum over the
    token's pairs, the routing weights applied) or with per_pair each pair's reply [n, topk, K]
    fp32; with push, the finished y goes into the peers' inboxes and their flags are raised (the
    two-part push sum, as the direct kernel does it)."""
    pack = prepared(w2)
    if pack is None:
        raise RuntimeError("b12x experts: this pool was not prepared (b12x_experts.prepare at load)")
    _impl, launch, _scratch, topk_sum = _modules()
    if indices is not None:
        ids = slots[(indices.reshape(-1) - start).long()].view(indices.shape)
    else:
        ids = slots
    ids = ids.to(torch.int32).contiguous()
    n, topk = ids.shape
    K = x_in.shape[1]
    x = x_in.contiguous()
    w = weights.to(torch.float32).contiguous()
    route = launch(scratch=pack.scratch(n, topk), a=x, topk_ids=ids, topk_weights=w, w13=pack.w13, w13_scales=pack.s13,
                   w2=pack.w2, w2_scales=pack.s2, alpha1=pack.alpha, alpha2=pack.alpha, input_scale=pack.one,
                   down_scale=pack.one, max_tokens=n, num_topk=topk, swiglu_limit=float(swiglu_limit), fast_math=FAST_MATH)
    if per_pair:
        return route.view(n, topk, K).float()
    out = torch.empty(n, K, dtype=torch.bfloat16, device=x.device)
    topk_sum(route, out, n, topk, K, "bf16", int(torch.cuda.current_stream(x.device).cuda_stream))
    y = out.float()
    if push is not None:
        _push(y, push)
    return y


def _push(y, push):
    """The two-part push sum's routed part after their kernel: y into the peers' inboxes, then the
    flags (triton_ops._fp4_push_arrive's protocol: one arrival per block, the last raises)."""
    from . import triton_ops
    import triton
    dests = (list(push["dests"]) + [y] * 3)[:3]
    flags = (list(push["flags"]) + [y] * 3)[:3]
    peers = len(push["flags"])
    flat = y.reshape(-1)
    block = 1024
    nblocks = triton.cdiv(flat.numel(), block)
    _b12x_push_kernel[(nblocks,)](flat, dests[0], dests[1], dests[2], push["counter"], flags[0], flags[1], flags[2],
                                  flat.numel(), nblocks, PEERS_OUT=peers, BLOCK=block)
    push["done"] = True


try:
    import triton
    import triton.language as tl
    from . import triton_ops as _tk

    @triton.jit
    def _b12x_push_kernel(y_ptr, pd0, pd1, pd2, pcnt_ptr, pf0, pf1, pf2, N, NSUM, PEERS_OUT: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = offs < N
        v = tl.load(y_ptr + offs, mask=live, other=0.0)
        _tk._fp4_push_rows(pd0, pd1, pd2, 0, offs, v, live, PEERS_OUT)
        _tk._fp4_push_arrive(pcnt_ptr, pf0, pf1, pf2, NSUM, PEERS_OUT)
except ImportError:  # the CPU-only selftest
    _b12x_push_kernel = None
