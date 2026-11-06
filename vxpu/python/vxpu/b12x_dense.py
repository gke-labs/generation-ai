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

"""Their dense linear behind our fp8 linear op (hero run 15, the last day's order: their kernels in
our step). The vLLM fork's block-FP8 linear (flashinfer's experimental b12x package,
gemm.block_fp8_linear: the activation quantized to MXFP8 in the launch, the checkpoint's fp8 weight
with its 32x32 e8m0 scales expanded once into the SM120 MMA layout, one CuTe DSL GEMM) runs in
place of triton_ops.fp8_gemv at the plain vxpu::fp8_linear sites, as the plan option
VXPU_DENSE_KERNEL=b12x. Measured on pod A (Nov 6, their autotuned plan against gemv_rows at the
decode step's TP4 shapes, the weight flushed from L2 each call): 5-15% less a call, both flat from
1 to 16 rows, about 100 against 115 us over the eight dense shapes.

Each weight is packed once, at its first eager call (the engine's warm-up runs one eager step before
any CUDA graph is captured), and each (shape, row capacity) gets their autotuned plan once: the
autotune races their tile configurations in this process (7-14 s a shape, cached under
VXPU_B12X_CACHE across runs); their default configuration without the autotune measured slower than
our kernel. A first call under capture is an error: the shape must be declared by the eager step.
VXPU_B12X_PATH names the directories holding the b12x package and the CuTe DSL (colon-separated)."""
import bisect
import os
import sys
import time

import torch

DENSE_KERNEL = os.environ.get("VXPU_DENSE_KERNEL", "vxpu")
B12X_PATH = os.environ.get("VXPU_B12X_PATH", "")
CACHE_DIR = os.environ.get("VXPU_B12X_CACHE") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "b12x-prep")
AUTOTUNE = os.environ.get("VXPU_B12X_AUTOTUNE", "1") == "1"
CAPACITIES = (1, 2, 4, 8, 16, 32, 64)  # row capacities a plan is prepared for (their server plans powers of two)
MAX_ROWS = CAPACITIES[-1]
_mods = None
_sessions = {}  # device ordinal -> PreparationSession
_plans = {}  # (ordinal, K, N, capacity) -> prepared execution state
_packed = {}  # (weight data_ptr, N, K) -> their packed weight
seconds = {"pack": 0.0, "plan": 0.0}  # load time spent in their layout and autotune (reported by the engine)


def on():
    """Whether the plan routes the plain fp8 linear sites to their kernel."""
    return DENSE_KERNEL == "b12x"


def _modules():
    global _mods
    if _mods is None:
        for p in reversed([p for p in B12X_PATH.split(":") if p]):
            if p not in sys.path:
                sys.path.insert(0, p)
        from b12x.gemm import block_fp8_linear as bfl
        from b12x.preparation import PreparationSession, PreparedCall, detect_device
        from b12x.preparation.types import require_prepared
        _mods = (bfl, PreparationSession, PreparedCall, detect_device, require_prepared)
    return _mods


def _session(device):
    bfl, PreparationSession, _, detect_device, _ = _modules()
    key = device.index
    if key not in _sessions:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _sessions[key] = PreparationSession(device=detect_device(device), autotune=AUTOTUNE, cache_dir=CACHE_DIR,
                                            compile_workers=0)
    return _sessions[key]


def _plan(device, K, N, capacity):
    key = (device.index, K, N, capacity)
    state = _plans.get(key)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"b12x dense: the shape {capacity}x{K}->{N} was not prepared before CUDA graph capture "
                               "(the eager warm-up step must see every fp8 linear shape)")
        bfl, _, PreparedCall, _, require_prepared = _modules()
        started = time.perf_counter()
        plan = bfl.plan(bfl.Caps(device=device, max_tokens=capacity, in_features=K, out_features=N, block_size=(32, 32),
                                 output_mode="functional"))
        if AUTOTUNE:
            # a weight of the shape for the race (random values: the time does not depend on them)
            w = torch.zeros(N, K, dtype=torch.float8_e4m3fn, device=device)
            s = torch.ones(N // 32, K // 32, dtype=torch.float8_e8m0fnu, device=device)
            packed = bfl.pack_weight(w, s, block_size=(32, 32))

            def make_call(st, packed=packed):
                src = torch.empty((st.query.max_tokens, st.query.in_features), device=device, dtype=torch.bfloat16)
                return PreparedCall(run=lambda: st.run(src, packed), produce=lambda: src.normal_(std=0.25), owners=(packed,))
            _session(device).prepare((plan.request(name=f"vxpu.fp8_linear.{N}x{K}.m{capacity}", prepare_call=make_call,
                                                   benchmark_call=make_call),))
            del packed, w, s
        state = require_prepared(plan, "gemm.block_fp8_linear")
        _plans[key] = state
        seconds["plan"] += time.perf_counter() - started
        print(f"[vxpu] b12x dense: plan {capacity}x{K}->{N} on {device}: {state.config.tile_m}x{state.config.tile_n}x"
              f"{state.config.tile_k} split-K {state.config.split_k_slices} ({time.perf_counter() - started:.1f}s)", flush=True)
    return state


def _pack(weight, scale):
    key = (weight.data_ptr(), weight.shape[0], weight.shape[1])
    packed = _packed.get(key)
    if packed is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("b12x dense: a weight was first seen under CUDA graph capture (the eager warm-up step must see it)")
        bfl = _modules()[0]
        started = time.perf_counter()
        s = scale if scale.dtype == torch.float8_e8m0fnu else scale.view(torch.float8_e8m0fnu)
        packed = bfl.pack_weight(weight, s, block_size=(32, 32))
        _packed[key] = packed
        seconds["pack"] += time.perf_counter() - started
    return packed


def run(x, weight, scale):
    """vxpu::fp8_linear's contract through their kernel: x [..., K] bf16, weight [N, K] fp8 e4m3 with
    [N/32, K/32] e8m0 scales -> [..., N] bf16."""
    N, K = weight.shape
    x2 = x.reshape(-1, K)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    rows = x2.shape[0]
    capacity = CAPACITIES[bisect.bisect_left(CAPACITIES, rows)]
    state = _plan(x2.device, K, N, capacity)
    y = state.run(x2, _pack(weight, scale))
    return y.reshape(tuple(x.shape[:-1]) + (N,)).to(x.dtype)
