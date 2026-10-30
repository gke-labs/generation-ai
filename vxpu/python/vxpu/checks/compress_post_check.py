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

"""Check vxpu::compress_post: the Triton path against the reference at the released model's
shapes (D = 512, rope dim 64) for 1, 3, 8, 32, 33, 48 and 64 rows, both quantization kinds,
per-row and shared frequencies, then times and kernel counts in a CUDA graph."""

import sys
import torch

from vxpu import costs, ops, stepgraph, triton_ops

D = 512
ROPE_DIM = 64
ROWS = (1, 3, 8, 32, 33, 48, 64)
QUANT_KINDS = ("e4m3", "e8m0")
FREQ_MODES = ("shared", "per-row")


def make_freqs(rows, rope_dim, mode, device):
    if mode == "per-row":
        angles = torch.randn(rows, 1, rope_dim // 2, device=device)
    else:
        angles = torch.randn(1, rope_dim // 2, device=device)
    return torch.polar(torch.ones_like(angles), angles)


def check_correctness(device):
    torch.manual_seed(42)
    print("=" * 80)
    print(f"Correctness check: Triton vs Reference on {device}")
    print(f"Shapes: D = {D}, rope_dim = {ROPE_DIM}")
    print(f"{'rows':>6} | {'quant':>6} | {'freqs':>8} | {'elements':>10} | {'diffs':>8} | result")
    print("-" * 80)
    all_ok = True
    for rows in ROWS:
        for quant_kind in QUANT_KINDS:
            for mode in FREQ_MODES:
                x = torch.randn(rows, 1, D, dtype=torch.bfloat16, device=device)
                freqs_cis = make_freqs(rows, ROPE_DIM, mode, device)
                ref = ops._compress_post_reference(x, freqs_cis, ROPE_DIM, quant_kind)
                freqs_rows = ops._freqs_rows(freqs_cis, rows, ROPE_DIM)
                cand = triton_ops.compress_post(x, freqs_rows, ROPE_DIM, quant_kind)
                diffs = (ref != cand).sum().item()
                elements = x.numel()
                res = "ok" if diffs == 0 else "DIFFERENT"
                if diffs > 0:
                    all_ok = False
                print(f"{rows:>6} | {quant_kind:>6} | {mode:>8} | {elements:>10} | {diffs:>8} | RESULT {res}")
    print("=" * 80)
    return all_ok


def check_costs(device):
    print("\nCUDA graph measurement (times and kernel counts as costs.py measures them):")
    print(f"{'rows':>6} | {'quant':>6} | {'time (us)':>10} | {'kernels':>8} | {'kernel names'}")
    print("-" * 80)
    driver = stepgraph._Driver()
    for rows in ROWS:
        for quant_kind in QUANT_KINDS:
            x = torch.randn(rows, 1, D, dtype=torch.bfloat16, device=device)
            freqs_cis = make_freqs(rows, ROPE_DIM, "per-row", device)
            freqs_rows = ops._freqs_rows(freqs_cis, rows, ROPE_DIM)
            flat, sets = costs._flat(
                lambda x_in, f_in: triton_ops.compress_post(x_in, f_in, ROPE_DIM, quant_kind),
                (x, freqs_rows), device)
            time_us = costs._replay_us(flat, sets)
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g):
                flat(*sets[0])
            names = {}
            try:
                counts = driver.node_counts(g.raw_cuda_graph(), names)
                kernels = counts.get("kernels", 0)
            except RuntimeError:
                kernels = 0
            k_names = ", ".join(f"{k}: {v}" for k, v in names.items())
            print(f"{rows:>6} | {quant_kind:>6} | {time_us:>10.2f} | {kernels:>8} | {k_names}")
    print("=" * 80)


def main():
    if not torch.cuda.is_available() or not triton_ops.available():
        print("CUDA/Triton not available on this machine.")
        print("Checking reference and fused implementations on CPU...")
        torch.manual_seed(42)
        all_ok = True
        for rows in ROWS:
            for quant_kind in QUANT_KINDS:
                for mode in FREQ_MODES:
                    x = torch.randn(rows, 1, D, dtype=torch.bfloat16)
                    freqs_cis = make_freqs(rows, ROPE_DIM, mode, "cpu")
                    ref = ops._compress_post_reference(x, freqs_cis, ROPE_DIM, quant_kind)
                    fused = ops._compress_post_fused(x, freqs_cis, ROPE_DIM, quant_kind)
                    diffs = (ref != fused).sum().item()
                    res = "ok" if diffs == 0 else "DIFFERENT"
                    if diffs > 0:
                        all_ok = False
                    print(f"{rows:>6} | {quant_kind:>6} | {mode:>8} | {x.numel():>10} | {diffs:>8} | RESULT {res}")
        if not all_ok:
            return 1
        print("\nPasses CPU verification. Run on a GPU machine for Triton kernel checks and CUDA graph measurements.")
        return 0

    device = torch.device("cuda:0")
    ok = check_correctness(device)
    if not ok:
        print("FAILED: Mismatches detected between Triton and Reference implementations!")
        return 1
    check_costs(device)
    print("All checks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
