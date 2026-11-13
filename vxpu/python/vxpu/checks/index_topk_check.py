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

"""Check vxpu::index_topk reference against gpu-triton implementation."""

from collections import Counter
import math
import sys
import torch

from vxpu import costs, ops, triton_ops
from vxpu.ops import _index_scores_reference
from vxpu.stepgraph import _Driver


def verify_row(r, ref_row, tri_row, scores_row, cl_val, topk, offset=0):
    """Verify one query row of indices: ascending order, -1 for unreachable,
    equal -1 count, and multiset equivalence of finite picks allowing for ties."""
    t_list = tri_row.tolist()
    r_list = ref_row.tolist()

    if len(t_list) != topk:
        return f"Row {r}: expected {topk} indices, got {len(t_list)}"
    if len(r_list) != topk:
        return f"Row {r}: reference expected {topk} indices, got {len(r_list)}"

    # Must be sorted ascending among valid picks; -1s must be at the end
    for i in range(topk - 1):
        if t_list[i] == -1 and t_list[i + 1] != -1:
            return f"Row {r}: Triton has -1 before valid index at pos {i}"
        if t_list[i] != -1 and t_list[i + 1] != -1 and t_list[i] > t_list[i + 1]:
            return f"Row {r}: Triton indices not sorted ascending: {t_list[:10]}..."

    # Check bounds on valid picks
    for idx in t_list:
        if idx != -1:
            orig_pos = idx - offset
            if orig_pos < 0 or orig_pos >= cl_val:
                return f"Row {r}: pick {idx} (pos {orig_pos}) >= compress_len {cl_val}"

    # Compare -1s by count
    t_negs = t_list.count(-1)
    r_negs = r_list.count(-1)
    if t_negs != r_negs:
        return f"Row {r}: unreachable count mismatch (triton {t_negs} vs ref {r_negs})"

    t_finite = [x for x in t_list if x != -1]
    r_finite = [x for x in r_list if x != -1]

    # Compare finite picks as multisets
    c_t = Counter(t_finite)
    c_r = Counter(r_finite)
    diff_t = list((c_t - c_r).elements())
    diff_r = list((c_r - c_t).elements())

    if diff_t or diff_r:
        if len(diff_t) != len(diff_r):
            return f"Row {r}: finite pick count mismatch ({len(diff_t)} vs {len(diff_r)})"

        # Check scores of differing elements
        t_scores = [scores_row[idx - offset].item() for idx in diff_t]
        r_scores = [scores_row[idx - offset].item() for idx in diff_r]

        ref_finite = [scores_row[i - offset].item() for i in r_finite]
        threshold = min(ref_finite) if ref_finite else 0.0
        band = 2.0 ** (math.frexp(abs(threshold))[1] - 8) if threshold != 0.0 else 0.0

        for s in t_scores + r_scores:
            if abs(s - threshold) > band:
                return (f"Row {r}: score mismatch on {len(diff_t)} differing picks beyond band {band:.6f} (threshold {threshold:.6f}): "
                        f"triton diff {diff_t} (scores {t_scores}) vs "
                        f"ref diff {diff_r} (scores {r_scores})")
        print(f"  Row {r}: {len(diff_t)} boundary picks swapped within band {band:.6f} (threshold {threshold:.6f})")

    return None


def run_checks(device, row_counts=None, topk=512):
    H = 32
    D = 128
    T = 4096
    if row_counts is None:
        row_counts = [1, 3, 8, 32, 33, 48, 64]

    print(f"Checking index_topk on {device}: H={H}, D={D}, T={T}, topk={topk}")
    all_ok = True

    for rows in row_counts:
        for has_cand in (False, True):
            torch.manual_seed(42 + rows + (100 if has_cand else 0))
            b, s = rows, 1

            q = torch.randn(b, s, H, D, dtype=torch.bfloat16, device=device)
            k = torch.randn(b, T, D, dtype=torch.bfloat16, device=device)
            w = torch.randn(b, s, H, dtype=torch.bfloat16, device=device)

            # compress_lens varying per row, with some rows having cl < topk
            cl = torch.randint(256, T + 1, (b, s, 1), dtype=torch.int32, device=device)
            if rows > 1:
                cl[0, 0, 0] = 300  # < topk to test unreachable -1

            if has_cand:
                cand = torch.zeros(b, s, T, dtype=torch.bool, device=device)
                for bi in range(b):
                    valid_len = min(int(cl[bi, 0, 0].item()), T)
                    # Random blocks of 8 enabled
                    n_blocks = valid_len // 8
                    if n_blocks > 0:
                        block_mask = torch.rand(n_blocks, device=device) > 0.3
                        cand[bi, 0, :n_blocks * 8] = block_mask.repeat_interleave(8)
            else:
                cand = None

            offset = 0

            # Run reference and triton
            ref_out = ops._index_topk_reference(q, k, w, cl, cand, topk, offset)
            tri_out = triton_ops.index_topk(q, k, w, cl, cand, topk, offset)

            # Intermediate scores for tie checking
            scores_ref = _index_scores_reference(q, k, w, cl, cand)

            mismatches = 0
            err_msg = None
            for r in range(rows):
                cl_val = int(cl[r, 0, 0].item())
                err = verify_row(r, ref_out[r, 0], tri_out[r, 0], scores_ref[r, 0], cl_val, topk, offset)
                if err:
                    mismatches += 1
                    if err_msg is None:
                        err_msg = err

            cand_str = "with cand" if has_cand else "no cand"
            if mismatches == 0:
                print(f"rows={rows:2d} {cand_str:10s}: 0 mismatches")
            else:
                print(f"rows={rows:2d} {cand_str:10s}: {mismatches} mismatches! First error: {err_msg}")
                all_ok = False

    return all_ok


def run_cuda_graph_benchmark(device, topk=512):
    if not torch.cuda.is_available() or device.type != "cuda":
        print("CUDA graph benchmark skipped: CUDA unavailable")
        return True

    driver = _Driver()
    H = 32
    D = 128
    T = 4096
    offset = 0

    print("\nCUDA graph kernel counts and timings:")
    for rows in (1, 64):
        torch.manual_seed(1000 + rows)
        b, s = rows, 1
        q = torch.randn(b, s, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(b, T, D, dtype=torch.bfloat16, device=device)
        w = torch.randn(b, s, H, dtype=torch.bfloat16, device=device)
        cl = torch.randint(1000, T + 1, (b, s, 1), dtype=torch.int32, device=device)
        cand = None

        # Capture and time reference
        ref_out = torch.empty((b, s, topk), dtype=torch.int32, device=device)
        def run_ref():
            out = ops._index_topk_reference(q, k, w, cl, cand, topk, offset)
            ref_out.copy_(out)

        g_ref = costs.capture(run_ref)
        t_ref = costs.replay_us(g_ref)
        nodes_ref = driver.node_counts(g_ref)

        # Capture and time Triton
        tri_out = torch.empty((b, s, topk), dtype=torch.int32, device=device)
        def run_tri():
            out = triton_ops.index_topk(q, k, w, cl, cand, topk, offset)
            tri_out.copy_(out)

        g_tri = costs.capture(run_tri)
        t_tri = costs.replay_us(g_tri)
        nodes_tri = driver.node_counts(g_tri)

        print(f"rows={rows:2d}:")
        print(f"  reference: {nodes_ref.get('kernel', 0)} kernels ({nodes_ref.get('total', 0)} nodes), {t_ref:.1f} us")
        print(f"  triton:    {nodes_tri.get('kernel', 0)} kernels ({nodes_tri.get('total', 0)} nodes), {t_tri:.1f} us")

        if nodes_tri.get("kernel", 0) > 2:
            print(f"ERROR: Triton path used {nodes_tri.get('kernel', 0)} kernels (expected at most 2)")
            return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check vxpu::index_topk")
    parser.add_argument("--rows", type=str, default=None,
                        help="Comma-separated row counts (default: all [1, 3, 8, 32, 33, 48, 64])")
    parser.add_argument("--topk", type=int, default=512,
                        help="Top-K count (default: 512, e.g. 500 for non-power-of-two)")
    parser.add_argument("--all", action="store_true",
                        help="Run all row counts even on CPU")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        row_counts = [1, 3, 8, 32, 33, 48, 64]
    else:
        device = torch.device("cpu")
        row_counts = [1, 3] if not args.all else [1, 3, 8, 32, 33, 48, 64]

    if args.rows:
        row_counts = [int(r.strip()) for r in args.rows.split(",") if r.strip()]

    ok = run_checks(device, row_counts=row_counts, topk=args.topk)
    graph_ok = run_cuda_graph_benchmark(device, topk=args.topk)

    if ok and graph_ok:
        print("\nRESULT ok")
        return 0
    else:
        print("\nRESULT DIFFERENT")
        return 1


if __name__ == "__main__":
    sys.exit(main())
