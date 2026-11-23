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

"""Unit tests for vxpu::rotary under the kernel contract."""

# pyright: strict

from typing import Any
import unittest

import torch

from . import ops, plan, triton_ops
from .kernel_tools import check
from .kernels.rotary import inputs


class RotaryKernelTest(unittest.TestCase):
    """Verifies that vxpu::rotary meets the kernel contract for both single and per-row positions."""

    def test_inputs(self) -> None:
        """Verifies synthetic input construction and shape parsing."""
        # Single position shared by all rows
        x, freqs, inv = inputs("1x1x64x64", device="cpu")
        self.assertEqual(x.shape, (1, 1, 64, 64))
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(freqs.shape, (1, 32))
        self.assertEqual(freqs.dtype, torch.complex64)
        self.assertFalse(inv)

        _, _, inv_true = inputs("1x1x64x64 inv", device="cpu")
        self.assertTrue(inv_true)

        # 3D input format
        x3, freqs3, inv3 = inputs("2x8x64", device="cpu")
        self.assertEqual(x3.shape, (2, 8, 64))
        self.assertEqual(freqs3.shape, (8, 32))
        self.assertFalse(inv3)

        # Per-row positions (several rows at different positions)
        x_row, freqs_row, inv_row = inputs("4x1x64x64 per_row", device="cpu")
        self.assertEqual(x_row.shape, (4, 1, 64, 64))
        self.assertEqual(freqs_row.shape, (4, 1, 32))
        self.assertFalse(inv_row)

        x_row_inv, freqs_row_inv, inv_row_true = inputs("4x1x64x64 per_row inv", device="cpu")
        self.assertEqual(x_row_inv.shape, (4, 1, 64, 64))
        self.assertEqual(freqs_row_inv.shape, (4, 1, 32))
        self.assertTrue(inv_row_true)

        # Hyphenated modifier
        _, freqs_hyphen, _ = inputs("4x1x64x64 per-row", device="cpu")
        self.assertEqual(freqs_hyphen.shape, (4, 1, 32))

        # Error cases
        with self.assertRaises(ValueError):
            inputs("", device="cpu")
        with self.assertRaises(ValueError):
            inputs("64x64", device="cpu")
        with self.assertRaises(ValueError):
            inputs("1x2x3x4x64", device="cpu")
        with self.assertRaises(ValueError):
            inputs("1x1x64x63", device="cpu")
        with self.assertRaises(ValueError):
            inputs("1x1x64x64 unknown", device="cpu")

    def test_reference_contract_and_refusals(self) -> None:
        """Verifies reference implementation behavior and shape validation."""
        ref = plan.reference_of("vxpu::rotary")
        self.assertIsNotNone(ref)
        assert ref is not None
        ops_mod: Any = ops

        # Valid 4D input with single-position freqs_cis [s, d/2]
        x4 = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16)
        freqs_single = torch.polar(torch.ones(3, 8), torch.randn(3, 8))
        y_single = ops_mod.call("vxpu::rotary", ref.fn, x4, freqs_single, False)
        self.assertEqual(y_single.shape, x4.shape)
        self.assertEqual(y_single.dtype, x4.dtype)

        # Valid 4D input with per-row freqs_cis [b, s, d/2]
        freqs_row = torch.polar(torch.ones(2, 3, 8), torch.randn(2, 3, 8))
        y_row = ops_mod.call("vxpu::rotary", ref.fn, x4, freqs_row, False)
        self.assertEqual(y_row.shape, x4.shape)
        self.assertEqual(y_row.dtype, x4.dtype)

        # Valid 3D input with per-row freqs_cis [b, s, d/2]
        x3 = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        y3_row = ops_mod.call("vxpu::rotary", ref.fn, x3, freqs_row, False)
        self.assertEqual(y3_row.shape, x3.shape)

        # Refusal of invalid x rank
        with self.assertRaisesRegex(ValueError, r"rank 3 .* or 4"):
            ops_mod.call("vxpu::rotary", ref.fn, torch.randn(2, 16), freqs_single, False)

        # Refusal of odd feature dimension
        with self.assertRaisesRegex(ValueError, r"last dimension .* must be even"):
            ops_mod.call("vxpu::rotary", ref.fn, torch.randn(1, 1, 15), torch.randn(1, 7), False)

        # Refusal of mismatched freqs_cis shape, naming the accepted shapes
        bad_freqs = torch.polar(torch.ones(5, 8), torch.randn(5, 8))
        with self.assertRaisesRegex(ValueError, r"needs freqs_cis \[3, 8\] or \[2, 3, 8\], got freqs_cis \(5, 8\)"):
            ops_mod.call("vxpu::rotary", ref.fn, x4, bad_freqs, False)

    def test_rotary_kernel_check_cpu(self) -> None:
        """Verifies kernel-check on CPU for both single-position and per-row position forms."""
        # Single position for all rows
        res_single = check.run_check("vxpu::rotary", "reference", "1x1x64x64", device="cpu")
        self.assertTrue(res_single.passed, f"rotary kernel-check single position failed: {res_single}")
        self.assertEqual(res_single.deviation, 0.0)
        print(check.line(res_single))

        # Different position per row
        res_row = check.run_check("vxpu::rotary", "reference", "4x1x64x64 per_row", device="cpu")
        self.assertTrue(res_row.passed, f"rotary kernel-check per-row failed: {res_row}")
        self.assertEqual(res_row.deviation, 0.0)
        print(check.line(res_row))

    def test_rotary_kernel_check_gpu(self) -> None:
        """Verifies kernel-check on GPU against reference body for both forms within rounding tolerance."""
        if not torch.cuda.is_available() or not triton_ops.available():
            self.skipTest("needs CUDA and Triton")

        # Form 1: one position for all rows
        res_single = check.run_check("vxpu::rotary", "gpu-triton", "1x1x64x64", device="cuda")
        self.assertTrue(res_single.passed, f"gpu-triton single position failed check: {res_single}")
        self.assertLessEqual(res_single.deviation, check.ROUNDING_TOLERANCE)
        print(check.line(res_single))

        # Form 2: a different position per row (several rows at different positions)
        res_row = check.run_check("vxpu::rotary", "gpu-triton", "4x1x64x64 per_row", device="cuda")
        self.assertTrue(res_row.passed, f"gpu-triton per-row failed check: {res_row}")
        self.assertLessEqual(res_row.deviation, check.ROUNDING_TOLERANCE)
        print(check.line(res_row))

        # Form 2 with inverse rotation
        res_row_inv = check.run_check("vxpu::rotary", "gpu-triton", "4x1x64x64 per_row inv", device="cuda")
        self.assertTrue(res_row_inv.passed, f"gpu-triton per-row inv failed check: {res_row_inv}")
        self.assertLessEqual(res_row_inv.deviation, check.ROUNDING_TOLERANCE)
        print(check.line(res_row_inv))

    def test_gpu_triton_contract_and_refusals(self) -> None:
        """Verifies triton_ops.rotary contract and error reporting naming shapes."""
        if not torch.cuda.is_available() or not triton_ops.available():
            self.skipTest("needs CUDA and Triton")

        dev = "cuda"
        b, s, h, d = 2, 3, 4, 16
        D = d // 2
        x = torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev)

        # Refusal of invalid x rank
        with self.assertRaisesRegex(ValueError, r"rank 3 .* or 4"):
            triton_ops.rotary(torch.randn(s, d, device=dev), torch.randn(s, D, device=dev), torch.randn(s, D, device=dev))

        # Refusal of mismatched shapes naming the shapes
        bad_cos = torch.randn(5, D, device=dev)
        bad_sin = torch.randn(5, D, device=dev)
        with self.assertRaisesRegex(ValueError, rf"needs cos and sin \[{s}, {D}\] or \[{b}, {s}, {D}\], got cos \(5, {D}\)"):
            triton_ops.rotary(x, bad_cos, bad_sin)

        # Refusal when cos and sin shapes do not match
        mismatched_sin = torch.randn(s, D, device=dev)
        with self.assertRaisesRegex(ValueError, r"needs cos and sin"):
            triton_ops.rotary(x, torch.randn(b, s, D, device=dev), mismatched_sin)


if __name__ == "__main__":
    unittest.main()
