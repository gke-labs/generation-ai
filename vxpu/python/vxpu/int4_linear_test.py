# pyright: strict
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

"""Tests for vxpu::int4_linear under the kernel contract."""

import json
import unittest
from typing import Any

import torch

from . import ops, plan, plan_pb2
from .kernel_tools.check import run_check
from .kernels import int4_linear as int4_mod
from .kernels.int4_linear.gpu_triton import INT4_GEMV_MAX_ROWS, gpu_triton


class Int4LinearKernelTest(unittest.TestCase):
    """Verifies vxpu::int4_linear packing, reference math, registration, and Triton CUDA kernel."""

    def test_reference_matches_explicit_unpack_and_matmul(self) -> None:
        """Verifies the reference matches explicit per-nibble dequantization across dtypes and group sizes."""
        gen = torch.Generator(device="cpu").manual_seed(17)
        for group_size in (64, 128):
            K = group_size * 2
            N = 32
            n_groups = K // group_size
            for dtype in (torch.bfloat16, torch.float16):
                # Include values 0..15 so top-bit nibbles (8..15) exercise negative int32 words.
                w_q = torch.randint(0, 16, (K, N), generator=gen, dtype=torch.int32)
                z_q = torch.randint(0, 16, (n_groups, N), generator=gen, dtype=torch.int32)
                qweight, qzeros = int4_mod.pack_int4(w_q, z_q)
                scales = torch.empty((n_groups, N), dtype=torch.float32).uniform_(0.01, 0.1, generator=gen).to(dtype)
                x = torch.randn((2, 3, K), generator=gen, dtype=dtype)

                # Explicit elementwise dequantize for ground truth.
                w_ref = torch.empty((K, N), dtype=torch.float32)
                for k in range(K):
                    g = k // group_size
                    w_ref[k] = (w_q[k].float() - z_q[g].float()) * scales[g].float()
                expected = torch.matmul(x.float(), w_ref).to(dtype)

                ops_mod: Any = ops
                got: torch.Tensor = ops_mod.int4_linear(x, qweight, scales, qzeros)
                self.assertEqual(got.shape, (2, 3, N))
                self.assertEqual(got.dtype, dtype)
                self.assertTrue(torch.equal(got, expected), f"mismatch for group_size={group_size}, dtype={dtype}")

    def test_validation_rejects_bad_shapes_and_dtypes(self) -> None:
        """Verifies argument validation rejects invalid shapes and dtypes."""
        x, qw, sc, qz = int4_mod.inputs("1x128->64", "cpu")
        ops_mod: Any = ops
        with self.assertRaises(ValueError):
            ops_mod.int4_linear(x, qw.to(torch.int64), sc, qz)
        with self.assertRaises(ValueError):
            ops_mod.int4_linear(x, qw, sc, qz.to(torch.int64))
        with self.assertRaises(ValueError):
            ops_mod.int4_linear(x[:, :64], qw, sc, qz)
        with self.assertRaises(ValueError):
            ops_mod.int4_linear(x, qw, sc, qz[:, :4])

    def test_input_builder_shapes_and_modifiers(self) -> None:
        """Verifies the input builder parses shapes, group sizes, and dtypes."""
        x, qw, sc, qz = int4_mod.inputs("4x256->128 g=64 float16", "cpu", seed=3)
        self.assertEqual(x.shape, (4, 256))
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(qw.shape, (32, 128))
        self.assertEqual(qw.dtype, torch.int32)
        self.assertEqual(sc.shape, (4, 128))
        self.assertEqual(sc.dtype, torch.float16)
        self.assertEqual(qz.shape, (4, 16))
        self.assertEqual(qz.dtype, torch.int32)

        with self.assertRaises(ValueError):
            int4_mod.inputs("1x100->64", "cpu")
        with self.assertRaises(ValueError):
            int4_mod.inputs("1x128->60", "cpu")
        with self.assertRaises(ValueError):
            int4_mod.inputs("1x128->64 bogus_mod", "cpu")

    def test_kernel_contract_registration(self) -> None:
        """Verifies vxpu::int4_linear registers reference and gpu_triton under the kernel contract."""
        op = "vxpu::int4_linear"
        self.assertIn(op, plan.REGISTRY)
        impls = plan.REGISTRY[op]
        names = [im.name for im in impls]
        self.assertIn("reference", names)
        self.assertIn("gpu_triton", names)
        self.assertGreater(names.index("gpu_triton"), names.index("reference"))

        ref_impl = next(im for im in impls if im.name == "reference")
        self.assertTrue(ref_impl.reference)
        self.assertEqual(ref_impl.numerics, "bit-identical")
        self.assertEqual(ref_impl.devices, ("cpu", "cuda"))
        self.assertEqual(ref_impl.kind, plan_pb2.SITE_KIND_DENSE_GEMV)

        triton_impl = next(im for im in impls if im.name == "gpu_triton")
        self.assertEqual(triton_impl.numerics, "rounding")
        self.assertEqual(triton_impl.devices, ("cuda",))
        self.assertEqual(triton_impl.kind, plan_pb2.SITE_KIND_DENSE_GEMV)
        self.assertIsNotNone(triton_impl.condition)
        assert triton_impl.condition is not None
        self.assertEqual(triton_impl.condition.max_rows, INT4_GEMV_MAX_ROWS)

        table = plan.load_table(json.dumps({"sites": [{"op": op, "impl": "gpu_triton"}]}))
        self.assertIsNotNone(table)

    def test_cpu_kernel_check(self) -> None:
        """Verifies kernel-check runs cleanly on CPU for the reference implementation."""
        res = run_check("vxpu::int4_linear", "reference", "1x256->128", device="cpu", repeat=3)
        self.assertTrue(res.passed, f"CPU reference check failed: {res}")
        self.assertEqual(res.deviation, 0.0)
        self.assertGreater(res.bytes, 0)

    def test_cuda_gpu_triton_accuracy_and_check(self) -> None:
        """Verifies gpu_triton matches the reference on CUDA across decode batch sizes, dtypes, and shapes."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        triton_impl = next((im for im in plan.REGISTRY["vxpu::int4_linear"] if im.name == "gpu_triton"), None)
        plan_mod: Any = plan
        if triton_impl is None or triton_impl.applies is None or not triton_impl.applies(plan_mod.default_plan("cuda"), rows=1):
            self.skipTest("gpu_triton does not apply on this environment")

        shapes_list = [
            "1x128->64",
            "1x256->128 g=64",
            "2x256->128",
            "3x256->128",
            "4x256->128 float16",
            "8x256->128",
            "1x2048->2048",
            "1x2048->6144",
            "1x4096->4096",
        ]
        for shape in shapes_list:
            res = run_check(
                "vxpu::int4_linear",
                "gpu_triton",
                shape,
                device="cuda",
                card=torch.cuda.get_device_name(0),
                bandwidth_gbs=300.0,
                floor="nominal",
                repeat=5,
            )
            self.assertTrue(res.passed, f"gpu_triton kernel-check failed on {shape}: {res}")

        # Verify 3D input preservation [B, S, K] -> [B, S, N].
        x2d, qw, sc, qz = int4_mod.inputs("4x256->128", "cuda", seed=5)
        x3d = x2d.view(2, 2, 256)
        ops_mod: Any = ops
        ref_impl = plan.reference_of("vxpu::int4_linear")
        self.assertIsNotNone(ref_impl)
        assert ref_impl is not None
        got_3d = gpu_triton(x3d, qw, sc, qz)
        want_3d: torch.Tensor = ops_mod.call("vxpu::int4_linear", ref_impl.fn, x3d, qw, sc, qz)
        self.assertEqual(got_3d.shape, (2, 2, 128))
        self.assertEqual(got_3d.dtype, x3d.dtype)
        torch.testing.assert_close(got_3d, want_3d, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
