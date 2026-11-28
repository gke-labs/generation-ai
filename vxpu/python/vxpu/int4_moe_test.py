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

"""Tests for vxpu::int4_moe under the kernel contract."""

import json
import unittest
from typing import Any

import torch

from . import ops, plan, plan_pb2
from .kernel_tools.check import run_check
from .kernels import int4_moe as int4_moe_mod
from .kernels.int4_moe.gpu_triton import INT4_MOE_MAX_ROWS, gpu_triton


def _dequant_explicit(
    weight_q: torch.Tensor,
    zero_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantizes one [K, N] expert weight matrix elementwise in float32."""
    K, N = int(weight_q.shape[0]), int(weight_q.shape[1])
    out = torch.empty((K, N), dtype=torch.float32)
    for k in range(K):
        g = k // group_size
        out[k] = (weight_q[k].float() - zero_q[g].float()) * scales[g].float()
    return out


class Int4MoeKernelTest(unittest.TestCase):
    """Verifies vxpu::int4_moe packing, reference math, validation, registration, and Triton CUDA kernel."""

    def test_reference_matches_explicit_per_expert_swiglu(self) -> None:
        """Verifies the reference matches explicit elementwise dequant, SwiGLU, and weighted sum."""
        gen = torch.Generator(device="cpu").manual_seed(23)
        E = 6
        top_k = 2
        for group_size in (64, 128):
            K = group_size * 2
            I = group_size
            n_groups_k = K // group_size
            n_groups_i = I // group_size
            for dtype in (torch.bfloat16, torch.float16):
                gate_wq = torch.randint(0, 16, (E, K, I), generator=gen, dtype=torch.int32)
                gate_zq = torch.randint(0, 16, (E, n_groups_k, I), generator=gen, dtype=torch.int32)
                gate_qw, gate_qz = int4_moe_mod.pack_int4_experts(gate_wq, gate_zq)
                gate_sc = torch.empty((E, n_groups_k, I), dtype=torch.float32).uniform_(0.01, 0.08, generator=gen).to(dtype)

                up_wq = torch.randint(0, 16, (E, K, I), generator=gen, dtype=torch.int32)
                up_zq = torch.randint(0, 16, (E, n_groups_k, I), generator=gen, dtype=torch.int32)
                up_qw, up_qz = int4_moe_mod.pack_int4_experts(up_wq, up_zq)
                up_sc = torch.empty((E, n_groups_k, I), dtype=torch.float32).uniform_(0.01, 0.08, generator=gen).to(dtype)

                down_wq = torch.randint(0, 16, (E, I, K), generator=gen, dtype=torch.int32)
                down_zq = torch.randint(0, 16, (E, n_groups_i, K), generator=gen, dtype=torch.int32)
                down_qw, down_qz = int4_moe_mod.pack_int4_experts(down_wq, down_zq)
                down_sc = torch.empty((E, n_groups_i, K), dtype=torch.float32).uniform_(0.01, 0.08, generator=gen).to(dtype)

                # Exercise 3D inputs [B=2, S=2, K] with shared experts across tokens.
                x = torch.randn((2, 2, K), generator=gen, dtype=dtype)
                topk_ids = torch.tensor([[[0, 3], [3, 5]], [[5, 0], [1, 3]]], dtype=torch.int64)
                topk_weights = torch.tensor([[[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.8], [0.55, 0.45]]], dtype=torch.float32)

                expected = torch.zeros((2, 2, K), dtype=torch.float32)
                for b in range(2):
                    for s in range(2):
                        xb = x[b, s].float()
                        for r in range(top_k):
                            e = int(topk_ids[b, s, r].item())
                            wg = _dequant_explicit(gate_wq[e], gate_zq[e], gate_sc[e], group_size)
                            wu = _dequant_explicit(up_wq[e], up_zq[e], up_sc[e], group_size)
                            wd = _dequant_explicit(down_wq[e], down_zq[e], down_sc[e], group_size)
                            h = torch.nn.functional.silu(xb @ wg) * (xb @ wu)
                            expected[b, s] += topk_weights[b, s, r].float() * (h @ wd)
                expected_out = expected.to(dtype)

                ops_mod: Any = ops
                got: torch.Tensor = ops_mod.int4_moe(
                    x,
                    topk_ids,
                    topk_weights,
                    gate_qw,
                    gate_sc,
                    gate_qz,
                    up_qw,
                    up_sc,
                    up_qz,
                    down_qw,
                    down_sc,
                    down_qz,
                )
                self.assertEqual(got.shape, (2, 2, K))
                self.assertEqual(got.dtype, dtype)
                torch.testing.assert_close(got, expected_out, rtol=1e-3, atol=1e-3)

    def test_validation_rejects_bad_shapes_and_dtypes(self) -> None:
        """Verifies upfront argument validation in eager, meta, and FakeTensorMode."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        args = int4_moe_mod.inputs("2x128->64 e=4 k=2 g=64", "cpu")
        x, ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz = args
        ops_mod: Any = ops

        bad_cases: list[tuple[torch.Tensor, ...]] = [
            (torch.tensor(1.0, dtype=x.dtype), ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (torch.empty((2, 0), dtype=x.dtype), ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x.to(torch.int32), ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids.to(torch.float32), wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids[:, :1], wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids, wts.to(torch.int32), gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids, wts, gqw.to(torch.int64), gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids, wts, gqw, gsc[0], gqz, uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids, wts, gqw, gsc, gqz[:, :, :4], uqw, usc, uqz, dqw, dsc, dqz),
            (x, ids, wts, gqw, gsc, gqz, uqw[:2], usc[:2], uqz[:2], dqw, dsc, dqz),
            (x, ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz[:, :, :4]),
        ]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                ops_mod.int4_moe(*bad)
            with self.assertRaises(ValueError):
                ops_mod.int4_moe(*(t.to("meta") for t in bad))
            with FakeTensorMode():
                fbad = tuple(torch.empty(t.shape, dtype=t.dtype) for t in bad)
                with self.assertRaises(ValueError):
                    ops_mod.int4_moe(*fbad)

        # Verify valid meta and fake calls return the expected shape and dtype.
        out_meta: torch.Tensor = ops_mod.int4_moe(*(t.to("meta") for t in args))
        self.assertEqual(out_meta.shape, x.shape)
        self.assertEqual(out_meta.dtype, x.dtype)
        with FakeTensorMode():
            fargs = tuple(torch.empty(t.shape, dtype=t.dtype) for t in args)
            out_fake: torch.Tensor = ops_mod.int4_moe(*fargs)
            self.assertEqual(out_fake.shape, x.shape)
            self.assertEqual(out_fake.dtype, x.dtype)

        # Verify out-of-range expert index fails cleanly in eager reference execution.
        bad_ids = ids.clone()
        bad_ids[0, 0] = 4
        with self.assertRaises(ValueError):
            ops_mod.int4_moe(x, bad_ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz)

    def test_input_builder_and_active_bytes(self) -> None:
        """Verifies the input builder and active_bytes accounting for sparse MoE."""
        args = int4_moe_mod.inputs("1x256->128 e=16 k=2 g=64 float16", "cpu", seed=7)
        x, ids, wts, gqw, gsc, gqz, uqw, usc, uqz, dqw, dsc, dqz = args
        self.assertEqual(x.shape, (1, 256))
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(ids.shape, (1, 2))
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(wts.shape, (1, 2))
        self.assertEqual(gqw.shape, (16, 32, 128))
        self.assertEqual(gsc.shape, (16, 4, 128))
        self.assertEqual(gqz.shape, (16, 4, 16))
        self.assertEqual(uqw.shape, (16, 32, 128))
        self.assertEqual(usc.shape, (16, 4, 128))
        self.assertEqual(uqz.shape, (16, 4, 16))
        self.assertEqual(dqw.shape, (16, 16, 256))
        self.assertEqual(dsc.shape, (16, 2, 256))
        self.assertEqual(dqz.shape, (16, 2, 32))

        # Active bytes must count only the 2 active experts out of 16 plus activations.
        out = torch.empty_like(x)
        active = int4_moe_mod.active_bytes(args, out)
        all_weight_bytes = sum(t.numel() * t.element_size() for t in args[3:])
        expected_active = (
            x.numel() * x.element_size()
            + ids.numel() * ids.element_size()
            + wts.numel() * wts.element_size()
            + out.numel() * out.element_size()
            + (2 * all_weight_bytes) // 16
        )
        self.assertEqual(active, expected_active)
        self.assertLess(active, all_weight_bytes)

        for bad_shape in ("1x256->128", "1x256->128 e=4 k=8", "1x200->128 e=4 k=2", "1x256->128 e=4 k=2 bogus"):
            with self.assertRaises(ValueError):
                int4_moe_mod.inputs(bad_shape, "cpu")

    def test_kernel_contract_registration(self) -> None:
        """Verifies vxpu::int4_moe registers reference and gpu_triton under the kernel contract."""
        op = "vxpu::int4_moe"
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
        self.assertEqual(ref_impl.kind, plan_pb2.SITE_KIND_EXPERT_OP)

        triton_impl = next(im for im in impls if im.name == "gpu_triton")
        self.assertEqual(triton_impl.numerics, "rounding")
        self.assertEqual(triton_impl.devices, ("cuda",))
        self.assertEqual(triton_impl.kind, plan_pb2.SITE_KIND_EXPERT_OP)
        self.assertIsNotNone(triton_impl.condition)
        assert triton_impl.condition is not None
        self.assertEqual(triton_impl.condition.max_rows, INT4_MOE_MAX_ROWS)

        table = plan.load_table(json.dumps({"sites": [{"op": op, "impl": "gpu_triton"}]}))
        self.assertIsNotNone(table)

    def test_cpu_kernel_check(self) -> None:
        """Verifies kernel-check runs cleanly on CPU and reports active expert bytes."""
        shape = "1x256->128 e=8 k=2"
        res = run_check("vxpu::int4_moe", "reference", shape, device="cpu", repeat=3)
        self.assertTrue(res.passed, f"CPU reference check failed: {res}")
        self.assertEqual(res.deviation, 0.0)
        args = int4_moe_mod.inputs(shape, "cpu")
        self.assertEqual(res.bytes, int4_moe_mod.active_bytes(args, torch.empty_like(args[0])))

    def test_cuda_gpu_triton_accuracy_and_check(self) -> None:
        """Verifies gpu_triton matches the reference on CUDA across decode batch sizes, dtypes, and shapes."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        triton_impl = next((im for im in plan.REGISTRY["vxpu::int4_moe"] if im.name == "gpu_triton"), None)
        plan_mod: Any = plan
        if triton_impl is None or triton_impl.applies is None or not triton_impl.applies(plan_mod.default_plan("cuda"), rows=1):
            self.skipTest("gpu_triton does not apply on this environment")

        shapes_list = [
            "1x256->128 e=8 k=2",
            "1x256->128 e=8 k=2 g=64",
            "4x256->128 e=8 k=2 float16",
            "8x256->128 e=16 k=4",
            "1x2048->768 e=128 k=8",
            "4x2048->768 e=128 k=8",
            "8x2048->768 e=128 k=8",
        ]
        for shape in shapes_list:
            res = run_check(
                "vxpu::int4_moe",
                "gpu_triton",
                shape,
                device="cuda",
                card=torch.cuda.get_device_name(0),
                bandwidth_gbs=300.0,
                floor="nominal",
                repeat=5,
            )
            self.assertTrue(res.passed, f"gpu_triton kernel-check failed on {shape}: {res}")

        # Verify 3D input shape preservation [B, S, K] -> [B, S, K] with int64 topk_ids.
        args = int4_moe_mod.inputs("4x256->128 e=8 k=2", "cuda", seed=11)
        x3d = args[0].view(2, 2, 256)
        ids3d = args[1].to(torch.int64).view(2, 2, 2)
        wts3d = args[2].to(torch.bfloat16).view(2, 2, 2)
        ops_mod: Any = ops
        ref_impl = plan.reference_of("vxpu::int4_moe")
        self.assertIsNotNone(ref_impl)
        assert ref_impl is not None
        got_3d = gpu_triton(x3d, ids3d, wts3d, *args[3:])
        want_3d: torch.Tensor = ops_mod.call("vxpu::int4_moe", ref_impl.fn, x3d, ids3d, wts3d, *args[3:])
        self.assertEqual(got_3d.shape, (2, 2, 256))
        self.assertEqual(got_3d.dtype, x3d.dtype)
        torch.testing.assert_close(got_3d, want_3d, rtol=1e-2, atol=1e-2)

        # Verify CUDA graph capture and replay succeed without host synchronization.
        cg_args = int4_moe_mod.inputs("1x256->128 e=8 k=2", "cuda", seed=17)
        want_cg: torch.Tensor = ops_mod.call("vxpu::int4_moe", ref_impl.fn, *cg_args)
        for _ in range(2):
            _ = gpu_triton(*cg_args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got_cg = gpu_triton(*cg_args)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got_cg, want_cg, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
