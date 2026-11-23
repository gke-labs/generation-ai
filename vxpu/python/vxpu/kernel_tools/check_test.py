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

"""Unit tests for kernel registration, the check runner, and arithmetic."""

# pyright: strict

import io
import json
import subprocess
import sys
import types
import unittest
import unittest.mock

import torch

from .. import plan
from ..plan import REGISTRY
from .check import Result, floor_us, line, main, run_check


class CheckTest(unittest.TestCase):
    """Verifies kernel check failure paths, registration naming, and arithmetic."""

    def test_arithmetic(self) -> None:
        self.assertEqual(f"{floor_us(157_500_000, 1460.0):.1f}", "107.9")

        res = Result(
            op="vxpu::test_op",
            variant="test_var",
            shapes="1x128",
            numerics="rounding",
            passed=True,
            deviation=0.001,
            bytes=1000,
            us=10.0,
            floor_us=None,
            fraction=None,
            card="unknown",
            timing="warm",
        )
        rendered = line(res)
        self.assertIn("fraction=n/a", rendered)
        self.assertIn("floor_us=n/a", rendered)
        self.assertNotIn("floor=", rendered)

    def test_floor_rendering(self) -> None:
        res_measured = Result(
            op="vxpu::test_op",
            variant="test_var",
            shapes="1x128",
            numerics="rounding",
            passed=True,
            deviation=0.001,
            bytes=1000,
            us=10.0,
            floor_us=8.0,
            fraction=0.8,
            card="NVIDIA L4",
            timing="warm",
            floor="measured",
        )
        rendered_measured = line(res_measured)
        self.assertIn("floor=measured", rendered_measured)
        self.assertIn('card="NVIDIA L4"', rendered_measured)
        self.assertNotIn("(>floor)", rendered_measured)

        res_nominal = Result(
            op="vxpu::test_op",
            variant="test_var",
            shapes="1x128",
            numerics="rounding",
            passed=True,
            deviation=0.001,
            bytes=1000,
            us=10.0,
            floor_us=8.0,
            fraction=0.8,
            card="NVIDIA A100-SXM4-40GB",
            timing="warm",
            floor="nominal",
        )
        rendered_nominal = line(res_nominal)
        self.assertIn("floor=nominal", rendered_nominal)

        res_exceeds = Result(
            op="vxpu::test_op",
            variant="test_var",
            shapes="1x128",
            numerics="rounding",
            passed=True,
            deviation=0.001,
            bytes=1000,
            us=5.0,
            floor_us=8.0,
            fraction=1.6,
            card="NVIDIA L4",
            timing="warm",
            floor="measured",
        )
        rendered_exceeds = line(res_exceeds)
        self.assertIn("fraction=1.60 (>floor)", rendered_exceeds)

    def test_naming(self) -> None:
        op = "vxpu::fp8_linear"
        self.assertIn(op, REGISTRY)
        impls = REGISTRY[op]
        names = [im.name for im in impls]
        self.assertIn("bf16_dequant", names)
        self.assertIn("reference", names)
        self.assertGreater(names.index("bf16_dequant"), names.index("reference"))

        var_impl = next(im for im in impls if im.name == "bf16_dequant")
        self.assertEqual(getattr(var_impl.fn, "__module__", ""), "vxpu.kernels.fp8_linear.bf16_dequant")

        table = plan.load_table(json.dumps({"sites": [{"op": op, "impl": "bf16_dequant"}]}))
        self.assertIsNotNone(table)

    def test_fp8_grouped_linear_cpu_kernel(self) -> None:
        """Verifies fp8_grouped_linear cpu_kernel registers and passes check."""
        op = "vxpu::fp8_grouped_linear"
        self.assertIn(op, REGISTRY)
        impls = REGISTRY[op]
        names = [im.name for im in impls]
        self.assertIn("reference", names)
        self.assertIn("cpu_kernel", names)
        var_impl = next(im for im in impls if im.name == "cpu_kernel")
        self.assertEqual(var_impl.numerics, "rounding")
        self.assertEqual(var_impl.devices, ("cpu",))
        self.assertIsNotNone(var_impl.condition)
        assert var_impl.condition is not None
        self.assertEqual(var_impl.condition.max_rows, 4)
        self.assertEqual(var_impl.condition.require_external_fact, "cpu_kernels")

        from .. import kernels
        if kernels.load() is None:
            self.skipTest("C++ kernels not available")

        try:
            import vxpu.deepseek_v41 as ds
            getattr(ds, "_fork")()
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"deepseek_v41 fork unavailable: {e}")

        for shape in ("1x8x4096->8x1024", "2x8x4096->8x1024"):
            res = run_check(op, "cpu_kernel", shape, device="cpu")
            self.assertTrue(res.passed)
            self.assertEqual(res.numerics, var_impl.numerics)

        from ..kernels import fp8_grouped_linear
        from ..kernels.fp8_grouped_linear.cpu_kernel import cpu_kernel
        o_5, w_5, s_5 = fp8_grouped_linear.inputs("5x8x4096->8x1024", "cpu")
        with self.assertRaises(RuntimeError):
            cpu_kernel(o_5, w_5, s_5)

    def test_failure_path(self) -> None:
        throwaway_op = "vxpu::throwaway_op"
        mod_name = "vxpu.kernels.throwaway_op"
        self.addCleanup(plan.REGISTRY.pop, throwaway_op, None)
        self.addCleanup(sys.modules.pop, mod_name, None)
        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        sys.modules[mod_name] = fake_mod

        @plan.implementation(throwaway_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(throwaway_op, "bad_plus_one", ("cpu",), "bit-identical", "plus one")
        def _bad_plus_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        r1 = run_check(throwaway_op, "bad_plus_one", "1x10", device="cpu")
        self.assertFalse(r1.passed)
        buf = io.StringIO()
        with unittest.mock.patch("sys.stderr", buf):
            exit_code = main([throwaway_op, "bad_plus_one", "--shapes", "1x10", "--device", "cpu"])
        self.assertEqual(exit_code, 1)
        self.assertIn("kernel-check: no --bandwidth-gb-s given; the floor is not reported", buf.getvalue())

        # Bandwidth without floor basis must be rejected.
        buf_no_basis = io.StringIO()
        with unittest.mock.patch("sys.stderr", buf_no_basis), self.assertRaises(SystemExit):
            main([
                throwaway_op,
                "bad_plus_one",
                "--shapes",
                "1x10",
                "--device",
                "cpu",
                "--bandwidth-gb-s",
                "100",
            ])
        self.assertIn("--floor-basis (measured or nominal) is required", buf_no_basis.getvalue())

        buf_with_bw = io.StringIO()
        with unittest.mock.patch("sys.stderr", buf_with_bw):
            exit_code_bw = main([
                throwaway_op,
                "bad_plus_one",
                "--shapes",
                "1x10",
                "--device",
                "cpu",
                "--bandwidth-gb-s",
                "100",
                "--floor-basis",
                "nominal",
            ])
        self.assertEqual(exit_code_bw, 1)
        self.assertNotIn("kernel-check: no --bandwidth-gb-s given", buf_with_bw.getvalue())

        @plan.implementation(throwaway_op, "round_pass", ("cpu",), "rounding", "scale 1.01")
        def _round_pass(x: torch.Tensor) -> torch.Tensor:
            return x * 1.01

        r2 = run_check(throwaway_op, "round_pass", "1x10", device="cpu")
        self.assertTrue(r2.passed)

        @plan.implementation(throwaway_op, "round_fail", ("cpu",), "rounding", "scale 1.1")
        def _round_fail(x: torch.Tensor) -> torch.Tensor:
            return x * 1.1

        r3 = run_check(throwaway_op, "round_fail", "1x10", device="cpu")
        self.assertFalse(r3.passed)

        @plan.implementation(throwaway_op, "round_nan", ("cpu",), "rounding", "returns nan")
        def _round_nan(x: torch.Tensor) -> torch.Tensor:
            return x * float("nan")

        r_nan = run_check(throwaway_op, "round_nan", "1x10", device="cpu")
        self.assertFalse(r_nan.passed)

        with self.assertRaises(ValueError) as ctx:
            run_check("vxpu::nonexistent_op", "foo", "1x10", device="cpu")
        self.assertIn("unknown op", str(ctx.exception))

    def test_engine_import_does_not_import_kernel_tools(self) -> None:
        """Verifies that importing vxpu.engine does not load check or score."""
        cmd = [
            sys.executable,
            "-c",
            "import sys, vxpu.engine; "
            "loaded = [m for m in sys.modules if 'kernel_tools' in m or 'check' in m or 'score' in m]; "
            "assert not any('kernel_tools' in m for m in sys.modules), f'kernel_tools loaded: {loaded}'; "
            "assert 'vxpu.kernel_tools.check' not in sys.modules; "
            "assert 'vxpu.kernel_tools.score' not in sys.modules",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"engine imported kernel tools:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_dense_op_input_builders(self) -> None:
        """Verifies input builders for the dense ops produce expected shapes and types."""
        from ..kernels import fp8_grouped_linear, hc_sinkhorn, head_linear, rotary

        # fp8_grouped_linear
        o, w_grp, s_grp = fp8_grouped_linear.inputs("1x8x4096->8x1024", "cpu")
        self.assertEqual(o.shape, (1, 1, 8, 4096))
        self.assertEqual(o.dtype, torch.bfloat16)
        self.assertEqual(w_grp.shape, (8192, 4096))
        self.assertEqual(w_grp.dtype, torch.float8_e4m3fn)
        self.assertEqual(s_grp.shape, (256, 128))
        self.assertEqual(s_grp.dtype, torch.float8_e8m0fnu)

        # head_linear
        x_head, w_head = head_linear.inputs("1x5120->129280 bfloat16", "cpu")
        self.assertEqual(x_head.shape, (1, 5120))
        self.assertEqual(x_head.dtype, torch.bfloat16)
        self.assertEqual(w_head.shape, (129280, 5120))
        self.assertEqual(w_head.dtype, torch.bfloat16)

        # head_linear accepts surrounding whitespace
        x_head_ws, w_head_ws = head_linear.inputs("  1x5120->129280  ", "cpu")
        self.assertEqual(x_head_ws.shape, (1, 5120))
        self.assertEqual(w_head_ws.shape, (129280, 5120))
        self.assertEqual(w_head_ws.dtype, torch.bfloat16)

        # head_linear rejects unknown dtypes or extra tokens
        with self.assertRaises(ValueError):
            head_linear.inputs("1x5120->129280 invalid_dtype", "cpu")
        with self.assertRaises(ValueError):
            head_linear.inputs("1x5120->129280 bfloat16 extra", "cpu")

        # hc_sinkhorn
        mixes, hc_scale, hc_base, hc_mult, iters, eps = hc_sinkhorn.inputs("tokens=1 hc=4", "cpu")
        self.assertEqual(mixes.shape, (1, 24))
        self.assertEqual(mixes.dtype, torch.float32)
        self.assertEqual(hc_scale.shape, (3,))
        self.assertEqual(hc_base.shape, (24,))
        self.assertEqual(hc_mult, 4)
        self.assertEqual(iters, 20)
        self.assertEqual(eps, 1e-06)

        # hc_sinkhorn rejects unknown tokens or non-positive dimensions
        with self.assertRaises(ValueError):
            hc_sinkhorn.inputs("tokens=1 hc=4 unknown=8", "cpu")
        with self.assertRaises(ValueError):
            hc_sinkhorn.inputs("tokens=0 hc=4", "cpu")
        with self.assertRaises(ValueError):
            hc_sinkhorn.inputs("tokens=1 hc=0", "cpu")

        # rotary
        x_rot, freqs_rot, inv = rotary.inputs("1x1x64x64", "cpu")
        self.assertEqual(x_rot.shape, (1, 1, 64, 64))
        self.assertEqual(x_rot.dtype, torch.bfloat16)
        self.assertEqual(freqs_rot.shape, (1, 32))
        self.assertEqual(freqs_rot.dtype, torch.complex64)
        self.assertFalse(inv)

        _, _, inv_true = rotary.inputs("1x1x64x64 inv", "cpu")
        self.assertTrue(inv_true)

        x_rot_row, freqs_rot_row, inv_row = rotary.inputs("4x1x64x64 per_row", "cpu")
        self.assertEqual(x_rot_row.shape, (4, 1, 64, 64))
        self.assertEqual(freqs_rot_row.shape, (4, 1, 32))
        self.assertFalse(inv_row)

        _, _, inv_row_true = rotary.inputs("4x1x64x64 per_row inv", "cpu")
        self.assertTrue(inv_row_true)

        with self.assertRaises(ValueError):
            rotary.inputs("4x1x64x64 invalid_modifier", "cpu")


if __name__ == "__main__":
    unittest.main()
