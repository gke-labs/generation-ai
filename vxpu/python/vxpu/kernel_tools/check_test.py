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


if __name__ == "__main__":
    unittest.main()
