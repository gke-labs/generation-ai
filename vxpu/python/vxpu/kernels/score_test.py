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

"""Unit tests for kernel scoring, injected timings arithmetic, and failure handling."""

# pyright: strict

import io
import sys
import types
import unittest
import unittest.mock

import torch

from .. import plan
from . import score
from .score import discover_scorable_ops, main, score_all


class ScoreTest(unittest.TestCase):
    """Verifies score calculation, injected timings, and failure behavior."""

    def register_cleanup(self, op: str, mod_name: str) -> None:
        """Registers a cleanup callback to remove test op and module after test run."""
        def cleanup() -> None:
            plan.REGISTRY.pop(op, None)
            sys.modules.pop(mod_name, None)

        self.addCleanup(cleanup)

    def test_score_arithmetic_with_injected_timings(self) -> None:
        test_op = "vxpu::test_op_score"
        mod_name = "vxpu.kernels.test_op_score"
        self.register_cleanup(test_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(test_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(test_op, "stub_var1", ("cpu",), "bit-identical", "stub 1")
        def _stub1(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(test_op, "stub_var2", ("cpu",), "bit-identical", "stub 2")
        def _stub2(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        timings: dict[str, float] = {
            "reference": 100.0,
            "stub_var1": 50.0,   # speedup: 100 / 50 = 2.0
            "stub_var2": 25.0,   # speedup: 100 / 25 = 4.0
        }

        buf = io.StringIO()
        exit_code = score_all(
            ops={test_op: ("1x10",)},
            out=buf,
            timer=lambda op, ref, vars, shape: timings,
        )

        self.assertEqual(exit_code, 0)
        output = buf.getvalue().splitlines()

        # Check kernel-check lines were printed.
        self.assertTrue(any("kernel-check vxpu::test_op_score reference 1x10" in line for line in output))
        self.assertTrue(any("kernel-check vxpu::test_op_score stub_var1 1x10" in line for line in output))
        self.assertTrue(any("kernel-check vxpu::test_op_score stub_var2 1x10" in line for line in output))

        # Check summary line before the last line.
        self.assertEqual(output[-2], "kernel-score vxpu::test_op_score best=stub_var2 speedup=4.00 shape=1x10")

        # Check last stdout line is the sum formatted %.3f.
        self.assertEqual(output[-1], "4.000")

    def test_variant_failure_exits_one(self) -> None:
        fail_op = "vxpu::test_op_fail"
        mod_name = "vxpu.kernels.test_op_fail"
        self.register_cleanup(fail_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(fail_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(fail_op, "bad_variant", ("cpu",), "bit-identical", "bad plus one")
        def _bad(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        buf = io.StringIO()
        exit_code = score_all(ops={fail_op: ("1x10",)}, out=buf)

        self.assertEqual(exit_code, 1)
        output = buf.getvalue().splitlines()

        # The failing check line must be printed.
        self.assertTrue(any("kernel-check vxpu::test_op_fail bad_variant 1x10" in line and "FAIL" in line for line in output))

        # No kernel-score summary line or final score line should be printed.
        self.assertFalse(any(line.startswith("kernel-score") for line in output))

    def test_op_without_variants_scores_zero(self) -> None:
        empty_op = "vxpu::test_op_empty"
        mod_name = "vxpu.kernels.test_op_empty"
        self.register_cleanup(empty_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(empty_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        buf = io.StringIO()
        exit_code = score_all(
            ops={empty_op: ("1x10",)},
            out=buf,
            timer=lambda op, ref, vars, shape: {ref.name: 100.0},
        )

        self.assertEqual(exit_code, 0)
        output = buf.getvalue().splitlines()
        self.assertEqual(output[-2], "kernel-score vxpu::test_op_empty best=none speedup=0.00 shape=none")
        self.assertEqual(output[-1], "0.000")

    def test_variant_skipped_when_not_applicable(self) -> None:
        skip_op = "vxpu::test_op_skip"
        mod_name = "vxpu.kernels.test_op_skip"
        self.register_cleanup(skip_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(skip_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def _unsupported_applies(plan: object, **_: object) -> bool:
            return False

        @plan.implementation(
            skip_op,
            "unsupported_var",
            ("cpu",),
            "bit-identical",
            "not supported here",
            applies=_unsupported_applies,
        )
        def _unsupported(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(skip_op, "supported_var", ("cpu",), "bit-identical", "supported")
        def _supported(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        timings: dict[str, float] = {
            "reference": 100.0,
            "supported_var": 50.0,
        }

        buf = io.StringIO()
        exit_code = score_all(
            ops={skip_op: ("1x10",)},
            out=buf,
            timer=lambda op, ref, vars, shape: timings,
        )

        self.assertEqual(exit_code, 0)
        output = buf.getvalue().splitlines()

        # Unsupported variant is skipped with reason printed.
        self.assertIn(
            "kernel-score vxpu::test_op_skip skipped=unsupported_var reason=not available on this machine",
            output,
        )
        # Supported variant is checked and scored.
        self.assertTrue(any("kernel-check vxpu::test_op_skip supported_var 1x10" in line for line in output))
        self.assertEqual(output[-2], "kernel-score vxpu::test_op_skip best=supported_var speedup=2.00 shape=1x10")
        self.assertEqual(output[-1], "2.000")

    def test_discover_scorable_ops(self) -> None:
        discovered = discover_scorable_ops()
        self.assertIn("vxpu::fp8_linear", discovered)
        self.assertEqual(discovered["vxpu::fp8_linear"], ("1x6144->25600",))
        # Ensure no test ops leaked into discovered ops.
        for op in discovered:
            self.assertFalse(op.startswith("vxpu::test_op_"))

    def test_main_cli(self) -> None:
        test_op = "vxpu::test_op_cli"
        mod_name = "vxpu.kernels.test_op_cli"
        self.register_cleanup(test_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(test_op, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        with unittest.mock.patch.object(score, "discover_scorable_ops", return_value={test_op: ("1x10",)}):
            exit_code = main([])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
