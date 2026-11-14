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
import math
import os
import re
import sys
import types
from typing import Any, Mapping, Sequence
import unittest
import unittest.mock

import torch

from .. import plan
from . import score
from .score import discover_scorable_ops, main, score_all


class ScoreTest(unittest.TestCase):
    """Verifies score calculation, injected timings, and failure behavior."""

    @staticmethod
    def _make_timer(timings: Mapping[str, float]) -> score.TimerFn:
        def _timer(op: str, ref: plan.Impl, vars: Sequence[plan.Impl], shape: str, **kwargs: Any) -> Mapping[str, float]:
            return timings
        return _timer

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
            timer=self._make_timer(timings),
        )

        self.assertEqual(exit_code, 0)
        output = buf.getvalue().splitlines()

        # Check kernel-check lines were printed.
        self.assertTrue(any("kernel-check vxpu::test_op_score reference 1x10" in line for line in output))
        self.assertTrue(any("kernel-check vxpu::test_op_score stub_var1 1x10" in line for line in output))
        self.assertTrue(any("kernel-check vxpu::test_op_score stub_var2 1x10" in line for line in output))

        # Check summary line before the last line.
        self.assertEqual(output[-2], "kernel-score vxpu::test_op_score best=stub_var2 speedup=4.00 shape=1x10")

        # Check last stdout line is the sum of log2 speedups formatted %.3f.
        self.assertEqual(output[-1], "2.000")

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
            timer=self._make_timer({"reference": 100.0}),
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
            timer=self._make_timer(timings),
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
        self.assertEqual(output[-1], "1.000")

    def test_score_padding_and_improvement_threshold(self) -> None:
        """Verifies that adding a non-faster variant does not move score past threshold,
        while a 10% speedup on an existing variant does."""
        op_base = "vxpu::test_op_base"
        mod_base = "vxpu.kernels.test_op_base"
        self.register_cleanup(op_base, mod_base)

        fake_mod_base = types.ModuleType(mod_base)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(1, 10, dtype=torch.float32, device=device),)

        setattr(fake_mod_base, "inputs", _inputs)
        setattr(fake_mod_base, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_base] = fake_mod_base

        @plan.implementation(op_base, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref_base(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(op_base, "variant_base", ("cpu",), "bit-identical", "base variant")
        def _var_base(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        # Baseline: variant_base is 4x faster (100 us ref / 25 us variant -> speedup 4.0, score 2.0).
        timings_base: dict[str, float] = {"reference": 100.0, "variant_base": 25.0}
        buf = io.StringIO()
        score_all(ops={op_base: ("1x10",)}, out=buf, timer=self._make_timer(timings_base))
        base_score = float(buf.getvalue().splitlines()[-1])
        self.assertAlmostEqual(base_score, 2.000, places=3)

        # 1. Padding: add a new op whose only variant matches reference speed (speedup 1.0).
        op_pad = "vxpu::test_op_pad"
        mod_pad = "vxpu.kernels.test_op_pad"
        self.register_cleanup(op_pad, mod_pad)

        fake_mod_pad = types.ModuleType(mod_pad)
        setattr(fake_mod_pad, "inputs", _inputs)
        setattr(fake_mod_pad, "SCORE_SHAPES", ("1x10",))
        sys.modules[mod_pad] = fake_mod_pad

        @plan.implementation(op_pad, "reference", ("cpu",), "bit-identical", "reference", reference=True)
        def _ref_pad(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(op_pad, "variant_pad", ("cpu",), "bit-identical", "matching variant")
        def _var_pad(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def _padded_timer(o: str, r: plan.Impl, v: Sequence[plan.Impl], s: str) -> dict[str, float]:
            if o == op_base:
                return timings_base
            return {"reference": 100.0, "variant_pad": 100.0}

        buf = io.StringIO()
        score_all(ops={op_base: ("1x10",), op_pad: ("1x10",)}, out=buf, timer=_padded_timer)
        padded_score = float(buf.getvalue().splitlines()[-1])
        # Score must not move past the threshold (in fact, adds 0.0).
        self.assertLess(abs(padded_score - base_score), score.THRESHOLD)
        self.assertAlmostEqual(padded_score, base_score, places=3)

        # 2. Improvement: make the existing variant 10% faster (latency = 25.0 / 1.10 = 22.727 us).
        timings_improved: dict[str, float] = {"reference": 100.0, "variant_base": 25.0 / 1.10}
        buf = io.StringIO()
        score_all(ops={op_base: ("1x10",)}, out=buf, timer=self._make_timer(timings_improved))
        improved_score = float(buf.getvalue().splitlines()[-1])

        # 10% faster variant must move the score past the threshold.
        improvement = improved_score - base_score
        self.assertGreater(improvement, score.THRESHOLD)
        self.assertAlmostEqual(improvement, math.log2(1.10), places=3)

    def test_goal_file_threshold_matches_score_threshold(self) -> None:
        """Verifies that score.THRESHOLD and .goals/create-cpu-kernels.md minImprovement agree."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        goal_path = os.path.join(repo_root, ".goals", "create-cpu-kernels.md")
        if not os.path.exists(goal_path):
            self.skipTest("the goal file is only present in a repository checkout")
        with open(goal_path, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"^\s*minImprovement:\s*([0-9.]+)", content, re.MULTILINE)
        self.assertIsNotNone(match, "minImprovement not found in .goals/create-cpu-kernels.md")
        assert match is not None
        goal_threshold = float(match.group(1))
        self.assertEqual(score.THRESHOLD, goal_threshold)

    def test_gpu_goal_file(self) -> None:
        """Verifies that .goals/create-gpu-kernels.md has manual schedule and valid metric parameters."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        goal_path = os.path.join(repo_root, ".goals", "create-gpu-kernels.md")
        if not os.path.exists(goal_path):
            self.skipTest("the goal file is only present in a repository checkout")
        with open(goal_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("schedule: manual", content)
        self.assertIn("direction: higher", content)
        self.assertIn("command: ./vxpu/dev/goals/gpu-kernel-score", content)
        self.assertIn("unit: summed fraction of memory-bandwidth floor gained over the reference", content)
        match = re.search(r"^\s*minImprovement:\s*([0-9.]+)", content, re.MULTILINE)
        self.assertIsNotNone(match, "minImprovement not found in .goals/create-gpu-kernels.md")

    def test_gpu_scoring_arithmetic(self) -> None:
        """Verifies that GPU scoring computes fraction gained over reference, zero for matching or slower."""
        test_op = "vxpu::test_op_gpu_score"
        mod_name = "vxpu.kernels.test_op_gpu_score"
        self.register_cleanup(test_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        # 100,000 float32 elements: input = 400 KB, output = 400 KB. Total = 800,000 bytes.
        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            return (torch.ones(100, 1000, dtype=torch.float32),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("100x1000",))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(test_op, "reference", ("cuda",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(test_op, "fast_var", ("cuda",), "bit-identical", "fast variant")
        def _fast(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        # Bandwidth: 1.0 GB/s = 1,000,000,000 bytes/sec = 1,000 bytes/us.
        # floor_us = 800,000 / (1.0 * 1000) = 800.0 us.
        # ref time: 4000.0 us -> ref_frac = 800.0 / 4000.0 = 0.200.
        # fast_var time: 1600.0 us -> var_frac = 800.0 / 1600.0 = 0.500.
        # Expected op_score = max(0.0, 0.500 - 0.200) = 0.300.
        timings_fast: dict[str, float] = {"reference": 4000.0, "fast_var": 1600.0}

        buf = io.StringIO()
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch.object(plan, "default_plan", return_value=plan.Plan("cpu")):
            # Also verifies indexed device strings like "cuda:0".
            exit_code = score_all(
                ops={test_op: ("100x1000",)},
                out=buf,
                timer=self._make_timer(timings_fast),
                device="cuda:0",
                card="test-card",
                bandwidth_gbs=1.0,
            )

        self.assertEqual(exit_code, 0)
        output = buf.getvalue().splitlines()
        summary_line = output[-2]
        self.assertIn("kernel-score vxpu::test_op_gpu_score best=fast_var fraction=0.50 speedup=2.50 shape=100x1000", summary_line)
        self.assertEqual(output[-1], "0.300")

        # Matching variant: latency matches reference (4000.0 us), score is 0.000.
        timings_matching: dict[str, float] = {"reference": 4000.0, "fast_var": 4000.0}
        buf = io.StringIO()
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch.object(plan, "default_plan", return_value=plan.Plan("cpu")):
            score_all(
                ops={test_op: ("100x1000",)},
                out=buf,
                timer=self._make_timer(timings_matching),
                device="cuda",
                card="test-card",
                bandwidth_gbs=1.0,
            )
        output = buf.getvalue().splitlines()
        self.assertEqual(output[-1], "0.000")

        # Slower variant: latency is worse than reference (5000.0 us), score is 0.000.
        timings_slower: dict[str, float] = {"reference": 4000.0, "fast_var": 5000.0}
        buf = io.StringIO()
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch.object(plan, "default_plan", return_value=plan.Plan("cpu")):
            score_all(
                ops={test_op: ("100x1000",)},
                out=buf,
                timer=self._make_timer(timings_slower),
                device="cuda",
                card="test-card",
                bandwidth_gbs=1.0,
            )
        output = buf.getvalue().splitlines()
        self.assertEqual(output[-1], "0.000")

    def test_cuda_no_card_fails_with_clear_message(self) -> None:
        """Verifies that requesting CUDA without a card fails with non-zero exit and clear message."""
        stderr_buf = io.StringIO()
        with unittest.mock.patch("torch.cuda.is_available", return_value=False), \
             unittest.mock.patch("sys.stderr", stderr_buf):
            exit_code = score_all(device="cuda")

        self.assertEqual(exit_code, 1)
        self.assertIn("no CUDA-capable card is available", stderr_buf.getvalue())

    def test_cuda_missing_bandwidth_fails(self) -> None:
        """Verifies that running CUDA scoring without bandwidth fails with clear message."""
        stderr_buf = io.StringIO()
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch("torch.cuda.get_device_name", return_value="test-card"), \
             unittest.mock.patch("sys.stderr", stderr_buf):
            exit_code = score_all(device="cuda")

        self.assertEqual(exit_code, 1)
        self.assertIn("requires --bandwidth-gb-s from the machine database", stderr_buf.getvalue())

    def test_timing_uses_specified_device(self) -> None:
        """Verifies that timing helper uses and synchronizes on the specified device under test."""
        devices_entered: list[str] = []

        class FakeDeviceCtx:
            def __init__(self, dev: str) -> None:
                self.dev = dev

            def __enter__(self) -> None:
                devices_entered.append(self.dev)

            def __exit__(self, *args: Any) -> None:
                pass

        sync_devices: list[str] = []

        def fake_sync(dev: str | None = None) -> None:
            if dev is not None:
                sync_devices.append(dev)

        fake_event = unittest.mock.MagicMock()
        fake_event.elapsed_time.return_value = 1.0

        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device", side_effect=FakeDeviceCtx), \
             unittest.mock.patch("torch.cuda.synchronize", side_effect=fake_sync), \
             unittest.mock.patch("torch.cuda.Event", return_value=fake_event):
            score.measure_us(lambda: None, on_cuda=True, rep=1, device="cuda:1")

        self.assertIn("cuda:1", devices_entered)
        self.assertIn("cuda:1", sync_devices)

    def test_gpu_shape_selection_by_gain_not_speedup(self) -> None:
        """Verifies that the winning shape is chosen by floor gain rather than speedup when they disagree."""
        test_op = "vxpu::test_op_shapes_disagree"
        mod_name = "vxpu.kernels.test_op_shapes_disagree"
        self.register_cleanup(test_op, mod_name)

        fake_mod = types.ModuleType(mod_name)

        def _inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
            if shapes == "shape_high_speedup":
                # 100 float32 elements = 400 bytes input + 400 bytes output = 800 bytes.
                return (torch.ones(10, 10, dtype=torch.float32),)
            # shape_high_gain: 100,000 float32 elements = 400,000 bytes in + 400,000 bytes out = 800,000 bytes.
            return (torch.ones(100, 1000, dtype=torch.float32),)

        setattr(fake_mod, "inputs", _inputs)
        setattr(fake_mod, "SCORE_SHAPES", ("shape_high_speedup", "shape_high_gain"))
        sys.modules[mod_name] = fake_mod

        @plan.implementation(test_op, "reference", ("cuda",), "bit-identical", "reference", reference=True)
        def _ref(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @plan.implementation(test_op, "my_var", ("cuda",), "bit-identical", "variant")
        def _var(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        # Bandwidth: 1.0 GB/s (1000 bytes / us).
        # shape_high_speedup (800 bytes):
        #   floor_us = 0.8 us.
        #   ref = 100.0 us (ref_frac = 0.008), var = 20.0 us (var_frac = 0.040).
        #   speedup = 5.0x, gain = 0.040 - 0.008 = 0.032.
        # shape_high_gain (800,000 bytes):
        #   floor_us = 800.0 us.
        #   ref = 4000.0 us (ref_frac = 0.200), var = 2000.0 us (var_frac = 0.400).
        #   speedup = 2.0x, gain = 0.400 - 0.200 = 0.200.
        # Rule by speedup would pick shape_high_speedup (5.0 > 2.0).
        # Rule by summed quantity (gain) must pick shape_high_gain (0.200 > 0.032).
        def _disagree_timer(op: str, ref: plan.Impl, vars: Sequence[plan.Impl], shape: str, **kwargs: Any) -> Mapping[str, float]:
            if shape == "shape_high_speedup":
                return {"reference": 100.0, "my_var": 20.0}
            return {"reference": 4000.0, "my_var": 2000.0}

        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
             unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch.object(plan, "default_plan", return_value=plan.Plan("cpu")):
            op_score = score.score_op(
                test_op,
                shapes=["shape_high_speedup", "shape_high_gain"],
                out=io.StringIO(),
                timer=_disagree_timer,
                device="cuda",
                card="test-card",
                bandwidth_gbs=1.0,
            )

        self.assertEqual(op_score.shape, "shape_high_gain")
        self.assertAlmostEqual(op_score.score, 0.200, places=3)
        self.assertAlmostEqual(op_score.speedup, 2.0, places=2)

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
