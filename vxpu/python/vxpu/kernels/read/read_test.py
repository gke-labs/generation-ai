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

"""Unit tests for vxpu::read."""

# pyright: strict

from typing import Any
import unittest

import torch

from . import inputs
from ... import ops, plan


class ReadOpTest(unittest.TestCase):
    """Verifies input builder and reference implementation of vxpu::read."""

    def test_inputs(self) -> None:
        (x,) = inputs("1048576", device="cpu")
        self.assertEqual(x.shape, (1048576,))
        self.assertEqual(x.dtype, torch.uint8)
        self.assertEqual(int(x.sum().item()), 1048576)

        with self.assertRaises(ValueError):
            inputs("0", device="cpu")
        with self.assertRaises(ValueError):
            inputs("-10", device="cpu")

    def test_reference(self) -> None:
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8)
        impl = plan.reference_of("vxpu::read")
        self.assertIsNotNone(impl)
        assert impl is not None
        ops_mod: Any = ops
        got: Any = ops_mod.call("vxpu::read", impl.fn, x)
        self.assertEqual(int(got.item()), 15)

    def test_gpu_triton(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        triton_impl = None
        for impl in plan.REGISTRY.get("vxpu::read", []):
            if impl.name == "gpu_triton":
                triton_impl = impl
                break
        if triton_impl is None:
            self.skipTest("gpu_triton variant not found")
        assert triton_impl is not None

        plan_mod: Any = plan
        dp: Any = plan_mod.default_plan("cuda")
        if triton_impl.applies and not triton_impl.applies(dp):
            self.skipTest("gpu_triton does not apply on this environment")

        x = torch.ones(1024, dtype=torch.uint8, device="cuda")
        ops_mod: Any = ops
        got: Any = ops_mod.call("vxpu::read", triton_impl.fn, x)
        self.assertEqual(int(got.item()), 1024)
