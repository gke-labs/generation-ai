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

"""Variant cpu_kernel of vxpu::fp8_grouped_linear."""

# pyright: strict

from typing import Any

import torch

from .. import load, variant

FP8_GROUPED_MAX_ROWS: int = 4


def _applies(plan: Any, rows: int = 0, **_: Any) -> bool:
    return bool(plan["kernels"] and rows <= FP8_GROUPED_MAX_ROWS and load() is not None)


@variant(
    devices=("cpu",),
    numerics="rounding",
    when=f"fused CPU GEMV reading the fp8 weight directly without dequantization, at most {FP8_GROUPED_MAX_ROWS} rows",
    applies=_applies,
    condition={"max_rows": FP8_GROUPED_MAX_ROWS, "require_external_fact": "cpu_kernels"},
)
def cpu_kernel(o: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Computes grouped fp8 linear using the fused CPU kernel."""
    mod: Any = load()
    if mod is None:
        raise RuntimeError("CPU kernels unavailable")
    res: torch.Tensor = mod.fp8_grouped_linear(o.contiguous(), weight.contiguous(), scale.contiguous())
    return res
