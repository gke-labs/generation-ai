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

"""Triton kernel variant for vxpu::index_topk."""

from typing import Any
import torch

from .. import variant
from ...ops import INDEX_TOPK_MAX_ROWS


@variant(
    devices=("cuda",),
    numerics="rounding",
    when="Triton kernels: score reduction and top-k selection in one pass over T",
    applies=lambda plan, rows=0, **_: plan["triton"] and rows <= INDEX_TOPK_MAX_ROWS,
    lowerable=True,
    condition={"max_rows": INDEX_TOPK_MAX_ROWS},
)
def triton(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    index_topk: int,
    offset: int = 0,
    tk: Any = None,
) -> torch.Tensor:
    """Triton implementation of score reduction and top-k selection."""
    if tk is not None:
        return tk.index_topk(q, index_k, weights, compress_lens, candidates, index_topk, offset)
    from ... import triton_ops
    return triton_ops.index_topk(q, index_k, weights, compress_lens, candidates, index_topk, offset)
