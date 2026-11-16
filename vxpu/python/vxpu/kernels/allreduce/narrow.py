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

"""Narrowed wire-format approximation variant for allreduce."""

import torch

from .. import variant


@variant(
    devices=("cpu", "cuda"),
    numerics="rounding",
    when="approximation: narrows wire format to bf16 and accumulates without fp32 promotion",
    condition={"require_external_fact": "approximate"},
)
def narrow(buffers: torch.Tensor) -> torch.Tensor:
    """Sums partials after narrowing wire precision to bfloat16.

    Today this narrows float32 inputs to bfloat16 and accumulates without float32
    promotion on stacked buffers on one device, standing in for an approximate
    cross-card reduction that trades precision for network bandwidth.
    It declares a catalog condition requiring an external fact, so the planner
    never selects it unless a kernel table explicitly names it.
    """
    ranks = buffers.shape[0]
    if ranks <= 1:
        return buffers[0].clone() if ranks == 1 else buffers.new_empty(buffers.shape[1:])
    if buffers.dtype == torch.float32:
        wire = buffers.bfloat16()
        acc: torch.Tensor = wire[0].float()
        for r in range(1, wire.shape[0]):
            acc = acc + wire[r].float()
        return acc.to(buffers.dtype)
    acc: torch.Tensor = buffers[0].clone()
    for r in range(1, buffers.shape[0]):
        acc = acc + buffers[r]
    return acc
