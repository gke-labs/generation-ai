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

"""One-shot cross-card sum variant for allreduce."""

import torch

from .. import variant


@variant(
    devices=("cpu", "cuda"),
    numerics="rounding",
    when="single-kernel reduction over the rank axis (dim=0) with float32 accumulation",
)
def one_shot(buffers: torch.Tensor) -> torch.Tensor:
    """Reduces the rank dimension in a single reduction kernel with float32 accumulation.

    Today this runs as a single tensor reduction across the stacked rank buffers
    on one device, standing in for the cross-card one-shot exchange and sum.
    Because parallel reduction reassociates additions across threads, it rounds
    differently from sequential rank order and declares the rounding numerics class.
    """
    if buffers.shape[0] <= 1:
        return buffers[0].clone() if buffers.shape[0] == 1 else buffers.new_empty(buffers.shape[1:])
    return torch.sum(buffers.float(), dim=0).to(buffers.dtype)
