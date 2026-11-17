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

"""Chunked cross-card sum variant for allreduce."""

import torch

from .. import variant


@variant(
    devices=("cpu", "cuda"),
    numerics="bit-identical",
    when="chunked reduction over rank axis standing in for reduce-scatter and all-gather",
)
def scatter_gather(buffers: torch.Tensor) -> torch.Tensor:
    """Computes cross-card sum via chunked rank reduction and concatenation.

    Today this partitions the last dimension of the stacked buffers into rank
    chunks, reduces each chunk across ranks with float32 accumulation, and
    concatenates the result, standing in for reduce-scatter and all-gather across cards.
    """
    ranks = buffers.shape[0]
    if ranks <= 1:
        return buffers[0].clone() if ranks == 1 else buffers.new_empty(buffers.shape[1:])
    dim = buffers.shape[-1]
    if buffers.ndim < 2 or dim < ranks or dim % ranks != 0:
        acc: torch.Tensor = buffers[0].float()
        for r in range(1, ranks):
            acc = acc + buffers[r].float()
        return acc.to(buffers.dtype)

    chunks = torch.chunk(buffers, ranks, dim=-1)
    gathered: list[torch.Tensor] = []
    for r in range(ranks):
        c = chunks[r]
        c_acc: torch.Tensor = c[0].float()
        for i in range(1, ranks):
            c_acc = c_acc + c[i].float()
        gathered.append(c_acc.to(buffers.dtype))
    return torch.cat(gathered, dim=-1)
