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

"""Variant gpu_triton of vxpu::read."""

import torch

from .. import variant

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None and tl is not None:
    @triton.jit
    def _read_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        v = tl.load(x_ptr + offsets, mask=mask)
        s = tl.sum(v.to(tl.int64))
        tl.atomic_add(out_ptr, s)
else:
    _read_kernel = None


def _triton_available(plan: object, **_: object) -> bool:
    return triton is not None and torch.cuda.is_available()


@variant(
    devices=("cuda",),
    numerics="bit-identical",
    when="sequential memory read with Triton, streaming through memory bus",
    applies=_triton_available,
)
def gpu_triton(x: torch.Tensor) -> torch.Tensor:
    """Streams x sequentially through the memory bus using Triton."""
    if _read_kernel is None:
        raise RuntimeError("Triton is not available")
    block_size = 2048
    num_warps = 8
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    out = torch.zeros((), dtype=torch.int64, device=x.device)
    with torch.cuda.device(x.device):
        _read_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out
