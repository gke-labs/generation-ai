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

"""Input builder for vxpu::read.

vxpu::read measures the card's memory read rate under the kernel contract
rather than serving model inference. It defines no SCORE_SHAPES because
speedup over the reference for a pure read is not a kernel getting faster.
"""

# pyright: strict

import torch

SCORE_SHAPES = None


def inputs(shapes: str, device: str, seed: int = 0) -> tuple[torch.Tensor]:
    """Builds a flat byte tensor of the requested size in bytes."""
    n_bytes = int(shapes)
    if n_bytes <= 0:
        raise ValueError(f"read shape bytes must be positive, got {shapes!r}")
    x = torch.ones(n_bytes, dtype=torch.uint8, device=device)
    return (x,)
