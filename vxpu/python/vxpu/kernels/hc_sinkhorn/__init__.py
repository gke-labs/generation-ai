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

"""Input builder for vxpu::hc_sinkhorn."""

# pyright: strict

import torch

SCORE_SHAPES: tuple[str, ...] = ("tokens=1 hc=4",)


def inputs(
    shapes: str,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
    """Builds synthetic inputs for hc_sinkhorn.

    Returns (mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps).
    Stands for the hyper-connection mixing projection (TransformerBlock.hc_mixes)
    in decode (tokens=1, hc_mult=4, sinkhorn_iters=20, eps=1e-6) in the
    released model.

    Parses shapes in 'tokens={tokens} hc={hc_mult}' format.
    """
    tokens = 1
    hc_mult = 4
    for part in shapes.split():
        if part.startswith("tokens="):
            tokens = int(part.removeprefix("tokens="))
        elif part.startswith("hc="):
            hc_mult = int(part.removeprefix("hc="))
        else:
            raise ValueError(f"unrecognized hc_sinkhorn shape token {part!r} in {shapes!r}")
    if tokens <= 0 or hc_mult <= 0:
        raise ValueError(f"tokens ({tokens}) and hc ({hc_mult}) must be positive")

    mix_hc = (2 + hc_mult) * hc_mult
    sinkhorn_iters = 20
    eps = 1e-06

    gen = torch.Generator(device="cpu").manual_seed(seed)
    mixes = torch.randn((tokens, mix_hc), generator=gen, dtype=torch.float32).to(device)
    hc_scale = torch.randn(3, generator=gen, dtype=torch.float32).to(device)
    hc_base = torch.randn(mix_hc, generator=gen, dtype=torch.float32).to(device)

    return mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
