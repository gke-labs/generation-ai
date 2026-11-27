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

"""Unit tests for GPU expert mover lifetime and fp4_experts_decode Triton kernel numerics."""

import gc
import threading
import time
from typing import Any
import unittest
import weakref

import torch

from . import deepseek_v41, experts_gpu, plan, triton_ops


class ExpertsGPUTest(unittest.TestCase):
    """Verifies Mover weakref lifecycle and fp4_experts_decode kernel output against reference."""

    def test_mover_lifetime(self) -> None:
        """Mover is garbage-collected when dropped and its worker thread exits cleanly."""
        mover_cls: Any = getattr(experts_gpu, "Mover")
        mover: Any = mover_cls(2, 0.25)
        ref = weakref.ref(mover)
        del mover
        gc.collect()
        self.assertIsNone(ref(), "Mover still referenced after drop")

        deadline = time.time() + 5.0
        while any(t.name == "vxpu-expert-moves" for t in threading.enumerate()) and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(
            any(t.name == "vxpu-expert-moves" for t in threading.enumerate()),
            "Mover worker thread outlived the Mover instance",
        )

    def test_fp4_experts_decode_matches_reference(self) -> None:
        """triton_ops.fp4_experts_decode matches DeepSeek MoE dequant/FFN reference on random fp4 banks."""
        triton_mod: Any = triton_ops
        if not torch.cuda.is_available() or not triton_mod.available():
            self.skipTest("needs CUDA and Triton")

        try:
            ds: Any = getattr(deepseek_v41, "_fork")()
        except (ImportError, ModuleNotFoundError) as e:
            if plan.require_model_source():
                raise
            self.skipTest(f"deepseek_v41 model source unavailable: {e}")

        k_dim, i_dim, n_slots, dev = 256, 128, 4, "cuda"
        g = torch.Generator(device=dev).manual_seed(0)

        def bank(n_out: int, n_in: int) -> tuple[torch.Tensor, torch.Tensor]:
            w = torch.randint(0, 256, (n_slots, n_out, n_in // 2), dtype=torch.uint8, device=dev, generator=g)
            sc = torch.randint(120, 125, (n_slots, n_out, n_in // 32), dtype=torch.uint8, device=dev, generator=g)
            return w, sc.view(torch.float8_e8m0fnu)

        (w1, s1), (w3, s3), (w2, s2) = bank(i_dim, k_dim), bank(i_dim, k_dim), bank(k_dim, i_dim)
        x = triton_mod.act_quant_dequant(torch.randn(1, k_dim, device=dev, generator=g).to(torch.bfloat16))
        weights = torch.rand(1, 3, device=dev, generator=g)
        slots = torch.tensor([[2, -1, 0]], device=dev)
        y = triton_mod.fp4_experts_decode(x, weights, slots, w1, s1, w3, s3, w2, s2, 10.0)

        class _RefModule:
            swiglu_limit = 10.0

            def _quantized_input(self, t: torch.Tensor) -> torch.Tensor:
                return ds.MoE._quantized_input(self, t)

        ref = torch.zeros(1, k_dim, device=dev)
        for k_idx in range(slots.shape[1]):
            sl = int(slots[0, k_idx].item())
            if sl < 0:
                continue
            deq = [ds.MoE._dequant(w[sl : sl + 1], sc[sl : sl + 1]) for w, sc in ((w1, s1), (w3, s3), (w2, s2))]
            gate = torch.bmm(deq[0], x.view(1, k_dim, 1)).squeeze(-1)
            up = torch.bmm(deq[1], x.view(1, k_dim, 1)).squeeze(-1)
            h = ds.MoE._ffn(_RefModule(), gate, up, weights[0, k_idx : k_idx + 1])
            ref[0] += torch.bmm(deq[2], h.unsqueeze(-1)).squeeze(-1).float()[0]

        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        self.assertLessEqual(rel, 1e-2, f"fp4_experts_decode differs from reference (relative {rel:.3g})")


if __name__ == "__main__":
    unittest.main()
