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

"""Unit tests for the Qwen3-MoE model adapter and tiny-config CI fixture."""

import os
import tempfile
import unittest
from typing import Any, cast

import torch
import transformers

from . import qwen3_moe
from .engine import Engine, SamplingParams
from .export import export_artifact
from .fixtures import tiny_qwen3_moe


def _engine_greedy(engine: Engine, session_id: str, prompt: list[int], max_new_tokens: int) -> tuple[list[int], list[torch.Tensor]]:
    """Runs greedy generation on an Engine session and captures tokens and logits."""
    tokens: list[int] = []
    logits_list: list[torch.Tensor] = []

    def sink(_pos: int, logits: torch.Tensor) -> None:
        logits_list.append(logits.detach().cpu().float().clone())

    gen: Any = getattr(engine, "generate")(
        session_id, prompt, max_new_tokens, SamplingParams(do_sample=False), logits_sink=sink
    )
    for chunk in cast(list[dict[str, Any]], list(gen)):
        tokens.extend(cast(list[int], chunk.get("token_ids", [])))
    return tokens, logits_list


def _eager_greedy(
    model: qwen3_moe.Qwen3MoeForCausalLM, prompt: list[int], max_new_tokens: int, device: str
) -> tuple[list[int], list[torch.Tensor]]:
    """Runs greedy prefill + decode on an eager Qwen3MoeForCausalLM."""
    tokens: list[int] = []
    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(len(prompt), dtype=torch.long, device=device)
        logits = model(ids, pos)
        step_logits = logits[0, -1]
        logits_list.append(step_logits.detach().cpu().float().clone())
        next_id = int(step_logits.argmax(-1).item())
        cur_pos = len(prompt)
        for _ in range(max_new_tokens):
            tokens.append(next_id)
            if len(tokens) == max_new_tokens:
                break
            logits = model(
                torch.tensor([[next_id]], dtype=torch.long, device=device),
                torch.tensor([cur_pos], dtype=torch.long, device=device),
            )
            cur_pos += 1
            step_logits = logits[0, -1]
            logits_list.append(step_logits.detach().cpu().float().clone())
            next_id = int(step_logits.argmax(-1).item())
    return tokens, logits_list


def _transformers_greedy(model_dir: str, prompt: list[int], max_new_tokens: int, device: str) -> list[int]:
    """Runs greedy generation through upstream transformers Qwen3MoeForCausalLM."""
    from_pretrained: Any = getattr(transformers.AutoModelForCausalLM, "from_pretrained")
    model: Any = from_pretrained(model_dir, dtype="auto").to(device).eval()
    with torch.no_grad():
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        out: torch.Tensor = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    tail = out[0, len(prompt) :]
    return [int(tail[i].item()) for i in range(int(tail.shape[0]))]


class Qwen3MoeAdapterTest(unittest.TestCase):
    """Verifies Qwen3-MoE export, Engine execution, INT4 layouts, and boundary isolation."""

    def test_bf16_export_and_engine_match_eager_and_transformers_cpu(self) -> None:
        """Verifies BF16 export + Engine matches both eager adapter and transformers on CPU."""
        self._verify_dense_variant(dtype=torch.bfloat16, device="cpu", check_transformers=True)

    def test_fp16_export_and_engine_match_eager_cpu(self) -> None:
        """Verifies FP16 export + Engine matches eager reference on CPU."""
        self._verify_dense_variant(dtype=torch.float16, device="cpu", check_transformers=False)

    def test_bf16_export_and_engine_match_eager_and_transformers_cuda(self) -> None:
        """Verifies BF16 export + Engine matches eager and transformers on CUDA when available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self._verify_dense_variant(dtype=torch.bfloat16, device="cuda", check_transformers=True)

    def _verify_dense_variant(self, *, dtype: torch.dtype, device: str, check_transformers: bool) -> None:
        prompt = [3, 17, 42, 91, 128, 7]
        max_new = 6
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = tiny_qwen3_moe.write(os.path.join(tmp, "model"), dtype=dtype, seed=21)
            art_dir = os.path.join(tmp, "artifact")
            _, binding = export_artifact(model_dir, art_dir, max_cache_len=64)

            self.assertGreater(len(binding["bound"]), 0)
            self.assertGreater(len(binding["stacked"]), 0)
            self.assertGreater(len(binding["state"]), 0)
            self.assertNotIn("quantization", binding)

            eager = qwen3_moe.load_eager(model_dir, max_cache_len=64, device=device)
            eager_tokens, eager_logits = _eager_greedy(eager, prompt, max_new, device)

            engine = Engine(art_dir, device=device)
            sid = engine.new_session()
            eng_tokens, eng_logits = _engine_greedy(engine, sid, prompt, max_new)
            self.assertEqual(eng_tokens, eager_tokens)
            for got, want in zip(eng_logits, eager_logits):
                torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)

            if check_transformers:
                tf_tokens = _transformers_greedy(model_dir, prompt, max_new, device)
                self.assertEqual(eng_tokens, tf_tokens)

            # Verify multi-turn suffix prefill reuses KV cache identically to fresh prompt.
            turn2_prompt = prompt + eng_tokens[:2] + [55, 66, 77]
            turn2_tokens, _ = _engine_greedy(engine, sid, turn2_prompt, 4)
            fresh_sid = engine.new_session()
            fresh_tokens, _ = _engine_greedy(engine, fresh_sid, turn2_prompt, 4)
            self.assertEqual(turn2_tokens, fresh_tokens)

    def test_int4_awq_export_and_engine_load_cleanly_cpu(self) -> None:
        """Verifies packed INT4 AWQ weights export, stay packed in binding, and match eager on CPU."""
        self._verify_int4_variant("awq", device="cpu")

    def test_int4_gptq_export_and_engine_load_cleanly_cpu(self) -> None:
        """Verifies packed INT4 GPTQ weights export, stay packed in binding, and match eager on CPU."""
        self._verify_int4_variant("gptq", device="cpu")

    def test_int4_awq_and_gptq_cuda(self) -> None:
        """Verifies packed INT4 AWQ and GPTQ export + Engine match eager on CUDA when available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self._verify_int4_variant("awq", device="cuda")
        self._verify_int4_variant("gptq", device="cuda")

    def _verify_int4_variant(self, quant_method: str, *, device: str) -> None:
        prompt = [11, 22, 33, 44, 55]
        max_new = 6
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = tiny_qwen3_moe.write(
                os.path.join(tmp, f"model_{quant_method}"),
                dtype=torch.bfloat16,
                quant_method=quant_method,
                seed=31,
            )
            art_dir = os.path.join(tmp, f"artifact_{quant_method}")
            _, binding = export_artifact(model_dir, art_dir, max_cache_len=64)

            quant_meta = cast(dict[str, Any], binding["quantization"])
            self.assertEqual(quant_meta["quant_method"], quant_method)
            self.assertEqual(quant_meta["bits"], 4)
            self.assertEqual(quant_meta["group_size"], tiny_qwen3_moe.GROUP_SIZE)
            self.assertGreater(quant_meta["int4_modules"], 0)

            # Packed int32 weights stay packed in bound and stacked tables.
            bound_qweights = [k for k in binding["bound"] if k.endswith(".qweight")]
            stacked_qweights = [k for k in binding["stacked"] if k.endswith("_qweight")]
            self.assertEqual(len(bound_qweights), tiny_qwen3_moe.NUM_LAYERS * 4)
            self.assertEqual(len(stacked_qweights), tiny_qwen3_moe.NUM_LAYERS * 2)
            for k in bound_qweights:
                self.assertEqual(binding["bound"][k]["dtype"], "I32")

            eager = qwen3_moe.load_eager(model_dir, max_cache_len=64, device=device)
            eager_tokens, eager_logits = _eager_greedy(eager, prompt, max_new, device)

            engine = Engine(art_dir, device=device)
            sid = engine.new_session()
            eng_tokens, eng_logits = _engine_greedy(engine, sid, prompt, max_new)
            self.assertEqual(eng_tokens, eager_tokens)
            for got, want in zip(eng_logits, eager_logits):
                torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)

    def test_pack_unpack_roundtrip_awq_and_gptq(self) -> None:
        """Verifies pack_int4 and unpack_int4 recover exact weights across full 0..15 nibbles."""
        gen = torch.Generator(device="cpu").manual_seed(99)
        k_dim, n_dim, group_size = 64, 32, 32
        n_groups = k_dim // group_size
        w_q = torch.randint(0, 16, (k_dim, n_dim), generator=gen, dtype=torch.int32)
        z_q = torch.randint(0, 16, (n_groups, n_dim), generator=gen, dtype=torch.int32)
        scales = torch.empty((n_groups, n_dim), dtype=torch.float32).uniform_(0.01, 0.1, generator=gen).to(torch.bfloat16)
        expected = (
            (w_q.reshape(n_groups, group_size, n_dim).float() - z_q.unsqueeze(1).float())
            * scales.unsqueeze(1).float()
        ).reshape(k_dim, n_dim)

        for layout in ("awq", "gptq"):
            qw, qz = qwen3_moe.pack_int4(w_q, z_q, layout)
            got = qwen3_moe.unpack_int4(qw, scales, qz, layout)
            torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)

    def test_tied_embeddings_and_dense_mlp_layer(self) -> None:
        """Verifies tie_word_embeddings and mlp_only_layers export and match eager."""
        prompt = [4, 8, 15, 16, 23]
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = tiny_qwen3_moe.write(
                os.path.join(tmp, "model"),
                dtype=torch.bfloat16,
                tie_word_embeddings=True,
                mlp_only_layers=(0,),
                seed=7,
            )
            art_dir = os.path.join(tmp, "artifact")
            export_artifact(model_dir, art_dir, max_cache_len=64)

            eager = qwen3_moe.load_eager(model_dir, max_cache_len=64, device="cpu")
            eager_tokens, _ = _eager_greedy(eager, prompt, 5, "cpu")
            engine = Engine(art_dir, device="cpu")
            eng_tokens, _ = _engine_greedy(engine, engine.new_session(), prompt, 5)
            self.assertEqual(eng_tokens, eager_tokens)

    def test_handles_leaves_fp8_to_transformers_and_rejects_invalid_quant(self) -> None:
        """Verifies FP8 qwen3_moe stays on the fp8 path and invalid INT4 configs fail clearly."""
        self.assertTrue(qwen3_moe.handles({"config": {"model_type": "qwen3_moe"}}))
        self.assertTrue(
            qwen3_moe.handles({"config": {"model_type": "qwen3_moe", "quantization_config": {"quant_method": "awq"}}})
        )
        self.assertFalse(
            qwen3_moe.handles({"config": {"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8"}}})
        )
        self.assertFalse(qwen3_moe.handles({"config": {"model_type": "llama"}}))

        base: dict[str, Any] = {
            "model_type": "qwen3_moe",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        }
        with self.assertRaises(ValueError):
            qwen3_moe.config_from_dict({**base, "quantization_config": {"quant_method": "gptq", "desc_act": True}})
        with self.assertRaises(ValueError):
            qwen3_moe.config_from_dict({**base, "quantization_config": {"quant_method": "awq", "bits": 8}})

    def test_model_boundary_has_no_qwen3_references_in_core_modules(self) -> None:
        """Verifies engine.py, ops.py, stepgraph.py, lower.py, and kernels/ stay Qwen3-free."""
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        core_files = [
            os.path.join(pkg_dir, "engine.py"),
            os.path.join(pkg_dir, "ops.py"),
            os.path.join(pkg_dir, "stepgraph.py"),
            os.path.join(pkg_dir, "lower.py"),
        ]
        kernels_dir = os.path.join(pkg_dir, "kernels")
        for root, _, files in os.walk(kernels_dir):
            for fname in files:
                if fname.endswith(".py"):
                    core_files.append(os.path.join(root, fname))

        for path in core_files:
            with open(path) as f:
                text = f.read().lower()
            self.assertNotIn("qwen3", text, f"unexpected Qwen3 reference in {path}")


if __name__ == "__main__":
    unittest.main()
