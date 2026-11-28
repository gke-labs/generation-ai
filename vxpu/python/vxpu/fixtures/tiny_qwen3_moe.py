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

"""Synthetic tiny-config Qwen3-MoE checkpoint fixture for fast CI tests.

Generates a deterministic 2-layer, 8-expert (top-2) Qwen3-MoE safetensors
checkpoint using standard Hugging Face tensor naming in BF16, FP16, packed
INT4 AWQ, or packed INT4 GPTQ format without downloading any external weights.
"""

import json
import os
import sys
from typing import Any

import safetensors.torch
import torch

VOCAB_SIZE: int = 256
HIDDEN_SIZE: int = 128
INTERMEDIATE_SIZE: int = 256
MOE_INTERMEDIATE_SIZE: int = 64
NUM_LAYERS: int = 2
NUM_ATTENTION_HEADS: int = 4
NUM_KEY_VALUE_HEADS: int = 2
HEAD_DIM: int = 32
NUM_EXPERTS: int = 8
NUM_EXPERTS_PER_TOK: int = 2
GROUP_SIZE: int = 32
_AWQ_PACK_ORDER: tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"unsupported fixture dtype {dtype}")


def pack_int4(
    weight_q: torch.Tensor,
    zero_q: torch.Tensor,
    layout: str,
    checkpoint_format: str = "gptq",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs unsigned 4-bit weights [K, N] and zero points [n_groups, N] into AWQ or GPTQ int32 tensors.

    AWQ packs 8 columns along N per int32 in interleave order (0, 2, 4, 6, 1, 3, 5, 7)
    with raw zero points. GPTQ packs 8 rows along K into qweight [K // 8, N] and 8
    columns along N into qzeros [n_groups, N // 8] in little-endian nibble order,
    subtracting 1 from zero points for standard v1 format (checkpoint_format != "gptq_v2").
    """
    k_dim, n_dim, n_groups = int(weight_q.shape[0]), int(weight_q.shape[1]), int(zero_q.shape[0])
    if k_dim % 8 != 0 or n_dim % 8 != 0:
        raise ValueError(f"pack_int4 requires K ({k_dim}) and N ({n_dim}) divisible by 8")
    shifts = torch.arange(0, 32, 4, device=weight_q.device, dtype=torch.int32)
    w_i = weight_q.to(torch.int32) & 0xF
    z_src = zero_q.to(torch.int32)
    if layout == "gptq" and checkpoint_format != "gptq_v2":
        if bool((z_src < 1).any().item()):
            raise ValueError("pack_int4: GPTQ v1 zero points must be >= 1")
        z_src = z_src - 1
    z_i = z_src & 0xF
    if layout == "awq":
        order = torch.tensor(_AWQ_PACK_ORDER, device=weight_q.device, dtype=torch.long)
        qw = torch.sum(w_i.reshape(k_dim, n_dim // 8, 8).index_select(2, order) << shifts.view(1, 1, 8), dim=2, dtype=torch.int32)
        qz = torch.sum(z_i.reshape(n_groups, n_dim // 8, 8).index_select(2, order) << shifts.view(1, 1, 8), dim=2, dtype=torch.int32)
        return qw.contiguous(), qz.contiguous()
    if layout == "gptq":
        qw = torch.sum(w_i.reshape(k_dim // 8, 8, n_dim) << shifts.view(1, 8, 1), dim=1, dtype=torch.int32)
        qz = torch.sum(z_i.reshape(n_groups, n_dim // 8, 8) << shifts.view(1, 1, 8), dim=2, dtype=torch.int32)
        return qw.contiguous(), qz.contiguous()
    raise ValueError(f"pack_int4: unknown INT4 layout {layout!r}")


def _rand_weight(shape: tuple[int, ...], gen: torch.Generator, dtype: torch.dtype, std: float = 0.04) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * std).to(dtype)


def _add_linear(
    state: dict[str, torch.Tensor],
    prefix: str,
    in_features: int,
    out_features: int,
    gen: torch.Generator,
    dtype: torch.dtype,
    quant_method: str | None,
    group_size: int,
    checkpoint_format: str,
) -> None:
    """Adds either a floating-point or packed INT4 linear projection to state."""
    if quant_method is None:
        state[f"{prefix}.weight"] = _rand_weight((out_features, in_features), gen, dtype)
        return
    n_groups = in_features // group_size
    w_q = torch.randint(0, 16, (in_features, out_features), generator=gen, dtype=torch.int32)
    z_q = torch.randint(4, 12, (n_groups, out_features), generator=gen, dtype=torch.int32)
    qweight, qzeros = pack_int4(w_q, z_q, quant_method, checkpoint_format=checkpoint_format)
    scales = torch.empty((n_groups, out_features), dtype=torch.float32).uniform_(0.005, 0.03, generator=gen).to(dtype)
    state[f"{prefix}.qweight"] = qweight
    state[f"{prefix}.qzeros"] = qzeros
    state[f"{prefix}.scales"] = scales
    if quant_method == "gptq":
        state[f"{prefix}.g_idx"] = torch.arange(in_features, dtype=torch.int32) // group_size


def write(
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    quant_method: str | None = None,
    checkpoint_format: str = "gptq",
    group_size: int = GROUP_SIZE,
    seed: int = 17,
    tie_word_embeddings: bool = False,
    mlp_only_layers: tuple[int, ...] = (),
) -> str:
    """Writes a tiny deterministic Qwen3-MoE checkpoint directory and returns its path."""
    if quant_method not in (None, "awq", "gptq"):
        raise ValueError(f"unsupported quant_method {quant_method!r}")
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    config: dict[str, Any] = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "hidden_act": "silu",
        "attention_bias": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": list(mlp_only_layers),
        "tie_word_embeddings": tie_word_embeddings,
        "max_position_embeddings": 1024,
        "torch_dtype": _dtype_name(dtype),
        "dtype": _dtype_name(dtype),
    }
    if quant_method == "awq":
        config["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": group_size,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["lm_head", "mlp.gate"],
        }
    elif quant_method == "gptq":
        qc: dict[str, Any] = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": group_size,
            "desc_act": False,
            "sym": False,
            "modules_to_not_convert": ["lm_head", "mlp.gate"],
        }
        if checkpoint_format != "gptq":
            qc["checkpoint_format"] = checkpoint_format
        config["quantization_config"] = qc

    state: dict[str, torch.Tensor] = {}
    state["model.embed_tokens.weight"] = _rand_weight((VOCAB_SIZE, HIDDEN_SIZE), gen, dtype, std=0.1)
    q_dim = NUM_ATTENTION_HEADS * HEAD_DIM
    kv_dim = NUM_KEY_VALUE_HEADS * HEAD_DIM

    for layer in range(NUM_LAYERS):
        lp = f"model.layers.{layer}"
        state[f"{lp}.input_layernorm.weight"] = (1.0 + _rand_weight((HIDDEN_SIZE,), gen, dtype, std=0.05)).to(dtype)
        state[f"{lp}.post_attention_layernorm.weight"] = (1.0 + _rand_weight((HIDDEN_SIZE,), gen, dtype, std=0.05)).to(dtype)
        state[f"{lp}.self_attn.q_norm.weight"] = (1.0 + _rand_weight((HEAD_DIM,), gen, dtype, std=0.05)).to(dtype)
        state[f"{lp}.self_attn.k_norm.weight"] = (1.0 + _rand_weight((HEAD_DIM,), gen, dtype, std=0.05)).to(dtype)
        _add_linear(state, f"{lp}.self_attn.q_proj", HIDDEN_SIZE, q_dim, gen, dtype, quant_method, group_size, checkpoint_format)
        _add_linear(state, f"{lp}.self_attn.k_proj", HIDDEN_SIZE, kv_dim, gen, dtype, quant_method, group_size, checkpoint_format)
        _add_linear(state, f"{lp}.self_attn.v_proj", HIDDEN_SIZE, kv_dim, gen, dtype, quant_method, group_size, checkpoint_format)
        _add_linear(state, f"{lp}.self_attn.o_proj", q_dim, HIDDEN_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)

        if layer in mlp_only_layers:
            _add_linear(state, f"{lp}.mlp.gate_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)
            _add_linear(state, f"{lp}.mlp.up_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)
            _add_linear(state, f"{lp}.mlp.down_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)
        else:
            # Keep router gates unquantized, matching real AWQ/GPTQ Qwen3-MoE checkpoints.
            state[f"{lp}.mlp.gate.weight"] = _rand_weight((NUM_EXPERTS, HIDDEN_SIZE), gen, dtype, std=0.1)
            for expert in range(NUM_EXPERTS):
                ep = f"{lp}.mlp.experts.{expert}"
                _add_linear(state, f"{ep}.gate_proj", HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)
                _add_linear(state, f"{ep}.up_proj", HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)
                _add_linear(state, f"{ep}.down_proj", MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, gen, dtype, quant_method, group_size, checkpoint_format)

    state["model.norm.weight"] = (1.0 + _rand_weight((HIDDEN_SIZE,), gen, dtype, std=0.05)).to(dtype)
    if not tie_word_embeddings:
        state["lm_head.weight"] = _rand_weight((VOCAB_SIZE, HIDDEN_SIZE), gen, dtype, std=0.1)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    save_file_fn: Any = getattr(safetensors.torch, "save_file")
    save_file_fn(state, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"do_sample": False}, f)
    return out_dir


if __name__ == "__main__":
    quant = sys.argv[2] if len(sys.argv) > 2 else None
    print(write(sys.argv[1], quant_method=quant))
