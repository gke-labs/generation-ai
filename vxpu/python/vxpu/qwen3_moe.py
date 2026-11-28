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

"""Thin model adapter for Qwen3-MoE (Qwen3-30B-A3B architecture).

Keeps Qwen3-MoE architecture, Hugging Face safetensors naming, and packed INT4
(AWQ and GPTQ) layout details inside the export boundary so engine.py, ops.py,
lower.py, stepgraph.py, kernels/, and the Go planner stay model-agnostic.
"""

from dataclasses import dataclass
import re
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE: str = "qwen3_moe"
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16, "BF16": torch.bfloat16,
    "float16": torch.float16, "F16": torch.float16,
    "float32": torch.float32, "F32": torch.float32,
}
_EXPERT_DENSE_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_INT4_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales)$")


@dataclass(frozen=True)
class Int4QuantConfig:
    """Packed 4-bit weight quantization settings parsed from config.json."""

    quant_method: str
    group_size: int = 128
    bits: int = 4
    checkpoint_format: str = "gptq"


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """Architecture settings parsed from a Qwen3-MoE config.json."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16
    quant: Int4QuantConfig | None = None


def handles(manifest: dict[str, Any]) -> bool:
    """Returns whether this adapter handles the manifest's model configuration."""
    config = cast(dict[str, Any], manifest.get("config") or {})
    if config.get("model_type") != MODEL_TYPE:
        return False
    qc_raw = config.get("quantization_config")
    if isinstance(qc_raw, dict) and cast(dict[str, Any], qc_raw).get("quant_method") == "fp8":
        return False
    return True


def config_from_dict(raw: dict[str, Any]) -> Qwen3MoeConfig:
    """Parses a Hugging Face config.json dictionary into Qwen3MoeConfig."""
    hidden_size = int(raw["hidden_size"])
    num_heads = int(raw["num_attention_heads"])
    rope_raw = raw.get("rope_parameters")
    rope_theta = float(
        cast(dict[str, Any], rope_raw).get("rope_theta", raw.get("rope_theta", 10000.0))
        if isinstance(rope_raw, dict) else raw.get("rope_theta", 10000.0)
    )
    dtype_name = str(raw.get("dtype") or raw.get("torch_dtype") or "bfloat16")
    if dtype_name not in _DTYPES:
        raise ValueError(f"qwen3_moe: unsupported dtype {dtype_name!r}")

    quant: Int4QuantConfig | None = None
    qc_raw = raw.get("quantization_config")
    if isinstance(qc_raw, dict):
        qc = cast(dict[str, Any], qc_raw)
        method = str(qc.get("quant_method") or "").lower()
        if method in ("awq", "gptq"):
            bits = int(qc.get("bits", 4))
            if bits != 4:
                raise ValueError(f"qwen3_moe: expected 4-bit quantization, got bits={bits}")
            if bool(qc.get("desc_act", False)):
                raise ValueError("qwen3_moe: GPTQ desc_act=True is not supported")
            group_size = int(qc.get("group_size", 128))
            if group_size <= 0 or group_size % 8 != 0:
                raise ValueError(f"qwen3_moe: group_size ({group_size}) must be a positive multiple of 8")
            ckpt_fmt = str(qc.get("checkpoint_format") or ("gptq" if method == "gptq" else "")).lower()
            quant = Int4QuantConfig(quant_method=method, group_size=group_size, bits=bits, checkpoint_format=ckpt_fmt)
        elif method:
            raise ValueError(f"qwen3_moe: unsupported quantization method {method!r}")

    return Qwen3MoeConfig(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(raw.get("intermediate_size", hidden_size * 3)),
        moe_intermediate_size=int(raw.get("moe_intermediate_size", hidden_size)),
        num_hidden_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw.get("head_dim") or (hidden_size // num_heads)),
        num_experts=int(raw.get("num_experts", raw.get("num_local_experts", 0))),
        num_experts_per_tok=int(raw.get("num_experts_per_tok", 1)),
        norm_topk_prob=bool(raw.get("norm_topk_prob", False)),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_theta=rope_theta,
        attention_bias=bool(raw.get("attention_bias", False)),
        decoder_sparse_step=int(raw.get("decoder_sparse_step", 1)),
        mlp_only_layers=tuple(int(i) for i in (raw.get("mlp_only_layers") or ())),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        dtype=_DTYPES[dtype_name],
        quant=quant,
    )


def unpack_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    layout: str,
    checkpoint_format: str = "gptq",
) -> torch.Tensor:
    """Unpacks and dequantizes 2D or 3D packed INT4 weights into float32 [..., K, N].

    Standard AutoGPTQ / GPTQModel v1 checkpoints (checkpoint_format == "gptq")
    store zero_point - 1 in qzeros, so this function adds 1 back when unpacking
    GPTQ v1 while leaving AWQ and gptq_v2 zero points unchanged.
    """
    if layout == "awq":
        even = torch.arange(0, 16, 4, device=qweight.device, dtype=torch.int32)
        odd = torch.arange(16, 32, 4, device=qweight.device, dtype=torch.int32)
        shifts = torch.stack((even, odd), dim=1).reshape(8)
        k_dim = int(qweight.shape[-2])
    elif layout == "gptq":
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
        k_dim = int(qweight.shape[-2]) * 8
    else:
        raise ValueError(f"qwen3_moe: unknown INT4 layout {layout!r}")
    prefix = qweight.shape[:-2]
    n_groups, n_dim = int(scales.shape[-2]), int(scales.shape[-1])
    w_q = (
        ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(*prefix, k_dim, n_dim)
        if layout == "awq"
        else ((qweight.unsqueeze(-2) >> shifts.view(8, 1)) & 0xF).reshape(*prefix, k_dim, n_dim)
    )
    z_q = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(*prefix, n_groups, n_dim)
    if layout == "gptq" and checkpoint_format != "gptq_v2":
        z_q = z_q + 1
    w_groups = w_q.reshape(*prefix, n_groups, k_dim // n_groups, n_dim).float()
    return ((w_groups - z_q.unsqueeze(-2).float()) * scales.unsqueeze(-2).float()).reshape(*prefix, k_dim, n_dim)


class RMSNorm(nn.Module):
    """Applies root mean square normalization in float32."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).to(hidden_states.dtype)


class Int4Linear(nn.Module):
    """Linear projection backed by packed INT4 AWQ or GPTQ safetensors weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        layout: str,
        scales_dtype: torch.dtype,
        checkpoint_format: str = "gptq",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.checkpoint_format = checkpoint_format
        n_groups = in_features // group_size
        qw_shape = (in_features, out_features // 8) if layout == "awq" else (in_features // 8, out_features)
        self.qweight = nn.Parameter(torch.empty(qw_shape, dtype=torch.int32), requires_grad=False)
        self.qzeros = nn.Parameter(torch.empty((n_groups, out_features // 8), dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.empty((n_groups, out_features), dtype=scales_dtype), requires_grad=False)
        self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features, dtype=scales_dtype), requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = unpack_int4(self.qweight, self.scales, self.qzeros, self.layout, self.checkpoint_format)
        out = torch.matmul(x.float(), weight).to(x.dtype)
        return out + self.bias.to(x.dtype) if self.bias is not None else out


def _int4_spec(
    prefix: str, in_features: int, tensors: dict[str, Any], quant: Int4QuantConfig | None
) -> tuple[str, int, torch.dtype, str] | None:
    qw_meta = tensors.get(f"{prefix}.qweight")
    if qw_meta is None:
        return None
    sc_meta = tensors[f"{prefix}.scales"]
    qw_shape = cast(list[int], qw_meta["shape"])
    layout = "awq" if qw_shape[0] == in_features else "gptq"
    if quant is not None and qw_shape[0] == (in_features if quant.quant_method == "awq" else in_features // 8):
        layout = quant.quant_method
    ckpt_fmt = quant.checkpoint_format if quant is not None else ("gptq" if layout == "gptq" else "")
    return layout, in_features // int(cast(list[int], sc_meta["shape"])[0]), _DTYPES[str(sc_meta["dtype"])], ckpt_fmt


def _make_linear(prefix: str, in_f: int, out_f: int, bias: bool, dtype: torch.dtype, tensors: dict[str, Any], quant: Int4QuantConfig | None) -> nn.Module:
    spec = _int4_spec(prefix, in_f, tensors, quant)
    if spec is not None:
        return Int4Linear(in_f, out_f, spec[1], spec[0], spec[2], checkpoint_format=spec[3], bias=bias)
    return nn.Linear(in_f, out_f, bias=bias, dtype=dtype)


class Qwen3MoeAttention(nn.Module):
    """Grouped-query self-attention with per-head QK-norm, RoPE, and static KV cache."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(self, cfg: Qwen3MoeConfig, layer_idx: int, max_cache_len: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.head_dim, self.scaling = cfg.head_dim, cfg.head_dim**-0.5
        p, q_dim, kv_dim = f"model.layers.{layer_idx}.self_attn", cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = _make_linear(f"{p}.q_proj", cfg.hidden_size, q_dim, cfg.attention_bias, cfg.dtype, tensors, cfg.quant)
        self.k_proj = _make_linear(f"{p}.k_proj", cfg.hidden_size, kv_dim, cfg.attention_bias, cfg.dtype, tensors, cfg.quant)
        self.v_proj = _make_linear(f"{p}.v_proj", cfg.hidden_size, kv_dim, cfg.attention_bias, cfg.dtype, tensors, cfg.quant)
        self.o_proj = _make_linear(f"{p}.o_proj", q_dim, cfg.hidden_size, cfg.attention_bias, cfg.dtype, tensors, cfg.quant)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, cfg.dtype)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, cfg.dtype)
        self.register_buffer("k_cache", torch.zeros(1, self.num_kv_heads, max_cache_len, self.head_dim, dtype=cfg.dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(1, self.num_kv_heads, max_cache_len, self.head_dim, dtype=cfg.dtype), persistent=False)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_position: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        half = self.head_dim // 2
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * c) + (torch.cat((-q[..., half:], q[..., :half]), dim=-1) * s)
        k = (k * c) + (torch.cat((-k[..., half:], k[..., :half]), dim=-1) * s)
        self.k_cache.index_copy_(2, cache_position, k)
        self.v_cache.index_copy_(2, cache_position, v)
        keys, values = self.k_cache, self.v_cache
        if self.num_kv_groups != 1:
            cache_len = self.k_cache.shape[2]
            keys = keys.unsqueeze(2).expand(bsz, self.num_kv_heads, self.num_kv_groups, cache_len, self.head_dim).reshape(bsz, self.num_heads, cache_len, self.head_dim)
            values = values.unsqueeze(2).expand(bsz, self.num_kv_heads, self.num_kv_groups, cache_len, self.head_dim).reshape(bsz, self.num_heads, cache_len, self.head_dim)
        weights = F.softmax(torch.matmul(q, keys.transpose(2, 3)) * self.scaling + causal_mask, dim=-1, dtype=torch.float32).to(q.dtype)
        return cast(torch.Tensor, self.o_proj(torch.matmul(weights, values).transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)))


class Qwen3MoeMLP(nn.Module):
    """Dense SwiGLU MLP used on non-sparse layers."""

    def __init__(self, cfg: Qwen3MoeConfig, layer_idx: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        p = f"model.layers.{layer_idx}.mlp"
        self.gate_proj = _make_linear(f"{p}.gate_proj", cfg.hidden_size, cfg.intermediate_size, False, cfg.dtype, tensors, cfg.quant)
        self.up_proj = _make_linear(f"{p}.up_proj", cfg.hidden_size, cfg.intermediate_size, False, cfg.dtype, tensors, cfg.quant)
        self.down_proj = _make_linear(f"{p}.down_proj", cfg.intermediate_size, cfg.hidden_size, False, cfg.dtype, tensors, cfg.quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def _grouped_routing(
    hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts routed token-expert pairs by expert and builds int32 grouped_mm offsets."""
    top_k = top_k_index.shape[1]
    expert_ids_g, perm = torch.sort(top_k_index.reshape(-1))
    tok_idx = perm // top_k
    x_g = hidden_states.index_select(0, tok_idx)
    weights_g = top_k_weights.reshape(-1).index_select(0, perm)
    counts = torch.zeros(num_experts, device=expert_ids_g.device, dtype=torch.int64)
    counts.scatter_add_(0, expert_ids_g, torch.ones_like(expert_ids_g))
    return x_g, tok_idx, weights_g, torch.cumsum(counts, dim=0, dtype=torch.int32)


def _grouped_mm_bf16(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Runs grouped_mm in bfloat16 (required across meta, CPU, and L4 CUDA) and restores x.dtype."""
    mm_fn: Any = getattr(F, "grouped_mm")
    return cast(torch.Tensor, mm_fn(x.to(torch.bfloat16), weight.to(torch.bfloat16), offs=offsets)).to(x.dtype)


def _eval_batched_experts(
    hidden_states: torch.Tensor, top_k_weights: torch.Tensor, w_gu: torch.Tensor, w_dn: torch.Tensor
) -> torch.Tensor:
    """Evaluates gathered SwiGLU expert weights [T*K, H, 2I] and [T*K, I, H] for 1-token decode."""
    tokens, hidden = hidden_states.shape
    top_k = top_k_weights.shape[1]
    x = hidden_states.repeat_interleave(top_k, dim=0)
    gu = torch.bmm(x.unsqueeze(1), w_gu.to(x.dtype)).squeeze(1)
    gate, up = gu.chunk(2, dim=-1)
    out = torch.bmm((F.silu(gate) * up).unsqueeze(1), w_dn.to(x.dtype)).squeeze(1)
    return (out.view(tokens, top_k, hidden) * top_k_weights.unsqueeze(-1)).sum(dim=1)


class DenseMoeExperts(nn.Module):
    """Stacked 3D BF16/FP16 expert bank using grouped_mm on prefill and bmm on decode."""

    def __init__(self, cfg: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = cfg.num_experts
        self.grouped = False
        self.gate_up_proj = nn.Parameter(torch.empty(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size, dtype=cfg.dtype), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, dtype=cfg.dtype), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        if self.grouped:
            x_g, tok_idx, weights_g, offsets = _grouped_routing(hidden_states, top_k_index, top_k_weights, self.num_experts)
            gate, up = _grouped_mm_bf16(x_g, self.gate_up_proj.transpose(1, 2), offsets).chunk(2, dim=-1)
            out_g = _grouped_mm_bf16((F.silu(gate) * up).contiguous(), self.down_proj.transpose(1, 2), offsets)
            out = torch.zeros_like(hidden_states)
            out.index_add_(0, tok_idx, out_g * weights_g.unsqueeze(-1))
            return out
        idx = top_k_index.reshape(-1)
        return _eval_batched_experts(hidden_states, top_k_weights, self.gate_up_proj.index_select(0, idx).transpose(1, 2), self.down_proj.index_select(0, idx).transpose(1, 2))


class Int4MoeExperts(nn.Module):
    """Stacked 3D packed INT4 expert bank using grouped_mm on prefill and active-only gather on decode."""

    def __init__(
        self, cfg: Qwen3MoeConfig, layout: str, group_size: int, scales_dtype: torch.dtype, checkpoint_format: str
    ) -> None:
        super().__init__()
        self.num_experts = cfg.num_experts
        self.layout = layout
        self.checkpoint_format = checkpoint_format
        self.grouped = False
        e, h, i = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.gate_up_qweight = nn.Parameter(torch.empty((e, h, (2 * i) // 8) if layout == "awq" else (e, h // 8, 2 * i), dtype=torch.int32), requires_grad=False)
        self.gate_up_qzeros = nn.Parameter(torch.empty((e, h // group_size, (2 * i) // 8), dtype=torch.int32), requires_grad=False)
        self.gate_up_scales = nn.Parameter(torch.empty((e, h // group_size, 2 * i), dtype=scales_dtype), requires_grad=False)
        self.down_qweight = nn.Parameter(torch.empty((e, i, h // 8) if layout == "awq" else (e, i // 8, h), dtype=torch.int32), requires_grad=False)
        self.down_qzeros = nn.Parameter(torch.empty((e, i // group_size, h // 8), dtype=torch.int32), requires_grad=False)
        self.down_scales = nn.Parameter(torch.empty((e, i // group_size, h), dtype=scales_dtype), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        if self.grouped:
            # Unpack each projection bank once across num_experts so peak weight memory is independent of T.
            x_g, tok_idx, weights_g, offsets = _grouped_routing(hidden_states, top_k_index, top_k_weights, self.num_experts)
            w_gu = unpack_int4(self.gate_up_qweight, self.gate_up_scales, self.gate_up_qzeros, self.layout, self.checkpoint_format)
            gate, up = _grouped_mm_bf16(x_g, w_gu, offsets).chunk(2, dim=-1)
            w_dn = unpack_int4(self.down_qweight, self.down_scales, self.down_qzeros, self.layout, self.checkpoint_format)
            out_g = _grouped_mm_bf16((F.silu(gate) * up).contiguous(), w_dn, offsets)
            out = torch.zeros_like(hidden_states)
            out.index_add_(0, tok_idx, out_g * weights_g.unsqueeze(-1))
            return out
        idx = top_k_index.reshape(-1)
        w_gu = unpack_int4(self.gate_up_qweight.index_select(0, idx), self.gate_up_scales.index_select(0, idx), self.gate_up_qzeros.index_select(0, idx), self.layout, self.checkpoint_format)
        w_dn = unpack_int4(self.down_qweight.index_select(0, idx), self.down_scales.index_select(0, idx), self.down_qzeros.index_select(0, idx), self.layout, self.checkpoint_format)
        return _eval_batched_experts(hidden_states, top_k_weights, w_gu, w_dn)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k router softmax paired with dense or packed INT4 SwiGLU experts."""

    def __init__(self, cfg: Qwen3MoeConfig, layer_idx: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        self.top_k, self.norm_topk_prob = cfg.num_experts_per_tok, cfg.norm_topk_prob
        p = f"model.layers.{layer_idx}.mlp"
        self.gate = _make_linear(f"{p}.gate", cfg.hidden_size, cfg.num_experts, False, cfg.dtype, tensors, cfg.quant)
        spec = _int4_spec(f"{p}.experts.0.gate_proj", cfg.hidden_size, tensors, cfg.quant)
        self.experts: nn.Module = Int4MoeExperts(cfg, spec[0], spec[1], spec[2], spec[3]) if spec is not None else DenseMoeExperts(cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        router_logits = cast(torch.Tensor, self.gate(flat))
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1, dtype=torch.float32), self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return cast(torch.Tensor, self.experts(flat, indices, weights.to(router_logits.dtype))).reshape(bsz, seq_len, hidden)


class Qwen3MoeDecoderLayer(nn.Module):
    """Single Qwen3-MoE transformer layer."""

    def __init__(self, cfg: Qwen3MoeConfig, layer_idx: int, max_cache_len: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype)
        self.self_attn = Qwen3MoeAttention(cfg, layer_idx, max_cache_len, tensors)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype)
        sparse = layer_idx not in cfg.mlp_only_layers and cfg.num_experts > 0 and (layer_idx + 1) % cfg.decoder_sparse_step == 0
        self.mlp: nn.Module = Qwen3MoeSparseMoeBlock(cfg, layer_idx, tensors) if sparse else Qwen3MoeMLP(cfg, layer_idx, tensors)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_position: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache_position, causal_mask)
        return x + cast(torch.Tensor, self.mlp(self.post_attention_layernorm(x)))


class Qwen3MoeModel(nn.Module):
    """Qwen3-MoE transformer backbone with shared RoPE and static causal mask positions."""

    inv_freq: torch.Tensor
    cache_positions: torch.Tensor

    def __init__(self, cfg: Qwen3MoeConfig, max_cache_len: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=cfg.dtype)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(cfg, i, max_cache_len, tensors) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype)
        self.register_buffer("inv_freq", 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)), persistent=False)
        self.register_buffer("cache_positions", torch.arange(max_cache_len, dtype=torch.long), persistent=False)

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        freqs = torch.outer(cache_position.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(hidden_states.dtype).unsqueeze(0), emb.sin().to(hidden_states.dtype).unsqueeze(0)
        allowed = self.cache_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
        seq_len = input_ids.shape[1]
        causal_mask = torch.zeros((1, 1, seq_len, self.max_cache_len), device=hidden_states.device, dtype=hidden_states.dtype).masked_fill(
            ~allowed.view(1, 1, seq_len, self.max_cache_len), torch.finfo(hidden_states.dtype).min
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, cache_position, causal_mask)
        return cast(torch.Tensor, self.norm(hidden_states))


class Qwen3MoeForCausalLM(nn.Module):
    """Causal language model exposing the engine's (input_ids, cache_position) interface."""

    def __init__(self, cfg: Qwen3MoeConfig, max_cache_len: int, tensors: dict[str, Any]) -> None:
        super().__init__()
        self.config = cfg
        self.model = Qwen3MoeModel(cfg, max_cache_len, tensors)
        self.lm_head = _make_linear("lm_head", cfg.hidden_size, cfg.vocab_size, False, cfg.dtype, tensors, cfg.quant)

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.lm_head(self.model(input_ids, cache_position)))


def set_experts_grouped(model: Qwen3MoeForCausalLM, grouped: bool) -> None:
    """Switches every MoE expert bank between grouped_mm prefill and batched decode."""
    for module in model.modules():
        if isinstance(module, (DenseMoeExperts, Int4MoeExperts)):
            module.grouped = grouped


def build(manifest: dict[str, Any], max_cache_len: int) -> tuple[Qwen3MoeForCausalLM, dict[str, torch.Tensor], Qwen3MoeConfig]:
    """Constructs a weightless meta Qwen3MoeForCausalLM and harvests its constant buffers."""
    cfg = config_from_dict(cast(dict[str, Any], manifest["config"]))
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg, max_cache_len, cast(dict[str, Any], manifest["tensors"])).eval()
    buffers: dict[str, torch.Tensor] = {
        "model.inv_freq": 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)),
        "model.cache_positions": torch.arange(max_cache_len, dtype=torch.long),
    }
    return model, buffers, cfg


def checkpoint_view(manifest_tensors: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Maps model parameter names to checkpoint safetensors names and stacked expert specs."""
    direct: dict[str, str] = {name: name for name in manifest_tensors}
    dense: dict[tuple[str, str], dict[int, str]] = {}
    int4: dict[tuple[str, str, str], dict[int, str]] = {}
    for name in manifest_tensors:
        m = _EXPERT_DENSE_RE.match(name)
        if m:
            dense.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = name
        elif (m4 := _EXPERT_INT4_RE.match(name)) is not None:
            int4.setdefault((m4.group(1), m4.group(3), m4.group(4)), {})[int(m4.group(2))] = name

    merged: dict[str, Any] = {}
    for p in sorted({p for p, _ in dense}):
        ex = sorted(dense[(p, "gate_proj")])
        merged[f"{p}.gate_up_proj"] = {"parts": [[dense[(p, "gate_proj")][e], dense[(p, "up_proj")][e]] for e in ex], "cat_dim": 0, "stack": True}
        merged[f"{p}.down_proj"] = {"parts": [[dense[(p, "down_proj")][e]] for e in ex], "cat_dim": 0, "stack": True}
    for p, kind in sorted({(p, k) for p, _, k in int4}):
        ex = sorted(int4[(p, "gate_proj", kind)])
        merged[f"{p}.gate_up_{kind}"] = {"parts": [[int4[(p, "gate_proj", kind)][e], int4[(p, "up_proj", kind)][e]] for e in ex], "cat_dim": 1, "stack": True}
        merged[f"{p}.down_{kind}"] = {"parts": [[int4[(p, "down_proj", kind)][e]] for e in ex], "cat_dim": 0, "stack": True}
    return direct, merged


def export_graphs(
    model: Qwen3MoeForCausalLM, max_cache_len: int, min_prefill_len: int
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram]:
    """Exports grouped_mm prefill and batched 1-token decode programs on the meta device."""
    import torch.fx.experimental._config as fx_config

    fx_config.backed_size_oblivious = True
    seq = torch.export.Dim("seq", min=min_prefill_len, max=max_cache_len)
    ex_len = min(max(min_prefill_len, 4), max_cache_len)
    with torch.device("meta"):
        set_experts_grouped(model, True)
        prefill = torch.export.export(
            model, (), kwargs={"input_ids": torch.zeros(1, ex_len, dtype=torch.long), "cache_position": torch.arange(ex_len, dtype=torch.long)},
            dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}}, strict=False,
        )
        set_experts_grouped(model, False)
        decode = torch.export.export(
            model, (), kwargs={"input_ids": torch.zeros(1, 1, dtype=torch.long), "cache_position": torch.zeros(1, dtype=torch.long)},
            strict=False,
        )
    return prefill, decode


def binding_extras(cfg: Qwen3MoeConfig, model: Qwen3MoeForCausalLM) -> dict[str, Any]:
    """Builds model-specific metadata recorded in binding.json."""
    int4_modules = sum(isinstance(m, (Int4Linear, Int4MoeExperts)) for m in model.modules())
    if cfg.quant is None and int4_modules == 0:
        return {}
    return {
        "quantization": {
            "quant_method": cfg.quant.quant_method if cfg.quant is not None else "int4",
            "bits": 4,
            "group_size": cfg.quant.group_size if cfg.quant is not None else 128,
            "int4_modules": int4_modules,
        }
    }
