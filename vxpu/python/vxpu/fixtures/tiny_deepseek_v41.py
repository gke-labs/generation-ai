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

"""A tiny random DeepSeek-V4.1 checkpoint in the published layout: fp8
dense weights with 32x32 block scales, fp4 experts (int8-packed, e8m0
scales per 32), compressed sparse attention with an indexer, hyper-
connections, and an engram layer with an fp8 table. Written by the
fork's own module, so the names are the checkpoint's. Needs a tokenizer
for the engram token map: a tiny one is fetched from the Hub.

    python -m vxpu.fixtures.tiny_deepseek_v41 OUT_DIR
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file

TOKENIZER = "trl-internal-testing/tiny-random-LlamaForCausalLM"

ARGS = dict(
    vocab_size=512, dim=128, moe_inter_dim=64, n_layers=4, n_mtp_layers=0, n_heads=4,
    n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    q_lora_rank=64, head_dim=64, rope_head_dim=16, o_groups=2, o_lora_rank=32,
    window_size=16, compress_ratios=(0, 2, 2, 1), kv_source_layers=(1, 3), index_source_layers=(1, 3),
    index_n_heads=2, index_head_dim=32, index_topk=4,
    hc_mult=4, engram_layer_ids=(1,), engram_num_embeddings=(4096,), engram_max_ngram_size=3,
    engram_vocab_size=1000, engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=0,
    dtype="fp8", expert_dtype="fp4", score_func="sqrtsoftplus", swiglu_limit=10.0,
)

DRAFTER_ARGS = dict(
    n_mtp_layers=2,
    compress_ratios=(0, 2, 2, 1, 0, 0),
    # the DSpark drafter: two stages after the backbone, drafting blocks of 3 tokens
    dspark_block_size=3,
    dspark_noise_token_id=3,
    dspark_target_layer_ids=(2, 3),
    dspark_markov_rank=16,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=2,
)


def write(out_dir: str, seed: int = 0, drafter: bool = False) -> str:
    from transformers import AutoTokenizer
    from vxpu.deepseek_v41 import _fork
    ds = _fork()
    import kernel_aten as ka
    from engram import build_compressed_token_map

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.save_pretrained(out_dir)

    args_dict = dict(ARGS)
    if drafter:
        args_dict.update(DRAFTER_ARGS)
    args_dict["engram_compressed_vocab_size"] = build_compressed_token_map(tokenizer)[1]
    args_dict["vocab_size"] = len(tokenizer)
    args = ds.ModelArgs(max_batch_size=1, max_seq_len=64, temperature=0, **args_dict)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = ds.Transformer(args, tokenizer)
    finally:
        torch.set_default_dtype(torch.float32)

    tensors = {}
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=0.05):
        return torch.randn(*shape, generator=g) * scale

    for name, p in model.named_parameters():
        if name.endswith("_scale") and name.split(".")[-1] in ("hc_attn_scale", "hc_ffn_scale"):
            tensors[name] = torch.ones(p.shape)
        elif p.dtype == torch.float4_e2m1fn_x2 or (p.dtype == torch.uint8 and p.dim() == 3):
            # an expert bank (the fork keeps it as uint8 bytes; torch.export cannot serialize
            # fp4): the checkpoint stores per-expert int8-packed fp4 rows
            e, out, half = p.shape
            vals = ka._to_fp4_rounded(rand(e, out, half * 2, scale=1.0).clamp(-6, 6))
            packed = ka.pack_fp4(vals).view(torch.int8)
            base = name.rsplit(".", 1)[0]
            for i in range(e):
                tensors[f"{base}.experts.{i}.{name.rsplit('.', 1)[1]}.weight"] = packed[i].contiguous()
        elif p.dtype == torch.float8_e8m0fnu and p.dim() == 3:  # a bank's scales
            e = p.shape[0]
            base, proj = name.rsplit(".", 1)[0], name.rsplit(".", 1)[1].replace("_scale", "")
            for i in range(e):
                tensors[f"{base}.experts.{i}.{proj}.scale"] = torch.full(p.shape[1:], 2.0 ** -6).to(torch.float8_e8m0fnu)
        elif p.dtype == torch.float8_e8m0fnu:
            tensors[name] = torch.full(p.shape, 2.0 ** -7).to(torch.float8_e8m0fnu)
        elif p.dtype == torch.float8_e4m3fn:
            tensors[name] = rand(*p.shape, scale=1.0).to(torch.float8_e4m3fn)
        elif p.dim() == 1 and ("norm" in name or name.endswith("_weight")):
            tensors[name] = (1.0 + rand(*p.shape, scale=0.1)).to(torch.bfloat16 if p.dtype == torch.bfloat16 else p.dtype)
        else:
            dtype = torch.bfloat16 if p.dtype in (torch.bfloat16, torch.float32) else p.dtype
            # the checkpoint stores head/wkv-compressor/confidence weights as bf16; f32 params read bf16
            tensors[name] = rand(*p.shape).to(dtype) if p.dim() > 1 else rand(*p.shape).to(dtype)
    # the gate bias and attention sinks are f32 in the checkpoint
    for name in list(tensors):
        if name.endswith("ffn.gate.bias") or name.endswith("attn_sink") or ".hc_" in name:
            tensors[name] = tensors[name].float()
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
                   "vocab_size": args.vocab_size, "eos_token_id": tokenizer.eos_token_id,
                   "bos_token_id": tokenizer.bos_token_id}, f, indent=1)
    with open(os.path.join(out_dir, "model_args.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in args_dict.items()}, f, indent=1)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"do_sample": False}, f)
    return out_dir


_shared_model_dir: str | None = None
_shared_artifact_dir: str | None = None


def get_model_dir(seed: int = 0) -> str:
    """Returns a shared temporary directory containing the tiny DeepSeek checkpoint."""
    global _shared_model_dir
    if _shared_model_dir is None or not os.path.isdir(_shared_model_dir):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="tiny-deepseek-model-")
        _shared_model_dir = write(tmp, seed=seed)
    return _shared_model_dir


def get_artifact_dir(max_cache_len: int = 64) -> str:
    """Returns a shared temporary directory containing the exported tiny DeepSeek artifact."""
    global _shared_artifact_dir
    if _shared_artifact_dir is None or not os.path.isdir(_shared_artifact_dir):
        import tempfile
        from vxpu.export import export_artifact
        model_dir = get_model_dir()
        tmp = tempfile.mkdtemp(prefix="tiny-deepseek-artifact-")
        export_artifact(model_dir, tmp, max_cache_len=max_cache_len)
        _shared_artifact_dir = tmp
    return _shared_artifact_dir


if __name__ == "__main__":
    print(write(sys.argv[1], drafter="--drafter" in sys.argv[2:]))
