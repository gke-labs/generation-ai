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

"""Export entry point for DeepSeek-V4.1-Flash (model_type ``deepseek_v41``).

There is no transformers implementation of this model; the exporter
captures the model maker's own reference module, carried as a fork in
``third_party/deepseek_v41`` with plain-aten versions of its kernels and
static cache state (see that repository's vxpu branch). The flow is the
transformers one: build on the meta device, harvest the constant
buffers, export prefill and decode, put the constants back, and bind
every parameter to the checkpoint by its own name. Routed experts are
``stacked`` banks assembled from the per-expert checkpoint tensors.

The model's arguments come from ``model_args.json`` next to a local
checkpoint when present (test fixtures), else from the fork's
``inference/config.json``, which is the released model's.
"""

import contextlib
import json
import os
import sys
import types

import torch

FORK = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "deepseek_v41", "inference")
MODEL_TYPE = "deepseek_v41"


def _fork():
    """Import the fork's modules (they import each other by bare name)."""
    path = os.path.abspath(os.environ.get("VXPU_DEEPSEEK_V41", FORK))
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("DEEPSEEK_KERNELS", "aten")
    import model as ds_model  # noqa: F401
    return ds_model


@contextlib.contextmanager
def bf16_default():
    """The reference runner sets torch's default dtype to bf16 for the whole
    run and the kernels size their outputs from it; construction, eager
    runs and export must happen under the same setting."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def model_args(repo_or_dir, max_cache_len, model_type_args=None):
    """ModelArgs for the checkpoint: ``model_args.json`` beside a local
    checkpoint, else the fork's released config."""
    ds = _fork()
    local = os.path.join(repo_or_dir, "model_args.json") if os.path.isdir(repo_or_dir) else None
    path = local if local and os.path.exists(local) else os.path.join(os.path.dirname(ds.__file__), "config.json")
    with open(path) as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if k in ds.ModelArgs.__dataclass_fields__}
    raw.update(max_batch_size=1, max_seq_len=max_cache_len, temperature=0)
    return ds.ModelArgs(**raw)


class Prefill(torch.nn.Module):
    """input_ids [1, L], cache_position [L] (always 0..L-1: this model's prefill starts at 0)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position):
        return self.model(input_ids, 0)[1].unsqueeze(1)  # [1, 1, vocab]: the last position's logits


class Decode(torch.nn.Module):
    """input_ids [1, 1], cache_position [1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position):
        return self.model(input_ids, cache_position[0])[1].unsqueeze(1)


def build(repo_or_dir, tokenizer, max_cache_len):
    """The model on the meta device with its constant buffers harvested.
    Returns (wrapper-less model, {buffer name: value})."""
    ds = _fork()
    args = model_args(repo_or_dir, max_cache_len)
    with torch.device("meta"), bf16_default():
        model = ds.Transformer(args, tokenizer).eval()
    # remove_duplicate=False: layers share one rotary table object (the fork
    # caches it), and every name needs its value
    constants = {name: buf.detach().clone()
                 for name, buf in model.named_buffers(remove_duplicate=False) if not buf.is_meta}
    model.to("meta")
    return model, constants, args


def set_routing(model, routing):
    for layer in model.layers:
        layer.ffn.routing = routing


def checkpoint_view(model, manifest_tensors):
    """(direct, merged) as export.checkpoint_view returns them, for this
    model: every parameter is named as in the checkpoint, except the
    expert banks, which stack the per-expert tensors."""
    import re
    direct, merged = {}, {}
    names = set(manifest_tensors)
    for name in names:
        direct[name] = name
    bank = re.compile(r"^(?P<prefix>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<i>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$")
    groups = {}
    for name in names:
        m = bank.match(name)
        if m:
            key = f"{m['prefix']}.{m['proj']}" + ("" if m["kind"] == "weight" else "_scale")
            groups.setdefault(key, []).append((int(m["i"]), name))
            direct.pop(name, None)
    for key, parts in groups.items():
        merged[key] = {"parts": [[n] for _, n in sorted(parts)], "cat_dim": 0, "stack": True}
    return direct, merged


def _hc_sinkhorn_op(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """The fork's hc_split_sinkhorn signature, routed to vxpu::hc_sinkhorn."""
    return torch.ops.vxpu.hc_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


def _head_forward(self, x, full_logits=False):
    """ParallelHead.forward (single process) with the linear as
    vxpu::head_linear over the bf16 weight."""
    if not full_logits:
        x = x[:, -1]
    return torch.ops.vxpu.head_linear(x, self.weight)


def _set_head(head, on):
    """The output head traced as vxpu::head_linear with its weight in the
    checkpoint's bf16 (the fork widens it to fp32 at construction, which
    the engine would then hold and read at twice the bytes)."""
    if on and not hasattr(head, "_vxpu_weight"):
        # kept outside the module's parameters, so the fp32 original is not
        # lifted into the exported graphs as a second, unused head
        head.__dict__["_vxpu_weight"] = head.weight
        w = head.weight
        head.weight = torch.nn.Parameter(torch.empty(w.shape, dtype=torch.bfloat16, device=w.device),
                                         requires_grad=False)
        head.forward = types.MethodType(_head_forward, head)
    elif not on and hasattr(head, "_vxpu_weight"):
        head.weight = head._vxpu_weight
        del head._vxpu_weight
        del head.forward


def set_external(model, on):
    """Route the expert evaluation and the n-gram lookups of every layer
    through the vxpu ops (see ops.py), or back to the module's own paths.
    The exporter traces with them on, so the graphs carry calls instead of
    the banks and tables; the engine answers the calls. The dense fp8
    linears, the hyper-connection mixing and the output head become ops
    too, for their device kernels."""
    from . import ops
    ds = _fork()
    ds.linear_fn = torch.ops.vxpu.fp8_linear if on else None  # every dense fp8 linear
    ds.grouped_linear_fn = torch.ops.vxpu.fp8_grouped_linear if on else None  # the attention's wo_a
    # the hyper-connection mixing: the fork calls it through a module-level
    # name, so it is swapped for the op while tracing and restored after
    if not hasattr(ds, "_vxpu_hc_split_sinkhorn"):
        ds._vxpu_hc_split_sinkhorn = ds.hc_split_sinkhorn
    ds.hc_split_sinkhorn = _hc_sinkhorn_op if on else ds._vxpu_hc_split_sinkhorn
    if getattr(ds, "world_size", 1) == 1:
        _set_head(model.head, on)
    for layer in model.layers:
        layer.ffn.experts_fn = torch.ops.vxpu.moe_experts if on else None
        if layer.engram is not None:
            layer.engram.embed.rows_fn = torch.ops.vxpu.engram_rows if on else None
            ops.ENGRAM_DIM[layer.layer_id] = layer.engram.embed.dim


def external_spec(model, view, args):
    """What the graphs call out for, from the checkpoint view: per layer,
    the expert banks (stacked specs) and the engram table (tensor names).
    Complete or an error: the graphs no longer reference these tensors,
    so the classifier's unbound check does not see a checkpoint that
    lacks them, and the failure would otherwise arrive on the executor
    after the whole fetch."""
    import re
    direct, merged = view
    experts, engram = {}, {}
    bank = re.compile(r"^layers\.(\d+)\.ffn\.(w[123](?:_scale)?)$")
    for key, spec in merged.items():
        m = bank.match(key)
        if m:
            experts.setdefault(m.group(1), {})[m.group(2)] = spec
    problems = []
    for layer in model.layers:
        want = {n for n, p in layer.ffn.named_parameters() if re.fullmatch(r"w[123](_scale)?", n)}
        got = experts.get(str(layer.layer_id), {})
        for name in sorted(want - set(got)):
            problems.append(f"layers.{layer.layer_id}.ffn.{name}: no expert tensors in the checkpoint")
        for name, spec in got.items():
            if len(spec["parts"]) != args.n_routed_experts:
                problems.append(f"layers.{layer.layer_id}.ffn.{name}: {len(spec['parts'])} of {args.n_routed_experts} experts")
        if layer.engram is not None:
            prefix = f"layers.{layer.layer_id}.engram.embed."
            if prefix + "weight" not in direct or prefix + "scale" not in direct:
                problems.append(f"{prefix}weight/scale: not in the checkpoint")
                continue
            engram[str(layer.layer_id)] = {"weight": direct[prefix + "weight"], "scale": direct[prefix + "scale"],
                                           "num_embeddings": layer.engram.embed.num_embeddings,
                                           "dim": layer.engram.embed.dim}
    if problems:
        raise RuntimeError(f"external tensors missing from the checkpoint (artifact would be incomplete): {problems[:5]}")
    return {"experts": experts, "engram": engram}


def host_experts(layer_id, args, banks):
    """The fork's MoE module for `layer_id` holding `banks` ({w1, w2, w3,
    w1_scale, ...}: tensors on the host) and nothing else real; its own
    routings compute on them."""
    ds = _fork()
    with torch.device("meta"), bf16_default():
        moe = ds.MoE(layer_id, args)
    for name, tensor in banks.items():
        if tensor.dtype == torch.int8:  # the checkpoint's packed fp4 bytes; the module keeps uint8
            tensor = tensor.view(torch.uint8)
        setattr(moe, name, torch.nn.Parameter(tensor, requires_grad=False))
    return moe


def host_engram(layer_id, num_embeddings, dim, weight, scale):
    """The fork's n-gram table module for `layer_id` over host tensors."""
    ds = _fork()
    with torch.device("meta"), bf16_default():
        embed = ds.ParallelEngramEmbedding(num_embeddings, dim)
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    embed.scale = torch.nn.Parameter(scale, requires_grad=False)
    embed.table_id = layer_id
    return embed


def export_graphs(model, max_cache_len, min_prefill_len):
    """Prefill (symbolic length) and decode programs. The expert banks and
    the n-gram tables stay out of the graphs (set_external): the executor
    places them."""
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    seq = torch.export.Dim("seq", min=min_prefill_len, max=max_cache_len)
    set_external(model, True)
    try:
        return _export_graphs(model, seq)
    finally:
        set_external(model, False)


def _export_graphs(model, seq):
    with bf16_default():
        set_routing(model, "grouped")
        # keyword arguments, as the engine calls the graphs
        prefill = torch.export.export(
            Prefill(model), (),
            kwargs={"input_ids": torch.zeros(1, 8, dtype=torch.long, device="meta"),
                    "cache_position": torch.arange(8, device="meta")},
            dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
            strict=False)
        set_routing(model, "batched")
        decode = torch.export.export(
            Decode(model), (),
            kwargs={"input_ids": torch.zeros(1, 1, dtype=torch.long, device="meta"),
                    "cache_position": torch.zeros(1, dtype=torch.long, device="meta")},
            strict=False)
    return prefill, decode


def binding_extras(args):
    """What the engine must know beyond the tensor classes."""
    return {
        "model_type": MODEL_TYPE,
        # The prefill graph starts at position 0: a changed prefix means a
        # fresh session, not a suffix prefill.
        "prefill_from_zero": True,
        "quantization": {"quant_method": "fp8", "block_size": [32, 32],
                         "expert_dtype": args.expert_dtype or "fp8", "scale_fmt": "ue8m0"},
    }
