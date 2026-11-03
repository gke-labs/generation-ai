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


def model_args(repo_or_dir, max_cache_len, model_type_args=None, batch=1, step_tokens=1, window_slack=None,
               draft_head_fp8=False):
    """ModelArgs for the checkpoint: ``model_args.json`` beside a local
    checkpoint, else the fork's released config. ``batch`` sequences decode
    together: every state buffer gets that leading dimension. A decode step
    takes ``step_tokens`` tokens per row: the window rings get that many
    slots minus one beyond the window, so a step's rejected tokens never
    overwrite a position the row still reads. ``window_slack`` sets those
    slots instead: a decode exported with fewer token rows to serve as a
    shorter verify of another step shares that step's rings (hero run 15),
    so it takes that step's slack, not its own (the rings' size and the
    position's slot in them follow it)."""
    ds = _fork()
    local = os.path.join(repo_or_dir, "model_args.json") if os.path.isdir(repo_or_dir) else None
    path = local if local and os.path.exists(local) else os.path.join(os.path.dirname(ds.__file__), "config.json")
    with open(path) as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if k in ds.ModelArgs.__dataclass_fields__}
    raw.update(max_batch_size=batch, max_seq_len=max_cache_len, temperature=0,
               window_slack=step_tokens - 1 if window_slack is None else window_slack,
               dspark_head_fp8=bool(draft_head_fp8))
    return ds.ModelArgs(**raw)


def has_drafter(args):
    """Whether the checkpoint carries its own drafter (DSpark stages, mtp.*)."""
    return args.dspark_block_size > 0 and args.n_mtp_layers > 0


class Prefill(torch.nn.Module):
    """input_ids [R, L], cache_position [L] (always 0..L-1: this model's prefill starts at 0).
    R prompts of one length at once (R is 1 unless the artifact was exported with prefill rows):
    they fill rows 0..R-1 of the state, and the engine moves each to its own row. With the
    drafter, it also seeds the drafter's rings with the prompts' hidden states."""

    def __init__(self, model, drafter=False):
        super().__init__()
        self.model = model
        self.drafter = drafter

    def forward(self, input_ids, cache_position):
        _, logits, main_hidden = self.model(input_ids, 0)
        if self.drafter:
            self.model.spec_observe(main_hidden, 0)
        return logits.unsqueeze(1)  # [R, 1, vocab]: each prompt's last position's logits


class Decode(torch.nn.Module):
    """input_ids [B, S], cache_position [B]: S tokens per row (one, or a token and the drafts to
    verify), each row's from its own position. Returns every token's logits, [B, S, vocab]. With
    the drafter, it also writes the step's hidden states into the drafter's rings."""

    def __init__(self, model, drafter=False):
        super().__init__()
        self.model = model
        self.drafter = drafter

    def forward(self, input_ids, cache_position):
        _, logits, main_hidden = self.model(input_ids, cache_position)
        if self.drafter:
            self.model.spec_observe(main_hidden, cache_position)
        return logits if input_ids.size(1) > 1 else logits.unsqueeze(1)


class Draft(torch.nn.Module):
    """input_ids [B, 1], cache_position [B]: each row's next token and its position. Returns one
    int64 tensor [B, 2 * block]: the drafter's tokens after it, then each draft's confidence (the
    confidence head's score through a sigmoid, times 1e6, rounded), so the native loop's one output
    buffer carries both and the loop reads a draft's confidence beside its acceptance (hero run 12;
    the binding's draft.confidence names the encoding). Reads the drafter's rings and writes no
    state."""

    CONFIDENCE_SCALE = 1e6

    def __init__(self, model, passes=1):
        super().__init__()
        self.model = model
        self.passes = passes  # blocks drafted in sequence (hero run 15); the binding's block is passes * the head's

    def forward(self, input_ids, cache_position):
        tokens, _, confidence = self.model.spec_draft(input_ids.view(-1), cache_position, passes=self.passes)
        score = (torch.sigmoid(confidence.float()) * self.CONFIDENCE_SCALE).round().to(torch.int64)
        return torch.cat([tokens[:, 1:], score.view(tokens.size(0), -1)], dim=1)


def build(repo_or_dir, tokenizer, max_cache_len, batch=1, step_tokens=1, window_slack=None, draft_head_fp8=False):
    """The model on the meta device with its constant buffers harvested.
    Returns (wrapper-less model, {buffer name: value})."""
    ds = _fork()
    args = model_args(repo_or_dir, max_cache_len, batch=batch, step_tokens=step_tokens, window_slack=window_slack,
                      draft_head_fp8=draft_head_fp8)
    with torch.device("meta"), bf16_default():
        model = ds.Transformer(args, tokenizer).eval()
    # remove_duplicate=False: layers share one rotary table object (the fork
    # caches it), and every name needs its value
    constants = {name: buf.detach().clone()
                 for name, buf in model.named_buffers(remove_duplicate=False) if not buf.is_meta}
    model.to("meta")
    return model, constants, args


def set_routing(model, routing):
    for layer in list(model.layers) + list(model.mtp):
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


def _rotary_op(x, freqs_cis, inverse=False):
    """The fork's apply_rotary_emb signature, routed to vxpu::rotary."""
    y = torch.ops.vxpu.rotary(x, freqs_cis, inverse)
    x.copy_(y)
    return x


def _head_forward(self, x, full_logits=False):
    """ParallelHead.forward with the linear as vxpu::head_linear over the bf16 weight; under
    tensor parallelism (the fork's world_size > 1) the weight is this rank's vocabulary slice and
    the ranks' logits are gathered (vxpu::tp_allgather) as the fork's all_gather would."""
    if not full_logits:
        x = x[:, -1]
    logits = torch.ops.vxpu.head_linear(x, self.weight)
    world = getattr(_fork(), "world_size", 1)
    if world > 1:
        logits = torch.ops.vxpu.tp_allgather(logits, world)
    return logits


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
    ds.ffn_front_fn = torch.ops.vxpu.ffn_front if on else None  # a block's FFN front half, up to the experts
    ds.hc_front_fn = torch.ops.vxpu.hc_front if on else None  # and the attention half's, up to the attention
    ds.shared_ffn_fn = torch.ops.vxpu.shared_ffn if on else None  # the MoE's shared expert
    ds.shared_ffn_front_fn = torch.ops.vxpu.shared_ffn_front if on else None  # the shared expert beside the experts
    ds.shared_ffn_combine_fn = torch.ops.vxpu.shared_ffn_combine if on else None  # its sum with theirs, and the mix
    ds.attn_q_fn = torch.ops.vxpu.attn_q if on else None  # the attention's query path
    ds.window_kv_fn = torch.ops.vxpu.window_kv if on else None  # and its window key
    ds.attn_out_fn = torch.ops.vxpu.attn_out if on else None  # and its output path
    ds.attn_core_fn = torch.ops.vxpu.attn_core if on else None  # the attention finishing the output's first step
    ds.attn_select_fn = torch.ops.vxpu.attn_select if on else None  # and forming its selection over the two caches
    ds.attn_out_partial_fn = torch.ops.vxpu.attn_out_partial if on else None  # the output path without the mix (TP)
    ds.tp_allreduce_hc_fn = torch.ops.vxpu.tp_allreduce_hc if on else None  # the ranks' sum with the block's mix (TP)
    # the hyper-connection mixing: the fork calls it through a module-level
    # name, so it is swapped for the op while tracing and restored after
    if not hasattr(ds, "_vxpu_hc_split_sinkhorn"):
        ds._vxpu_hc_split_sinkhorn = ds.hc_split_sinkhorn
    ds.hc_split_sinkhorn = _hc_sinkhorn_op if on else ds._vxpu_hc_split_sinkhorn
    # rotary embeddings: swapped for vxpu::rotary while tracing and restored after
    if not hasattr(ds, "_vxpu_apply_rotary_emb"):
        ds._vxpu_apply_rotary_emb = ds.apply_rotary_emb
    ds.apply_rotary_emb = _rotary_op if on else ds._vxpu_apply_rotary_emb
    # the sparse attention: the fork calls it through a module-level name too
    if not hasattr(ds, "_vxpu_sparse_attn"):
        ds._vxpu_sparse_attn = ds.sparse_attn
    ds.sparse_attn = torch.ops.vxpu.sparse_attn if on else ds._vxpu_sparse_attn
    _set_head(model.head, on)
    if getattr(ds, "world_size", 1) > 1:
        # Tensor parallelism (hero run 14): the placed output path (attn_out) and the placed shared
        # expert mix the hyper-connection stream inside the op, which needs the summed output, so
        # under a row split they stay off and the fork's own path runs: the projection as
        # fp8_linear / fp8_grouped_linear ops, torch.distributed's all_reduce (vxpu::tp_allreduce
        # through _FakeDist) and the block's hc_post as tensor work.
        ds.attn_out_fn = ds.attn_core_fn = None
        # The selection and the output path stay placed (attn_select, attn_out_partial): each rank
        # attends with its heads over the caches and projects through its slices of wo_a and wo_b to
        # a partial the ranks' all_reduce sums; the block mixes the stream itself (hero run 15),
        # where before the fork's own path ran here (the window's ids, the caches' concatenation,
        # the inverse rotary, act_quant, the grouped projection and wo_b as ops and glue).
        ds.shared_ffn_fn = ds.shared_ffn_combine_fn = None
        # The shared expert's front stays on: each rank runs it over its slice of the expert
        # (parallel_slices) and the fork adds the partial to the routed experts' before their
        # all_reduce (hero run 15), where before every rank ran the whole shared expert as three
        # fp8_linear ops and their SwiGLU glue.
    for layer in list(model.layers) + list(model.mtp):
        layer.ffn.experts_fn = torch.ops.vxpu.moe_experts if on else None
        if layer.engram is not None:
            layer.engram.embed.rows_fn = torch.ops.vxpu.engram_rows if on else None
            ops.ENGRAM_DIM[layer.layer_id] = layer.engram.embed.dim


def external_spec(model, view, args, drafter=False):
    """What the graphs call out for, from the checkpoint view: per layer,
    the expert banks (stacked specs) and the engram table (tensor names).
    With the drafter, its stages' banks too, by their layer ids (after the
    backbone's).
    Complete or an error: the graphs no longer reference these tensors,
    so the classifier's unbound check does not see a checkpoint that
    lacks them, and the failure would otherwise arrive on the executor
    after the whole fetch."""
    import re
    direct, merged = view
    experts, engram = {}, {}
    bank = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.(w[123](?:_scale)?)$")
    for key, spec in merged.items():
        m = bank.match(key)
        if m and (m.group(1) == "layers" or drafter):
            layer_id = int(m.group(2)) + (args.n_layers if m.group(1) == "mtp" else 0)
            experts.setdefault(str(layer_id), {})[m.group(3)] = spec
    problems = []
    for layer in list(model.layers) + (list(model.mtp) if drafter else []):
        want = {n for n, p in layer.ffn.named_parameters() if re.fullmatch(r"w[123](_scale)?", n)}
        got = experts.get(str(layer.layer_id), {})
        n_experts = args.get_moe_config(layer.layer_id)[0]
        for name in sorted(want - set(got)):
            problems.append(f"layer {layer.layer_id} ffn.{name}: no expert tensors in the checkpoint")
        for name, spec in got.items():
            if len(spec["parts"]) != n_experts:
                problems.append(f"layer {layer.layer_id} ffn.{name}: {len(spec['parts'])} of {n_experts} experts")
        if getattr(layer, "engram", None) is not None:
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


def host_engram(layer_id, num_embeddings, dim, weight=None, scale=None):
    """The fork's n-gram table module for `layer_id` over host tensors."""
    ds = _fork()
    with torch.device("meta"), bf16_default():
        embed = ds.ParallelEngramEmbedding(num_embeddings, dim)
    if weight is None:
        weight = torch.empty((0, dim), dtype=torch.uint8, device="cpu")
    if scale is None:
        scale = torch.empty((0, dim // embed.block_size), dtype=torch.uint8, device="cpu")
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    embed.scale = torch.nn.Parameter(scale, requires_grad=False)
    embed.table_id = layer_id
    return embed


class _FakeDist:
    """Stands in for torch.distributed in the fork while a tensor-parallel program is built and
    traced (hero run 14): the fork sizes its parallel linears, embedding and head from
    get_world_size()/get_rank(), and its all_reduce and all_gather become vxpu::tp_allreduce and
    vxpu::tp_allgather calls in the graph, answered by the executor (parallel.py, stepgraph.py)."""

    def __init__(self, world_size, rank):
        self.world_size, self.rank = world_size, rank

    def is_initialized(self):
        return self.world_size > 1

    def get_world_size(self, *a, **k):
        return self.world_size

    def get_rank(self, *a, **k):
        return self.rank

    def all_reduce(self, tensor, *a, **k):  # in place, as torch.distributed's
        tensor.copy_(torch.ops.vxpu.tp_allreduce(tensor))

    def all_gather(self, tensor_list, tensor, *a, **k):
        gathered = torch.ops.vxpu.tp_allgather(tensor, len(tensor_list))
        n = tensor.shape[-1]
        for i, out in enumerate(tensor_list):
            out.copy_(gathered[..., i * n:(i + 1) * n])


@contextlib.contextmanager
def tensor_parallel(world_size, rank):
    """Build and trace the fork's model as rank `rank` of `world_size` (its slices of the
    parallel weights, the collectives as graph calls); world_size 1 changes nothing."""
    ds = _fork()
    previous = ds.dist
    if world_size > 1:
        ds.dist = _FakeDist(world_size, rank)
    try:
        yield
    finally:
        ds.dist = previous
        if world_size > 1:  # the module globals the Transformer set from the fake
            ds.world_size, ds.rank = 1, 0


def parallel_slices(model):
    """How each parameter of a tensor-parallel rank is cut from the checkpoint's full tensor:
    {parameter fqn: dim} for the parallel modules (column-parallel linears and the head and
    embedding by rows, row-parallel linears by columns; an fp8 weight's block scales the same
    way; the attention sink by head), from the fork's module types. The engine slices the
    full tensors by these at load, so one checkpoint serves every rank."""
    ds = _fork()
    out = {}
    for name, module in model.named_modules(remove_duplicate=False):  # the head and embedding are shared with the drafter's last stage
        if isinstance(module, ds.ColumnParallelLinear):
            out[f"{name}.weight"] = 0
            if getattr(module, "scale", None) is not None:
                out[f"{name}.scale"] = 0
        elif isinstance(module, ds.RowParallelLinear):
            out[f"{name}.weight"] = 1
            if getattr(module, "scale", None) is not None:
                out[f"{name}.scale"] = 1
        elif isinstance(module, (ds.ParallelEmbedding, ds.ParallelHead)):
            out[f"{name}.weight"] = 0
            if getattr(module, "scale", None) is not None:  # the drafter's fp8 head: its block scales by rows too
                out[f"{name}.scale"] = 0
        elif isinstance(module, ds.ParallelEngramEmbedding):
            out[f"{name}.weight"] = 0
            out[f"{name}.scale"] = 0
        elif isinstance(module, ds.Attention):
            out[f"{name}.attn_sink"] = 0
        elif isinstance(module, ds.Expert) and name.endswith(".shared_experts"):
            # the MoE's shared expert, split like a column- then row-parallel pair: w1 and w3 by
            # rows (the expert's width), w2 by columns, their block scales the same way; each
            # rank's partial joins the routed experts' partial in the FFN's all_reduce (hero run 15)
            for linear, dim in (("w1", 0), ("w3", 0), ("w2", 1)):
                out[f"{name}.{linear}.weight"] = dim
                if getattr(getattr(module, linear), "scale", None) is not None:
                    out[f"{name}.{linear}.scale"] = dim
    return out


def export_graphs(model, max_cache_len, min_prefill_len, batch=1, step_tokens=1, drafter=False, prefill=True,
                  prefill_rows=1, draft_program=True, draft_passes=1):
    """Prefill (symbolic length) and decode programs, and with the drafter
    the draft program (else None; draft_program=False: the decode writes the
    drafter's rings but no draft program is exported, for a rank's program
    of a four-rank step, whose drafter is the one program the engine binds
    to rank 0). The expert banks and the n-gram tables
    stay out of the graphs (set_external): the executor places them.
    ``prefill_rows`` above 1 gives the prefill a symbolic row count too (1 to
    that many prompts of one length at once, hero run 15: a batch's rows
    prefilled together share their expert reads)."""
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    seq = torch.export.Dim("seq", min=min_prefill_len, max=max_cache_len)
    # the row count's floor is 2: export specializes a dimension that may be 1 (the engine prefills
    # a lone prompt twice, in rows 0 and 1, and keeps row 0's)
    # (a two-row prefill is a fixed shape: Dim refuses min == max)
    rows = (torch.export.Dim("rows", min=2, max=prefill_rows) if prefill_rows > 2
            else 2 if prefill_rows == 2 else None)
    set_external(model, True)
    try:
        return _export_graphs(model, seq, batch, step_tokens, drafter, prefill, rows, draft_program, draft_passes)
    finally:
        set_external(model, False)


def _export_graphs(model, seq, batch, step_tokens, drafter, prefill=True, rows=None, draft_program=True, draft_passes=1):
    """``rows`` is the prefill's row dimension: None for one row, the integer 2 for a fixed two rows,
    else a Dim from 2 up."""
    with bf16_default():
        set_routing(model, "grouped")
        # keyword arguments, as the engine calls the graphs
        prefill = torch.export.export(
            Prefill(model, drafter), (),
            kwargs={"input_ids": torch.zeros(2 if rows is not None else 1, 8, dtype=torch.long, device="meta"),
                    "cache_position": torch.arange(8, device="meta")},
            dynamic_shapes={"input_ids": ({0: rows, 1: seq} if rows is not None and not isinstance(rows, int)
                                          else {1: seq}),
                            "cache_position": {0: seq}},
            strict=False) if prefill else None
        set_routing(model, "batched")
        positions = {"cache_position": torch.zeros(batch, dtype=torch.long, device="meta")}
        decode = torch.export.export(
            Decode(model, drafter), (),
            kwargs={"input_ids": torch.zeros(batch, step_tokens, dtype=torch.long, device="meta"), **positions},
            strict=False)
        draft = None
        if drafter and draft_program:
            draft = torch.export.export(
                Draft(model, draft_passes), (),
                kwargs={"input_ids": torch.zeros(batch, 1, dtype=torch.long, device="meta"), **positions},
                strict=False)
    return prefill, decode, draft


def binding_extras(args, step_tokens=1, drafter=False, draft_passes=1):
    """What the engine must know beyond the tensor classes."""
    return {
        # tokens per row in a decode step, and the drafter's block (draft.pt2)
        "step_tokens": step_tokens,
        # the slots beyond the window and compressor rings (step_tokens - 1 unless the export set
        # it): programs sharing one state must agree on it, so the engine checks a shorter verify's
        "window_slack": args.window_slack,
        **({"draft": {"block": args.dspark_block_size * draft_passes, "confidence": "sigmoid_x1e6",
                      "passes": draft_passes}} if drafter else {}),
        "model_type": MODEL_TYPE,
        # The prefill graph starts at position 0: a changed prefix means a
        # fresh session, not a suffix prefill.
        "prefill_from_zero": True,
        "quantization": {"quant_method": "fp8", "block_size": [32, 32],
                         "expert_dtype": args.expert_dtype or "fp8", "scale_fmt": "ue8m0"},
    }
