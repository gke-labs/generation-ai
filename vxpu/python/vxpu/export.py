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

"""Export a model to a vXPU artifact — without loading its weights.

The model is instantiated on torch's meta device (shapes and dtypes,
no storage), so a laptop can export any size of model. Two graphs are
captured via transformers' export wrapper:

  prefill.pt2 — dynamic sequence length, explicit cache positions
  decode.pt2  — constant shapes, so executors can compile it once

Every tensor the graphs reference is classified:

  bound   — a content-addressed weight reference from the manifest
  derived — computed from config at load time (RoPE tables, embedding
            scales); never stored in checkpoints
  state   — per-session, zero-initialized (KV caches, counters);
            mutated in place by the graphs

An artifact directory is the unit of interchange:
manifest.json + binding.json + prefill.pt2 + decode.pt2 — a few tens
of MB describing models of any size.

Usage:
    python -m vxpu.export google/gemma-4-E4B-it -o gemma-e4b/
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import executorch as hf_executorch
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
)

from .fp8 import apply_fp8, fp8_config, set_experts_implementation
from .manifest import build_manifest

def _head_shapes(config):
    """Per-layer (num_kv_heads, head_dim) for the static cache.

    transformers' own helper slices ``layer_types[:-num_kv_shared_layers]``,
    which is empty when a model shares no KV layers (``[:-0]`` is ``[:0]``)
    — e.g. Gemma 4 31B. Fall back to the full layer list in that case;
    otherwise defer to transformers.
    """
    num_heads, head_dim = _upstream_head_shapes(config)
    if not isinstance(num_heads, list) or num_heads:
        return num_heads, head_dim
    layer_types = list(config.layer_types)
    shared = getattr(config, "num_kv_shared_layers", 0) or 0
    if shared:
        layer_types = layer_types[:-shared]
    head_dim = [config.global_head_dim
                if layer == "full_attention" and config.global_head_dim
                else config.head_dim for layer in layer_types]
    num_heads = [config.num_global_key_value_heads
                 if layer == "full_attention" and config.attention_k_eq_v
                 else config.num_key_value_heads for layer in layer_types]
    return num_heads, head_dim


_upstream_head_shapes = hf_executorch.get_head_shapes
hf_executorch.get_head_shapes = _head_shapes


def checkpoint_view(model, manifest_tensors):
    """The checkpoint as the model's parameters see it, per transformers.

    ``from_pretrained`` does not look checkpoint keys up by name: it runs
    the model's conversion mapping, which renames keys (``model.
    language_model.X`` -> ``model.X`` for a text-only class) and merges
    several checkpoint tensors into one parameter (per-expert
    projections into an expert bank, a sharded table into one table).
    Replay exactly that on the manifest, without the bytes.

    Returns (direct, merged): ``direct`` maps a model key to the manifest
    name holding it; ``merged`` maps a model key to an assembly spec,
    ``{"parts": [[names...], ...], "cat_dim": d, "stack": bool}`` (rows
    concatenated along ``cat_dim``, then stacked along a new leading
    dimension if ``stack``), as the executor's fetch_stacked expects.
    """
    import re
    from transformers.conversion_mapping import get_model_conversion_mapping

    renamings = get_model_conversion_mapping(model)
    direct, groups = {}, {}
    for name in sorted(manifest_tensors):
        key, converter, pattern = name, None, None
        for renaming in renamings:
            new_key, matched = renaming.rename_source_key(key)
            if matched is not None and getattr(renaming, "operations", None):
                converter, pattern = renaming, matched
            key = new_key
        if converter is None:
            direct[key] = name
            continue
        group = groups.setdefault(key, {"converter": converter, "sources": {}})
        # The index the pattern's "*" stood for, so shards and experts
        # assemble in numeric order (shard_10 after shard_9).
        regex = re.escape(pattern).replace(r"\*", r"([^.]+)")
        match = re.search(regex, name)
        index = match.group(1) if match else "0"
        index = int(index) if index.isdigit() else index
        group["sources"].setdefault(pattern, []).append((index, name))

    merged = {}
    for key, group in groups.items():
        ops = [type(op).__name__ for op in group["converter"].operations]
        columns = [[n for _, n in sorted(v)] for v in group["sources"].values()]
        if ops == ["MergeModulelist", "Concatenate"]:
            # Stack each pattern's tensors over the index, then concatenate the
            # patterns along `dim` of the stacked result: per row that is a
            # concatenation along dim - 1.
            dim = group["converter"].operations[1].dim
            rows = [list(parts) for parts in zip(*columns)]
            merged[key] = {"parts": rows, "cat_dim": dim - 1, "stack": True}
        elif ops == ["MergeModulelist"]:
            merged[key] = {"parts": [[n] for n in columns[0]], "cat_dim": 0,
                           "stack": True}
        elif ops == ["Concatenate"]:
            dim = group["converter"].operations[0].dim
            merged[key] = {"parts": [[n for col in columns for n in col]],
                           "cat_dim": dim, "stack": False}
        else:
            raise NotImplementedError(
                f"checkpoint conversion {ops} for {key} is not supported")
    return direct, merged


@contextlib.contextmanager
def meta_parameters():
    """Construct a model with parameters on the meta device and buffers real.

    Used once, to harvest buffer values (buffer_values): buffers are
    what the model computes from its config at construction (rotary
    tables, embedding scales, softcaps), and the artifact must carry
    exactly those values rather than have the executor re-derive them.
    (The re-derived p-RoPE table of Gemma 4 was wrong past its first
    entries and long prompts diverged; see issue #228.) Export itself
    still runs on an all-meta model; the values are put back into the
    exported programs afterwards.

    Parameters are allocated on the CPU untouched (virtual memory only;
    transformers' no_init_weights keeps initializers from writing) and
    moved to meta as they are registered.
    """
    import torch.nn as nn
    try:
        from transformers.initialization import no_init_weights  # >= 5.x
    except ImportError:
        from transformers.modeling_utils import no_init_weights

    original = nn.Module.register_parameter

    def register_parameter(module, name, param):
        if param is not None and param.device.type != "meta":
            param = nn.Parameter(param.detach().to("meta"),
                                 requires_grad=param.requires_grad)
        original(module, name, param)

    nn.Module.register_parameter = register_parameter
    try:
        with no_init_weights():
            yield
    finally:
        nn.Module.register_parameter = original


def buffer_values(config, dtype):
    """{buffer name: tensor} for every buffer the model computes at
    construction, from a throwaway model with meta parameters."""
    with meta_parameters():
        probe = AutoModelForCausalLM.from_config(config, dtype=dtype)
    values = {name: tensor.detach().clone()
              for name, tensor in probe.named_buffers()
              if tensor.device.type != "meta"}
    del probe
    return values


def inject_buffer_values(exported, values, param_fqns, mutated):
    """Give the exported program's meta buffers their real values.

    Matches by peeling wrapper prefixes from the exported name (the
    wrapper holds the model as ``model``). Parameters and state are
    left alone. Returns the fqns that received values.
    """
    given = []
    for table in (exported.state_dict, exported.constants):
        for fqn in list(table):
            tensor = table[fqn]
            if (fqn in param_fqns or fqn in mutated
                    or not isinstance(tensor, torch.Tensor) or not tensor.is_meta):
                continue
            parts = fqn.split(".")
            for i in range(len(parts)):
                value = values.get(".".join(parts[i:]))
                if value is not None:
                    if tuple(value.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"buffer {fqn}: model computes shape {tuple(value.shape)}, "
                            f"graph expects {tuple(tensor.shape)}")
                    table[fqn] = value.to(tensor.dtype)
                    given.append(fqn)
                    break
    return given


VIEW_OPS = ("aten.slice.", "aten.select.", "aten.view.", "aten.reshape.",
            "aten.alias.", "aten.as_strided.", "aten.squeeze.", "aten.unsqueeze.",
            "aten.transpose.", "aten.permute.", "aten.expand.", "aten.narrow.",
            "aten._unsafe_view.", "aten.detach.")


def mutated_fqns(exported):
    """Names of tensors the graph writes in place: the model's state.

    Two forms: a functionalized program reports mutations in its
    signature; the export wrapper's cache updates survive as in-place
    aten ops (index_copy_, copy_) on a placeholder or a view of one, so
    those are traced back through view ops to their placeholder.
    """
    from torch.export.graph_signature import OutputKind
    sig = exported.graph_signature
    kinds = {OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION}
    mutated = {spec.target for spec in sig.output_specs
               if spec.kind in kinds and spec.target}

    placeholder_fqn = {}
    for mapping in (sig.inputs_to_parameters, sig.inputs_to_buffers,
                    sig.inputs_to_lifted_tensor_constants):
        placeholder_fqn.update(mapping)

    def root(node):
        seen = 0
        while node is not None and node.op == "call_function" and seen < 64:
            if not any(str(node.target).startswith(v) for v in VIEW_OPS):
                return None
            node = node.args[0] if node.args and hasattr(node.args[0], "op") else None
            seen += 1
        return node

    for node in exported.graph.nodes:
        if node.op != "call_function":
            continue
        parts = str(node.target).split(".")  # aten.index_copy_.default
        if len(parts) < 2 or not parts[1].endswith("_"):
            continue
        target = root(node.args[0]) if node.args and hasattr(node.args[0], "op") else None
        if target is not None and target.op == "placeholder":
            fqn = placeholder_fqn.get(target.target)
            if fqn is not None:
                mutated.add(fqn)
    return mutated


def strip_state_values(exported, state_fqns):
    """Replace state tensors' (zero) values with meta tensors before
    saving: a session's cache belongs to the executor, not the artifact."""
    for table in (exported.state_dict, exported.constants):
        for fqn in list(table):
            tensor = table[fqn]
            if fqn in state_fqns and isinstance(tensor, torch.Tensor) and not tensor.is_meta:
                table[fqn] = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")


def constant_aliases(exportable, *programs):
    """Lifted constants that are the same tensor as a wrapper buffer.

    Non-strict export captures a tensor reached through a Python object
    (the static cache's layers) as a lifted constant, even when the same
    tensor is also a registered buffer of the wrapper; both names end up
    in the graph. They are one tensor and must stay one on the executor
    (a cache written under one name and read under the other would not
    be a cache), so map each alias to the buffer it is.
    """
    # Identities come from the programs' own tables, under the names the
    # graphs use: a buffer is in state_dict (persistent) or constants
    # (non-persistent) under its signature name; a lifted constant that
    # is the same object as a buffer is an alias of it.
    aliases = {}
    for program in programs:
        sig = program.graph_signature
        by_id = {}
        for name in sig.inputs_to_buffers.values():
            tensor = program.state_dict.get(name, program.constants.get(name))
            if isinstance(tensor, torch.Tensor):
                by_id.setdefault(id(tensor), name)
        for name in sig.inputs_to_lifted_tensor_constants.values():
            canonical = by_id.get(id(program.constants.get(name)))
            if canonical is not None and canonical != name:
                if aliases.get(name, canonical) != canonical:
                    raise RuntimeError(
                        f"constant {name} aliases different buffers in the "
                        f"two graphs ({aliases[name]} vs {canonical})")
                aliases[name] = canonical
    return aliases


def _tensor_key(tensor):
    return (
        tensor.dtype,
        tuple(tensor.shape),
        hashlib.sha256(
            tensor.detach()
            .cpu()
            .contiguous()
            .view(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).digest(),
    )


def _rewire(program, dup, keep, ph_by_name, spec_by_target):
    del program.constants[dup]
    assert dup in spec_by_target and keep in spec_by_target, (
        f"missing input spec for constant rewire: {dup} -> {keep}"
    )
    dup_spec = spec_by_target.pop(dup)
    keep_spec = spec_by_target[keep]
    dup_node = ph_by_name.pop(dup_spec.arg.name)
    keep_node = ph_by_name[keep_spec.arg.name]
    dup_node.replace_all_uses_with(keep_node)
    program.graph.erase_node(dup_node)
    program.graph_signature.input_specs = [
        s for s in program.graph_signature.input_specs
        if s.arg.name != dup_node.name
    ]


def deduplicate_constants(aliases, *programs):
    """Deduplicate constant tensors with identical content across exported programs.

    Finds lifted constants and non-persistent buffers with identical dtype, shape,
    and byte content. For each distinct content, one canonical constant is kept,
    and all other occurrences in the graphs are rewritten to reference that
    canonical constant. The duplicates are removed from the programs' constants
    and added to `aliases`.
    """
    aliases = dict(aliases or {})
    exclude = set().union(*(mutated_fqns(p) for p in programs))
    exclude |= {a for a, c in aliases.items() if c in exclude}
    exclude |= {c for a, c in aliases.items() if a in exclude}

    # 1. Key every constant once; a name whose content differs between programs is never shared.
    key_of, conflicting = {}, set()
    for p in programs:
        for fqn, t in p.constants.items():
            if fqn in exclude or not isinstance(t, torch.Tensor) or t.is_meta:
                continue
            k = _tensor_key(t)
            if key_of.setdefault(fqn, k) != k:
                conflicting.add(fqn)

    # 2. One pass per program: the first name seen for a content is canonical everywhere.
    canon_of = {}
    for p in programs:
        ph_by_name = {n.name: n for n in p.graph.nodes if n.op == "placeholder"}
        spec_by_target = {s.target: s for s in p.graph_signature.input_specs if s.target}
        here = {}
        modified = False

        for fqn in list(p.constants):
            if fqn in exclude or fqn in conflicting or fqn not in key_of:
                continue
            k = key_of[fqn]
            canon = canon_of.setdefault(k, fqn)
            keep = canon if canon in p.constants else here.setdefault(k, fqn)
            if keep != fqn:
                _rewire(p, fqn, keep, ph_by_name, spec_by_target)
                aliases[fqn] = keep
                modified = True

        targets = [s.target for s in p.graph_signature.input_specs if s.target]
        assert len(targets) == len(set(targets)), (
            f"duplicate input spec targets in {p}: {targets}"
        )
        if modified:
            p.graph.lint()
            p.graph_module.recompile()

    # Flatten chained aliases
    for a in list(aliases):
        c = aliases[a]
        while c in aliases and aliases[c] != c:
            c = aliases[c]
        aliases[a] = c

    return aliases


def assert_no_duplicate_constants(program_or_path):
    """Assert that no two constants in an exported graph have identical content."""
    if isinstance(program_or_path, (str, os.PathLike)):
        program = torch.export.load(program_or_path)
    else:
        program = program_or_path
    seen = {}
    for fqn, val in program.constants.items():
        if isinstance(val, torch.Tensor) and not val.is_meta:
            key = _tensor_key(val)
            if key in seen:
                raise AssertionError(
                    f"duplicate constant in exported program: {fqn} has "
                    f"identical content to {seen[key]} (dtype={val.dtype}, shape={tuple(val.shape)})")
            seen[key] = fqn


MIN_PREFILL_LEN = 2


def used_fqns_of(exported):
    """Names of the tensors the program's graph reads (non-strict export
    lifts every tensor of the module tree, used or not)."""
    sig = exported.graph_signature
    used_placeholders = {
        node.target for node in exported.graph.nodes
        if node.op == "placeholder" and node.users}
    return {fqn for mapping in (sig.inputs_to_parameters, sig.inputs_to_buffers,
                                sig.inputs_to_lifted_tensor_constants)
            for placeholder, fqn in mapping.items() if placeholder in used_placeholders}


def classify(exported, manifest, view, aliases=None, also_mutated=(), also_used=()):
    """Classify every tensor the exported program references.

    Returns (bound, derived, state, stacked, unbound, unused, constant).
    ``state`` is what the graph mutates (caches); ``constant`` is a
    buffer whose values the artifact carries (rotary tables, scales);
    ``derived`` is a buffer without values, recomputed by the executor
    (artifacts from before real buffers were captured). ``view`` is the
    checkpoint as the model's parameters see it (checkpoint_view);
    ``stacked`` maps a merged parameter to its assembly spec. ``aliases``
    maps a lifted name to the tensor it is (constant_aliases); aliases
    are not classified themselves. ``also_used`` names tensors that
    another program of the artifact uses (used_fqns), so they are bound
    even where this one leaves them unused.
    """
    direct, merged = view
    aliases = dict(aliases or {})
    skip = set(aliases)
    sig = exported.graph_signature
    # Non-strict export lifts every buffer and constant of the module
    # tree, used or not (a text decode graph still names the vision and
    # audio towers' rotary tables and softcaps). Those placeholders have
    # no users; they get a placeholder tensor on the executor, nothing
    # is fetched or derived for them.
    # A tensor counts as used if any of its names is (a cache tensor is
    # read through its lifted-constant alias and written through its
    # buffer name, or the other way round).
    used_fqns = {aliases.get(fqn, fqn) for fqn in used_fqns_of(exported) | set(also_used)}
    # Tied weights collapse to one placeholder in the signature but the
    # state dict keeps both names — classify over the union.
    param_fqns = set(sig.inputs_to_parameters.values()) | set(
        exported.state_dict.keys())
    fqns = list(dict.fromkeys(
        list(param_fqns)
        + list(sig.inputs_to_buffers.values())
        + list(exported.constants.keys())))

    def resolve(fqn, table):
        # Wrapper nesting adds prefixes; match by peeling components.
        parts = fqn.split(".")
        for i in range(len(parts)):
            value = table.get(".".join(parts[i:]))
            if value is not None:
                return value
        return None

    def lookup(fqn):
        name = resolve(fqn, direct)
        return manifest["tensors"][name] if name is not None else None

    tied_ref = next(
        (ref for name, ref in manifest["tensors"].items()
         if name.endswith("embed_tokens.weight")), None)

    # State is whatever either graph writes: a linear-attention state
    # written by prefill and only read by decode is state all the same.
    mutated = {aliases.get(f, f) for f in mutated_fqns(exported) | set(also_mutated)}
    bound, derived, state, stacked, unbound, unused, constant = {}, [], [], {}, [], [], []
    for fqn in fqns:
        if fqn in skip:
            continue
        if aliases.get(fqn, fqn) not in used_fqns:
            unused.append(fqn)
            continue
        ref = lookup(fqn)
        if ref is None and fqn.endswith("lm_head.weight"):
            ref = tied_ref  # weight tying: an alias, not a second copy
        spec = resolve(fqn, merged) if ref is None else None
        if ref is not None:
            bound[fqn] = ref
        elif spec is not None:
            stacked[fqn] = spec
        elif fqn in param_fqns:
            unbound.append(fqn)
        elif aliases.get(fqn, fqn) in mutated:
            state.append(fqn)
        else:
            value = exported.state_dict.get(fqn, exported.constants.get(fqn))
            if isinstance(value, torch.Tensor) and not value.is_meta:
                constant.append(fqn)
            else:
                derived.append(fqn)  # pre-real-buffer artifacts
    return bound, derived, state, stacked, unbound, unused, constant


def export_artifact(repo_id, out_dir, max_cache_len=1024, revision="main", batch=1, step_tokens=None, draft=False):
    """Build the complete artifact directory for repo_id. ``batch`` sequences
    decode together, a decode step takes ``step_tokens`` tokens per row, and
    ``draft`` exports the checkpoint's own drafter (DeepSeek-V4.1 only, for
    now)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_manifest(repo_id, revision)
    if manifest["config"].get("model_type") == "deepseek_v41":
        # No transformers implementation: the model maker's own module,
        # from the fork in third_party (see deepseek_v41.py).
        return export_deepseek_v41(repo_id, out_dir, manifest, max_cache_len, revision, batch, step_tokens, draft)
    if batch != 1 or (step_tokens or 1) != 1 or draft:
        raise ValueError("only DeepSeek-V4.1 artifacts export batched, multi-token or draft decode so far")
    config = AutoConfig.from_pretrained(repo_id, revision=revision)

    quantization = fp8_config(config)
    # Size-oblivious reasoning for the prefill's symbolic length: without
    # it export case-splits "could this dimension be 1?" on derived sizes
    # (a chunk count of a linear-attention scan) and specializes the
    # length to the example's range.
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    # The model's compute dtype. Composite configs (text + vision) may
    # carry it only on the text config; from_pretrained would read the
    # checkpoint's dtypes, which a weightless export cannot.
    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    dtype = (getattr(config, "dtype", None)
             or getattr(text_config, "dtype", None) or torch.bfloat16)
    buffers = buffer_values(config, dtype)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
        # A model-maker fp8 checkpoint: transformers' own fp8 modules,
        # whose graph consumes the fp8 weights and block scales directly
        # (see fp8.py and third_party/transformers).
        view = checkpoint_view(model, manifest["tensors"])
        swapped = (apply_fp8(model, config, view, manifest["tensors"])
                   if quantization else 0)
        model.generation_config.cache_implementation = "static"
        exportable = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=max_cache_len)

        # A prefill of one token is served by the decode graph (see the
        # engine), so the prefill graph starts at two: the symbolic
        # reasoner otherwise case-splits "is the length 1?" at every
        # transposed-then-reshaped activation.
        seq = torch.export.Dim("seq", min=MIN_PREFILL_LEN, max=max_cache_len)
        # Experts: grouped_mm for prefill (routed pairs sorted by expert,
        # one grouped matmul per projection), batched_mm for decode (one
        # gathered block per pair). Both are static-shape; both are the
        # sparse routing transformers runs eagerly.
        set_experts_implementation(model, "grouped_mm")
        # Non-strict export runs the model's Python as is on fake tensors
        # (strict mode re-traces it through Dynamo, which trips on the
        # Python-heavy hybrid models); the captured graph is the same.
        prefill = exportable.export(
            input_ids=torch.zeros(1, 8, dtype=torch.long),
            cache_position=torch.arange(8),
            dynamic_shapes={"input_ids": {1: seq},
                            "cache_position": {0: seq}},
            strict=False)
        set_experts_implementation(model, "batched_mm")
        # Linear-attention / conv states branch in Python on whether a
        # previous state exists; the decode graph is the "yes" branch.
        set_state_flag = getattr(exportable.model, "set_has_previous_state", None)
        if set_state_flag is not None:
            set_state_flag(True)
        decode = exportable.export(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            cache_position=torch.zeros(1, dtype=torch.long),
            strict=False)

    aliases = constant_aliases(exportable, prefill, decode)
    state_fqns = set().union(*(mutated_fqns(p) for p in (prefill, decode)))
    state_fqns |= {a for a, c in aliases.items() if c in state_fqns}
    state_fqns |= {c for a, c in aliases.items() if a in state_fqns}
    for program in (prefill, decode):
        strip_state_values(program, state_fqns)
        params = set(program.graph_signature.inputs_to_parameters.values())
        inject_buffer_values(program, buffers, params, state_fqns)
    aliases = deduplicate_constants(aliases, prefill, decode)
    torch.export.save(prefill, os.path.join(out_dir, "prefill.pt2"))
    torch.export.save(decode, os.path.join(out_dir, "decode.pt2"))

    bound, derived, state, stacked, unbound, unused, constant = classify(
        decode, manifest, view, aliases=aliases,
        also_mutated=mutated_fqns(prefill))
    if unbound:
        raise RuntimeError(
            f"unbound tensors (artifact would be incomplete): {unbound}")
    binding = {"bound": bound, "derived": derived, "state": state,
               "stacked": stacked, "aliases": aliases, "unused": unused,
               "constant": constant,
               "max_cache_len": max_cache_len,
               "min_prefill_len": MIN_PREFILL_LEN}
    if quantization:
        binding["quantization"] = {
            "quant_method": "fp8",
            "block_size": list(quantization["block_size"]),
            "fp8_modules": swapped}

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "binding.json"), "w") as f:
        json.dump(binding, f, indent=1, sort_keys=True)
    return manifest, binding


def export_deepseek_v41(repo_id, out_dir, manifest, max_cache_len, revision, batch=1, step_tokens=None, draft=False):
    """The artifact for a DeepSeek-V4.1 checkpoint: the same steps as the
    transformers path, over the fork's module. With ``draft``, the
    checkpoint's DSpark drafter becomes draft.pt2, and a decode step
    verifies its block: step_tokens defaults to the block plus one."""
    from transformers import AutoTokenizer
    from . import deepseek_v41 as ds

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)  # the engram token map
    if draft:
        if not ds.has_drafter(ds.model_args(repo_id, max_cache_len)):
            raise ValueError(f"{repo_id} has no drafter (DSpark stages) to export")
        step_tokens = step_tokens or ds.model_args(repo_id, max_cache_len).dspark_block_size + 1
    step_tokens = step_tokens or 1
    model, buffers, args = ds.build(repo_id, tokenizer, max_cache_len, batch=batch, step_tokens=step_tokens)
    view = ds.checkpoint_view(model, manifest["tensors"])
    prefill, decode, drafter = ds.export_graphs(model, max_cache_len, MIN_PREFILL_LEN, batch=batch,
                                                step_tokens=step_tokens, drafter=draft)
    programs = [p for p in (prefill, decode, drafter) if p is not None]

    aliases = constant_aliases(ds.Prefill(model), *programs)
    state_fqns = set().union(*(mutated_fqns(p) for p in programs))
    state_fqns |= {a for a, c in aliases.items() if c in state_fqns}
    state_fqns |= {c for a, c in aliases.items() if a in state_fqns}
    for program in programs:
        strip_state_values(program, state_fqns)
        params = set(program.graph_signature.inputs_to_parameters.values())
        inject_buffer_values(program, buffers, params, state_fqns)
    aliases = deduplicate_constants(aliases, *programs)
    torch.export.save(prefill, os.path.join(out_dir, "prefill.pt2"))
    torch.export.save(decode, os.path.join(out_dir, "decode.pt2"))
    if drafter is not None:
        torch.export.save(drafter, os.path.join(out_dir, "draft.pt2"))

    bound, derived, state, stacked, unbound, unused, constant = classify(
        decode, manifest, view, aliases=aliases, also_mutated=mutated_fqns(prefill),
        also_used=used_fqns_of(drafter) if drafter is not None else ())
    if unbound:
        raise RuntimeError(f"unbound tensors (artifact would be incomplete): {unbound}")
    # The graphs call out for the expert banks and the n-gram tables
    # (ops.py); the engine fetches them from these specs and places them.
    external = ds.external_spec(model, view, args, drafter=draft)
    binding = {"bound": bound, "derived": derived, "state": state,
               "stacked": stacked, "aliases": aliases, "unused": unused,
               "constant": constant, "external": external,
               "max_cache_len": max_cache_len, "batch": batch,
               "min_prefill_len": MIN_PREFILL_LEN, **ds.binding_extras(args, step_tokens, draft)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "binding.json"), "w") as f:
        json.dump(binding, f, indent=1, sort_keys=True)
    return manifest, binding  # main() prints the summary, as for every exporter


def main():
    parser = argparse.ArgumentParser(
        description="Export a model to a vXPU artifact (no weights "
                    "downloaded)")
    parser.add_argument("repo_id")
    parser.add_argument("-o", "--out", required=True,
                        help="artifact output directory")
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch", type=int, default=1,
                        help="rows that decode together, each at its own position (DeepSeek-V4.1)")
    parser.add_argument("--step-tokens", type=int, default=None,
                        help="tokens per row in a decode step, to verify drafts (DeepSeek-V4.1; default 1)")
    parser.add_argument("--draft", action="store_true",
                        help="export the checkpoint's own drafter as draft.pt2 (DeepSeek-V4.1's DSpark)")
    args = parser.parse_args()

    manifest, binding = export_artifact(
        args.repo_id, args.out, args.max_cache_len, args.revision, args.batch, args.step_tokens, args.draft)
    total = sum(f["size"] for f in manifest["files"].values())
    print(f"artifact: {args.out}")
    print(f"  model: {args.repo_id} ({len(manifest['tensors'])} tensors, "
          f"{total / 1e9:.2f} GB of weights referenced, not included)")
    print(f"  binding: {len(binding['bound'])} bound, "
          f"{len(binding['derived'])} derived, "
          f"{len(binding['state'])} state, "
          f"{len(binding['stacked'])} stacked, "
          f"{len(binding['aliases'])} aliases, "
          f"{len(binding['unused'])} unused, "
          f"{len(binding['constant'])} constant"
          + (f" (fp8, {binding['quantization'].get('fp8_modules', '?')} modules)"
             if binding.get("quantization") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
