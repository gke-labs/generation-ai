"""Stage B': weightless export of the KV-cached generation graphs.

Two graphs from one wrapper (transformers' export-friendly module, which
registers a StaticCache as mutable buffers):

  prefill.pt2 — input_ids [1, S] (S dynamic) + cache_position [S]
  decode.pt2  — input_ids [1, 1] + cache_position [1], constant shapes
                so the executor can torch.compile it once and reuse it
                for every token.

The binding gains a third category beyond bound/derived: `state` —
cache buffers that come from nowhere: the executor zero-initializes
them, and the graphs mutate them in place.
"""

import argparse
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
)


def classify(ep, manifest):
    """Map every lifted parameter/buffer to a manifest ref, a config
    derivation, or executor-initialized state."""
    # Tied weights collapse to one placeholder in the signature, but the
    # state dict keeps both entries — classify over the union.
    sig = ep.graph_signature
    param_fqns = set(sig.inputs_to_parameters.values()) | set(
        ep.state_dict.keys()
    )
    fqns = list(
        dict.fromkeys(
            list(param_fqns)
            + list(sig.inputs_to_buffers.values())
            + list(ep.constants.keys())
        )
    )

    def lookup(fqn):
        # Wrapper nesting adds leading prefixes; match by peeling them.
        parts = fqn.split(".")
        for i in range(len(parts)):
            ref = manifest["tensors"].get(".".join(parts[i:]))
            if ref is not None:
                return ref
        return None

    tied_ref = (
        manifest["tensors"].get("model.embed_tokens.weight")
        if manifest["config"].get("tie_word_embeddings")
        else None
    )

    bound, derived, state, unbound = {}, [], [], []
    for fqn in fqns:
        ref = lookup(fqn)
        if ref is None and fqn.endswith("lm_head.weight"):
            ref = tied_ref
        if ref is not None:
            bound[fqn] = ref
        elif "rotary_emb" in fqn:
            derived.append(fqn)
        elif fqn not in param_fqns:
            # Every other buffer is runtime state: zero-initialized by
            # the executor, mutated by the graphs (KV cache, counters).
            state.append(fqn)
        else:
            unbound.append(fqn)
    return bound, derived, state, unbound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    config = AutoConfig.from_pretrained(args.repo_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config).eval()
        model.generation_config.cache_implementation = "static"
        exportable = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=args.max_cache_len
        )

        seq = torch.export.Dim("seq", min=1, max=args.max_cache_len)
        ep_prefill = exportable.export(
            input_ids=torch.zeros(1, 8, dtype=torch.long),
            cache_position=torch.arange(8),
            dynamic_shapes={
                "input_ids": {1: seq},
                "cache_position": {0: seq},
            },
        )
        ep_decode = exportable.export(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            cache_position=torch.zeros(1, dtype=torch.long),
        )

    import os

    for name, ep in (("prefill", ep_prefill), ("decode", ep_decode)):
        torch.export.save(ep, f"{name}.pt2")
        print(f"{name}.pt2: {len(list(ep.graph.nodes))} nodes, "
              f"{os.path.getsize(f'{name}.pt2') / 1e6:.1f} MB")

    bound, derived, state, unbound = classify(ep_decode, manifest)
    with open("cached.binding.json", "w") as f:
        json.dump(
            {"bound": bound, "derived": derived, "state": state,
             "unbound": unbound, "max_cache_len": args.max_cache_len},
            f, indent=1, sort_keys=True,
        )
    print(f"binding: {len(bound)} bound, {len(derived)} derived, "
          f"{len(state)} state (zero-init cache), unbound: {unbound or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
