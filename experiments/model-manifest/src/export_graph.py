"""Stage B: capture the model's graph WITHOUT weights.

The meta device instantiates the architecture with shapes/dtypes but no
storage, so the thin client can torch.export a frontier-scale model on a
laptop. The ExportedProgram lifts parameters as named inputs; we join
those FQNs against the stage-A manifest to bind every graph parameter to
its (file_sha256, offset, length, dtype, shape) content reference.

Interchange artifact = weightless graph (.pt2) + manifest + binding.
"""

import argparse
import collections
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM


class ForwardLogits(torch.nn.Module):
    """Stateless single forward returning logits (cache-free first cut)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--out", default="model.pt2")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    config = AutoConfig.from_pretrained(args.repo_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config).eval()
        wrapper = ForwardLogits(model)
        example = torch.zeros(1, args.seq_len, dtype=torch.long)

    seq = torch.export.Dim("seq", min=2, max=4096)
    ep = torch.export.export(
        wrapper, (example,), dynamic_shapes=((torch.export.Dim.STATIC, seq),)
    )

    # Join lifted parameters/buffers against the manifest's tensor table.
    sig = ep.graph_signature
    param_fqns = list(sig.inputs_to_parameters.values())
    buffer_fqns = list(sig.inputs_to_buffers.values())
    strip = lambda fqn: fqn.removeprefix("model.")  # ForwardLogits wrapper prefix
    # Weight tying: aliases resolved from config, not stored twice.
    tied = {"lm_head.weight": "model.embed_tokens.weight"} if getattr(
        config, "tie_word_embeddings", False) else {}
    bound, derived, unbound = {}, [], []
    for fqn in param_fqns + buffer_fqns:
        name = strip(fqn)
        ref = manifest["tensors"].get(name) or manifest["tensors"].get(tied.get(name, ""))
        if ref is not None:
            bound[fqn] = ref
        elif "rotary_emb" in fqn:
            derived.append(fqn)  # computed from config (RoPE frequencies)
        else:
            unbound.append(fqn)

    ops = collections.Counter(
        str(n.target) for n in ep.graph.nodes if n.op == "call_function"
    )
    layers = set()
    for n in ep.graph.nodes:
        for path, _cls in n.meta.get("nn_module_stack", {}).values():
            if "layers." in path:
                layers.add(path.split("layers.")[1].split(".")[0])

    torch.export.save(ep, args.out)
    with open(args.out + ".binding.json", "w") as f:
        json.dump(
            {"bound": bound, "derived": derived, "unbound": unbound},
            f, indent=1, sort_keys=True,
        )

    import os

    print(f"exported {args.repo_id} on meta device: {len(list(ep.graph.nodes))} nodes, "
          f"{len(param_fqns)} params, {len(buffer_fqns)} buffers")
    print(f"  recognized decoder layers: {len(layers)}")
    print(f"  top ops: {', '.join(f'{op.split('.')[-2]}x{n}' for op, n in ops.most_common(5))}")
    print(f"  binding: {len(bound)} bound to manifest, {len(derived)} derived from "
          f"config, unbound: {unbound or 'none'}")
    print(f"  artifact: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), "
          f"weights referenced: "
          f"{sum(t['length'] for t in bound.values()) / 1e6:.0f} MB (not included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
