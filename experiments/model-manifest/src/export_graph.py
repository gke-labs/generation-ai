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
    def lookup(fqn):
        # Wrapper nesting adds leading prefixes; match by peeling them.
        parts = fqn.split(".")
        for i in range(len(parts)):
            ref = manifest["tensors"].get(".".join(parts[i:]))
            if ref is not None:
                return ref
        return None

    # Weight tying: the alias resolves to whatever embed_tokens tensor the
    # checkpoint stores (its nesting varies across architectures).
    tied_ref = next(
        (r for n, r in manifest["tensors"].items()
         if n.endswith("embed_tokens.weight")), None)
    # Buffers computed from config at init, never stored in checkpoints.
    DERIVED_MARKERS = ("rotary_emb", "embed_scale", "softcap", "inv_timescales")
    bound, derived, unbound = {}, [], []
    for fqn in param_fqns + buffer_fqns:
        ref = lookup(fqn)
        if ref is None and fqn.endswith("lm_head.weight"):
            ref = tied_ref
        if ref is not None:
            bound[fqn] = ref
        elif any(marker in fqn for marker in DERIVED_MARKERS):
            derived.append(fqn)
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
