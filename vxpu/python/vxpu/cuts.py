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

"""Where a model's step can be cut into ranges of whole layers (#387).

A layout that runs ranges of whole layers on different machines (the
planner's layer-ranges, #419) cuts the step between two layers: everything
before runs on one machine, everything after on the next, and what one side
computes for the other crosses the network once a step. A cut goes right
after a layer's experts node, so layer L's attention starts the next range.

Not every boundary can be cut. DeepSeek-V4's later layers read state an
earlier layer writes every step (layer 20's compressed KV cache, read by
layers 21-39): a cut between them would have to ship that cache both ways
every step. So for each boundary this reports:

  legal          no mutable state (the binding's "state": KV caches and the
                 like, written in place) is touched on both sides
  state          the state that is, when it is not
  message_bytes  the activations crossing it, a step (a decode step's: one
                 token's)
  copied         read-only tensors both sides read (weights, constants):
                 each range needs a copy

A graph is cut at a boundary only if every graph the step runs (decode and
prefill) can be: analyze both and take the intersection (legal_layers).
"""

import collections

import torch

from . import nodes, segments


def _nbytes(val):
    try:
        return int(val.numel()) * val.element_size()
    except (AttributeError, TypeError):
        return 0


def analyze(split, binding):
    """{layer: {legal, state, message_bytes, copied}} for each boundary
    before an experts layer but the first, in a split step graph."""
    # a resident read under an alias is the same tensor as its canonical name
    aliases = binding.get("aliases") or {}
    state = {aliases.get(s, s) for s in binding.get("state") or []}
    order = segments.segments_of(split)
    pos = {name: i for i, (name, _) in enumerate(order)}
    subs = dict(order)
    # where each experts layer's node sits: a cut before layer L goes after layer L-1's
    experts_at = {}
    for name, sub in order:
        if nodes.is_native(sub):
            kind, index = nodes.host_call(sub)
            if kind == "experts":
                experts_at[index] = pos[name]
    # each value a segment produces (its call, or an item of its outputs), and who reads it
    produced = {}
    touched = collections.defaultdict(set)  # resident -> positions of the segments that take it
    reads = []  # (producing position, value, reading position, its placeholder)
    for n in split.graph.nodes:
        if n.op == "call_module" and n.target in pos:
            produced[n] = pos[n.target]
            phs = [p for p in subs[n.target].graph.nodes if p.op == "placeholder"]
            for a, ph in zip(n.args, phs):
                if not isinstance(a, torch.fx.Node):
                    continue
                if a.op == "get_attr":
                    touched[aliases.get(a.target, a.target)].add(pos[n.target])
                elif a in produced:
                    reads.append((produced[a], a, pos[n.target], ph))
        elif n.op == "call_function" and n.args and isinstance(n.args[0], torch.fx.Node) and n.args[0] in produced:
            produced[n] = produced[n.args[0]]  # an item of a segment's outputs
    out = {}
    for layer in sorted(experts_at):
        if layer - 1 not in experts_at:
            continue
        cut = experts_at[layer - 1] + 1  # the first position after the cut
        spanning = sorted(r for r, ps in touched.items() if min(ps) < cut <= max(ps))
        crossing = {}
        for src, value, dst, ph in reads:
            if src < cut <= dst:
                crossing[value] = _nbytes(ph.meta.get("val"))
        bad = [r for r in spanning if r in state]
        out[layer] = {"legal": not bad, "state": bad, "message_bytes": sum(crossing.values()),
                      "copied": [r for r in spanning if r not in state]}
    return out


def legal_layers(*analyses):
    """The layers a step can be cut before, in every graph analysed."""
    common = None
    for a in analyses:
        ok = {layer for layer, c in a.items() if c["legal"]}
        common = ok if common is None else common & ok
    return sorted(common or [])


def describe(artifact_dir):
    """The cuts of an artifact's decode and prefill steps: {"decode": ...,
    "prefill": ..., "legal": [layers]}."""
    import json
    import os
    from . import ops  # noqa: F401  (registers the vxpu ops the graphs call)
    with open(os.path.join(artifact_dir, "binding.json")) as f:
        binding = json.load(f)
    out = {}
    seg_defs = segments.read_artifact_segments(artifact_dir)
    for graph in ("decode", "prefill"):
        path = os.path.join(artifact_dir, graph + ".pt2")
        if not os.path.exists(path):
            continue
        split, _ = segments.split_by_segments(torch.export.load(path).module(), seg_defs)
        out[graph] = analyze(split, binding)
    out["legal"] = legal_layers(*(out[g] for g in ("decode", "prefill") if g in out))
    return out


def main(argv=None):
    """python -m vxpu.cuts ARTIFACT_DIR: the cuts, as JSON."""
    import argparse
    import json
    p = argparse.ArgumentParser(prog="python -m vxpu.cuts", description=__doc__.split("\n\n")[0])
    p.add_argument("artifact")
    args = p.parse_args(argv)
    print(json.dumps(describe(args.artifact), indent=1))


if __name__ == "__main__":
    main()
