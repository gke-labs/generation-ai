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

"""An expert placement that spreads every layer's experts over the GPUs (hero run 14: the
experts per card). Each layer's experts go to the GPUs in a fixed rotation by id (expert e of
every layer to GPU e mod G), so each card holds a quarter of every layer and the memory is
balanced; a popularity-balanced variant takes the expert trace's counts and assigns experts to
cards greedily by load.

    python ep_placement.py --from placement-ranges4c.json --gpus 4 -o placement-ep4.json
    python ep_placement.py --from placement-ranges4c.json --gpus 4 --counts trace-counts.json -o placement-ep4-pop.json

The input placement supplies the layers and their expert ids (and the bytes per expert); the
output is the same format (vxpu-placement/2) with the ids redistributed."""

import argparse
import json


def spread(layers, gpus, counts=None):
    """{layer: [ids]} -> [{layer: [ids]} per GPU]. Without counts, expert e goes to GPU e mod
    gpus. With counts ({layer: {expert: selections}}), the experts of each layer are assigned
    most popular first to the GPU with the least load so far, ties by id."""
    out = [dict() for _ in range(gpus)]
    for layer, ids in sorted(layers.items()):
        if counts and counts.get(layer):
            # most popular first to the least loaded card, each card capped at its equal share of the
            # layer's experts so the memory stays balanced too; the never-selected experts fill the rest
            c = counts[layer]
            cap = -(-len(ids) // gpus)
            order = sorted(ids, key=lambda e: (-float(c.get(str(e), c.get(e, 0))), e))
            load = [0.0] * gpus
            held = [[] for _ in range(gpus)]
            for e in order:
                g = min((k for k in range(gpus) if len(held[k]) < cap), key=lambda k: (load[k], k))
                held[g].append(e)
                load[g] += float(c.get(str(e), c.get(e, 0)))
            for g in range(gpus):
                out[g][layer] = sorted(held[g])
        else:
            for g in range(gpus):
                out[g][layer] = [e for e in ids if e % gpus == g]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from", dest="src", required=True, help="a placement whose layers and ids to redistribute")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--counts", default=None, help="{layer: {expert: selections}} to balance by popularity")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args(argv)
    doc = json.load(open(a.src))
    src_gpus = doc["gpus"] if "gpus" in doc else [doc]
    layers = {}
    total_bytes = total_experts = 0
    for g in src_gpus:
        for layer, ids in g["layers"].items():
            layers.setdefault(int(layer), []).extend(int(e) for e in ids)
        total_bytes += g.get("bytes", 0)
        total_experts += g.get("experts", 0)
    per_expert = total_bytes / total_experts if total_experts else 0
    counts = None
    if a.counts:
        counts = {int(k): v for k, v in json.load(open(a.counts)).items()}
    gpus = spread(layers, a.gpus, counts)
    out = {"format": "vxpu-placement/2", "gpus": []}
    for g in gpus:
        n = sum(len(v) for v in g.values())
        out["gpus"].append({"layers": {str(k): v for k, v in sorted(g.items())}, "experts": n,
                            "bytes": int(n * per_expert)})
    json.dump(out, open(a.out, "w"))
    print(f"{a.gpus} GPUs, {len(layers)} layers, experts per GPU "
          f"{[g['experts'] for g in out['gpus']]}, GB per GPU {[round(g['bytes'] / 1e9, 1) for g in out['gpus']]}")


if __name__ == "__main__":
    main()
