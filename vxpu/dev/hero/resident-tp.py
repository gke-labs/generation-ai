# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hero run 15, the resident executor in the tensor-parallel step's regime, emulated shape-true
and node-count-true on four cards: every card walks 43 layers of the one-row TP4 + EP4 step's
dense chain (the real one-row profile's 29 kernels a layer: the quarter-width GEMVs at their
real shapes, the small kernels as 256 x 128 stand-ins that cost the per-node floor, the experts'
1.5 pairs a card as GEMVs of the same bytes) with the real vxpu::allreduce at the two sum points
a layer, captured per card and replayed together, two ways:

    launched: every node its own kernel, as the step graph does it (about 960 launches a card);
    resident: one generated kernel per segment between sums (one program per SM, a counter
              barrier only where a node depends on another, independent nodes without one),
              the all-reduce op between them (about 160 launches a card).

An emulation: the bodies are the plain fp8 GEMV, not the templates with their epilogues, so the
number says what the executor's form is worth against the graph's form at this node count and
these bytes, not what the real step will measure.

    python resident-tp.py [--layers 43] [--rounds 10] [--out results.json]
"""

import argparse
import importlib
import json
import os
import sys
import tempfile
import uuid

import torch
import triton

from vxpu import allreduce, resident

DIM = 5120
TINY = ("tiny", 256, 128)
# one layer of the one-row TP4 + EP4 step on one card, in the real profile's order, as
# (name, N, K, deps by name); the quarter-width shapes: columns in (N / 4), rows out (K / 4)
ATTN = [("recode_q", *TINY[1:], ()), ("wq_a", 320, 5120, ("recode_q",)), ("wkv", 128, 5120, ("recode_q",)),
        ("post_q", *TINY[1:], ("wq_a",)), ("post_kv", *TINY[1:], ("wkv",)), ("recode_qb", *TINY[1:], ("post_q",)),
        ("wq_b", 8192, 1280, ("recode_qb",)), ("attn", *TINY[1:], ("wq_b", "post_kv")), ("combine", *TINY[1:], ("attn",)),
        ("recode_o", *TINY[1:], ("combine",)), ("wo_a", 8192, 1024, ("recode_o",))]
FFN = [("wo_b", 5120, 2048, ()), ("stats", *TINY[1:], ("wo_b",)), ("dots", *TINY[1:], ("stats",)),
       ("front_b", *TINY[1:], ("dots",)), ("recode_s", *TINY[1:], ("front_b",)), ("w1", 576, 5120, ("recode_s",)),
       ("w3", 576, 5120, ("recode_s",)), ("swiglu", *TINY[1:], ("w1", "w3")), ("recode_2", *TINY[1:], ("swiglu",)),
       ("w2", 5120, 576, ("recode_2",)), ("gate_up", 3456, 5120, ("front_b",)), ("down", 2048, 5120, ("gate_up",)),
       ("post_f", *TINY[1:], ("down", "w2"))]


def make_weight(n, k, device):
    w = (torch.randn(n, k, device=device) * 0.05).to(torch.float8_e4m3fn)
    s = torch.full((n // 32, k // 32), 127, dtype=torch.uint8, device=device)
    s[::3] = 126
    return w, s


class Segment:
    """One segment's nodes on one card: tensors per layer, the generated kernel shared by every
    layer (same shapes), and the launched form."""

    def __init__(self, spec, device, layers, bn=16, bm=16, bk=128):
        self.spec, self.device, self.layers = spec, device, layers
        self.bn, self.bm, self.bk = bn, bm, bk
        self.names = [s[0] for s in spec]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.per_layer = []  # [(x, w, s, y) per node] per layer
        for _ in range(layers):
            nodes = []
            for name, n, k, deps in spec:
                x = torch.randn(1, k, device=device).to(torch.bfloat16)
                w, s = make_weight(n, k, device)
                y = torch.empty(1, n, dtype=torch.bfloat16, device=device)
                nodes.append((x, w, s, y))
            self.per_layer.append(nodes)
        self.done = torch.zeros(layers * len(spec), dtype=torch.int32, device=device)
        self.kernel = self._generate()

    def _generate(self):
        n = len(self.spec)
        args = ", ".join(f"x{i}, w{i}, s{i}, y{i}" for i in range(n))
        consts = ", ".join(f"N{i}: tl.constexpr, K{i}: tl.constexpr, KS{i}: tl.constexpr, T{i}: tl.constexpr" for i in range(n))
        body = []
        for i, (name, nn, kk, deps) in enumerate(self.spec):
            for d in deps:
                body.append(f"    while tl.atomic_add(done_ptr + {self.index[d]}, 0, sem='acquire') < P:\n        pass")
            body.append(f"    for t in range(pid, T{i}, P):\n        _gemv_tile(x{i}, w{i}, s{i}, y{i}, 1, N{i}, K{i}, KS{i}, t, BM, BN, BK)")
            body.append(f"    tl.atomic_add(done_ptr + {i}, 1, sem='release')")
        src = f"""import triton
import triton.language as tl
from vxpu.resident import _gemv_tile

@triton.jit
def segment_kernel({args}, done_ptr, P, {consts}, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
{os.linesep.join(body)}
"""
        d = tempfile.mkdtemp()
        mod = "seg_" + uuid.uuid4().hex[:8]
        open(os.path.join(d, mod + ".py"), "w").write(src)
        sys.path.insert(0, d)
        return importlib.import_module(mod).segment_kernel

    def consts(self):
        out = {}
        for i, (_, nn, kk, _) in enumerate(self.spec):
            out[f"N{i}"] = nn
            out[f"K{i}"] = kk
            out[f"KS{i}"] = kk // 32
            out[f"T{i}"] = triton.cdiv(nn, self.bn)
        return out

    def run_resident(self, layer, programs):
        tensors = [t for node in self.per_layer[layer] for t in node]
        done = self.done[layer * len(self.spec):(layer + 1) * len(self.spec)]
        self.kernel[(programs,)](*tensors, done, programs, **self.consts(), BM=self.bm, BN=self.bn, BK=self.bk,
                                 num_warps=4, num_stages=4)

    def run_launched(self, layer):
        for (x, w, s, y), (_, nn, kk, _) in zip(self.per_layer[layer], self.spec):
            resident._node_kernel[(triton.cdiv(nn, self.bn),)](x, w, s, y, 1, nn, kk, kk // 32,
                                                               BM=self.bm, BN=self.bn, BK=self.bk, num_warps=4, num_stages=4)

    def outputs(self):
        return [tuple(node[3].clone() for node in nodes) for nodes in self.per_layer]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--programs", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    n = torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(n)]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    programs = args.programs or sms
    sites = 2 * args.layers
    c = allreduce.setup(devices, sites=sites, max_elems=DIM, dtype=torch.bfloat16)
    streams = [torch.cuda.Stream(device=d) for d in devices]
    print(f"{torch.cuda.get_device_name(0)}: {n} cards, {sms} SMs, {programs} resident programs, {args.layers} layers", flush=True)
    attn = [Segment(ATTN, d, args.layers) for d in devices]
    ffn = [Segment(FFN, d, args.layers) for d in devices]
    xs = [torch.randn(1, DIM, dtype=torch.bfloat16, device=d) for d in devices]
    nodes_a_card = args.layers * (len(ATTN) + len(FFN))
    gb = sum(w.numel() + s.numel() for seg in (attn[0], ffn[0]) for nodes in seg.per_layer for _, w, s, _ in nodes) / 1e9
    print(f"{nodes_a_card} nodes and {sites} sums a card, {gb:.2f} GB of weights a card", flush=True)

    def step(k, form, sums=True):
        # eagerly, the sums are left out: a card's eager step of a thousand launches fills the
        # launch queue while its first sum spins for peers not yet launched (a deadlock the
        # captured replay, which starts all four cards' graphs at once, does not have)
        for layer in range(args.layers):
            if form == "resident":
                attn[k].run_resident(layer, programs)
            else:
                attn[k].run_launched(layer)
            if sums:
                allreduce.allreduce(xs[k], 2 * layer)
            if form == "resident":
                ffn[k].run_resident(layer, programs)
            else:
                ffn[k].run_launched(layer)
            if sums:
                allreduce.allreduce(xs[k], 2 * layer + 1)

    results = {"cards": n, "sms": sms, "programs": programs, "layers": args.layers, "nodes_a_card": nodes_a_card,
               "sums_a_card": sites, "weights_gb_a_card": gb}
    outs = {}
    for form in ("launched", "resident"):
        # warm eagerly on every card (kernels compiled before any capture), then capture one graph per card
        for k in range(n):
            with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                for seg in (attn[k], ffn[k]):
                    seg.done.zero_()
                step(k, form, sums=False)
            torch.cuda.synchronize(k)
        print(f"{form}: warm, capturing", flush=True)
        outs[form] = [(attn[k].outputs(), ffn[k].outputs()) for k in range(n)]
        graphs = []
        for k in range(n):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(k), torch.cuda.graph(g, stream=streams[k]):
                attn[k].done.zero_()
                ffn[k].done.zero_()
                step(k, form)
            graphs.append(g)
        allreduce._all_sync(c)
        times = []
        for r in range(args.rounds + 3):
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(streams[0])
            for k in range(n):
                with torch.cuda.device(k), torch.cuda.stream(streams[k]):
                    graphs[k].replay()
            en.record(streams[0])
            allreduce._all_sync(c)
            if r >= 3:
                times.append(st.elapsed_time(en))
        times.sort()
        med, p90 = times[len(times) // 2], times[int(len(times) * 0.9)]
        launches = nodes_a_card + sites if form == "launched" else 2 * args.layers + sites
        print(f"{form}: {med:.2f} ms a step (p90 {p90:.2f}), {launches} launches a card", flush=True)
        results[form] = {"ms": med, "p90": p90, "launches_a_card": launches}
        del graphs
    # the two forms agree node for node on every card
    same = all(torch.equal(a, b) for k in range(n) for sa, sb in zip(outs["launched"][k], outs["resident"][k])
               for la, lb in zip(sa, sb) for a, b in zip(la, lb))
    print(f"outputs identical between the forms on every card: {same}", flush=True)
    results["identical"] = same
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
