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

# The barrier's cost against the program count: the generated resident kernel with and without the
# dependency barriers at 188, 376 and 752 programs (one launch for the 64-node chain), on GPU 0.
import os, sys, torch, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "resident-gen.py"))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resident-gen.py")).read()
# reuse the generator and the bench helpers without running the module's main sweep
head = src[:src.index("dev = torch.device")]
ns = {}
exec(compile(head, "gen-head", "exec"), ns)
generate, bench = ns["generate"], ns["bench"]
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
sms = torch.cuda.get_device_properties(dev).multi_processor_count
for with_deps in (False, True):
    wl = bench.chain(dev, 8, 1, bench.LAYER, deps=with_deps)
    nb = bench.bytes_of(wl)
    kernel, _ = generate(wl, deps=with_deps)
    tensors = [t for node in wl.nodes for t in node[:4]]
    consts = {}
    for i, (x, w, s_, y, _) in enumerate(wl.nodes):
        consts[f"M{i}"] = x.shape[0]; consts[f"N{i}"] = w.shape[0]; consts[f"K{i}"] = w.shape[1]
        consts[f"KS{i}"] = s_.shape[1]; consts[f"T{i}"] = triton.cdiv(w.shape[0], wl.bn)
    for P in (sms, 2 * sms, 4 * sms):
        def step():
            wl.done.zero_()
            kernel[(P,)](*tensors, wl.done, P, **consts, BM=wl.bm, BN=wl.bn, BK=wl.bk, num_warps=4, num_stages=4)
        med, p90 = bench.time_graph(step)
        print(f"generated deps={with_deps} P={P}: {med:.3f} ms ({nb / med / 1e6:.0f} GB/s), p90 {p90:.3f}", flush=True)
    del wl; torch.cuda.empty_cache()
print("DONE")
