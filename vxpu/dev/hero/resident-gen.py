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

# The resident kernel generated per work list (the compiled-query form): every node's shapes and
# tile count are constexprs and its tensors are kernel arguments, so each node's K loop is a
# straight-line pipelined block, against the launched graph and the data-driven resident kernel.
import os, sys, importlib, tempfile, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vxpu import resident
import importlib.util
spec = importlib.util.spec_from_file_location("bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "resident-bench.py")); bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

POLL = 'atomic'

def generate(wl, deps=True):
    n = wl.count
    args = ", ".join(f"x{i}, w{i}, s{i}, y{i}" for i in range(n))
    consts = ", ".join(f"M{i}: tl.constexpr, N{i}: tl.constexpr, K{i}: tl.constexpr, KS{i}: tl.constexpr, T{i}: tl.constexpr" for i in range(n))
    body = []
    for i, (x, w, s, y, d) in enumerate(wl.nodes):
        if deps and d:
            for dep in d:
                if POLL == "volatile":
                    body.append(f"    while tl.load(done_ptr + {dep}, volatile=True) < P:\n        pass\n    tl.atomic_add(done_ptr + {dep}, 0, sem='acquire')")
                else:
                    body.append(f"    while tl.atomic_add(done_ptr + {dep}, 0, sem='acquire') < P:\n        pass")
        body.append(f"    for t in range(pid, T{i}, P):\n        _gemv_tile(x{i}, w{i}, s{i}, y{i}, M{i}, N{i}, K{i}, KS{i}, t, BM, BN, BK)")
        body.append(f"    tl.atomic_add(done_ptr + {i}, 1, sem='release')")
    src = f"""import triton
import triton.language as tl
from vxpu.resident import _gemv_tile

@triton.jit
def step_kernel({args}, done_ptr, P, {consts}, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
{os.linesep.join(body)}
"""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "gen_step.py")
    open(path, "w").write(src)
    sys.path.insert(0, d)
    import uuid
    name = "gen_step_" + uuid.uuid4().hex[:8]
    os.rename(path, os.path.join(d, name + ".py"))
    mod = importlib.import_module(name)
    return mod.step_kernel, src

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
sms = torch.cuda.get_device_properties(dev).multi_processor_count
for with_deps, poll in ((True, "volatile"), (False, "atomic"), (True, "atomic")):
    POLL = poll
    wl = bench.chain(dev, 8, 1, bench.LAYER, deps=with_deps)
    nb = bench.bytes_of(wl)
    kernel, src = generate(wl, deps=with_deps)
    tensors = [t for node in wl.nodes for t in node[:4]]
    consts = {}
    for i, (x, w, s_, y, _) in enumerate(wl.nodes):
        consts[f"M{i}"] = x.shape[0]; consts[f"N{i}"] = w.shape[0]; consts[f"K{i}"] = w.shape[1]
        consts[f"KS{i}"] = s_.shape[1]; consts[f"T{i}"] = triton.cdiv(w.shape[0], wl.bn)
    for P in (4 * sms,):
        for ns in (4,):
            def step():
                wl.done.zero_()
                kernel[(P,)](*tensors, wl.done, P, **consts, BM=wl.bm, BN=wl.bn, BK=wl.bk, num_warps=4, num_stages=ns)
            med, p90 = bench.time_graph(step)
            torch.cuda.synchronize()
            err = max((y.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-6) for (_, _, _, y, _), r in zip(wl.nodes, wl.reference()))
            print(f"generated deps={with_deps} poll={poll} P={P} stages={ns}: {med:.3f} ms ({nb / med / 1e6:.0f} GB/s), p90 {p90:.3f}, err {err:.1e}", flush=True)
    med, _ = bench.time_graph(lambda: wl.run_launched(num_stages=4))
    print(f"launched stages=4: {med:.3f} ms ({nb / med / 1e6:.0f} GB/s)", flush=True)
    del wl; torch.cuda.empty_cache()
# registers, Triton 3.8: the device caches on the JITFunction
for name, k in (("node", resident._node_kernel), ("static", resident._resident_static_kernel)):
    try:
        caches = getattr(k, "device_caches", None) or getattr(k, "cache", None)
        found = []
        for c in (caches.values() if isinstance(caches, dict) else []):
            entries = c[0].values() if isinstance(c, tuple) else (c.values() if isinstance(c, dict) else [])
            for ck in entries:
                found.append((getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)))
        print(name, "regs/spills", found[:6])
    except Exception as e:
        print(name, "regs: n/a", str(e)[:80])
print("DONE")
