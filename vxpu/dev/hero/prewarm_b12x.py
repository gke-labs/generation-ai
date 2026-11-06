"""Pre-warm the b12x dense option's autotune selection cache for the decode step's 14 plain fp8_linear shapes at the
capacities the one-stream and width programs run (1..128), so another pod with the same card copies a complete cache
and loads under VXPU_B12X_CACHE_ONLY=1 with no race. Random weight values: the race's choice does not depend on them."""
import os, sys, time, torch
sys.path.insert(0, os.environ.get("VXPU_PY", "/mnt/ssd/src-b12x-dense"))
os.environ["VXPU_VERIFY_ARTIFACTS"] = ""
from vxpu import b12x_dense
dev = torch.device("cuda", 0)
shapes = [(25600, 6144), (5120, 15360), (4096, 5120), (1024, 1280), (512, 5120)]
caps = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "1,2,4,8,16,32,64,128").split(",")]
if "--from-sites" in sys.argv:  # the union of every (K, N, capacity) any load prepared (b12x_dense._record_site)
    sites = os.path.join(b12x_dense.CACHE_DIR, "sites.txt")
    seen = set()
    for line in open(sites):
        K, N, cap = (int(v) for v in line.split())
        seen.add((N, K))
    shapes = sorted(set(shapes) | seen)
    print(f"PREWARM shapes from {sites}: {len(shapes)}", flush=True)
t0 = time.perf_counter()
for N, K in shapes:
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn); s = torch.ones(N // 32, K // 32, device=dev).to(torch.float8_e8m0fnu)
    packed = b12x_dense._pack(w, s)
    for cap in caps:
        t1 = time.perf_counter(); b12x_dense._plan(dev, K, N, cap, packed); torch.cuda.synchronize()
        print(f"PREWARM {N}x{K} cap {cap}: {time.perf_counter()-t1:.1f}s", flush=True)
    del w, s, packed; torch.cuda.empty_cache()
print(f"PREWARM done in {time.perf_counter()-t0:.0f}s; cache {b12x_dense.CACHE_DIR}", flush=True)
