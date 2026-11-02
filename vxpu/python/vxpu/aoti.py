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


"""Dense decode segments compiled ahead of time and run in our own CUDA
graphs (#277).

With the plan lowered into the decode graph (lower.py), a dense segment
is torch ops and Triton kernels only, so AOTInductor can compile it to a
package: native code with the kernels inside, no Python at run time. This
replaces torch.compile for those segments (the plan's segment_runtime =
aoti):

  build    on a segment's first call, torch.export on the arguments it
           was called with, then aoti_compile_and_package, in inductor's
           deterministic mode: no choice that changes numerics is made by
           timing kernels on the device, so a package computes the same
           bits whichever machine built it (and costs nothing measurable).
           Packages are cached on disk under a key of the segment's code
           with its values named by position (so the layers that compute
           the same thing on different weights share one package), its
           argument shapes, dtypes and strides, the kernels' source, the
           torch version, the device it is compiled for (the GPU, or the
           CPU's features) and the inductor settings that change
           numerics: the compiled plan, reused by every later load of the
           same artifact on the same kind of machine
  graph    the package is loaded single-threaded (the default runner's
           model pool queries CUDA events, which a capture forbids); the
           first call runs it eagerly, the second captures it into a CUDA
           graph and replays it, later calls replay. Every call executes
           the segment exactly once: segments update the caches in place.
           Arguments that are the engine's own tensors (weights, the
           resident state: attributes of the split graph) are passed by
           address; the others (the step's inputs, other segments'
           outputs) are copied into the graph's input buffers first.

A replay's outputs are the graph's own buffers, rewritten by the next
replay of the same segment, one step later: the host ops and the next
segments read them within the step, and the step's result is cloned
(segments.Split).
"""

import hashlib
import os
import threading
import time

import torch

from . import triton_ops

FORMAT = "vxpu-aoti/1"  # in the cache key: a change here invalidates every package
NUMERICS_CONFIG = ("deterministic", "prologue_fusion", "emulate_precision_casts")


def canonical_code(sub, args_of=None):
    """The segment's graph with its values named by position: what it
    computes, without the names it was given. Layer 9's attention and
    layer 10's differ only in names (their weights are arguments), so
    they share one package. Targets and literal arguments are kept, so
    graphs that compute different things stay apart. args_of: {node:
    args} to render in place of a node's own (a native node's host call,
    whose module index the lowering consumes)."""
    names = {n: f"v{i}" for i, n in enumerate(sub.graph.nodes)}
    lines = []
    for n in sub.graph.nodes:
        t = n.target
        if n.op == "placeholder":
            target = ""  # an input's name: its position is what counts
        elif isinstance(t, str):
            target = t
        elif isinstance(t, torch._ops.OpOverload):
            target = str(t)
        else:
            target = f"{getattr(t, '__module__', '')}.{getattr(t, '__qualname__', repr(t))}"
        args = (args_of or {}).get(n, n.args)
        render = torch.fx.node.map_arg((args, n.kwargs), lambda a: names[a])
        lines.append(f"{names[n]} = {n.op} {target} {render!r}")
    return "\n".join(lines)


def _target_device(args):
    """What a package is compiled for: the GPU (its name, compute
    capability, CUDA version) when it runs on one, else the CPU, which
    inductor compiles for natively (-march=native): its feature flags."""
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_cuda:
            props = torch.cuda.get_device_properties(a.device)
            return f"{props.name} {props.major}.{props.minor} cuda {torch.version.cuda}"
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith(("flags", "Features"))), "")
    except OSError:
        import platform
        flags = platform.processor()
    return "cpu " + " ".join(sorted(flags.split(":", 1)[-1].split()))


def cache_key(sub, args, extra="", code=None):
    """The content key of a segment's package: extra is what else it
    depends on (a native node's lowering), code its graph's code (by
    default canonical_code)."""
    h = hashlib.sha256()
    h.update(FORMAT.encode())
    h.update((code if code is not None else canonical_code(sub)).encode())
    h.update(extra.encode())
    for a in args:
        if isinstance(a, torch.Tensor):
            h.update(f"T{tuple(a.shape)}{a.stride()}{a.dtype}{a.device.type}".encode())
        else:
            h.update(repr(a).encode())
    import inspect
    h.update(inspect.getsource(triton_ops).encode())
    h.update(torch.__version__.encode())
    import torch._inductor.config as inductor_config
    for name in NUMERICS_CONFIG:  # inductor settings that change what a package computes
        h.update(f"{name}={getattr(inductor_config, name, None)}".encode())
    h.update(_target_device(args).encode())
    return h.hexdigest()[:32]


def _report_device_mix(sub, args):
    """A segment's export failed on tensors of two devices: replay the segment eagerly under a
    dispatch mode that names the first operator whose arguments sit on different devices, with
    their shapes and devices, and say where the segment's arguments are. Diagnostic only."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    where = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            where.append(f"arg{i}:{tuple(a.shape)}@{a.device}")
    print(f"[vxpu] segment export mixed devices; its arguments: {' '.join(where)}", flush=True)

    class Report(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), kw=None):
            kw = kw or {}
            flat, _ = tree_flatten((a, kw))
            ts = [t for t in flat if isinstance(t, torch.Tensor) and t.device.type == "cuda"]
            devs = {t.device for t in ts}
            if len(devs) > 1:
                print(f"[vxpu] first mixed op: {func} with " + " ".join(f"{tuple(t.shape)}{t.dtype}@{t.device}" for t in ts),
                      flush=True)
                raise RuntimeError(f"mixed devices at {func}")
            return func(*a, **kw)

    try:
        with Report():
            sub(*args)
        print("[vxpu] the eager replay mixed no devices: the mix is in export's own placement", flush=True)
    except Exception as e:
        print(f"[vxpu] eager replay: {type(e).__name__}: {str(e)[:200]}", flush=True)



_BUILD_LOCK = threading.Lock()


class Packages:
    """The package cache: a directory of <key>.pt2."""

    def __init__(self, cache_dir, expected=0):
        # made and probed here, while the engine loads, so a directory that
        # cannot be written fails the load, not the first request
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        probe = os.path.join(self.cache_dir, f".probe.{os.getpid()}")
        with open(probe, "w"):
            pass
        os.remove(probe)
        self.expected = expected  # segments that will ask: the summary is printed when all have
        self.built = self.loaded = 0
        self.build_s = 0.0
        # the key names the kernel source too: a lowered segment's canonical code renders a
        # vxpu_triton.* node the same whatever the kernel behind it became, and a package built
        # from an older kernel source (or an older lowering choice at the same graph) was served
        # for the engram layers' projections (#490: the compiler-fused reduction ran where the
        # GEMV had been chosen)
        from . import lower
        self.extra = f" kernels={lower.kernel_digest()}"

    def get(self, sub, args, extra="", code=None):
        """The segment's package, loaded (built first if not cached)."""
        path = self.path(sub, args, extra, code)
        index = next((a.device.index for a in args if isinstance(a, torch.Tensor) and a.is_cuda), None)
        runner = torch._inductor.aoti_load_package(path, run_single_threaded=True,
                                                   device_index=-1 if index is None else index)
        if self.built + self.loaded == self.expected:
            print(f"[vxpu] AOTInductor segments: {self.loaded} loaded from the cache, {self.built} built"
                  + (f" in {self.build_s:.0f} s" if self.built else ""), flush=True)
        return runner

    def path(self, sub, args, extra="", code=None, dynamic_shapes=None):
        """The segment's package file, built first if not cached."""
        import torch._inductor.config as inductor_config
        with inductor_config.patch(deterministic=True):
            return self._path(sub, args, extra + getattr(self, "extra", "") + repr(dynamic_shapes), code, dynamic_shapes)

    def _path(self, sub, args, extra, code, dynamic_shapes=None):
        path = os.path.join(self.cache_dir, cache_key(sub, args, extra, code) + ".pt2")
        # one build at a time in this process: torch.export keeps tracing state in module globals,
        # and the ranks of a tensor-parallel step reach their first (eager) step on threads of
        # their own (parallel.World), each building the packages its segments lack
        with _BUILD_LOCK:
            return self._build(path, sub, args, dynamic_shapes)

    def _build(self, path, sub, args, dynamic_shapes):
        if os.path.exists(path):
            self.loaded += 1
        else:
            if not self.built:
                print(f"[vxpu] building AOTInductor packages (once per artifact, plan and "
                      f"machine; kept in {self.cache_dir})", flush=True)
            t = time.perf_counter()
            # every node carries a "custom" meta entry before the re-trace: torch's annotate context
            # (entered by a prim wrapper while the tracer interprets a node) deletes the key on exit
            # when the node had none, and the tracer may have swapped its current meta by then
            # (KeyError: 'custom' on the four-rank drafter step's ring write, hero run 15); with
            # the key present it restores the entry instead
            for node in sub.graph.nodes:
                node.meta.setdefault("custom", {})
            try:
                ep = torch.export.export(sub, tuple(args), dynamic_shapes=dynamic_shapes, strict=False)
            except Exception as e:
                if "same device" in str(e):
                    _report_device_mix(sub, args)
                raise
            tmp = f"{path[:-len('.pt2')]}.{os.getpid()}.tmp.pt2"  # aoti_compile_and_package wants .pt2
            torch._inductor.aoti_compile_and_package(ep, package_path=tmp)
            os.replace(tmp, path)  # another process building the same key writes the same package
            self.build_s += time.perf_counter() - t
            self.built += 1
        return path


class GraphedSegment(torch.nn.Module):
    """One dense segment: its AOTInductor package, run in a CUDA graph.
    static[i] marks argument i as the engine's own tensor (same address
    every call)."""

    def __init__(self, name, sub, static, packages, pool):
        super().__init__()
        self.name, self.sub, self.static = name, sub, static
        self.packages, self.pool = packages, pool
        self.runner = None
        self.graph = None

    @property
    def _orig_mod(self):
        """The segment's graph, under the name a compiled module gives it
        (segments.segments_of, telemetry)."""
        return self.sub

    def forward(self, *args):
        if self.runner is None:
            self.runner = self.packages.get(self.sub, args)
            return self.runner(*args)  # the first call runs eagerly
        if self.graph is None:
            return self._capture(args)
        for i, a in enumerate(args):
            if not self.static[i] and isinstance(a, torch.Tensor):
                self.inputs[i].copy_(a)
        self.graph.replay()
        return self.outputs

    def _capture(self, args):
        inputs = [a.clone() if not self.static[i] and isinstance(a, torch.Tensor) else a
                  for i, a in enumerate(args)]
        device = next((a.device for a in inputs if isinstance(a, torch.Tensor) and a.is_cuda), None)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        # on the segment's GPU, with a capture stream of its own: torch.cuda.graph's default capture
        # stream is one stream for the process, made on whichever GPU captured first
        with torch.cuda.device(device), torch.cuda.graph(graph, pool=self.pool, stream=torch.cuda.Stream(device=device)):
            outputs = self.runner(*inputs)
        # kept only once captured: a capture that fails (out of memory, say)
        # leaves the segment as it was, and the next call tries again
        self.inputs, self.outputs, self.graph = inputs, outputs, graph
        graph.replay()  # capturing executed nothing: this is the call's one execution
        return outputs


def static_arguments(split, name):
    """For the call of submodule `name` in the split graph: which arguments
    are attributes of the graph (the engine's weights and resident state,
    the same tensors every call)."""
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == name:
            return [isinstance(a, torch.fx.Node) and a.op == "get_attr" for a in node.args]
    raise KeyError(name)
