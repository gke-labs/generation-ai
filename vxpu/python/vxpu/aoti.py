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

  build    during engine load (before the engine reports ready),
           torch.export on example arguments for the segment, then
           aoti_compile_and_package, in inductor's deterministic mode: no
           choice that changes numerics is made by timing kernels on the
           device, so a package computes the same bits whichever machine
           built it (and costs nothing measurable). Packages are cached on
           disk under a key of the segment's code with its values named by
           position (so the layers that compute the same thing on different
           weights share one package), its argument shapes, dtypes and
           strides, the kernels' source, the torch version, the device it is
           compiled for (the GPU, or the CPU's features) and the inductor
           settings that change numerics: the compiled plan, reused by
           every later load of the same artifact on the same kind of machine
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

import collections
from collections.abc import Mapping
import hashlib
import multiprocessing as mp
import os
import queue
import shutil
import time
from typing import Any, Callable, NamedTuple
import zipfile

import torch

from . import triton_ops
from .localblobs import _alive

FORMAT = "vxpu-aoti/1"  # in the cache key: a change here invalidates every package
NUMERICS_CONFIG = ("deterministic", "prologue_fusion", "emulate_precision_casts")


def clean_cache_strays(cache_dir):
    """Remove temporary files and directories left in cache_dir by dead processes."""
    if not os.path.exists(cache_dir):
        return
    for name in os.listdir(cache_dir):
        if ".tmp" in name:
            prefix = name.split(".tmp", 1)[0]
            pid_string = prefix.rsplit(".", 1)[-1]
            if pid_string.isdigit():
                pid = int(pid_string)
                if 0 < pid <= 2147483647 and pid != os.getpid() and not _alive(pid):
                    target = os.path.join(cache_dir, name)
                    if os.path.isdir(target):
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        try:
                            os.remove(target)
                        except OSError:
                            pass


def extract(path):
    """Extract a .pt2 package archive once into a keyed .d directory beside it.

    Returns (dir_path, extracted_bytes). Reuses an existing directory.
    """
    if os.path.isdir(path) and path.endswith(".d"):
        return path, 0
    package_dir = (path[:-len(".pt2")] if path.endswith(".pt2") else path) + ".d"
    if os.path.isdir(package_dir):
        return package_dir, 0
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    staging_dir = f"{package_dir}.{time.time_ns()}.{os.getpid()}.tmp"
    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(staging_dir)
        extracted_bytes = sum(f.file_size for f in archive.infolist())
    try:
        os.replace(staging_dir, package_dir)
    except OSError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if not os.path.isdir(package_dir):
            raise
    return package_dir, extracted_bytes


def load_package(path):
    """Load an AOTInductor package from an extracted directory or .pt2 archive."""
    if os.path.isdir(path):
        from torch.export.pt2_archive._package import AOTICompiledModel
        loader = torch._C._aoti.AOTIModelPackageLoader(path, "model", True, 1, -1)
        return AOTICompiledModel(loader)
    return torch._inductor.aoti_load_package(path, run_single_threaded=True)


def open_package(cache_dir: str, key: str) -> Callable[..., Any]:
    """Opens a pre-staged AOTInductor package <cache_dir>/<key>.pt2 without compiling."""
    pt2 = os.path.join(os.path.abspath(cache_dir), f"{key}.pt2")
    if not os.path.isfile(pt2):
        raise FileNotFoundError(pt2)
    package_dir, _ = extract(pt2)
    return load_package(package_dir)


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
    # The package key names the Triton kernels' source and in-segment knobs:
    # a lowered segment's canonical code renders a vxpu_triton.* node the same
    # whatever the kernel source behind it became, and a package built under
    # different in-segment knobs was served for the engram layers' projections
    # (#490: the compiler-fused reduction ran where the GEMV had been chosen).
    from . import lower
    h.update(lower.kernel_digest().encode())
    h.update(torch.__version__.encode())
    import torch._inductor.config as inductor_config
    for name in NUMERICS_CONFIG:  # inductor settings that change what a package computes
        h.update(f"{name}={getattr(inductor_config, name, None)}".encode())
    h.update(_target_device(args).encode())
    return h.hexdigest()[:32]


def is_toolchain_compile_error(e: BaseException) -> bool:
    """Returns True if e represents a compiler toolchain error building the segment.

    Only compiler errors (e.g. CppCompileError, CUDACompileError,
    CppWrapperCodegenError, XPUCompileError in torch 2.14.1) are counted. Wrapping
    exceptions such as BackendCompilerFailed or InductorError are unwrapped to
    inspect the underlying exception; an error in our own lowering or execution
    fails the task rather than being recorded as a compilation failure.
    """
    import torch._inductor.exc as inductor_exc

    compiler_error_types: tuple[type[BaseException], ...] = (
        inductor_exc.CppCompileError,
        inductor_exc.CUDACompileError,
        inductor_exc.CppWrapperCodegenError,
        inductor_exc.XPUCompileError,
    )
    curr: BaseException | None = e
    while curr is not None:
        if isinstance(curr, compiler_error_types):
            return True
        inner = getattr(curr, "inner_exception", None)
        if isinstance(inner, BaseException):
            curr = inner
        elif curr.__cause__ is not None:
            curr = curr.__cause__
        elif curr.__context__ is not None:
            curr = curr.__context__
        else:
            break
    return False


def compile_package(ep: Any, pt2_path: str) -> None:
    """Compile an ExportedProgram with AOTInductor and atomically place it at pt2_path."""
    import torch._inductor.config as inductor_config
    stem = pt2_path[:-len(".pt2")] if pt2_path.endswith(".pt2") else pt2_path
    staging_file = f"{stem}.{time.time_ns()}.{os.getpid()}.tmp.pt2"
    try:
        with inductor_config.patch(deterministic=True):
            torch._inductor.aoti_compile_and_package(ep, package_path=staging_file)
            os.replace(staging_file, pt2_path)
    finally:
        if os.path.exists(staging_file):
            try:
                os.remove(staging_file)
            except OSError:
                pass


def _compile_worker(task: tuple[str, str, str]) -> tuple[str, str | None, bool]:
    """Worker process: loads an ExportedProgram from ep_path and compiles it."""
    name, ep_path, pt2_path = task
    try:
        if not os.path.exists(pt2_path):
            from . import nodes as _nodes, ops as _ops  # register custom ops (vxpu_native, vxpu) in worker process
            ep = torch.export.load(ep_path)
            compile_package(ep, pt2_path)
        return name, None, False
    except Exception as e:
        return name, str(e), is_toolchain_compile_error(e)
    finally:
        if os.path.exists(ep_path):
            try:
                os.remove(ep_path)
            except OSError:
                pass


def _worker_loop(worker_id: int, tq: Any, rq: Any) -> None:
    from . import nodes as _nodes, ops as _ops
    _nodes._register_fakes()
    while True:
        task = tq.get()
        if task is None:
            break
        name, err, is_compile_err = _compile_worker(task)
        rq.put((worker_id, name, err, is_compile_err))


def build_segment_package(
    sub: Any,
    args: tuple[Any, ...],
    pt2_path: str,
    dynamic_shapes: Any = None,
) -> None:
    """Compiles a segment submodule with AOTInductor to pt2_path.

    This is the shared build procedure used by both the compilation task
    and in-engine compilation during model load.
    """
    ep = torch.export.export(sub, tuple(args), dynamic_shapes=dynamic_shapes, strict=False)
    compile_package(ep, pt2_path)


def parse_package_arg(arg: str) -> tuple[str, str]:
    """Parses a single package argument in the form 'name=key'.

    Refuses any argument that is empty or does not follow 'name=key'.
    """
    if not isinstance(arg, str) or "=" not in arg:
        raise ValueError(f"malformed package argument {arg!r}: expected name=key")
    name, key = arg.split("=", 1)
    name = name.strip()
    key = key.strip()
    if not name or not key:
        raise ValueError(f"malformed package argument {arg!r}: name and key must not be empty")
    return name, key


class Packages:
    """The package cache: a directory of <key>.pt2 and extracted <key>.d."""

    def __init__(self, cache_dir: str, expected: int = 0, parallel_builds: int | None = None,
                 packages: Mapping[str, str] | None = None):
        # made and probed here, while the engine loads, so a directory that
        # cannot be written fails the load, not the first request
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        probe = os.path.join(self.cache_dir, f".probe.{os.getpid()}")
        with open(probe, "w"):
            pass
        os.remove(probe)
        clean_cache_strays(self.cache_dir)
        self.expected = expected  # segments that will ask: the summary is printed when all have
        self.parallel_builds = parallel_builds
        self.packages = dict(packages) if packages is not None else None
        self.failures: dict[str, str] = {}
        self._built_keys: set[str] = set()
        self._loaded_built_keys: set[str] = set()
        self._retrieved = 0
        self.built = self.loaded = 0
        self.build_s = 0.0
        self.extracted = 0
        self.extracted_bytes = 0

    def report(self):
        """Print the summary of packages loaded from cache or built."""
        print(f"[vxpu] AOTInductor segments: {self.loaded} loaded from the cache, {self.built} built"
              + (f" in {self.build_s:.0f} s" if self.built else "")
              + f", {self.extracted} extracted ({self.extracted_bytes / 1e9:.1f} GB)", flush=True)

    def get(self, sub, args, extra="", code=None, name=None):
        """The segment's package, loaded (built first if not cached)."""
        path = self.path(sub, args, extra, code, name=name)
        runner = load_package(path)
        self._retrieved += 1
        if self.expected and self._retrieved == self.expected:
            self.report()
        return runner

    def build_all(self, tasks: list[tuple[str, Any, tuple[Any, ...]]], allow_failures: bool = False) -> None:
        """Build any packages needed by tasks that are not yet cached, in parallel."""
        import torch._inductor.config as inductor_config
        with inductor_config.patch(deterministic=True):
            self._build_all(tasks, allow_failures=allow_failures)

    def _build_all(self, tasks: list[tuple[str, Any, tuple[Any, ...]]], allow_failures: bool = False) -> None:
        # Group tasks by cache key so segments that share a package are built once.
        unique_tasks: dict[str, tuple[str, Any, tuple[Any, ...], str, Any]] = {}
        key_to_names: dict[str, list[str]] = collections.defaultdict(list)
        name_to_key: dict[str, str] = {}
        for item in tasks:
            name, sub, args = item[0], item[1], item[2]
            dynamic_shapes = item[3] if len(item) > 3 else None
            extra = (item[4] if len(item) > 4 and item[4] is not None else "") + repr(dynamic_shapes)
            code = item[5] if len(item) > 5 else None
            if name and self.packages and name in self.packages:
                key = self.packages[name]
            else:
                key = cache_key(sub, args, extra=extra, code=code)
            key_to_names[key].append(name)
            name_to_key[name] = key
            if key not in unique_tasks:
                pt2 = os.path.join(self.cache_dir, key + ".pt2")
                package_dir = (pt2[:-len(".pt2")] if pt2.endswith(".pt2") else pt2) + ".d"
                if not os.path.exists(pt2) and not os.path.isdir(package_dir):
                    unique_tasks[key] = (name, sub, args, pt2, dynamic_shapes)

        if not unique_tasks:
            return

        if not self.built:
            print(f"[vxpu] building AOTInductor packages (once per artifact, plan and "
                  f"machine; kept in {self.cache_dir})", flush=True)

        t_start = time.perf_counter()
        to_build = list(unique_tasks.items())
        concurrency = self.parallel_builds if self.parallel_builds is not None else (os.cpu_count() or 1)
        concurrency = max(1, min(concurrency, len(to_build)))

        if concurrency == 1:
            for key, (name, sub, args, pt2, dynamic_shapes) in to_build:
                if os.path.exists(pt2):
                    self._built_keys.add(key)
                    continue
                try:
                    build_segment_package(sub, args, pt2, dynamic_shapes=dynamic_shapes)
                    self._built_keys.add(key)
                except Exception as e:
                    if allow_failures and is_toolchain_compile_error(e):
                        for n in key_to_names[key]:
                            self.failures[n] = str(e)
                        continue
                    raise RuntimeError(f"failed to load or build package for segment {name}: {e}") from e
        else:
            ep_files: list[str] = []
            compile_tasks: list[tuple[str, str, str]] = []
            try:
                for key, (name, sub, args, pt2, dynamic_shapes) in to_build:
                    stem = pt2[:-len(".pt2")] if pt2.endswith(".pt2") else pt2
                    ep_file = f"{stem}.ep.{time.time_ns()}.{os.getpid()}.tmp.pt2"
                    ep_files.append(ep_file)
                    try:
                        ep = torch.export.export(sub, tuple(args), dynamic_shapes=dynamic_shapes, strict=False)
                        torch.export.save(ep, ep_file)
                    except Exception as e:
                        if allow_failures and is_toolchain_compile_error(e):
                            for n in key_to_names[key]:
                                self.failures[n] = str(e)
                            continue
                        raise RuntimeError(f"failed to load or build package for segment {name}: {e}") from e
                    compile_tasks.append((name, ep_file, pt2))

                if not compile_tasks:
                    return

                actual_concurrency = min(concurrency, len(compile_tasks))
                ctx = mp.get_context("spawn")
                task_queues = [ctx.Queue() for _ in range(actual_concurrency)]
                rq = ctx.Queue()
                workers = [
                    ctx.Process(target=_worker_loop, args=(i, task_queues[i], rq))
                    for i in range(actual_concurrency)
                ]
                for w in workers:
                    w.start()

                active_task: dict[int, str] = {}
                pending = list(compile_tasks)
                for i in range(actual_concurrency):
                    if pending:
                        task = pending.pop(0)
                        active_task[i] = task[0]
                        task_queues[i].put(task)
                    else:
                        task_queues[i].put(None)

                completed = 0
                failed = None
                while completed < len(compile_tasks):
                    try:
                        worker_id, name, err, is_compile_err = rq.get(timeout=0.2)
                        completed += 1
                        if err:
                            if allow_failures and is_compile_err:
                                failed_key = name_to_key.get(name, "")
                                for n in key_to_names.get(failed_key, [name]):
                                    self.failures[n] = err
                            else:
                                failed = (name, err)
                                break
                        if pending:
                            task = pending.pop(0)
                            active_task[worker_id] = task[0]
                            task_queues[worker_id].put(task)
                        else:
                            active_task.pop(worker_id, None)
                            task_queues[worker_id].put(None)
                    except queue.Empty:
                        for i, w in enumerate(workers):
                            if not w.is_alive() and w.exitcode not in (0, None):
                                seg_name = active_task.get(i, "unknown")
                                failed = (seg_name, f"worker process {w.pid} died with exit code {w.exitcode}")
                                break
                        if failed:
                            break

                if failed:
                    for w in workers:
                        if w.is_alive():
                            w.terminate()
                    for w in workers:
                        w.join(timeout=5)
                        if w.is_alive():
                            w.kill()
                    raise RuntimeError(f"failed to load or build package for segment {failed[0]}: {failed[1]}")

                for w in workers:
                    w.join()

                for key, (name, *_) in to_build:
                    if name not in self.failures:
                        self._built_keys.add(key)
            finally:
                for p in ep_files:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        newly_built = sum(1 for _, (name, *_) in to_build if name not in self.failures)
        self.built += newly_built
        self.build_s += time.perf_counter() - t_start

    def path(self, sub, args, extra="", code=None, dynamic_shapes=None, name=None):
        """The segment's package directory (extracted from its .pt2), built first if not cached."""
        import torch._inductor.config as inductor_config
        with inductor_config.patch(deterministic=True):
            return self._path(sub, args, extra + repr(dynamic_shapes), code, dynamic_shapes, name=name)

    def _path(self, sub, args, extra, code, dynamic_shapes=None, name=None):
        if name and self.packages and name in self.packages:
            key = self.packages[name]
        else:
            key = cache_key(sub, args, extra, code)
        pt2 = os.path.join(self.cache_dir, key + ".pt2")
        if os.path.exists(pt2):
            if key in self._built_keys:
                if key in self._loaded_built_keys:
                    self.loaded += 1
                else:
                    self._loaded_built_keys.add(key)
            else:
                self.loaded += 1
        else:
            if not self.built:
                print(f"[vxpu] building AOTInductor packages (once per artifact, plan and "
                  f"machine; kept in {self.cache_dir})", flush=True)
            t = time.perf_counter()
            try:
                build_segment_package(sub, args, pt2, dynamic_shapes=dynamic_shapes)
            except Exception:
                raise
            self.build_s += time.perf_counter() - t
            self.built += 1
            self._built_keys.add(key)
            self._loaded_built_keys.add(key)
        package_dir, extracted_bytes = extract(pt2)
        if extracted_bytes:
            self.extracted += 1
            self.extracted_bytes += extracted_bytes
        return package_dir


class ArgSpec(NamedTuple):
    """The shape, stride, dtype and device expected of a tensor argument,
    so GraphedSegment can verify runtime calls without keeping example
    tensors resident."""
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class Segment(torch.nn.Module):
    """One dense segment: its AOTInductor package, called directly.
    static[i] marks argument i as the engine's own tensor (same address
    every call)."""

    def __init__(
        self,
        name: str,
        sub: torch.fx.GraphModule,
        static: list[bool],
        runner: Callable[..., Any],
        example_args: tuple[Any, ...] | None = None,
        packages: Any = None,
    ) -> None:
        super().__init__()
        if runner is None:
            raise ValueError(f"segment {name}: runner must not be None")
        self.name, self.sub, self.static = name, sub, static
        self.runner, self.packages = runner, packages
        self.expected_specs: tuple[Any, ...] | None = (
            tuple(
                ArgSpec(tuple(a.shape), a.stride(), a.dtype, a.device)
                if isinstance(a, torch.Tensor)
                else a
                for a in example_args
            )
            if example_args is not None
            else None
        )
        self._warmed = False

    @property
    def _orig_mod(self):
        """The segment's graph, under the name a compiled module gives it
        (segments.segments_of, telemetry)."""
        return self.sub

    def _check_args(self, args: tuple[Any, ...]) -> None:
        if self.expected_specs is None:
            return
        if len(args) != len(self.expected_specs):
            raise RuntimeError(
                f"segment {self.name}: expected {len(self.expected_specs)} arguments but got {len(args)}"
            )
        for i, (actual, spec) in enumerate(zip(args, self.expected_specs)):
            if isinstance(spec, ArgSpec):
                if not isinstance(actual, torch.Tensor):
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} expected tensor but got {type(actual).__name__}"
                    )
                if actual.shape != spec.shape:
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} shape {tuple(actual.shape)} does not match expected {spec.shape}"
                    )
                if actual.stride() != spec.stride:
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} stride {actual.stride()} does not match expected {spec.stride}"
                    )
                if actual.dtype != spec.dtype:
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} dtype {actual.dtype} does not match expected {spec.dtype}"
                    )
                if actual.device != spec.device:
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} device {actual.device} does not match expected {spec.device}"
                    )
            else:
                if actual != spec:
                    raise RuntimeError(
                        f"segment {self.name}: argument {i} value {actual!r} does not match expected {spec!r}"
                    )

    def forward(self, *args):
        if not self._warmed:
            self._warmed = True
            if self.expected_specs is not None:
                self._check_args(args)
        return self.runner(*args)


class GraphedSegment(Segment):
    """One dense segment: its AOTInductor package, run in a CUDA graph."""

    def __init__(
        self,
        name: str,
        sub: torch.fx.GraphModule,
        static: list[bool],
        runner: Callable[..., Any],
        pool: Any,
        example_args: tuple[Any, ...] | None = None,
        packages: Any = None,
    ) -> None:
        super().__init__(name, sub, static, runner, example_args=example_args, packages=packages)
        self.pool = pool
        self.graph = None

    def forward(self, *args):
        if not self._warmed:
            self._warmed = True
            if self.expected_specs is not None:
                self._check_args(args)
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
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            outputs = self.runner(*inputs)
        # kept only once captured: a capture that fails (out of memory, say)
        # leaves the segment as it was, and the next call tries again
        self.inputs, self.outputs, self.graph = inputs, outputs, graph
        graph.replay()  # capturing executed nothing: this is the call's one execution
        return outputs


def example_args(
    split: torch.fx.GraphModule,
    name: str,
    sub: torch.fx.GraphModule,
    device: str | torch.device,
) -> tuple[Any, ...]:
    """Example arguments for compiling segment `name`: the real tensors
    for attributes of the split graph (the engine's weights and resident state),
    and zero-filled tensors matching the shapes and dtypes of activations."""
    call_node: torch.fx.Node | None = None
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == name:
            call_node = node
            break
    if call_node is None:
        raise KeyError(f"segment {name!r} not found in split graph")
    if call_node.kwargs:
        raise ValueError(
            f"segment {name!r}: keyword arguments on call node are not supported: {list(call_node.kwargs.keys())}"
        )

    orig = getattr(sub, "_orig_mod", sub)
    placeholders = [n for n in orig.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(call_node.args):
        raise ValueError(
            f"segment {name!r}: arity mismatch: {len(placeholders)} placeholders != {len(call_node.args)} call arguments"
        )

    args: list[Any] = []
    for p, a in zip(placeholders, call_node.args):
        if not isinstance(a, torch.fx.Node):
            args.append(a)
        elif a.op == "get_attr":
            obj = split
            for part in str(a.target).split("."):
                obj = getattr(obj, part)
            args.append(obj)
        else:
            val = p.meta.get("val")
            if val is None:
                val = a.meta.get("val")
            if val is None:
                raise ValueError(
                    f"segment {name!r}: placeholder {p.name!r} has no recorded value in graph metadata"
                )
            if isinstance(val, torch.Tensor):
                shape = tuple(int(s) for s in val.shape)
                args.append(torch.zeros(shape, dtype=val.dtype, device=device))
            else:
                args.append(val)
    return tuple(args)


def static_arguments(split, name):
    """For the call of submodule `name` in the split graph: which arguments
    are attributes of the graph (the engine's weights and resident state,
    the same tensors every call)."""
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == name:
            return [isinstance(a, torch.fx.Node) and a.op == "get_attr" for a in node.args]
    raise KeyError(name)
