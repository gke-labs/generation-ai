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

"""The executor's physical plan: what it chose to run, and why.

The artifact is the logical plan (the model's computation, #316); how to
run it on a given machine is the executor's choice, as a database picks a
physical plan for a query. A Plan is built once per engine from the
hardware and the configuration, and it is the one place those choices are
made and recorded:

  hardware     what the engine runs on (device, GPU, CPU, SIMD, RAM)
  choices      global decisions (split decode, compile mode, where the
               experts live, the bf16 weight cache, ...), each with its
               value, where the value came from (default, environment or
               hint) and the alternatives
  ops          for each logical op the graphs call (vxpu::fp8_linear,
               vxpu::moe_experts, ...), the registered implementations,
               when each applies, and its numerics class: bit-identical to
               the reference computation, or equal to rounding
  selections   which implementation each call site actually got, keyed by
               op, implementation and input shapes

Ops dispatch through Plan.select instead of deciding inline, so what runs
is what the plan says and what explain() reports. Selections are recorded
when Python dispatches an op: eager calls, tracing and CUDA-graph
recording. A replayed CUDA graph does not call Python, so the record says
which implementation a shape got, not how many times it ran.

`python -m vxpu.plan --artifact DIR [--device D]` explains the plan for an
artifact without loading its weights.
"""

import argparse
import collections
import contextvars
import dataclasses
import json
import math
import os
import platform
import tempfile
import threading

# ---------------------------------------------------------------------------
# configuration: one parser for every switch (#320)

TRUE = ("1", "true", "on", "yes")
FALSE = ("0", "false", "off", "no")

# The switches the coordinator set for the engine being built
# (LoadModelRequest.plan): the same names and values as the environment's,
# read ahead of it, with source "coordinator" (#394).
_given = contextvars.ContextVar("vxpu_plan_given", default=None)

# The switches the coordinator may set: how the engine runs the model,
# where it holds the experts, and the node servers it wired them to
# (VXPU_REMOTE_NODES: other executors it asked to serve nodes; the router
# never forwards it from a client's plan). Not the ones naming the
# machine's own resources (directories, the weight bucket): those stay
# with whoever runs the executor. Nor the ones the kernels
# read for themselves at first use (VXPU_KERNELS, VXPU_KERNEL_THREADS),
# which a value from the coordinator would not reach.
COORDINATOR_SWITCHES = frozenset({
    "VXPU_FUSED_EXPERTS", "VXPU_SPLIT_DECODE", "VXPU_COMPILE_MODE", "VXPU_SPLIT_COMPILE", "VXPU_DIRECT_HOST_OPS",
    "VXPU_LOWER_DECODE", "VXPU_SEGMENT_RUNTIME", "VXPU_EXPERT_MAILBOXES", "VXPU_REMOTE_NODES", "VXPU_DATAFLOW",
    "VXPU_STEP_GRAPH", "VXPU_LAYER_RANGE", "VXPU_REMOTE_RANGES", "VXPU_FUSE", "VXPU_LAYER_DEVICES",
    "VXPU_DATAFLOW_PLACE", "VXPU_GPU_EXPERTS", "VXPU_GPU_EXPERTS_MOVES", "VXPU_GPU_EXPERTS_REQUEST_WEIGHT",
    "VXPU_EXPERTS_IN_RAM", "VXPU_BF16_WEIGHT_CACHE", "VXPU_BF16_CACHE_HEADROOM_GB",
})


class ConfigError(ValueError):
    """A switch with a value the parser does not understand."""


def _raw(name):
    """A switch's text and where it came from: the coordinator, else the
    environment."""
    given = _given.get()
    if given is not None and name in given:
        return given[name], "coordinator"
    return os.environ.get(name), "env"


def env_flag(name, default, extra_false=()):
    """A boolean switch: 1/true/on/yes or 0/false/off/no, any case;
    anything else is an error rather than a guess. Returns (value, source)."""
    raw, src = _raw(name)
    if raw is None or raw == "":
        return default, "default"
    v = raw.strip().lower()
    if v in TRUE:
        return True, src
    if v in FALSE or v in extra_false:
        return False, src
    raise ConfigError(f"{name}={raw!r}: expected one of {', '.join(TRUE + FALSE + tuple(extra_false))}")


def env_number(name, default, kind=float, minimum=None):
    """A numeric switch; refuses what does not parse. Integers are plain
    ASCII digits (what C's atol reads the same way: the CPU kernels read
    VXPU_KERNEL_THREADS themselves); floats must be finite. Returns
    (value, source)."""
    raw, src = _raw(name)
    if raw is None or raw == "":
        return default, "default"
    text = raw.strip()
    try:
        if kind is int and not (text.isascii() and text.lstrip("+").isdigit()):
            raise ValueError(text)
        value = kind(text)
    except ValueError:
        raise ConfigError(f"{name}={raw!r}: expected {'an integer' if kind is int else 'a number'}") from None
    if kind is float and not math.isfinite(value):
        raise ConfigError(f"{name}={raw!r}: must be a finite number")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={raw!r}: must be at least {minimum}")
    return value, src


def env_choice(name, default, allowed):
    """A switch with a fixed set of values; refuses anything else.
    Returns (value, source)."""
    raw, src = _raw(name)
    if raw is None or raw == "":
        return default, "default"
    if raw.strip() not in allowed:
        raise ConfigError(f"{name}={raw!r}: expected one of {', '.join(allowed)}")
    return raw.strip(), src


def env_text(name, default):
    raw, src = _raw(name)
    if raw is None or raw == "":
        return default, "default"
    return raw, src


# ---------------------------------------------------------------------------
# hardware

def probe_hardware(device):
    """What the engine runs on, without building or loading anything."""
    hw = {"device": str(device), "cpu": platform.processor() or platform.machine(),
          "cpu_threads": os.cpu_count(), "simd": _simd()}
    try:
        hw["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        if platform.system() == "Darwin":
            import subprocess
            try:
                hw["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                           text=True, timeout=2).stdout.strip() or hw["cpu"]
            except (OSError, subprocess.SubprocessError):
                pass
    if str(device).startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            props = torch.cuda.get_device_properties(idx)
            hw.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                      gpu_capability=f"{props.major}.{props.minor}", gpu_count=torch.cuda.device_count())
    return hw


def _simd():
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "neon"
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith("flags")), "")
        return "avx512" if " avx512f" in flags else ("avx2" if " avx2" in flags else "sse")
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# implementations

@dataclasses.dataclass(frozen=True)
class Impl:
    """One physical implementation of a logical op."""
    op: str
    name: str
    fn: object
    devices: tuple            # device types it runs on ("cpu", "cuda")
    numerics: str             # "bit-identical" or "rounding", against the reference computation
    when: str                 # human description of when it applies
    applies: object = None    # applies(plan, **ctx) -> bool, beyond the device check
    reference: bool = False   # the reference computation: always applicable, on any device
    host: bool = False        # runs where its data lives on the host (the expert banks, the n-gram tables)
    lowerable: bool = False   # fn(..., tk=triton_ops.lowered) is traceable: the plan's lowering may inline it (lower.py)
    fused: bool = False       # a decomposed body for the compiler to fuse, chosen where measured cheaper (costs.py)


REGISTRY = collections.defaultdict(list)  # op -> [Impl], in preference order


def implementation(op, name, devices, numerics, when, applies=None, reference=False, host=False, lowerable=False,
                   fused=False):
    """Register fn as an implementation of op. Registration order is the
    preference order: the first one that applies is selected. host=True
    marks an op that runs where its data is (on the host for host-placed
    banks and tables), whatever the engine's device. lowerable=True marks
    one whose fn takes its Triton kernels as tk= and, given the traced
    ones, is pure tensor work a compiler can trace through. fused=True
    marks a lowerable body of plain tensor operations, which the lowering
    chooses over the kernel it would replace only where it measures
    cheaper (costs.py)."""
    def register(fn):
        REGISTRY[op].append(Impl(op, name, fn, tuple(devices), numerics, when, applies, reference, host, lowerable,
                                 fused))
        return fn
    return register


def _available(impl, device_type):
    """Whether impl can run for an engine on device_type."""
    return impl.reference or device_type in impl.devices or (impl.host and "cpu" in impl.devices)


@dataclasses.dataclass
class Choice:
    value: object
    source: str
    alternatives: tuple = ()
    doc: str = ""


# ---------------------------------------------------------------------------
# the plan

# input shapes recorded per implementation; later shapes are counted
# together, so an engine serving many prompt lengths keeps a bounded record
MAX_SHAPES_RECORDED = 64


class Plan:
    """The physical plan of one engine. See the module docstring."""

    def __init__(self, device, compile_decode=False, external_dir=None, triton_available=None, given=None):
        """given: the switches the coordinator set ({VXPU_NAME: value}),
        read ahead of the environment."""
        given = dict(given or {})
        refused = sorted(set(given) - COORDINATOR_SWITCHES)
        if refused:
            raise ConfigError(f"the coordinator may not set {', '.join(refused)} (it may set "
                              f"{', '.join(sorted(COORDINATOR_SWITCHES))})")
        empty = sorted(k for k, v in given.items() if not v.strip())
        if empty:
            raise ConfigError(f"the coordinator set {', '.join(empty)} to nothing: leave a switch out instead")
        if "VXPU_GPU_EXPERTS" in given:  # checked here, so a bad one fails the call, not the load
            try:
                placement = json.loads(given["VXPU_GPU_EXPERTS"])
            except (ValueError, RecursionError):  # nested past the interpreter's limit: refused too
                placement = None
            if not isinstance(placement, dict):
                raise ConfigError("the coordinator's VXPU_GPU_EXPERTS must be the placement itself (a JSON object), "
                                  "not a path")
        token = _given.set(given)
        try:
            self._choose(device, compile_decode, external_dir, triton_available)
        finally:
            _given.reset(token)

    def _choose(self, device, compile_decode, external_dir, triton_available):
        self.device = str(device)
        self.device_type = self.device.split(":")[0]
        self.hardware = probe_hardware(device)
        self.choices = {}
        self.notes = []
        self._selections = collections.Counter()
        self._shapes = collections.Counter()  # distinct input shapes recorded per (op, implementation)
        self._lock = threading.Lock()
        cuda = self.device_type == "cuda"
        if triton_available is None:
            from . import triton_ops
            triton_available = triton_ops.available()
        self._set("triton", triton_available, "probe", (True, False),
                  "Triton importable with a GPU: the GPU kernels (activation quantization, fp8 GEMV, Sinkhorn, head)")

        # the CPU kernels (csrc/): built on first use; off means the aten paths
        on, src = env_flag("VXPU_KERNELS", True, extra_false=("aten",))
        self._set("kernels", on, src, (True, False), "the fused C++ CPU kernels (fp4 experts, fp8 linears)")
        on, src = env_flag("VXPU_FUSED_EXPERTS", True)
        self._set("fused_experts", on, src, (True, False),
                  "one C++ call per decode layer's experts, instead of three kernel calls with the module's SwiGLU between")
        threads, src = env_number("VXPU_KERNEL_THREADS", 0, int, minimum=0)
        self._set("kernel_threads", threads or "torch's thread count", src, (),
                  "threads for the CPU kernels (read by the extension; 0 means torch's count)")

        # decode compilation: the split at host ops and the compile mode
        split, src = env_flag("VXPU_SPLIT_DECODE", True)
        self._set("compile_decode", bool(compile_decode), "argument", (True, False), "torch.compile the decode graph")
        self._set("split_decode", split and bool(compile_decode), src if compile_decode else "requires compile_decode",
                  (True, False), "cut the decode graph at its host ops; each dense segment compiled on its own")
        mode_default = ("reduce-overhead" if split else "max-autotune-no-cudagraphs") if cuda else "default"
        modes = ("reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune", "default")  # torch.compile's
        mode, src = env_choice("VXPU_COMPILE_MODE", mode_default, modes)
        self._set("compile_mode", mode, src, modes,
                  "torch.compile mode (reduce-overhead gives each split segment a CUDA graph)")
        on, src = env_flag("VXPU_SPLIT_COMPILE", True)
        self._set("split_compile", on, src, (True, False),
                  "compile the dense segments (off: the split alone, segments eager, a check of the cut)")
        on, src = env_flag("VXPU_DIRECT_HOST_OPS", True)
        self._set("direct_host_ops", on, src, (True, False),
                  "eager host-op segments call the op's implementation directly, not through torch's dispatcher")
        # on a GPU the default is the step graph's: AOTInductor segments, the dataflow runner, one CUDA
        # graph a step with the host ops as native mailboxes (and with rows, the native decode loop)
        aoti_default = "aoti" if cuda and self["split_decode"] and self["split_compile"] else "jit"
        runtime, rsrc = env_choice("VXPU_SEGMENT_RUNTIME", aoti_default, ("jit", "aoti"))
        if runtime == "aoti" and not (cuda and self["split_decode"] and self["split_compile"]):
            runtime, rsrc = "jit", "aoti requires CUDA and a compiled split decode"
        on, src = env_flag("VXPU_LOWER_DECODE", False)
        if runtime == "aoti":
            on, src = True, "required by segment_runtime=aoti"
        self._set("lower_decode", on and self["split_decode"], src if self["split_decode"] else "requires split_decode",
                  (True, False), "before compiling, rewrite each dense segment's vxpu op calls into the selected "
                  "implementation, its Triton kernels as torch ops (lower.py): the compiler sees the kernels")
        fuse, src = env_text("VXPU_FUSE", "attn_out,attn_q,compress_post,ffn_front,fp8_linear,hc_front,hc_sinkhorn,matmul,rotary,shared_ffn,window_kv")
        fuse = tuple(sorted(op.strip() for op in fuse.split(",") if op.strip()))
        self._set("fuse", fuse if self["lower_decode"] else (), src if self["lower_decode"] else "requires lower_decode",
                  (), "what the lowering may rewrite into plain tensor operations, which the compiler fuses with "
                  "their neighbours (comma-separated): the dense ops fp8_linear, rotary, hc_sinkhorn and ffn_front, in place of "
                  "our hand-written kernels, and matmul, small aten matmuls (linear, einsum) as reductions; each call "
                  "is rewritten only where that measures cheaper at its shapes (lower.py, costs.py)")
        self._set("segment_runtime", runtime, rsrc, ("jit", "aoti"),
                  "how dense segments run: jit (torch.compile, CUDA graphs by its cudagraph trees) or aoti "
                  "(AOTInductor packages, built once and cached, in the engine's own CUDA graphs; aoti.py)")
        on, src = env_flag("VXPU_EXPERT_MAILBOXES", False)
        self._set("expert_mailboxes", on, src, (True, False),
                  "decode's routed experts as mailboxes (mailboxes.py, #372): each (token, k) pair answered by its "
                  "expert's owner (the CPU, a GPU pool), the replies summed in k order whoever answered")
        nodes, src = env_text("VXPU_REMOTE_NODES", None)
        self._set("remote_nodes", nodes, src, (),
                  "host nodes node servers serve (name=tcp://host:port/index[,...]; nodeserver.py): their residents "
                  "are not loaded here, and their host ops send their inputs to the server's mailbox")
        rng, src = env_text("VXPU_LAYER_RANGE", None)
        if rng:
            try:
                first, _, last = rng.partition(":")
                rng = (int(first), int(last) if last else None)
                if rng[0] < 0 or (rng[1] is not None and rng[1] <= rng[0]):
                    raise ValueError
            except ValueError:
                raise ConfigError(f"VXPU_LAYER_RANGE={rng!r}: START:END (END exclusive, or empty for the last layer)")
        self._set("layer_range", rng, src, (),
                  "run only layers START..END-1 of the step (START:END, END empty for the last layer): their slice of "
                  "decode and prefill, their experts and n-gram tables; other executors run the other ranges (#387)")
        devices, src = env_text("VXPU_LAYER_DEVICES", None)
        self._set("layer_devices", devices, src, (),
                  "the GPUs the layers run on, as START:END=cuda:N,... (layer sharding): each range's dense weights, "
                  "caches and segments live on its GPU and the step graph is one CUDA graph per GPU, crossing once "
                  "per range; the head goes with the last range (requires dataflow)")
        ranges, src = env_text("VXPU_REMOTE_RANGES", None)
        self._set("remote_ranges", ranges, src, (),
                  "where other executors serve the ranges of layers after this engine's range, in order, as "
                  "START:END=tcp://host:port,... (an empty END means the last layer). This engine sends each step "
                  "on to them (rangeserver.py, #387)")
        on, src = env_flag("VXPU_DATAFLOW", cuda and self["segment_runtime"] == "aoti")
        self._set("dataflow", on, src, (True, False),
                  "run the decode step as a dataflow graph: its segments as nodes, each fired when the inboxes "
                  "its inputs are written into are full (dataflow.py, the reference runner; #372)")
        if self["layer_range"] is not None and not on:  # a range is a slice of the dataflow graph
            raise ConfigError("VXPU_LAYER_RANGE requires VXPU_DATAFLOW=1")
        if self["remote_ranges"] and not on:
            raise ConfigError("VXPU_REMOTE_RANGES requires VXPU_DATAFLOW=1")
        on, src = env_flag("VXPU_STEP_GRAPH", True)
        if on and not (cuda and self["dataflow"] and self["segment_runtime"] == "aoti" and self["kernels"]):
            on, src = False, "requires CUDA, dataflow, segment_runtime=aoti and the CPU kernels (the step host)"
        self._set("step_graph", on, src, (True, False),
                  "capture the whole decode step in one CUDA graph: the AOTInductor segments, and the host ops as "
                  "mailboxes the GPU waits on, answered by a native thread (stepgraph.py, #372, #409)")
        place, src = env_text("VXPU_DATAFLOW_PLACE", None)
        if place:
            try:
                kinds = dict(item.split("=", 1) for item in place.split(","))
            except ValueError:
                raise ValueError(f"VXPU_DATAFLOW_PLACE={place!r}: expected kind=device[,kind=device...]") from None
            unknown = set(kinds) - {"experts", "engram"}  # a dense node's weights are not moved, so it stays put
            if unknown:
                raise ValueError(f"VXPU_DATAFLOW_PLACE: kinds {sorted(unknown)} cannot be placed (experts, engram can)")
            place = ",".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        if place and not self["dataflow"]:
            place, src = None, "requires dataflow"
        self._set("dataflow_placement", place, src, (),
                  "where the dataflow graph's nodes run, by kind (e.g. experts=cpu,engram=cpu); unnamed kinds run "
                  "where their resident tensors are")
        home = os.path.expanduser("~")
        base = os.path.join(home, ".cache") if home != "~" else tempfile.gettempdir()  # no HOME: not a relative "~"
        cache, src = env_text("VXPU_AOTI_CACHE_DIR", os.path.join(base, "vxpu", "aoti"))
        self._set("aoti_cache_dir", cache, src, (), "where AOTInductor segment packages are kept (the compiled plans)")

        # inductor settings: prologue fusion folds dequantization into GEMM operand loads on CUDA
        if not cuda:
            on, src = False, "CUDA only"
        elif not compile_decode:
            on, src = False, "requires compile_decode"
        else:
            on, src = True, "default"
        self._set("prologue_fusion", on, src, (True, False),
                  "inductor prologue fusion folds dequantization into GEMM operand loads on CUDA")

        # where the external tensors live
        if external_dir:  # an explicit argument wins over the environment, as it always has
            ext_dir, src = external_dir, "argument"
        else:
            ext_dir, src = env_text("VXPU_EXTERNAL_DIR", None)
        self._set("external_dir", ext_dir, src, (), "directory for memory-mapped external tensors")
        blob_dir, src = env_text("VXPU_BLOB_DIR", None)
        self._set("blob_dir", blob_dir, src, (),
                  "this machine's local weight tier: checkpoint files by sha256, read instead of their sources (localblobs.py)")
        bucket, src = env_text("VXPU_BLOB_BUCKET", None)
        self._set("blob_bucket", bucket, src, (),
                  "the weight bucket (gs://BUCKET, modelstore's layout) that fills blob_dir with files it lacks")
        hot, src = env_text("VXPU_GPU_EXPERTS", None)
        if hot and not (cuda and self["triton"] and self["kernels"]):
            # the hot implementation needs all three: without one, holding
            # banks on the GPU would cost its memory and never run
            hot, src = None, "requires CUDA, Triton and the CPU kernels"
        self._set("gpu_experts", hot, src, (),
                  "a placement, inline JSON or a file (per GPU, {layer: [expert ids]}, from expert_stats --placement-gb): those experts "
                  "are held on the GPUs and computed there alongside the CPU's (experts_gpu.py)")
        moves, src = env_number("VXPU_GPU_EXPERTS_MOVES", 0, int, minimum=0)
        if moves and not hot:
            moves, src = 0, "needs gpu_experts"
        self._set("gpu_experts_moves", moves, src, (),
                  "experts moved between the host and the GPUs at once as the traffic's hot set drifts (0: the "
                  "placement stays as loaded; results then depend only on the placement, not on earlier traffic)")
        weight, src = env_number("VXPU_GPU_EXPERTS_REQUEST_WEIGHT", 0.25, float, minimum=0)
        if weight > 1:
            raise ValueError(f"VXPU_GPU_EXPERTS_REQUEST_WEIGHT={weight}: a share, at most 1")
        self._set("gpu_experts_request_weight", weight, src, (),
                  "with gpu_experts_moves: the current request's own selections' share of an expert's score, the "
                  "rest all the traffic's (0.25: the best of the offline simulator's, expert_sim.py)")
        in_ram, src = env_flag("VXPU_EXPERTS_IN_RAM", False)
        placement = "ram" if (in_ram or not ext_dir) else "mmap"
        self._set("experts_placement", placement, src if in_ram else ("default" if not ext_dir else "external_dir"),
                  ("ram", "mmap"), "expert banks held in RAM, or memory-mapped from external_dir (the page cache as the RAM tier)")

        # the GPU's bf16 copies of fp8 weights (prefill, and decode without Triton)
        on, src = env_flag("VXPU_BF16_WEIGHT_CACHE", True)
        self._set("bf16_weight_cache", on, src, (True, False),
                  "keep a bf16 copy of each dense fp8 weight on the GPU (faster prefill, twice the dense weights' memory)")
        gb, src = env_number("VXPU_BF16_CACHE_HEADROOM_GB", 4.0, float, minimum=0)
        self._set("bf16_cache_headroom_gb", gb, src, (), "device memory the bf16 copies must leave free")

    def _set(self, name, value, source, alternatives=(), doc=""):
        self.choices[name] = Choice(value, source, tuple(alternatives), doc)

    def __getitem__(self, name):
        return self.choices[name].value

    def note(self, text):
        """A runtime event worth reporting with the plan (once each)."""
        with self._lock:
            if text not in self.notes:
                self.notes.append(text)

    # ---- dispatch

    def candidates(self, op, device_type):
        return [impl for impl in REGISTRY[op] if _available(impl, device_type)]

    def select(self, op, device_type, key=None, **ctx):
        """The implementation of op to run for this call: the first
        registered one that runs on device_type and applies under this plan
        and ctx. key (input shapes, say) is recorded with the selection."""
        for impl in REGISTRY[op]:
            # the reference computation runs anywhere torch does
            if impl.reference or (device_type in impl.devices and (impl.applies is None or impl.applies(self, **ctx))):
                with self._lock:
                    record = (op, impl.name, key)
                    if record not in self._selections:
                        if self._shapes[(op, impl.name)] >= MAX_SHAPES_RECORDED:
                            record = (op, impl.name, "(other shapes)")
                        else:
                            self._shapes[(op, impl.name)] += 1
                    self._selections[record] += 1
                return impl
        raise RuntimeError(f"no implementation of {op} for {device_type} ({ctx})")

    # ---- explain

    def explain(self, ops=None):
        """The plan as JSON-able data. ops limits the op table to the ops an
        artifact calls (default: every registered op)."""
        names = sorted(ops) if ops is not None else sorted(REGISTRY)
        selections = collections.defaultdict(list)
        with self._lock:
            for (op, impl, key), n in sorted(self._selections.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
                selections[op].append({"impl": impl, "inputs": key, "dispatches": n})
        table = {}
        for op in names:
            table[op] = {
                "implementations": [
                    {"name": i.name, "devices": list(i.devices), "numerics": i.numerics, "when": i.when,
                     "on_this_device": _available(i, self.device_type), "lowerable": i.lowerable}
                    for i in REGISTRY[op]],
                "selected": selections.get(op, []),
            }
        return {
            "format": "vxpu-plan/1",
            "hardware": self.hardware,
            "choices": {k: {"value": c.value, "source": c.source, "alternatives": list(c.alternatives), "doc": c.doc}
                        for k, c in self.choices.items()},
            "ops": table,
            "notes": list(self.notes),
        }

    def summary(self, ops=None):
        """A compact, human-readable form of explain() for logs."""
        e = self.explain(ops)
        hw = e["hardware"]
        lines = [f"plan on {hw['device']}: {hw.get('gpu', hw['cpu'])}"
                 + (f", {hw['gpu_memory_bytes'] / 1e9:.0f} GB" if "gpu_memory_bytes" in hw else "")
                 + f", {hw.get('ram_bytes', 0) / 1e9:.0f} GB RAM, {hw['cpu_threads']} CPU threads, {hw['simd']}"]
        lines.append("  " + ", ".join(f"{k}={_brief(c['value'])}" + ("" if c["source"] in ("default", "probe") else f" ({c['source']})")
                                      for k, c in e["choices"].items()))
        for op, row in e["ops"].items():
            here = [i["name"] for i in row["implementations"] if i["on_this_device"]]
            got = sorted({s["impl"] for s in row["selected"]})
            lines.append(f"  {op}: {' > '.join(here) or '-'}" + (f"   selected: {', '.join(got)}" if got else ""))
        lines += [f"  note: {n}" for n in e["notes"]]
        return "\n".join(lines)


def _brief(value, limit=80):
    """A choice's value for a log line: an inline placement is long."""
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit // 2]}... ({len(text)} chars)"


# choices whose values are paths on the executor's own disk: a record
# (which leaves the process, on the Generate stream) says only whether
# they are set
LOCAL_PATHS = ("external_dir", "aoti_cache_dir", "blob_dir", "blob_bucket", "gpu_experts")


def record(explained):
    """The compact form of an explain() a result keeps (vectors files,
    telemetry): each choice's value (for a local path, whether it is set),
    the implementations each op ran and their numerics classes, and the
    plan's notes (runtime events: the lowering's report, a bf16 cache that
    stopped), so a result says which plan produced it."""
    return {"choices": {k: (c["value"] is not None) if k in LOCAL_PATHS else c["value"]
                        for k, c in explained["choices"].items()},
            "selected": {op: sorted({s["impl"] for s in row["selected"]}) for op, row in explained["ops"].items()},
            "numerics": {op: sorted({i["numerics"] for i in row["implementations"]
                                     if i["name"] in {s["impl"] for s in row["selected"]}})
                         for op, row in explained["ops"].items()},
            "notes": list(explained["notes"])}


_default = {}


def default_plan(device_type):
    """The plan for op calls made outside any engine (direct calls in tests
    and tools): built from the environment once per device type."""
    plan = _default.get(device_type)
    if plan is None:
        plan = _default[device_type] = Plan(device_type)
    return plan


# ---------------------------------------------------------------------------
# EXPLAIN without loading weights

def artifact_ops(artifact_dir):
    """The vxpu ops an artifact's graphs call."""
    import torch
    from . import ops as _ops  # noqa: F401  (registers the ops the graphs reference)
    found = set()
    for name in ("prefill.pt2", "decode.pt2"):
        path = os.path.join(artifact_dir, name)
        if not os.path.exists(path):
            continue
        ep = torch.export.load(path)
        for node in ep.graph.nodes:
            target = str(node.target)
            if node.op == "call_function" and target.startswith("vxpu."):
                found.add("vxpu::" + target.split(".")[1])
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description="Explain the physical plan for an artifact, without loading weights.")
    p.add_argument("--artifact", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true", help="the plan for a compiled decode (the executor's GPU default)")
    p.add_argument("--json", action="store_true", help="print the full explain() JSON")
    args = p.parse_args(argv)
    # run as `python -m vxpu.plan` this file is __main__, a second copy of
    # the module: the ops register into vxpu.plan's registry, so use that one
    from . import ops as _ops  # noqa: F401  (registers the implementations)
    from . import plan as registered
    plan = registered.Plan(args.device, compile_decode=args.compile)
    used = registered.artifact_ops(args.artifact)
    print(json.dumps(plan.explain(used), indent=1) if args.json else plan.summary(used))


if __name__ == "__main__":
    main()
