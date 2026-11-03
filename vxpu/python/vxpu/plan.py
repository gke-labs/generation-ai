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
import contextlib
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
    "VXPU_EXPERTS_IN_RAM", "VXPU_BF16_WEIGHT_CACHE", "VXPU_BF16_CACHE_HEADROOM_GB", "VXPU_PLAN_TABLE",
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
            copy, read = gpu_bandwidth(idx)
            if copy:
                hw["gpu_bandwidth_gb_s"] = copy
            if read:
                hw["gpu_read_bandwidth_gb_s"] = read
    return hw


def gpu_bandwidth(index=0, nbytes=1 << 30):
    """The GPU's memory bandwidth as measured on the card itself (GB/s, the
    best of three; (None, None) without CUDA): a device-to-device copy of
    nbytes (read and write counted) and a single-tensor read of nbytes (a
    sum). The byte floors in the kernel record are over the read figure
    (pkg/kernelplan: Ledger), since the step's kernels read weights and
    write little: on an RTX PRO 6000 the copy measures about 1,460 GB/s
    and the read about 1,550, and the head's GEMV was seen at 1,530 GB/s,
    above the copy figure. The record carries both."""
    import torch
    if not torch.cuda.is_available():
        return None, None
    try:
        with torch.cuda.device(index):
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            dst.copy_(src)
            as_f32 = src.view(torch.float32)
            as_f32.sum()

            def best_ms(fn):
                best = None
                for _ in range(3):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    best = ms if best is None or ms < best else best
                return best
            copy_ms = best_ms(lambda: dst.copy_(src))
            read_ms = best_ms(lambda: as_f32.sum())
            del src, dst, as_f32
        return (round(2 * nbytes / (copy_ms * 1e-3) / 1e9, 1), round(nbytes / (read_ms * 1e-3) / 1e9, 1))
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None, None


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
    counters: object = None   # counters(**ctx) -> how many zeroed int32 arrival counters a call needs (the lowering's scratch)


REGISTRY = collections.defaultdict(list)  # op -> [Impl], in preference order


def implementation(op, name, devices, numerics, when, applies=None, reference=False, host=False, lowerable=False,
                   fused=False, counters=None):
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
                                 fused, counters))
        return fn
    return register


_checking = threading.local()


@contextlib.contextmanager
def checking(wrap):
    """While the block runs in this thread, every Plan.select returns
    wrap(plan, op, impl, device_type, key) instead of impl: the reference
    check (checks.py) wraps each chosen implementation with one that runs
    the reference beside it. Thread-local, so the step host's threads and
    other batches' loops dispatch as usual."""
    previous = getattr(_checking, "wrap", None)
    _checking.wrap = wrap
    try:
        yield
    finally:
        _checking.wrap = previous


def reference_of(op):
    """The registered reference implementation of op, or None."""
    return next((i for i in REGISTRY[op] if i.reference), None)


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
# the kernel table and the record: the planner's choices as data (hero run 8)
#
# The table is what the planner (pkg/kernelplan, Go) hands an engine: per
# call site (an op, the module that calls it, its input shapes) the
# implementation body to run, and the kernels' knobs (the numbers
# triton_ops sets by hand today). The record is what the engine writes
# back: per site the candidates, what was measured, what was chosen and
# why, and the kernels inlined for it, plus the step's statistics. The
# planner reads records and emits tables; an engine given a table applies
# it exactly and measures nothing.

TABLE_FORMAT = "vxpu-kernel-table/1"
RECORD_FORMAT = "vxpu-kernel-record/1"

# triton_ops' hand-set numbers a table may set (module attributes read at
# each call): name -> type. The bool ones are triton_ops' "1"/"0" A/B
# overrides; None (the default) leaves the kernel to decide from the card.
KNOBS = {
    "GEMV_ROWS_MAX": int, "FP4_DIRECT_MAX": int, "FFN_FRONT_EB": int, "FFN_FRONT_BD": int,
    "FRONT_SPLIT_BD": int, "FRONT_SPLIT_EB": int, "FP4_KSPLIT": int, "FP4_NATIVE": bool, "FP4_DIRECT": bool,
    "MXFP8_RECODE": bool,  # the native GEMV bodies recode x in their prologue (true) or in a launch before each (default)
    # the direct fp4 expert kernels' tiles: rows a program, K an iteration, warps, pipeline stages
    "FP4_GU_BN": int, "FP4_GU_BK": int, "FP4_GU_WARPS": int, "FP4_GU_STAGES": int,
    "FP4_DN_BN": int, "FP4_DN_BK": int, "FP4_DN_WARPS": int, "FP4_DN_STAGES": int,
    "FOLD_POST": bool, "FOLD_POST_RB": int,  # window_kv's row finish in its GEMV's last program (hero run 11), rows at a time
    "FP4_ONE_LAUNCH": bool, "FP4_ONE_PER_SM": int,  # the direct expert path as one launch a layer (hero run 12), programs an SM
    "FP4_ONE_WARPS": int, "FP4_ONE_STAGES": int, "FP4_PDL": bool,  # its warps and stages; the two-launch path's down kernel as a programmatic dependent launch
    # the direct kernels under expert parallelism (hero run 15): the compact grid, the held pairs' programs first in
    # dispatch order and the unheld pairs' empty programs last
    "FP4_DIRECT_COMPACT": bool,
    # the direct expert kernels' K splits (hero run 15): gate/up over K, down over I; a per-rows choice (knobs_by_rows)
    "FP4_GU_KS": int, "FP4_DN_KS": int,
    # the expert GEMM form (hero run 15): from this many pairs a layer the grouped schedule's blocks run through the
    # descriptor-fed kernels, gate and up as separate programs; their tiles (pairs a block, expert rows a program,
    # K a step, warps, stages) for gate/up and down
    "FP4_GEMM_MIN_PAIRS": int,
    "FP4_GG_BM": int, "FP4_GG_BN": int, "FP4_GG_BK": int, "FP4_GG_WARPS": int, "FP4_GG_STAGES": int,
    "FP4_GD_BM": int, "FP4_GD_BN": int, "FP4_GD_BK": int, "FP4_GD_WARPS": int, "FP4_GD_STAGES": int,
    # vxpu::allreduce (hero run 14): the reduce-scatter form forced, the rows it starts at, the elements a program moves
    "ALLREDUCE_SCATTER": bool, "ALLREDUCE_SCATTER_MIN_ROWS": int, "ALLREDUCE_BLOCK": int,
    "GEMV_PUSH_BN": int,  # the fused push's GEMV: weight rows per program, the span its epilogue pushes (hero run 15)
}


def load_table(text):
    """A kernel table from its JSON (inline, or a path to a file), checked:
    the format, every op and implementation registered, every knob known
    and of its type. Returns the table dict with an "index" the matching
    uses."""
    source = text
    if not text.lstrip().startswith("{"):
        try:
            with open(os.path.expanduser(text)) as f:
                text = f.read()
        except OSError as e:
            raise ConfigError(f"VXPU_PLAN_TABLE={source!r}: cannot read it: {e}") from None
    try:
        table = json.loads(text)
    except ValueError as e:
        raise ConfigError(f"VXPU_PLAN_TABLE: not JSON: {e}") from None
    if not isinstance(table, dict) or table.get("format") != TABLE_FORMAT:
        raise ConfigError(f"VXPU_PLAN_TABLE: expected a JSON object with format {TABLE_FORMAT!r}")
    knobs = table.get("knobs") or {}
    if not isinstance(knobs, dict):
        raise ConfigError("VXPU_PLAN_TABLE: knobs must be an object")
    for name, value in knobs.items():
        kind = KNOBS.get(name)
        if kind is None:
            raise ConfigError(f"VXPU_PLAN_TABLE: unknown knob {name!r} (known: {', '.join(sorted(KNOBS))})")
        if kind is bool and not isinstance(value, bool) or kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"VXPU_PLAN_TABLE: knob {name} must be {'true/false' if kind is bool else 'an integer'}, got {value!r}")
    by_rows = table.get("knobs_by_rows") or []
    if not isinstance(by_rows, list):
        raise ConfigError("VXPU_PLAN_TABLE: knobs_by_rows must be a list of {rows_max, knobs}")
    for i, entry in enumerate(by_rows):
        if not isinstance(entry, dict) or not isinstance(entry.get("rows_max"), int) or isinstance(entry.get("rows_max"), bool) \
                or not isinstance(entry.get("knobs"), dict):
            raise ConfigError(f"VXPU_PLAN_TABLE: knobs_by_rows[{i}] needs an integer rows_max and a knobs object")
        for name, value in entry["knobs"].items():
            kind = KNOBS.get(name)
            if kind is None:
                raise ConfigError(f"VXPU_PLAN_TABLE: knobs_by_rows[{i}]: unknown knob {name!r}")
            if kind is bool and not isinstance(value, bool) or kind is int and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigError(f"VXPU_PLAN_TABLE: knobs_by_rows[{i}]: knob {name} must be "
                                  f"{'true/false' if kind is bool else 'an integer'}, got {value!r}")
    index = {"site": {}, "shapes": {}, "op": {}}
    for i, entry in enumerate(table.get("sites") or []):
        if not isinstance(entry, dict) or "op" not in entry or "impl" not in entry:
            raise ConfigError(f"VXPU_PLAN_TABLE: sites[{i}] needs op and impl")
        op, impl = entry["op"], entry["impl"]
        if op not in REGISTRY:
            raise ConfigError(f"VXPU_PLAN_TABLE: sites[{i}]: unknown op {op!r}")
        if impl not in {im.name for im in REGISTRY[op]}:
            raise ConfigError(f"VXPU_PLAN_TABLE: sites[{i}]: {op} has no implementation {impl!r} "
                              f"(it has {', '.join(im.name for im in REGISTRY[op])})")
        if entry.get("site"):
            key, where = (op, entry["site"]), index["site"]
        elif entry.get("shapes"):
            key, where = (op, entry["shapes"]), index["shapes"]
        else:
            key, where = op, index["op"]
        if key in where and where[key]["impl"] != impl:
            raise ConfigError(f"VXPU_PLAN_TABLE: sites[{i}] names {impl} where an earlier entry named "
                              f"{where[key]['impl']} for the same {key}")
        where[key] = entry
    table["index"] = index
    # the GEMV templates' blocks per shape class (triton_ops.gemv_blocks): what `vxpu kernels measure` chose
    blocks = table.get("blocks") or []
    if not isinstance(blocks, list):
        raise ConfigError("VXPU_PLAN_TABLE: blocks must be a list")
    from . import triton_ops
    seen = {}
    for i, entry in enumerate(blocks):
        key = entry.get("shapes") if isinstance(entry, dict) else None
        values = entry.get("blocks") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key or not isinstance(values, dict) or not values:
            raise ConfigError(f"VXPU_PLAN_TABLE: blocks[{i}] needs shapes (the shape class) and blocks (an object)")
        for name, value in values.items():
            if name not in triton_ops.GEMV_BLOCK_NAMES:
                raise ConfigError(f"VXPU_PLAN_TABLE: blocks[{i}] ({key}): unknown block {name!r} "
                                  f"(known: {', '.join(triton_ops.GEMV_BLOCK_NAMES)})")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"VXPU_PLAN_TABLE: blocks[{i}] ({key}): {name} must be a positive integer, got {value!r}")
        if key in seen and seen[key] != values:
            raise ConfigError(f"VXPU_PLAN_TABLE: blocks[{i}] names {key} again with other blocks")
        seen[key] = dict(values)
    table["blocks_index"] = seen
    return table


def table_entry(table, op, site=None, shapes=None):
    """The table's entry for a call: the one naming this site, else the one
    naming the op at these shapes, else the one for the op alone; None when
    the table says nothing about it."""
    if not table:
        return None
    index = table["index"]
    if site is not None and (op, site) in index["site"]:
        return index["site"][(op, site)]
    if shapes is not None and (op, shapes) in index["shapes"]:
        return index["shapes"][(op, shapes)]
    return index["op"].get(op)


def apply_knobs(knobs):
    """Set triton_ops' knobs from the table. The kernels read them at each
    call, so this takes effect for everything launched afterwards (and
    before any graph is captured, since the plan is built first)."""
    if not knobs:
        return
    from . import triton_ops
    for name, value in knobs.items():
        if os.environ.get("VXPU_" + name) not in (None, ""):
            continue  # the environment is the A/B override of a table's value
        if KNOBS[name] is bool:
            setattr(triton_ops, name, "1" if value else "0")
        elif name in ("FP4_KSPLIT", "FP4_GU_KS", "FP4_DN_KS"):
            setattr(triton_ops, name, str(int(value)))
        else:
            setattr(triton_ops, name, int(value))
    cached = getattr(triton_ops, "_fp4_native", None)  # memoized per device from FP4_NATIVE
    if cached is not None and hasattr(cached, "cache_clear"):
        cached.cache_clear()


def knobs_for_rows(table, rows):
    """The table's knobs for a step of `rows` token rows: the first knobs_by_rows entry whose
    rows_max covers rows (the entries in rows_max order), or {} (hero run 15: the direct expert
    kernels' pair limit, their K splits and the MXFP8 recode are choices by rows, not per process)."""
    for entry in sorted(table.get("knobs_by_rows") or [], key=lambda e: e["rows_max"]):
        if rows <= entry["rows_max"]:
            return dict(entry["knobs"])
    return {}


def apply_blocks(blocks):
    """Set the GEMV templates' blocks per shape class from the table (triton_ops.GEMV_BLOCKS,
    read at each call, so in effect for everything launched afterwards)."""
    from . import triton_ops
    triton_ops.GEMV_BLOCKS.clear()
    if blocks:
        triton_ops.GEMV_BLOCKS.update({k: dict(v) for k, v in blocks.items()})


def effective_knobs():
    """The knobs' values in effect, typed as the table would give them
    (None: the kernel decides from the card)."""
    try:
        from . import triton_ops
    except ImportError:
        return {}
    out = {}
    for name, kind in KNOBS.items():
        value = getattr(triton_ops, name, None)
        if value is None:
            out[name] = None
        elif kind is bool:
            out[name] = value in (True, "1")
        else:
            out[name] = int(value)
    return out


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
        fuse, src = env_text("VXPU_FUSE", "attn_out,attn_q,ffn_front,fp8_linear,hc_front,hc_sinkhorn,matmul,rotary,shared_ffn,shared_ffn_combine,shared_ffn_front,window_kv")
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
        # Tensor parallelism of the dense layers (hero run 14; the plan's "parallel" section, rendered by
        # physplan as these variables): dense_tp ranks run the same step on dense_tp GPUs, each with its
        # slice of every dense weight (the artifact exported with --tp), summing where the fork
        # all-reduces; the experts stay where the placement puts them and each rank runs only the
        # pairs its GPU holds (expert_ep is the placement's spread, for the record).
        tp_text, src = env_text("VXPU_DENSE_TP", None)
        try:
            tp = int(tp_text) if tp_text else 1
            if tp < 1:
                raise ValueError
        except ValueError:
            raise ConfigError(f"VXPU_DENSE_TP={tp_text!r}: a rank count of 1 or more")
        self._set("dense_tp", tp, src, (),
                  "run the decode step as this many tensor-parallel ranks, one GPU each, the dense weights split "
                  "across them and summed with vxpu::tp_allreduce where the fork all-reduces (parallel.py; requires "
                  "an artifact exported with --tp N)")
        gpus_text, src = env_text("VXPU_TP_GPUS", None)
        gpus = [int(x) for x in gpus_text.split(",")] if gpus_text else list(range(tp))
        if tp > 1 and len(gpus) != tp:
            raise ConfigError(f"VXPU_TP_GPUS={gpus_text!r}: {tp} GPU indices for dense_tp={tp}")
        self._set("tp_gpus", ",".join(str(g) for g in gpus), src, (), "the GPUs of the tensor-parallel ranks, in rank order")
        ep_text, src = env_text("VXPU_EXPERT_EP", None)
        self._set("expert_ep", int(ep_text) if ep_text else 1, src, (),
                  "how many GPUs each layer's experts are spread over by the placement (the record's field; the "
                  "placement file decides)")
        red, src = env_text("VXPU_REDUCE", "one-shot")
        self._set("reduce", red, src, ("one-shot", "ring", "skip"),
                  "the cross-card sum in the captured tensor-parallel step: one-shot, the all-reduce template "
                  "(vxpu::allreduce, allreduce.py: a push kernel and a sum kernel, the reduce-scatter form where its "
                  "table wants it); ring, the ring write per peer with the stream's flag waits and an aten sum; "
                  "skip, no sum at all (the step's timing floor; the tokens are wrong)")
        on, src = env_flag("VXPU_FUSED_PUSH", False)
        self._set("fused_push", on, src, (True, False),
                  "the fused push (hero run 15): the GEMV that produces a mixing sum's partial (the attention's wo_b, "
                  "the shared expert's w2 with the routed sum added) runs inside the collective and writes the partial "
                  "into the peers' inboxes from its epilogue, so the one-shot form's push kernel is gone and the sum "
                  "kernel only waits and mixes (lower.fuse_pushes, allreduce.gemv_allreduce_hc; one-shot only)")
        draft_device, src = env_text("VXPU_DRAFT_DEVICE", None)
        if draft_device is not None and not draft_device.startswith("cuda:"):
            raise ConfigError(f"VXPU_DRAFT_DEVICE={draft_device!r}: cuda:N")
        self._set("draft_device", draft_device, src, (),
                  "the GPU that runs the drafter's stages (the layers after the backbone's) and holds their "
                  "experts, instead of the layer_devices range that covers their ids (the last one, where every "
                  "batch's step ends); the planner's drafter.gpu (hero run 13)")
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
        # the kernel table: the planner's choice of body per call site and the kernels' knobs
        text, src = env_text("VXPU_PLAN_TABLE", None)
        self.table = load_table(text) if text else None
        if self.table is not None:
            wanted = self.table.get("gpu")
            if wanted and self.hardware.get("gpu") and wanted != self.hardware["gpu"]:
                raise ConfigError(f"VXPU_PLAN_TABLE: the table is for {wanted!r}, this engine runs on "
                                  f"{self.hardware['gpu']!r}")
            apply_knobs(self.table.get("knobs"))
            apply_blocks(self.table.get("blocks_index"))
        sites = len(self.table.get("sites") or []) if self.table else 0
        nblocks = len(self.table.get("blocks_index") or {}) if self.table else 0
        self._set("plan_table", (f"{sites} entries, {len(self.table.get('knobs') or {})} knobs, {nblocks} block classes"
                                 if self.table else None),
                  src, (), "the kernel table (inline JSON or a file): which implementation body each call site runs "
                  "and the kernels' knobs; an engine given one applies it exactly and measures nothing "
                  "(plan.py: load_table; the planner writes it, pkg/kernelplan)")
        path, src = env_text("VXPU_PLAN_RECORD", None)
        self._record_path = path
        self._set("plan_record", path, src, (), "where the engine writes the kernel record: per call site the "
                  "candidates, the measured costs, the choice and its source, the kernels inlined; the planner's input")
        self.sites = []  # the record's site entries (lower.py appends)
        self.stats = {}  # per-step statistics (whoever measures them: Plan.stat)
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
        ep, src = env_number("VXPU_EXPERT_EP", 0, int, minimum=0)
        if ep and not hot:
            ep, src = 0, "needs gpu_experts"
        self._set("expert_ep", ep, src, (),
                  "the experts split over this many GPUs with each GPU computing only the pairs of the experts it holds "
                  "and handing on the partial sum (hero run 14, with the dense layers split across the cards): no pair "
                  "goes to the host or to another GPU's pool; the step's reduce adds the GPUs' partials (0: off)")
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

        # the standing checks (checks.py), on by default in development: each
        # off by its switch, and the record carries whether it ran and what it cost
        on, src = env_flag("VXPU_CHECK_GOLDEN", True)
        self._set("check_golden", on, src, (True, False),
                  "after warm-up, decode the model's golden prompts and compare the tokens (golden.json in the "
                  "artifact, or fixtures/golden/<model>.json)")
        n, src = env_number("VXPU_CHECK_DUPLICATE_ROWS", 1, int, minimum=0)
        self._set("check_duplicate_rows", n, src, (0, 1),
                  "rows a generate_rows call pads with a copy of a decoding row, whose tokens must equal the "
                  "original's every step (0: off)")
        n, src = env_number("VXPU_CHECK_REFERENCE", 50, int, minimum=0)
        self._set("check_reference_every", n, src, (0, 50),
                  "every N decode steps, run the step again through the reference implementations beside the "
                  "chosen bodies and compare (0: off)")
        budget, src = env_number("VXPU_CHECK_REFERENCE_BUDGET", 0.02, float, minimum=0)
        self._set("check_reference_budget", budget, src, (),
                  "the reference check's share of the decode time: a sample waits until the run has decoded for "
                  "cost / budget since the last one")
        mode, src = env_choice("VXPU_CHECK_ON_MISMATCH", "stop", ("stop", "warn"))
        self._set("check_on_mismatch", mode, src, ("stop", "warn"),
                  "what a failed check does: stop the run (the engine refuses to serve, a generate call fails) "
                  "or print a warning and note it in the plan")

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

    def table_choice(self, op, device_type, key=None, site=None, **ctx):
        """The implementation the kernel table names for this call, or None
        when the table says nothing about it. A named body that cannot run
        here is an error when the lowering asks (the table is the planner's
        word and must be applied exactly); an eager call falls back to the
        registry's choice and the plan notes it once, since eager dispatch
        sees shapes the planner never measured (prefill, warm-up)."""
        entry = table_entry(self.table, op, site, key)
        if entry is None:
            return None
        impl = next(i for i in REGISTRY[op] if i.name == entry["impl"])
        runs = impl.reference or (device_type in impl.devices and (impl.applies is None or impl.applies(self, **ctx)))
        if runs:
            return impl
        where = f"{op} at {site or key or 'every site'}"
        if ctx.get("lowering"):
            raise ConfigError(f"VXPU_PLAN_TABLE names {impl.name} for {where}, which cannot run here "
                              f"(device {device_type}, {impl.when})")
        self.note(f"the kernel table names {impl.name} for {where}; an eager call could not run it and took the "
                  f"registry's choice")
        return None

    def select(self, op, device_type, key=None, **ctx):
        """The implementation of op to run for this call: the kernel table's
        when it names one, else the first registered one that runs on
        device_type and applies under this plan and ctx. key (input shapes,
        say) is recorded with the selection; ctx may carry site (the module
        calling the op) when the lowering asks."""
        impl = self.table_choice(op, device_type, key, **ctx)
        if impl is None:
            for candidate in REGISTRY[op]:
                # the reference computation runs anywhere torch does
                if candidate.reference or (device_type in candidate.devices
                                           and (candidate.applies is None or candidate.applies(self, **ctx))):
                    impl = candidate
                    break
            else:
                raise RuntimeError(f"no implementation of {op} for {device_type} ({ctx})")
        with self._lock:
            record = (op, impl.name, key)
            if record not in self._selections:
                if self._shapes[(op, impl.name)] >= MAX_SHAPES_RECORDED:
                    record = (op, impl.name, "(other shapes)")
                else:
                    self._shapes[(op, impl.name)] += 1
            self._selections[record] += 1
        wrap = getattr(_checking, "wrap", None)
        if wrap is not None:  # the reference check runs in this thread: it sees every dispatch (checks.py)
            return wrap(self, op, impl, device_type, key)
        return impl

    # ---- the record

    def record_site(self, **entry):
        """One call site's row of the kernel record (lower.py: op, site, node,
        shapes, layer, chosen, source, candidates, measured_us, kernels)."""
        with self._lock:
            self.sites.append(entry)

    def stat(self, name, value):
        """A per-step statistic for the record (kernels a step, crossings,
        distinct experts a layer, ...), from whoever measures it."""
        with self._lock:
            self.stats[name] = value
        self.write_record()

    def kernel_record(self):
        """The kernel record as JSON-able data: the hardware, the choices,
        the knobs in effect, every call site the lowering saw, the step's
        statistics."""
        import torch
        hw = dict(self.hardware)
        hw["torch"] = torch.__version__
        if hw.get("gpu"):
            hw["cuda"] = torch.version.cuda
        try:
            import triton
            hw["triton"] = triton.__version__
        except ImportError:
            pass
        with self._lock:
            sites = [dict(e) for e in self.sites]
            stats = dict(self.stats)
        try:
            from . import triton_ops
            # the GEMV shape classes launched so far, with the blocks that ran (the measuring step's input);
            # a process that launched none (the CPU, a test) writes no entry
            if triton_ops.GEMV_CLASSES:
                stats["gemv_classes"] = {k: dict(v, blocks=dict(v["blocks"])) for k, v in triton_ops.GEMV_CLASSES.items()}
        except ImportError:
            pass
        return {
            "format": RECORD_FORMAT,
            "hardware": hw,
            "choices": {k: (c.value is not None) if k in LOCAL_PATHS else c.value for k, c in self.choices.items()},
            "knobs": effective_knobs(),
            "sites": sites,
            "stats": stats,
        }

    def write_record(self):
        """Write the kernel record where VXPU_PLAN_RECORD says, if it does
        (rewritten whole each time: after the lowering, and as statistics
        arrive)."""
        path = getattr(self, "_record_path", None)
        if not path:
            return None
        record = self.kernel_record()
        tmp = f"{path}.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(record, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path

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


def check_record(table, record):
    """The sites of a record that did not run what the table names for
    them: [(site entry, expected impl)]. A table applied exactly gives
    none; anything else is the failure the planner cannot tolerate (a
    choice measured, recorded and not applied: #490)."""
    bad = []
    for entry in record.get("sites", []):
        want = table_entry(table, entry["op"], entry.get("site"), entry.get("shapes"))
        if want is not None and want["impl"] != entry.get("chosen"):
            bad.append((entry, want["impl"]))
    return bad


def describe_mismatches(bad):
    return "; ".join(f"{e['op']} at {e.get('site') or e.get('shapes') or 'a site'}: ran {e.get('chosen')}, "
                     f"the table names {want}" for e, want in bad)


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


def profile_step(artifact, table=None, steps=4, heavy_us=50.0):
    """On a GPU: load the engine for artifact (the environment's plan, the
    table given or VXPU_PLAN_TABLE's), run decode steps under the CUDA
    profiler and report what launched: every kernel name with its count and
    time in one step, next to the table's choices. Two checks: the record's
    sites against the table (check_record), and the compiler-fused kernels
    (inductor's, named triton_*_fused_*) heavier than heavy_us, which no
    chosen body accounts for when every site of their op chose a kernel
    body: that is how a body the table did not choose shows itself (#490).
    Returns (report dict, problems list)."""
    import torch
    from .engine import Engine, SamplingParams
    if table is not None:
        os.environ["VXPU_PLAN_TABLE"] = table
    engine = Engine(artifact, device="cuda", compile_decode=True)
    greedy = SamplingParams(do_sample=False)
    prompt = [(11 * j + 5) % 30000 + 100 for j in range(24)]
    engine.generate_rows([prompt], 4, greedy)  # warm-up: the step captured
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        engine.generate_rows([prompt], steps, greedy)
    events = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                    key=lambda e: e.time_range.start)
    nk = (getattr(engine._decode_split, "node_counts", None) or {}).get("kernels")
    last = events[-nk:] if nk and nk <= len(events) else events
    counts = collections.Counter(e.name for e in last)
    times = collections.defaultdict(float)
    for e in last:
        times[e.name] += e.time_range.elapsed_us()
    record = engine.plan.kernel_record()
    problems = []
    if engine.plan.table is not None:
        bad = check_record(engine.plan.table, record)
        if bad:
            problems.append("sites that did not run the table's body: " + describe_mismatches(bad))
    fused_ops = {e["op"] for e in record["sites"] if e.get("source") != "kept" and e.get("kernels") == []}
    heavy = [(name, counts[name], times[name] / counts[name]) for name in counts
             if "_fused_" in name and name.startswith("triton_") and times[name] / counts[name] > heavy_us]
    if heavy and not fused_ops:
        problems.append("compiler-fused kernels heavier than %.0f us with no site choosing a fused body: " % heavy_us
                        + ", ".join(f"{n} x{c} {us:.0f} us" for n, c, us in heavy))
    report = {"kernels_in_step": len(last), "by_kernel": [{"name": n, "count": counts[n], "us": round(times[n], 1)}
                                                          for n in sorted(counts, key=lambda n: -times[n])],
              "record": record, "problems": problems}
    return report, problems


def main(argv=None):
    p = argparse.ArgumentParser(description="Explain the physical plan for an artifact, without loading weights; "
                                "check a kernel record against a kernel table; or profile a step on a GPU.")
    p.add_argument("--artifact", help="the artifact to explain (or to run, with --profile-step)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true", help="the plan for a compiled decode (the executor's GPU default)")
    p.add_argument("--json", action="store_true", help="print the full explain() JSON")
    p.add_argument("--table", help="a kernel table (file or inline JSON) for --check-record or --profile-step")
    p.add_argument("--check-record", metavar="RECORD", help="check a kernel record file against --table (no GPU)")
    p.add_argument("--profile-step", action="store_true",
                   help="on a GPU: load the engine for --artifact, profile a decode step and report what launched")
    args = p.parse_args(argv)
    # run as `python -m vxpu.plan` this file is __main__, a second copy of
    # the module: the ops register into vxpu.plan's registry, so use that one
    from . import ops as _ops  # noqa: F401  (registers the implementations)
    from . import plan as registered
    if args.check_record:
        if not args.table:
            p.error("--check-record needs --table")
        table = registered.load_table(args.table)
        with open(args.check_record) as f:
            record = json.load(f)
        bad = registered.check_record(table, record)
        print(f"{len(record.get('sites', []))} sites, {len(bad)} did not run the table's body"
              + (": " + registered.describe_mismatches(bad) if bad else ""))
        raise SystemExit(1 if bad else 0)
    if not args.artifact:
        p.error("--artifact is required")
    if args.profile_step:
        report, problems = registered.profile_step(args.artifact, args.table)
        print(json.dumps(report if args.json else {k: v for k, v in report.items() if k != "record"}, indent=1))
        for problem in problems:
            print("PROBLEM: " + problem)
        raise SystemExit(1 if problems else 0)
    plan = registered.Plan(args.device, compile_decode=args.compile)
    used = registered.artifact_ops(args.artifact)
    print(json.dumps(plan.explain(used), indent=1) if args.json else plan.summary(used))


if __name__ == "__main__":
    main()
