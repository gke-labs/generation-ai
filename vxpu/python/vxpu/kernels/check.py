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

"""Runs a kernel variant against its reference and measures efficiency."""

# pyright: strict

import argparse
import dataclasses
import importlib
import math
import sys
import time
from typing import Any, Sequence

import torch

from .. import ops, plan
from ..plan import Numerics, REGISTRY, reference_of
from .cards import bandwidth_for

ROUNDING_TOLERANCE: float = 0.05


@dataclasses.dataclass(frozen=True)
class Result:
    """The result of checking one kernel variant on one shape."""

    op: str
    variant: str
    shapes: str
    numerics: Numerics
    passed: bool
    deviation: float
    bytes: int
    us: float
    floor_us: float | None
    fraction: float | None
    card: str
    timing: str = "warm"


def rel_deviation(got: torch.Tensor, want: torch.Tensor) -> float:
    """Computes max |got - want| over max |want|."""
    diff = (got.float() - want.float()).abs().max().item()
    denom = want.float().abs().max().item()
    if denom == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return float(diff / denom)


def floor_us(tensor_bytes: int, bandwidth_gbs: float) -> float:
    """Calculates theoretical memory floor latency in microseconds."""
    return tensor_bytes / (bandwidth_gbs * 1e3)


def _extract_tensors(val: Any) -> list[torch.Tensor] | None:
    if isinstance(val, torch.Tensor):
        return [val]
    if type(val) in (tuple, list):
        return [t for t in val if isinstance(t, torch.Tensor)]
    return None


def _judge(numerics: Numerics, got: Any, want: Any) -> tuple[bool, float]:
    """Judges got against want according to the numerics class."""
    got_tensors = _extract_tensors(got)
    want_tensors = _extract_tensors(want)
    if got_tensors is None or want_tensors is None:
        return False, float("inf")

    if len(got_tensors) != len(want_tensors):
        return False, float("inf")

    for g, w in zip(got_tensors, want_tensors):
        if g.shape != w.shape or g.dtype != w.dtype:
            return False, float("inf")

    if numerics == "bit-identical":
        all_equal = all(torch.equal(g, w) for g, w in zip(got_tensors, want_tensors))
        if all_equal:
            return True, 0.0
        devs = [rel_deviation(g, w) for g, w in zip(got_tensors, want_tensors) if g.is_floating_point()]
        max_dev = max(devs) if devs else float("inf")
        return False, max_dev

    if numerics == "rounding":
        devs: list[float] = []
        for g, w in zip(got_tensors, want_tensors):
            if g.is_floating_point():
                dev = rel_deviation(g, w)
                devs.append(dev)
                if not math.isfinite(dev) or dev > ROUNDING_TOLERANCE:
                    return False, dev
            else:
                if not torch.equal(g, w):
                    return False, float("inf")
        max_dev = max(devs) if devs else 0.0
        return True, max_dev

    raise ValueError(f"unknown numerics class {numerics!r}")


def _find_inputs(op: str) -> Any:
    op_bare = op.removeprefix("vxpu::")
    mod_name = f"vxpu.kernels.{op_bare}"
    try:
        return getattr(importlib.import_module(mod_name), "inputs")
    except (ImportError, AttributeError):
        raise ValueError(f"no input builder found for op {op!r} at {mod_name}.inputs") from None


def run_check(
    op: str,
    name: str,
    shapes: str,
    device: str = "cpu",
    card: str = "cpu",
    bandwidth_gbs: float | None = None,
    repeat: int = 20,
) -> Result:
    """Runs a kernel variant against its reference and measures efficiency."""
    if op not in REGISTRY:
        raise ValueError(f"unknown op {op!r}; registered: {', '.join(REGISTRY.keys())}")
    impl = next((i for i in REGISTRY[op] if i.name == name), None)
    if impl is None:
        available = [i.name for i in REGISTRY[op]]
        raise ValueError(f"unknown variant {name!r} for {op}; available: {', '.join(available)}")
    ref = reference_of(op)
    if ref is None:
        raise ValueError(f"no reference registered for {op}")

    inputs_fn = _find_inputs(op)
    args: tuple[Any, ...] = inputs_fn(shapes, device)

    cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    ops_mod: Any = ops
    got: Any = ops_mod.call(op, impl.fn, *args)
    want: Any = ops_mod.call(op, ref.fn, *cloned_args)
    passed, dev = _judge(impl.numerics, got, want)

    # Tensor bytes summed directly over the one-node graph's inputs and outputs.
    total_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))
    if (ts := _extract_tensors(got)) is not None:
        total_bytes += sum(t.numel() * t.element_size() for t in ts)

    on_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Timing: 3 warm calls then best of repeat.
    for _ in range(3):
        ops_mod.call(op, impl.fn, *args)
    if on_cuda:
        torch.cuda.synchronize()

    best_us = float("inf")
    for _ in range(max(1, repeat)):
        if on_cuda:
            start_event: Any = torch.cuda.Event(enable_timing=True)
            end_event: Any = torch.cuda.Event(enable_timing=True)
            start_event.record()
            ops_mod.call(op, impl.fn, *args)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_us: float = float(start_event.elapsed_time(end_event)) * 1000.0
        else:
            t0 = time.perf_counter()
            ops_mod.call(op, impl.fn, *args)
            t1 = time.perf_counter()
            elapsed_us = (t1 - t0) * 1e6
        if elapsed_us < best_us:
            best_us = elapsed_us

    if bandwidth_gbs is not None and bandwidth_gbs > 0:
        fl_us = floor_us(total_bytes, bandwidth_gbs)
        frac = fl_us / best_us if best_us > 0 else None
    else:
        fl_us = None
        frac = None

    return Result(
        op=op,
        variant=name,
        shapes=shapes,
        numerics=impl.numerics,
        passed=passed,
        deviation=dev,
        bytes=total_bytes,
        us=best_us,
        floor_us=fl_us,
        fraction=frac,
        card=card,
        timing="warm",
    )


def line(r: Result) -> str:
    """Renders one result line for a kernel check."""
    status = "PASS" if r.passed else "FAIL"
    fl_str = f"{r.floor_us:.1f}" if r.floor_us is not None else "n/a"
    fr_str = f"{r.fraction:.2f}" if r.fraction is not None else "n/a"
    return (
        f"kernel-check {r.op} {r.variant} {r.shapes} {r.numerics} "
        f"dev={r.deviation:.1e} {status} bytes={r.bytes / 1e6:.1f}MB "
        f"us={r.us:.1f} floor_us={fl_str} fraction={fr_str} "
        f'card="{r.card}" timing={r.timing}'
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parses arguments, executes kernel checks across shapes, and reports results."""
    parser = argparse.ArgumentParser(description="Check a kernel variant against its reference.")
    parser.add_argument("op", help="Op name (e.g. vxpu::fp8_linear)")
    parser.add_argument("variant", help="Variant name (e.g. bf16_dequant)")
    parser.add_argument("--shapes", action="append", required=True, help="Input shapes (e.g. 1x128->64)")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--card", default=None, help="Card name for bandwidth lookup")
    parser.add_argument("--bandwidth-gb-s", type=float, default=None, help="Override bandwidth in GB/s")
    parser.add_argument("--repeat", type=int, default=20, help="Number of timing iterations")

    args = parser.parse_args(argv)

    card = args.card
    if card is None:
        if args.device.startswith("cuda"):
            hw = plan.probe_hardware(args.device)
            card = str(hw.get("gpu", "unknown"))
        else:
            card = "cpu"

    bw_gbs = args.bandwidth_gb_s
    if bw_gbs is None:
        bw_gbs = bandwidth_for(card)

    all_passed = True
    for shapes in args.shapes:
        res = run_check(
            op=args.op,
            name=args.variant,
            shapes=shapes,
            device=args.device,
            card=card,
            bandwidth_gbs=bw_gbs,
            repeat=args.repeat,
        )
        print(line(res), flush=True)
        if not res.passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
