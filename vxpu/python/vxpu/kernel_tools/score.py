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

"""Scores kernel variants on CPU or GPU against their reference implementations."""

# pyright: strict

import argparse
import dataclasses
import importlib
import math
import pkgutil
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

import torch

from .. import kernels, ops, plan
from . import check

THRESHOLD: float = 0.1
TimerFn = Callable[..., Mapping[str, float]]


class CheckFailedError(Exception):
    """Raised when a kernel variant fails exactness verification."""


@dataclasses.dataclass(frozen=True)
class OpScore:
    """The scoring result for one op across variants and shapes."""

    op: str
    best_variant: str
    speedup: float
    score: float
    shape: str
    fraction: float = 0.0


def _make_result(
    op: str,
    variant: str,
    shape: str,
    numerics: plan.Numerics,
    passed: bool,
    deviation: float,
    total_bytes: int,
    us: float = 0.0,
    floor_us: float | None = None,
    fraction: float | None = None,
    card: str = "cpu",
) -> check.Result:
    """Constructs a Result instance for checking or reporting."""
    return check.Result(
        op=op,
        variant=variant,
        shapes=shape,
        numerics=numerics,
        passed=passed,
        deviation=deviation,
        bytes=total_bytes,
        us=us,
        floor_us=floor_us,
        fraction=fraction,
        card=card,
        timing="warm",
    )


def measure_us(fn: Callable[[], Any], on_cuda: bool, rep: int = 1, device: str = "cpu") -> float:
    """Runs a function rep times and returns latency per repetition in microseconds."""
    if on_cuda:
        with torch.cuda.device(device):
            start_ev: Any = torch.cuda.Event(enable_timing=True)
            end_ev: Any = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(rep):
                fn()
            end_ev.record()
            torch.cuda.synchronize(device)
            return (float(start_ev.elapsed_time(end_ev)) * 1000.0) / rep
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return ((time.perf_counter() - t0) * 1e6) / rep


def percentile(data: Sequence[float], pct: float) -> float:
    """Returns the value at the given percentile (0.0 to 1.0) from a float sequence."""
    if not data:
        raise ValueError("cannot compute percentile of empty sequence")
    s = sorted(data)
    idx = int(round(pct * (len(s) - 1)))
    return s[idx]


def time_impls(
    op: str,
    ref: plan.Impl,
    active_variants: Sequence[plan.Impl],
    shape: str,
    device: str = "cpu",
) -> dict[str, float]:
    """Times reference and active variants interleaved across warm and timing rounds."""
    inputs_fn: Any = check.find_inputs(op)
    args: tuple[Any, ...] = inputs_fn(shape, device)
    cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    ops_mod: Any = ops
    on_cuda = device.split(":")[0] == "cuda" and torch.cuda.is_available()

    target_us = 20_000.0 if on_cuda else 100_000.0
    max_reps = 10 if on_cuda else 50

    warm_ref_times: list[float] = []
    warm_var_times: dict[str, list[float]] = {v.name: [] for v in active_variants}
    for _ in range(5):
        ops_mod.call(op, ref.fn, *cloned_args)
        if on_cuda:
            torch.cuda.synchronize(device)
        warm_ref_times.append(measure_us(lambda: ops_mod.call(op, ref.fn, *cloned_args), on_cuda, device=device))
        for v in active_variants:
            warm_var_times[v.name].append(measure_us(lambda: ops_mod.call(op, v.fn, *args), on_cuda, device=device))

    med_warm_ref_us = statistics.median(warm_ref_times)
    ref_rep = 1 if on_cuda else max(1, min(max_reps, int(target_us / max(med_warm_ref_us, 1.0))))

    repeats: dict[str, int] = {}
    for v in active_variants:
        med_warm_us = statistics.median(warm_var_times[v.name])
        repeats[v.name] = max(1, min(max_reps, int(target_us / max(med_warm_us, 1.0))))

    ref_times: list[float] = []
    var_times: dict[str, list[float]] = {v.name: [] for v in active_variants}
    ratios: dict[str, list[float]] = {v.name: [] for v in active_variants}

    t_start = time.perf_counter()
    rounds = 0
    while rounds < 20 or (time.perf_counter() - t_start < 3.0 and rounds < 50):
        t_ref = measure_us(lambda: ops_mod.call(op, ref.fn, *cloned_args), on_cuda, rep=ref_rep, device=device)
        ref_times.append(t_ref)

        for v in active_variants:
            rep = repeats[v.name]
            t_var = measure_us(lambda: ops_mod.call(op, v.fn, *args), on_cuda, rep=rep, device=device)
            var_times[v.name].append(t_var)
            if on_cuda:
                ratios[v.name].append(t_ref / t_var if t_var > 0 else 1.0)
        rounds += 1

    if on_cuda:
        ref_med = statistics.median(ref_times) if ref_times else 0.0
        timings: dict[str, float] = {ref.name: ref_med}
        for v in active_variants:
            var_ratios = ratios[v.name]
            med_ratio = statistics.median(var_ratios) if var_ratios else 1.0
            timings[v.name] = (ref_med / med_ratio) if med_ratio > 0 else ref_med
        return timings

    timings = {ref.name: percentile(ref_times, 0.20)}
    for v in active_variants:
        timings[v.name] = percentile(var_times[v.name], 0.20)
    return timings


def _parse_shapes(raw: Any) -> tuple[str, ...] | None:
    """Parses raw SCORE_SHAPES into a tuple of strings, or None if empty/invalid."""
    if type(raw) not in (tuple, list):
        return None
    result: list[str] = []
    for item in raw:
        result.append(str(item))
    if len(result) == 0:
        return None
    return tuple(result)


def discover_scorable_ops() -> dict[str, tuple[str, ...]]:
    """Finds all ops that have an input builder and SCORE_SHAPES defined."""
    _ = ops  # ensure ops are imported and registered
    kernels.discover()

    scorable: dict[str, tuple[str, ...]] = {}
    for op_info in pkgutil.iter_modules(kernels.__path__):
        if not op_info.ispkg or op_info.name.startswith("_"):
            continue
        mod_name = f"{kernels.__name__}.{op_info.name}"
        mod: Any = importlib.import_module(mod_name)
        inputs_fn: Any = getattr(mod, "inputs", None)
        shapes = _parse_shapes(getattr(mod, "SCORE_SHAPES", None))
        if callable(inputs_fn) and shapes is not None:
            op_name = f"vxpu::{op_info.name}"
            scorable[op_name] = shapes

    return scorable


def variants_for_op(op: str, device: str = "cpu") -> list[plan.Impl]:
    """Returns the registered variants for an op on device, excluding the reference."""
    if op not in plan.REGISTRY:
        return []
    device_type = device.split(":")[0]
    return [impl for impl in plan.REGISTRY[op] if not impl.reference and device_type in impl.devices]


def score_shape(
    op: str,
    ref: plan.Impl,
    active_variants: Sequence[plan.Impl],
    shape: str,
    out: TextIO = sys.stdout,
    timer_fn: TimerFn = time_impls,
    device: str = "cpu",
    card: str = "cpu",
    bandwidth_gbs: float | None = None,
) -> tuple[float, str, float, float]:
    """Verifies and times variants for one shape, returning best speedup, variant, variant fraction, and ref fraction."""
    inputs_fn: Any = check.find_inputs(op)
    args: tuple[Any, ...] = inputs_fn(shape, device)
    cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    ops_mod: Any = ops

    want: Any = ops_mod.call(op, ref.fn, *cloned_args)
    total_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))
    if (ts := check.extract_tensors(want)) is not None:
        total_bytes += sum(t.numel() * t.element_size() for t in ts)

    fl_us = check.floor_us(total_bytes, bandwidth_gbs) if bandwidth_gbs is not None and bandwidth_gbs > 0 else None

    devs: dict[str, float] = {}
    for var in active_variants:
        got: Any = ops_mod.call(op, var.fn, *args)
        passed, dev = check.judge(var.numerics, got, want)
        devs[var.name] = dev
        if not passed:
            fail_res = _make_result(
                op=op,
                variant=var.name,
                shape=shape,
                numerics=var.numerics,
                passed=False,
                deviation=dev,
                total_bytes=total_bytes,
                floor_us=fl_us,
                card=card,
            )
            print(check.line(fail_res), file=out, flush=True)
            raise CheckFailedError(f"{op} variant {var.name} failed {var.numerics} on {shape}")

    try:
        timings = timer_fn(op, ref, active_variants, shape, device=device)
    except TypeError:
        timings = timer_fn(op, ref, active_variants, shape)
    ref_us = timings[ref.name]
    ref_frac = (fl_us / ref_us) if fl_us is not None and ref_us > 0 else 0.0

    ref_res = _make_result(
        op=op,
        variant=ref.name,
        shape=shape,
        numerics=ref.numerics,
        passed=True,
        deviation=0.0,
        total_bytes=total_bytes,
        us=ref_us,
        floor_us=fl_us,
        fraction=ref_frac if fl_us is not None else None,
        card=card,
    )
    print(check.line(ref_res), file=out, flush=True)

    best_speedup = 0.0
    best_variant = "none"
    best_frac = 0.0
    for v in active_variants:
        v_us = timings[v.name]
        v_frac = (fl_us / v_us) if fl_us is not None and v_us > 0 else 0.0
        v_res = _make_result(
            op=op,
            variant=v.name,
            shape=shape,
            numerics=v.numerics,
            passed=True,
            deviation=devs[v.name],
            total_bytes=total_bytes,
            us=v_us,
            floor_us=fl_us,
            fraction=v_frac if fl_us is not None else None,
            card=card,
        )
        print(check.line(v_res), file=out, flush=True)

        speedup = (ref_us / v_us) if v_us > 0 else 0.0
        if speedup > best_speedup:
            best_speedup = speedup
            best_variant = v.name
            best_frac = v_frac

    return best_speedup, best_variant, best_frac, ref_frac


def score_op(
    op: str,
    shapes: Sequence[str],
    out: TextIO = sys.stdout,
    timer: TimerFn | None = None,
    device: str = "cpu",
    card: str = "cpu",
    bandwidth_gbs: float | None = None,
) -> OpScore:
    """Times an op's reference and variants across shapes, returning its score.

    Raises CheckFailedError if any variant fails its declared numerics class.
    """
    ref = plan.reference_of(op)
    if ref is None:
        raise ValueError(f"no reference registered for {op}")

    plan_mod: Any = plan
    dp: Any = plan_mod.default_plan(device)
    active_variants: list[plan.Impl] = []
    for var in variants_for_op(op, device=device):
        if var.applies is not None and not bool(var.applies(dp)):
            print(f"kernel-score {op} skipped={var.name} reason=not available on this machine", file=out, flush=True)
            continue
        active_variants.append(var)

    if len(active_variants) == 0:
        return OpScore(op=op, best_variant="none", speedup=0.0, score=0.0, shape="none", fraction=0.0)

    timer_fn = timer or time_impls
    device_type = device.split(":")[0]
    best_score = 0.0
    best_speedup = 0.0
    best_variant = active_variants[0].name
    best_shape = shapes[0] if shapes else "none"
    best_frac = 0.0

    for shape in shapes:
        speedup, variant, frac, ref_f = score_shape(
            op=op,
            ref=ref,
            active_variants=active_variants,
            shape=shape,
            out=out,
            timer_fn=timer_fn,
            device=device,
            card=card,
            bandwidth_gbs=bandwidth_gbs,
        )
        if device_type == "cuda":
            shape_score = max(0.0, frac - ref_f)
        else:
            shape_score = math.log2(speedup) if speedup > 1.0 else 0.0

        if shape_score > best_score:
            best_score = shape_score
            best_speedup = speedup
            best_variant = variant
            best_shape = shape
            best_frac = frac
        elif best_score == 0.0 and speedup > best_speedup:
            best_speedup = speedup
            best_variant = variant
            best_shape = shape
            best_frac = frac

    return OpScore(
        op=op,
        best_variant=best_variant,
        speedup=best_speedup,
        score=best_score,
        shape=best_shape,
        fraction=best_frac,
    )


def score_all(
    ops: Mapping[str, tuple[str, ...]] | None = None,
    out: TextIO = sys.stdout,
    timer: TimerFn | None = None,
    device: str = "cpu",
    card: str | None = None,
    bandwidth_gbs: float | None = None,
) -> int:
    """Scores all qualifying ops across their score shapes and outputs results."""
    device_type = device.split(":")[0]
    card_name: str
    if device_type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            print("kernel-score: CUDA requested but no CUDA-capable card is available", file=sys.stderr)
            return 1
        if card is None:
            idx = torch.device(device).index or 0
            card_name = torch.cuda.get_device_name(idx)
        else:
            card_name = card
        if bandwidth_gbs is None or bandwidth_gbs <= 0:
            print("kernel-score: CUDA scoring requires --bandwidth-gb-s from the machine database", file=sys.stderr)
            return 1
    else:
        card_name = "cpu"

    if ops is None:
        target_ops = discover_scorable_ops()
    else:
        target_ops = dict(ops)

    op_scores: list[OpScore] = []
    for op in sorted(target_ops.keys()):
        shapes = target_ops[op]
        try:
            op_score = score_op(
                op,
                shapes,
                out=out,
                timer=timer,
                device=device,
                card=card_name,
                bandwidth_gbs=bandwidth_gbs,
            )
        except CheckFailedError:
            return 1
        op_scores.append(op_score)

    for s in op_scores:
        if device_type == "cuda":
            print(
                f"kernel-score {s.op} best={s.best_variant} fraction={s.fraction:.2f} speedup={s.speedup:.2f} shape={s.shape}",
                file=out,
                flush=True,
            )
        else:
            print(
                f"kernel-score {s.op} best={s.best_variant} speedup={s.speedup:.2f} shape={s.shape}",
                file=out,
                flush=True,
            )

    total_score = sum(s.score for s in op_scores)
    print(f"{total_score:.3f}", file=out, flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parses arguments and runs kernel scoring across all qualifying ops."""
    parser = argparse.ArgumentParser(description="Score kernel variants on CPU or GPU against reference.")
    parser.add_argument("--device", default="cpu", help="Device to score (cpu or cuda)")
    parser.add_argument("--card", default=None, help="Card name (probed on cuda if unset)")
    parser.add_argument("--bandwidth-gb-s", type=float, default=None, help="Card bandwidth in GB/s")
    args = parser.parse_args(argv)
    return score_all(device=args.device, card=args.card, bandwidth_gbs=args.bandwidth_gb_s)


if __name__ == "__main__":
    sys.exit(main())
