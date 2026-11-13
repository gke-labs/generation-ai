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

"""Scores kernel variants on the CPU against their reference implementations."""

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
TimerFn = Callable[[str, plan.Impl, Sequence[plan.Impl], str], Mapping[str, float]]


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


def _make_result(
    op: str,
    variant: str,
    shape: str,
    numerics: plan.Numerics,
    passed: bool,
    deviation: float,
    total_bytes: int,
    us: float = 0.0,
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
        floor_us=None,
        fraction=None,
        card="cpu",
        timing="warm",
    )


def time_impls(
    op: str,
    ref: plan.Impl,
    active_variants: Sequence[plan.Impl],
    shape: str,
) -> dict[str, float]:
    """Times reference and active variants interleaved across warm and timing rounds."""
    inputs_fn: Any = check.find_inputs(op)
    args: tuple[Any, ...] = inputs_fn(shape, "cpu")
    cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    ops_mod: Any = ops

    warm_var_times: dict[str, list[float]] = {v.name: [] for v in active_variants}
    for _ in range(5):
        ops_mod.call(op, ref.fn, *cloned_args)
        for v in active_variants:
            t0 = time.perf_counter()
            ops_mod.call(op, v.fn, *args)
            warm_var_times[v.name].append((time.perf_counter() - t0) * 1e6)

    repeats: dict[str, int] = {}
    for v in active_variants:
        med_warm_us = statistics.median(warm_var_times[v.name])
        repeats[v.name] = max(1, min(10, int(20_000 / max(med_warm_us, 1.0))))

    ref_times: list[float] = []
    ratios: dict[str, list[float]] = {v.name: [] for v in active_variants}

    t_start = time.perf_counter()
    rounds = 0
    while rounds < 20 or (time.perf_counter() - t_start < 3.0 and rounds < 50):
        t0 = time.perf_counter()
        ops_mod.call(op, ref.fn, *cloned_args)
        t_ref = (time.perf_counter() - t0) * 1e6
        ref_times.append(t_ref)

        for v in active_variants:
            rep = repeats[v.name]
            t0 = time.perf_counter()
            for _ in range(rep):
                ops_mod.call(op, v.fn, *args)
            t_var = ((time.perf_counter() - t0) * 1e6) / rep
            ratios[v.name].append(t_ref / t_var if t_var > 0 else 1.0)
        rounds += 1

    ref_med = statistics.median(ref_times) if ref_times else 0.0
    timings: dict[str, float] = {ref.name: ref_med}
    for v in active_variants:
        var_ratios = ratios[v.name]
        med_ratio = statistics.median(var_ratios) if var_ratios else 1.0
        timings[v.name] = (ref_med / med_ratio) if med_ratio > 0 else ref_med
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


def variants_for_op(op: str) -> list[plan.Impl]:
    """Returns the registered CPU variants for an op, excluding the reference."""
    if op not in plan.REGISTRY:
        return []
    return [impl for impl in plan.REGISTRY[op] if not impl.reference and "cpu" in impl.devices]


def score_shape(
    op: str,
    ref: plan.Impl,
    active_variants: Sequence[plan.Impl],
    shape: str,
    out: TextIO = sys.stdout,
    timer_fn: TimerFn = time_impls,
) -> tuple[float, str]:
    """Verifies and times variants for one shape, printing check lines and returning best speedup."""
    inputs_fn: Any = check.find_inputs(op)
    args: tuple[Any, ...] = inputs_fn(shape, "cpu")
    cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    ops_mod: Any = ops

    want: Any = ops_mod.call(op, ref.fn, *cloned_args)
    total_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))
    if (ts := check.extract_tensors(want)) is not None:
        total_bytes += sum(t.numel() * t.element_size() for t in ts)

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
            )
            print(check.line(fail_res), file=out, flush=True)
            raise CheckFailedError(f"{op} variant {var.name} failed {var.numerics} on {shape}")

    timings = timer_fn(op, ref, active_variants, shape)
    ref_us = timings[ref.name]
    var_us = {v.name: timings[v.name] for v in active_variants}

    ref_res = _make_result(
        op=op,
        variant=ref.name,
        shape=shape,
        numerics=ref.numerics,
        passed=True,
        deviation=0.0,
        total_bytes=total_bytes,
        us=ref_us,
    )
    print(check.line(ref_res), file=out, flush=True)

    best_speedup = 0.0
    best_variant = "none"
    for v in active_variants:
        v_us = var_us[v.name]
        v_res = _make_result(
            op=op,
            variant=v.name,
            shape=shape,
            numerics=v.numerics,
            passed=True,
            deviation=devs[v.name],
            total_bytes=total_bytes,
            us=v_us,
        )
        print(check.line(v_res), file=out, flush=True)

        speedup = (ref_us / v_us) if v_us > 0 else 0.0
        if speedup > best_speedup:
            best_speedup = speedup
            best_variant = v.name

    return best_speedup, best_variant


def score_op(
    op: str,
    shapes: Sequence[str],
    out: TextIO = sys.stdout,
    timer: TimerFn | None = None,
) -> OpScore:
    """Times an op's reference and CPU variants across shapes, returning its score.

    Raises CheckFailedError if any variant fails its declared numerics class.
    """
    ref = plan.reference_of(op)
    if ref is None:
        raise ValueError(f"no reference registered for {op}")

    plan_mod: Any = plan
    dp: Any = plan_mod.default_plan("cpu")
    active_variants: list[plan.Impl] = []
    for var in variants_for_op(op):
        if var.applies is not None and not bool(var.applies(dp)):
            print(f"kernel-score {op} skipped={var.name} reason=not available on this machine", file=out, flush=True)
            continue
        active_variants.append(var)

    if len(active_variants) == 0:
        return OpScore(op=op, best_variant="none", speedup=0.0, score=0.0, shape="none")

    timer_fn = timer or time_impls
    best_speedup = 0.0
    best_variant = "none"
    best_shape = "none"

    for shape in shapes:
        speedup, variant = score_shape(
            op=op,
            ref=ref,
            active_variants=active_variants,
            shape=shape,
            out=out,
            timer_fn=timer_fn,
        )
        if speedup > best_speedup:
            best_speedup = speedup
            best_variant = variant
            best_shape = shape

    op_score = math.log2(best_speedup) if best_speedup > 1.0 else 0.0
    return OpScore(
        op=op,
        best_variant=best_variant,
        speedup=best_speedup,
        score=op_score,
        shape=best_shape,
    )


def score_all(
    ops: Mapping[str, tuple[str, ...]] | None = None,
    out: TextIO = sys.stdout,
    timer: TimerFn | None = None,
) -> int:
    """Scores all qualifying ops across their score shapes and outputs results."""
    if ops is None:
        target_ops = discover_scorable_ops()
    else:
        target_ops = dict(ops)

    op_scores: list[OpScore] = []
    for op in sorted(target_ops.keys()):
        shapes = target_ops[op]
        try:
            op_score = score_op(op, shapes, out=out, timer=timer)
        except CheckFailedError:
            return 1
        op_scores.append(op_score)

    for s in op_scores:
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
    parser = argparse.ArgumentParser(description="Score kernel variants on CPU against reference.")
    parser.parse_args(argv)
    return score_all()


if __name__ == "__main__":
    sys.exit(main())
