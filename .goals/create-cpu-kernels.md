---
metric:
  command: ./vxpu/dev/goals/cpu-kernel-score
  direction: higher
  unit: summed log2 speedup over the reference
  minImprovement: 0.1
  runs: 3
  timeout: 30m
verify: ./dev/ci/presubmits/test-vxpu-python
budget:
  maxDuration: 2h
  maxCostUSD: 20
schedule: manual
---
# Write faster CPU kernel variants

vXPU runs a model as a graph of ops. Each op has one reference implementation and any number of variants; the planner picks a variant per call site from measurements. Your job is to add one variant that computes the same result as the reference, faster, on the CPU.

## The contract

- A variant is one file: `vxpu/python/vxpu/kernels/<op>/<variant>.py`. The file stem is the variant's name. Read `vxpu/python/vxpu/kernels/fp8_linear/bf16_dequant.py` first; it is the worked example.
- Register it with `@variant(devices=("cpu",), numerics=..., when=...)` from `vxpu.kernels`. `when` is one sentence that says when this variant is the right choice.
- Declare the numerics class honestly. `"bit-identical"` means equal to the reference in every bit. `"rounding"` means the largest deviation is within the check's tolerance. The check enforces the class you declare.
- A variant takes the op's arguments and returns the op's result. It knows tensors, shapes and dtypes. It does not know the model: no layer names, no checkpoint names, no model-specific constants.
- New files start with `# pyright: strict` and annotate every signature. Docstrings use the active voice and lead with what the function returns.
- To score an op that has no input builder yet, add `inputs(shapes, device, seed)` and `SCORE_SHAPES` to `vxpu/python/vxpu/kernels/<op>/__init__.py`, with shapes the op meets in a real model. Do that in its own change, before the variant.

## How to measure

```
python -m vxpu.kernel_tools.check vxpu::<op> <variant> --shapes <shape> --device cpu
./vxpu/dev/goals/cpu-kernel-score
```

The first command checks one variant against the reference and prints its time. The second scores every op; its last line is the number this goal raises.

## Rules

- One variant per change. Keep it small enough to read in one sitting.
- Do not edit the reference implementations, `kernel_tools/check.py`, `kernel_tools/score.py`, the tolerance, the score shapes of an existing op, or the tests that guard them. A faster number from a weaker check is a failure.
- A C++ kernel belongs in `vxpu/python/vxpu/csrc/` and is called from the variant file; the variant must fall back cleanly (not register) when the extension cannot be built.
- Put the `kernel-check` lines for your variant and the reference in the commit body, with the machine they ran on.
- Follow `AGENTS.md`.
