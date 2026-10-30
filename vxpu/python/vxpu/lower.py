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


"""The plan, lowered into the graph: each vxpu op call replaced by the
implementation the plan selects for it (#316, #277).

Why. In the exported graph (the artifact, the logical plan) a dense op
is one opaque node: vxpu::fp8_linear is a custom op whose body is a
Python function, which asks the plan which implementation to run
(ops.py, Plan.select) and launches its Triton kernels directly. That is
deliberate: the artifact stays portable and the executor chooses. But to
a compiler (inductor, behind torch.compile and AOTInductor) the node is
"call this Python function": it cannot see the kernels, plan their
memory or launch them from its own code, and AOTInductor, which
generates C++ with no Python in it, cannot keep the call at all.

What. In compiler terms, devirtualization and inlining. The custom op is
an indirect call; for the decode graph the plan's choice is known before
running (each call site's shapes are fixed and recorded on the graph's
nodes), so the indirect call is replaced by the selected implementation's
body. For a decode-step fp8 linear on CUDA:

    before   y  = vxpu.fp8_linear(x, w, s)            # opaque; Python picks
                                                       # gpu-fp8-gemv per call
    after    xq = vxpu_triton.act_quant_dequant(x)     # Triton kernel launches
             y0 = vxpu_triton.fp8_gemv(xq.view(1, K), w, s, 0, bf16)  # the
             y  = y0.view(1, 1, N)                     # compiler can see

The same kernels run on the same inputs; what changes is who launches
them (inductor's generated code instead of our Python).

How. The body is the implementation's own function, called with its
kernels as torch ops (tk=triton_ops.lowered: each Triton kernel
registered with torch.library.triton_op, which tells PyTorch's tracer
"this is a Triton kernel launch, with these inputs and outputs" -- the
wrapping is for the tracer, Triton compiles the kernel the same either
way). It is traced once on the node's shapes (make_fx, fake tensors:
nothing is computed) and the resulting ops are inlined in place of the
node, so the graph calls torch ops only and nothing in Python is left
between the kernels (a node pointing at a Python function would be
traced through, and guarded on, at every call).

What stays. Only implementations registered lowerable (pure tensor work:
the fp8 GEMV, the Sinkhorn, head and rotary kernels) are inlined; a call whose
selection is not lowerable (a bf16 weight copy kept in the engine's
cache, say) stays the op it was, and still runs, through its op. Host
ops (the experts, the n-gram rows) are never lowered: they are the
segment boundaries. The report (describe(), kept as a plan note) says
what was lowered and what was kept.

Fusion. Where an op has a decomposed body (plain tensor operations,
registered fused) and the plan fuses that op, the body competes with the
kernel at each call's shapes, and the cheaper one, measured on the
engine's GPU, is inlined (costs.py). Small aten matmuls likewise become
reductions where that measures cheaper (_fuse_matmuls).

Numerics. A lowered kernel call is bit-identical to the op, but only because
the kernels guard against two ways a compiler's launch differs from
Triton's own (see triton_ops.py), which any new lowerable kernel must
also respect:
  - inductor passes a Python float argument as fp64 where Triton's
    launcher passes fp32: a kernel converts such arguments to fp32
    itself, or its arithmetic silently becomes fp64;
  - a torch-side cast before a kernel (x.float()) is fused into x's
    producer, skipping the rounding to x's dtype the eager path does:
    a kernel loads its inputs' own dtype and widens in registers.
A fused body or reduction sums in its own order, so it is equal to the
op up to rounding. And inductor's on-disk caches did not notice a changed kernel source
(#341), so while lowering the cache directory is keyed on triton_ops.py.
"""

import collections
import hashlib
import inspect
import operator
import os

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from . import costs, ops, triton_ops
from .segments import HOST_OPS


class _NoShapes(Exception):
    pass


def _value(gm, arg):
    """A node argument as the op would see it: an attribute's tensor, a
    node's recorded value (a fake tensor with the shape, dtype and
    device), anything else as is."""
    if isinstance(arg, torch.fx.Node):
        if arg.op == "get_attr":
            obj = gm
            for part in arg.target.split("."):
                obj = getattr(obj, part)
            return obj
        if "val" not in arg.meta:
            raise _NoShapes(arg.name)
        return arg.meta["val"]
    if isinstance(arg, (list, tuple)):
        return type(arg)(_value(gm, a) for a in arg)
    return arg


def _implementation(op, impl):
    """A function computing op's result with impl, its kernels traced."""
    def fn(*args):
        return ops.call(op, lambda *a: impl.fn(*a, tk=triton_ops.lowered), *args)
    return fn


def _inline(gm, node, fn, values):
    """Replace node with fn traced to torch ops on the node's values
    (shapes, dtypes and devices: fake tensors, nothing is computed).
    Non-tensor arguments are constants of the traced graph."""
    leaves, spec = pytree.tree_flatten(values)
    arg_leaves, _ = pytree.tree_flatten(node.args)
    at = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]

    def flat(*tensors):
        vals = list(leaves)
        for i, t in zip(at, tensors):
            vals[i] = t
        return fn(*pytree.tree_unflatten(vals, spec))

    with FakeTensorMode():
        examples = [torch.empty_strided(leaves[i].shape, leaves[i].stride(), dtype=leaves[i].dtype,
                                        device=leaves[i].device) for i in at]
    traced = make_fx(flat, tracing_mode="fake")(*examples)
    placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
    with gm.graph.inserting_before(node):
        out = gm.graph.graph_copy(traced.graph, {ph: arg_leaves[i] for ph, i in zip(placeholders, at)})
    if isinstance(out, (tuple, list)):  # a multi-output op: its users take items
        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                user.replace_all_uses_with(out[user.args[1]])
                gm.graph.erase_node(user)
    else:
        node.replace_all_uses_with(out)
    if not node.users:
        gm.graph.erase_node(node)


_keyed_cache_dir = None


def key_cache_dir():
    """Point inductor's on-disk caches at a directory keyed by the Triton
    kernels' source. A lowered graph embeds the kernels, and inductor's
    caches did not notice when a kernel's source changed: they served the
    code compiled from the old source (#341). Under the directory the
    caches would have used, so TORCHINDUCTOR_CACHE_DIR still chooses where.
    Idempotent; returns the directory."""
    global _keyed_cache_dir
    if _keyed_cache_dir is None:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        digest = hashlib.sha256(inspect.getsource(triton_ops).encode()).hexdigest()[:16]
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
        _keyed_cache_dir = os.path.join(os.path.abspath(base), f"vxpu-kernels-{digest}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = _keyed_cache_dir
    return _keyed_cache_dir


def lower(gm, plan):
    """Rewrite gm's vxpu op calls into their selected implementations,
    where lowerable. The engine lowers the whole decode graph before
    splitting it (segments.py): the host ops are left where they are, so
    the cut is the same. Returns {op: {"lowered": {impl: n}, "kept": {impl: n}}}
    for the calls seen; gm is recompiled if anything changed."""
    key_cache_dir()
    report = collections.defaultdict(lambda: {"lowered": collections.Counter(), "kept": collections.Counter(),
                                              "measured": {}})
    for node in list(gm.graph.nodes):
        target = str(node.target)
        if node.op != "call_function" or not target.startswith("vxpu.") or target in HOST_OPS:
            continue
        op = "vxpu::" + target.split(".")[1]
        if op not in ops.SELECTION or node.kwargs:
            report[op]["kept"]["(not lowerable)"] += 1
            continue
        try:
            args = _value(gm, node.args)
        except _NoShapes:
            report[op]["kept"]["(no shapes)"] += 1
            continue
        impl = ops.select(op, *args, plan=plan, lowering=True)
        if impl.fused:
            impl = _cheaper(op, impl, ops.select(op, *args, plan=plan), args, report)
        if not impl.lowerable:
            report[op]["kept"][impl.name] += 1
            continue
        _inline(gm, node, _implementation(op, impl), args)
        report[op]["lowered"][impl.name] += 1
    if "matmul" in plan["fuse"]:
        _fuse_matmuls(gm, report)
    if any(r["lowered"] for r in report.values()):
        gm.graph.lint()
        gm.recompile()
    return {op: {k: dict(v) for k, v in r.items() if v} for op, r in report.items()}


def _device(values):
    return next(v.device for v in pytree.tree_leaves(values) if isinstance(v, torch.Tensor))


def _cheaper(op, fused, other, args, report):
    """fused (a decomposed body) or other (the implementation selected without it), by
    their measured cost at these shapes (costs.py). Only a lowerable other competes: one
    that is not runs through its op, as the plan chose it (a bf16 weight copy kept in the
    engine's cache, say), which a measurement on its own could not reproduce."""
    if other.name == fused.name:
        return other
    if not other.lowerable:
        # the reference computation (plain tensor operations) is better inlined as the body
        # than kept as a call the compiler cannot take; any other unlowerable choice stays
        return fused if other.reference else other
    chosen, times = costs.choose(op, {fused.name: (_implementation(op, fused), True),
                                      other.name: (_implementation(op, other), False)}, args, _device(args))
    if times:
        report[op]["measured"][" ".join(costs.shapes(args))] = times
    return fused if chosen == fused.name else other


def _linear_reduction(x, w, b=None):
    """aten.linear as a multiply and a sum over K, in fp32 (cuBLAS also
    accumulates bf16 in fp32), rounded to x's dtype."""
    y = (x.float()[..., None, :] * w.float()).sum(-1)
    if b is not None:
        y = y + b.float()
    return y.to(x.dtype)


def _einsum_reduction(equation, a, b):
    """A two-operand einsum as a broadcast multiply and a sum over the
    contracted letters, in fp32. Both operands are permuted and viewed onto
    one letter order (the output's letters, then the contracted ones)."""
    (la, lb), out = _einsum_parts(equation)
    letters = list(dict.fromkeys(out + la + lb))
    sizes = {c: n for l, t in ((la, a), (lb, b)) for c, n in zip(l, t.shape)}

    def onto(t, l):
        t = t.float().permute([l.index(c) for c in letters if c in l])
        return t.reshape([sizes[c] if c in l else 1 for c in letters])

    y = onto(a, la) * onto(b, lb)
    if len(letters) > len(out):
        y = y.sum(tuple(range(len(out), len(letters))))
    return y.to(torch.result_type(a, b))


def _einsum_parts(equation):
    """((lhs letters, rhs letters), output letters), or None for an einsum
    this rewrite does not handle: other than two operands, an ellipsis, or
    a letter repeated within one operand (a diagonal)."""
    if "->" not in equation or "." in equation:
        return None
    ins, out = equation.replace(" ", "").split("->")
    ins = ins.split(",")
    if len(ins) != 2 or any(len(set(l)) != len(l) for l in ins):
        return None
    return tuple(ins), out


def _matmul_rows(node, gm):
    """The fewer free rows of node's two sides if it is a contraction this
    rewrite handles (aten.linear, a two-operand aten.einsum), else None.
    A side's free rows are the output elements only its operand indexes: a
    linear's are its input's leading dimensions."""
    try:
        if node.target is torch.ops.aten.linear.default:
            x, w = _value(gm, node.args[:2])
            return x.numel() // x.shape[-1] if w.dim() == 2 else None
        if node.target is torch.ops.aten.einsum.default and len(node.args[1]) == 2:
            parts = _einsum_parts(node.args[0])
            if parts is None:
                return None
            (la, lb), out = parts
            a, b = _value(gm, node.args[1])
            rows = [1, 1]
            for i, (l, other, t) in enumerate(((la, lb, a), (lb, la, b))):
                for c, n in zip(l, t.shape):
                    if c in out and c not in other:
                        rows[i] *= n
            return min(rows)
    except _NoShapes:
        pass
    return None


def _fuse_matmuls(gm, report):
    """Rewrite small contractions into reductions the compiler fuses with
    their neighbours (plan fuse "matmul"), where that measures cheaper. To
    inductor a matmul is a library call (cuBLAS), one kernel launch with its
    inputs written out first, which for small tensors costs more than the
    arithmetic; but a reduction reads the weight again for every row, so
    for large weights the library call wins (costs.py). Only up to
    ops.FUSED_GEMV_MAX_ROWS rows, as the fused fp8 linear, and on CUDA,
    where the costs are measured."""
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        rows = _matmul_rows(node, gm)
        if rows is None:
            continue
        op = "aten::" + node.target.__name__.split(".")[0]
        values = _value(gm, node.args)
        if rows > ops.FUSED_GEMV_MAX_ROWS or _device(values).type != "cuda":
            report[op]["kept"]["aten"] += 1
            continue
        if node.target is torch.ops.aten.linear.default:
            body = _linear_reduction
        else:  # the equation is a constant of the traced body
            def body(equation, operands):
                return _einsum_reduction(equation, *operands)
        aten = node.target
        chosen, times = costs.choose(op, {"reduction": (body, True), "aten": (lambda *a: aten(*a), False)},
                                     values, _device(values))
        if times:
            report[op]["measured"][" ".join(costs.shapes(values))] = times
        if chosen == "aten":
            report[op]["kept"]["aten"] += 1
            continue
        _inline(gm, node, body, values)
        report[op]["lowered"]["reduction"] += 1


REPORT_PREFIX = "lowered the dense segments' op calls: "


def describe(report):
    """One line for logs and plan notes."""
    parts = []
    for op, r in report.items():
        done = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("lowered", {}).items()))
        kept = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("kept", {}).items()))
        parts.append(f"{op.split('::')[1]}: " + "; ".join(x for x in (done and f"lowered {done}", kept and f"kept {kept}") if x))
    return REPORT_PREFIX + (" | ".join(parts) or "none")


def describe_measured(report):
    """The measured costs behind the choices (costs.py), one line, or None."""
    parts = [f"{op.split('::')[1]} {shape}: " + ", ".join(f"{name} {us} us" for name, us in sorted(times.items()))
             for op, r in report.items() for shape, times in r.get("measured", {}).items()]
    return "lowering costs, measured per call shape: " + " | ".join(parts) if parts else None
