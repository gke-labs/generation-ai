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
the fp8 GEMV, the Sinkhorn, head, rotary, and index_topk kernels) are inlined; a call whose
selection is not lowerable (a bf16 weight copy kept in the engine's
cache, say) stays the op it was, and still runs, through its op. Host
ops (the experts, the n-gram rows) are never lowered: they are the
segment boundaries. The report (describe(), kept as a plan note) says
what was lowered and what was kept.

Numerics. A lowered call is bit-identical to the op, but only because
the kernels guard against two ways a compiler's launch differs from
Triton's own (see triton_ops.py), which any new lowerable kernel must
also respect:
  - inductor passes a Python float argument as fp64 where Triton's
    launcher passes fp32: a kernel converts such arguments to fp32
    itself, or its arithmetic silently becomes fp64;
  - a torch-side cast before a kernel (x.float()) is fused into x's
    producer, skipping the rounding to x's dtype the eager path does:
    a kernel loads its inputs' own dtype and widens in registers.
And inductor's on-disk caches did not notice a changed kernel source
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

from . import ops, triton_ops
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
    report = collections.defaultdict(lambda: {"lowered": collections.Counter(), "kept": collections.Counter()})
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
        impl = ops.select(op, *args, plan=plan)
        if not impl.lowerable:
            report[op]["kept"][impl.name] += 1
            continue
        _inline(gm, node, _implementation(op, impl), args)
        report[op]["lowered"][impl.name] += 1
    if any(r["lowered"] for r in report.values()):
        gm.graph.lint()
        gm.recompile()
    return {op: {k: dict(v) for k, v in r.items() if v} for op, r in report.items()}


REPORT_PREFIX = "lowered the dense segments' op calls: "


def describe(report):
    """One line for logs and plan notes."""
    parts = []
    for op, r in report.items():
        done = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("lowered", {}).items()))
        kept = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("kept", {}).items()))
        parts.append(f"{op.split('::')[1]}: " + "; ".join(x for x in (done and f"lowered {done}", kept and f"kept {kept}") if x))
    return REPORT_PREFIX + (" | ".join(parts) or "none")
