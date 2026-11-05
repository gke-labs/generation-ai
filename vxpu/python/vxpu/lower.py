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
from . import segments
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


# ops that alias their first argument: a view of a parameter is still the parameter's bytes
_ALIASING = {"view", "reshape", "permute", "transpose", "t", "getitem", "slice", "select", "squeeze", "unsqueeze",
             "expand", "contiguous", "alias", "detach", "_to_copy", "to", "clone", "view_as", "flatten", "narrow",
             "unbind", "split", "chunk", "as_strided"}


def _root(node):
    """The node whose storage a view-like chain of nodes aliases: the
    parameter behind a permute, the activation behind a reshape."""
    seen = 0
    while isinstance(node, torch.fx.Node) and node.op in ("call_function", "call_method") and seen < 16:
        packet = getattr(node.target, "overloadpacket", None)  # aten.permute.default -> permute
        name = (packet.__name__ if packet is not None else getattr(node.target, "__name__", str(node.target)))
        name = name.split(".")[-1].rstrip("_")
        if name not in _ALIASING or not node.args or not isinstance(node.args[0], torch.fx.Node):
            break
        node = node.args[0]
        seen += 1
    return node


def site_bytes(gm, node):
    """The bytes a call site must move, from the graph alone: the
    parameters it reads (weights and scales: a get_attr, or the export's
    lifted placeholders p_* and c_*), the activations it reads and writes
    (every other input, and its outputs), and the state buffers it names
    (b_*: the caches, of which an op may read a window, so an upper bound
    kept apart). Together with the measured microseconds this is the
    site's line in the roofline ledger (pkg/kernelplan: Ledger); the floor
    is weights plus activations over the card's bandwidth."""
    out = {"weights": 0, "activations": 0, "state": 0}
    seen = set()
    for arg in pytree.tree_leaves(node.args):
        if not isinstance(arg, torch.fx.Node):
            continue
        try:
            value = _value(gm, arg)
        except _NoShapes:
            continue
        root = _root(arg)
        if root.op == "get_attr" or root.name.startswith(("p_", "c_")):
            kind = "weights"
        elif root.name.startswith("b_"):
            kind = "state"
        else:
            kind = "activations"
        for t in pytree.tree_leaves(value):
            if isinstance(t, torch.Tensor) and (root.name, kind) not in seen:
                out[kind] += t.numel() * t.element_size()
        seen.add((root.name, kind))
    for t in pytree.tree_leaves(node.meta.get("val")):
        if isinstance(t, torch.Tensor):
            out["activations"] += t.numel() * t.element_size()
    return out


def _implementation(op, impl):
    """A function computing op's result with impl, its kernels traced; with counters (the
    lowering's scratch slice, for an impl that declared it needs them), handed to impl.fn."""
    def fn(*args, counters=None):
        if counters is None:
            return ops.call(op, lambda *a: impl.fn(*a, tk=triton_ops.lowered), *args)
        return ops.call(op, lambda *a: impl.fn(*a, tk=triton_ops.lowered, counters=counters), *args)
    return fn


def _site(node):
    """The call site's name and layer: the module that called the op (the
    export's nn_module_stack, e.g. model.layers.14.engram.wkv), which is
    the same across exports of one model, and the layer index parsed from
    it (None for the head, the embedding). The kernel table names sites
    this way."""
    stack = node.meta.get("nn_module_stack") or {}
    path = None
    for value in stack.values():
        path = value[0] if isinstance(value, (tuple, list)) else str(value)
    layer = None
    if path:
        parts = path.split(".")
        for i, part in enumerate(parts[:-1]):
            if part in ("layers", "mtp") and parts[i + 1].isdigit():
                layer = int(parts[i + 1])
                break
    return path, layer


def _inline(gm, node, fn, values, device=None, counters=None):
    """Replace node with fn traced to torch ops on the node's values
    (shapes, dtypes and devices: fake tensors, nothing is computed).
    Non-tensor arguments are constants of the traced graph. Returns the
    kernels the body launches, as the traced graph names them (the
    vxpu_triton ops): empty for a body of plain tensor operations, which
    the compiler fuses into kernels of its own.

    device: the GPU the node's segment will run on under layer sharding
    (_segment_devices). The body is traced on it, so what it derives from
    its inputs' device (an arange on start.device, a cast's copy) names
    that GPU and not the one the lowering happens to run on; and any
    device literal the trace still carries for another GPU is rewritten
    before the copy into gm. Without it the values' own devices stand.

    counters: (n, scratch node) when the body needs n zeroed int32 arrival
    counters a step (Impl.counters): the node, a slice of the segment's
    scratch, is traced as one more input and handed to fn as counters=."""
    leaves, spec = pytree.tree_flatten(values)
    arg_leaves, _ = pytree.tree_flatten(node.args)
    at = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]

    def flat(*tensors):
        vals = list(leaves)
        for i, t in zip(at, tensors[:len(at)]):
            vals[i] = t
        if counters is not None:
            return fn(*pytree.tree_unflatten(vals, spec), counters=tensors[len(at)])
        return fn(*pytree.tree_unflatten(vals, spec))

    with FakeTensorMode():
        examples = [torch.empty_strided(leaves[i].shape, leaves[i].stride(), dtype=leaves[i].dtype,
                                        device=device if device is not None and leaves[i].is_cuda else leaves[i].device)
                    for i in at]
        if counters is not None:
            examples.append(torch.empty(counters[0], dtype=torch.int32,
                                        device=device if device is not None else _device(values)))
    traced = make_fx(flat, tracing_mode="fake")(*examples)
    if device is not None:
        _retarget(traced.graph, device)
    kernels = [str(n.target).removesuffix(".default") for n in traced.graph.nodes
               if n.op == "call_function" and str(n.target).startswith("vxpu_triton.")]
    placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
    mapping = {ph: arg_leaves[i] for ph, i in zip(placeholders, at)}
    if counters is not None:
        mapping[placeholders[len(at)]] = counters[1]
    with gm.graph.inserting_before(node):
        out = gm.graph.graph_copy(traced.graph, mapping)
    if isinstance(out, (tuple, list)):  # a multi-output op: its users take items
        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                user.replace_all_uses_with(out[user.args[1]])
                gm.graph.erase_node(user)
    else:
        node.replace_all_uses_with(out)
    if not node.users:
        gm.graph.erase_node(node)
    return kernels


def _retarget(graph, device):
    """Point every device literal of graph that names a CUDA device other
    than `device` at device: the kwargs of factories and copies (arange,
    zeros, _to_copy) and positional torch.device arguments. Returns how
    many nodes changed."""
    changed = 0
    for n in graph.nodes:
        def fix(v):
            if isinstance(v, (str, torch.device)):
                try:
                    d = torch.device(v)
                except (RuntimeError, TypeError):
                    return v
                if d.type == "cuda" and d != device:
                    return device
            return v
        kwargs = {k: fix(v) for k, v in n.kwargs.items()}
        args = tuple(fix(v) if isinstance(v, torch.device) else v for v in n.args)
        if kwargs != dict(n.kwargs) or args != tuple(n.args):
            n.kwargs, n.args = kwargs, args
            changed += 1
    return changed


def _segment_devices(gm, layer_device):
    """The GPU each node's segment runs on under layer sharding, by the
    same rule the sharding applies after the split (engine._shard_layers):
    the host ops cut the graph into segments, a segment goes to the GPU of
    the highest layer its nodes come from (a layer's tail and the next
    layer's head share one segment and go with the later layer), and a
    segment with no layer (the head) goes where the one before it went.
    layer_device: layer -> torch.device or None. {} without sharding."""
    if layer_device is None:
        return {}
    devices, segment, layers, prev = {}, [], set(), None

    def flush():
        nonlocal prev
        dev = layer_device(max(layers)) if layers else prev
        for n in segment:
            devices[n] = dev
        if dev is not None:
            prev = dev

    for n in gm.graph.nodes:
        if n.op == "call_function" and str(n.target) in HOST_OPS:
            flush()
            segment, layers = [], set()
            continue
        segment.append(n)
        _, layer = _site(n)
        if layer is not None:
            layers.add(layer)
    flush()
    return devices


_keyed_cache_dir = None


def kernel_digest():
    """A digest of the Triton kernels' source and of the knobs in effect (plan.KNOBS: the table's
    and the A/B switches' values): what the compiled packages and inductor's caches are keyed by,
    so a changed kernel, or a knob that changes what a kernel body launches, is never served from
    an older compilation. The packages embed the launches a body makes (a package built with the
    MXFP8 recoding as a separate kernel was served to a process that had turned the prologue on:
    the same source, the same graph, a different step)."""
    from . import plan
    knobs = repr(sorted(plan.effective_knobs().items()))
    blocks = repr(sorted((k, sorted(v.items())) for k, v in triton_ops.GEMV_BLOCKS.items()))  # the table's blocks per class
    return hashlib.sha256((inspect.getsource(triton_ops) + knobs + blocks).encode()).hexdigest()[:16]


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
        digest = kernel_digest()
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
        _keyed_cache_dir = os.path.join(os.path.abspath(base), f"vxpu-kernels-{digest}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = _keyed_cache_dir
    return _keyed_cache_dir


# The shared expert joined back into one op (shared_ffn) when its front is not an island: "1" turns it on,
# off by default. Measured on four RTX PRO 6000 (hero 10, Nov 1): 40 launches fewer a step (1,082 -> 1,042 at
# one row) and the step unchanged at one row (13.3 ms) and nine (22.4 / 18.2 / 35.5 one prompt / same / different),
# because the joined GEMV's hyper-connection epilogue costs what the combine kernel did (the w2 GEMV 10.6 ->
# 12.9 us against the combine's 1.0); and the tokens are not the pair's: the epilogue sums the mix in another
# order than the compiler's kernel, a rounding-level difference that moved the greedy choice at token 26 of
# 64 at one row and in eight of nine rows at nine. Off until the epilogue and the combine agree bit for bit.
SHARED_FFN_JOIN = os.environ.get("VXPU_SHARED_FFN_JOIN", "0") == "1"

_FRONT, _COMBINE, _JOINED = "vxpu.shared_ffn_front.default", "vxpu.shared_ffn_combine.default", "vxpu.shared_ffn.default"


def join_shared_ffn(gm):
    """Rewrite each shared_ffn_front -> shared_ffn_combine pair into one shared_ffn call: the
    exported graph calls the shared expert in two ops so that the step graph can run the front
    beside the routed experts (segments.ISLAND_OPS); when the front is not an island the two run
    one after the other in one segment, and the pair costs a kernel the joined op does not have.
    The combine is elementwise work the compiler gives a kernel of its own (the add of the
    experts' sum, the rounding, the hyper-connection mix: 0.9 us at one row, 40 launches a step),
    where shared_ffn's Triton body does the same in its last GEMV's epilogue (HAS_ADD, EPI_HC).
    Joins a pair only when the front's one user is the combine. The new node keeps the combine's
    recorded value and the front's call site (the site the kernel table names). Returns how many
    pairs were joined."""
    joined = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != _COMBINE or node.kwargs:
            continue
        front = node.args[0]
        if not (isinstance(front, torch.fx.Node) and front.op == "call_function" and str(front.target) == _FRONT
                and len(front.users) == 1 and not front.kwargs):
            continue
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(torch.ops.vxpu.shared_ffn.default, tuple(front.args) + tuple(node.args[1:]))
        new.meta.update(node.meta)
        for key in ("nn_module_stack", "stack_trace", "source_fn_stack"):
            if key in front.meta:
                new.meta[key] = front.meta[key]
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        gm.graph.erase_node(front)
        joined += 1
    if joined:
        gm.graph.lint()
        gm.recompile()
    return joined


# The attention's two front projections as one op (hero run 15): attn_q and window_kv of a layer read the same
# x; joined, their down-projections are one GEMV launch (VXPU_QKV_JOIN=1 turns it on; an A/B until its ladder is
# on record with the tokens checked, since the joined GEMV's K split may differ from the narrow one's).
QKV_JOIN = os.environ.get("VXPU_QKV_JOIN", "0") == "1"
_ATTN_Q, _WINDOW_KV = "vxpu.attn_q.default", "vxpu.window_kv.default"


def join_qkv(gm):
    """Rewrite each attn_q + window_kv pair that reads the same x (and the same `quantized`) into one
    attn_qkv call placed where the earlier of the two stood (its arguments are x and the graph's
    inputs, so both are ready there); attn_q's getitem users move to the new node, window_kv's
    users take its third output. Returns how many pairs were joined."""
    import operator

    def source(n):  # the tensor behind alias nodes (an earlier pass wraps an op's input in aten.alias)
        while isinstance(n, torch.fx.Node) and n.op == "call_function" and str(n.target) in ("aten.alias.default", "aten.detach.default"):
            n = n.args[0]
        return n

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    qs_by_src = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and str(n.target) == _ATTN_Q and not n.kwargs:
            qs_by_src.setdefault(source(n.args[0]), []).append(n)
    joined = 0
    for kv in [n for n in list(gm.graph.nodes) if n.op == "call_function" and str(n.target) == _WINDOW_KV and not n.kwargs]:
        x = source(kv.args[0])
        if not isinstance(x, torch.fx.Node):
            continue
        q = next((u for u in qs_by_src.get(x, []) if u.args[13] == kv.args[9]), None)
        if q is None or any(not (u.op == "call_function" and u.target is operator.getitem) for u in q.users):
            continue
        qs_by_src[x].remove(q)
        first = q if order[q] < order[kv] else kv
        with gm.graph.inserting_before(first):  # the earlier op's own x: defined there, the later op's alias is not
            # window_kv's own `quantized` (its last argument) is attn_q's, which the match required: dropped
            new = gm.graph.call_function(torch.ops.vxpu.attn_qkv.default, (first.args[0],) + tuple(q.args[1:]) + tuple(kv.args[1:-1]))
            kv_out = gm.graph.call_function(operator.getitem, (new, 2))
        new.meta.update(q.meta)
        qval, kval = q.meta.get("val"), kv.meta.get("val")
        if qval is not None and kval is not None:
            new.meta["val"] = tuple(qval) + (kval,)
        kv_out.meta.update(kv.meta)
        for u in list(q.users):
            u.args = (new, u.args[1])
        kv.replace_all_uses_with(kv_out)
        gm.graph.erase_node(kv)
        gm.graph.erase_node(q)
        joined += 1
    if joined:
        gm.graph.lint()
        gm.recompile()
    return joined


_TP_SUM_HC = "vxpu.tp_allreduce_hc.default"
_ATTN_PARTIAL = "vxpu.attn_out_partial.default"
_PASS_THROUGH = ("aten.view.default", "aten.reshape.default", "aten.to.dtype", "aten._to_copy.default")


def _producer(node):
    """The node a mixing sum's partial comes from, looking back through the exported views and
    the cast to fp32 (each with that one user), with the nodes passed on the way; None where
    anything else intervenes."""
    chain = []
    while (isinstance(node, torch.fx.Node) and node.op == "call_function" and str(node.target) in _PASS_THROUGH
           and len(node.users) == 1):
        target = str(node.target)
        if target == "aten.to.dtype":
            if node.kwargs or node.args[1] is not torch.float32:
                return None, chain
        elif target == "aten._to_copy.default":
            if dict(node.kwargs) != {"dtype": torch.float32}:
                return None, chain
        elif node.kwargs:
            return None, chain
        chain.append(node)
        node = node.args[0]
    return (node if isinstance(node, torch.fx.Node) and node.op == "call_function" else None), chain


def drop_partial_casts(gm):
    """Drop the cast to fp32 between a bf16 partial and its mixing sum (hero run 15's many-row
    account): the attention block's partial is attn_out_partial's bf16 output cast to fp32 for the
    ranks' sum, so every push carried twice the bytes of the values it moved (131 us a site at 80
    rows, the same as the FFN sites' fp32 partials). The sum kernel converts each partial to fp32
    as it loads it and adds the ranks in order, so the totals are bit-identical with the cast gone.
    Only a cast with that one user, feeding the sum through views, goes; the FFN sites' fp32
    partials (the routed sum plus the shared expert's) are not casts and stay. Returns how many
    sums now take a bf16 partial."""
    dropped = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != _TP_SUM_HC or node.kwargs:
            continue
        _, chain = _producer(node.args[0])
        casts = [n for n in chain if str(n.target) in ("aten.to.dtype", "aten._to_copy.default")]
        if len(casts) != 1:
            continue
        cast = casts[0]
        src = cast.args[0]
        val = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
        if val is None or val.dtype != torch.bfloat16:
            continue
        cast.replace_all_uses_with(src)
        gm.graph.erase_node(cast)
        for n in chain:  # the views between the cast and the sum now carry bf16
            if n is not cast and "val" in n.meta:
                n.meta["val"] = n.meta["val"].to(torch.bfloat16)
        dropped += 1
    if dropped:
        gm.graph.lint()
        gm.recompile()
    return dropped


def fuse_pushes(gm, shared=True):
    """Rewrite each mixing sum whose partial a GEMV produces so that the GEMV runs inside the
    collective (the fused push, hero run 15): vxpu.tp_allreduce_hc(float(attn_out_partial(..)))
    becomes tp_gemv_allreduce_hc(attn_out_mid(..), wo_b, sb, None, ..), and
    tp_allreduce_hc(view(y_experts + view(float(shared_ffn_front(..))))) becomes
    tp_gemv_allreduce_hc(shared_ffn_mid(..), w2, s2, y_experts, ..). The step graph then runs the
    GEMV with the peers' inboxes as its epilogue's destinations and the sum kernel only waits and
    mixes (allreduce.gemv_allreduce_hc); the numbers are the same, added in the same order. A
    site whose partial comes any other way is left as it is. The new nodes keep the producer's
    call site (the kernel table's name) and the sum's recorded value. Returns how many sites were
    rewritten. `shared` False leaves the shared expert's sites as they are (the front stays one op,
    shared_ffn_front, which the step graph can run whole on the parallel branch beside the routed
    experts; the sum then takes the one-shot push of y_experts + float(front)): the A/B of the
    fused push against the hidden GEMV (hero run 15)."""
    rewritten = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != _TP_SUM_HC or node.kwargs:
            continue
        x, residual, post, comb = node.args
        base, chain = _producer(x)
        if base is None:
            continue
        mid_target = mid_args = weight = scale = add = None
        if str(base.target) == _ATTN_PARTIAL and len(base.users) == 1 and not base.kwargs:
            o, freqs, start, seqlen, wo_a, sa, groups, rank, wo_b, sb, rope_dim, prepared = base.args
            mid_target, mid_args = torch.ops.vxpu.attn_out_mid.default, (o, freqs, start, seqlen, wo_a, sa, groups,
                                                                          rank, rope_dim, prepared)
            weight, scale = wo_b, sb
            dead = chain + [base]
        elif shared and str(base.target) == "aten.add.Tensor" and len(base.users) == 1 and not base.kwargs:
            front = front_chain = other = None
            for side, rest in ((base.args[0], base.args[1]), (base.args[1], base.args[0])):
                cand, cchain = _producer(side)
                if (cand is not None and str(cand.target) == _FRONT and len(cand.users) == 1
                        and not cand.kwargs and isinstance(rest, torch.fx.Node)):
                    front, front_chain, other = cand, cchain, rest
                    break
            if front is None:
                continue
            x_in, w1, s1, w3, s3, w2, s2, limit, quantized = front.args
            val = other.meta.get("val")
            if val is None or val.dtype != torch.float32:
                continue
            mid_target, mid_args = torch.ops.vxpu.shared_ffn_mid.default, (x_in, w1, s1, w3, s3, limit, quantized)
            weight, scale, add = w2, s2, other
            dead = chain + [base] + front_chain + [front]
        else:
            continue
        fake_mode = getattr(node.meta.get("val"), "fake_mode", None)
        with gm.graph.inserting_before(node):
            mid = gm.graph.call_function(mid_target, mid_args)
            new = gm.graph.call_function(torch.ops.vxpu.tp_gemv_allreduce_hc.default,
                                         (mid, weight, scale, add, residual, post, comb))
        src = dead[-1]  # the producer: its call site names the kernel table's entry
        for key in ("nn_module_stack", "stack_trace", "source_fn_stack"):
            if key in src.meta:
                mid.meta[key] = src.meta[key]
        new.meta.update(node.meta)
        if fake_mode is not None:
            with fake_mode:
                mid.meta["val"] = mid_target(*_value(gm, mid_args))
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        for n in dead:
            if not n.users:
                gm.graph.erase_node(n)
        rewritten += 1
    if rewritten:
        gm.graph.lint()
        gm.recompile()
    return rewritten


def lower(gm, plan, layer_device=None):
    """Rewrite gm's vxpu op calls into their selected implementations,
    where lowerable. The engine lowers the whole decode graph before
    splitting it (segments.py): the host ops are left where they are, so
    the cut is the same. Each call site's choice goes into the plan's
    kernel record (Plan.record_site): the site, its shapes, the candidates,
    what was measured, what was chosen and why (the table, a measurement,
    the registry's order), the kernels inlined for it, and the device it
    was traced for. With a kernel table (VXPU_PLAN_TABLE) the table's body
    is inlined and nothing is measured.

    layer_device (layer -> torch.device, the engine's layer sharding):
    each site's body is traced on the GPU its segment will run on
    (_segment_devices), not on the GPU the lowering runs on. A body traced
    on the wrong GPU bakes that GPU's index into what it derives from its
    inputs' device (an arange on start.device, type_as's copy), and the
    segment then dies on another GPU with "Expected all tensors to be on
    the same device": the 9- and 33-row steps did, in the segment spanning
    a range boundary, whenever the lowering picked a fused body there.

    Returns {op: {"lowered": {impl: n}, "kept": {impl: n}}} for the calls
    seen; gm is recompiled if anything changed."""
    key_cache_dir()
    report = collections.defaultdict(lambda: {"lowered": collections.Counter(), "kept": collections.Counter(),
                                              "measured": {}})
    if SHARED_FFN_JOIN and _FRONT not in segments.ISLAND_OPS:
        joined = join_shared_ffn(gm)
        if joined:
            report["vxpu::shared_ffn_front"]["joined"] = joined
    if QKV_JOIN:
        joined = join_qkv(gm)
        seen = sum(1 for n in gm.graph.nodes if n.op == "call_function" and str(n.target) in (_ATTN_Q, _WINDOW_KV, "vxpu.attn_qkv.default"))
        print(f"[vxpu] qkv join: {joined} attn_q + window_kv pairs joined ({seen} such calls in the graph after it)", flush=True)
        if joined:
            report["vxpu::attn_q"]["joined"] = joined
    devices = _segment_devices(gm, layer_device)
    scratch = _Scratch()
    for node in list(gm.graph.nodes):
        target = str(node.target)
        if node.op != "call_function" or not target.startswith("vxpu.") or target in HOST_OPS:
            continue
        op = "vxpu::" + target.split(".")[1]
        site, layer = _site(node)
        if op not in ops.SELECTION or node.kwargs:
            report[op]["kept"]["(not lowerable)"] += 1
            plan.record_site(op=op, site=site, node=node.name, layer=layer, chosen=None, source="kept",
                             why="not lowerable", bytes=site_bytes(gm, node))
            continue
        try:
            args = _value(gm, node.args)
        except _NoShapes:
            report[op]["kept"]["(no shapes)"] += 1
            plan.record_site(op=op, site=site, node=node.name, layer=layer, chosen=None, source="kept",
                             why="no shapes")
            continue
        device_type, key, ctx = ops.SELECTION[op](*args)
        available = plan.candidates(op, device_type)
        candidates = [i.name for i in available]
        fused = [i.name for i in available if i.fused]  # bodies the compiler fuses: credited a launch (costs.py)
        times = why = None
        impl = plan.table_choice(op, device_type, key, site=site, lowering=True, **ctx)
        if impl is not None:
            source = "table"
        else:
            impl = ops.select(op, *args, plan=plan, lowering=True, site=site)
            source = "registry"
            if impl.fused:
                impl, times, why = _cheaper(op, impl, ops.select(op, *args, plan=plan, site=site), args, report)
                if times:
                    source = "measured"
        target = devices.get(node)
        entry = dict(op=op, site=site, node=node.name, shapes=key, layer=layer, chosen=impl.name, source=source,
                     candidates=candidates, fused=fused, measured_us=times or None, why=why,
                     device=str(target) if target is not None else None, bytes=site_bytes(gm, node))
        if not impl.lowerable:
            report[op]["kept"][impl.name] += 1
            plan.record_site(kernels=None, **entry)
            continue
        counters = None
        wanted = impl.counters(**ctx) if impl.counters is not None else 0
        if wanted:
            counters = (wanted, scratch.take(gm, node, wanted, target if target is not None else _device(args)))
        kernels = _inline(gm, node, _implementation(op, impl), args, device=target, counters=counters)
        report[op]["lowered"][impl.name] += 1
        plan.record_site(kernels=kernels, **entry)
    scratch.close()
    if "matmul" in plan["fuse"]:
        _fuse_matmuls(gm, report, devices)
    if any(r["lowered"] for r in report.values()):
        gm.graph.lint()
        gm.recompile()
    plan.write_record()
    return {op: {k: (dict(v) if isinstance(v, dict) else v) for k, v in r.items() if v} for op, r in report.items()}


def _device(values):
    return next(v.device for v in pytree.tree_leaves(values) if isinstance(v, torch.Tensor))


class _Scratch:
    """The segment's arrival counters (hero run 11): one zeros node per segment graph and GPU,
    sliced per site that declared it needs counters (Impl.counters). The zeros is an
    intermediate of the compiled segment, so every captured graph has its own, zeroed by one
    fill a step per segment; no kernel owns a global that two batches in flight could share."""

    def __init__(self):
        self.nodes = {}  # device -> (zeros node, count so far)

    def take(self, gm, before, n, device):
        key = str(device)
        if key not in self.nodes:
            first = next(nd for nd in gm.graph.nodes if nd.op != "placeholder")
            with gm.graph.inserting_before(first):
                zeros = gm.graph.call_function(torch.zeros, ([0],), {"dtype": torch.int32, "device": torch.device(device)})
            self.nodes[key] = [zeros, 0]
        zeros, off = self.nodes[key]
        self.nodes[key][1] = off + n
        with gm.graph.inserting_before(before):
            return gm.graph.call_function(torch.ops.aten.slice.Tensor, (zeros, 0, off, off + n))

    def close(self):
        for zeros, total in self.nodes.values():
            zeros.args = ([total],)


def _cheaper(op, fused, other, args, report):
    """fused (a decomposed body) or other (the implementation selected without it), by
    their measured cost at these shapes (costs.py), with the times measured ({} when a
    cached decision answered, None when nothing was measured). Only a lowerable other
    competes: one that is not runs through its op, as the plan chose it (a bf16 weight copy
    kept in the engine's cache, say), which a measurement on its own could not reproduce."""
    if other.name == fused.name:
        return other, None, None
    if not other.lowerable:
        # the reference computation (plain tensor operations) is better inlined as the body
        # than kept as a call the compiler cannot take; any other unlowerable choice stays
        return (fused if other.reference else other), None, None
    chosen, times, why = costs.choose(op, {fused.name: (_implementation(op, fused), True),
                                           other.name: (_implementation(op, other), False)}, args, _device(args))
    if times:
        report[op]["measured"][" ".join(costs.shapes(args))] = times
    return (fused if chosen == fused.name else other), times, why


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


def _fuse_matmuls(gm, report, devices=None):
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
        chosen, times, _ = costs.choose(op, {"reduction": (body, True), "aten": (lambda *a: aten(*a), False)},
                                        values, _device(values))
        if times:
            report[op]["measured"][" ".join(costs.shapes(values))] = times
        if chosen == "aten":
            report[op]["kept"]["aten"] += 1
            continue
        _inline(gm, node, body, values, device=(devices or {}).get(node))
        report[op]["lowered"]["reduction"] += 1


REPORT_PREFIX = "lowered the dense segments' op calls: "


def describe(report):
    """One line for logs and plan notes."""
    parts = []
    for op, r in report.items():
        done = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("lowered", {}).items()))
        kept = ", ".join(f"{n} {impl}" for impl, n in sorted(r.get("kept", {}).items()))
        joined = r.get("joined") and (f"{r['joined']} joined with window_kv into attn_qkv" if op == "vxpu::attn_q" else
                                      f"{r['joined']} joined with shared_ffn_combine into shared_ffn")
        parts.append(f"{op.split('::')[1]}: " + "; ".join(x for x in (done and f"lowered {done}", kept and f"kept {kept}", joined) if x))
    return REPORT_PREFIX + (" | ".join(parts) or "none")


def describe_measured(report):
    """The measured costs behind the choices (costs.py), one line, or None."""
    parts = [f"{op.split('::')[1]} {shape}: " + ", ".join(f"{name} {us} us" for name, us in sorted(times.items()))
             for op, r in report.items() for shape, times in r.get("measured", {}).items()]
    return "lowering costs, measured per call shape: " + " | ".join(parts) if parts else None
