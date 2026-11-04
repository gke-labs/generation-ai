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

"""The lowering under layer sharding: a fused body traced for a segment on
another GPU names that GPU, not the one the lowering runs on.

    python -m unittest vxpu.lower_test
"""

import unittest

import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from . import costs, lower


def _stack(path):
    return {"model": ("model", "Model"), path: (path, "Module")}


class SegmentDevices(unittest.TestCase):
    """_segment_devices mirrors the sharding's rule for the cut: a segment
    goes to the GPU of the highest layer its nodes come from, the head
    goes where the segment before it went."""

    def test_rule(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        a7 = g.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
        a7.meta["nn_module_stack"] = _stack("model.layers.7.attn")
        host7 = g.call_function(torch.ops.vxpu.moe_experts.default, (a7, a7, a7, 7))  # the experts host op
        tail7 = g.call_function(torch.ops.aten.mul.Tensor, (host7, 3.0))
        tail7.meta["nn_module_stack"] = _stack("model.layers.7.ffn.shared")
        head8 = g.call_function(torch.ops.aten.mul.Tensor, (tail7, 4.0))
        head8.meta["nn_module_stack"] = _stack("model.layers.8.attn")
        host8 = g.call_function(torch.ops.vxpu.moe_experts.default, (head8, head8, head8, 8))
        head = g.call_function(torch.ops.aten.mul.Tensor, (host8, 5.0))  # the model's head: no layer
        g.output(head)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        def layer_device(layer):
            return torch.device("cuda", 0 if layer < 8 else 1)

        devices = lower._segment_devices(gm, layer_device)
        self.assertEqual(devices[a7], torch.device("cuda", 0))
        self.assertEqual(devices[tail7], torch.device("cuda", 1), "a layer's tail goes with the next layer's head")
        self.assertEqual(devices[head8], torch.device("cuda", 1))
        self.assertEqual(devices[head], torch.device("cuda", 1), "the head goes where the segment before it went")
        self.assertEqual(lower._segment_devices(gm, None), {})


class Retarget(unittest.TestCase):
    def test_literals(self):
        g = torch.fx.Graph()
        n = g.call_function(torch.ops.aten.arange.default, (4,), {"device": torch.device("cuda", 0), "dtype": torch.int64})
        m = g.call_function(torch.ops.aten._to_copy.default, (n,), {"device": "cuda:0", "dtype": torch.float32})
        k = g.call_function(torch.ops.aten.zeros.default, ([3],), {"device": torch.device("cpu")})
        g.output((m, k))
        self.assertEqual(lower._retarget(g, torch.device("cuda", 1)), 2)
        self.assertEqual(n.kwargs["device"], torch.device("cuda", 1))
        self.assertEqual(m.kwargs["device"], torch.device("cuda", 1))
        self.assertEqual(k.kwargs["device"], torch.device("cpu"), "a CPU literal is not a GPU index")


@unittest.skipUnless(torch.cuda.is_available(), "fake CUDA tensors need a CUDA build")
class InlineOnDevice(unittest.TestCase):
    """A body that derives devices from its inputs (an arange on start.device,
    type_as, a zeros on x.device), traced for a segment on cuda:1 while the
    values sit on cuda:0: every device literal in the inlined graph names
    cuda:1."""

    def test_inline(self):
        def body(s, x):
            positions = s.reshape(-1, 1) + torch.arange(4, device=s.device)
            y = (positions.float() * 2).type_as(x)
            return y, torch.zeros(3, device=x.device) + y.sum()

        with FakeTensorMode():
            s = torch.empty(2, dtype=torch.int64, device="cuda:0")
            x = torch.empty(2, 4, dtype=torch.bfloat16, device="cuda:0")
        gm = make_fx(lambda a, b: (a.reshape(-1, 1) + b.sum().to(torch.int64), b * 1), tracing_mode="fake")(s, x)
        node = next(n for n in gm.graph.nodes if n.op == "call_function" and "mul" in str(n.target))
        lower._inline(gm, node, body, (s, x), device=torch.device("cuda", 1))
        literals = [v for n in gm.graph.nodes for v in list(n.args) + list(n.kwargs.values())
                    if isinstance(v, (torch.device, str)) and str(v).startswith("cuda")]
        self.assertTrue(literals, "the body's factories and casts carry device literals")
        self.assertEqual({str(v) for v in literals}, {"cuda:1"})


class JoinSharedFfn(unittest.TestCase):
    """join_shared_ffn: a shared_ffn_front -> shared_ffn_combine pair becomes one shared_ffn call
    with the front's inputs, the combine's inputs, the combine's value and the front's call site;
    a front with another reader stays."""

    def _graph(self, extra_reader=False):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        ws = [g.placeholder(n) for n in ("w1", "s1", "w3", "s3", "w2", "s2")]
        y = g.placeholder("y")
        residual, post, comb = (g.placeholder(n) for n in ("residual", "post", "comb"))
        front = g.call_function(torch.ops.vxpu.shared_ffn_front.default, (x, *ws, 10.0, True))
        front.meta["nn_module_stack"] = _stack("model.layers.3.ffn.shared_experts")
        yf = g.call_function(torch.ops.aten._to_copy.default, (y,), {"dtype": torch.float32})
        combine = g.call_function(torch.ops.vxpu.shared_ffn_combine.default, (front, yf, residual, post, comb))
        combine.meta["val"] = "stream"
        combine.meta["nn_module_stack"] = _stack("model.layers.3")
        out = [combine]
        if extra_reader:
            out.append(g.call_function(torch.ops.aten.mul.Tensor, (front, 2.0)))
        g.output(tuple(out))
        return torch.fx.GraphModule(torch.nn.Module(), g), x, ws, yf, (residual, post, comb)

    def test_joined(self):
        gm, x, ws, yf, hc = self._graph()
        self.assertEqual(lower.join_shared_ffn(gm), 1)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertNotIn("vxpu.shared_ffn_front.default", targets)
        self.assertNotIn("vxpu.shared_ffn_combine.default", targets)
        node = next(n for n in gm.graph.nodes if str(n.target) == "vxpu.shared_ffn.default")
        self.assertEqual(node.args, (x, *ws, 10.0, True, yf, *hc))
        self.assertEqual(node.meta["val"], "stream", "the combine's recorded value")
        self.assertEqual(lower._site(node), ("model.layers.3.ffn.shared_experts", 3), "the front's call site")
        self.assertEqual([str(n.target) for n in gm.graph.nodes if n.op == "call_function"],
                         ["aten._to_copy.default", "vxpu.shared_ffn.default"])

    def test_report_carries_the_count(self):
        report = {"vxpu::shared_ffn_front": {"lowered": {}, "kept": {}, "measured": {}, "joined": 40}}
        self.assertIn("40 joined with shared_ffn_combine into shared_ffn", lower.describe(report))
        # lower() renders the report with the same comprehension: a count is not a mapping
        self.assertEqual({k: (dict(v) if isinstance(v, dict) else v) for k, v in report["vxpu::shared_ffn_front"].items() if v},
                         {"joined": 40})

    def test_front_with_another_reader_stays(self):
        gm, *_ = self._graph(extra_reader=True)
        self.assertEqual(lower.join_shared_ffn(gm), 0)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn("vxpu.shared_ffn_front.default", targets)
        self.assertIn("vxpu.shared_ffn_combine.default", targets)


class JoinQkv(unittest.TestCase):
    """join_qkv: an attn_q and a window_kv that read the same x become one attn_qkv call placed where
    the earlier stood; attn_q's getitem readers move to it and window_kv's readers take its third
    output; a pair whose `quantized` flags differ stays."""

    def _graph(self, same_quant=True, kv_first=False):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        qa = [g.placeholder(n) for n in ("wq_a", "sa", "qnw", "wq_b", "sb", "freqs", "start")]
        ka = [g.placeholder(n) for n in ("wkv", "skv", "kvnw", "kfreqs", "kstart")]
        q_args = (x, qa[0], qa[1], qa[2], 1e-6, qa[3], qa[4], qa[5], qa[6], 1, 2, 64, 32, True)
        kv_args = (x, ka[0], ka[1], ka[2], 1e-6, ka[3], ka[4], 1, 32, True if same_quant else False)
        if kv_first:
            kv = g.call_function(torch.ops.vxpu.window_kv.default, kv_args)
            q = g.call_function(torch.ops.vxpu.attn_q.default, q_args)
        else:
            q = g.call_function(torch.ops.vxpu.attn_q.default, q_args)
            kv = g.call_function(torch.ops.vxpu.window_kv.default, kv_args)
        import operator
        qn = g.call_function(operator.getitem, (q, 0))
        qq = g.call_function(operator.getitem, (q, 1))
        y = g.call_function(torch.ops.aten.add.Tensor, (qn, kv))
        g.output((qq, y))
        return torch.fx.GraphModule(torch.nn.Module(), g), q_args, kv_args

    def test_joined(self):
        for kv_first in (False, True):
            gm, q_args, kv_args = self._graph(kv_first=kv_first)
            self.assertEqual(lower.join_qkv(gm), 1)
            targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
            self.assertNotIn("vxpu.attn_q.default", targets)
            self.assertNotIn("vxpu.window_kv.default", targets)
            node = next(n for n in gm.graph.nodes if str(n.target) == "vxpu.attn_qkv.default")
            self.assertEqual(node.args, q_args + kv_args[1:])
            import operator
            items = [n for n in gm.graph.nodes if n.target is operator.getitem]
            self.assertEqual(sorted(n.args[1] for n in items), [0, 1, 2])
            self.assertTrue(all(n.args[0] is node for n in items))
            gm.graph.lint()

    def test_different_quantized_flags_stay(self):
        gm, *_ = self._graph(same_quant=False)
        self.assertEqual(lower.join_qkv(gm), 0)

    def test_report_line(self):
        report = {"vxpu::attn_q": {"lowered": {}, "kept": {}, "measured": {}, "joined": 40}}
        self.assertIn("40 joined with window_kv into attn_qkv", lower.describe(report))


class SiteBytes(unittest.TestCase):
    """site_bytes classifies what a call reads by the graph alone: the
    parameters (get_attr, or the export's p_ and c_ placeholders, also
    behind a view), the state buffers (b_), and the activations (every
    other input, and the outputs)."""

    def test_classification(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")  # an activation, 1x16 bf16: 32 bytes
        p = g.placeholder("p_model_w")  # a parameter, 16x8 fp8: 128 bytes
        b = g.placeholder("b_model_cache")  # a state buffer, 64x8 bf16: 1024 bytes
        pt = g.call_function(torch.ops.aten.permute.default, (p, [1, 0]))  # a view of the parameter
        out = g.call_function(torch.ops.aten.mm.default, (x, pt))
        g.call_function(torch.ops.aten.add.Tensor, (out, b))
        g.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        with FakeTensorMode() as fake:
            x.meta["val"] = fake.from_tensor(torch.zeros(1, 16, dtype=torch.bfloat16))
            p.meta["val"] = fake.from_tensor(torch.zeros(16, 8, dtype=torch.float8_e4m3fn))
            b.meta["val"] = fake.from_tensor(torch.zeros(64, 8, dtype=torch.bfloat16))
            pt.meta["val"] = p.meta["val"].permute(1, 0)
            out.meta["val"] = fake.from_tensor(torch.zeros(1, 8, dtype=torch.bfloat16))
        add = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.add.Tensor)
        self.assertEqual(lower.site_bytes(gm, out), {"weights": 128, "activations": 32 + 16, "state": 0})
        with FakeTensorMode() as fake:
            add.meta["val"] = fake.from_tensor(torch.zeros(64, 8, dtype=torch.bfloat16))
        self.assertEqual(lower.site_bytes(gm, add), {"weights": 0, "activations": 16 + 1024, "state": 1024})

    def test_get_attr_is_a_parameter(self):
        root = torch.nn.Module()
        root.register_buffer("w", torch.zeros(4, 4, dtype=torch.float32))
        g = torch.fx.Graph()
        x = g.placeholder("x")
        w = g.get_attr("w")
        out = g.call_function(torch.ops.aten.mm.default, (x, w))
        g.output(out)
        gm = torch.fx.GraphModule(root, g)
        with FakeTensorMode() as fake:
            x.meta["val"] = fake.from_tensor(torch.zeros(1, 4))
            out.meta["val"] = fake.from_tensor(torch.zeros(1, 4))
        self.assertEqual(lower.site_bytes(gm, out), {"weights": 64, "activations": 16 + 16, "state": 0})



class NoiseBandTest(unittest.TestCase):
    """A fused body displaces the kernel only when it wins by more than the measurement's noise."""

    def test_inside_the_band_keeps_the_kernel(self):
        # the attn_out pair hero 11 found flipping between runs
        chosen, why = costs.decide({"gpu-fused": 58.49, "gpu-triton": 57.60}, {"gpu-fused": 0.0}, "gpu-triton")
        self.assertEqual(chosen, "gpu-triton")
        self.assertIn("noise band", why)
        chosen, _ = costs.decide({"gpu-fused": 56.6, "gpu-triton": 57.60}, {"gpu-fused": 1.0}, "gpu-triton")
        self.assertEqual(chosen, "gpu-triton")  # 55.6 effective against 57.6 - 2.0: on the band, not under it

    def test_a_clear_win_takes_the_fused_body(self):
        chosen, why = costs.decide({"gpu-fused": 50.0, "gpu-triton": 57.60}, {"gpu-fused": 2.0}, "gpu-triton")
        self.assertEqual(chosen, "gpu-fused")
        self.assertIn("beats", why)
        # a small kernel: the band is the microseconds floor, not the share
        chosen, _ = costs.decide({"gpu-fused": 1.5, "gpu-triton": 3.0}, {"gpu-fused": 0.0}, "gpu-triton")
        self.assertEqual(chosen, "gpu-triton")
        chosen, _ = costs.decide({"gpu-fused": 0.9, "gpu-triton": 3.0}, {"gpu-fused": 0.0}, "gpu-triton")
        self.assertEqual(chosen, "gpu-fused")
class ScratchCounters(unittest.TestCase):
    """A segment's arrival counters (hero run 11): one zeros per segment graph
    and GPU, sized to every site's request at the end, each site inlined with
    its own slice as an extra input (Impl.counters); the body's result is
    computed from that input, and a CPU run of the inlined graph sees zeros."""

    def test_one_zeros_per_segment_sliced_per_site(self):
        def body(x, counters=None):
            return x * 2 + counters.sum().to(x.dtype)

        with FakeTensorMode():
            x = torch.empty(2, 4, dtype=torch.float32, device="cpu")
        gm = make_fx(lambda a: ((a * 1) + 1, (a * 1) + 2), tracing_mode="fake")(x)
        muls = [n for n in gm.graph.nodes if n.op == "call_function" and "mul" in str(n.target)]
        self.assertEqual(len(muls), 2)
        scratch = lower._Scratch()
        for node, n in zip(muls, (1, 3)):
            counters = (n, scratch.take(gm, node, n, torch.device("cpu")))
            lower._inline(gm, node, body, (x,), counters=counters)
        scratch.close()
        zeros = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.zeros]
        self.assertEqual(len(zeros), 1, "one zeros per segment and device")
        self.assertEqual(zeros[0].args, ([4],))
        self.assertEqual(zeros[0].kwargs["dtype"], torch.int32)
        slices = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.ops.aten.slice.Tensor]
        self.assertEqual([(n.args[2], n.args[3]) for n in slices], [(0, 1), (1, 4)])
        gm.graph.lint()
        gm.recompile()
        a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        out = gm(a)
        torch.testing.assert_close(out[0], a * 2 + 1)
        torch.testing.assert_close(out[1], a * 2 + 2)


if __name__ == "__main__":
    unittest.main()


class DropPartialCasts(unittest.TestCase):
    """drop_partial_casts: a mixing sum fed by the fp32 cast of a bf16 partial, through views,
    takes the bf16 partial (hero run 15's many-row account); a cast with another reader, a cast
    of an fp32 value and the FFN site's add of a cast stay."""

    def _graph(self, src_dtype=torch.bfloat16, extra_reader=False, ffn=False):
        g = torch.fx.Graph()
        o = g.placeholder("o")
        residual, post, comb = (g.placeholder(n) for n in ("residual", "post", "comb"))
        part = g.call_function(torch.ops.aten.mul.Tensor, (o, 2.0))
        part.meta["val"] = torch.empty(1, 1, 8, dtype=src_dtype, device="meta")
        cast = g.call_function(torch.ops.aten._to_copy.default, (part,), {"dtype": torch.float32})
        cast.meta["val"] = torch.empty(1, 1, 8, dtype=torch.float32, device="meta")
        x = g.call_function(torch.ops.aten.view.default, (cast, [1, 1, 8]))
        x.meta["val"] = cast.meta["val"]
        if ffn:
            y = g.placeholder("y")
            x = g.call_function(torch.ops.aten.add.Tensor, (y, x))
        total = g.call_function(torch.ops.vxpu.tp_allreduce_hc.default, (x, residual, post, comb))
        out = [total]
        if extra_reader:
            out.append(g.call_function(torch.ops.aten.mul.Tensor, (cast, 2.0)))
        g.output(tuple(out))
        return torch.fx.GraphModule(torch.nn.Module(), g), part

    def test_the_cast_goes_and_the_sum_takes_the_bf16_partial(self):
        gm, part = self._graph()
        self.assertEqual(lower.drop_partial_casts(gm), 1)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertNotIn("aten._to_copy.default", targets)
        total = next(n for n in gm.graph.nodes if str(n.target) == "vxpu.tp_allreduce_hc.default")
        view = total.args[0]
        self.assertEqual(str(view.target), "aten.view.default")
        self.assertIs(view.args[0], part)
        self.assertEqual(view.meta["val"].dtype, torch.bfloat16)

    def test_casts_that_must_stay(self):
        for kwargs in ({"extra_reader": True}, {"src_dtype": torch.float32}, {"ffn": True}):
            gm, _ = self._graph(**kwargs)
            self.assertEqual(lower.drop_partial_casts(gm), 0, kwargs)
            targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
            self.assertIn("aten._to_copy.default", targets, kwargs)
