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

"""Tests for stepgraph in-graph stamps and timeline accounting."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import dataflow, stepgraph, triton_ops


class PhaseDecodeTest(unittest.TestCase):
    """The decode of a hand-filled buffer shaped [PHASE_STEPS, nodes, 2] returns steps
    oldest first and skips empty slots."""

    def test_decode_wrapped_indices(self):
        nodes = [
            ("n0", "dense", [0], ["dense"]),
            ("n1", "experts", [0], ["experts"]),
        ]
        buf = torch.zeros(stepgraph.PHASE_STEPS, len(nodes), 2, dtype=torch.int64)

        # Write three steps at wrapped indices for k=10 (step 7 -> slot 7, step 8 -> slot 0, step 9 -> slot 1).
        # Slots 2, 3, 4, 5, 6 remain 0.
        buf[7, 0, :] = torch.tensor([700, 750])
        buf[7, 1, :] = torch.tensor([750, 800])

        buf[0, 0, :] = torch.tensor([800, 850])
        buf[0, 1, :] = torch.tensor([850, 900])

        buf[1, 0, :] = torch.tensor([900, 950])
        buf[1, 1, :] = torch.tensor([950, 1000])

        table = stepgraph.decode_phase_table(buf, nodes, k=10)
        self.assertEqual(len(table), 3)

        # Oldest first: step 7 (slot 7), then step 8 (slot 0), then step 9 (slot 1)
        step7 = table[0]
        self.assertEqual(step7[0], ("n0", "dense", [0], ["dense"], 700, 750))
        self.assertEqual(step7[1], ("n1", "experts", [0], ["experts"], 750, 800))

        step8 = table[1]
        self.assertEqual(step8[0], ("n0", "dense", [0], ["dense"], 800, 850))
        self.assertEqual(step8[1], ("n1", "experts", [0], ["experts"], 850, 900))

        step9 = table[2]
        self.assertEqual(step9[0], ("n0", "dense", [0], ["dense"], 900, 950))
        self.assertEqual(step9[1], ("n1", "experts", [0], ["experts"], 950, 1000))


class PhaseAccountTest(unittest.TestCase):
    """The account per class on a hand-made table: two dense nodes and one host node
    whose span contains a dense island."""

    def test_account_hand_made_table(self):
        # Two dense nodes (attn_q, attn_out) and one host node (moe) containing a dense island (shared_ffn).
        table = [
            ("attn_q", "dense", [0], ["attention"], 100, 200),
            ("moe", "experts", [0], ["experts"], 300, 800),
            ("shared_ffn", "dense", [0], ["shared_expert"], 400, 600),
            ("attn_out", "dense", [0], ["attention"], 900, 1000),
        ]

        acc = stepgraph.phase_account(table)

        # Wall: first start (100) to last end (1000)
        self.assertEqual(acc["wall"], 900)

        # Covered: union of [100, 200], [300, 800] enclosing [400, 600], and [900, 1000] = 100 + 500 + 100 = 700
        self.assertEqual(acc["covered"], 700)

        # Gaps: wall (900) - covered (700) = 200
        self.assertEqual(acc["gaps"], 200)

        # Per-class sums: host node span and dense spans inside it are reported as the design's columns do
        self.assertEqual(acc["classes"]["attention"], 200)
        self.assertEqual(acc["classes"]["experts"], 500)
        self.assertEqual(acc["classes"]["shared_expert"], 200)


class NodeOpsTest(unittest.TestCase):
    """_node_ops returns vxpu ops for dense segments and kind for host nodes."""

    def test_host_node(self):
        node = dataflow.Node("experts_0", "experts", [0], [], {})
        self.assertEqual(stepgraph._node_ops(None, node), ["experts"])

    def test_synthetic_dense_submodules(self):
        from . import ops as _  # register torch.ops.vxpu

        class SubA(torch.nn.Module):
            def forward(self, x):
                return x + 1

        sub_a = torch.fx.symbolic_trace(SubA())

        # Construct fx GraphModule with a vxpu custom op call
        g = torch.fx.Graph()
        x = g.placeholder("x")
        out = g.call_function(torch.ops.vxpu.fp8_linear.default, (x, x, x))
        g.output(out)
        sub_b = torch.fx.GraphModule(torch.nn.Module(), g)

        split = torch.nn.Module()
        split.submod_0 = sub_a
        split.submod_1 = sub_b

        node0 = dataflow.Node("submod_0", "dense", [0], [], {})
        node1 = dataflow.Node("submod_1", "dense", [0], [], {})

        self.assertEqual(stepgraph._node_ops(split, node0), ["dense"])
        self.assertEqual(stepgraph._node_ops(split, node1), ["fp8_linear"])

    def test_node_ops_tiny_model_cpu_split(self):
        from .engine import Engine
        from .export import export_artifact
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-llama")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)
            saved = {k: os.environ.get(k) for k in ("VXPU_DATAFLOW", "VXPU_SPLIT_DECODE")}
            os.environ["VXPU_DATAFLOW"] = "1"
            os.environ["VXPU_SPLIT_DECODE"] = "1"
            try:
                eng = Engine(artifact, compile_decode=True)
                runner = eng._decode_split
                self.assertIsNotNone(runner)
                for node in runner.graph.nodes:
                    ops = stepgraph._node_ops(runner.split, node)
                    if node.kind == "dense":
                        self.assertTrue(isinstance(ops, list) and len(ops) > 0)
                    else:
                        self.assertEqual(ops, [node.kind])
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


@unittest.skipUnless(torch.cuda.is_available() and triton_ops.available(), "needs CUDA and Triton")
class PhaseTraceGPUTest(unittest.TestCase):
    """The tiny model's step captured with VXPU_PHASE_TRACE=1 on a GPU."""

    def test_gpu_phase_trace(self):
        from .engine import Engine, SamplingParams
        from .export import export_artifact
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-llama")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)
            aoti_dir = os.path.join(td, "aoti")
            env = {
                "VXPU_DATAFLOW": "1",
                "VXPU_SEGMENT_RUNTIME": "aoti",
                "VXPU_AOTI_CACHE_DIR": aoti_dir,
                "VXPU_STEP_GRAPH": "1",
            }
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                steps = 5
                # Reference run without in-graph stamps
                with patch.object(stepgraph, "PHASE_TRACE", False):
                    ref_eng = Engine(artifact, device="cuda", compile_decode=True)
                    ref_sid = ref_eng.new_session()
                    ref_events = list(ref_eng.generate(ref_sid, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                    ref_tokens = [ev["token"] for ev in ref_events if not ev.get("done")]

                # Traced run with in-graph stamps enabled
                with patch.object(stepgraph, "PHASE_TRACE", True):
                    eng = Engine(artifact, device="cuda", compile_decode=True)
                    sg = eng._decode_split
                    self.assertIsInstance(sg, stepgraph.StepGraph)
                    sid = eng.new_session()
                    events = list(eng.generate(sid, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                    tokens = [ev["token"] for ev in events if not ev.get("done")]

                self.assertEqual(len(tokens), steps)
                self.assertEqual(tokens, ref_tokens)

                table = sg.phase_table()
                self.assertTrue(len(table) > 0)
                # The first decode step runs eagerly; subsequent decode steps replay the captured graph.
                self.assertEqual(sg.phase_replays, steps - 1)

                for step_idx, step_rows in enumerate(table):
                    for name, kind, layers, ops, start, end in step_rows:
                        self.assertLess(start, end, f"node {name} in step {step_idx} must have start < end")
                    starts = [r[4] for r in step_rows]
                    self.assertEqual(starts, sorted(starts))
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
