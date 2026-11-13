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
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from google.protobuf import json_format
import torch

from . import dataflow, stepgraph, stepgraph_pb2, triton_ops
from .plan import ConfigError, Plan


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


class SelectPhaseRowsTest(unittest.TestCase):
    """select_phase_rows keeps the newest rows corresponding to the recorded call."""

    def test_selection_cases(self):
        old_rows = [f"old_{i}" for i in range(8)]

        # Eight old rows and two new ones give the two
        ring_after_2 = [f"old_{i}" for i in range(2, 8)] + ["new_0", "new_1"]
        self.assertEqual(
            stepgraph.select_phase_rows(8, 10, ring_after_2),
            ["new_0", "new_1"],
        )

        # Ten new ones give the newest eight
        ten_new_ring = [f"new_{i}" for i in range(2, 10)]
        self.assertEqual(
            stepgraph.select_phase_rows(0, 10, ten_new_ring),
            ten_new_ring,
        )

        # Zero new ones give none
        self.assertEqual(
            stepgraph.select_phase_rows(8, 8, old_rows),
            [],
        )

        # A counter that wrapped the ring more than once still gives the newest eight
        wrapped_ring = [f"step_{i}" for i in range(17, 25)]
        self.assertEqual(
            stepgraph.select_phase_rows(0, 25, wrapped_ring),
            wrapped_ring,
        )


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
                    ref_tokens = [ev.get("token", ev["token_ids"][0]) for ev in ref_events if not ev.get("done") and (ev.get("token") or ev.get("token_ids"))]

                # Traced run with in-graph stamps enabled
                with patch.object(stepgraph, "PHASE_TRACE", True):
                    eng = Engine(artifact, device="cuda", compile_decode=True)
                    sg = eng._decode_split
                    self.assertIsInstance(sg, stepgraph.StepGraph)
                    sid = eng.new_session()
                    events = list(eng.generate(sid, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                    tokens = [ev.get("token", ev["token_ids"][0]) for ev in events if not ev.get("done") and (ev.get("token") or ev.get("token_ids"))]

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


def run_vxpu_plan_step(graph_file: str, out_file: str, serialize: bool = False) -> None:
    """Executes vxpu plan step from graph_file to out_file."""
    cmd = ["vxpu", "plan", "step", "-graph", graph_file, "-out", out_file]
    if serialize:
        cmd.append("-serialize")
    if shutil.which("vxpu"):
        subprocess.run(cmd, check=True)
    else:
        vxpu_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        go_cmd = ["go", "run", "./cmd/vxpu", "plan", "step", "-graph", graph_file, "-out", out_file]
        if serialize:
            go_cmd.append("-serialize")
        subprocess.run(go_cmd, cwd=vxpu_dir, check=True)


class StepPlanOrderingTest(unittest.TestCase):
    """GPU-free tests for step plan validation and ordering."""

    def test_no_plan_call_order(self):
        """Without a plan, nodes run on the main stream in call order."""
        node0 = dataflow.Node("n0", "dense", [0], [], {})
        node1 = dataflow.Node("n1", "dense", [0], [], {})
        node2 = dataflow.Node("n2", "dense", [0], [], {})
        ordered = stepgraph.plan_order([node0, node1, node2], None)
        self.assertEqual(len(ordered), 3)
        self.assertEqual([n.name for n, _, _ in ordered], ["n0", "n1", "n2"])
        for _, stream, waits in ordered:
            self.assertEqual(stream, stepgraph_pb2.Stream.STREAM_MAIN)
            self.assertEqual(waits, [])

    def test_read_step_plan_file_not_found(self):
        """read_step_plan raises FileNotFoundError on missing file."""
        with self.assertRaises(FileNotFoundError):
            stepgraph.read_step_plan("/nonexistent/path/plan.json")

    def test_plan_order_invalid_inputs(self):
        """Table test for invalid plan structures: unknown waits, later waits, self-waits,
        invalid stream values, and out-of-order node plans."""
        nodes = [
            dataflow.Node("n0", "dense", [0], [], {}),
            dataflow.Node("n1", "dense", [0], [], {}),
            dataflow.Node("n2", "dense", [0], [], {}),
        ]
        cases = [
            (
                "wait on ghost",
                stepgraph_pb2.StepPlan(nodes=[
                    stepgraph_pb2.NodePlan(name="n0", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n1", stream=stepgraph_pb2.Stream.STREAM_MAIN, waits=["ghost"]),
                    stepgraph_pb2.NodePlan(name="n2", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                ]),
                "ghost",
            ),
            (
                "wait on later node",
                stepgraph_pb2.StepPlan(nodes=[
                    stepgraph_pb2.NodePlan(name="n0", stream=stepgraph_pb2.Stream.STREAM_MAIN, waits=["n2"]),
                    stepgraph_pb2.NodePlan(name="n1", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n2", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                ]),
                "n2",
            ),
            (
                "wait on self",
                stepgraph_pb2.StepPlan(nodes=[
                    stepgraph_pb2.NodePlan(name="n0", stream=stepgraph_pb2.Stream.STREAM_MAIN, waits=["n0"]),
                    stepgraph_pb2.NodePlan(name="n1", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n2", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                ]),
                "n0",
            ),
            (
                "stream value outside enum",
                stepgraph_pb2.StepPlan(nodes=[
                    stepgraph_pb2.NodePlan(name="n0", stream=3),
                    stepgraph_pb2.NodePlan(name="n1", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n2", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                ]),
                "invalid stream",
            ),
            (
                "reverse order",
                stepgraph_pb2.StepPlan(nodes=[
                    stepgraph_pb2.NodePlan(name="n2", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n1", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                    stepgraph_pb2.NodePlan(name="n0", stream=stepgraph_pb2.Stream.STREAM_MAIN),
                ]),
                "mismatch at index 0",
            ),
        ]
        for name, plan, err_substr in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError) as ctx:
                    stepgraph.plan_order(nodes, plan)
                self.assertIn(err_substr, str(ctx.exception))

    def test_step_plan_file_requires_step_graph(self):
        """VXPU_STEP_PLAN_FILE requires VXPU_STEP_GRAPH=1."""
        saved = {k: os.environ.get(k) for k in ("VXPU_STEP_PLAN_FILE", "VXPU_STEP_GRAPH")}
        os.environ["VXPU_STEP_PLAN_FILE"] = "/tmp/plan.json"
        os.environ["VXPU_STEP_GRAPH"] = "0"
        try:
            with self.assertRaises(ConfigError) as ctx:
                Plan("cpu")
            self.assertIn("VXPU_STEP_PLAN_FILE requires VXPU_STEP_GRAPH=1", str(ctx.exception))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


@unittest.skipUnless(torch.cuda.is_available() and triton_ops.available(), "needs CUDA and Triton")
class StepPlanGPUTest(unittest.TestCase):
    """The tiny DeepSeek model's step executed with a StepPlan on GPU."""

    def test_gpu_step_plan_32_steps(self):
        from .engine import Engine, SamplingParams
        from .export import export_artifact
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir = tiny_deepseek_v41.write(os.path.join(td, "model"))
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 dependencies unavailable: {e}") from e

            artifact = os.path.join(td, "artifact")
            export_artifact(model_dir, artifact, max_cache_len=64)
            aoti_dir = os.path.join(td, "aoti")
            desc_file = os.path.join(td, "stepgraph.json")
            plan_file = os.path.join(td, "stepplan.json")
            plan_parallel_file = os.path.join(td, "stepplan_parallel.json")

            # 1. Build initial graph and write StepGraphDescription
            env_desc = {
                "VXPU_DATAFLOW": "1",
                "VXPU_SEGMENT_RUNTIME": "aoti",
                "VXPU_AOTI_CACHE_DIR": aoti_dir,
                "VXPU_STEP_GRAPH": "1",
                "VXPU_STEP_GRAPH_FILE": desc_file,
            }
            with patch.dict(os.environ, env_desc):
                eng_desc = Engine(artifact, device="cuda", compile_decode=True)
                self.assertTrue(os.path.exists(desc_file))
                dense_names = {n.name for n in eng_desc._decode_split.runner.graph.nodes if n.kind == "dense"}
                self.assertEqual(len(dense_names), 6)

            # 2. Compute plan with vxpu plan step (every node on main stream)
            run_vxpu_plan_step(desc_file, plan_file, serialize=False)
            self.assertTrue(os.path.exists(plan_file))

            # 3. Create a hand-written plan that keeps the planner's waits exactly
            # and moves the dense nodes to STREAM_PARALLEL, leaving the host ops on main.
            plan_msg = stepgraph.read_step_plan(plan_file)
            branched_count = 0
            for np in plan_msg.nodes:
                if np.name in dense_names:
                    np.stream = stepgraph_pb2.Stream.STREAM_PARALLEL
                    branched_count += 1
                else:
                    np.stream = stepgraph_pb2.Stream.STREAM_MAIN
            self.assertEqual(branched_count, 6)
            with open(plan_parallel_file, "w", encoding="utf-8") as f:
                f.write(json_format.MessageToJson(plan_msg, indent=2))

            steps = 32
            env_base = {
                "VXPU_DATAFLOW": "1",
                "VXPU_SEGMENT_RUNTIME": "aoti",
                "VXPU_AOTI_CACHE_DIR": aoti_dir,
                "VXPU_STEP_GRAPH": "1",
            }

            # Capture 1: No plan
            with patch.dict(os.environ, env_base):
                eng_none = Engine(artifact, device="cuda", compile_decode=True)
                sg_none = eng_none._decode_split
                self.assertIsInstance(sg_none, stepgraph.StepGraph)
                sid_none = eng_none.new_session()
                events_none = list(eng_none.generate(sid_none, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                tokens_none = [ev.get("token", ev["token_ids"][0]) for ev in events_none if not ev.get("done") and (ev.get("token") or ev.get("token_ids"))]

            # Capture 2: Plan from vxpu plan step (nodes on main stream)
            env_plan = dict(env_base, VXPU_STEP_PLAN_FILE=plan_file)
            with patch.dict(os.environ, env_plan):
                eng_plan = Engine(artifact, device="cuda", compile_decode=True)
                sg_plan = eng_plan._decode_split
                self.assertIsInstance(sg_plan, stepgraph.StepGraph)
                sid_plan = eng_plan.new_session()
                events_plan = list(eng_plan.generate(sid_plan, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                tokens_plan = [ev.get("token", ev["token_ids"][0]) for ev in events_plan if not ev.get("done") and (ev.get("token") or ev.get("token_ids"))]
                self.assertEqual(sg_plan._branched, 0)

            # Capture 3: Hand-written plan with dense nodes on STREAM_PARALLEL
            env_parallel = dict(env_base, VXPU_STEP_PLAN_FILE=plan_parallel_file)
            with patch.dict(os.environ, env_parallel):
                eng_par = Engine(artifact, device="cuda", compile_decode=True)
                sg_par = eng_par._decode_split
                self.assertIsInstance(sg_par, stepgraph.StepGraph)
                sid_par = eng_par.new_session()
                events_par = list(eng_par.generate(sid_par, [1, 2, 3], steps, SamplingParams(do_sample=False)))
                tokens_par = [ev.get("token", ev["token_ids"][0]) for ev in events_par if not ev.get("done") and (ev.get("token") or ev.get("token_ids"))]
                self.assertEqual(sg_par._branched, 6)

            # Verify token equality across all three captures
            self.assertEqual(len(tokens_none), steps)
            self.assertEqual(len(tokens_plan), steps)
            self.assertEqual(len(tokens_par), steps)
            self.assertEqual(tokens_none, tokens_plan)
            self.assertEqual(tokens_plan, tokens_par)


if __name__ == "__main__":
    unittest.main()
