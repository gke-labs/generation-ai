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

"""Tests for describing the step graph as a document."""

import os
import tempfile
import unittest

from google.protobuf import json_format

from . import dataflow, stepgraph_pb2
from .engine import Engine
from .export import export_artifact
from .plan import ConfigError


class StepGraphDescriptionTest(unittest.TestCase):
    """Verifies that the step graph description records facts accurately."""

    def test_deepseek_split_description(self) -> None:
        """Verifies that the tiny DeepSeek fixture describes alternating host ops and cache writes."""
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir = tiny_deepseek_v41.write(os.path.join(td, "model"))
            except ImportError as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 fork unavailable: {e}") from e

            artifact = os.path.join(td, "artifact")
            export_artifact(model_dir, artifact, max_cache_len=64)
            saved = os.environ.get("VXPU_DATAFLOW")
            os.environ["VXPU_DATAFLOW"] = "1"
            try:
                eng = Engine(artifact, device="cpu", compile_decode=True)
                runner = getattr(eng, "_decode_split")
                self.assertIsNotNone(runner)
                desc = runner.graph.description()
            finally:
                if saved is None:
                    os.environ.pop("VXPU_DATAFLOW", None)
                else:
                    os.environ["VXPU_DATAFLOW"] = saved

            # 1. Nodes alternate through host ops in call order (6 dense, 4 experts, 1 engram).
            kinds = [n.kind for n in desc.nodes]
            self.assertEqual(len(kinds), 11)
            self.assertEqual(kinds.count("dense"), 6)
            self.assertEqual(kinds.count("experts"), 4)
            self.assertEqual(kinds.count("engram"), 1)
            for i, kind in enumerate(kinds):
                if i % 2 == 0:
                    self.assertEqual(kind, "dense")
                else:
                    self.assertIn(kind, ("experts", "engram"))

            # 1b. Execution reflects how nodes run: 6 graph, 5 exchange.
            executions = [n.execution for n in desc.nodes]
            self.assertEqual(executions.count(stepgraph_pb2.Execution.EXECUTION_GRAPH), 6)
            self.assertEqual(executions.count(stepgraph_pb2.Execution.EXECUTION_EXCHANGE), 5)
            for i, execution in enumerate(executions):
                if i % 2 == 0:
                    self.assertEqual(execution, stepgraph_pb2.Execution.EXECUTION_GRAPH)
                else:
                    self.assertEqual(execution, stepgraph_pb2.Execution.EXECUTION_EXCHANGE)

            # 2. Some dense node lists a compress_kv_cache resident under writes.
            written_compress = [
                (i, n, w)
                for i, n in enumerate(desc.nodes)
                if n.kind == "dense"
                for w in n.writes
                if "compress_kv_cache" in w
            ]
            self.assertTrue(len(written_compress) > 0)

            # 3. A later node reads that same resident and does not list it under writes.
            writer_idx, _, target_res = written_compress[0]
            found_later_reader = False
            for later_node in desc.nodes[writer_idx + 1:]:
                read_residents = [r.resident for r in later_node.reads if r.WhichOneof("ref") == "resident"]
                if target_res in read_residents:
                    self.assertNotIn(target_res, later_node.writes)
                    found_later_reader = True
                    break
            self.assertTrue(found_later_reader)

            # JSON round-trips through json_format.
            json_str = json_format.MessageToJson(desc, preserving_proto_field_name=True)
            parsed = json_format.Parse(json_str, stepgraph_pb2.StepGraphDescription())
            self.assertEqual(parsed, desc)

    def test_engine_writes_step_graph(self) -> None:
        """Verifies that VXPU_STEP_GRAPH_FILE causes the engine to write the description file."""
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-llama")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)
            out_file = os.path.join(td, "stepgraph.json")
            saved = {k: os.environ.get(k) for k in ("VXPU_DATAFLOW", "VXPU_STEP_GRAPH_FILE")}
            os.environ["VXPU_DATAFLOW"] = "1"
            os.environ["VXPU_STEP_GRAPH_FILE"] = out_file
            try:
                eng = Engine(artifact, device="cpu", compile_decode=True)
                self.assertIsNotNone(eng)
                self.assertTrue(os.path.exists(out_file))
                with open(out_file, encoding="utf-8") as f:
                    desc = json_format.Parse(f.read(), stepgraph_pb2.StepGraphDescription())
                self.assertGreater(len(desc.nodes), 0)
                self.assertGreater(len(desc.inputs), 0)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_step_graph_file_requires_dataflow(self) -> None:
        """Verifies that VXPU_STEP_GRAPH_FILE requires VXPU_DATAFLOW=1."""
        with tempfile.TemporaryDirectory() as td:
            artifact = os.path.join(td, "tiny-llama")
            export_artifact("trl-internal-testing/tiny-random-LlamaForCausalLM", artifact, max_cache_len=64)
            saved = {k: os.environ.get(k) for k in ("VXPU_DATAFLOW", "VXPU_STEP_GRAPH_FILE")}
            os.environ["VXPU_DATAFLOW"] = "0"
            os.environ["VXPU_STEP_GRAPH_FILE"] = os.path.join(td, "out.json")
            try:
                with self.assertRaises(ConfigError) as ctx:
                    Engine(artifact, device="cpu", compile_decode=True)
                self.assertIn("VXPU_STEP_GRAPH_FILE requires VXPU_DATAFLOW=1", str(ctx.exception))
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_graph_description_invalid_write_raises(self) -> None:
        """Verifies that out-of-bounds writes or non-resident writes raise ValueError."""
        # Out-of-bounds write
        node = dataflow.Node("n1", "dense", [0], [], {}, writes=[0])
        graph = dataflow.Graph([node], [], None)
        with self.assertRaises(ValueError) as ctx:
            graph.description()
        self.assertIn("node 'n1'", str(ctx.exception))

        # Non-resident write (e.g. edge or input)
        node_edge = dataflow.Node("n2", "dense", [0], [dataflow.Ref("edge", node="n1")], {}, writes=[0])
        graph_edge = dataflow.Graph([node_edge], [], None)
        with self.assertRaises(ValueError) as ctx:
            graph_edge.description()
        self.assertIn("node 'n2'", str(ctx.exception))

    def test_node_execution_validation(self) -> None:
        """Verifies that Node validates execution string and consistency with host call."""
        # Valid cases
        n_graph = dataflow.Node("g", "dense", [0], [], {})
        self.assertEqual(n_graph.execution, "graph")
        n_exchange = dataflow.Node("e", "experts", [0], [], {}, host=("experts", 0))
        self.assertEqual(n_exchange.execution, "exchange")

        # Unknown execution string rejected
        with self.assertRaises(ValueError) as ctx:
            dataflow.Node("bad", "dense", [0], [], {}, execution="unknown")
        self.assertIn("unknown execution 'unknown'", str(ctx.exception))

        # Graph node cannot have host call
        with self.assertRaises(ValueError) as ctx:
            dataflow.Node("bad", "dense", [0], [], {}, host=("experts", 0), execution="graph")
        self.assertIn("graph node cannot have a host call", str(ctx.exception))

        # Exchange node must have host call
        with self.assertRaises(ValueError) as ctx:
            dataflow.Node("bad", "experts", [0], [], {}, execution="exchange")
        self.assertIn("exchange node must have a host call", str(ctx.exception))

    def test_write_step_graph_bare_filename(self) -> None:
        """Verifies that writing to a bare file name creates temporary files in the target directory, not /tmp."""
        import glob
        desc = stepgraph_pb2.StepGraphDescription()
        sys_tmp = tempfile.gettempdir()
        before_sys = set(glob.glob(os.path.join(sys_tmp, "stepgraph-*.tmp")))
        with tempfile.TemporaryDirectory() as td:
            orig_cwd = os.getcwd()
            os.chdir(td)
            try:
                dataflow.write_step_graph("stepgraph.json", desc)
                self.assertTrue(os.path.exists("stepgraph.json"))
                self.assertEqual(glob.glob("stepgraph-*.tmp"), [])
                after_sys = set(glob.glob(os.path.join(sys_tmp, "stepgraph-*.tmp")))
                self.assertEqual(after_sys - before_sys, set())
            finally:
                os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
