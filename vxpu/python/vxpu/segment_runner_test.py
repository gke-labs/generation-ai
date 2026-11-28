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

"""Tests for sequential plan-segment execution through run_segment."""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from . import engine, physplan, plan_pb2, segment_runner, segments, vxpu_pb2
from .fixtures import tiny_deepseek_v41


def _make_submod(input_names: list[str], output_names: list[str]) -> torch.fx.GraphModule:
    """Builds a minimal GraphModule with the given placeholder and output node names."""
    graph = torch.fx.Graph()
    phs = [graph.placeholder(name) for name in input_names]
    anchor = phs[0]
    out_nodes: list[torch.fx.Node] = []
    for name in output_names:
        node = graph.create_node("call_function", torch.clone, args=(anchor,), name=name)
        out_nodes.append(node)
    if len(out_nodes) == 1:
        graph.output(out_nodes[0])
    else:
        graph.output(tuple(out_nodes))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


class _RecordingObserver:
    """Minimal SegmentRecorder capturing event order for tests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def begin_step(self) -> None:
        self.events.append("begin")

    def after_segment(self, index: int) -> None:
        self.events.append(f"seg:{index}")

    def end_step(self) -> None:
        self.events.append("end")


class SegmentRunnerUnitTest(unittest.TestCase):
    """Unit tests for run_segment and build_segment_runner slot wiring."""

    def test_multi_segment_reordered_reads_writes_and_resident_commit(self) -> None:
        """Verifies slot wiring strictly from Segment.reads/writes including resident updates."""
        split = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        split.register_buffer("weight", torch.tensor([2.0, 3.0]))
        split.register_buffer("cache", torch.tensor([10.0, 20.0]))

        # submod_0 expects (x, weight_attr, cache_attr) — deliberately reversed from plan reads.
        # submod_0 returns (h1, cache_new) — deliberately reversed from plan writes.
        sub0 = _make_submod(["x", "weight_attr", "cache_attr"], ["h1", "cache_new"])
        # submod_1 reads h1 (intermediate), cache_attr (pre-step resident), cache_new (updated resident).
        sub1 = _make_submod(["h1", "cache_attr", "cache_new"], ["logits"])

        calls: list[str] = []

        def fn0(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x, w, c = args
            calls.append("submod_0")
            return x * w, c + 1.0

        def fn1(*args: torch.Tensor) -> torch.Tensor:
            h1, old_c, new_c = args
            calls.append("submod_1")
            return h1 + old_c + new_c

        pf = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    host=False,
                    reads=[
                        plan_pb2.Value(name="b_cache", kind=plan_pb2.VALUE_KIND_RESIDENT, resident="cache"),
                        plan_pb2.Value(name="p_weight", kind=plan_pb2.VALUE_KIND_RESIDENT, resident="weight"),
                        plan_pb2.Value(name="x", kind=plan_pb2.VALUE_KIND_STEP_INPUT),
                    ],
                    writes=[
                        plan_pb2.Value(name="cache_new", kind=plan_pb2.VALUE_KIND_RESIDENT, resident="cache"),
                        plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE),
                    ],
                ),
                plan_pb2.Segment(
                    name="submod_1",
                    host=True,
                    reads=[
                        plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE),
                        plan_pb2.Value(name="b_cache", kind=plan_pb2.VALUE_KIND_RESIDENT, resident="cache"),
                        plan_pb2.Value(name="cache_new", kind=plan_pb2.VALUE_KIND_RESIDENT, resident="cache"),
                    ],
                    writes=[
                        plan_pb2.Value(name="logits", kind=plan_pb2.VALUE_KIND_STEP_OUTPUT),
                    ],
                ),
            ]
        )

        runner = segment_runner.build_segment_runner(
            split=split,
            plan_file=pf,
            callables={"submod_0": fn0, "submod_1": fn1},
            submodules={"submod_0": sub0, "submod_1": sub1},
            attr_targets={"weight_attr": "weight", "cache_attr": "cache"},
            loaded=1,
        )
        self.assertEqual(runner.vxpu_segments, (2, 1))
        self.assertEqual([s.name for s in runner.segments], ["submod_0", "submod_1"])
        self.assertEqual([s.host for s in runner.segments], [False, True])

        obs = _RecordingObserver()
        runner.set_recorder(obs)

        # Step 1: x=[1, 2], w=[2, 3] -> h1=[2, 6]; old_c=[10, 20], new_c=[11, 21] -> logits=[23, 47].
        out1 = runner(x=torch.tensor([1.0, 2.0]))
        self.assertIsInstance(out1, torch.Tensor)
        assert isinstance(out1, torch.Tensor)
        self.assertTrue(torch.equal(out1, torch.tensor([23.0, 47.0])))
        self.assertEqual(calls, ["submod_0", "submod_1"])
        self.assertEqual(obs.events, ["begin", "seg:0", "seg:1", "end"])
        # Resident buffer was committed to [11, 21] at end of step 1.
        self.assertTrue(torch.equal(split.get_buffer("cache"), torch.tensor([11.0, 21.0])))

        # Step 2 (positional call): old_c=[11, 21], new_c=[12, 22] -> logits=[25, 49].
        runner.set_recorder(None)
        out2 = runner(torch.tensor([1.0, 2.0]))
        assert isinstance(out2, torch.Tensor)
        self.assertTrue(torch.equal(out2, torch.tensor([25.0, 49.0])))
        self.assertTrue(torch.equal(split.get_buffer("cache"), torch.tensor([12.0, 22.0])))

    def test_validation_rejects_broken_plan_wiring(self) -> None:
        """Rejects plans that read unwritten intermediates or declare no step outputs."""
        split = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        sub0 = _make_submod(["missing"], ["out"])
        pf_bad_read = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    reads=[plan_pb2.Value(name="missing", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                    writes=[plan_pb2.Value(name="out", kind=plan_pb2.VALUE_KIND_STEP_OUTPUT)],
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "before any segment writes it"):
            segment_runner.build_segment_runner(
                split=split,
                plan_file=pf_bad_read,
                callables={"submod_0": sub0},
                submodules={"submod_0": sub0},
                attr_targets={},
                loaded=0,
            )

        sub_no_out = _make_submod(["x"], ["mid"])
        pf_no_step_out = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    reads=[plan_pb2.Value(name="x", kind=plan_pb2.VALUE_KIND_STEP_INPUT)],
                    writes=[plan_pb2.Value(name="mid", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "no step outputs"):
            segment_runner.build_segment_runner(
                split=split,
                plan_file=pf_no_step_out,
                callables={"submod_0": sub_no_out},
                submodules={"submod_0": sub_no_out},
                attr_targets={},
                loaded=0,
            )

    def test_run_segment_arity_errors(self) -> None:
        """run_segment fails fast when callable output arity disagrees with output_slots."""
        env = [torch.tensor([1.0]), torch.empty(0), torch.empty(0)]
        seg_expect_one = segment_runner.SegmentExecution(
            name="s0",
            host=False,
            callable=lambda *args: (args[0], args[0]),
            input_slots=(0,),
            output_slots=(1,),
        )
        with self.assertRaisesRegex(RuntimeError, "expected 1 tensor output"):
            segment_runner.run_segment(seg_expect_one, env)

        seg_expect_two = segment_runner.SegmentExecution(
            name="s1",
            host=False,
            callable=lambda *args: args[0],
            input_slots=(0,),
            output_slots=(1, 2),
        )
        with self.assertRaisesRegex(RuntimeError, "expected 2 outputs"):
            segment_runner.run_segment(seg_expect_two, env)

    def test_positional_step_input_order_follows_split_placeholders(self) -> None:
        """Positional calls follow split.graph placeholder order even when seg.reads is reversed."""
        top_graph = torch.fx.Graph()
        p_first = top_graph.placeholder("first")
        top_graph.placeholder("second")
        top_graph.output(p_first)
        split = torch.fx.GraphModule(torch.nn.Module(), top_graph)

        # seg.reads lists "second" before "first", reversed from split.graph's (first, second).
        sub0 = _make_submod(["second", "first"], ["out"])

        def fn0(*args: torch.Tensor) -> torch.Tensor:
            second, first = args
            return first * 10.0 + second

        pf = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    reads=[
                        plan_pb2.Value(name="second", kind=plan_pb2.VALUE_KIND_STEP_INPUT),
                        plan_pb2.Value(name="first", kind=plan_pb2.VALUE_KIND_STEP_INPUT),
                    ],
                    writes=[plan_pb2.Value(name="out", kind=plan_pb2.VALUE_KIND_STEP_OUTPUT)],
                )
            ]
        )

        runner = segment_runner.build_segment_runner(
            split=split,
            plan_file=pf,
            callables={"submod_0": fn0},
            submodules={"submod_0": sub0},
            attr_targets={},
            loaded=0,
        )

        expected = torch.tensor([23.0])
        out_pos = runner(torch.tensor([2.0]), torch.tensor([3.0]))
        assert isinstance(out_pos, torch.Tensor)
        self.assertTrue(torch.equal(out_pos, expected))

        out_kw = runner(first=torch.tensor([2.0]), second=torch.tensor([3.0]))
        assert isinstance(out_kw, torch.Tensor)
        self.assertTrue(torch.equal(out_kw, expected))

        out_mixed = runner(torch.tensor([2.0]), second=torch.tensor([3.0]))
        assert isinstance(out_mixed, torch.Tensor)
        self.assertTrue(torch.equal(out_mixed, expected))

        with self.assertRaisesRegex(TypeError, "multiple values for step input 'first'"):
            runner(torch.tensor([2.0]), first=torch.tensor([3.0]))

        with self.assertRaisesRegex(TypeError, "unexpected step input keyword argument 'bogus'"):
            runner(torch.tensor([2.0]), bogus=torch.tensor([3.0]))


def _emit_plan(artifact_dir: str, out_path: str, machine: str, goal: str) -> plan_pb2.PlanFile:
    """Emits a PlanFile from artifact_dir using the Go planner."""
    vxpu_bin = shutil.which("vxpu")
    if vxpu_bin is not None:
        cmd = [vxpu_bin, "plan", "emit"]
        cwd = None
    elif shutil.which("go") is not None:
        cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cmd = ["go", "run", "./cmd/vxpu", "plan", "emit"]
    else:
        raise unittest.SkipTest("neither vxpu binary nor go is available on PATH")
    cmd.extend([
        "--artifact",
        artifact_dir,
        "--machine",
        machine,
        "--goal",
        goal,
        "--out",
        out_path,
        "--tree-commit",
        "0123456789abcdef0123456789abcdef01234567",
    ])
    subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=dict(os.environ, GOTOOLCHAIN="auto"))
    return physplan.read_plan_file(out_path)


class TinyDeepSeekSegmentRunnerTest(unittest.TestCase):
    """End-to-end equivalence between SegmentRunner and segments.Split on tiny DeepSeek."""

    def test_matches_split_bit_identically_on_cpu(self) -> None:
        """SegmentRunner produces bit-identical logits and greedy tokens as Split on tiny DeepSeek."""
        try:
            art = tiny_deepseek_v41.get_artifact_dir()
        except Exception as e:
            self.skipTest(f"tiny_deepseek_v41 unavailable: {e}")

        with tempfile.TemporaryDirectory() as td:
            pf = _emit_plan(art, os.path.join(td, "plan.pb"), "c3d-standard-60", "pass-through")
            # Run all segments as plain torch to compare loop vs Split directly without package compilation.
            plain_pf = plan_pb2.PlanFile()
            plain_pf.CopyFrom(pf)
            for seg in plain_pf.segments:
                seg.run = plan_pb2.SEGMENT_RUN_PLAIN_TORCH
                seg.package_key = ""

            eng = engine.Engine(art, device="cpu", plan=plain_pf)
            runner = eng._decode_split
            self.assertIsInstance(runner, segment_runner.SegmentRunner)
            assert isinstance(runner, segment_runner.SegmentRunner)
            self.assertEqual(len(runner.segments), len(plain_pf.segments))

            prompt_ids = [1, 42, 7, 13]
            params = engine.SamplingParams(do_sample=False)

            loop_logits: list[torch.Tensor] = []
            loop_tokens: list[int] = []
            tel_segments = 0
            sess_a = eng.new_session()
            with eng.external_scope():
                for ev in eng.generate(
                    sess_a,
                    prompt_ids,
                    max_new_tokens=6,
                    params=params,
                    logits_sink=lambda _step, t: loop_logits.append(t.detach().clone()),
                    telemetry=vxpu_pb2.TELEMETRY_SUMMARY,
                    ignore_eos=True,
                ):
                    if ev["done"]:
                        tel_segments = len(ev["telemetry"]["segments"])
                    else:
                        loop_tokens.extend(ev["token_ids"])
            self.assertEqual(tel_segments, len(plain_pf.segments))

            # Run a second session with a different prompt, then re-run prompt_ids in a third
            # session to confirm resident buffer switching through Engine._bind stays isolated.
            other_sess = eng.new_session()
            repeat_tokens: list[int] = []
            repeat_sess = eng.new_session()
            with eng.external_scope():
                for _ in eng.generate(other_sess, [2, 99, 5, 3], max_new_tokens=3, params=params, ignore_eos=True):
                    pass
                for ev in eng.generate(repeat_sess, prompt_ids, max_new_tokens=6, params=params, ignore_eos=True):
                    if not ev["done"]:
                        repeat_tokens.extend(ev["token_ids"])
            self.assertEqual(repeat_tokens, loop_tokens)

            # Compare against legacy Split over the exact same split module and resident state.
            split_module_runner = segments.Split(runner.split, *runner.vxpu_segments)
            split_logits: list[torch.Tensor] = []
            split_tokens: list[int] = []
            sess_b = eng.new_session()
            with (
                mock.patch.object(eng, "_decode_run", split_module_runner),
                mock.patch.object(eng, "_decode_split", split_module_runner),
                eng.external_scope(),
            ):
                for ev in eng.generate(
                    sess_b,
                    prompt_ids,
                    max_new_tokens=6,
                    params=params,
                    logits_sink=lambda _step, t: split_logits.append(t.detach().clone()),
                    ignore_eos=True,
                ):
                    if not ev["done"]:
                        split_tokens.extend(ev["token_ids"])

            self.assertEqual(loop_tokens, split_tokens)
            self.assertEqual(len(loop_logits), len(split_logits))
            for step, (a, b) in enumerate(zip(loop_logits, split_logits)):
                self.assertTrue(
                    torch.equal(a, b),
                    f"step {step} logits differed: max diff {(a - b).abs().max().item()}",
                )


if __name__ == "__main__":
    unittest.main()
