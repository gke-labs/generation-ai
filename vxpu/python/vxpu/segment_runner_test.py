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

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from . import engine, kernels, physplan, plan_pb2, segment_runner, segments, vxpu_pb2
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

    def test_run_segment_recording_cpu_host_remote_and_hot_path(self) -> None:
        """run_segment records segment times and local host bodies while leaving remote sites unmeasured."""
        split = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        sub0 = _make_submod(["x"], ["h0"])
        sub1 = _make_submod(["h0"], ["h1"])
        sub2 = _make_submod(["h1"], ["out"])

        host_calls = 0
        remote_calls = 0

        def dense_fn(*args: torch.Tensor) -> torch.Tensor:
            return args[0] + 1.0

        def local_hot_fn(*args: torch.Tensor) -> torch.Tensor:
            return args[0] * 2.0

        def local_host_recorded(*args: torch.Tensor) -> segment_runner.HostCallResult:
            nonlocal host_calls
            host_calls += 1
            x = args[0]
            body = "cpu-fused" if x.shape[0] == 1 else "cpu-grouped"
            return segment_runner.HostCallResult(output=x * 2.0, body=body, rows=int(x.shape[0]))

        def remote_hot_fn(*args: torch.Tensor) -> torch.Tensor:
            return args[0] + 3.0

        def remote_host_recorded(*args: torch.Tensor) -> segment_runner.HostCallResult:
            nonlocal remote_calls
            remote_calls += 1
            return segment_runner.HostCallResult(output=args[0] + 3.0, body=None, rows=None)

        pf = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    host=False,
                    reads=[plan_pb2.Value(name="x", kind=plan_pb2.VALUE_KIND_STEP_INPUT)],
                    writes=[plan_pb2.Value(name="h0", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                ),
                plan_pb2.Segment(
                    name="submod_1",
                    host=True,
                    call_sites=[plan_pb2.CallSite(node="moe_experts", op="vxpu::moe_experts")],
                    reads=[plan_pb2.Value(name="h0", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                    writes=[plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                ),
                plan_pb2.Segment(
                    name="submod_2",
                    host=True,
                    call_sites=[plan_pb2.CallSite(node="moe_experts_1", op="vxpu::moe_experts")],
                    reads=[plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                    writes=[plan_pb2.Value(name="out", kind=plan_pb2.VALUE_KIND_STEP_OUTPUT)],
                ),
            ]
        )

        runner = segment_runner.build_segment_runner(
            split=split,
            plan_file=pf,
            callables={"submod_0": dense_fn, "submod_1": local_hot_fn, "submod_2": remote_hot_fn},
            submodules={"submod_0": sub0, "submod_1": sub1, "submod_2": sub2},
            attr_targets={},
            loaded=0,
            host_callables={"submod_1": local_host_recorded, "submod_2": remote_host_recorded},
            host_keys={"submod_1": ("experts", 0), "submod_2": ("experts", 1)},
        )

        # Hot path with recording off must never invoke host_callables.
        hot_out = runner(torch.tensor([5.0]))
        assert isinstance(hot_out, torch.Tensor)
        self.assertTrue(torch.equal(hot_out, torch.tensor([15.0])))
        self.assertEqual(host_calls, 0)
        self.assertEqual(remote_calls, 0)

        # Active recording captures per-segment time, local host bodies/row counts, and leaves remote unmeasured.
        remote_nodes: dict[tuple[str, int], object] = {("experts", 1): object()}
        self.assertEqual(runner.local_host_sites(remote_nodes), ("moe_experts",))
        rec = segment_runner.ExecutionRecord(device="cpu", local_host_sites=runner.local_host_sites(remote_nodes))
        runner.set_execution_record(rec)
        try:
            out1 = runner(torch.tensor([5.0]))
            out2 = runner(torch.tensor([5.0, 6.0, 7.0, 8.0]))
        finally:
            runner.set_execution_record(None)

        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        self.assertTrue(torch.equal(out1, torch.tensor([15.0])))
        self.assertTrue(torch.equal(out2, torch.tensor([15.0, 17.0, 19.0, 21.0])))
        self.assertEqual(host_calls, 2)
        self.assertEqual(remote_calls, 2)

        summary = rec.summary()
        self.assertEqual(set(summary.segment_measurements), {"submod_0", "submod_1", "submod_2"})
        for seg_us in summary.segment_measurements.values():
            self.assertGreaterEqual(seg_us, 0.0)
        self.assertEqual(set(summary.site_measurements), {"moe_experts"})
        self.assertEqual(set(summary.site_executed_bodies), {"moe_experts"})
        self.assertEqual(
            [(eb.body, list(eb.row_counts)) for eb in summary.site_executed_bodies["moe_experts"]],
            [("cpu-fused", [1]), ("cpu-grouped", [4])],
        )

        # Zero decode steps marks local host sites as measured-and-not-run and leaves remote sites unmeasured.
        zero_rec = segment_runner.ExecutionRecord(device="cpu", local_host_sites=runner.local_host_sites(remote_nodes))
        zero_summary = zero_rec.summary()
        self.assertEqual(zero_summary.segment_measurements, {})
        self.assertEqual(zero_summary.site_measurements, {})
        self.assertEqual(zero_summary.site_executed_bodies, {"moe_experts": []})


def _vxpu_cmd(*subargs: str) -> tuple[list[str], str | None]:
    """Resolves the vxpu CLI command using the installed binary or `go run`."""
    vxpu_bin = shutil.which("vxpu")
    if vxpu_bin is not None:
        return [vxpu_bin, *subargs], None
    if shutil.which("go") is not None:
        cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return ["go", "run", "./cmd/vxpu", *subargs], cwd
    raise unittest.SkipTest("neither vxpu binary nor go is available on PATH")


def _emit_plan(
    artifact_dir: str,
    out_path: str,
    machine: str,
    goal: str,
    table_path: str | None = None,
) -> plan_pb2.PlanFile:
    """Emits a PlanFile from artifact_dir using the Go planner."""
    cmd, cwd = _vxpu_cmd(
        "plan",
        "emit",
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
    )
    if table_path is not None:
        cmd.extend(["--table", table_path])
    subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, env=dict(os.environ, GOTOOLCHAIN="auto"))
    return physplan.read_plan_file(out_path)


def _explain_analyze_tree(artifact_dir: str, plan_path: str, record_path: str) -> dict[str, object]:
    """Runs `vxpu plan tree --json` on a plan and run record."""
    cmd, cwd = _vxpu_cmd(
        "plan",
        "tree",
        "--artifact",
        artifact_dir,
        "--plan",
        plan_path,
        "--record",
        record_path,
        "--json",
    )
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, GOTOOLCHAIN="auto"),
    )
    return json.loads(proc.stdout)


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

    def test_run_record_and_explain_analyze_on_tiny_deepseek(self) -> None:
        """run_segment populates segment times and host call-site facts matching EXPLAIN ANALYZE."""
        try:
            art = tiny_deepseek_v41.get_artifact_dir()
        except Exception as e:
            self.skipTest(f"tiny_deepseek_v41 unavailable: {e}")

        table_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "pkg", "kernelplan", "testdata", "kernel_table.json")
        )
        with tempfile.TemporaryDirectory() as td:
            plan_path = os.path.join(td, "plan.pb")
            record_path = os.path.join(td, "record.pb")
            pf = _emit_plan(art, plan_path, "c3d-standard-60", "pass-through", table_path=table_path)
            plain_pf = plan_pb2.PlanFile()
            plain_pf.CopyFrom(pf)
            for seg in plain_pf.segments:
                seg.run = plan_pb2.SEGMENT_RUN_PLAIN_TORCH
                seg.package_key = ""
            physplan.write_plan_file(plan_path, plain_pf)

            eng = engine.Engine(art, device="cpu", plan=plain_pf, run_record_file=record_path)
            params = engine.SamplingParams(do_sample=False)

            # 1. Local execution: 5 local host sites measured, 62 compiled/dense sites not measured.
            sess = eng.new_session()
            for _ in eng.generate(sess, [1, 42, 7, 13], max_new_tokens=3, params=params, ignore_eos=True):
                pass

            rec = physplan.read_plan_file(record_path)
            self.assertEqual(len(rec.segments), 11)
            for seg in rec.segments:
                self.assertTrue(seg.HasField("total_measured_time_us"), f"segment {seg.name} missing timing")
                self.assertGreater(seg.total_measured_time_us, 0.0)

            host_ids = {"moe_experts", "engram_rows", "moe_experts_1", "moe_experts_2", "moe_experts_3"}
            sites_by_id = {s.id: s for s in rec.sites}
            use_cpu_kernels = bool(eng.plan["kernels"]) and kernels.available()
            expected_moe_body = "cpu-fused" if use_cpu_kernels else "reference"
            expected_engram_body = "host-file" if use_cpu_kernels else "host-lookup"
            for sid, site in sites_by_id.items():
                if sid in host_ids:
                    self.assertTrue(site.measured, f"host site {sid} must be measured")
                    self.assertTrue(site.HasField("stats"), f"host site {sid} missing stats")
                    self.assertGreater(site.stats.measured_us, 0.0)
                    if sid == "engram_rows":
                        self.assertEqual([(eb.body, list(eb.row_counts)) for eb in site.executed_bodies], [(expected_engram_body, [])])
                    else:
                        self.assertEqual([(eb.body, list(eb.row_counts)) for eb in site.executed_bodies], [(expected_moe_body, [1])])
                else:
                    self.assertFalse(site.measured, f"non-host site {sid} must be unmeasured")
                    self.assertEqual(len(site.executed_bodies), 0)
                    self.assertFalse(site.HasField("stats"))

            tree = _explain_analyze_tree(art, plan_path, record_path)
            self.assertEqual(tree["total_call_sites"], 67)
            self.assertEqual(tree["measured_call_sites"], 5)
            self.assertEqual(tree["unmeasured_call_sites"], 62)
            expected_differing = 4 if use_cpu_kernels else 1
            self.assertEqual(tree["differing_call_sites"], expected_differing)
            tree_segments = tree["segments"]
            assert isinstance(tree_segments, list)
            self.assertEqual(len(tree_segments), 11)
            for seg_doc in tree_segments:
                assert isinstance(seg_doc, dict)
                self.assertIn("total_measured_time_us", seg_doc)

            # 2. Remote host op on ("experts", 0) must leave moe_experts unmeasured.
            def fake_remote_moe(
                x_in: torch.Tensor,
                weights: torch.Tensor,
                indices: torch.Tensor,
            ) -> torch.Tensor:
                return torch.zeros((x_in.shape[0], x_in.shape[1]), dtype=torch.float32, device=x_in.device)

            eng.external.remote_nodes[("experts", 0)] = fake_remote_moe
            try:
                sess_remote = eng.new_session()
                for _ in eng.generate(sess_remote, [1, 42, 7, 13], max_new_tokens=2, params=params, ignore_eos=True):
                    pass

                rec_remote = physplan.read_plan_file(record_path)
                remote_sites = {s.id: s for s in rec_remote.sites}
                self.assertFalse(remote_sites["moe_experts"].measured)
                self.assertEqual(len(remote_sites["moe_experts"].executed_bodies), 0)
                self.assertFalse(remote_sites["moe_experts"].HasField("stats"))
                for sid in host_ids - {"moe_experts"}:
                    self.assertTrue(remote_sites[sid].measured)
                    self.assertEqual(len(remote_sites[sid].executed_bodies), 1)

                tree_remote = _explain_analyze_tree(art, plan_path, record_path)
                self.assertEqual(tree_remote["measured_call_sites"], 4)
                self.assertEqual(tree_remote["unmeasured_call_sites"], 63)

                # 3. Zero decode steps (short prompt stepped through _decode_run during prefill only)
                # must not leak prefill calls into the decode run record.
                sess_zero = eng.new_session()
                for _ in eng.generate(sess_zero, [1], max_new_tokens=0, params=params, ignore_eos=True):
                    pass

                rec_zero = physplan.read_plan_file(record_path)
                for seg in rec_zero.segments:
                    self.assertFalse(seg.HasField("total_measured_time_us"))
                zero_sites = {s.id: s for s in rec_zero.sites}
                self.assertFalse(zero_sites["moe_experts"].measured)
                for sid in host_ids - {"moe_experts"}:
                    self.assertTrue(zero_sites[sid].measured)
                    self.assertEqual(len(zero_sites[sid].executed_bodies), 0)
                    self.assertFalse(zero_sites[sid].HasField("stats"))
            finally:
                eng.external.remote_nodes.pop(("experts", 0), None)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class SegmentRunnerCUDATest(unittest.TestCase):
    """CUDA event timing and recording verification for run_segment and ExecutionRecord."""

    def test_run_segment_recording_cuda_events(self) -> None:
        """ExecutionRecord drains CUDA timing events and records non-zero segment and host times."""
        device = torch.device("cuda")
        split = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        sub0 = _make_submod(["x"], ["h0"])
        sub1 = _make_submod(["h0"], ["h1"])
        sub2 = _make_submod(["h1"], ["out"])

        weight = torch.eye(64, device=device, dtype=torch.float32) * 2.0
        host_calls = 0
        remote_calls = 0

        def dense_fn(*args: torch.Tensor) -> torch.Tensor:
            return args[0] @ weight

        def local_hot_fn(*args: torch.Tensor) -> torch.Tensor:
            return (args[0].cpu() + 1.0).to(device)

        def local_host_recorded(*args: torch.Tensor) -> segment_runner.HostCallResult:
            nonlocal host_calls
            host_calls += 1
            x = args[0]
            body = "cpu-fused" if x.shape[0] == 1 else "cpu-grouped"
            return segment_runner.HostCallResult(output=(x.cpu() + 1.0).to(device), body=body, rows=int(x.shape[0]))

        def remote_hot_fn(*args: torch.Tensor) -> torch.Tensor:
            return args[0] + 3.0

        def remote_host_recorded(*args: torch.Tensor) -> segment_runner.HostCallResult:
            nonlocal remote_calls
            remote_calls += 1
            return segment_runner.HostCallResult(output=args[0] + 3.0, body=None, rows=None)

        pf = plan_pb2.PlanFile(
            segments=[
                plan_pb2.Segment(
                    name="submod_0",
                    host=False,
                    reads=[plan_pb2.Value(name="x", kind=plan_pb2.VALUE_KIND_STEP_INPUT)],
                    writes=[plan_pb2.Value(name="h0", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                ),
                plan_pb2.Segment(
                    name="submod_1",
                    host=True,
                    call_sites=[plan_pb2.CallSite(node="moe_experts", op="vxpu::moe_experts")],
                    reads=[plan_pb2.Value(name="h0", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                    writes=[plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                ),
                plan_pb2.Segment(
                    name="submod_2",
                    host=True,
                    call_sites=[plan_pb2.CallSite(node="moe_experts_1", op="vxpu::moe_experts")],
                    reads=[plan_pb2.Value(name="h1", kind=plan_pb2.VALUE_KIND_INTERMEDIATE)],
                    writes=[plan_pb2.Value(name="out", kind=plan_pb2.VALUE_KIND_STEP_OUTPUT)],
                ),
            ]
        )

        runner = segment_runner.build_segment_runner(
            split=split,
            plan_file=pf,
            callables={"submod_0": dense_fn, "submod_1": local_hot_fn, "submod_2": remote_hot_fn},
            submodules={"submod_0": sub0, "submod_1": sub1, "submod_2": sub2},
            attr_targets={},
            loaded=0,
            host_callables={"submod_1": local_host_recorded, "submod_2": remote_host_recorded},
            host_keys={"submod_1": ("experts", 0), "submod_2": ("experts", 1)},
        )

        # Hot path on CUDA without recording must skip host_callables.
        x1 = torch.ones((1, 64), device=device, dtype=torch.float32)
        hot_out = runner(x1)
        assert isinstance(hot_out, torch.Tensor)
        self.assertTrue(torch.equal(hot_out, torch.full((1, 64), 6.0, device=device)))
        self.assertEqual(host_calls, 0)
        self.assertEqual(remote_calls, 0)

        # Run 10 steps (> _CUDA_DRAIN_EVERY=8) to exercise both periodic end_step() and final summary() draining.
        remote_nodes: dict[tuple[str, int], object] = {("experts", 1): object()}
        rec = segment_runner.ExecutionRecord(device=device, local_host_sites=runner.local_host_sites(remote_nodes))
        self.assertTrue(rec.cuda)
        x4 = torch.ones((4, 64), device=device, dtype=torch.float32)
        runner.set_execution_record(rec)
        try:
            for step in range(10):
                inp = x1 if step < 8 else x4
                out = runner(inp)
                assert isinstance(out, torch.Tensor)
                self.assertTrue(torch.equal(out, torch.full_like(inp, 6.0)))
        finally:
            runner.set_execution_record(None)

        self.assertEqual(host_calls, 10)
        self.assertEqual(remote_calls, 10)
        summary = rec.summary()
        self.assertEqual(set(summary.segment_measurements), {"submod_0", "submod_1", "submod_2"})
        for seg_name, seg_us in summary.segment_measurements.items():
            self.assertGreater(seg_us, 0.0, f"segment {seg_name} CUDA time must be positive")
        self.assertEqual(set(summary.site_measurements), {"moe_experts"})
        self.assertGreater(summary.site_measurements["moe_experts"], 0.0)
        self.assertEqual(
            [(eb.body, list(eb.row_counts)) for eb in summary.site_executed_bodies["moe_experts"]],
            [("cpu-fused", [1]), ("cpu-grouped", [4])],
        )

    def test_tiny_deepseek_run_record_on_cuda(self) -> None:
        """Engine records non-zero CUDA segment times and local host bodies on tiny DeepSeek."""
        try:
            art = tiny_deepseek_v41.get_artifact_dir()
        except Exception as e:
            self.skipTest(f"tiny_deepseek_v41 unavailable: {e}")

        with tempfile.TemporaryDirectory() as td:
            plan_path = os.path.join(td, "plan.pb")
            record_path = os.path.join(td, "record.pb")
            pf = _emit_plan(art, plan_path, "g2-standard-24", "pass-through")
            plain_pf = plan_pb2.PlanFile()
            plain_pf.CopyFrom(pf)
            for seg in plain_pf.segments:
                seg.run = plan_pb2.SEGMENT_RUN_PLAIN_TORCH
                seg.package_key = ""
            physplan.write_plan_file(plan_path, plain_pf)

            eng = engine.Engine(art, device="cuda", plan=plain_pf, run_record_file=record_path)
            try:
                params = engine.SamplingParams(do_sample=False)
                sess = eng.new_session()
                for _ in eng.generate(sess, [1, 42, 7, 13], max_new_tokens=3, params=params, ignore_eos=True):
                    pass
            finally:
                del eng
                torch.cuda.empty_cache()

            rec = physplan.read_plan_file(record_path)
            self.assertEqual(len(rec.segments), 11)
            for seg in rec.segments:
                self.assertTrue(seg.HasField("total_measured_time_us"), f"segment {seg.name} missing timing")
                self.assertGreater(seg.total_measured_time_us, 0.0)

            host_ids = {"moe_experts", "engram_rows", "moe_experts_1", "moe_experts_2", "moe_experts_3"}
            sites_by_id = {s.id: s for s in rec.sites}
            for sid, site in sites_by_id.items():
                if sid in host_ids:
                    self.assertTrue(site.measured, f"host site {sid} must be measured")
                    self.assertTrue(site.HasField("stats"), f"host site {sid} missing stats")
                    self.assertGreater(site.stats.measured_us, 0.0)
                    self.assertEqual(len(site.executed_bodies), 1)
                else:
                    self.assertFalse(site.measured, f"non-host site {sid} must be unmeasured")
                    self.assertEqual(len(site.executed_bodies), 0)
                    self.assertFalse(site.HasField("stats"))


if __name__ == "__main__":
    unittest.main()
