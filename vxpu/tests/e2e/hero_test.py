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

"""End-to-end hero journey test for vxpu.

The hero journey exercises the full lifecycle of a model:
1. Upload through the Python front (vxpu.AutoModelForCausalLM.from_pretrained).
2. Pass-through execution with an unaltering plan matching eager transformers.
3. Optimized execution with planner-chosen machine optimizations.
4. Measurements and comparison of planner choices and speedup.

The test reports each step separately as passed, failed, or not yet possible,
printing a summary table to the job log and GitHub Actions step summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM as TransformersModel
from transformers import AutoTokenizer

from vxpu import AutoModelForCausalLM
from vxpu import modelstore as vxpu_modelstore
from vxpu.client import ARTIFACT_FILES

# Steps expected not to pass yet, each with the reason naming the missing piece.
EXPECTED_NOT_YET_POSSIBLE: dict[str, str] = {}


@dataclass
class StepReport:
    """Report for an individual step in the hero journey."""

    step: str
    result: str  # "passed", "failed", "not yet possible"
    missing: str
    seconds: float
    error_detail: str = ""


def get_plan_identity(plan_path: str) -> Optional[str]:
    """Reads the plan identity field computed by the planner in the plan message."""
    from vxpu import physplan

    pf = physplan.read_plan_file(plan_path)
    ident = getattr(pf, "plan_hash", "")
    return ident if ident else None


def get_engine_plan_identity(model: Any) -> Optional[str]:
    """Returns the plan identity the model is running under, as reported by the engine."""
    plan_hash = getattr(model, "plan_hash", None)
    return str(plan_hash) if plan_hash else None


def diff_plans(plan1_path: str, plan2_path: str) -> tuple[bool, list[str]]:
    """Compares two plan files and returns whether they differ and what changed."""
    with open(plan1_path, "rb") as f1, open(plan2_path, "rb") as f2:
        b1, b2 = f1.read(), f2.read()
    if b1 == b2:
        return False, []

    from vxpu import physplan, plan_pb2

    pf1 = physplan.read_plan_file(plan1_path)
    pf2 = physplan.read_plan_file(plan2_path)

    differences: list[str] = []

    # Check goal
    if pf1.goal != pf2.goal:
        g1 = plan_pb2.PlanGoal.Name(pf1.goal)
        g2 = plan_pb2.PlanGoal.Name(pf2.goal)
        differences.append(f"goal changed: {g1} -> {g2}")

    # Check knobs
    if pf1.knobs != pf2.knobs:
        differences.append(f"knobs differ: {pf2.knobs}")

    # Check planned sites and bodies
    sites1 = {s.id: s for s in pf1.sites}
    sites2 = {s.id: s for s in pf2.sites}
    for site_id, s2 in sites2.items():
        s1 = sites1.get(site_id)
        body2 = s2.body or s2.strategy
        if s1 is None:
            differences.append(f"site {site_id} added with body {body2}")
        else:
            body1 = s1.body or s1.strategy
            if body1 != body2:
                differences.append(f"site {site_id} body changed: {body1} -> {body2}")

    # Check segments
    segs1 = {s.name: s for s in pf1.segments}
    segs2 = {s.name: s for s in pf2.segments}
    for name, s2 in segs2.items():
        s1 = segs1.get(name)
        if s1 is None:
            differences.append(f"segment {name} added ({plan_pb2.SegmentRun.Name(s2.run)})")
        elif s1.run != s2.run:
            r1 = plan_pb2.SegmentRun.Name(s1.run)
            r2 = plan_pb2.SegmentRun.Name(s2.run)
            differences.append(f"segment {name} run changed: {r1} -> {r2}")
        elif str(s1) != str(s2):
            differences.append(f"segment {name} runs differently")

    if not differences:
        return False, []

    return True, differences


def format_table(reports: list[StepReport]) -> str:
    """Formats step reports into a plaintext console table."""
    col_step = "Step"
    col_res = "Result"
    col_miss = "What is missing"
    col_time = "Time"

    max_step = max(len(col_step), max(len(r.step) for r in reports))
    max_res = max(len(col_res), max(len(r.result) for r in reports))
    max_miss = max(len(col_miss), max(len(r.missing) for r in reports))
    max_time = 8

    width = max_step + max_res + max_miss + max_time + 6
    border = "=" * width
    divider = "-" * width

    header = (
        f"{col_step:<{max_step}}  "
        f"{col_res:<{max_res}}  "
        f"{col_miss:<{max_miss}}  "
        f"{col_time:>{max_time}}"
    )
    lines = [
        border,
        "The Hero Journey (End-to-End Test)".center(width),
        border,
        header,
        divider,
    ]
    for r in reports:
        time_str = f"{r.seconds:.2f}s"
        line = (
            f"{r.step:<{max_step}}  "
            f"{r.result:<{max_res}}  "
            f"{r.missing:<{max_miss}}  "
            f"{time_str:>{max_time}}"
        )
        lines.append(line)
    lines.append(border)

    failures = [r for r in reports if r.error_detail]
    if failures:
        lines.append("\nFailure details:")
        for r in failures:
            lines.append(f"  {r.step}: {r.error_detail}")

    return "\n".join(lines)


def format_markdown_table(reports: list[StepReport], model_id: str, device: str = "cpu") -> str:
    """Formats step reports into a GitHub-flavored Markdown table for job summary."""
    lines = [
        "### The Hero Journey (End-to-End Test)",
        "",
        f"**Model:** `{model_id}` | **Device:** `{device}`",
        "",
        "| Step | Result | What is missing | Seconds |",
        "|:---|:---|:---|---:|",
    ]
    for r in reports:
        lines.append(f"| {r.step} | `{r.result}` | {r.missing} | {r.seconds:.2f}s |")
    lines.append("")

    failures = [r for r in reports if r.error_detail]
    if failures:
        lines.append("<details>")
        lines.append("<summary>Failure details</summary>\n")
        for r in failures:
            lines.append(f"- **{r.step}:** {r.error_detail}")
        lines.append("\n</details>\n")

    return "\n".join(lines)


def record_step(
    reports: list[StepReport],
    step: str,
    passed: bool,
    seconds: float,
    failure_detail: str = "",
) -> None:
    """Records the outcome of a step against EXPECTED_NOT_YET_POSSIBLE."""
    if passed:
        if step in EXPECTED_NOT_YET_POSSIBLE:
            print(
                f"STEP PASSED: {step} now passes and must come off EXPECTED_NOT_YET_POSSIBLE!",
                file=sys.stderr,
            )
            reports.append(
                StepReport(
                    step=step,
                    result="failed",
                    missing="step passed unexpectedly; remove from EXPECTED_NOT_YET_POSSIBLE",
                    seconds=seconds,
                    error_detail=f"step passed unexpectedly: {failure_detail}",
                )
            )
        else:
            reports.append(
                StepReport(
                    step=step,
                    result="passed",
                    missing="-",
                    seconds=seconds,
                )
            )
    else:
        if step in EXPECTED_NOT_YET_POSSIBLE:
            reports.append(
                StepReport(
                    step=step,
                    result="not yet possible",
                    missing=EXPECTED_NOT_YET_POSSIBLE[step],
                    seconds=seconds,
                    error_detail=failure_detail,
                )
            )
        else:
            reports.append(
                StepReport(
                    step=step,
                    result="failed",
                    missing=failure_detail or "step execution failed",
                    seconds=seconds,
                    error_detail=failure_detail,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hero Journey End-to-End Test")
    parser.add_argument(
        "--model",
        default="trl-internal-testing/tiny-random-LlamaForCausalLM",
        help="Model repository ID to test",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional path to existing exported artifact directory",
    )
    parser.add_argument(
        "--router",
        default=None,
        help="Router host:port (if omitted, connects via kubectl port-forward)",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Directly reachable vxpu-agent host:port for building speed packages (if omitted, uses kubectl exec on the executor pod)",
    )
    parser.add_argument(
        "--vxpu-bin",
        default=None,
        help="Path to vxpu CLI binary",
    )
    parser.add_argument(
        "--modelstore",
        default=None,
        help="Host-reachable modelstore URL for uploading artifacts and checking packages",
    )
    parser.add_argument(
        "--in-cluster-modelstore",
        default=None,
        help="In-cluster modelstore URL used by the executor agent and compiler",
    )
    parser.add_argument(
        "--machine-kind",
        default=None,
        help="Cluster executor machine kind (e.g. cpu:...:avx512)",
    )
    parser.add_argument(
        "--software-version",
        default=None,
        help="Executor software version (git tree commit)",
    )
    parser.add_argument(
        "--agent-tls-dir",
        default=None,
        help="Host directory containing ca.crt, tls.crt, and tls.key for coordinator-to-agent mTLS",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Path to write GitHub step summary to (defaults to GITHUB_STEP_SUMMARY env var)",
    )
    return parser.parse_args()


def _copy_file_to_pod(
    pod_name: str, local_path: str, remote_path: str, executable: bool = False, mode: Optional[str] = None
) -> None:
    """Writes local_path into remote_path inside pod_name via kubectl exec."""
    with open(local_path, "rb") as f:
        data = f.read()
    shell_cmd = f"cat > {remote_path}"
    if mode is not None:
        shell_cmd += f" && chmod {mode} {remote_path}"
    elif executable:
        shell_cmd += f" && chmod +x {remote_path}"
    res = subprocess.run(
        ["kubectl", "exec", "-i", pod_name, "--", "sh", "-c", shell_cmd],
        input=data,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"copying {local_path} to {pod_name}:{remote_path} failed: "
            f"{res.stderr.decode(errors='replace').strip()}"
        )


def build_speed_packages(
    vxpu_bin: str,
    artifact_dir: str,
    opt_plan_file: str,
    model_id: str,
    modelstore_url: str,
    in_cluster_modelstore_url: str,
    machine_kind: Optional[str],
    agent_addr: Optional[str] = None,
    agent_tls_dir: Optional[str] = None,
) -> None:
    """Uploads the artifact to modelstore and builds the speed plan's packages through the executor agent."""
    art_repo = vxpu_modelstore.artifact_repository(model_id)
    art_tag = "hero"
    with tempfile.TemporaryDirectory() as stage_dir:
        for name in ARTIFACT_FILES:
            shutil.copy2(os.path.join(artifact_dir, name), os.path.join(stage_dir, name))
        vxpu_modelstore.push_directory(modelstore_url, art_repo, art_tag, stage_dir)
    art_ref = f"{art_repo}:{art_tag}"

    pod_name: Optional[str] = None
    remote_tls_dir: Optional[str] = None
    try:
        if agent_addr:
            pkg_cmd = [
                vxpu_bin,
                "plan",
                "packages",
                "--plan",
                opt_plan_file,
                "--artifact",
                art_ref,
                "--model",
                model_id,
                "--modelstore",
                in_cluster_modelstore_url,
                "--build",
                "--agent",
                agent_addr,
            ]
            if agent_tls_dir:
                pkg_cmd.extend(["--tls-dir", agent_tls_dir])
        else:
            pod_res = subprocess.run(
                ["kubectl", "get", "pod", "-l", "app=vxpu-executor", "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True,
                text=True,
            )
            pod_name = pod_res.stdout.strip()
            if pod_res.returncode != 0 or not pod_name:
                raise RuntimeError(f"finding running executor pod: {pod_res.stderr.strip()}")

            remote_vxpu = "/tmp/vxpu-hero"
            remote_plan = "/tmp/hero-opt-plan.pb"
            _copy_file_to_pod(pod_name, vxpu_bin, remote_vxpu, executable=True)
            _copy_file_to_pod(pod_name, opt_plan_file, remote_plan, executable=False)

            pkg_cmd = [
                "kubectl",
                "exec",
                pod_name,
                "--",
                remote_vxpu,
                "plan",
                "packages",
                "--plan",
                remote_plan,
                "--artifact",
                art_ref,
                "--model",
                model_id,
                "--modelstore",
                in_cluster_modelstore_url,
                "--build",
                "--agent",
                "127.0.0.1:50052",
            ]
            if agent_tls_dir:
                remote_tls_dir = "/tmp/hero-coord-tls"
                mk_res = subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        pod_name,
                        "--",
                        "sh",
                        "-c",
                        f"mkdir -p {remote_tls_dir} && chmod 0700 {remote_tls_dir}",
                    ],
                    capture_output=True,
                    text=True,
                )
                if mk_res.returncode != 0:
                    raise RuntimeError(f"creating {remote_tls_dir} in {pod_name}: {mk_res.stderr.strip()}")
                _copy_file_to_pod(
                    pod_name, os.path.join(agent_tls_dir, "ca.crt"), f"{remote_tls_dir}/ca.crt", mode="0444"
                )
                _copy_file_to_pod(
                    pod_name, os.path.join(agent_tls_dir, "tls.crt"), f"{remote_tls_dir}/tls.crt", mode="0400"
                )
                _copy_file_to_pod(
                    pod_name, os.path.join(agent_tls_dir, "tls.key"), f"{remote_tls_dir}/tls.key", mode="0400"
                )
                pkg_cmd.extend(["--tls-dir", remote_tls_dir])

        if machine_kind:
            pkg_cmd.append(f"--machine-kind={machine_kind}")
        else:
            pkg_cmd.append("--machine=c3d-standard-60")

        build_res = subprocess.run(pkg_cmd, capture_output=True, text=True)
        if build_res.returncode != 0:
            raise RuntimeError(
                f"building speed plan packages failed (exit {build_res.returncode}):\n"
                f"stdout: {build_res.stdout.strip()}\nstderr: {build_res.stderr.strip()}"
            )
        if build_res.stdout.strip():
            print(build_res.stdout.strip(), flush=True)
    finally:
        if pod_name and remote_tls_dir:
            subprocess.run(
                ["kubectl", "exec", pod_name, "--", "rm", "-rf", remote_tls_dir],
                capture_output=True,
            )


def main() -> None:
    args = parse_args()
    model_id: str = args.model
    device_label: str = args.machine_kind or "cpu"
    reports: list[StepReport] = []

    print(f"Starting hero journey for model {model_id!r} on {device_label}...", flush=True)

    # Locate vxpu CLI binary
    vxpu_bin: Optional[str] = args.vxpu_bin or shutil.which("vxpu")

    ref_tokens: Optional[torch.Tensor] = None
    expected_pt_identity: Optional[str] = None
    step2_tokens: Optional[torch.Tensor] = None
    step2_ms_per_token: Optional[float] = None
    step3_tokens: Optional[torch.Tensor] = None
    step3_ms_per_token: Optional[float] = None

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
    prompt_text = "Is the sky blue?"
    inputs: Any = tokenizer(prompt_text, return_tensors="pt")
    max_new_tokens = 10

    # -----------------------------------------------------------------
    # Step 1: Upload model through Python front and verify baseline
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    step1_passed = False
    step1_err = ""
    try:
        print("1. Upload: loading reference eager transformers model...", flush=True)
        ref_model: Any = TransformersModel.from_pretrained(model_id, dtype="auto").eval()
        ref_tokens = ref_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        print("1. Upload: loading model through vxpu.AutoModelForCausalLM.from_pretrained...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            router=args.router,
            artifact_dir=args.artifact_dir,
        )
        try:
            vxpu_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            if ref_tokens is not None and torch.equal(vxpu_out, ref_tokens):
                step1_passed = True
            else:
                ref_list: list[int] = ref_tokens.tolist() if ref_tokens is not None else []
                step1_err = f"token mismatch: vxpu={vxpu_out.tolist()} vs ref={ref_list}"
        finally:
            model.close()
    except Exception as e:
        step1_err = str(e)
    record_step(reports, "1. Upload", step1_passed, time.perf_counter() - t0, step1_err)

    # Resolve artifact directory for planner commands
    artifact_dir: str = args.artifact_dir or os.path.join(
        os.path.expanduser("~/.cache/vxpu/artifacts"),
        model_id.replace("/", "--"),
        "main",
        "2048",
    )

    plan_work_dir = tempfile.TemporaryDirectory()
    try:
        # -----------------------------------------------------------------
        # Step 2: Pass-through plan emitted by planner, verifying equivalence
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        step2_passed = False
        step2_err = ""
        pt_plan_file = os.path.join(plan_work_dir.name, "plan_passthrough.pb")
        if not vxpu_bin or not os.path.isfile(vxpu_bin) or not os.access(vxpu_bin, os.X_OK):
            step2_err = f"vxpu binary not found or not executable: {vxpu_bin}"
            reports.append(
                StepReport(
                    step="2. Pass-through",
                    result="failed",
                    missing=step2_err,
                    seconds=time.perf_counter() - t0,
                    error_detail=step2_err,
                )
            )
        else:
            print("2. Pass-through: emitting plan with planner...", flush=True)
            pt_emit_cmd = [vxpu_bin, "plan", "emit", "--artifact", artifact_dir, "--out", pt_plan_file]
            if args.machine_kind:
                pt_emit_cmd.append(f"--machine-kind={args.machine_kind}")
            if args.software_version:
                pt_emit_cmd.append(f"--tree-commit={args.software_version}")
            emit_res = subprocess.run(
                pt_emit_cmd,
                capture_output=True,
                text=True,
            )
            if emit_res.returncode != 0:
                step2_err = f"plan emit failed: {emit_res.stderr.strip()}"
                reports.append(
                    StepReport(
                        step="2. Pass-through",
                        result="failed",
                        missing=step2_err,
                        seconds=time.perf_counter() - t0,
                        error_detail=step2_err,
                    )
                )
            else:
                expected_pt_identity = get_plan_identity(pt_plan_file)
                if expected_pt_identity is None:
                    step2_err = "plan has no identity"
                else:
                    print(f"2. Pass-through: emitted plan identity {expected_pt_identity[:12]}", flush=True)

                # Step 2 passes only when the executor, through the router, reports
                # the identity of the plan it loaded and that equals the identity of
                # the plan emit produced.
                if expected_pt_identity is not None:
                    try:
                        model_pt = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            router=args.router,
                            artifact_dir=artifact_dir,
                        )
                        try:
                            pt_out = model_pt.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                            )
                            engine_plan_id = get_engine_plan_identity(model_pt)
                            if engine_plan_id is None:
                                step2_err = (
                                    f"engine did not report loaded plan identity "
                                    f"(expected {expected_pt_identity[:12]}, got None)"
                                )
                            elif engine_plan_id != expected_pt_identity:
                                step2_err = (
                                    f"engine reported plan identity {engine_plan_id[:12]}, "
                                    f"expected {expected_pt_identity[:12]}"
                                )
                            elif ref_tokens is not None and torch.equal(pt_out, ref_tokens):
                                step2_tokens = pt_out
                                step2_passed = True
                                if model_pt.last and hasattr(model_pt.last, "ms_per_token"):
                                    step2_ms_per_token = float(model_pt.last.ms_per_token)
                            else:
                                step2_err = "tokens differ from plain eager transformers"
                        finally:
                            model_pt.close()
                    except Exception as e:
                        step2_err = f"load with plan failed: {e}"

                record_step(reports, "2. Pass-through", step2_passed, time.perf_counter() - t0, step2_err)

        # -----------------------------------------------------------------
        # Step 3: Optimized plan emitted with optimizations for the machine
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        step3_passed = False
        step3_err = ""
        opt_plan_file = os.path.join(plan_work_dir.name, "plan_optimized.pb")
        if not vxpu_bin or not os.path.isfile(vxpu_bin) or not os.access(vxpu_bin, os.X_OK):
            step3_err = f"vxpu binary not found or not executable: {vxpu_bin}"
            reports.append(
                StepReport(
                    step="3. Optimized",
                    result="failed",
                    missing=step3_err,
                    seconds=time.perf_counter() - t0,
                    error_detail=step3_err,
                )
            )
        else:
            print("3. Optimized: emitting speed plan with planner...", flush=True)
            opt_emit_cmd = [
                vxpu_bin,
                "plan",
                "emit",
                "--artifact",
                artifact_dir,
                "--out",
                opt_plan_file,
                "--goal=speed",
            ]
            if args.machine_kind:
                opt_emit_cmd.append(f"--machine-kind={args.machine_kind}")
            else:
                opt_emit_cmd.append("--machine=c3d-standard-60")
            if args.software_version:
                opt_emit_cmd.append(f"--tree-commit={args.software_version}")
            emit_res = subprocess.run(
                opt_emit_cmd,
                capture_output=True,
                text=True,
            )
            if emit_res.returncode != 0:
                step3_err = f"plan emit failed: {emit_res.stderr.strip()}"
                reports.append(
                    StepReport(
                        step="3. Optimized",
                        result="failed",
                        missing=step3_err,
                        seconds=time.perf_counter() - t0,
                        error_detail=step3_err,
                    )
                )
            else:
                # Step 3 passes only if the plan differs from step 2's in a way it prints,
                # the engine reports that plan's identity, and the tokens equal step 2's.
                plans_differ = False
                diff_lines: list[str] = []
                if os.path.isfile(pt_plan_file):
                    plans_differ, diff_lines = diff_plans(pt_plan_file, opt_plan_file)

                if not plans_differ:
                    step3_err = "the planner chose nothing different"
                else:
                    print("3. Optimized: planner chose differences from pass-through plan:", flush=True)
                    for diff in diff_lines:
                        print(f"  {diff}", flush=True)

                    expected_opt_identity = get_plan_identity(opt_plan_file)
                    if expected_opt_identity is None:
                        step3_err = "plan has no identity"
                    elif expected_pt_identity is not None and expected_opt_identity == expected_pt_identity:
                        step3_err = (
                            f"speed plan identity {expected_opt_identity[:12]} equals pass-through plan identity"
                        )
                    elif step2_tokens is None:
                        step3_err = "step 2 did not produce tokens to compare against"
                    elif not args.modelstore:
                        step3_err = "--modelstore is required to build and stage speed plan packages"
                    else:
                        print(f"3. Optimized: emitted speed plan identity {expected_opt_identity[:12]}", flush=True)
                        try:
                            print("3. Optimized: building required .pt2 packages into modelstore...", flush=True)
                            build_speed_packages(
                                vxpu_bin=vxpu_bin,
                                artifact_dir=artifact_dir,
                                opt_plan_file=opt_plan_file,
                                model_id=model_id,
                                modelstore_url=args.modelstore,
                                in_cluster_modelstore_url=args.in_cluster_modelstore or args.modelstore,
                                machine_kind=args.machine_kind,
                                agent_addr=args.agent,
                                agent_tls_dir=args.agent_tls_dir,
                            )
                            print("3. Optimized: loading model with speed plan through vxpu front...", flush=True)
                            model_opt = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                router=args.router,
                                artifact_dir=artifact_dir,
                                plan=opt_plan_file,
                            )
                            try:
                                opt_out = model_opt.generate(
                                    **inputs,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                )
                                engine_plan_id = get_engine_plan_identity(model_opt)
                                if engine_plan_id is None:
                                    step3_err = (
                                        f"engine did not report loaded plan identity "
                                        f"(expected {expected_opt_identity[:12]}, got None)"
                                    )
                                elif engine_plan_id != expected_opt_identity:
                                    step3_err = (
                                        f"engine reported plan identity {engine_plan_id[:12]}, "
                                        f"expected {expected_opt_identity[:12]}"
                                    )
                                elif expected_pt_identity is not None and engine_plan_id == expected_pt_identity:
                                    step3_err = (
                                        f"engine reported pass-through plan identity {engine_plan_id[:12]} "
                                        "instead of speed plan"
                                    )
                                elif torch.equal(opt_out, step2_tokens):
                                    step3_tokens = opt_out
                                    step3_passed = True
                                    if model_opt.last and hasattr(model_opt.last, "ms_per_token"):
                                        step3_ms_per_token = float(model_opt.last.ms_per_token)
                                else:
                                    step3_err = (
                                        f"tokens differ from step 2: "
                                        f"opt={opt_out.tolist()} vs step2={step2_tokens.tolist()}"
                                    )
                            finally:
                                model_opt.close()
                        except Exception as e:
                            step3_err = f"load with optimized plan failed: {e}"

                record_step(reports, "3. Optimized", step3_passed, time.perf_counter() - t0, step3_err)
    finally:
        plan_work_dir.cleanup()

    # -----------------------------------------------------------------
    # Step 4: Comparison & Metrics
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    step4_passed = False
    step4_err = ""
    if step2_ms_per_token is not None and step3_ms_per_token is not None:
        speedup = (
            step2_ms_per_token / step3_ms_per_token
            if step3_ms_per_token > 0
            else 1.0
        )
        print(
            f"4. Comparison: Step 2: {step2_ms_per_token:.1f} ms/token, "
            f"Step 3: {step3_ms_per_token:.1f} ms/token ({speedup:.2f}x speedup)",
            flush=True,
        )
        step4_passed = True
    else:
        step4_err = "requires tokens and timings from steps 2 and 3"
    record_step(reports, "4. Comparison", step4_passed, time.perf_counter() - t0, step4_err)

    # Print summary table to job log
    table_str = format_table(reports)
    print("\n" + table_str + "\n", flush=True)

    # Write Markdown summary table to GITHUB_STEP_SUMMARY if available
    summary_path = args.summary_file or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            md_table = format_markdown_table(reports, model_id, device=device_label)
            with open(summary_path, "a") as f:
                f.write(md_table + "\n")
            print(f"Wrote job summary table to {summary_path}", flush=True)
        except Exception as e:
            print(f"Failed to write job summary to {summary_path}: {e}", file=sys.stderr)

    # Determine exit status
    has_failed = any(r.result == "failed" for r in reports)
    has_incomplete = any(r.result == "not yet possible" for r in reports)

    if has_failed:
        print("Hero journey failed with errors.", file=sys.stderr)
        sys.exit(1)
    if has_incomplete:
        print("Hero journey finished with incomplete steps (failing visibly).", file=sys.stderr)
        sys.exit(1)

    print("Hero journey completed successfully (all steps passed)!", flush=True)


if __name__ == "__main__":
    main()
