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

"""Compiles native packages as a task apart from an engine.

The planner determines which compiled packages are needed on a machine kind.
Given an artifact, a plan, and the packages named by the planner, this task
compiles exactly those packages and saves each to <out_dir>/<key>.pt2.
"""

from collections.abc import Sequence
import argparse
import os
import sys
import time
from typing import Any, TypedDict

import torch

from . import aoti, engine, plan as plan_mod, plan_pb2, segments


class CompiledPackage(TypedDict):
    """The outcome of compiling one package."""
    name: str
    key: str
    path: str
    bytes: int


def read_package_report(path: str) -> plan_pb2.PackageReport:
    """Reads a PackageReport from a JSON or protobuf file."""
    report = plan_pb2.PackageReport()
    if path.endswith(".json"):
        from google.protobuf import json_format

        with open(path, "r", encoding="utf-8") as f:
            json_format.Parse(f.read(), report)
    else:
        with open(path, "rb") as f:
            report.ParseFromString(f.read())
    return report


def write_package_report(path: str, report: plan_pb2.PackageReport) -> None:
    """Writes a PackageReport to a JSON or protobuf file."""
    if path.endswith(".json"):
        from google.protobuf import json_format

        with open(path, "w", encoding="utf-8") as f:
            f.write(json_format.MessageToJson(report, indent=2))
    else:
        with open(path, "wb") as f:
            f.write(report.SerializeToString())


def compile_packages(
    artifact_dir: str,
    out_dir: str,
    packages: Sequence[plan_pb2.Package],
    plan: plan_mod.Plan,
    machine_kind: str,
    software_version: str,
    device: str = "cpu",
    parallel_builds: int | None = None,
) -> tuple[list[CompiledPackage], list[plan_pb2.CompileFailure]]:
    """Compiles the named packages from artifact_dir into out_dir.

    The task refuses packages it was not given enough to build:
    - unknown segments or host op segments
    - empty package names or keys
    - empty machine_kind or software_version
    """
    if not machine_kind or not machine_kind.strip():
        raise ValueError("machine_kind is required")
    if not software_version or not software_version.strip():
        raise ValueError("software_version is required")
    machine_kind = machine_kind.strip()
    software_version = software_version.strip()

    if not packages:
        raise ValueError("no packages specified to compile")

    pkg_map: dict[str, str] = {}
    for pkg in packages:
        name = pkg.name.strip()
        key = pkg.key.strip()
        if not name:
            raise ValueError("package has empty segment name")
        if not key:
            raise ValueError(f"package for segment {name!r} has empty key")
        pkg_map[name] = key

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gm = engine.load_dense_decode_artifact(artifact_dir, plan, device=device)
    split, host = segments.decode_segments(gm, plan, device=device)
    available_segments = dict(segments.segments_of(split))

    tasks: list[tuple[str, torch.fx.GraphModule, tuple[Any, ...]]] = []
    for name in pkg_map:
        if name not in available_segments:
            raise ValueError(f"cannot compile package: unknown segment {name!r}")
        if name in host:
            raise ValueError(f"cannot compile package: segment {name!r} is a host op")
        sub = available_segments[name]
        assert isinstance(sub, torch.fx.GraphModule)
        tasks.append((name, sub, aoti.example_args(split, name, sub, device)))

    pkg_cache = aoti.Packages(out_dir, parallel_builds=parallel_builds, packages=pkg_map)
    pkg_cache.build_all(tasks, allow_failures=True)

    results: list[CompiledPackage] = []
    failures: list[plan_pb2.CompileFailure] = []

    for name, key in pkg_map.items():
        if name in pkg_cache.failures:
            failures.append(
                plan_pb2.CompileFailure(
                    package=plan_pb2.Package(name=name, key=key),
                    machine_kind=machine_kind,
                    software_version=software_version,
                    reason=pkg_cache.failures[name],
                )
            )
            continue
        pt2_path = os.path.join(out_dir, f"{key}.pt2")
        if not os.path.exists(pt2_path):
            raise RuntimeError(f"expected package {pt2_path} was not built")
        size = os.path.getsize(pt2_path)
        results.append(CompiledPackage(name=name, key=key, path=pt2_path, bytes=size))
    return results, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile native packages for a model from its artifact and plan.",
    )
    parser.add_argument("--artifact", required=True, help="path to artifact directory")
    parser.add_argument("--plan", required=True, help="path to plan.pb file")
    parser.add_argument("--out", required=True, help="output directory for compiled .pt2 packages")
    parser.add_argument("--device", default="cpu", help="compilation device (e.g. cpu, cuda:0)")
    parser.add_argument("--machine-kind", required=True, help="machine kind compilation is targeting")
    parser.add_argument("--software-version", required=True, help="software version of the executor")
    parser.add_argument("--packages-file", help="path to package report file (.json or .pb)")
    parser.add_argument("--parallel-builds", type=int, default=None, help="maximum parallel compilation workers")
    parser.add_argument("--out-report", help="path to write output package report (.json or .pb)")
    parser.add_argument("packages", nargs="*", help="packages to compile as name=key pairs")

    args = parser.parse_args()

    pkgs: list[plan_pb2.Package] = []
    if args.packages_file:
        report = read_package_report(args.packages_file)
        pkgs.extend(report.missing)
    for arg in args.packages:
        name, key = aoti.parse_package_arg(arg)
        pkgs.append(plan_pb2.Package(name=name, key=key))

    if not pkgs:
        print("[vxpu] error: no packages specified (use --packages-file or name=key arguments)", file=sys.stderr)
        sys.exit(1)

    from . import physplan

    pf = physplan.read_plan_file(args.plan)
    plan_given = physplan.apply_env(pf)
    plan = plan_mod.Plan(args.device, given=plan_given)

    machine_kind = args.machine_kind
    software_version = args.software_version

    t0 = time.perf_counter()
    results, failures = compile_packages(
        args.artifact,
        args.out,
        pkgs,
        plan,
        machine_kind=machine_kind,
        software_version=software_version,
        device=args.device,
        parallel_builds=args.parallel_builds,
    )
    elapsed = time.perf_counter() - t0
    for r in results:
        print(f"[vxpu] compiled package {r['name']} -> {r['key']}.pt2 ({r['bytes']} bytes)")
    for f in failures:
        print(f"[vxpu] failed to compile package {f.package.name} ({f.package.key}): {f.reason}")
    print(f"[vxpu] compiled {len(results)} package(s), {len(failures)} failed in {elapsed:.1f} s into {args.out}", flush=True)

    if args.out_report:
        out_report = plan_pb2.PackageReport(
            required=pkgs,
            present=[plan_pb2.Package(name=r["name"], key=r["key"]) for r in results],
            missing=[],
            failed=failures,
            machine_kind=machine_kind,
        )
        write_package_report(args.out_report, out_report)


if __name__ == "__main__":
    main()
