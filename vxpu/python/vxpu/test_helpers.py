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

"""Test utilities for creating plans with the Go planner in tests."""

import os
import shutil
import subprocess
import torch
from . import lower, plan


def emit_test_plan(artifact: str, device: str = "cpu", out: str | None = None, table: str | None = None) -> str:
    """Emits a physical plan for an artifact in test setup using vxpu plan emit."""
    plan_path = out or os.path.join(artifact, f"plan{'_table' if table else ''}.pb")
    if os.path.exists(plan_path):
        return plan_path

    vxpu_bin = shutil.which("vxpu")
    if not vxpu_bin:
        raise RuntimeError("vxpu binary not found on PATH; build vxpu before running tests")

    decode_path = os.path.join(artifact, "decode.pt2")
    ep = torch.export.load(decode_path)
    rec = lower.describe_sites(ep.module())
    cat = plan.build_catalog()

    sites_path = os.path.join(artifact, "sites.pb")
    cat_path = os.path.join(artifact, "catalog.pb")
    plan.write_kernel_record(sites_path, rec)
    plan.write_catalog(cat_path, cat)

    cmd = [
        vxpu_bin,
        "plan",
        "emit",
        "--sites",
        sites_path,
        "--catalog",
        cat_path,
        "--device",
        device,
        "--tree-commit",
        "0123456789abcdef0123456789abcdef01234567",
        "--out",
        plan_path,
    ]
    if table:
        cmd.extend(["--table", table])
    subprocess.run(cmd, check=True)
    return plan_path
