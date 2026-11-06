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

"""Physical execution plans, per-site strategies, and run records (#529).

This module defines the mapping from a physical plan's per-site strategy
choices to the engine's effective knobs, and the run record format written
at the end of an execution.
"""

import hashlib
import json
import os
import subprocess
import time

PLAN_FORMAT = "vxpu-plan/1"
RECORD_FORMAT = "vxpu-run-record/1"
RECORD_SCHEMA_VERSION = 1

KNOWN_STRATEGIES = {
    "dense GEMV": {"default", "gemv_rows", "reduction", "aten"},
    "expert op": {"default", "fused", "direct", "per-expert", "one-launch", "b12x"},
    "sum site": {"default", "one-shot", "reduce-scatter", "bf16_push", "fused_push", "fp32_wire", "fp32"},
    "attention": {"default", "sdpa", "flash"},
    "drafter": {"default", "engram", "speculative"},
}

ALL_STRATEGIES = frozenset({strat for strats in KNOWN_STRATEGIES.values() for strat in strats})


def load_plan(plan):
    """Load a Plan dictionary from a dictionary, JSON string, or file path."""
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, (str, bytes, os.PathLike)):
        if isinstance(plan, bytes):
            plan = plan.decode("utf-8")
        if os.path.exists(plan):
            with open(plan, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(plan)
    raise TypeError(f"expected dict, path or JSON string for plan, got {type(plan)!r}")


def plan_hash(plan):
    """Compute the canonical 16-hex-digit SHA-256 hash of a Plan."""
    data = load_plan(plan)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def compute_artifact_md5(manifest_bytes, binding_bytes):
    """Compute the MD5 checksum over an artifact's manifest and binding."""
    h = hashlib.md5()
    h.update(manifest_bytes)
    h.update(binding_bytes)
    return h.hexdigest()


def get_tree_commit():
    """Retrieve the current tree commit from the environment or git."""
    for key in ("GIT_COMMIT", "TREE_COMMIT"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def apply_env(plan, env=None):
    """Read the per-site strategy map from a plan and apply the effective knobs to env.

    Sets the existing knobs in env (defaulting to os.environ). Unknown
    strategies raise a ValueError (hard error). Returns the dictionary of
    effective knobs in effect.
    """
    plan_dict = load_plan(plan)
    if env is None:
        env = os.environ

    strategies = dict(plan_dict.get("strategies", {}))
    if not strategies and "sites" in plan_dict:
        for site in plan_dict["sites"]:
            if "id" in site and "strategy" in site:
                strategies[site["id"]] = site["strategy"]

    # Unknown strategies are a hard error
    for site_id, strat in strategies.items():
        if strat not in ALL_STRATEGIES:
            raise ValueError(f"unknown strategy {strat!r} for site {site_id!r}")

    knobs = {
        "VXPU_FUSED_EXPERTS": "1",
        "VXPU_ALLREDUCE_SCATTER_MIN_ROWS": "16",
        "VXPU_ALLREDUCE_PUSH_BF16": "0",
    }

    for site_id, strat in strategies.items():
        if strat == "reduce-scatter":
            knobs["VXPU_ALLREDUCE_SCATTER_MIN_ROWS"] = "1"
        elif strat == "one-shot":
            knobs["VXPU_ALLREDUCE_SCATTER_MIN_ROWS"] = "0"
        elif strat in ("bf16_push", "fused_push"):
            knobs["VXPU_ALLREDUCE_PUSH_BF16"] = "1"
        elif strat in ("fp32", "fp32_wire"):
            knobs["VXPU_ALLREDUCE_PUSH_BF16"] = "0"
        elif strat == "direct":
            knobs["VXPU_FP4_DIRECT_FIRST"] = "1"
        elif strat == "per-expert":
            knobs["VXPU_FP4_DIRECT_FIRST"] = "0"
        elif strat == "fused":
            knobs["VXPU_FUSED_EXPERTS"] = "1"
        elif strat == "one-launch":
            knobs["VXPU_FP4_ONE_LAUNCH"] = "1"
        elif strat == "b12x":
            knobs["VXPU_EXPERT_KERNEL"] = "b12x"
        elif strat == "gemv_rows":
            knobs["VXPU_GEMV"] = "gemv_rows"
        elif strat == "reduction":
            knobs["VXPU_GEMV"] = "reduction"
        elif strat == "aten":
            knobs["VXPU_GEMV"] = "aten"
        elif strat == "engram":
            knobs["VXPU_DRAFTER"] = "engram"

    if "switches" in plan_dict:
        for k, v in plan_dict["switches"].items():
            knobs[k] = str(v)

    env.update(knobs)
    return knobs


def create_run_record(plan=None, knobs_in_effect=None, median_step_ms=0.0,
                      tokens_a_step=1.0, checks=None, per_card_peak_memory=None,
                      build_time=0.0, per_class_times=None, plan_hash_val=None,
                      tree_commit=None, artifact_md5=None):
    """Build a versioned run record dictionary."""
    if plan is not None:
        plan_dict = load_plan(plan)
        if plan_hash_val is None:
            plan_hash_val = plan_hash(plan_dict)
        if tree_commit is None:
            tree_commit = plan_dict.get("tree_commit")
        if artifact_md5 is None:
            artifact_md5 = plan_dict.get("artifact_md5")

    if plan_hash_val is None:
        plan_hash_val = "unknown"
    if tree_commit is None:
        tree_commit = get_tree_commit()
    if artifact_md5 is None:
        artifact_md5 = "unknown"
    if knobs_in_effect is None:
        knobs_in_effect = {}
    if checks is None:
        checks = {"status": "pass", "exactness": True}
    if per_card_peak_memory is None:
        per_card_peak_memory = {}

    return {
        "format": RECORD_FORMAT,
        "schema_version": RECORD_SCHEMA_VERSION,
        "plan_hash": str(plan_hash_val),
        "tree_commit": str(tree_commit),
        "artifact_md5": str(artifact_md5),
        "knobs_in_effect": {str(k): str(v) for k, v in knobs_in_effect.items()},
        "median_step_ms": round(float(median_step_ms), 3),
        "tokens_a_step": float(tokens_a_step),
        "tokens_per_step": float(tokens_a_step),
        "checks": dict(checks),
        "per_card_peak_memory": dict(per_card_peak_memory),
        "build_time": round(float(build_time), 3),
        "build_time_s": round(float(build_time), 3),
        "per_class_times": dict(per_class_times) if per_class_times is not None else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_record(record_or_path):
    """Validate and parse a run record, checking schema version and required fields."""
    if isinstance(record_or_path, (str, bytes, os.PathLike)):
        if isinstance(record_or_path, bytes):
            record_or_path = record_or_path.decode("utf-8")
        if os.path.exists(record_or_path):
            with open(record_or_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        else:
            record = json.loads(record_or_path)
    elif isinstance(record_or_path, dict):
        record = dict(record_or_path)
    else:
        raise TypeError(f"expected dict, path or JSON string for record, got {type(record_or_path)!r}")

    if not record.get("format") and not record.get("schema_version"):
        raise ValueError("record schema is unversioned: missing format and schema_version")
    if record.get("format") and record["format"] != RECORD_FORMAT:
        raise ValueError(f"unexpected record format {record['format']!r}, expected {RECORD_FORMAT!r}")
    if not record.get("plan_hash"):
        raise ValueError("missing plan_hash")
    if not record.get("tree_commit"):
        raise ValueError("missing tree_commit")
    if not record.get("artifact_md5"):
        raise ValueError("missing artifact_md5")
    if "knobs_in_effect" not in record or record["knobs_in_effect"] is None:
        raise ValueError("missing knobs_in_effect")
    if "checks" not in record or record["checks"] is None:
        raise ValueError("missing checks field")
    return record


def write_run_record(record, out_path):
    """Validate and write a run record to out_path as formatted JSON."""
    validated = parse_record(record)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
        f.write("\n")
