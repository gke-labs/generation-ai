# pyright: strict
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

"""Physical execution plans and per-site strategies (#529, #583).

This module translates a physical plan file's per-site strategy choices
into the engine's execution switches.
"""

from __future__ import annotations

import hashlib
import os
from typing import Mapping, Sequence
from google.protobuf import json_format

from . import plan_pb2

# Per-kind non-default strategies. Kinds absent here have no non-default strategies.
KNOWN_STRATEGIES: dict[int, frozenset[str]] = {
    plan_pb2.SITE_KIND_EXPERT_OP: frozenset({"fused", "per-expert", "mailbox"}),
}

SETTABLE_KNOBS: frozenset[str] = frozenset({
    "VXPU_FUSED_EXPERTS",
    "VXPU_EXPERT_MAILBOXES",
})


def read_plan_file(path: str | bytes | os.PathLike[str]) -> plan_pb2.PlanFile:
    """Reads and parses a PlanFile protobuf message from JSON on disk."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    if not os.path.exists(path):
        raise FileNotFoundError(f"plan file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pf = plan_pb2.PlanFile()
    try:
        json_format.Parse(text, pf)
    except json_format.ParseError as e:
        raise ValueError(f"invalid plan JSON: {e}") from e
    return pf


def write_plan_file(
    path: str | bytes | os.PathLike[str], plan: plan_pb2.PlanFile
) -> None:
    """Writes a PlanFile protobuf message as JSON to disk atomically.

    When recording a run, replaces the file so readers see the latest
    completed call. Writes through a temporary file and os.replace so readers
    never see a partial document.
    """
    str_path: str = os.fspath(path) if not isinstance(path, bytes) else path.decode("utf-8")
    parent = os.path.dirname(str_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = json_format.MessageToJson(
        plan, indent=2, preserving_proto_field_name=True, sort_keys=True
    )
    tmp_path = f"{str_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, str_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def plan_hash(path: str | bytes | os.PathLike[str]) -> str:
    """Computes the SHA-256 hash of a physical execution plan file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_artifact_md5(artifact_dir: str | bytes | os.PathLike[str]) -> str:
    """Computes the MD5 checksum over manifest.json and binding.json bytes."""
    path: str = os.fspath(artifact_dir) if not isinstance(artifact_dir, bytes) else artifact_dir.decode("utf-8")
    m = hashlib.md5()
    for name in ("manifest.json", "binding.json"):
        with open(os.path.join(path, name), "rb") as f:
            m.update(f.read())
    return m.hexdigest()


def fill_run_record(
    plan: plan_pb2.PlanFile | None = None,
    *,
    plan_hash: str | None = None,
    tree_commit: str | None = None,
    artifact_md5: str | None = None,
    knobs: Mapping[str, int | bool | None] | None = None,
    median_step_ms: float | None = None,
    tokens_a_stream_a_step: float | None = None,
    accepted_drafts_a_step: float | None = None,
    checks: Mapping[str, bool] | None = None,
    per_card_peak_memory: Sequence[int] | None = None,
    per_class_device_times: Mapping[str, float] | None = None,
    site_measurements: Mapping[str, float] | None = None,
) -> plan_pb2.PlanFile:
    """Fills a PlanFile protobuf message from executor run measurements."""
    record = plan_pb2.PlanFile()
    if plan is not None:
        record.CopyFrom(plan)

    # When given an already-measured plan, clear previous run measurements
    # so earlier runs do not leak stale actuals into the new record.
    record.ClearField("stats")
    for site in record.sites:
        site.ClearField("stats")

    if tree_commit:
        record.tree_commit = tree_commit
    if artifact_md5:
        record.artifact_md5 = artifact_md5

    stats = plan_pb2.PlanExecutionStats()
    has_stats = False

    if plan_hash:
        stats.plan_hash = plan_hash
        has_stats = True

    if knobs:
        knobs_msg = plan_pb2.Knobs()
        has_knob = False
        for name, val in knobs.items():
            if val is not None:
                setattr(knobs_msg, name, val)
                has_knob = True
        if has_knob:
            stats.knobs.CopyFrom(knobs_msg)
            has_stats = True

    if median_step_ms is not None:
        stats.median_step_ms = median_step_ms
        has_stats = True
    if tokens_a_stream_a_step is not None:
        stats.tokens_a_stream_a_step = tokens_a_stream_a_step
        has_stats = True
    if accepted_drafts_a_step is not None:
        stats.accepted_drafts_a_step = accepted_drafts_a_step
        has_stats = True

    if checks:
        for k, v in checks.items():
            stats.checks[str(k)] = bool(v)
        has_stats = True

    if per_card_peak_memory:
        stats.per_card_peak_memory.extend(int(p) for p in per_card_peak_memory)
        has_stats = True

    if per_class_device_times:
        for k, v in per_class_device_times.items():
            stats.per_class_device_times[str(k)] = float(v)
        has_stats = True

    if has_stats:
        record.stats.CopyFrom(stats)

    if site_measurements:
        for site in record.sites:
            if site.id in site_measurements:
                site.stats.measured_us = float(site_measurements[site.id])

    return record


def _set_knob(switches: dict[str, str], name: str, value: str, sid: str) -> None:
    if name not in SETTABLE_KNOBS:
        raise ValueError(f"knob {name} has no reader in the tree")
    if name in switches and switches[name] != value:
        raise ValueError(
            f"conflicting values for knob {name} from site {sid!r}: {switches[name]!r} vs {value!r}"
        )
    switches[name] = value


def _kind_name(kind: int) -> str:
    if kind in plan_pb2.SiteKind.values():
        return plan_pb2.SiteKind.Name(kind)
    return str(kind)


def apply_env(plan: plan_pb2.PlanFile) -> dict[str, str]:
    """Translates per-site strategy choices in a PlanFile into engine switches.

    Validates each site's kind and strategy. A non-default strategy for a
    kind without strategies, or an unknown strategy, raises ValueError.
    Returns a dictionary of {switch_name: value} to merge into plan_given.
    A value the coordinator also set that differs is a ConfigError in the engine.
    """
    switches: dict[str, str] = {}

    for site in plan.sites:
        kind = site.kind
        strat = site.strategy or "default"

        if kind == plan_pb2.SITE_KIND_UNSPECIFIED or kind not in plan_pb2.SiteKind.values():
            raise ValueError(f"unknown site kind {_kind_name(kind)!r} for site {site.id!r}")

        if strat == "default":
            continue

        valid_strategies = KNOWN_STRATEGIES.get(kind, frozenset())
        if strat not in valid_strategies:
            raise ValueError(f"unknown strategy {strat!r} for site {site.id!r} (kind {_kind_name(kind)!r})")

        if kind == plan_pb2.SITE_KIND_EXPERT_OP:
            if strat == "fused":
                _set_knob(switches, "VXPU_FUSED_EXPERTS", "1", site.id)
            elif strat == "per-expert":
                _set_knob(switches, "VXPU_FUSED_EXPERTS", "0", site.id)
            elif strat == "mailbox":
                _set_knob(switches, "VXPU_EXPERT_MAILBOXES", "1", site.id)

    return switches
