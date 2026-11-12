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

import os
from google.protobuf import json_format

from . import plan_pb2

# Call site kinds emitted by pkg/kernelplan (dense GEMV, expert op, sum site, attention, drafter).
KNOWN_CLASSES: frozenset[str] = frozenset({
    "dense GEMV",
    "expert op",
    "sum site",
    "attention",
    "drafter",
})

# Per-kind non-default strategies. Kinds absent here have no non-default strategies.
KNOWN_STRATEGIES: dict[str, frozenset[str]] = {
    "expert op": frozenset({"fused", "per-expert", "mailbox"}),
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


def _set_knob(switches: dict[str, str], name: str, value: str, sid: str) -> None:
    if name not in SETTABLE_KNOBS:
        raise ValueError(f"knob {name} has no reader in the tree")
    if name in switches and switches[name] != value:
        raise ValueError(
            f"conflicting values for knob {name} from site {sid!r}: {switches[name]!r} vs {value!r}"
        )
    switches[name] = value


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

        if kind not in KNOWN_CLASSES:
            raise ValueError(f"unknown site kind {kind!r} for site {site.id!r}")

        if strat == "default":
            continue

        valid_strategies = KNOWN_STRATEGIES.get(kind, frozenset())
        if strat not in valid_strategies:
            raise ValueError(f"unknown strategy {strat!r} for site {site.id!r} (kind {kind!r})")

        if kind == "expert op":
            if strat == "fused":
                _set_knob(switches, "VXPU_FUSED_EXPERTS", "1", site.id)
            elif strat == "per-expert":
                _set_knob(switches, "VXPU_FUSED_EXPERTS", "0", site.id)
            elif strat == "mailbox":
                _set_knob(switches, "VXPU_EXPERT_MAILBOXES", "1", site.id)

    return switches
