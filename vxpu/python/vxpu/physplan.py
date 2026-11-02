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

"""The physical plan file (`vxpu-plan/1`, pkg/kernelplan/pipeline.go): the layer ranges per
GPU, the pipeline of batches over them and the kernel table to apply. `VXPU_PLAN=path` hands
the engine one; the engine reads it before its Plan reads the environment and renders it as
the switches the Plan knows (VXPU_LAYER_DEVICES, VXPU_PLAN_TABLE), so the plan file and the
switches are one mechanism with the file in front. The batch count is for the caller
(default_batches: Engine.generate_rows(batches=...)); the speculation section (the draft length
and accept policy, hero run 10) is for the drafter path (speculation_policy). An expert placement other than the
placement file given in VXPU_GPU_EXPERTS is not rendered yet: the engine needs the experts per
layer to build one, and "with-layers" (every expert of a range's layers on the range's GPU) is
what the hand file placement-ranges4c.json holds for DeepSeek-V4.1-Flash. The parallel section
(hero run 14: the dense layers split across the cards of a group and the experts spread over
them) is rendered as VXPU_DENSE_TP, VXPU_EXPERT_EP, VXPU_TP_GPUS ("0,1,2,3"), VXPU_EP_EXPERTS
("by-id", "by-popularity" or a placement file) and VXPU_REDUCE ("one-shot", "ring" or empty for
the engine's choice by payload); the engine's tensor-parallel path reads those (parallel_section
returns the section as given)."""

import json
import os

from vxpu import speculation

FORMAT = "vxpu-plan/1"
_loaded = None


def load(path):
    """The plan file as a dict, checked: format, ranges ("a:b" per GPU), pipeline.batches."""
    try:
        with open(path) as f:
            plan = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"VXPU_PLAN={path!r}: cannot read it: {e}") from None
    if not isinstance(plan, dict) or plan.get("format") != FORMAT:
        raise ValueError(f"VXPU_PLAN={path!r}: expected a JSON object with format {FORMAT!r}")
    ranges = plan.get("ranges") or []
    for r in ranges:
        layers = str(r.get("layers", ""))
        if layers.count(":") != 1 or not all(p.isdigit() for p in layers.split(":")) or "gpu" not in r:
            raise ValueError(f"VXPU_PLAN={path!r}: a range needs a gpu and layers 'first:end', got {r!r}")
    batches = (plan.get("pipeline") or {}).get("batches", 1)
    if not isinstance(batches, int) or batches < 1:
        raise ValueError(f"VXPU_PLAN={path!r}: pipeline.batches must be a positive integer")
    drafter = plan.get("drafter")
    if drafter is not None:
        gpu = drafter.get("gpu") if isinstance(drafter, dict) else None
        if not isinstance(gpu, int) or gpu < 0 or (ranges and gpu not in {int(r["gpu"]) for r in ranges}):
            raise ValueError(f"VXPU_PLAN={path!r}: drafter.gpu must name a GPU that holds a range, got {drafter!r}")
    try:
        speculation.Policy.from_plan(plan.get("speculation"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"VXPU_PLAN={path!r}: {e}") from None
    check_parallel(plan.get("parallel"), ranges, path)
    return plan


def check_parallel(section, ranges, path="<plan>"):
    """The parallel section as pkg/kernelplan validates it: dense_tp 1, 2 or 4; expert_ep 1 or dense_tp;
    with dense_tp above 1 the group's gpus (distinct, dense_tp of them) and one range, the whole step,
    on the group's first card; experts "", "by-id", "by-popularity" or a placement file; reduce "",
    "one-shot" or "ring"."""
    if section is None:
        return
    if not isinstance(section, dict):
        raise ValueError(f"VXPU_PLAN={path!r}: parallel must be an object")
    tp = section.get("dense_tp", 1)
    ep = section.get("expert_ep", 1)
    if tp not in (1, 2, 4):
        raise ValueError(f"VXPU_PLAN={path!r}: parallel dense_tp {tp!r}: want 1, 2 or 4")
    if ep not in (1, tp):
        raise ValueError(f"VXPU_PLAN={path!r}: parallel expert_ep {ep!r}: want 1 or dense_tp ({tp})")
    gpus = section.get("gpus") or []
    if tp > 1:
        if len(gpus) != tp or len(set(gpus)) != tp or not all(isinstance(g, int) and g >= 0 for g in gpus):
            raise ValueError(f"VXPU_PLAN={path!r}: parallel gpus {gpus!r}: want the {tp} distinct cards of the group")
        if len(ranges) != 1:
            raise ValueError(f"VXPU_PLAN={path!r}: parallel dense_tp {tp} with {len(ranges)} ranges: a tensor-parallel plan has one range")
        if int(ranges[0].get("gpu", -1)) != gpus[0]:
            raise ValueError(f"VXPU_PLAN={path!r}: the range's gpu must be the group's first card {gpus[0]}")
    experts = section.get("experts", "")
    if experts not in ("", "by-id", "by-popularity") and not str(experts).endswith(".json"):
        raise ValueError(f"VXPU_PLAN={path!r}: parallel experts {experts!r}: want by-id, by-popularity or a placement file")
    if section.get("reduce", "") not in ("", "one-shot", "ring"):
        raise ValueError(f"VXPU_PLAN={path!r}: parallel reduce {section.get('reduce')!r}: want one-shot or ring")


def parallel_env(section):
    """The switches the parallel section renders to (empty without one or with dense_tp 1)."""
    if not section or section.get("dense_tp", 1) <= 1:
        return {}
    return {
        "VXPU_DENSE_TP": str(section["dense_tp"]),
        "VXPU_EXPERT_EP": str(section.get("expert_ep", 1)),
        "VXPU_TP_GPUS": ",".join(str(g) for g in section.get("gpus") or []),
        "VXPU_EP_EXPERTS": section.get("experts") or "by-id",
        "VXPU_REDUCE": section.get("reduce") or "",
    }


def layer_devices(plan):
    """The plan's ranges as the VXPU_LAYER_DEVICES string: '0:8=cuda:0,8:20=cuda:1,...'."""
    return ",".join(f"{r['layers']}=cuda:{int(r['gpu'])}" for r in plan.get("ranges") or [])


def apply_env():
    """With VXPU_PLAN set: read the plan once and render it into the switches the engine's Plan
    reads, unless a switch is already set by hand (the hand switch wins and is reported). Returns
    the plan, or None without VXPU_PLAN."""
    global _loaded
    path = os.environ.get("VXPU_PLAN")
    if not path:
        return None
    if _loaded is None:
        plan = load(path)
        notes = []
        devices = layer_devices(plan)
        if devices:
            if os.environ.get("VXPU_LAYER_DEVICES") and os.environ["VXPU_LAYER_DEVICES"] != devices:
                notes.append(f"VXPU_LAYER_DEVICES set by hand ({os.environ['VXPU_LAYER_DEVICES']}) overrides the plan's {devices}")
            else:
                os.environ["VXPU_LAYER_DEVICES"] = devices
        drafter = plan.get("drafter")
        if drafter:  # the drafter's stages and their experts on this GPU, whatever range covers their layer ids
            device = f"cuda:{int(drafter['gpu'])}"
            if os.environ.get("VXPU_DRAFT_DEVICE") and os.environ["VXPU_DRAFT_DEVICE"] != device:
                notes.append(f"VXPU_DRAFT_DEVICE set by hand ({os.environ['VXPU_DRAFT_DEVICE']}) overrides the plan's {device}")
            else:
                os.environ["VXPU_DRAFT_DEVICE"] = device
        kernels = plan.get("kernels")
        if kernels:
            table = kernels if isinstance(kernels, str) else json.dumps(kernels)
            if os.environ.get("VXPU_PLAN_TABLE") and os.environ["VXPU_PLAN_TABLE"] != table:
                notes.append("VXPU_PLAN_TABLE set by hand overrides the plan's kernel table")
            else:
                os.environ["VXPU_PLAN_TABLE"] = table
        experts = plan.get("experts")
        if experts and experts != "with-layers" and not os.environ.get("VXPU_GPU_EXPERTS"):
            os.environ["VXPU_GPU_EXPERTS"] = experts  # a placement file
        for name, value in parallel_env(plan.get("parallel")).items():
            if os.environ.get(name) and os.environ[name] != value:
                notes.append(f"{name} set by hand ({os.environ[name]}) overrides the plan's {value}")
            else:
                os.environ[name] = value
        print(f"[vxpu] plan {path}: {len(plan.get('ranges') or [])} ranges, "
              f"{(plan.get('pipeline') or {}).get('batches', 1)} batches in flight"
              + (f"; {'; '.join(notes)}" if notes else ""), flush=True)
        _loaded = plan
    return _loaded


def speculation_policy():
    """The plan's speculation section as a speculation.Policy (no drafts without a plan or a
    section). The engine's drafter path applies it through speculation.draft_length,
    plan_drafts and accept; the export's step tokens bound the draft."""
    plan = apply_env()
    return speculation.Policy.from_plan(plan.get("speculation") if plan else None)


def speculation_section():
    """The plan's speculation section as given, or None when the plan has none (an engine whose
    artifact has a drafter then issues every draft, as before plans named a policy)."""
    plan = apply_env()
    return (plan or {}).get("speculation")


def parallel_section():
    """The plan's parallel section as given, or None (each range whole on its GPU)."""
    plan = apply_env()
    return (plan or {}).get("parallel")


def default_batches():
    """The plan's pipeline.batches, or 1 without a plan."""
    plan = apply_env()
    return int((plan.get("pipeline") or {}).get("batches", 1)) if plan else 1
