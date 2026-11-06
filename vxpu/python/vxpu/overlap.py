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

"""Measure real concurrency and overlap per run (busy, covered, wall) from in-graph device stamps.

The phase trace (VXPU_PHASE_TRACE) stamps each node on its stream before and after execution with
the GPU's global timer (%globaltimer). This module summarizes:
  - busy: sum of node spans
  - covered: length of the union of node spans
  - wall: elapsed time from the start of the first node to the end of the last node
  - idle: wall - covered (gaps)
  - overlap: busy - covered (concurrency)
  - per-class breakdown: busy, covered, and side-branch share
  - hidden nodes: nodes whose span lies entirely inside another branch's span
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def classify_node(name: str, kind: str, ops: Optional[List[str]] = None) -> str:
    """Classify a step graph node into a timeline class matching hero 15's timelines."""
    ops = ops or []
    ops_lower = [str(op).lower() for op in ops]
    name_lower = name.lower()

    if kind == "experts" or ("expert" in name_lower and "shared" not in name_lower):
        return "routed_experts"

    if kind in ("allreduce", "allgather", "collective") or any("allreduce" in op for op in ops_lower) or name_lower.startswith("allreduce"):
        return "sums"

    if any(op in ("shared_ffn_front", "shared_ffn_mid", "shared_ffn", "shared_ffn_combine") for op in ops_lower) or "shared_ffn" in name_lower or "shared_expert" in name_lower:
        return "shared_expert"

    if any(op in ("attn_q", "window_kv", "attn_qkv") for op in ops_lower) or name_lower.startswith("attn_q") or name_lower.startswith("window_kv"):
        return "attention_projections"

    if any(op in ("attn_core", "attn_select", "sparse_attn", "attn_out", "rotary") for op in ops_lower) or name_lower.startswith("attn_core") or name_lower.startswith("attn_out"):
        return "attention_core"

    if any(op in ("ffn_front", "ffn_front_split") for op in ops_lower) or name_lower.startswith("ffn_front"):
        return "ffn_front"

    if any(op in ("hc_front", "hc_front_split", "hc_sinkhorn") for op in ops_lower) or name_lower.startswith("hc_front"):
        return "hc_front"

    if kind == "engram" or any("engram" in op or "wkv" in op for op in ops_lower):
        return "engram"

    if kind == "embed_rows" or any("embed" in op for op in ops_lower):
        return "embed"

    return "dense_other"


def union_interval_length(intervals: List[Tuple[int, int]]) -> int:
    """Calculate the total length of the union of integer intervals [start, end]."""
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return 0
    valid.sort(key=lambda x: (x[0], x[1]))
    merged = []
    curr_s, curr_e = valid[0]
    for s, e in valid[1:]:
        if s <= curr_e:
            curr_e = max(curr_e, e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = s, e
    merged.append((curr_s, curr_e))
    return sum(e - s for s, e in merged)


def interval_intersection_length(s1: int, e1: int, intervals: List[Tuple[int, int]]) -> int:
    """Calculate the overlap length between [s1, e1] and a set of disjoint merged intervals."""
    if e1 <= s1:
        return 0
    total = 0
    for s2, e2 in intervals:
        start = max(s1, s2)
        end = min(e1, e2)
        if end > start:
            total += end - start
    return total


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals into sorted disjoint intervals."""
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    merged = []
    curr_s, curr_e = valid[0]
    for s, e in valid[1:]:
        if s <= curr_e:
            curr_e = max(curr_e, e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = s, e
    merged.append((curr_s, curr_e))
    return merged


def measure_step_overlap(nodes: List[Union[Dict[str, Any], Tuple, List]]) -> Dict[str, Any]:
    """Measure concurrency, overlap, and per-class metrics for a step of stamped nodes.

    Each node may be:
      - a dict with keys: name, start_ns, end_ns, and optional class, kind, ops, branch
      - a tuple: (name, kind, layers, ops, start_ns, end_ns) or (name, kind, layers, ops, branch, start_ns, end_ns)
    """
    parsed_nodes = []
    for item in nodes:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            kind = str(item.get("kind", "dense"))
            ops = item.get("ops", [])
            cls_name = item.get("class") or classify_node(name, kind, ops)
            start_ns = int(item.get("start_ns", item.get("start", 0)))
            end_ns = int(item.get("end_ns", item.get("end", 0)))
            branch = item.get("branch")
        elif isinstance(item, (tuple, list)):
            if len(item) == 6:
                name, kind, layers, ops, start_ns, end_ns = item
                branch = None
            elif len(item) >= 7:
                name, kind, layers, ops, branch, start_ns, end_ns = item[:7]
            else:
                continue
            name = str(name)
            kind = str(kind)
            cls_name = classify_node(name, kind, ops)
            start_ns = int(start_ns)
            end_ns = int(end_ns)
        else:
            continue

        if end_ns > start_ns:
            parsed_nodes.append({
                "name": name,
                "kind": kind,
                "class": cls_name,
                "branch": branch,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
            })

    if not parsed_nodes:
        return {
            "wall_ms": 0.0,
            "busy_ms": 0.0,
            "covered_ms": 0.0,
            "overlap_ms": 0.0,
            "idle_ms": 0.0,
            "ratio": 1.0,
            "classes": {},
            "hidden_nodes": [],
        }

    # If branches are not explicitly assigned, detect side branches:
    # A node whose span is entirely inside another node's span is on a side branch.
    has_explicit_branch = any(n["branch"] is not None for n in parsed_nodes)
    if not has_explicit_branch:
        for i, n1 in enumerate(parsed_nodes):
            s1, e1 = n1["start_ns"], n1["end_ns"]
            for j, n2 in enumerate(parsed_nodes):
                if i != j and n2["start_ns"] <= s1 and e1 <= n2["end_ns"] and (n2["duration_ns"] > n1["duration_ns"]):
                    n1["branch"] = "par"
                    break
            if n1["branch"] is None:
                n1["branch"] = "main"
    else:
        for n in parsed_nodes:
            if n["branch"] is None:
                n["branch"] = "main"

    # 1. Busy: sum of all node durations
    busy_ns = sum(n["duration_ns"] for n in parsed_nodes)

    # 2. Covered: union of all node intervals
    all_intervals = [(n["start_ns"], n["end_ns"]) for n in parsed_nodes]
    covered_ns = union_interval_length(all_intervals)

    # 3. Wall: span from minimum start to maximum end
    min_start = min(n["start_ns"] for n in parsed_nodes)
    max_end = max(n["end_ns"] for n in parsed_nodes)
    wall_ns = max(0, max_end - min_start)

    # 4. Overlap and Idle
    overlap_ns = max(0, busy_ns - covered_ns)
    idle_ns = max(0, wall_ns - covered_ns)
    ratio = (busy_ns / covered_ns) if covered_ns > 0 else 1.0

    # 5. Hidden nodes: nodes whose span lies entirely inside another branch's span
    hidden_nodes = []
    for i, n1 in enumerate(parsed_nodes):
        s1, e1 = n1["start_ns"], n1["end_ns"]
        for j, n2 in enumerate(parsed_nodes):
            if i == j:
                continue
            s2, e2 = n2["start_ns"], n2["end_ns"]
            # Different branches or concurrent containers:
            is_contained = (s2 <= s1 and e1 <= e2)
            if is_contained:
                # If durations are identical, only mark one as hidden
                if n2["duration_ns"] == n1["duration_ns"] and i > j:
                    continue
                # Ensure they are on different branches or n1 is on side branch
                if n1["branch"] == "par" or n2["branch"] != n1["branch"] or n2["duration_ns"] > n1["duration_ns"]:
                    hidden_nodes.append({
                        "name": n1["name"],
                        "class": n1["class"],
                        "inside": n2["name"],
                        "inside_class": n2["class"],
                        "duration_ms": round(n1["duration_ns"] / 1e6, 3),
                        "start_ns": s1,
                        "end_ns": e1,
                    })
                    break

    # Sort hidden nodes by duration descending
    hidden_nodes.sort(key=lambda h: -h["duration_ms"])

    # 6. Main chain intervals
    main_intervals = merge_intervals([(n["start_ns"], n["end_ns"]) for n in parsed_nodes if n["branch"] != "par"])

    # 7. Per-class metrics
    classes_dict = {}
    class_nodes = {}
    for n in parsed_nodes:
        class_nodes.setdefault(n["class"], []).append(n)

    for cls_name, c_nodes in class_nodes.items():
        c_busy_ns = sum(n["duration_ns"] for n in c_nodes)
        c_covered_ns = union_interval_length([(n["start_ns"], n["end_ns"]) for n in c_nodes])

        # Side-branch work: nodes in this class that ran on par/side branch beside the main chain
        side_nodes = [n for n in c_nodes if n["branch"] == "par"]
        side_branch_ns = 0
        for sn in side_nodes:
            # Overlap of this side-branch node with the main chain
            inter = interval_intersection_length(sn["start_ns"], sn["end_ns"], main_intervals)
            # If main chain is empty, the side node's own duration counts if it ran alongside other nodes
            side_branch_ns += inter if main_intervals else sn["duration_ns"]

        share = (side_branch_ns / c_busy_ns) if c_busy_ns > 0 else 0.0

        classes_dict[cls_name] = {
            "busy_ms": round(c_busy_ns / 1e6, 3),
            "covered_ms": round(c_covered_ns / 1e6, 3),
            "side_branch_ms": round(side_branch_ns / 1e6, 3),
            "side_branch_share": round(share, 3),
        }

    return {
        "wall_ms": round(wall_ns / 1e6, 3),
        "busy_ms": round(busy_ns / 1e6, 3),
        "covered_ms": round(covered_ns / 1e6, 3),
        "overlap_ms": round(overlap_ns / 1e6, 3),
        "idle_ms": round(idle_ns / 1e6, 3),
        "ratio": round(ratio, 2),
        "classes": classes_dict,
        "hidden_nodes": hidden_nodes,
    }


def measure_from_phase_table(phase_table: List[List[Any]]) -> Dict[str, Any]:
    """Measure overlap from the output of StepGraph.phase_table()."""
    if not phase_table:
        return {}

    steps_results = []
    for step_idx, step_nodes in enumerate(phase_table):
        res = measure_step_overlap(step_nodes)
        res["step"] = step_idx
        steps_results.append(res)

    # Use the latest step as primary summary, and keep the list of steps
    latest = dict(steps_results[-1])
    latest["steps"] = steps_results
    return latest
