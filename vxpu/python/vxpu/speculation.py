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

"""The speculation policy the plan names (`vxpu-plan/1`'s `speculation` section, pkg/kernelplan)
and the pure functions that apply it, for the engine's drafter path to call. Nothing here touches
a tensor: the inputs are Python lists per row, so the same functions serve the Python rows loop,
a test, and the native loop through a thin binding.

The policies (pkg/planner/speculation.go scores them against the cost model):
- fixed: every step drafts `draft` tokens; a draft is kept while it equals the verified choice
  before it (greedy acceptance, the output is what one token a step would produce);
- confidence: the drafter stops at the first draft whose top probability is under
  `min_confidence`, so a step verifies fewer token rows where the drafter is unsure;
- experts: among the drafter's candidate continuations at a position the one whose experts the
  step already reads is preferred, and the drafts are cut where they would add more than
  `max_new_experts` experts a layer beyond what the batch reads for its first tokens.

Every step's decisions feed `Stats`, which the record carries (per position: drafts issued,
accepted, and the confidence of accepted and rejected drafts), so the planner's estimates of a
policy's precision are replaced by measurements."""

import dataclasses

FIXED, CONFIDENCE, EXPERTS = "fixed", "confidence", "experts"
# the drafter's confidence encodings (the binding's draft.confidence): the score's scale in the
# drafter's int64 output (deepseek_v41.Draft: a sigmoid of the confidence head, times 1e6)
CONFIDENCE_SCALES = {"sigmoid_x1e6": 1e6}
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclasses.dataclass(frozen=True)
class Policy:
    draft: int = 0  # tokens drafted a step; 0: no drafter
    policy: str = FIXED
    min_confidence: float = 0.0
    max_new_experts: int = 0
    candidates: int = 1

    @classmethod
    def from_plan(cls, section):
        """The plan file's `speculation` section as a Policy (None or {}: no drafter), checked as
        the planner checks it."""
        if not section:
            return cls()
        p = cls(draft=int(section.get("draft", 0)), policy=section.get("policy", FIXED),
                min_confidence=float(section.get("min_confidence", 0.0)),
                max_new_experts=int(section.get("max_new_experts", 0)), candidates=int(section.get("candidates", 1)))
        p.check()
        return p

    def check(self):
        if not 0 <= self.draft <= 8:
            raise ValueError(f"speculation draft {self.draft}: want 0 to 8")
        if self.policy == CONFIDENCE and not 0 < self.min_confidence < 1:
            raise ValueError(f"speculation policy {self.policy!r}: min_confidence {self.min_confidence}, want a probability in (0, 1)")
        if self.policy == EXPERTS and (self.max_new_experts < 0 or self.candidates < 1):
            raise ValueError(f"speculation policy {self.policy!r}: max_new_experts {self.max_new_experts}, candidates {self.candidates}")
        if self.policy not in (FIXED, CONFIDENCE, EXPERTS):
            raise ValueError(f"speculation policy {self.policy!r}: the engine knows {FIXED!r}, {CONFIDENCE!r} and {EXPERTS!r}")
        return self


def accept(draft_tokens, target_choices):
    """How many of a row's drafts the verify keeps: each draft that equals the choice the verify
    made at the position before it, up to the first that does not (the native loop's rule). The
    policy does not change acceptance, only what is drafted; the output stays what one token a
    step would produce."""
    n = 0
    for draft, choice in zip(draft_tokens, target_choices):
        if draft != choice:
            break
        n += 1
    return n


def draft_length(draft_probs, policy):
    """How many of the drafter's proposed tokens a row issues this step under the policy, from
    the top probability the drafter gave each (in order). fixed: all of them, up to the policy's
    draft; confidence: up to the first under min_confidence."""
    k = min(policy.draft, len(draft_probs))
    if policy.policy != CONFIDENCE:
        return k
    for i in range(k):
        if draft_probs[i] < policy.min_confidence:
            return i
    return k


def compose_step(first, drafts, draft_probs, policy, steps):
    """One row's token rows for a verify step of `steps` token rows: the row's next token, then
    the drafts it issues under the policy (draft_length over their confidence; every draft without a
    policy), then the last issued token again in the positions left (the first token when none is
    issued), since the exported verify graph has `steps` token rows whatever is issued. A padded
    position the verify agrees with is a correct greedy token, so acceptance stays exact; the
    statistics count only the issued drafts. Returns (token rows, issued). The native loop
    (node_server.cpp's DecodeLoop with set_policy) composes the same rows."""
    k = min(len(drafts), steps - 1)
    if policy is not None:
        k = draft_length(draft_probs if draft_probs is not None else [1.0] * k, policy) if policy.policy == CONFIDENCE \
            else min(k, policy.draft)
    issued = list(drafts[:k])
    pad = issued[-1] if issued else first
    return [first] + issued + [pad] * (steps - 1 - k), k


def new_experts(token_experts, reading):
    """The experts a draft token adds, per layer, beyond those in `reading` (per layer, the ids
    the batch reads for its first tokens): the sum over layers of the ids not already read."""
    return sum(len(set(ids) - set(reading[layer])) for layer, ids in enumerate(token_experts))


def prefer(candidates, reading):
    """Among a position's candidate continuations, the index of the one whose experts the step
    already reads most (fewest new experts); ties go to the first, the drafter's own order (its
    most probable candidate). `candidates` is a list of per-layer expert id lists, one per
    candidate; `reading` is the per-layer ids already read."""
    best, best_new = 0, None
    for i, token_experts in enumerate(candidates):
        n = new_experts(token_experts, reading)
        if best_new is None or n < best_new:
            best, best_new = i, n
    return best


def plan_drafts(candidates_by_position, reading, policy):
    """The experts policy applied to one row's step: at each position choose the preferred
    candidate, stop where the drafts so far would add more than max_new_experts experts a layer
    on average beyond what the batch reads. Returns the chosen candidate index per issued
    position. `candidates_by_position[i]` is the list of candidates at draft position i, each a
    per-layer list of expert ids; the experts of a chosen draft count as read for the next."""
    layers = len(reading)
    budget = policy.max_new_experts * layers
    chosen, added = [], 0
    read = [set(ids) for ids in reading]
    for position, candidates in enumerate(candidates_by_position[:policy.draft]):
        if not candidates:
            break
        i = prefer(candidates, read) if policy.policy == EXPERTS else 0
        cost = new_experts(candidates[i], read)
        if policy.policy == EXPERTS and added + cost > budget:
            break
        added += cost
        chosen.append(i)
        for layer, ids in enumerate(candidates[i]):
            read[layer].update(ids)
    return chosen


class Stats:
    """Per draft position: drafts issued, accepted, and the drafter's confidence on the accepted
    and the rejected ones, for the record (`speculation` stat) and the planner's next fit."""

    def __init__(self, positions=8):
        self.issued = [0] * positions
        self.accepted = [0] * positions
        self.confidence_accepted = [0.0] * positions
        self.confidence_rejected = [0.0] * positions
        self.steps = 0
        self.events = []  # (accepted, [confidence per issued draft] or None) per row and step, for evaluate

    def record(self, issued, accepted, draft_probs=None):
        """One row's step: `issued` drafts, of which the first `accepted` were kept."""
        self.steps += 1
        self.events.append((accepted, list(draft_probs[:issued]) if draft_probs is not None else None))
        for i in range(issued):
            self.issued[i] += 1
            if i < accepted:
                self.accepted[i] += 1
            if draft_probs is not None and i < len(draft_probs):
                if i < accepted:
                    self.confidence_accepted[i] += draft_probs[i]
                else:
                    self.confidence_rejected[i] += draft_probs[i]

    def summary(self):
        """Per position the acceptance rate and the mean confidence of accepted and rejected
        drafts; `alpha` is the per-position rate the planner's geometric model uses (the mean
        over positions weighted by what was issued)."""
        positions = []
        for i in range(len(self.issued)):
            if not self.issued[i]:
                continue
            rejected = self.issued[i] - self.accepted[i]
            positions.append({"position": i + 1, "issued": self.issued[i], "accepted": self.accepted[i],
                              "rate": round(self.accepted[i] / self.issued[i], 3),
                              "confidence_accepted": round(self.confidence_accepted[i] / self.accepted[i], 3) if self.accepted[i] else None,
                              "confidence_rejected": round(self.confidence_rejected[i] / rejected, 3) if rejected else None})
        issued, accepted = sum(self.issued), sum(self.accepted)
        out = {"steps": self.steps, "issued": issued, "accepted": accepted,
               "alpha": round(accepted / issued, 3) if issued else None, "positions": positions}
        thresholds = self.thresholds()
        if thresholds:
            out["thresholds"] = thresholds
        return out

    def thresholds(self, thresholds=THRESHOLDS):
        """Per confidence threshold, what the planner's confidence policy needs measured: the
        share of the rejected drafts the threshold would have foreseen (precision) and the share
        of the accepted drafts it would have forgone (loss). Empty without confidences."""
        accepted_c, rejected_c = [], []
        for n, confs in self.events:
            if confs is None:
                continue
            accepted_c += confs[:n]
            rejected_c += confs[n:n + 1]  # the first rejected draft is the one a threshold could have stopped at
        if not accepted_c and not rejected_c:
            return {}
        out = {}
        for t in thresholds:
            out[str(t)] = {"precision": round(sum(c < t for c in rejected_c) / len(rejected_c), 3) if rejected_c else None,
                           "loss": round(sum(c < t for c in accepted_c) / len(accepted_c), 3) if accepted_c else None}
        return out


def stats_from_events(events, positions, scale=0.0):
    """The native loop's records (rows of [accepted, score_1 .. score_k], a score of -1 without a
    confidence output, -2 at a position the policy did not issue) as Stats; `scale` turns the
    scores into probabilities (0: no confidence)."""
    stats = Stats(positions)
    for row in events:
        n, scores = int(row[0]), [int(s) for s in row[1:1 + positions]]
        scores = [s for s in scores if s != -2]  # -2: a position the policy did not issue
        confs = [s / scale for s in scores] if scale and all(s >= 0 for s in scores) else None
        stats.record(len(scores), min(n, len(scores)), confs)
    return stats


def evaluate(stats, policy):
    """A policy replayed over the recorded steps, exactly: under fixed k only the first k drafts
    count (a draft past k is never issued), under confidence the drafts up to the first under the
    threshold; the tokens a step are one plus the drafts accepted among those issued (greedy
    acceptance is exact, so the policy changes cost and yield, never the tokens). Returns the mean
    tokens and token rows a step, the acceptance among the issued, and the steps it rests on."""
    if not stats.events:
        return None
    tokens = issued_total = accepted_total = 0
    usable = 0
    for n, confs in stats.events:
        if policy.policy == CONFIDENCE:
            if confs is None:
                continue
            issued = draft_length(confs, policy)
        else:
            issued = min(policy.draft, len(confs) if confs is not None else policy.draft)
        kept = min(n, issued)
        usable += 1
        issued_total += issued
        accepted_total += kept
        tokens += 1 + kept
    if not usable:
        return None
    return {"policy": policy.policy, "draft": policy.draft,
            **({"min_confidence": policy.min_confidence} if policy.policy == CONFIDENCE else {}),
            "steps": usable, "tokens_per_step": round(tokens / usable, 3),
            "token_rows_per_step": round(1 + issued_total / usable, 3),
            "acceptance": round(accepted_total / issued_total, 3) if issued_total else None}


def evaluate_all(stats, draft, thresholds=THRESHOLDS):
    """Every policy the plan could name, replayed over the same records: fixed 1..draft, and
    confidence at each threshold with up to `draft` drafts (where the records carry confidence)."""
    out = [evaluate(stats, Policy(draft=k)) for k in range(1, draft + 1)]
    out += [evaluate(stats, Policy(draft=draft, policy=CONFIDENCE, min_confidence=t)) for t in thresholds]
    return [o for o in out if o is not None]
