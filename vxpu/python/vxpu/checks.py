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

"""The engine's standing self-checks (hero run 10).

Every wrong-answer bug of hero runs 8 and 9 (the expert kernels' arrival
counters shared across batches, the drafter's strided output copied flat,
a replica refreshed on the wrong batch's graph, an island corrupting the
routed experts' buffers) showed as rows that should have been identical
and were not, or tokens that should have matched a reference and did not,
and each was found by a person running a check. These three run on their
own, on by default in development, each off by a plan switch, and the
kernel record carries whether they ran, what they found and what they cost
(the "checks" statistic):

  golden values at load (VXPU_CHECK_GOLDEN): after warm-up the engine
      decodes the model's golden prompts greedily and compares the tokens
      with the stored ones; golden.json in the artifact directory, or for
      the test fixtures fixtures/golden/<model>.json in the repository.
      A mismatch names the prompt and the first differing position, and
      the engine refuses to serve (or warns, VXPU_CHECK_ON_MISMATCH=warn).
  the duplicate-row invariant (VXPU_CHECK_DUPLICATE_ROWS): a generate_rows
      call with a free row pads it with a copy of a decoding row (same
      prompt, admitted at the same position) and compares the two rows'
      tokens as they are emitted; the copy is never in the results or the
      statistics, and a real prompt that needs the row gets it (the copy
      is cancelled). A duplicate row selects the experts the original
      does, so the step reads no more bytes; its cost is one more row of
      compute.
  the reference alongside (VXPU_CHECK_REFERENCE): on a sample of steps the
      step runs again, on a copy of the state, through the logical graph
      (the artifact's own ops, before the lowering replaced them), with
      every chosen implementation run beside the registered reference on
      the same inputs: per op the largest deviation seen, judged by the
      numerics class the implementation declares (bit-identical, or
      rounding), and the step's logits against the captured step's. The
      sample is every N steps, held back until the run has decoded for
      cost / budget since the last one (VXPU_CHECK_REFERENCE_BUDGET), so
      the check costs a few percent at most.

The hooks in engine.py are at the boundaries of its generation paths: a
Checks object per engine, the golden check at the end of the load, a
Padding object per generate_rows call, _Request.emit calling row_token,
and the reference check after a step (the eager rows) or between steps
with the native loop paused (the captured step graphs)."""

import json
import os
import time

import torch

from . import plan as planmod

GOLDEN_FORMAT = "vxpu-golden/1"
LOGIT_TOLERANCE = 0.025  # the step's logits against the reference's, relative to the row's largest logit (vectors_test's)
# a "rounding" implementation against the reference, relative to the output's largest value: bf16 work summed in
# another order drifts by a few percent of the largest element at the real model's sizes (vxpu::attn_select over
# 640 selected tokens measured 0.025 on DeepSeek-V4.1-Flash, every other op under 0.009)
ROUNDING_TOLERANCE = 0.05


class CheckFailed(RuntimeError):
    """A standing check found a mismatch and the plan says to stop."""


def golden_path(artifact_dir, manifest):
    """Where this model's golden values are: golden.json in the artifact directory, else the
    repository's fixtures/golden/<name>.json, <name> being the last part of the model path the
    artifact was exported from (the test fixtures write their checkpoint to a directory of that
    name). None when neither exists."""
    path = os.path.join(artifact_dir, "golden.json")
    if os.path.exists(path):
        return path
    repo = str(manifest.get("source", {}).get("repo", ""))
    name = os.path.basename(repo.rstrip("/"))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden", name + ".json")
    return path if name and os.path.exists(path) else None


def write_golden(engine, prompts, path, tokens=32, source=""):
    """Record the greedy tokens of `prompts` (lists of ids) from engine, `tokens` each, as a
    golden file at path. Returns the document."""
    from .engine import SamplingParams
    out = engine.generate_rows([list(p) for p in prompts], tokens, SamplingParams(do_sample=False))
    doc = {"format": GOLDEN_FORMAT, "tokens_per_prompt": tokens, "source": source,
           "prompts": [{"ids": [int(t) for t in p], "tokens": [int(t) for t in got]}
                       for p, got in zip(prompts, out["token_ids"])]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return doc


def _rel_deviation(got, want):
    """max |got - want| relative to max |want| (0 when both are empty)."""
    got = got.detach().float()
    want = want.detach().float().to(got.device)
    if got.shape != want.shape:
        return float("inf")
    if got.numel() == 0:
        return 0.0
    scale = want.abs().max().item()
    diff = (got - want).abs().max().item()
    return diff / scale if scale > 0 else (0.0 if diff == 0 else float("inf"))


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    return []


class Checks:
    """One engine's checks: the switches read from its plan, the results kept for the record."""

    def __init__(self, engine):
        self.engine = engine
        plan = engine.plan
        self.golden_on = bool(plan["check_golden"])
        self.duplicate_rows = int(plan["check_duplicate_rows"])
        self.reference_every = int(plan["check_reference_every"])
        self.reference_budget = float(plan["check_reference_budget"])
        self.stop = plan["check_on_mismatch"] == "stop"
        self.results = {"golden": {"ran": False}, "duplicate_rows": {"on": self.duplicate_rows > 0, "padded": 0,
                                                                     "compared": 0, "mismatches": 0},
                        "reference": {"on": self.reference_every > 0, "samples": 0, "ms": 0.0, "ops": {}}}
        self._program = None  # the logical decode graph (engine._build keeps it before the lowering)
        self._runner = None
        self._state_copy = None
        self._steps_seen = 0
        self._last_sample = 0  # the first sample at step N: a run's first steps are its warm-up
        self._last_sample_at = time.perf_counter()
        self._last_cost_ms = 0.0
        self._site_choice = None
        self._expected = {}  # (request id, token index) -> what the reference check says the row emits there

    # ---- reporting

    def _record(self):
        self.engine.plan.stat("checks", json.loads(json.dumps(self.results)))

    def mismatch(self, text):
        """A check failed: note it, print it, and stop (CheckFailed) unless the plan says warn."""
        self.engine.plan.note("check failed: " + text)
        print("[vxpu] CHECK FAILED: " + text, flush=True)
        self._record()
        if self.stop:
            raise CheckFailed(text)

    def status_text(self):
        """One line for the log at the end of the engine's build: what each standing check will do
        on this engine, so that a run never has a check silently absent (hero run 15: the
        tensor-parallel step ran for days with none of the three wired in)."""
        g = self.results["golden"]
        golden = ("ran" if g.get("ran") else f"skipped ({g.get('skipped', 'not run')})") if self.golden_on else "off"
        dup = f"{self.duplicate_rows} row a generate_rows call" if self.duplicate_rows > 0 else "off"
        ref = f"every {self.reference_every} steps" if self.reference_every > 0 else "off"
        return f"checks: golden {golden}; duplicate rows {dup}; reference alongside {ref}"

    # ---- the tensor-parallel step (engine._generate_rows_tp): the same checks, on its own loop

    def tp_shadow(self, n, rows):
        """Whether the tensor-parallel loop should fill a free row with a copy of row 0 (the
        duplicate-row invariant on that loop): n prompts in a group of `rows`, at least one free."""
        if self.duplicate_rows <= 0 or n <= 0 or n >= rows:
            return False
        self.results["duplicate_rows"]["padded"] += 1
        self._record()
        return True

    def tp_compare(self, chosen, original, shadow, step):
        """The copy's greedy choice against the original's after a tensor-parallel step."""
        res = self.results["duplicate_rows"]
        res["compared"] += 1
        if chosen[shadow] != chosen[original]:
            res["mismatches"] += 1
            self.mismatch(f"duplicate row: the copy in row {shadow} chose {chosen[shadow]} where row {original} chose "
                          f"{chosen[original]} at step {step} (tensor-parallel step)")

    def reference_tp(self, ids, positions, active, logits, steps, live):
        """The reference alongside on the tensor-parallel loop: when a sample is due, run the step
        (ids, positions) through the logical graph on a copy of the state and compare the step's
        logits of the active rows. The engine under tensor parallelism keeps each rank's shard of
        the weights, so the logical graph can run here only when the engine still holds a whole
        program (a single rank, or layer sharding); otherwise the check is skipped by name, once,
        rather than silently."""
        if not self.reference_due(steps):
            return False
        res = self.results["reference"]
        if self._program is None or live is None:
            if res.get("skipped") != "tensor-parallel step: no logical program with whole weights":
                res["skipped"] = "tensor-parallel step: no logical program with whole weights"
                self._record()
                print(f"[vxpu] reference check skipped: {res['skipped']}", flush=True)
            self._last_sample = steps // max(1, self.reference_every)
            return False
        try:
            self.snapshot(live)
            self.reference_step(torch.as_tensor(ids), torch.as_tensor(positions), active, logits=logits, steps=steps)
            return True
        except Exception as exc:  # the check must not take the step down; it says why it did not run
            res["skipped"] = f"tensor-parallel step: {type(exc).__name__}: {exc}"[:300]
            self._record()
            print(f"[vxpu] reference check skipped: {res['skipped']}", flush=True)
            self._last_sample = steps // max(1, self.reference_every)
            return False

    def ranges_text(self):
        """The step's layer ranges and their GPUs, for a mismatch message."""
        devices = self.engine.plan["layer_devices"]
        return devices if devices else f"one range, all layers on {self.engine.device}"

    # ---- 1. golden values at load

    def golden(self):
        """Decode the model's golden prompts after warm-up and compare the tokens (module docstring).
        Skipped for a range engine (it serves a slice of layers) and when there is no golden file."""
        e = self.engine
        result = self.results["golden"]
        if not self.golden_on:
            result["skipped"] = "off"
            return
        if e.plan["layer_range"]:
            result["skipped"] = "range engine"
            return
        path = golden_path(e.artifact_dir, e.manifest)
        if path is None:
            result["skipped"] = "no golden file"
            e.plan.note("golden check: no golden.json in the artifact and no fixtures/golden file for this model")
            return
        with open(path) as f:
            doc = json.load(f)
        if doc.get("format") != GOLDEN_FORMAT:
            raise planmod.ConfigError(f"{path}: not a {GOLDEN_FORMAT} file")
        n = int(doc.get("tokens_per_prompt", 32))
        prompts = [list(p["ids"]) for p in doc["prompts"]]
        from .engine import SamplingParams
        started = time.perf_counter()
        out = e.generate_rows(prompts, n, SamplingParams(do_sample=False))
        ms = (time.perf_counter() - started) * 1e3
        first_diff = []
        for p, got in zip(doc["prompts"], out["token_ids"]):
            want = list(p["tokens"])[:n]
            got = list(got)[:n]
            k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            if k is None and len(got) != len(want):
                k = min(len(got), len(want))
            first_diff.append(k)
        bad = [(i, k) for i, k in enumerate(first_diff) if k is not None]
        result.update({"ran": True, "path": path, "prompts": len(prompts), "tokens": n, "ms": round(ms, 1),
                       "ok": not bad, "first_diff": first_diff})
        self._record()
        msg = (f"golden check: {len(prompts)} prompts, {n} tokens each, {'all match' if not bad else 'MISMATCH'} "
               f"({ms:.0f} ms)")
        print(f"[vxpu] {msg}", flush=True)
        e.plan.note(msg)
        if bad:
            i, k = bad[0]
            want = doc["prompts"][i]["tokens"]
            got = out["token_ids"][i]
            self.mismatch(f"golden values: prompt {i} differs first at position {k}: got {got[k] if k < len(got) else None}, "
                          f"expected {want[k] if k < len(want) else None} ({path})")

    # ---- 2. the duplicate-row invariant

    def padding(self, requests, rows, batches=1, batch_of=None, telemetry=False):
        """The Padding of one generate_rows call, or None when the check is off, the run records
        telemetry (its row counts are the measurement), or a request samples (two sampled rows
        differ by design)."""
        if self.duplicate_rows <= 0 or telemetry or not requests:
            return None
        if any(r.sampler.do_sample for r in requests):
            return None
        return Padding(self, requests, rows, batches, batch_of)

    def row_token(self, request):
        """request emitted a token: compare it with its original's, or with its copies', at the
        same position. Called from _Request.emit for requests a Padding paired."""
        if self._expected:
            self.expected_token(request)
        k = len(request.token_ids) - 1
        token = request.token_ids[k]
        pairs = []
        if request.shadow_of is not None:
            pairs.append((request, request.shadow_of))
        for shadow in request.shadows:
            pairs.append((shadow, request))
        for shadow, original in pairs:
            other = original if shadow is request else shadow
            if len(other.token_ids) <= k:
                continue
            self.results["duplicate_rows"]["compared"] += 1
            if other.token_ids[k] == token:
                continue
            self.results["duplicate_rows"]["mismatches"] += 1
            self.mismatch(
                f"duplicate rows: row {shadow.row} of batch {shadow.batch} is a copy of row {original.row} (the same "
                f"prompt of {len(original.prompt)} tokens, position {len(original.prompt) + k}); at generated token "
                f"{k} (the row's step {k}) the copy chose {shadow.token_ids[k]} and the original {original.token_ids[k]}; "
                f"the step's layer ranges: {self.ranges_text()}; the row's tokens leave the last range")

    def rows_active(self, steps, shadows):
        """The loop's rows active per step ((ms, n) pairs from DecodeLoop.take_steps) without the
        padded rows: a copy admitted when the loop had run s steps decodes for len(tokens) - 1
        steps from there at one token a step; with the drafter, until the step its finish was seen."""
        counts = [n for _, n in steps if n]
        for shadow in shadows:
            start = getattr(shadow, "admitted_at", None)
            if start is None:
                continue
            if self.engine.step_tokens == 1 or getattr(shadow, "finished_at", None) is None:
                end = start + max(0, len(shadow.token_ids) - 1)
            else:
                end = shadow.finished_at
            for i in range(start, min(end, len(counts))):
                counts[i] = max(0, counts[i] - 1)
        return counts

    # ---- 3. the reference alongside

    def keep_logical(self, decode):
        """Called by engine._build before the lowering: keep the decode graph as the artifact
        exported it (its vxpu op calls intact), sharing the weights: a GraphModule over the same
        root with a copy of the graph."""
        if self.reference_every <= 0:
            return
        import copy
        self._program = torch.fx.GraphModule(decode, copy.deepcopy(decode.graph))

    def reference_due(self, steps):
        """Whether the reference check should run now: every N steps (`steps` the run's count so
        far), and only when the time decoded since the last sample covers its cost at the budget."""
        if self.reference_every <= 0 or self._program is None:
            return False
        if steps < self._steps_seen:  # a new call counts its steps from the start
            self._last_sample = 0
        self._steps_seen = steps
        if steps // self.reference_every <= self._last_sample:
            return False
        since_ms = (time.perf_counter() - self._last_sample_at) * 1e3
        if self._last_cost_ms and self.reference_budget > 0 and since_ms < self._last_cost_ms / self.reference_budget:
            return False
        return True

    def snapshot(self, live):
        """Copy the live state (the tensors the step about to run reads and writes) into the
        check's own copy: the reference runs that step on the copy. Before the step, since a step
        is not a function of its result: a recurrent state (Qwen4's linear attention) run twice
        differs."""
        self._prepare(live)
        for fqn, t in live.items():
            self._state_copy[fqn].copy_(t)

    def reference_native(self, loop, graph, live, requests):
        """The reference check on a native loop: pause it between steps, take the state as it is
        (the next step's input), predict that step's inputs (each active row feeds the token it
        chose at its next position), run the reference on a copy, and expect each greedy row's
        next token to be the reference's choice (or a near tie, within LOGIT_TOLERANCE) when the
        loop emits it (row_token). `requests` maps row -> the request decoding there."""
        loop.pause()
        try:
            inputs = dict(zip(graph.runner.graph.inputs, graph._inputs))
            ids, positions = inputs["input_ids"], inputs["cache_position"]
            if ids.shape[1] != 1:  # the drafter's steps feed several tokens the loop composes: not predicted here
                self.results["reference"]["skipped"] = "steps of several tokens on the native loop"
                self._last_sample = max(self._last_sample, loop.steps() // max(1, self.reference_every))
                return
            chosen = graph._chosen.reshape(ids.shape[0], -1)[:, :1]
            free = set(loop.free_rows())
            active = [r for r, q in requests.items() if r not in free and q.finish_reason is None and not q.cancelled]
            if not active:
                return
            self.snapshot(live)
            next_ids, next_pos = chosen.clone().to(ids.dtype), positions.clone() + 1
            self.reference_step(next_ids, next_pos, active, expect={r: requests[r] for r in active}, steps=loop.steps())
        finally:
            loop.resume()

    def reference_step(self, ids, positions, active, logits=None, expect=None, steps=None):
        """Run the step whose inputs are (ids, positions) on the snapshot of the state (snapshot),
        through the logical graph with every chosen implementation beside its reference; compare
        per op, and the logits of the active rows with `logits` (the step's own) when given, or
        record what each row in `expect` (row -> request, greedy) must emit next."""
        e = self.engine
        if not active or self._state_copy is None:
            return
        self._last_sample = (steps if steps is not None else self._steps_seen) // max(1, self.reference_every)
        started = time.perf_counter()
        deviations = {}
        import contextlib

        def wrap(plan, op, impl, device_type, key):
            ref = planmod.reference_of(op)
            if ref is None or impl.host or ref.host or impl is ref:
                return impl
            chosen = self._chosen_for(op, key, impl)
            fn = _Beside(op, chosen, ref, key, deviations)
            return planmod.Impl(op, chosen.name, fn, chosen.devices, chosen.numerics, chosen.when, None,
                                chosen.reference, chosen.host, chosen.lowerable, chosen.fused)

        with e.external_scope(), planmod.checking(wrap), torch.no_grad(), \
                torch.cuda.device(e.device) if str(e.device).startswith("cuda") else contextlib.nullcontext():
            run = self._runner if self._runner is not None else self._program
            dev = torch.device(e.device)
            got = run(input_ids=ids.to(dev), cache_position=positions.to(dev))
        got = got[0] if isinstance(got, (tuple, list)) and len(got) == 1 else got
        if str(e.device).startswith("cuda"):
            torch.cuda.synchronize()
        got = got.detach().float()
        got = got.reshape(ids.shape[0], ids.shape[1], -1)
        res = self.results["reference"]
        worst, token_flips = 0.0, 0
        if logits is not None:
            logits = logits.detach().float().reshape(got.shape)
            got = got.to(logits.device)
            for r in active:
                worst = max(worst, _rel_deviation(logits[r], got[r]))
                token_flips += int((logits[r].argmax(-1) != got[r].argmax(-1)).sum().item())
            res["logits_max_rel"] = round(max(res.get("logits_max_rel", 0.0), worst), 5)
            res["token_flips"] = res.get("token_flips", 0) + token_flips
        if expect:
            last = got[:, -1].cpu()
            for r, request in expect.items():
                if request.sampler.do_sample:
                    continue
                top = last[r].topk(2)
                scale = last[r].abs().max().item()
                margin = (top.values[0] - top.values[1]).item() / scale if scale > 0 else 0.0
                index = int(positions[r].item()) - len(request.prompt) + 1  # the token this step chooses
                request.checks = self
                self._expected[(id(request), index)] = (int(top.indices[0].item()), margin, r, int(positions[r].item()))
                res["expected"] = res.get("expected", 0) + 1
        cost_ms = (time.perf_counter() - started) * 1e3
        self._last_cost_ms = cost_ms
        self._last_sample_at = time.perf_counter()
        res["samples"] += 1
        res["ms"] = round(res["ms"] + cost_ms, 1)
        res["last_ms"] = round(cost_ms, 1)
        over = []
        for op, d in deviations.items():
            row = res["ops"].setdefault(op, {"impl": d["impl"], "numerics": d["numerics"], "calls": 0, "max_rel": 0.0,
                                             "int_differ": 0.0})
            row["calls"] += d["calls"]
            if d["max_rel"] > row["max_rel"]:
                row.update({"max_rel": round(d["max_rel"], 6), "worst_shapes": d["worst_key"]})
            row["int_differ"] = round(max(row["int_differ"], d["int_differ"]), 6)
            exact = d["numerics"] == "bit-identical"
            if d["max_rel"] > (0.0 if exact else ROUNDING_TOLERANCE) or (exact and d["int_differ"] > 0):
                over.append(f"{op} ({d['impl']}, {d['numerics']}): {d['max_rel']:.3g} at {d['worst_key']}"
                            + (f", {d['int_differ']:.3g} of the integer outputs differ" if d["int_differ"] else ""))
        self._record()
        where = steps if steps is not None else self._steps_seen
        if logits is not None:
            print(f"[vxpu] reference check at step {where}: {len(deviations)} ops compared, logits within {worst:.3g} "
                  f"of the reference, {token_flips} top-1 differences, {cost_ms:.0f} ms", flush=True)
        else:
            print(f"[vxpu] reference check after step {where}: {len(deviations)} ops compared, the next token of "
                  f"{len(active)} rows expected, {cost_ms:.0f} ms", flush=True)
        if over or worst > LOGIT_TOLERANCE:
            self.mismatch("reference alongside: " + "; ".join(
                over + ([f"the step's logits deviate {worst:.3g} from the reference (limit {LOGIT_TOLERANCE})"]
                        if worst > LOGIT_TOLERANCE else [])))

    def expected_token(self, request):
        """request emitted a token the reference check predicted: compare (row_token)."""
        k = len(request.token_ids) - 1
        want = self._expected.pop((id(request), k), None)
        if want is None:
            return
        token, margin, row, position = want
        res = self.results["reference"]
        got = request.token_ids[k]
        if got == token:
            res["tokens_ok"] = res.get("tokens_ok", 0) + 1
            self._record()
        elif margin <= LOGIT_TOLERANCE:
            res["near_ties"] = res.get("near_ties", 0) + 1
            self._record()
        else:
            res["token_mismatches"] = res.get("token_mismatches", 0) + 1
            self.mismatch(f"reference alongside: row {row} at position {position} chose {got} where the reference "
                          f"chooses {token} by a margin of {margin:.3g} of the largest logit (generated token {k})")

    def _chosen_for(self, op, key, impl):
        """The body the lowering chose for this op and shape, when its record names exactly one
        (the eager dispatch may pick differently: a fused body applies only under the lowering);
        else the dispatch's choice."""
        if self._site_choice is None:
            choices = {}
            for site in self.engine.plan.sites:
                if site.get("chosen"):
                    choices.setdefault((site["op"], site.get("shapes")), set()).add(site["chosen"])
            self._site_choice = {k: next(iter(v)) for k, v in choices.items() if len(v) == 1}
        name = self._site_choice.get((op, key))
        if name is None or name == impl.name:
            return impl
        return next((i for i in planmod.REGISTRY[op] if i.name == name), impl)

    def _prepare(self, live):
        """The state copy and, under layer sharding, the dataflow runner over the logical graph
        (made once; returns the time it took, 0 after the first call)."""
        if self._state_copy is not None:
            return 0.0
        started = time.perf_counter()
        e = self.engine
        self._state_copy = {fqn: torch.empty_like(t) for fqn, t in live.items()}
        from . import dataflow, segments
        if e.plan["layer_devices"]:
            split, _ = segments.split_at_host_ops(self._program)
            graph = e._graph(split)
            runner = dataflow.Runner(split, len(graph.nodes), 0, device=e.device, graph=graph,
                                     placement=e._host_placement(graph, "decode"))
            runner.state = set(e.binding["state"])
            self._runner = runner
            _point(split, self._state_copy, e.aliases)
        else:
            _point(self._program, self._state_copy, e.aliases)
        return (time.perf_counter() - started) * 1e3


class _Beside:
    """An implementation's fn with the reference run beside it on the same inputs: the deviation
    goes into `deviations` per op, and the reference's result flows on (so each op's deviation is
    its own, not its inputs')."""

    def __init__(self, op, impl, ref, key, deviations):
        self.op, self.impl, self.ref, self.key, self.deviations = op, impl, ref, key, deviations

    def __call__(self, *args, **kwargs):
        from . import ops
        saved = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
        got = ops.call(self.op, self.impl.fn, *args, **kwargs)
        want = ops.call(self.op, self.ref.fn, *saved, **kwargs)
        got_t, want_t = _tensors(got), _tensors(want)
        rel, ints = 0.0, 0.0
        if len(got_t) != len(want_t):
            rel = float("inf")
        else:
            got_t, want_t = _sorted_choices(got_t), _sorted_choices(want_t)
        for g, w in zip(got_t, want_t):
            if g.dtype.is_floating_point or g.dtype.is_complex:
                rel = max(rel, _rel_deviation(g, w))
            elif g.shape != w.shape:
                ints = 1.0
            else:  # integer outputs (an expert choice, a position): the share of elements that differ.
                # A top-k choice flips at a near tie under rounding, so a rounding implementation is
                # not judged on it; a bit-identical one must agree exactly
                ints = max(ints, (g != w.to(g.device)).float().mean().item() if g.numel() else 0.0)
        d = self.deviations.setdefault(self.op, {"impl": self.impl.name, "numerics": self.impl.numerics, "calls": 0,
                                                 "max_rel": 0.0, "worst_key": None, "int_differ": 0.0})
        d["calls"] += 1
        if rel > d["max_rel"] or d["worst_key"] is None:
            d["max_rel"], d["worst_key"] = rel, self.key
        d["int_differ"] = max(d["int_differ"], ints)
        return want


def _sorted_choices(tensors):
    """A choice an op returns as an integer tensor with a float tensor of the same shape beside it
    (the gate's expert indices and their weights) is a set along the last dimension: two
    implementations may list it in different orders (by score, by index). Sort each such pair by
    the integers so the comparison is order-free; everything else passes through."""
    out = list(tensors)
    for i, t in enumerate(out):
        if t.dtype.is_floating_point or t.dtype.is_complex or t.dim() == 0 or t.shape[-1] < 2:
            continue
        order = t.argsort(dim=-1)
        out[i] = t.gather(-1, order)
        for j, f in enumerate(out):
            if j != i and (f.dtype.is_floating_point) and f.shape == t.shape:
                out[j] = f.gather(-1, order.to(f.device))
    return out


def _point(module, state, aliases):
    """Point module's state buffers (and their aliases) at `state`'s tensors, by path, as the
    engine binds a session (engine._bind_pointers / _rebind)."""
    from . import dataflow
    state = {**state, **{alias: state[canonical] for alias, canonical in aliases.items() if canonical in state}}
    for fqn, tensor in state.items():
        parent, name = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
        try:
            target = dataflow._attr(module, parent) if parent else module
        except AttributeError:
            continue
        if name in target._buffers:
            target._buffers[name] = tensor
        elif isinstance(target.__dict__.get(name), torch.Tensor):
            setattr(target, name, tensor)


class Padding:
    """The padded rows of one generate_rows call: which batches have room, the copies made, and
    the cancelling of a copy when a real prompt needs its row."""

    def __init__(self, checks, requests, rows, batches=1, batch_of=None):
        self.checks = checks
        self.shadows = []
        self.rows = rows
        self.pending = {b: min(checks.duplicate_rows, max(0, rows - 1)) for b in range(batches)}
        self.made = {b: 0 for b in range(batches)}

    def next(self, admitted, b=0):
        """A copy to admit in batch b, of one of `admitted` (that batch's decoding real requests),
        or None when the batch is padded already or nothing decodes there yet."""
        from .engine import _Request
        originals = [r for r in admitted if r is not None and r.shadow_of is None and r.finish_reason is None
                     and not r.cancelled]
        if self.pending.get(b, 0) <= 0 or not originals:
            return None
        original = originals[self.made[b] % len(originals)]
        shadow = _Request(original.prompt, original.max_new_tokens, original.sampler, original.eos_ids)
        shadow.shadow_of, shadow.checks, shadow.batch = original, self.checks, b
        original.shadows.append(shadow)
        original.checks = self.checks
        self.pending[b] -= 1
        self.made[b] += 1
        self.shadows.append(shadow)
        self.checks.results["duplicate_rows"]["padded"] += 1
        return shadow

    def make_room(self, b=0):
        """A real prompt waits for a row in batch b: cancel one of its copies (the row frees at
        the next step). Returns whether one was."""
        for shadow in self.shadows:
            if shadow.batch == b and not shadow.cancelled and shadow.finish_reason is None:
                shadow.cancelled = True
                return True
        return False


def main(argv=None):
    """python -m vxpu.checks golden ARTIFACT -o golden.json [--prompts ids.json] [--tokens 32]: record
    the greedy tokens of each prompt (a JSON list of id lists; default: vectors_test's ID_PROMPTS)
    with an engine loaded with the checks off."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("golden", help="record a golden file from an engine")
    g.add_argument("artifact")
    g.add_argument("-o", "--out", required=True)
    g.add_argument("--prompts", help="JSON file: a list of id lists")
    g.add_argument("--tokens", type=int, default=32)
    g.add_argument("--device", default="cpu")
    g.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    for name in ("VXPU_CHECK_GOLDEN", "VXPU_CHECK_REFERENCE"):
        os.environ.setdefault(name, "0")
    from .engine import Engine
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    else:
        from .vectors_test import ID_PROMPTS
        prompts = [p["ids"] for p in ID_PROMPTS]
    engine = Engine(args.artifact, device=args.device, compile_decode=args.compile)
    doc = write_golden(engine, prompts, args.out, tokens=args.tokens, source=os.path.abspath(args.artifact))
    print(f"wrote {args.out}: {len(doc['prompts'])} prompts, {args.tokens} tokens each")


if __name__ == "__main__":
    main()
