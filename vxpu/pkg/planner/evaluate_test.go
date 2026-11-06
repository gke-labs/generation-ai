// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package planner

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadCandidates(t *testing.T) {
	// 1. Array of plans
	arrayData := []byte(`[
		{"format": "vxpu-plan/1", "name": "plan-a", "ranges": [{"gpu": 0, "layers": "0:8"}]},
		{"format": "vxpu-plan/1", "name": "plan-b", "ranges": [{"gpu": 0, "layers": "0:16"}]}
	]`)
	cands, err := ReadCandidates(arrayData)
	if err != nil {
		t.Fatalf("ReadCandidates array: %v", err)
	}
	if len(cands) != 2 {
		t.Fatalf("want 2 candidates, got %d", len(cands))
	}
	if cands[0].Name != "plan-a" || cands[1].Name != "plan-b" {
		t.Errorf("unexpected names: %q, %q", cands[0].Name, cands[1].Name)
	}

	// 2. Object with candidates array wrapping nested plans
	wrapperData := []byte(`{
		"candidates": [
			{"name": "cand-1", "plan": {"format": "vxpu-plan/1", "knobs": {"FUSED": 1}}},
			{"name": "cand-2", "plan": {"format": "vxpu-plan/1", "knobs": {"FUSED": 0}}}
		]
	}`)
	cands, err = ReadCandidates(wrapperData)
	if err != nil {
		t.Fatalf("ReadCandidates wrapper: %v", err)
	}
	if len(cands) != 2 {
		t.Fatalf("want 2 candidates, got %d", len(cands))
	}
	if cands[0].Name != "cand-1" {
		t.Errorf("want cand-1, got %q", cands[0].Name)
	}

	// 3. Single plan
	singleData := []byte(`{"format": "vxpu-plan/1", "ranges": [{"gpu": 0, "layers": "0:8"}]}`)
	cands, err = ReadCandidates(singleData)
	if err != nil {
		t.Fatalf("ReadCandidates single: %v", err)
	}
	if len(cands) != 1 {
		t.Fatalf("want 1 candidate, got %d", len(cands))
	}
}

func TestCheckGates(t *testing.T) {
	tests := []struct {
		name       string
		checks     any
		rec        map[string]any
		wantPassed bool
		wantSubstr string
	}{
		{
			name:       "clean passing string checks",
			checks:     "checks: golden=ran, duprow=on, ref=every 100, floor: 0 sites above 2x",
			wantPassed: true,
		},
		{
			name:       "skipped golden check is hard error",
			checks:     "checks: golden=skipped:no golden.json, duprow=on, ref=off",
			wantPassed: false,
			wantSubstr: "skipped check",
		},
		{
			name:       "skipped check with override is hard error",
			checks:     "checks: golden=ran, duprow=off(override), ref=off",
			wantPassed: false,
			wantSubstr: "skipped check",
		},
		{
			name:       "explicit mismatch in checks line",
			checks:     "checks: golden=failed:mismatch at position 26, duprow=on, ref=off",
			wantPassed: false,
			wantSubstr: "mismatch",
		},
		{
			name:       "exactness failed string",
			checks:     "checks: golden=ran, duprow=on, exactness=failed: drafted != undrafted",
			wantPassed: false,
			wantSubstr: "exactness mismatch",
		},
		{
			name:       "logit error under tolerance passes",
			checks:     "checks: golden=ran, duprow=on, ref=every 100, logit_error=0.0125",
			wantPassed: true,
		},
		{
			name:       "logit error over tolerance fails",
			checks:     "checks: golden=ran, duprow=on, ref=every 100, logit_error=0.0380",
			wantPassed: false,
			wantSubstr: "logit error 0.0380 exceeds tolerance 0.025",
		},
		{
			name:   "top level match prompt count mismatch fails",
			checks: "checks: golden=ran, duprow=on, ref=every 100",
			rec: map[string]any{
				"match": 31,
				"of":    32,
			},
			wantPassed: false,
			wantSubstr: "exactness mismatch (31 of 32 prompts matched)",
		},
		{
			name:   "top level same_rows false fails",
			checks: "checks: golden=ran, duprow=on, ref=every 100",
			rec: map[string]any{
				"same_rows": false,
			},
			wantPassed: false,
			wantSubstr: "duplicate rows invariant failed",
		},
		{
			name: "map checks passing",
			checks: map[string]any{
				"golden":    "ran",
				"duprow":    "on",
				"exactness": true,
			},
			wantPassed: true,
		},
		{
			name: "map checks skipped",
			checks: map[string]any{
				"golden": "skipped:no file",
				"duprow": "on",
			},
			wantPassed: false,
			wantSubstr: "skipped",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res := CheckGates(tc.checks, tc.rec)
			if res.Passed != tc.wantPassed {
				t.Fatalf("CheckGates() passed = %v, want %v (reason: %s)", res.Passed, tc.wantPassed, res.Reason)
			}
			if !tc.wantPassed && tc.wantSubstr != "" && !strings.Contains(res.Reason, tc.wantSubstr) {
				t.Errorf("CheckGates() reason = %q, want substring %q", res.Reason, tc.wantSubstr)
			}
		})
	}
}

func TestReadRunRecordRefusesMissingChecks(t *testing.T) {
	// Missing checks field
	noChecks := []byte(`{"median_step_ms": 14.5}`)
	_, err := ReadRunRecord(noChecks)
	if err == nil {
		t.Fatalf("expected error for record missing checks, got nil")
	}
	if !strings.Contains(err.Error(), "checks") {
		t.Errorf("expected missing checks error, got: %v", err)
	}

	// Present checks field
	withChecks := []byte(`{"median_step_ms": 14.5, "checks": "checks: golden=ran, duprow=on, ref=off"}`)
	rec, err := ReadRunRecord(withChecks)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rec["checks"] == nil {
		t.Errorf("checks field missing from parsed map")
	}
}

func TestEvaluateLoopWithFakeExecutor(t *testing.T) {
	dir := t.TempDir()

	controlPath := filepath.Join(dir, "control.json")
	controlRecord := map[string]any{
		"median_step_ms": 15.0,
		"checks":         "checks: golden=ran, duprow=on, ref=every 100",
	}
	controlBytes, _ := json.Marshal(controlRecord)
	if err := os.WriteFile(controlPath, controlBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	candidatesPath := filepath.Join(dir, "candidates.json")
	candidates := []map[string]any{
		{
			"name": "cand-fast-pass",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"id":     "fast-pass",
			},
		},
		{
			"name": "cand-fast-gate-fail",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"id":     "fast-gate-fail",
			},
		},
		{
			"name": "cand-slow-pass",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"id":     "slow-pass",
			},
		},
		{
			"name": "cand-build-fail",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"id":     "build-fail",
			},
		},
	}
	candsBytes, _ := json.Marshal(candidates)
	if err := os.WriteFile(candidatesPath, candsBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	// Create fake executor script
	fakeExecScript := filepath.Join(dir, "fake_executor.sh")
	scriptContent := `#!/bin/sh
set -e

PLAN_FILE=""
OUT_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --plan) PLAN_FILE="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -z "$PLAN_FILE" ]; then
  PLAN_FILE="$VXPU_PLAN"
fi
if [ -z "$OUT_FILE" ]; then
  OUT_FILE="$VXPU_PLAN_RECORD"
fi

if grep -q "build-fail" "$PLAN_FILE"; then
  echo "fatal: compilation error in lowering" >&2
  exit 1
fi

if grep -q "fast-pass" "$PLAN_FILE"; then
  echo '{"median_step_ms": 14.2, "checks": "checks: golden=ran, duprow=on, ref=every 100, exactness=passed"}' > "$OUT_FILE"
  exit 0
fi

if grep -q "fast-gate-fail" "$PLAN_FILE"; then
  echo '{"median_step_ms": 13.5, "checks": "checks: golden=ran, duprow=on, exactness=failed: drafted != undrafted"}' > "$OUT_FILE"
  exit 0
fi

if grep -q "slow-pass" "$PLAN_FILE"; then
  echo '{"median_step_ms": 14.8, "checks": "checks: golden=ran, duprow=on, ref=every 100"}' > "$OUT_FILE"
  exit 0
fi

echo "unknown plan" >&2
exit 2
`
	if err := os.WriteFile(fakeExecScript, []byte(scriptContent), 0o755); err != nil {
		t.Fatal(err)
	}

	currentPlanPath := filepath.Join(dir, "current_plan.json")
	ledgerPath := filepath.Join(dir, "ledger.jsonl")

	opts := EvaluationOptions{
		CandidatesPath:  candidatesPath,
		ControlPath:     controlPath,
		CurrentPlanPath: currentPlanPath,
		LedgerPath:      ledgerPath,
		ExecutorCmd:     fakeExecScript,
	}

	verdicts, err := Evaluate(t.Context(), opts)
	if err != nil {
		t.Fatalf("Evaluate returned unexpected error: %v", err)
	}

	if len(verdicts) != 4 {
		t.Fatalf("want 4 verdicts, got %d", len(verdicts))
	}

	// Verdict 0: fast-pass -> kept
	if verdicts[0].Verdict != "kept" {
		t.Errorf("candidate 0: want kept, got %s (reason: %s)", verdicts[0].Verdict, verdicts[0].Reason)
	}
	if verdicts[0].StepMs != 14.2 {
		t.Errorf("candidate 0: want stepMs 14.2, got %v", verdicts[0].StepMs)
	}
	if verdicts[0].BestStepMs != 15.0 {
		t.Errorf("candidate 0: want bestStepMs 15.0, got %v", verdicts[0].BestStepMs)
	}

	// Verdict 1: fast-gate-fail -> rejected with reason "gate failed"
	if verdicts[1].Verdict != "rejected" {
		t.Errorf("candidate 1: want rejected, got %s", verdicts[1].Verdict)
	}
	if !strings.Contains(verdicts[1].Reason, "gate failed") {
		t.Errorf("candidate 1: want gate failed reason, got %q", verdicts[1].Reason)
	}

	// Verdict 2: slow-pass -> rejected with reason "slower" (compared against 14.2)
	if verdicts[2].Verdict != "rejected" {
		t.Errorf("candidate 2: want rejected, got %s", verdicts[2].Verdict)
	}
	if !strings.Contains(verdicts[2].Reason, "slower") {
		t.Errorf("candidate 2: want slower reason, got %q", verdicts[2].Reason)
	}

	// Verdict 3: build-fail -> rejected with reason "build failed"
	if verdicts[3].Verdict != "rejected" {
		t.Errorf("candidate 3: want rejected, got %s", verdicts[3].Verdict)
	}
	if !strings.Contains(verdicts[3].Reason, "build failed") {
		t.Errorf("candidate 3: want build failed reason, got %q", verdicts[3].Reason)
	}

	// Verify current plan file contains candidate 0's plan (the kept plan)
	keptPlanData, err := os.ReadFile(currentPlanPath)
	if err != nil {
		t.Fatalf("reading current plan file: %v", err)
	}
	if !strings.Contains(string(keptPlanData), "fast-pass") {
		t.Errorf("current plan file does not contain fast-pass: %s", string(keptPlanData))
	}

	// Verify ledger file has all 4 verdicts
	ledgerData, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("reading ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledgerData)), "\n")
	if len(lines) != 4 {
		t.Fatalf("want 4 lines in ledger, got %d", len(lines))
	}
	var v0 Verdict
	if err := json.Unmarshal([]byte(lines[0]), &v0); err != nil {
		t.Fatalf("parsing ledger line 0: %v", err)
	}
	if v0.Verdict != "kept" {
		t.Errorf("ledger line 0: want kept, got %s", v0.Verdict)
	}
}

func TestEvaluateRefusesCandidateWithoutChecks(t *testing.T) {
	dir := t.TempDir()

	controlPath := filepath.Join(dir, "control.json")
	controlRecord := map[string]any{
		"median_step_ms": 15.0,
		"checks":         "checks: golden=ran, duprow=on, ref=every 100",
	}
	controlBytes, _ := json.Marshal(controlRecord)
	os.WriteFile(controlPath, controlBytes, 0o644)

	candidatesPath := filepath.Join(dir, "candidates.json")
	candidates := []map[string]any{
		{
			"name": "cand-no-checks",
			"plan": map[string]any{"format": "vxpu-plan/1", "id": "no-checks"},
		},
	}
	candsBytes, _ := json.Marshal(candidates)
	os.WriteFile(candidatesPath, candsBytes, 0o644)

	fakeExecScript := filepath.Join(dir, "fake_executor.sh")
	scriptContent := `#!/bin/sh
OUT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [ -z "$OUT_FILE" ]; then
  OUT_FILE="$VXPU_PLAN_RECORD"
fi
echo '{"median_step_ms": 12.0}' > "$OUT_FILE"
`
	os.WriteFile(fakeExecScript, []byte(scriptContent), 0o755)

	opts := EvaluationOptions{
		CandidatesPath: candidatesPath,
		ControlPath:    controlPath,
		ExecutorCmd:    fakeExecScript,
	}

	_, err := Evaluate(t.Context(), opts)
	if err == nil {
		t.Fatalf("expected Evaluate to refuse record without checks field, got nil")
	}
	if !strings.Contains(err.Error(), "record refused") || !strings.Contains(err.Error(), "checks") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestEvaluateHero15FoldVerdict(t *testing.T) {
	// Acceptance test:
	// Two candidates (the fold on and off at 8 streams) evaluated against the control reproduce the hero 15 verdict.
	// Hero 15 at 8 streams: control (direct/unfolded) is 29.21 ms.
	// Fold on (fused push on) is slower at 29.62 ms (0.4 ms slower) -> rejected: slower.
	// Fold off (fused push off) wins and improves to 29.12 ms -> kept.
	dir := t.TempDir()

	controlPath := filepath.Join(dir, "control.json")
	controlRecord := map[string]any{
		"median_step_ms": 29.21,
		"checks":         "checks: golden=ran, duprow=on, ref=every 100",
	}
	controlBytes, _ := json.Marshal(controlRecord)
	os.WriteFile(controlPath, controlBytes, 0o644)

	candidatesPath := filepath.Join(dir, "candidates.json")
	candidates := []map[string]any{
		{
			"name": "fold-on-8-streams",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"knobs":  map[string]any{"FUSED_PUSH": 1},
			},
		},
		{
			"name": "fold-off-8-streams",
			"plan": map[string]any{
				"format": "vxpu-plan/1",
				"knobs":  map[string]any{"FUSED_PUSH": 0},
			},
		},
	}
	candsBytes, _ := json.Marshal(candidates)
	os.WriteFile(candidatesPath, candsBytes, 0o644)

	fakeExecScript := filepath.Join(dir, "hero15_executor.sh")
	scriptContent := `#!/bin/sh
set -e
PLAN_FILE=""
OUT_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --plan) PLAN_FILE="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -z "$PLAN_FILE" ]; then
  PLAN_FILE="$VXPU_PLAN"
fi
if [ -z "$OUT_FILE" ]; then
  OUT_FILE="$VXPU_PLAN_RECORD"
fi

if grep -q 'FUSED_PUSH.*1' "$PLAN_FILE"; then
  # Fold on: slower at 8 streams (29.62 ms vs control 29.21 ms)
  echo '{"median_step_ms": 29.62, "checks": "checks: golden=ran, duprow=on, ref=every 100"}' > "$OUT_FILE"
  exit 0
fi

if grep -q 'FUSED_PUSH.*0' "$PLAN_FILE"; then
  # Fold off: faster at 8 streams (29.12 ms vs control 29.21 ms)
  echo '{"median_step_ms": 29.12, "checks": "checks: golden=ran, duprow=on, ref=every 100"}' > "$OUT_FILE"
  exit 0
fi

exit 1
`
	os.WriteFile(fakeExecScript, []byte(scriptContent), 0o755)

	planPath := filepath.Join(dir, "current_plan.json")
	ledgerPath := filepath.Join(dir, "ledger.jsonl")

	opts := EvaluationOptions{
		CandidatesPath:  candidatesPath,
		ControlPath:     controlPath,
		CurrentPlanPath: planPath,
		LedgerPath:      ledgerPath,
		ExecutorCmd:     fakeExecScript,
	}

	verdicts, err := Evaluate(t.Context(), opts)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	if len(verdicts) != 2 {
		t.Fatalf("want 2 verdicts, got %d", len(verdicts))
	}

	// fold-on is rejected as slower
	if verdicts[0].Verdict != "rejected" || !strings.Contains(verdicts[0].Reason, "slower") {
		t.Errorf("fold-on verdict: want rejected slower, got %s: %s", verdicts[0].Verdict, verdicts[0].Reason)
	}

	// fold-off is kept (improves time from 29.21 to 29.12)
	if verdicts[1].Verdict != "kept" {
		t.Errorf("fold-off verdict: want kept, got %s: %s", verdicts[1].Verdict, verdicts[1].Reason)
	}
	if verdicts[1].StepMs != 29.12 {
		t.Errorf("fold-off stepMs: want 29.12, got %v", verdicts[1].StepMs)
	}

	// Verify kept plan written to planPath
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("reading plan file: %v", err)
	}
	if !strings.Contains(string(data), `"FUSED_PUSH": 0`) {
		t.Errorf("kept plan does not have fold-off: %s", string(data))
	}
}
