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

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCmdPlanEvaluate(t *testing.T) {
	ctx := t.Context()

	// 1. Missing flags should fail
	if err := cmdPlan(ctx, []string{"evaluate"}); err == nil {
		t.Fatalf("cmdPlan evaluate with no flags should fail")
	}

	dir := t.TempDir()
	controlPath := filepath.Join(dir, "control.json")
	controlRecord := map[string]any{
		"median_step_ms": 15.0,
		"checks":         "checks: golden=ran, duprow=on, ref=every 100",
	}
	data, _ := json.Marshal(controlRecord)
	os.WriteFile(controlPath, data, 0o644)

	candidatesPath := filepath.Join(dir, "candidates.json")
	candidates := []map[string]any{
		{
			"name": "faster-candidate",
			"plan": map[string]any{"format": "vxpu-plan/1", "name": "fast"},
		},
	}
	data, _ = json.Marshal(candidates)
	os.WriteFile(candidatesPath, data, 0o644)

	fakeExecutor := filepath.Join(dir, "exec.sh")
	script := `#!/bin/sh
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
echo '{"median_step_ms": 13.8, "checks": "checks: golden=ran, duprow=on, ref=every 100"}' > "$OUT_FILE"
`
	os.WriteFile(fakeExecutor, []byte(script), 0o755)

	planPath := filepath.Join(dir, "plan.json")
	ledgerPath := filepath.Join(dir, "ledger.jsonl")

	args := []string{
		"evaluate",
		"--candidates", candidatesPath,
		"--control", controlPath,
		"--plan", planPath,
		"--ledger", ledgerPath,
		"--executor", fakeExecutor,
	}

	if err := cmdPlan(ctx, args); err != nil {
		t.Fatalf("cmdPlan evaluate failed: %v", err)
	}

	// Verify plan was written
	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("reading written plan file: %v", err)
	}
	if !strings.Contains(string(planBytes), "fast") {
		t.Errorf("plan file does not contain candidate plan: %s", string(planBytes))
	}

	// Verify ledger was written
	ledgerBytes, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("reading ledger: %v", err)
	}
	if !strings.Contains(string(ledgerBytes), `"verdict":"kept"`) {
		t.Errorf("ledger does not contain kept verdict: %s", string(ledgerBytes))
	}
}
