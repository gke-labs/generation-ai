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
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func TestEndToEndSkeletonRoundTrip(t *testing.T) {
	pythonPath, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not found")
	}

	tmpDir := t.TempDir()
	manifest := `{
		"tensors": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		},
		"config": {
			"model_type": "deepseek_v41",
			"n_layers": 1,
			"dim": 128
		}
	}`
	binding := `{
		"bound": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		}
	}`

	artDir := filepath.Join(tmpDir, "artifact")
	if err := os.MkdirAll(artDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artDir, "manifest.json"), []byte(manifest), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artDir, "binding.json"), []byte(binding), 0644); err != nil {
		t.Fatal(err)
	}

	planPath := filepath.Join(tmpDir, "plan.json")
	runPath := filepath.Join(tmpDir, "run.json")
	ledgerPath := filepath.Join(tmpDir, "ledger.jsonl")
	ctx := t.Context()

	// 1. vxpu plan emit
	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	// 2. Python executor apply_env and record write
	pyScript := fmt.Sprintf(`
import os
import sys
from vxpu import physplan

plan_path = %q
run_path = %q

plan = physplan.load_plan(plan_path)
fake_env = {}
knobs = physplan.apply_env(plan, env=fake_env)
assert "VXPU_FUSED_EXPERTS" in fake_env

record = physplan.create_run_record(
    plan=plan,
    knobs_in_effect=knobs,
    median_step_ms=21.4,
    tokens_a_step=1.0,
    checks={"status": "pass", "exactness": True},
    build_time=0.8,
)
physplan.write_run_record(record, run_path)
`, planPath, runPath)

	repoRoot, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	pythonDir := filepath.Join(repoRoot, "python")

	cmd := exec.Command(pythonPath, "-c", pyScript)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+pythonDir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Python script failed: %v\nOutput: %s", err, string(out))
	}

	// 3. vxpu plan record
	if err := cmdPlan(ctx, []string{"record", "--in", runPath, "--ledger", ledgerPath}); err != nil {
		t.Fatalf("cmdPlan record failed: %v", err)
	}

	// 4. Validate ledger content
	ledgerBytes, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("ReadFile ledger failed: %v", err)
	}
	rec, err := kernelplan.ValidateRecord(ledgerBytes)
	if err != nil {
		t.Fatalf("ValidateRecord on ledger line failed: %v", err)
	}
	if rec.MedianStepMs != 21.4 {
		t.Errorf("rec.MedianStepMs = %f, want 21.4", rec.MedianStepMs)
	}
}
