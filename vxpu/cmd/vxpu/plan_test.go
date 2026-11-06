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
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func TestCmdPlanEmitAndRecord(t *testing.T) {
	tmpDir := t.TempDir()

	manifest := `{
		"tensors": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		},
		"config": {
			"model_type": "deepseek_v41",
			"n_layers": 1,
			"dim": 128
		}
	}`
	binding := `{
		"bound": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
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
	ctx := t.Context()

	// 1. Emit
	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath, "--rows", "1,5,8"}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("ReadFile plan failed: %v", err)
	}

	var plan kernelplan.Plan
	if err := json.Unmarshal(planBytes, &plan); err != nil {
		t.Fatalf("Unmarshal plan failed: %v", err)
	}
	if plan.Format != kernelplan.PlanFormat {
		t.Errorf("plan format = %q, want %q", plan.Format, kernelplan.PlanFormat)
	}
	if len(plan.Sites) == 0 {
		t.Error("expected at least one site in plan")
	}

	planHash, err := kernelplan.HashPlan(&plan)
	if err != nil {
		t.Fatalf("HashPlan failed: %v", err)
	}

	// 2. Record
	runRec := kernelplan.RunRecord{
		Format:        kernelplan.RecordFormat,
		SchemaVersion: 1,
		PlanHash:      planHash,
		TreeCommit:    plan.TreeCommit,
		ArtifactMD5:   plan.ArtifactMD5,
		KnobsInEffect: map[string]string{"VXPU_FUSED_EXPERTS": "1"},
		MedianStepMs:  15.5,
		TokensAStep:   1.0,
		Checks:        map[string]any{"status": "pass"},
	}
	runPath := filepath.Join(tmpDir, "run.json")
	runBytes, err := json.Marshal(runRec)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(runPath, runBytes, 0644); err != nil {
		t.Fatal(err)
	}

	ledgerPath := filepath.Join(tmpDir, "ledger.jsonl")
	if err := cmdPlan(ctx, []string{"record", "--in", runPath, "--ledger", ledgerPath}); err != nil {
		t.Fatalf("cmdPlan record failed: %v", err)
	}

	ledgerBytes, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("ReadFile ledger failed: %v", err)
	}
	if len(ledgerBytes) == 0 {
		t.Error("expected non-empty ledger file")
	}
}
