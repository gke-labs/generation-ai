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

package kernelplan

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestEmitFromArtifact(t *testing.T) {
	manifest := `{
		"tensors": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"e0.w1": {"length": 100}, "e1.w1": {"length": 100}
		},
		"config": {
			"model_type": "deepseek_v41",
			"n_layers": 1,
			"dim": 128,
			"n_heads": 4,
			"head_dim": 32,
			"n_routed_experts": 8,
			"moe_inter_dim": 64,
			"engram_layer_ids": [0],
			"engram_vocab_size": 1000
		}
	}`
	binding := `{
		"bound": {
			"model.layers.0.self_attn.q_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"},
			"model.layers.0.mlp.down_proj.weight": {"length": 2048, "shape": [128, 128], "dtype": "F8_E4M3"}
		},
		"external": {
			"experts": {
				"0": {
					"w1": {"parts": [["e0.w1"], ["e1.w1"]]}
				}
			},
			"engram": {
				"0": {}
			}
		}
	}`

	plan, err := EmitFromArtifact([]byte(manifest), []byte(binding), "commit123", []int{1, 5, 8})
	if err != nil {
		t.Fatalf("EmitFromArtifact failed: %v", err)
	}

	if plan.Format != PlanFormat {
		t.Errorf("plan.Format = %q, want %q", plan.Format, PlanFormat)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan.TreeCommit = %q, want commit123", plan.TreeCommit)
	}
	wantMD5 := ComputeArtifactMD5([]byte(manifest), []byte(binding))
	if plan.ArtifactMD5 != wantMD5 {
		t.Errorf("plan.ArtifactMD5 = %q, want %q", plan.ArtifactMD5, wantMD5)
	}

	classesSeen := make(map[string]int)
	for _, s := range plan.Sites {
		classesSeen[s.Class]++
		if !slices.Equal(s.RowCounts, []int{1, 5, 8}) {
			t.Errorf("site %s row counts = %v, want [1, 5, 8]", s.ID, s.RowCounts)
		}
		if s.Strategy != "default" {
			t.Errorf("site %s strategy = %q, want default", s.ID, s.Strategy)
		}
		if plan.Strategies[s.ID] != s.Strategy {
			t.Errorf("strategies map for %s = %q, want %q", s.ID, plan.Strategies[s.ID], s.Strategy)
		}
	}

	for _, c := range []string{ClassDenseGEMV, ClassExpertOp, ClassSumSite, ClassAttention, ClassDrafter} {
		if classesSeen[c] == 0 {
			t.Errorf("expected at least one site of class %q, got 0", c)
		}
	}
}

func TestValidateRecord(t *testing.T) {
	valid := RunRecord{
		Format:        RecordFormat,
		SchemaVersion: 1,
		PlanHash:      "hash123",
		TreeCommit:    "commit123",
		ArtifactMD5:   "md5_abc",
		KnobsInEffect: map[string]string{
			"VXPU_FUSED_EXPERTS": "1",
		},
		MedianStepMs: 12.34,
		TokensAStep:  1.0,
		Checks: map[string]any{
			"status": "pass",
		},
	}
	data, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}

	rec, err := ValidateRecord(data)
	if err != nil {
		t.Fatalf("ValidateRecord failed on valid record: %v", err)
	}
	if rec.PlanHash != "hash123" {
		t.Errorf("rec.PlanHash = %q, want hash123", rec.PlanHash)
	}

	// Missing schema / unversioned
	unversioned := valid
	unversioned.Format = ""
	unversioned.SchemaVersion = 0
	data, _ = json.Marshal(unversioned)
	if _, err := ValidateRecord(data); err == nil {
		t.Error("expected error for unversioned record, got nil")
	}

	// Missing checks field
	noChecks := valid
	noChecks.Checks = nil
	data, _ = json.Marshal(noChecks)
	if _, err := ValidateRecord(data); err == nil {
		t.Error("expected error for record missing checks, got nil")
	}

	// Missing plan_hash
	noHash := valid
	noHash.PlanHash = ""
	data, _ = json.Marshal(noHash)
	if _, err := ValidateRecord(data); err == nil {
		t.Error("expected error for record missing plan_hash, got nil")
	}
}

func TestAppendLedger(t *testing.T) {
	tmpDir := t.TempDir()
	ledgerPath := filepath.Join(tmpDir, "ledger.jsonl")

	rec := &RunRecord{
		Format:        RecordFormat,
		SchemaVersion: 1,
		PlanHash:      "hash456",
		TreeCommit:    "commit456",
		ArtifactMD5:   "md5_def",
		KnobsInEffect: map[string]string{
			"VXPU_ALLREDUCE_SCATTER_MIN_ROWS": "16",
		},
		MedianStepMs: 28.6,
		TokensAStep:  1.0,
		Checks: map[string]any{
			"exactness": true,
		},
	}

	if err := AppendLedger(ledgerPath, rec); err != nil {
		t.Fatalf("AppendLedger failed: %v", err)
	}

	file, err := os.Open(ledgerPath)
	if err != nil {
		t.Fatalf("Open ledger failed: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		t.Fatal("expected at least one line in ledger")
	}

	readRec, err := ValidateRecord(scanner.Bytes())
	if err != nil {
		t.Fatalf("ValidateRecord on read line failed: %v", err)
	}
	if readRec.PlanHash != "hash456" {
		t.Errorf("readRec.PlanHash = %q, want hash456", readRec.PlanHash)
	}
	if readRec.MedianStepMs != 28.6 {
		t.Errorf("readRec.MedianStepMs = %f, want 28.6", readRec.MedianStepMs)
	}
}
