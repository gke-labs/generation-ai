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
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const testTreeCommit = "0123456789abcdef0123456789abcdef01234567"

// The testdata fixture is modelled on the exporter's output: the names, dtypes,
// ranks and expert-bank structure as export_deepseek_v41 writes them; the wo_a,
// wo_b, scale and expert shapes are the real artifact's, the other shapes are
// illustrative.
func TestEmitGolden(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	plan, err := Emit(fixtureDir, testTreeCommit, nil)
	if err != nil {
		t.Fatalf("Emit failed: %v", err)
	}

	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent failed: %v", err)
	}
	data = append(data, '\n')

	goldenPath := filepath.Join(fixtureDir, "golden_plan.json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.WriteFile(goldenPath, data, 0644); err != nil {
				t.Fatalf("WriteFile golden failed: %v", err)
			}
			goldenBytes = data
		} else {
			t.Fatalf("ReadFile golden failed: %v", err)
		}
	}

	if !bytes.Equal(data, goldenBytes) {
		t.Errorf("emitted plan does not match golden file %s\ngot:\n%s\nwant:\n%s", goldenPath, string(data), string(goldenBytes))
	}
}

func TestLinearGEMVFiltering(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	plan, err := Emit(fixtureDir, testTreeCommit, nil)
	if err != nil {
		t.Fatalf("Emit failed: %v", err)
	}

	var gemvSites []string
	for _, s := range plan.Sites {
		if s.Class == ClassDenseGEMV {
			gemvSites = append(gemvSites, s.ID)
		}
	}

	wantGEMV := []string{
		"model.head.weight",
		"model.layers.0.attn.wkv.weight",
		"model.layers.0.attn.wo_a.weight",
		"model.layers.0.attn.wo_b.weight",
		"model.layers.0.attn.wq_a.weight",
		"model.layers.0.attn.wq_b.weight",
		"model.layers.0.ffn.gate.weight",
		"model.layers.0.ffn.shared_experts.w1.weight",
		"model.layers.1.attn.wo_a.weight",
		"model.layers.1.attn.wo_b.weight",
		"model.layers.1.attn.wq_a.weight",
		"model.layers.1.ffn.gate.weight",
	}
	if !slices.Equal(gemvSites, wantGEMV) {
		t.Errorf("gemvSites = %v, want %v", gemvSites, wantGEMV)
	}

	for _, s := range plan.Sites {
		if s.ID == "model.embed.weight" || s.ID == "model.layers.0.attn.wo_a.scale" || s.ID == "model.layers.0.hc_attn_scale" {
			t.Errorf("non-linear or scale tensor %q should not be emitted as a site", s.ID)
		}
	}
}

func TestStackedMoEExpertSites(t *testing.T) {
	manifest := []byte(`{
		"tensors": {
			"model.layers.0.mlp.experts.0.down_proj.weight": {"shape": [2048, 768]},
			"model.layers.0.mlp.experts.0.gate_proj.weight": {"shape": [768, 2048]}
		}
	}`)
	binding := []byte(`{
		"bound": {},
		"stacked": {
			"model.layers.0.mlp.experts.gate_up_proj": {
				"parts": [
					["model.layers.0.mlp.experts.0.gate_proj.weight"],
					["model.layers.0.mlp.experts.0.gate_proj.weight"]
				]
			},
			"model.layers.0.mlp.experts.down_proj": {
				"parts": [
					["model.layers.0.mlp.experts.0.down_proj.weight"],
					["model.layers.0.mlp.experts.0.down_proj.weight"]
				]
			}
		}
	}`)

	// Run multiple times to verify deterministic selection (down_proj sorts before gate_up_proj)
	for range 50 {
		plan, err := EmitFromArtifact(manifest, binding, testTreeCommit, nil)
		if err != nil {
			t.Fatalf("EmitFromArtifact failed: %v", err)
		}

		var expertSites []Site
		for _, s := range plan.Sites {
			if s.Class == ClassExpertOp {
				expertSites = append(expertSites, s)
			}
		}
		if len(expertSites) != 1 {
			t.Fatalf("expected 1 expert site, got %d", len(expertSites))
		}
		site := expertSites[0]
		if site.ID != "layers.0.experts" {
			t.Errorf("site.ID = %q, want layers.0.experts", site.ID)
		}
		wantShape := []int{2, 2048, 768}
		if !slices.Equal(site.Shape, wantShape) {
			t.Fatalf("site.Shape = %v, want %v", site.Shape, wantShape)
		}
	}
}

func TestUnknownVersionRefused(t *testing.T) {
	invalidJSON := []byte(`{
		"format": "vxpu-plan/unknown",
		"artifact_md5": "abc123",
		"tree_commit": "commit456",
		"sites": []
	}`)
	if _, err := Parse(invalidJSON); err == nil {
		t.Error("expected error for unknown plan format, got nil")
	}

	unversionedJSON := []byte(`{
		"format": "",
		"artifact_md5": "abc123",
		"tree_commit": "commit456",
		"sites": []
	}`)
	if _, err := Parse(unversionedJSON); err == nil {
		t.Error("expected error for unversioned plan format, got nil")
	}
}

func TestRefuseUnknownTreeCommit(t *testing.T) {
	manifest := []byte(`{"tensors": {}}`)
	binding := []byte(`{"bound": {}}`)

	if _, err := EmitFromArtifact(manifest, binding, "unknown", nil); err == nil {
		t.Error("expected error refusing unknown tree_commit, got nil")
	}
	if _, err := EmitFromArtifact(manifest, binding, "", nil); err == nil {
		t.Error("expected error refusing empty tree_commit, got nil")
	}
}
