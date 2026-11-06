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
	"math"
	"path/filepath"
	"strings"
	"testing"
)

func TestMeasureOverlapHero15Block7(t *testing.T) {
	fixturePath := filepath.Join("testdata", "block7-hero15-run.json")
	_, rec, err := ReadRunRecordFile(fixturePath)
	if err != nil {
		t.Fatalf("ReadRunRecordFile(%s) failed: %v", fixturePath, err)
	}

	// 1. Busy: sum of spans: 6.34 + 2.31 + 1.08 + 0.48 + 1.26 + 2.63 + 0.78 + 0.76 + 0.75 = 16.39 ms
	if math.Abs(rec.BusyMs-16.39) > 0.05 {
		t.Errorf("BusyMs = %f, want ~16.39", rec.BusyMs)
	}

	// 2. Covered: union of spans: 6.34 + 1.08 + 1.26 + 2.63 + 0.78 + 0.76 + 0.75 = 13.60 ms
	if math.Abs(rec.CoveredMs-13.60) > 0.05 {
		t.Errorf("CoveredMs = %f, want ~13.60", rec.CoveredMs)
	}

	// 3. Wall: 14.45 ms
	if math.Abs(rec.WallMs-14.45) > 0.05 {
		t.Errorf("WallMs = %f, want ~14.45", rec.WallMs)
	}

	// 4. Overlap: busy - covered = 16.39 - 13.60 = 2.79 ms (~3 ms)
	if math.Abs(rec.OverlapMs-2.79) > 0.05 {
		t.Errorf("OverlapMs = %f, want ~2.79", rec.OverlapMs)
	}

	// 5. Idle: wall - covered = 14.45 - 13.60 = 0.85 ms
	if math.Abs(rec.IdleMs-0.85) > 0.05 {
		t.Errorf("IdleMs = %f, want ~0.85", rec.IdleMs)
	}

	// 6. Hidden work: shared_ffn_front_0 lies entirely inside experts_0
	var foundSharedHidden bool
	for _, h := range rec.HiddenNodes {
		if h.Name == "shared_ffn_front_0" {
			foundSharedHidden = true
			if h.Inside != "experts_0" {
				t.Errorf("shared_ffn_front_0 hidden inside %s, want experts_0", h.Inside)
			}
			if math.Abs(h.DurationMs-2.31) > 0.05 {
				t.Errorf("shared_ffn_front_0 duration = %f, want ~2.31", h.DurationMs)
			}
		}
	}
	if !foundSharedHidden {
		t.Errorf("expected shared_ffn_front_0 in hidden nodes, got %+v", rec.HiddenNodes)
	}

	// 7. Per-class metrics
	sharedClass, ok := rec.Classes["shared_expert"]
	if !ok {
		t.Fatalf("missing shared_expert class in %+v", rec.Classes)
	}
	if math.Abs(sharedClass.SideBranchShare-1.0) > 0.01 {
		t.Errorf("shared_expert SideBranchShare = %f, want 1.0", sharedClass.SideBranchShare)
	}

	attnProj, ok := rec.Classes["attention_projections"]
	if !ok {
		t.Fatalf("missing attention_projections class in %+v", rec.Classes)
	}
	if attnProj.SideBranchMs <= 0 || attnProj.SideBranchShare <= 0 {
		t.Errorf("expected attention_projections side-branch work, got %+v", attnProj)
	}

	// Format table
	output := FormatOverlapTable(rec)
	if !strings.Contains(output, "Step concurrency:") {
		t.Errorf("expected 'Step concurrency:' in formatted table:\n%s", output)
	}
	if !strings.Contains(output, "shared_ffn_front_0") {
		t.Errorf("expected 'shared_ffn_front_0' in formatted table:\n%s", output)
	}
	if !strings.Contains(output, "experts_0") {
		t.Errorf("expected 'experts_0' in formatted table:\n%s", output)
	}
}

func TestClassifyNode(t *testing.T) {
	tests := []struct {
		name string
		kind string
		ops  []string
		want string
	}{
		{"experts_0", "experts", nil, "routed_experts"},
		{"shared_ffn_front_1", "dense", []string{"shared_ffn_front"}, "shared_expert"},
		{"attn_q_2", "dense", []string{"attn_q"}, "attention_projections"},
		{"window_kv_2", "dense", []string{"window_kv"}, "attention_projections"},
		{"attn_core_2", "dense", []string{"attn_core"}, "attention_core"},
		{"allreduce_1", "allreduce", nil, "sums"},
		{"ffn_front_3", "dense", []string{"ffn_front"}, "ffn_front"},
		{"hc_front_3", "dense", []string{"hc_front"}, "hc_front"},
		{"engram_0", "engram", nil, "engram"},
		{"other_node", "dense", []string{"something_else"}, "dense_other"},
	}

	for _, tc := range tests {
		got := ClassifyNode(tc.name, tc.kind, tc.ops)
		if got != tc.want {
			t.Errorf("ClassifyNode(%q, %q, %v) = %q, want %q", tc.name, tc.kind, tc.ops, got, tc.want)
		}
	}
}
