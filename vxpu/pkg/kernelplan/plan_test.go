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
	"os"
	"path/filepath"
	"slices"
	"testing"

	"google.golang.org/protobuf/reflect/protoreflect"

	planv1alpha1 "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
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

	data, err := Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

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
		if s.Kind == planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV {
			gemvSites = append(gemvSites, s.Id)
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
		if s.Id == "model.embed.weight" || s.Id == "model.layers.0.attn.wo_a.scale" || s.Id == "model.layers.0.hc_attn_scale" {
			t.Errorf("non-linear or scale tensor %q should not be emitted as a site", s.Id)
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

		var expertSites []*PlanSite
		for _, s := range plan.Sites {
			if s.Kind == planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP {
				expertSites = append(expertSites, s)
			}
		}
		if len(expertSites) != 1 {
			t.Fatalf("expected 1 expert site, got %d", len(expertSites))
		}
		site := expertSites[0]
		if site.Id != "layers.0.experts" {
			t.Errorf("site.Id = %q, want layers.0.experts", site.Id)
		}
		wantShape := []int32{2, 2048, 768}
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
		"unknown_field": "foo",
		"artifact_md5": "abc123",
		"tree_commit": "commit456",
		"sites": []
	}`)
	if _, err := Parse(unversionedJSON); err == nil {
		t.Error("expected error for unversioned plan format with unknown field, got nil")
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

func TestKernelTableRoundTrip(t *testing.T) {
	gpu := "NVIDIA H100"
	gemvRows := int32(16)
	fp4Native := false
	kt := &planv1alpha1.KernelTable{
		Gpu: &gpu,
		Knobs: &planv1alpha1.Knobs{
			GEMV_ROWS_MAX: &gemvRows,
			FP4_NATIVE:    &fp4Native,
		},
		Sites: []*planv1alpha1.SiteEntry{
			{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1.engram.wkv"), Impl: "reference"},
			{Op: "vxpu::fp8_linear", Shapes: ptr("1x8->16"), Impl: "reference"},
			{Op: "vxpu::rotary", Impl: "reference"},
		},
	}

	data, err := MarshalKernelTable(kt)
	if err != nil {
		t.Fatalf("MarshalKernelTable failed: %v", err)
	}

	parsed, err := ParseKernelTable(data)
	if err != nil {
		t.Fatalf("ParseKernelTable failed: %v", err)
	}
	if parsed.GetGpu() != gpu {
		t.Errorf("parsed gpu = %q, want %q", parsed.GetGpu(), gpu)
	}
	if parsed.GetKnobs().GetGEMV_ROWS_MAX() != 16 {
		t.Errorf("parsed GEMV_ROWS_MAX = %d, want 16", parsed.GetKnobs().GetGEMV_ROWS_MAX())
	}
	if parsed.GetKnobs().GetFP4_NATIVE() != false {
		t.Errorf("parsed FP4_NATIVE = %v, want false", parsed.GetKnobs().GetFP4_NATIVE())
	}
	if len(parsed.Sites) != 3 {
		t.Fatalf("parsed sites length = %d, want 3", len(parsed.Sites))
	}
}

func TestKnobTypes(t *testing.T) {
	types := KnobTypes()
	if len(types) != 53 {
		t.Fatalf("KnobTypes count = %d, want 53", len(types))
	}
	if types["GEMV_ROWS_MAX"] != protoreflect.Int32Kind {
		t.Errorf("GEMV_ROWS_MAX kind = %v, want Int32Kind", types["GEMV_ROWS_MAX"])
	}
	if types["FP4_NATIVE"] != protoreflect.BoolKind {
		t.Errorf("FP4_NATIVE kind = %v, want BoolKind", types["FP4_NATIVE"])
	}
}

func TestRunRecordHasPlanStructureAndMeasurements(t *testing.T) {
	recordJSON, err := os.ReadFile(filepath.Join("testdata", "run_record.json"))
	if err != nil {
		t.Fatalf("ReadFile run_record.json failed: %v", err)
	}

	rec, err := Parse(recordJSON)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if rec.GetArtifactMd5() != "artifact123" {
		t.Errorf("artifact_md5 = %q, want artifact123", rec.GetArtifactMd5())
	}
	if rec.GetTreeCommit() != "commit456" {
		t.Errorf("tree_commit = %q, want commit456", rec.GetTreeCommit())
	}
	if len(rec.Sites) != 2 {
		t.Fatalf("sites length = %d, want 2", len(rec.Sites))
	}

	// For each call site of the plan: what was chosen and what was measured,
	// in one place, with no join by name.
	site0 := rec.Sites[0]
	if site0.Id != "layers.0.experts" {
		t.Errorf("site0.Id = %q, want layers.0.experts", site0.Id)
	}
	if site0.Strategy != "fused" {
		t.Errorf("site0.Strategy = %q, want fused", site0.Strategy)
	}
	if site0.Stats == nil {
		t.Fatal("site0.Stats is nil, want non-nil")
	}
	if site0.Stats.MeasuredUs == nil {
		t.Fatal("site0.Stats.MeasuredUs is nil, want non-nil")
	}
	if *site0.Stats.MeasuredUs != 42.5 {
		t.Errorf("*site0.Stats.MeasuredUs = %f, want 42.5", *site0.Stats.MeasuredUs)
	}

	site1 := rec.Sites[1]
	if site1.Id != "model.layers.0.attn.wo_a.weight" {
		t.Errorf("site1.Id = %q, want model.layers.0.attn.wo_a.weight", site1.Id)
	}
	if site1.Strategy != "default" {
		t.Errorf("site1.Strategy = %q, want default", site1.Strategy)
	}
	// An unmeasured site reads back with no measurement field set.
	if site1.Stats != nil {
		t.Errorf("site1.Stats = %v, want nil (not measured)", site1.Stats)
	}

	// Plan-level measurements:
	if rec.Stats == nil {
		t.Fatal("rec.Stats is nil, want non-nil")
	}
	if rec.Stats.PlanHash == nil || *rec.Stats.PlanHash != "hash789" {
		t.Errorf("PlanHash = %v, want hash789", rec.Stats.PlanHash)
	}
	if rec.Stats.MedianStepMs == nil || *rec.Stats.MedianStepMs != 15.5 {
		t.Errorf("MedianStepMs = %v, want 15.5", rec.Stats.MedianStepMs)
	}
	if rec.Stats.TokensAStreamAStep == nil || *rec.Stats.TokensAStreamAStep != 1.0 {
		t.Errorf("TokensAStreamAStep = %v, want 1.0", rec.Stats.TokensAStreamAStep)
	}
	if rec.Stats.AcceptedDraftsAStep == nil || *rec.Stats.AcceptedDraftsAStep != 0.75 {
		t.Errorf("AcceptedDraftsAStep = %v, want 0.75", rec.Stats.AcceptedDraftsAStep)
	}
	if rec.Stats.GetKnobs().GetGEMV_ROWS_MAX() != 32 {
		t.Errorf("GEMV_ROWS_MAX = %d, want 32", rec.Stats.GetKnobs().GetGEMV_ROWS_MAX())
	}
	if !rec.Stats.GetChecks()["exactness"] || rec.Stats.GetChecks()["bounds"] {
		t.Errorf("Checks = %v, want exactness:true, bounds:false", rec.Stats.GetChecks())
	}
	if len(rec.Stats.PerCardPeakMemory) != 1 || rec.Stats.PerCardPeakMemory[0] != 524288000 {
		t.Errorf("PerCardPeakMemory = %v, want [524288000]", rec.Stats.PerCardPeakMemory)
	}
	if rec.Stats.PerClassDeviceTimes["dense"] != 12.3 {
		t.Errorf("PerClassDeviceTimes[dense] = %v, want 12.3", rec.Stats.PerClassDeviceTimes["dense"])
	}

	// Round-trip marshal
	marshaled, err := Marshal(rec)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	reparsed, err := Parse(marshaled)
	if err != nil {
		t.Fatalf("reparse failed: %v", err)
	}
	if reparsed.Sites[0].GetStats().GetMeasuredUs() != 42.5 {
		t.Errorf("reparsed site0 measured_us = %f, want 42.5", reparsed.Sites[0].GetStats().GetMeasuredUs())
	}
}

func TestPlanWithoutRunHasNoMeasurements(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	plan, err := Emit(fixtureDir, testTreeCommit, nil)
	if err != nil {
		t.Fatalf("Emit failed: %v", err)
	}

	// Top-level run measurements must be nil
	if plan.Stats != nil {
		t.Errorf("plan.Stats = %v, want nil", plan.Stats)
	}

	// Every site's Stats must be nil (not measured != 0)
	for _, site := range plan.Sites {
		if site.Stats != nil {
			t.Errorf("site %q has Stats = %v, want nil", site.Id, site.Stats)
		}
	}
}

// TestReadKernelRecordFixture verifies that Go can read a call-site facts
// document (KernelRecord) written by Python and checked in as test data.
func TestReadKernelRecordFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "kernel_record.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read testdata fixture %s: %v", fixturePath, err)
	}

	kr, err := ParseKernelRecord(data)
	if err != nil {
		t.Fatalf("ParseKernelRecord failed: %v", err)
	}

	if len(kr.Sites) != 67 {
		t.Fatalf("len(kr.Sites) = %d, want 67", len(kr.Sites))
	}

	// Verify the first site: facts and candidates are present, but no choices.
	s0 := kr.Sites[0]
	if s0.Op != "vxpu::hc_sinkhorn" {
		t.Errorf("s0.Op = %q, want vxpu::hc_sinkhorn", s0.Op)
	}
	if s0.GetSite() != "model.layers.0" {
		t.Errorf("s0.Site = %q, want model.layers.0", s0.GetSite())
	}
	if s0.Node != "hc_sinkhorn" {
		t.Errorf("s0.Node = %q, want hc_sinkhorn", s0.Node)
	}
	if s0.GetShapes() != "tokens=1 hc=4" {
		t.Errorf("s0.Shapes = %q, want tokens=1 hc=4", s0.GetShapes())
	}
	if s0.GetLayer() != 0 {
		t.Errorf("s0.Layer = %d, want 0", s0.GetLayer())
	}
	if s0.Bytes == nil || s0.Bytes.Weights != 108 || s0.Bytes.Activations != 192 {
		t.Errorf("s0.Bytes = %+v, want weights:108, activations:192", s0.Bytes)
	}

	// The description records no choice:
	if s0.Chosen != "" {
		t.Errorf("s0.Chosen = %q, want empty", s0.Chosen)
	}
	if s0.Source != "" {
		t.Errorf("s0.Source = %q, want empty", s0.Source)
	}
	if s0.Why != nil {
		t.Errorf("s0.Why = %v, want nil", s0.Why)
	}
	if len(s0.Kernels) != 0 {
		t.Errorf("s0.Kernels = %v, want empty", s0.Kernels)
	}
	if s0.Fused != nil {
		t.Errorf("s0.Fused = %v, want nil", s0.Fused)
	}

	// Candidates at site 0:
	if len(s0.Candidates) != 2 {
		t.Fatalf("len(s0.Candidates) = %d, want 2", len(s0.Candidates))
	}
	c0 := s0.Candidates[0]
	if c0.Name != "gpu-triton" || !slices.Equal(c0.Devices, []string{"cuda"}) || c0.Numerics != "rounding" || !c0.Lowerable || !c0.Applies {
		t.Errorf("c0 = %+v, want gpu-triton (cuda, rounding, lowerable, applies)", c0)
	}
	c1 := s0.Candidates[1]
	if c1.Name != "reference" || !slices.Equal(c1.Devices, []string{"cpu", "cuda"}) || c1.Numerics != "bit-identical" || c1.Lowerable || !c1.Applies {
		t.Errorf("c1 = %+v, want reference (cpu+cuda, bit-identical, not lowerable, applies)", c1)
	}

	// Verify site 1: fp8 linear with candidate variants
	s1 := kr.Sites[1]
	if s1.Op != "vxpu::fp8_linear" || s1.GetSite() != "model.layers.0.attn.wq_a" || s1.GetShapes() != "1x128->64" {
		t.Errorf("s1 = %+v, unexpected op/site/shapes", s1)
	}
	var candNames []string
	for _, c := range s1.Candidates {
		candNames = append(candNames, c.Name)
	}
	wantCands := []string{"cpu-kernel", "gpu-fp8-gemv", "gpu-bf16-copy", "reference", "bf16_dequant"}
	if !slices.Equal(candNames, wantCands) {
		t.Errorf("s1 candidate names = %v, want %v", candNames, wantCands)
	}

	// Verify site 10: moe_experts host op call site is included
	s10 := kr.Sites[10]
	if s10.Op != "vxpu::moe_experts" || s10.Node != "moe_experts" || s10.GetSite() != "model.layers.0.ffn" {
		t.Errorf("s10 = %+v, want moe_experts at model.layers.0.ffn", s10)
	}
	var moeCands []string
	for _, c := range s10.Candidates {
		moeCands = append(moeCands, c.Name)
	}
	wantMoeCands := []string{"mailboxes", "gpu-hot+cpu", "cpu-fused", "cpu-three-call", "cpu-grouped", "reference"}
	if !slices.Equal(moeCands, wantMoeCands) {
		t.Errorf("s10 candidate names = %v, want %v", moeCands, wantMoeCands)
	}
}

func ptr[T any](v T) *T {
	return &v
}
