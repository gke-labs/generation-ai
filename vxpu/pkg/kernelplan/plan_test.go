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
	"strings"
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

	// Verify site 0: facts recorded:
	if !s0.GetSinkhornSupported() {
		t.Errorf("s0.SinkhornSupported = %v, want true", s0.GetSinkhornSupported())
	}

	// Verify site 1: fp8 linear with row count
	s1 := kr.Sites[1]
	if s1.Op != "vxpu::fp8_linear" || s1.GetSite() != "model.layers.0.attn.wq_a" || s1.GetShapes() != "1x128->64" {
		t.Errorf("s1 = %+v, unexpected op/site/shapes", s1)
	}
	if s1.GetRows() != 1 {
		t.Errorf("s1.Rows = %d, want 1", s1.GetRows())
	}

	// Verify site 10: moe_experts host op call site is included
	s10 := kr.Sites[10]
	if s10.Op != "vxpu::moe_experts" || s10.Node != "moe_experts" || s10.GetSite() != "model.layers.0.ffn" {
		t.Errorf("s10 = %+v, want moe_experts at model.layers.0.ffn", s10)
	}
	if s10.GetRows() != 1 {
		t.Errorf("s10.Rows = %d, want 1", s10.GetRows())
	}

	// Verify site 14: engram_rows host op call site is included
	s14 := kr.Sites[14]
	if s14.Op != "vxpu::engram_rows" || s14.Node != "engram_rows" {
		t.Errorf("s14 = %+v, want engram_rows", s14)
	}
}

func TestReadKernelCatalogFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "kernel_catalog.json")
	var cat planv1alpha1.KernelCatalog
	if err := ReadJSON(fixturePath, &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}
	if len(cat.Bodies) == 0 {
		t.Fatalf("len(cat.Bodies) = 0, want > 0")
	}

	findBody := func(op, name string) *planv1alpha1.KernelBody {
		for _, b := range cat.Bodies {
			if b.Op == op && b.Name == name {
				return b
			}
		}
		return nil
	}

	mailboxes := findBody("vxpu::moe_experts", "mailboxes")
	if mailboxes == nil || mailboxes.Kind != planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP {
		t.Fatalf("mailboxes body = %+v, want EXPERT_OP", mailboxes)
	}
	if !mailboxes.Host {
		t.Errorf("mailboxes.Host = false, want true")
	}
	if mailboxes.Condition == nil || mailboxes.Condition.GetMaxRows() != 8 {
		t.Errorf("mailboxes condition = %+v, want max_rows: 8", mailboxes.Condition)
	}

	gemv := findBody("vxpu::fp8_linear", "gpu-fp8-gemv")
	if gemv == nil || gemv.Kind != planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV {
		t.Fatalf("gemv body = %+v, want DENSE_GEMV", gemv)
	}
	if gemv.Condition == nil || gemv.Condition.GetMaxRows() != 4 {
		t.Errorf("gemv condition = %+v, want max_rows: 4", gemv.Condition)
	}

	sinkhorn := findBody("vxpu::hc_sinkhorn", "gpu-triton")
	if sinkhorn == nil || sinkhorn.Kind != planv1alpha1.SiteKind_SITE_KIND_ATTENTION {
		t.Fatalf("sinkhorn body = %+v, want ATTENTION", sinkhorn)
	}
	if sinkhorn.Condition == nil || !sinkhorn.Condition.GetRequireSinkhornSupported() {
		t.Errorf("sinkhorn condition = %+v, want require_sinkhorn_supported: true", sinkhorn.Condition)
	}
}

func TestKernelBodySelectionPrecedence(t *testing.T) {
	catalog := &planv1alpha1.KernelCatalog{
		Bodies: []*planv1alpha1.KernelBody{
			{Op: "vxpu::fp8_linear", Name: "candidate-a", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "site-impl", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "shapes-impl", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "op-impl", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::rotary", Name: "rotary-impl", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_ATTENTION},
			{
				Op:        "vxpu::moe_experts",
				Name:      "cpu-fused",
				Devices:   []string{"cpu"},
				Kind:      planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP,
				Condition: &planv1alpha1.KernelCondition{ExactRows: ptr(int32(1))},
				Host:      true,
			},
			{
				Op:        "vxpu::moe_experts",
				Name:      "cpu-grouped",
				Devices:   []string{"cpu"},
				Kind:      planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP,
				Condition: &planv1alpha1.KernelCondition{MinRows: ptr(int32(2))},
				Host:      true,
			},
		},
	}

	table := &planv1alpha1.KernelTable{
		Sites: []*planv1alpha1.SiteEntry{
			{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Impl: "site-impl",
			},
			{
				Op:     "vxpu::fp8_linear",
				Shapes: ptr("1x8->16"),
				Impl:   "shapes-impl",
			},
			{
				Op:   "vxpu::fp8_linear",
				Impl: "op-impl",
			},
			{
				Op:   "vxpu::rotary",
				Impl: "rotary-impl",
			},
		},
	}

	tests := []struct {
		name       string
		table      *planv1alpha1.KernelTable
		site       *planv1alpha1.SiteRecord
		device     string
		wantBody   string
		wantKind   planv1alpha1.SiteKind
		wantErrMsg string
	}{
		{
			name:  "exact site match beats shapes and op",
			table: table,
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::fp8_linear",
				Site:   ptr("model.layers.1.engram.wkv"),
				Shapes: ptr("1x8->16"),
				Node:   "fp8_linear_0",
			},
			device:   "cpu",
			wantBody: "site-impl",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name:  "shapes match applies when site does not match",
			table: table,
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::fp8_linear",
				Site:   ptr("model.layers.3.attn.wo"),
				Shapes: ptr("1x8->16"),
				Node:   "fp8_linear_1",
			},
			device:   "cpu",
			wantBody: "shapes-impl",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name:  "op match applies everywhere",
			table: table,
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::rotary",
				Site:   ptr("anywhere"),
				Shapes: ptr("any"),
				Node:   "rotary_0",
			},
			device:   "cpu",
			wantBody: "rotary-impl",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_ATTENTION,
		},
		{
			name:  "op level fallback for fp8_linear when site and shapes do not match",
			table: table,
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::fp8_linear",
				Site:   ptr("model.layers.3.attn.wo"),
				Shapes: ptr("1x9->16"),
				Node:   "fp8_linear_2",
			},
			device:   "cpu",
			wantBody: "op-impl",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name: "no table at all falls back to first applicable candidate in catalog",
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::fp8_linear",
				Site:   ptr("model.layers.1.engram.wkv"),
				Shapes: ptr("1x8->16"),
				Node:   "fp8_linear_3",
			},
			device:   "cpu",
			wantBody: "candidate-a",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name:  "empty table falls back to first applicable candidate in catalog",
			table: &planv1alpha1.KernelTable{},
			site: &planv1alpha1.SiteRecord{
				Op:     "vxpu::fp8_linear",
				Site:   ptr("model.layers.1.engram.wkv"),
				Shapes: ptr("1x8->16"),
				Node:   "fp8_linear_4",
			},
			device:   "cpu",
			wantBody: "candidate-a",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name:  "condition filtering in fallback selects matching body",
			table: &planv1alpha1.KernelTable{},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::moe_experts",
				Node: "moe_experts_0",
				Rows: ptr(int32(4)),
			},
			device:   "cpu",
			wantBody: "cpu-grouped",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP,
		},
		{
			name:  "condition filtering for single row selects exact_rows 1 body",
			table: &planv1alpha1.KernelTable{},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::moe_experts",
				Node: "moe_experts_1",
				Rows: ptr(int32(1)),
			},
			device:   "cpu",
			wantBody: "cpu-fused",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP,
		},
		{
			name:  "host body applies on cuda device",
			table: &planv1alpha1.KernelTable{},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::moe_experts",
				Node: "moe_experts_cuda",
				Rows: ptr(int32(1)),
			},
			device:   "cuda",
			wantBody: "cpu-fused",
			wantKind: planv1alpha1.SiteKind_SITE_KIND_EXPERT_OP,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			desc := &planv1alpha1.KernelRecord{Sites: []*planv1alpha1.SiteRecord{tc.site}}
			sites, err := PlanSites(desc, catalog, tc.table, tc.device)
			if tc.wantErrMsg != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Fatalf("PlanSites() error = %v, want substring %q", err, tc.wantErrMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("PlanSites() unexpected error: %v", err)
			}
			if len(sites) != 1 {
				t.Fatalf("PlanSites() len = %d, want 1", len(sites))
			}
			if sites[0].Body != tc.wantBody {
				t.Errorf("PlanSites() body = %q, want %q", sites[0].Body, tc.wantBody)
			}
			if sites[0].Kind != tc.wantKind {
				t.Errorf("PlanSites() kind = %v, want %v", sites[0].Kind, tc.wantKind)
			}
		})
	}
}

func TestKernelBodySelectionRefusals(t *testing.T) {
	catalog := &planv1alpha1.KernelCatalog{
		Bodies: []*planv1alpha1.KernelBody{
			{Op: "vxpu::fp8_linear", Name: "reference", Devices: []string{"cpu", "cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "gpu-fp8-gemv", Devices: []string{"cuda"}, Kind: planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV},
			{
				Op:        "vxpu::fp8_linear",
				Name:      "small-rows-only",
				Devices:   []string{"cpu"},
				Kind:      planv1alpha1.SiteKind_SITE_KIND_DENSE_GEMV,
				Condition: &planv1alpha1.KernelCondition{MaxRows: ptr(int32(4))},
			},
		},
	}

	tests := []struct {
		name       string
		table      *planv1alpha1.KernelTable
		site       *planv1alpha1.SiteRecord
		device     string
		wantErrSub []string
	}{
		{
			name: "table names body not in catalog",
			table: &planv1alpha1.KernelTable{
				Sites: []*planv1alpha1.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "nonexistent-body"},
				},
			},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "nonexistent-body", "does not apply"},
		},
		{
			name: "table names body not available for device",
			table: &planv1alpha1.KernelTable{
				Sites: []*planv1alpha1.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "gpu-fp8-gemv"},
				},
			},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "gpu-fp8-gemv", "does not apply"},
		},
		{
			name: "table names body whose condition does not hold",
			table: &planv1alpha1.KernelTable{
				Sites: []*planv1alpha1.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "small-rows-only"},
				},
			},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Node: "fp8_linear_0",
				Rows: ptr(int32(16)),
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "small-rows-only", "does not apply"},
		},
		{
			name: "condition names fact site does not report",
			table: &planv1alpha1.KernelTable{
				Sites: []*planv1alpha1.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "small-rows-only"},
				},
			},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Node: "fp8_linear_0",
				// Rows is nil!
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "small-rows-only", "does not apply"},
		},
		{
			name: "unknown op at call site refused",
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::unknown_op",
				Node: "unknown_0",
			},
			device:     "cpu",
			wantErrSub: []string{"unknown_0", "unknown op", "vxpu::unknown_op"},
		},
		{
			name: "conflicting table entries for same site",
			table: &planv1alpha1.KernelTable{
				Sites: []*planv1alpha1.SiteEntry{
					{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1"), Impl: "impl-a"},
					{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1"), Impl: "impl-b"},
				},
			},
			site: &planv1alpha1.SiteRecord{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1"),
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"conflicting impl"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			desc := &planv1alpha1.KernelRecord{Sites: []*planv1alpha1.SiteRecord{tc.site}}
			_, err := PlanSites(desc, catalog, tc.table, tc.device)
			if err == nil {
				t.Fatalf("PlanSites() expected error, got nil")
			}
			for _, sub := range tc.wantErrSub {
				if !strings.Contains(err.Error(), sub) {
					t.Errorf("PlanSites() error = %q, want substring %q", err.Error(), sub)
				}
			}
		})
	}
}

func TestTinyDeepSeekPlanSitesGolden(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	var kr planv1alpha1.KernelRecord
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_record.json"), &kr); err != nil {
		t.Fatalf("ReadJSON kernel_record.json: %v", err)
	}

	var cat planv1alpha1.KernelCatalog
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}

	var kt planv1alpha1.KernelTable
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_table.json"), &kt); err != nil {
		t.Fatalf("ReadJSON kernel_table.json: %v", err)
	}

	sites, err := PlanSites(&kr, &cat, &kt, "cpu")
	if err != nil {
		t.Fatalf("PlanSites: %v", err)
	}

	if len(sites) != 67 {
		t.Fatalf("len(sites) = %d, want 67", len(sites))
	}

	for i, s := range sites {
		if s.Body == "" {
			t.Errorf("site[%d] %q has empty body", i, s.Id)
		}
		if s.Strategy != "" {
			t.Errorf("site[%d] %q has strategy = %q, want unset", i, s.Id, s.Strategy)
		}
		if s.Kind == planv1alpha1.SiteKind_SITE_KIND_UNSPECIFIED {
			t.Errorf("site[%d] %q has unspecified kind", i, s.Id)
		}
	}

	plan := &planv1alpha1.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  testTreeCommit,
		Sites:       sites,
	}

	data, err := Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal plan: %v", err)
	}

	goldenPath := filepath.Join(fixtureDir, "golden_call_sites_plan.json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		if os.IsNotExist(err) && os.Getenv("UPDATE_GOLDEN") != "" {
			if err := os.WriteFile(goldenPath, data, 0644); err != nil {
				t.Fatalf("WriteFile golden failed: %v", err)
			}
			goldenBytes = data
		} else {
			t.Fatalf("ReadFile golden failed: %v", err)
		}
	}

	if !bytes.Equal(data, goldenBytes) {
		if os.Getenv("UPDATE_GOLDEN") != "" {
			if err := os.WriteFile(goldenPath, data, 0644); err != nil {
				t.Fatalf("WriteFile golden failed: %v", err)
			}
			return
		}
		t.Errorf("emitted plan does not match golden file %s (run with UPDATE_GOLDEN=1 to regenerate)\ngot:\n%s\nwant:\n%s", goldenPath, string(data), string(goldenBytes))
	}
}

func TestTinyDeepSeekPlanSitesNoTable(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	var kr planv1alpha1.KernelRecord
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_record.json"), &kr); err != nil {
		t.Fatalf("ReadJSON kernel_record.json: %v", err)
	}
	var cat planv1alpha1.KernelCatalog
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}

	sites, err := PlanSites(&kr, &cat, nil, "cpu")
	if err != nil {
		t.Fatalf("PlanSites with nil table: %v", err)
	}
	if len(sites) != 67 {
		t.Fatalf("len(sites) = %d, want 67", len(sites))
	}
	for _, s := range sites {
		if s.Body == "" {
			t.Errorf("site %q has empty body", s.Id)
		}
		wantBody := "reference"
		if s.Id == "head_linear" {
			wantBody = "f32-copy"
		} else if strings.HasPrefix(s.Id, "engram_rows") {
			wantBody = "host-lookup"
		}
		if s.Body != wantBody {
			t.Errorf("site %q body = %q, want %q", s.Id, s.Body, wantBody)
		}
	}
}

func ptr[T any](v T) *T {
	return &v
}
