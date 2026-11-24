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
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
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

	binData, err := Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	var parsedBin planpb.PlanFile
	if err := proto.Unmarshal(binData, &parsedBin); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if len(parsedBin.Sites) != len(plan.Sites) {
		t.Errorf("len(parsedBin.Sites) = %d, want %d", len(parsedBin.Sites), len(plan.Sites))
	}

	data, err := MarshalAsJSON(plan)
	if err != nil {
		t.Fatalf("MarshalAsJSON failed: %v", err)
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
		if s.Kind == planpb.SiteKind_SITE_KIND_DENSE_GEMV {
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

		var expertSites []*planpb.PlanSite
		for _, s := range plan.Sites {
			if s.Kind == planpb.SiteKind_SITE_KIND_EXPERT_OP {
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
	tmpDir := t.TempDir()
	invalidJSON := []byte(`{
		"format": "vxpu-plan/unknown",
		"artifact_md5": "abc123",
		"tree_commit": "commit456",
		"sites": []
	}`)
	invFile := filepath.Join(tmpDir, "invalid.json")
	if err := os.WriteFile(invFile, invalidJSON, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := ReadPlan(invFile); err == nil {
		t.Error("expected error for unknown plan format, got nil")
	}

	unversionedJSON := []byte(`{
		"unknown_field": "foo",
		"artifact_md5": "abc123",
		"tree_commit": "commit456",
		"sites": []
	}`)
	unvFile := filepath.Join(tmpDir, "unversioned.json")
	if err := os.WriteFile(unvFile, unversionedJSON, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := ReadPlan(unvFile); err == nil {
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
	kt := &planpb.KernelTable{
		Gpu: &gpu,
		Knobs: &planpb.Knobs{
			GEMV_ROWS_MAX: &gemvRows,
			FP4_NATIVE:    &fp4Native,
		},
		Sites: []*planpb.SiteEntry{
			{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1.engram.wkv"), Impl: "reference"},
			{Op: "vxpu::fp8_linear", Shapes: ptr("1x8->16"), Impl: "reference"},
			{Op: "vxpu::rotary", Impl: "reference"},
		},
	}

	data, err := MarshalKernelTableAsJSON(kt)
	if err != nil {
		t.Fatalf("MarshalKernelTableAsJSON failed: %v", err)
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
	var rec planpb.PlanFile
	if err := ReadJSON(filepath.Join("testdata", "run_record.json"), &rec); err != nil {
		t.Fatalf("ReadJSON run_record.json failed: %v", err)
	}

	binData, err := Marshal(&rec)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var parsedRec planpb.PlanFile
	if err := proto.Unmarshal(binData, &parsedRec); err != nil {
		t.Fatalf("proto.Unmarshal failed: %v", err)
	}

	if parsedRec.GetArtifactMd5() != "artifact123" {
		t.Errorf("artifact_md5 = %q, want artifact123", parsedRec.GetArtifactMd5())
	}
	if parsedRec.GetTreeCommit() != "commit456" {
		t.Errorf("tree_commit = %q, want commit456", parsedRec.GetTreeCommit())
	}
	if len(parsedRec.Sites) != 2 {
		t.Fatalf("sites length = %d, want 2", len(parsedRec.Sites))
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
	marshaled, err := Marshal(&rec)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	var reparsed planpb.PlanFile
	if err := proto.Unmarshal(marshaled, &reparsed); err != nil {
		t.Fatalf("proto.Unmarshal failed: %v", err)
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

// TestReadKernelRecord verifies that Go can read a lowering KernelRecord document in JSON and binary format.
func TestReadKernelRecord(t *testing.T) {
	tmpDir := t.TempDir()
	rec := &planpb.KernelRecord{
		Device: "cpu",
		Sites: []*planpb.SiteRecord{
			{
				Op:                "vxpu::hc_sinkhorn",
				Node:              "hc_sinkhorn",
				Site:              ptr("model.layers.0"),
				Shapes:            ptr("tokens=1 hc=4"),
				Layer:             ptr(int32(0)),
				SinkhornSupported: ptr(true),
				Bytes: &planpb.SiteBytes{
					Weights:     108,
					Activations: 192,
				},
			},
		},
	}
	jsonData, err := MarshalAsJSON(rec)
	if err != nil {
		t.Fatalf("MarshalAsJSON: %v", err)
	}
	jsonPath := filepath.Join(tmpDir, "record.json")
	if err := os.WriteFile(jsonPath, jsonData, 0644); err != nil {
		t.Fatalf("WriteFile record.json: %v", err)
	}
	kr, err := ReadKernelRecord(jsonPath)
	if err != nil {
		t.Fatalf("ReadKernelRecord failed: %v", err)
	}

	binData, err := proto.Marshal(kr)
	if err != nil {
		t.Fatalf("proto.Marshal failed: %v", err)
	}
	tmpFile := filepath.Join(tmpDir, "record.pb")
	if err := os.WriteFile(tmpFile, binData, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	parsedKR, err := ReadKernelRecord(tmpFile)
	if err != nil {
		t.Fatalf("ReadKernelRecord binary failed: %v", err)
	}

	if len(parsedKR.Sites) != 1 {
		t.Fatalf("len(parsedKR.Sites) = %d, want 1", len(parsedKR.Sites))
	}

	s0 := parsedKR.Sites[0]
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
	if !s0.GetSinkhornSupported() {
		t.Errorf("s0.SinkhornSupported = %v, want true", s0.GetSinkhornSupported())
	}
}

func TestReadKernelCatalogFixture(t *testing.T) {
	fixturePath := filepath.Join("catalog", "kernel_catalog.json")
	cat, err := ReadKernelCatalog(fixturePath)
	if err != nil {
		t.Fatalf("ReadKernelCatalog failed: %v", err)
	}
	if len(cat.Bodies) == 0 {
		t.Fatalf("len(cat.Bodies) = 0, want > 0")
	}

	binData, err := proto.Marshal(cat)
	if err != nil {
		t.Fatalf("proto.Marshal failed: %v", err)
	}
	tmpFile := filepath.Join(t.TempDir(), "catalog.pb")
	if err := os.WriteFile(tmpFile, binData, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	binCat, err := ReadKernelCatalog(tmpFile)
	if err != nil {
		t.Fatalf("ReadKernelCatalog binary failed: %v", err)
	}
	if len(binCat.Bodies) != len(cat.Bodies) {
		t.Fatalf("binary bodies len = %d, want %d", len(binCat.Bodies), len(cat.Bodies))
	}

	findBody := func(op, name string) *planpb.KernelBody {
		for _, b := range cat.Bodies {
			if b.Op == op && b.Name == name {
				return b
			}
		}
		return nil
	}

	mailboxes := findBody("vxpu::moe_experts", "mailboxes")
	if mailboxes == nil || mailboxes.Kind != planpb.SiteKind_SITE_KIND_EXPERT_OP {
		t.Fatalf("mailboxes body = %+v, want EXPERT_OP", mailboxes)
	}
	if !mailboxes.Host {
		t.Errorf("mailboxes.Host = false, want true")
	}
	if mailboxes.Condition == nil || mailboxes.Condition.GetMaxRows() != 8 {
		t.Errorf("mailboxes condition = %+v, want max_rows: 8", mailboxes.Condition)
	}

	gemv := findBody("vxpu::fp8_linear", "gpu-fp8-gemv")
	if gemv == nil || gemv.Kind != planpb.SiteKind_SITE_KIND_DENSE_GEMV {
		t.Fatalf("gemv body = %+v, want DENSE_GEMV", gemv)
	}
	if gemv.Condition == nil || gemv.Condition.GetMaxRows() != 4 {
		t.Errorf("gemv condition = %+v, want max_rows: 4", gemv.Condition)
	}

	sinkhorn := findBody("vxpu::hc_sinkhorn", "gpu-triton")
	if sinkhorn == nil || sinkhorn.Kind != planpb.SiteKind_SITE_KIND_ATTENTION {
		t.Fatalf("sinkhorn body = %+v, want ATTENTION", sinkhorn)
	}
	if sinkhorn.Condition == nil || !sinkhorn.Condition.GetRequireSinkhornSupported() {
		t.Errorf("sinkhorn condition = %+v, want require_sinkhorn_supported: true", sinkhorn.Condition)
	}
}

func TestKernelBodySelectionPrecedence(t *testing.T) {
	catalog := &planpb.KernelCatalog{
		Bodies: []*planpb.KernelBody{
			{Op: "vxpu::fp8_linear", Name: "candidate-a", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "site-impl", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "shapes-impl", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "op-impl", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::rotary", Name: "rotary-impl", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_ATTENTION},
			{
				Op:      "vxpu::moe_experts",
				Name:    "cpu-fused",
				Devices: []string{"cpu"},
				Kind:    planpb.SiteKind_SITE_KIND_EXPERT_OP,
				Host:    true,
			},
			{
				Op:        "vxpu::moe_experts",
				Name:      "cpu-grouped",
				Devices:   []string{"cpu"},
				Kind:      planpb.SiteKind_SITE_KIND_EXPERT_OP,
				Condition: &planpb.KernelCondition{MinRows: ptr(int32(2))},
				Host:      true,
			},
		},
	}

	table := &planpb.KernelTable{
		Sites: []*planpb.SiteEntry{
			{
				Op:   "vxpu::fp8_linear",
				Site: ptr("model.layers.1.engram.wkv"),
				Impl: "site-impl",
			},
			{
				Op:     "vxpu::fp8_linear",
				Shapes: ptr("8->16"),
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

	t.Run("table precedence: site beats shapes beats op", func(t *testing.T) {
		sites := []*planpb.CallSite{
			{
				Op:     "vxpu::fp8_linear",
				Site:   "model.layers.1.engram.wkv",
				Shapes: "8->16",
				Node:   "fp8_linear_0",
			},
			{
				Op:     "vxpu::fp8_linear",
				Site:   "model.layers.2.other",
				Shapes: "8->16",
				Node:   "fp8_linear_1",
			},
			{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.3.attn.wo",
				Node: "fp8_linear_2",
			},
			{
				Op:   "vxpu::rotary",
				Site: "anywhere",
				Node: "rotary_0",
			},
		}
		res, err := PlanSites(sites, catalog, table, "cpu")
		if err != nil {
			t.Fatalf("PlanSites() unexpected error: %v", err)
		}
		if res[0].Body != "site-impl" {
			t.Errorf("site 0 body = %q, want site-impl", res[0].Body)
		}
		if res[1].Body != "shapes-impl" {
			t.Errorf("site 1 body = %q, want shapes-impl", res[1].Body)
		}
		if res[2].Body != "op-impl" {
			t.Errorf("site 2 body = %q, want op-impl", res[2].Body)
		}
		if res[3].Body != "rotary-impl" {
			t.Errorf("site 3 body = %q, want rotary-impl", res[3].Body)
		}
	})

	tests := []struct {
		name       string
		table      *planpb.KernelTable
		site       *planpb.CallSite
		device     string
		wantBody   string
		wantKind   planpb.SiteKind
		wantErrMsg string
	}{
		{
			name: "no table at all falls back to first applicable candidate in catalog",
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1.engram.wkv",
				Node: "fp8_linear_3",
			},
			device:   "cpu",
			wantBody: "candidate-a",
			wantKind: planpb.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name:  "empty table falls back to first applicable candidate in catalog",
			table: &planpb.KernelTable{},
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1.engram.wkv",
				Node: "fp8_linear_4",
			},
			device:   "cpu",
			wantBody: "candidate-a",
			wantKind: planpb.SiteKind_SITE_KIND_DENSE_GEMV,
		},
		{
			name: "host body applies on cuda device",
			table: &planpb.KernelTable{
				Sites: []*planpb.SiteEntry{
					{Op: "vxpu::moe_experts", Impl: "cpu-fused"},
				},
			},
			site: &planpb.CallSite{
				Op:   "vxpu::moe_experts",
				Node: "moe_experts_cuda",
			},
			device:   "cuda",
			wantBody: "cpu-fused",
			wantKind: planpb.SiteKind_SITE_KIND_EXPERT_OP,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sites, err := PlanSites([]*planpb.CallSite{tc.site}, catalog, tc.table, tc.device)
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
	catalog := &planpb.KernelCatalog{
		Bodies: []*planpb.KernelBody{
			{Op: "vxpu::fp8_linear", Name: "reference", Devices: []string{"cpu", "cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{Op: "vxpu::fp8_linear", Name: "gpu-fp8-gemv", Devices: []string{"cuda"}, Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV},
			{
				Op:        "vxpu::fp8_linear",
				Name:      "small-rows-only",
				Devices:   []string{"cpu"},
				Kind:      planpb.SiteKind_SITE_KIND_DENSE_GEMV,
				Condition: &planpb.KernelCondition{MaxRows: ptr(int32(4))},
			},
		},
	}

	tests := []struct {
		name       string
		table      *planpb.KernelTable
		site       *planpb.CallSite
		device     string
		wantErrSub []string
	}{
		{
			name: "table names body not in catalog",
			table: &planpb.KernelTable{
				Sites: []*planpb.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "nonexistent-body"},
				},
			},
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1.engram.wkv",
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "nonexistent-body", "does not apply"},
		},
		{
			name: "table names body not available for device",
			table: &planpb.KernelTable{
				Sites: []*planpb.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "gpu-fp8-gemv"},
				},
			},
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1.engram.wkv",
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "gpu-fp8-gemv", "does not apply"},
		},
		{
			name: "table names body whose condition does not hold",
			table: &planpb.KernelTable{
				Sites: []*planpb.SiteEntry{
					{Op: "vxpu::fp8_linear", Impl: "small-rows-only"},
				},
			},
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1.engram.wkv",
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"fp8_linear_0", "small-rows-only", "does not apply"},
		},
		{
			name: "conflicting table entries for same site",
			table: &planpb.KernelTable{
				Sites: []*planpb.SiteEntry{
					{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1"), Impl: "impl-a"},
					{Op: "vxpu::fp8_linear", Site: ptr("model.layers.1"), Impl: "impl-b"},
				},
			},
			site: &planpb.CallSite{
				Op:   "vxpu::fp8_linear",
				Site: "model.layers.1",
				Node: "fp8_linear_0",
			},
			device:     "cpu",
			wantErrSub: []string{"conflicting impl"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PlanSites([]*planpb.CallSite{tc.site}, catalog, tc.table, tc.device)
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

	// Stated rule: ops not in catalog execute their default implementation without planner substitution (body left unset).
	t.Run("uncataloged op emitted without body", func(t *testing.T) {
		site := &planpb.CallSite{
			Op:   "vxpu::hc_front",
			Node: "hc_front_0",
		}
		sites, err := PlanSites([]*planpb.CallSite{site}, catalog, nil, "cpu")
		if err != nil {
			t.Fatalf("PlanSites() unexpected error: %v", err)
		}
		if len(sites) != 1 {
			t.Fatalf("len(sites) = %d, want 1", len(sites))
		}
		if sites[0].Body != "" {
			t.Errorf("site body = %q, want empty", sites[0].Body)
		}
		if sites[0].Kind != planpb.SiteKind_SITE_KIND_UNSPECIFIED {
			t.Errorf("site kind = %v, want SITE_KIND_UNSPECIFIED", sites[0].Kind)
		}
	})

	t.Run("table_entry_matches_no_call_site", func(t *testing.T) {
		site := &planpb.CallSite{
			Op:     "vxpu::fp8_linear",
			Site:   "model.layers.1.engram.wkv",
			Node:   "fp8_linear_0",
			Shapes: "128->64",
		}
		table := &planpb.KernelTable{
			Sites: []*planpb.SiteEntry{
				{
					Op:     "vxpu::fp8_linear",
					Shapes: ptr("NOPE"),
					Impl:   "reference",
				},
			},
		}
		_, err := PlanSites([]*planpb.CallSite{site}, catalog, table, "cpu")
		if err == nil {
			t.Fatal("PlanSites() expected error for unmatched table entry, got nil")
		}
		if !strings.Contains(err.Error(), "NOPE") || !strings.Contains(err.Error(), "applies to no call site") {
			t.Errorf("PlanSites() error = %q, want substrings NOPE and applies to no call site", err.Error())
		}
	})
}

func findGitRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func getTinyDeepSeekArtifactDir(t *testing.T) string {
	t.Helper()
	pyBin := "python3"
	root := findGitRoot()
	if root != "" {
		venvPy := filepath.Join(root, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			pyBin = venvPy
		}
	}
	cmd := exec.Command(pyBin, "-c", "import sys; from vxpu.fixtures import tiny_deepseek_v41; sys.stdout.write(tiny_deepseek_v41.get_artifact_dir(max_cache_len=64))")
	if root != "" {
		cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(root, "vxpu/python"))
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Skipf("tiny DeepSeek artifact unavailable: %v (stderr: %s)", err, stderr.String())
	}
	dir := strings.TrimSpace(stdout.String())
	if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err != nil {
		t.Skipf("decode.pt2 not found in generated artifact dir %s: %v", dir, err)
	}
	return dir
}

func loadTinyDeepSeekCallSites(t *testing.T) []*planpb.CallSite {
	t.Helper()
	artDir := getTinyDeepSeekArtifactDir(t)
	sites, err := ReadCallSites(artDir)
	if err != nil {
		t.Fatalf("ReadCallSites failed: %v", err)
	}
	for _, cs := range sites {
		cs.Rows = 1
		cs.RowCounts = []int32{1}
	}
	return sites
}

func TestTinyDeepSeekPlanSitesGolden(t *testing.T) {
	fixtureDir := filepath.Join("testdata")
	callSites := loadTinyDeepSeekCallSites(t)
	for _, cs := range callSites {
		cs.Rows = 1
		cs.RowCounts = []int32{1}
	}

	var cat planpb.KernelCatalog
	if err := ReadJSON(filepath.Join("catalog", "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}

	var kt planpb.KernelTable
	if err := ReadJSON(filepath.Join(fixtureDir, "kernel_table.json"), &kt); err != nil {
		t.Fatalf("ReadJSON kernel_table.json: %v", err)
	}

	sites, err := PlanSites(callSites, &cat, &kt, "cpu")
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
		if s.Kind == planpb.SiteKind_SITE_KIND_UNSPECIFIED {
			t.Errorf("site[%d] %q has unspecified kind", i, s.Id)
		}
	}

	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  testTreeCommit,
		Sites:       sites,
	}

	binData, err := Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal plan: %v", err)
	}
	var parsedBin planpb.PlanFile
	if err := proto.Unmarshal(binData, &parsedBin); err != nil {
		t.Fatalf("proto.Unmarshal failed: %v", err)
	}
	if len(parsedBin.Sites) != len(plan.Sites) {
		t.Errorf("len(parsedBin.Sites) = %d, want %d", len(parsedBin.Sites), len(plan.Sites))
	}

	goldenPath := filepath.Join(fixtureDir, "golden_call_sites_plan.json")
	var goldenPlan planpb.PlanFile
	if err := ReadJSON(goldenPath, &goldenPlan); err != nil {
		t.Fatalf("ReadJSON golden failed: %v", err)
	}
	if len(sites) != len(goldenPlan.Sites) {
		t.Fatalf("sites count = %d, golden sites count = %d", len(sites), len(goldenPlan.Sites))
	}
	for i, s := range sites {
		if s.Id != goldenPlan.Sites[i].Id {
			t.Errorf("site[%d] id = %q, want %q", i, s.Id, goldenPlan.Sites[i].Id)
		}
		if s.Body != goldenPlan.Sites[i].Body {
			t.Errorf("site[%d] %q body = %q, want %q", i, s.Id, s.Body, goldenPlan.Sites[i].Body)
		}
		if s.Kind != goldenPlan.Sites[i].Kind {
			t.Errorf("site[%d] %q kind = %v, want %v", i, s.Id, s.Kind, goldenPlan.Sites[i].Kind)
		}
	}

	// Verify JoinPlan populates body on every call site and checks node matching.
	segments := []*planpb.Segment{
		{Name: "submod_0", CallSites: callSites},
	}
	if err := JoinPlan(segments, plan); err != nil {
		t.Fatalf("JoinPlan failed: %v", err)
	}
	for i, cs := range callSites {
		if cs.Body != sites[i].Body {
			t.Errorf("call site %q body = %q, want %q", cs.Node, cs.Body, sites[i].Body)
		}
	}

	// Unmatched on either side must be refused by name
	missingInPlan := &planpb.PlanFile{
		ArtifactMd5: plan.ArtifactMd5,
		TreeCommit:  plan.TreeCommit,
		Sites:       sites[1:],
	}
	if err := JoinPlan(segments, missingInPlan); err == nil || !strings.Contains(err.Error(), callSites[0].Node) {
		t.Errorf("JoinPlan with missing site in plan error = %v, want substring %q", err, callSites[0].Node)
	}

	missingInGraph := &planpb.PlanFile{
		ArtifactMd5: plan.ArtifactMd5,
		TreeCommit:  plan.TreeCommit,
		Sites:       append(slices.Clone(sites), &planpb.PlanSite{Id: "extra_site", Body: "ref"}),
	}
	if err := JoinPlan(segments, missingInGraph); err == nil || !strings.Contains(err.Error(), "extra_site") {
		t.Errorf("JoinPlan with extra site in plan error = %v, want substring %q", err, "extra_site")
	}
}

func TestTinyDeepSeekPlanSitesNoTable(t *testing.T) {
	callSites := loadTinyDeepSeekCallSites(t)
	for _, cs := range callSites {
		cs.Rows = 1
		cs.RowCounts = []int32{1}
	}

	var cat planpb.KernelCatalog
	if err := ReadJSON(filepath.Join("catalog", "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}

	sites, err := PlanSites(callSites, &cat, nil, "cpu")
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

func TestTinyDeepSeekPlanSitesGPU(t *testing.T) {
	callSites := loadTinyDeepSeekCallSites(t)
	for _, cs := range callSites {
		cs.Rows = 1
		cs.RowCounts = []int32{1}
	}

	var cat planpb.KernelCatalog
	if err := ReadJSON(filepath.Join("catalog", "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}

	// 1. Single row decode on CUDA: conditions on rows (<=4, <=8), dtype (bfloat16),
	// and sinkhorn_supported match the GPU variants. Compare against checked-in golden GPU plan.
	sites, err := PlanSites(callSites, &cat, nil, "cuda")
	if err != nil {
		t.Fatalf("PlanSites on cuda: %v", err)
	}
	if len(sites) != 67 {
		t.Fatalf("len(sites) = %d, want 67", len(sites))
	}

	gpuGoldenPath := filepath.Join("testdata", "golden_call_sites_plan_gpu.json")
	var gpuGolden planpb.PlanFile
	if err := ReadJSON(gpuGoldenPath, &gpuGolden); err != nil {
		t.Fatalf("ReadJSON golden_call_sites_plan_gpu.json: %v", err)
	}
	if len(sites) != len(gpuGolden.Sites) {
		t.Fatalf("len(sites) = %d, want %d", len(sites), len(gpuGolden.Sites))
	}
	for i, s := range sites {
		if s.Id != gpuGolden.Sites[i].Id {
			t.Errorf("site[%d] id = %q, want %q", i, s.Id, gpuGolden.Sites[i].Id)
		}
		if s.Body != gpuGolden.Sites[i].Body {
			t.Errorf("site[%d] %q body = %q, want golden body %q", i, s.Id, s.Body, gpuGolden.Sites[i].Body)
		}
		if s.Kind != gpuGolden.Sites[i].Kind {
			t.Errorf("site[%d] %q kind = %v, want golden kind %v", i, s.Id, s.Kind, gpuGolden.Sites[i].Kind)
		}
	}

	// 2. Wide row count (rows=16) on CUDA: rows > max_rows(4) falls back from gpu-fp8-gemv
	wideSites := make([]*planpb.CallSite, len(callSites))
	for i, cs := range callSites {
		clone := proto.Clone(cs).(*planpb.CallSite)
		clone.Rows = 16
		clone.RowCounts = []int32{16}
		wideSites[i] = clone
	}
	sitesWide, err := PlanSites(wideSites, &cat, nil, "cuda")
	if err != nil {
		t.Fatalf("PlanSites on cuda (rows=16): %v", err)
	}
	for _, s := range sitesWide {
		if strings.HasPrefix(s.Id, "fp8_linear") {
			// At rows=16, gpu-fp8-gemv (max_rows: 4) does not apply. It falls back to gpu-bf16-copy.
			if s.Body != "gpu-bf16-copy" {
				t.Errorf("wide site %q body = %q, want gpu-bf16-copy fallback", s.Id, s.Body)
			}
		}
		if s.Id == "head_linear" {
			// At rows=16, gpu-triton-bf16 (max_rows: 8) does not apply. It falls back to f32-copy.
			if s.Body != "f32-copy" {
				t.Errorf("wide site %q body = %q, want f32-copy fallback", s.Id, s.Body)
			}
		}
	}
}

func TestReadKernelCatalog123ByteBody(t *testing.T) {
	// A catalog whose first body encodes to 123 bytes has field 1 tag 0x0a (newline)
	// followed by length 123 (0x7b, '{'). Verify that ReadKernelCatalog reads this as
	// binary protobuf according to the file extension, rather than misinterpreting as JSON.
	var cat *planpb.KernelCatalog
	var binData []byte
	for pad := 1; pad < 200; pad++ {
		c := &planpb.KernelCatalog{
			Bodies: []*planpb.KernelBody{
				{
					Name:    strings.Repeat("a", pad),
					Op:      "vxpu::fp8_linear",
					Devices: []string{"cpu"},
				},
			},
		}
		d, err := proto.Marshal(c)
		if err != nil {
			t.Fatalf("proto.Marshal: %v", err)
		}
		if len(d) > 1 && d[0] == 0x0a && d[1] == byte(123) {
			cat = c
			binData = d
			break
		}
	}
	if binData == nil {
		t.Fatalf("could not construct 123-byte body")
	}
	if binData[1] != '{' {
		t.Fatalf("expected second byte to be '{' (0x7b), got %#x", binData[1])
	}
	tmpFile := filepath.Join(t.TempDir(), "catalog.pb")
	if err := os.WriteFile(tmpFile, binData, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	readCat, err := ReadKernelCatalog(tmpFile)
	if err != nil {
		t.Fatalf("ReadKernelCatalog failed on 123-byte body: %v", err)
	}
	if len(readCat.Bodies) != 1 || readCat.Bodies[0].Name != cat.Bodies[0].Name {
		t.Errorf("read catalog body mismatch, got %v, want %v", readCat, cat)
	}
}

func TestReadDocumentUnknownFieldsRefused(t *testing.T) {
	tmpDir := t.TempDir()

	// Write a binary KernelRecord
	rec := &planpb.KernelRecord{
		Device: "cpu",
		Sites: []*planpb.SiteRecord{
			{Op: "vxpu::fp8_linear", Node: "fp8_0"},
		},
	}
	recData, err := proto.Marshal(rec)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	recPath := filepath.Join(tmpDir, "record.pb")
	if err := os.WriteFile(recPath, recData, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// Handing a KernelRecord to ReadKernelCatalog must be refused because of unknown fields
	if _, err := ReadKernelCatalog(recPath); err == nil {
		t.Error("expected error reading KernelRecord as KernelCatalog, got nil")
	}

	// Handing a KernelCatalog to ReadKernelRecord must be refused because of unknown fields
	cat := &planpb.KernelCatalog{
		Bodies: []*planpb.KernelBody{
			{Name: "ref_linear", Op: "vxpu::fp8_linear"},
		},
	}
	catData, err := proto.Marshal(cat)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	catPath := filepath.Join(tmpDir, "catalog.pb")
	if err := os.WriteFile(catPath, catData, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, err := ReadKernelRecord(catPath); err == nil {
		t.Error("expected error reading KernelCatalog as KernelRecord, got nil")
	}
}

func ptr[T any](v T) *T {
	return &v
}

func TestPlanHash_ComputesDeterministicIdentity(t *testing.T) {
	plan := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  "commit123",
		Sites: []*planpb.PlanSite{
			{
				Id:        "model.layers.0.attn.wq.weight",
				Kind:      planpb.SiteKind_SITE_KIND_DENSE_GEMV,
				Shape:     []int32{128, 256},
				RowCounts: []int32{1, 8},
				Strategy:  "gpu-fp8-gemv",
				Body:      "gemv_body",
			},
		},
		Knobs: &planpb.Knobs{
			GEMV_ROWS_MAX: proto.Int32(32),
		},
	}

	h1, err := PlanHash(plan)
	if err != nil {
		t.Fatalf("PlanHash: %v", err)
	}
	if len(h1) != 64 {
		t.Errorf("PlanHash len = %d, want 64", len(h1))
	}

	h2, err := PlanHash(plan)
	if err != nil {
		t.Fatalf("PlanHash repeat: %v", err)
	}
	if h1 != h2 {
		t.Errorf("PlanHash is not deterministic: %s != %s", h1, h2)
	}

	// Execution measurements (Stats on PlanFile and on PlanSite) must not affect PlanHash.
	withStats := proto.Clone(plan).(*planpb.PlanFile)
	withStats.Stats = &planpb.PlanExecutionStats{
		MedianStepMs:       proto.Float64(12.5),
		TokensAStreamAStep: proto.Float64(2.0),
	}
	withStats.Sites[0].Stats = &planpb.SiteExecutionStats{
		MeasuredUs: proto.Float64(42.5),
	}
	hWithStats, err := PlanHash(withStats)
	if err != nil {
		t.Fatalf("PlanHash with stats: %v", err)
	}
	if hWithStats != h1 {
		t.Errorf("PlanHash with stats = %s, want %s (measurements must not alter plan identity)", hWithStats, h1)
	}

	// An existing PlanHash field must not affect PlanHash computation.
	withHash := proto.Clone(plan).(*planpb.PlanFile)
	withHash.PlanHash = "existing_different_hash"
	hWithHash, err := PlanHash(withHash)
	if err != nil {
		t.Fatalf("PlanHash with existing hash: %v", err)
	}
	if hWithHash != h1 {
		t.Errorf("PlanHash with existing hash = %s, want %s", hWithHash, h1)
	}

	// Changing knobs must produce a different hash.
	diffKnobs := proto.Clone(plan).(*planpb.PlanFile)
	diffKnobs.Knobs.GEMV_ROWS_MAX = proto.Int32(64)
	hDiffKnobs, err := PlanHash(diffKnobs)
	if err != nil {
		t.Fatalf("PlanHash with different knobs: %v", err)
	}
	if hDiffKnobs == h1 {
		t.Errorf("different knobs produced the same hash: %s", hDiffKnobs)
	}

	// Changing sites must produce a different hash.
	diffSites := proto.Clone(plan).(*planpb.PlanFile)
	diffSites.Sites[0].Strategy = "reference"
	hDiffSites, err := PlanHash(diffSites)
	if err != nil {
		t.Fatalf("PlanHash with different sites: %v", err)
	}
	if hDiffSites == h1 {
		t.Errorf("different sites produced the same hash: %s", hDiffSites)
	}
}

func TestEnsurePlanIdentity(t *testing.T) {
	plan := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  "commit123",
	}
	if err := EnsurePlanIdentity(plan); err != nil {
		t.Fatalf("EnsurePlanIdentity: %v", err)
	}
	if plan.PlanHash == "" {
		t.Fatal("EnsurePlanIdentity did not set PlanHash")
	}

	expected, err := PlanHash(plan)
	if err != nil {
		t.Fatalf("PlanHash: %v", err)
	}
	if plan.PlanHash != expected {
		t.Errorf("PlanHash = %s, want %s", plan.PlanHash, expected)
	}

	// EnsurePlanIdentity accepts a matching PlanHash.
	customPlan := &planpb.PlanFile{
		ArtifactMd5: "abc",
	}
	correctHash, err := PlanHash(customPlan)
	if err != nil {
		t.Fatalf("PlanHash: %v", err)
	}
	customPlan.PlanHash = correctHash
	if err := EnsurePlanIdentity(customPlan); err != nil {
		t.Fatalf("EnsurePlanIdentity with correct hash: %v", err)
	}
	if customPlan.PlanHash != correctHash {
		t.Errorf("PlanHash changed: got %s, want %s", customPlan.PlanHash, correctHash)
	}

	// EnsurePlanIdentity refuses a bogus or mismatched PlanHash.
	bogusPlan := &planpb.PlanFile{
		ArtifactMd5: "abc",
		PlanHash:    "bogus",
	}
	if err := EnsurePlanIdentity(bogusPlan); err == nil {
		t.Fatal("EnsurePlanIdentity accepted bogus plan_hash, want error")
	}
}
