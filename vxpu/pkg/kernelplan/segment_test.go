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
	"archive/zip"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/proto"
)

func createTestArchive(t *testing.T, format, version string, major, minor int, nodesJSON string) string {
	t.Helper()
	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	if format != "" {
		w, err := zw.Create("decode/archive_format")
		if err != nil {
			t.Fatalf("create archive_format: %v", err)
		}
		if _, err := w.Write([]byte(format)); err != nil {
			t.Fatalf("write archive_format: %v", err)
		}
	}

	if version != "" {
		w, err := zw.Create("decode/archive_version")
		if err != nil {
			t.Fatalf("create archive_version: %v", err)
		}
		if _, err := w.Write([]byte(version)); err != nil {
			t.Fatalf("write archive_version: %v", err)
		}
	}

	w, err := zw.Create("decode/models/model.json")
	if err != nil {
		t.Fatalf("create model.json: %v", err)
	}
	modelJSON := fmt.Sprintf(`{
		"schema_version": {"major": %d, "minor": %d},
		"graph_module": {
			"graph": {
				"nodes": %s
			}
		}
	}`, major, minor, nodesJSON)
	if _, err := w.Write([]byte(modelJSON)); err != nil {
		t.Fatalf("write model.json: %v", err)
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}

	tmpDir := t.TempDir()
	pt2Path := filepath.Join(tmpDir, "decode.pt2")
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write test archive: %v", err)
	}
	return tmpDir
}

func TestReadSegments_Valid(t *testing.T) {
	nodes := `[
		{
			"name": "aten_select",
			"target": "torch.ops.aten.select.int"
		},
		{
			"name": "fp8_linear_0",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"metadata": {
				"nn_module_stack": "L__self__,,Decode;L__self__model,model,Transformer;L__self__model.layers.0.attn.wq_a,model.layers.0.attn.wq_a,Linear"
			}
		},
		{
			"name": "moe_experts_0",
			"target": "torch.ops.vxpu.moe_experts.default",
			"metadata": {
				"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.ffn,model.layers.0.ffn,MoE"
			}
		},
		{
			"name": "fp8_linear_1",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"metadata": {
				"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.ffn.shared_experts.w1,model.layers.0.ffn.shared_experts.w1,Linear"
			}
		}
	]`

	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	if len(segments) != 3 {
		t.Fatalf("got %d segments, want 3", len(segments))
	}

	// submod_0: dense
	if segments[0].Name != "submod_0" || segments[0].Host {
		t.Errorf("segments[0] = %+v, want submod_0 (dense)", segments[0])
	}
	if len(segments[0].CallSites) != 1 {
		t.Fatalf("segments[0] call sites = %d, want 1", len(segments[0].CallSites))
	}
	site0 := segments[0].CallSites[0]
	if site0.Op != "vxpu::fp8_linear" || site0.Node != "fp8_linear_0" || site0.Site != "model.layers.0.attn.wq_a" {
		t.Errorf("site0 = %+v, unexpected fields", site0)
	}

	// submod_1: host
	if segments[1].Name != "submod_1" || !segments[1].Host {
		t.Errorf("segments[1] = %+v, want submod_1 (host)", segments[1])
	}
	if len(segments[1].CallSites) != 1 {
		t.Fatalf("segments[1] call sites = %d, want 1", len(segments[1].CallSites))
	}
	site1 := segments[1].CallSites[0]
	if site1.Op != "vxpu::moe_experts" || site1.Node != "moe_experts_0" || site1.Site != "model.layers.0.ffn" {
		t.Errorf("site1 = %+v, unexpected fields", site1)
	}

	// submod_2: dense
	if segments[2].Name != "submod_2" || segments[2].Host {
		t.Errorf("segments[2] = %+v, want submod_2 (dense)", segments[2])
	}
	if len(segments[2].CallSites) != 1 {
		t.Fatalf("segments[2] call sites = %d, want 1", len(segments[2].CallSites))
	}
	site2 := segments[2].CallSites[0]
	if site2.Op != "vxpu::fp8_linear" || site2.Node != "fp8_linear_1" || site2.Site != "model.layers.0.ffn.shared_experts.w1" {
		t.Errorf("site2 = %+v, unexpected fields", site2)
	}
}

func TestReadSegments_RefuseUnsupportedFormat(t *testing.T) {
	dir := createTestArchive(t, "pt1", "0", 8, 20, "[]")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for unsupported archive format pt1, got nil")
	}
	if !strings.Contains(err.Error(), "pt1") {
		t.Errorf("error %q should name the refused format pt1", err.Error())
	}
}

func TestReadSegments_RefuseUnsupportedArchiveVersion(t *testing.T) {
	dir := createTestArchive(t, "pt2", "1", 8, 20, "[]")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for unsupported archive version 1, got nil")
	}
	if !strings.Contains(err.Error(), "1") {
		t.Errorf("error %q should name the refused archive version 1", err.Error())
	}
}

func TestReadSegments_RefuseUnsupportedSchemaVersion(t *testing.T) {
	dir := createTestArchive(t, "pt2", "0", 9, 0, "[]")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for unsupported schema version 9.0, got nil")
	}
	if !strings.Contains(err.Error(), "9.0") {
		t.Errorf("error %q should name the refused schema version 9.0", err.Error())
	}
}

func TestReadSegments_RefuseMalformedOperatorTarget(t *testing.T) {
	nodes := `[
		{
			"name": "broken_op",
			"target": "torch.ops.vxpu..default"
		}
	]`
	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for malformed operator target, got nil")
	}
	if !strings.Contains(err.Error(), "empty operator name") {
		t.Errorf("error %q should mention empty operator name", err.Error())
	}
}

func TestReadSegments_NestedGraph(t *testing.T) {
	nodes := `[
		{
			"name": "hop_node",
			"target": "torch.ops.higher_order.wrap_with_set_grad_enabled",
			"inputs": [
				{
					"name": "subgraph",
					"arg": {
						"as_graph": {
							"graph": {
								"nodes": [
									{"name": "sub_op", "target": "torch.ops.vxpu.fp8_linear.default"}
								]
							}
						}
					}
				}
			]
		}
	]`
	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)
	segs, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("unexpected error for nested graph: %v", err)
	}
	if len(segs) != 1 {
		t.Fatalf("len(segs) = %d, want 1", len(segs))
	}
	if len(segs[0].CallSites) != 1 || segs[0].CallSites[0].Node != "sub_op" {
		t.Errorf("call sites = %v, want 1 sub_op site", segs[0].CallSites)
	}
}

func TestReadSegments_NestedGraphWithHostOp(t *testing.T) {
	nodes := `[
		{
			"name": "hop_node",
			"target": "torch.ops.higher_order.wrap_with_set_grad_enabled",
			"inputs": [
				{
					"name": "subgraph",
					"arg": {
						"as_graph": {
							"graph": {
								"nodes": [
									{"name": "sub_op1", "target": "torch.ops.vxpu.fp8_linear.default"},
									{"name": "sub_host", "target": "torch.ops.vxpu.moe_experts.default"},
									{"name": "sub_op2", "target": "torch.ops.vxpu.fp8_linear.default"}
								]
							}
						}
					}
				}
			]
		}
	]`
	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)
	segs, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("unexpected error for nested graph with host op: %v", err)
	}
	if len(segs) != 3 {
		t.Fatalf("len(segs) = %d, want 3", len(segs))
	}
	// submod_0 (ordinary): sub_op1
	if segs[0].Host || len(segs[0].CallSites) != 1 || segs[0].CallSites[0].Node != "sub_op1" {
		t.Errorf("seg 0 = %+v, want ordinary segment with sub_op1", segs[0])
	}
	// submod_1 (host): sub_host
	if !segs[1].Host || len(segs[1].CallSites) != 1 || segs[1].CallSites[0].Node != "sub_host" {
		t.Errorf("seg 1 = %+v, want host segment with sub_host", segs[1])
	}
	// submod_2 (ordinary): sub_op2
	if segs[2].Host || len(segs[2].CallSites) != 1 || segs[2].CallSites[0].Node != "sub_op2" {
		t.Errorf("seg 2 = %+v, want ordinary segment with sub_op2", segs[2])
	}
}

func TestParseOperator(t *testing.T) {
	tests := []struct {
		target  string
		wantOp  string
		wantErr bool
	}{
		{"torch.ops.aten.select.int", "", false},
		{"torch.ops.vxpu.fp8_linear.default", "vxpu::fp8_linear", false},
		{"vxpu.moe_experts.default", "vxpu::moe_experts", false},
		{"torch.ops.vxpu..default", "", true},
		{"vxpu.", "", true},
		{"vxpu", "", false},
	}

	for _, tt := range tests {
		op, err := ParseOperator(tt.target)
		if (err != nil) != tt.wantErr {
			t.Errorf("ParseOperator(%q) error = %v, wantErr = %v", tt.target, err, tt.wantErr)
		}
		if op != tt.wantOp {
			t.Errorf("ParseOperator(%q) = %q, want %q", tt.target, op, tt.wantOp)
		}
	}
}

func TestReadCallSites(t *testing.T) {
	nodes := `[
		{"name": "n1", "target": "torch.ops.vxpu.fp8_linear.default"},
		{"name": "h1", "target": "torch.ops.vxpu.moe_experts.default"},
		{"name": "n2", "target": "torch.ops.vxpu.head_linear.default"}
	]`
	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)

	sites, err := ReadCallSites(dir)
	if err != nil {
		t.Fatalf("ReadCallSites failed: %v", err)
	}

	if len(sites) != 3 {
		t.Fatalf("ReadCallSites returned %d sites, want 3", len(sites))
	}
	wantOps := []string{"vxpu::fp8_linear", "vxpu::moe_experts", "vxpu::head_linear"}
	wantNodes := []string{"n1", "h1", "n2"}
	for i := range sites {
		if sites[i].Op != wantOps[i] {
			t.Errorf("site[%d].Op = %q, want %q", i, sites[i].Op, wantOps[i])
		}
		if sites[i].Node != wantNodes[i] {
			t.Errorf("site[%d].Node = %q, want %q", i, sites[i].Node, wantNodes[i])
		}
	}
}

func TestAttachPlan_DifferingBodies(t *testing.T) {
	tree := BuildModelTree("test_model", []*planpb.Segment{
		{
			Name: "submod_0",
			CallSites: []*planpb.CallSite{
				{Node: "site1", Op: "vxpu::fp8_linear"},
				{Node: "site2", Op: "vxpu::moe_experts"},
				{Node: "site3", Op: "vxpu::hc_front"},
			},
		},
	})

	plan := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", Body: "reference"},
			{Id: "site2", Body: "cpu-fused"},
			{Id: "site3", Body: ""}, // default implementation
		},
	}

	record := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{
				Id: "site1",
				ExecutedBodies: []*planpb.ExecutedBody{
					{Body: "reference", RowCounts: []int32{1}},
				},
			},
			{
				Id: "site2",
				ExecutedBodies: []*planpb.ExecutedBody{
					{Body: "cpu-grouped", RowCounts: []int32{2}},
					{Body: "cpu-fused", RowCounts: []int32{1}},
				},
			},
			{
				Id: "site3",
				ExecutedBodies: []*planpb.ExecutedBody{
					{Body: ""},
				},
			},
		},
	}

	if err := AttachPlan(tree, "", plan, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}

	if tree.DifferingCallSites != 1 {
		t.Errorf("tree.DifferingCallSites = %d, want 1", tree.DifferingCallSites)
	}

	cs2 := tree.Segments[0].CallSites[1]
	if len(cs2.ExecutedBodies) != 2 {
		t.Errorf("len(cs2.ExecutedBodies) = %d, want 2", len(cs2.ExecutedBodies))
	}
}

func TestAttachPlan_UnmatchedMeasurementsKept(t *testing.T) {
	tree := BuildModelTree("test_model", []*planpb.Segment{
		{
			Name: "submod_0",
			CallSites: []*planpb.CallSite{
				{Node: "site1", Op: "vxpu::fp8_linear"},
			},
		},
	})

	plan := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", Body: "reference"},
		},
	}

	record := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", ExecutedBodies: []*planpb.ExecutedBody{{Body: "reference"}}},
			{Id: "extra_record", ExecutedBodies: []*planpb.ExecutedBody{{Body: "reference"}}},
		},
	}

	if err := AttachPlan(tree, "", plan, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}

	if len(tree.UnmatchedMeasurements) != 1 {
		t.Errorf("got %d unmatched measurements, want 1", len(tree.UnmatchedMeasurements))
	}
	if tree.UnmatchedMeasurements[0].Id != "extra_record" {
		t.Errorf("tree.UnmatchedMeasurements[0].Id = %q, want 'extra_record'", tree.UnmatchedMeasurements[0].Id)
	}
}

func TestAttachPlan_PlanSiteAbsentFromRecordListed(t *testing.T) {
	tree := BuildModelTree("test_model", []*planpb.Segment{
		{
			Name: "submod_0",
			CallSites: []*planpb.CallSite{
				{Node: "site1", Op: "vxpu::fp8_linear"},
				{Node: "site2", Op: "vxpu::fp8_linear"},
			},
		},
	})

	plan := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", Body: "reference"},
			{Id: "site2", Body: "reference"},
		},
	}

	record := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			// site1 is absent from the record
			{Id: "site2", ExecutedBodies: []*planpb.ExecutedBody{{Body: "reference"}}},
		},
	}

	if err := AttachPlan(tree, "", plan, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}

	if len(tree.UnmatchedPlanSites) != 1 {
		t.Errorf("got %d unmatched plan sites, want 1", len(tree.UnmatchedPlanSites))
	} else if tree.UnmatchedPlanSites[0].Id != "site1" {
		t.Errorf("tree.UnmatchedPlanSites[0].Id = %q, want 'site1'", tree.UnmatchedPlanSites[0].Id)
	}
}

func TestAttachPlan_ThreeStates(t *testing.T) {
	tree := BuildModelTree("test_model", []*planpb.Segment{
		{
			Name: "submod_0",
			CallSites: []*planpb.CallSite{
				{Node: "site_ran_match", Op: "vxpu::fp8_linear"},
				{Node: "site_ran_diff", Op: "vxpu::fp8_linear"},
				{Node: "site_did_not_run", Op: "vxpu::moe_experts"},
				{Node: "site_not_measured", Op: "vxpu::engram_rows"},
			},
		},
	})

	plan := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site_ran_match", Body: "reference"},
			{Id: "site_ran_diff", Body: "reference"},
			{Id: "site_did_not_run", Body: "reference"},
			{Id: "site_not_measured", Body: "reference"},
		},
	}

	record := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			// State 1a: Measured and executed, matches plan
			{
				Id: "site_ran_match",
				ExecutedBodies: []*planpb.ExecutedBody{
					{Body: "reference", RowCounts: []int32{1}},
				},
			},
			// State 1b: Measured and executed, differs from plan
			{
				Id: "site_ran_diff",
				ExecutedBodies: []*planpb.ExecutedBody{
					{Body: "cpu-kernel", RowCounts: []int32{1}},
				},
			},
			// State 2: Measured and did not run (measured is true, executed_bodies is empty)
			{
				Id:             "site_did_not_run",
				Measured:       true,
				ExecutedBodies: nil,
			},
			// State 3: Not measured (measured is false, executed_bodies is empty)
			{
				Id:             "site_not_measured",
				Measured:       false,
				ExecutedBodies: nil,
			},
		},
	}

	if err := AttachPlan(tree, "", plan, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}

	// State 1b and State 2 count as differing; State 3 is excluded from differing and counted as unmeasured
	if tree.DifferingCallSites != 2 {
		t.Errorf("tree.DifferingCallSites = %d, want 2", tree.DifferingCallSites)
	}
	if tree.UnmeasuredCallSites != 1 {
		t.Errorf("tree.UnmeasuredCallSites = %d, want 1", tree.UnmeasuredCallSites)
	}

	sites := tree.Segments[0].CallSites
	// site_ran_match
	if !sites[0].GetInRecord() || !sites[0].Measured || len(sites[0].ExecutedBodies) != 1 {
		t.Errorf("site_ran_match state unexpected: InRecord=%v, Measured=%v, ExecutedBodies=%v",
			sites[0].GetInRecord(), sites[0].Measured, sites[0].ExecutedBodies)
	}
	// site_ran_diff
	if !sites[1].GetInRecord() || !sites[1].Measured || len(sites[1].ExecutedBodies) != 1 {
		t.Errorf("site_ran_diff state unexpected: InRecord=%v, Measured=%v, ExecutedBodies=%v",
			sites[1].GetInRecord(), sites[1].Measured, sites[1].ExecutedBodies)
	}
	// site_did_not_run
	if !sites[2].GetInRecord() || !sites[2].Measured || len(sites[2].ExecutedBodies) != 0 {
		t.Errorf("site_did_not_run state unexpected: InRecord=%v, Measured=%v, ExecutedBodies=%v",
			sites[2].GetInRecord(), sites[2].Measured, sites[2].ExecutedBodies)
	}
	// site_not_measured
	if !sites[3].GetInRecord() || sites[3].Measured || len(sites[3].ExecutedBodies) != 0 {
		t.Errorf("site_not_measured state unexpected: InRecord=%v, Measured=%v, ExecutedBodies=%v",
			sites[3].GetInRecord(), sites[3].Measured, sites[3].ExecutedBodies)
	}
}

func TestAttachPlan_SilenceReadsAsNotMeasured(t *testing.T) {
	// A run record written without execution measurement flags (as written by
	// main's Python today) leaves measured false and executed_bodies empty.
	// Silence in the record must read as not measured, not as differing.
	tree := BuildModelTree("test_model", []*planpb.Segment{
		{
			Name: "submod_0",
			CallSites: []*planpb.CallSite{
				{Node: "site1", Op: "vxpu::fp8_linear"},
				{Node: "site2", Op: "vxpu::moe_experts"},
				{Node: "site3", Op: "vxpu::engram_rows"},
			},
		},
	})

	plan := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", Body: "reference"},
			{Id: "site2", Body: "cpu-fused"},
			{Id: "site3", Body: "host-file"},
		},
	}

	// Record contains site timing stats but no executed bodies or measured flag
	record := &planpb.PlanFile{
		Sites: []*planpb.PlanSite{
			{Id: "site1", Stats: &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(10.0)}},
			{Id: "site2", Stats: &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(20.0)}},
			{Id: "site3", Stats: &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(30.0)}},
		},
	}

	if err := AttachPlan(tree, "", plan, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}

	if tree.DifferingCallSites != 0 {
		t.Errorf("tree.DifferingCallSites = %d, want 0", tree.DifferingCallSites)
	}
	if tree.UnmeasuredCallSites != 3 {
		t.Errorf("tree.UnmeasuredCallSites = %d, want 3", tree.UnmeasuredCallSites)
	}
	if tree.MeasuredCallSites != 3 {
		t.Errorf("tree.MeasuredCallSites = %d, want 3", tree.MeasuredCallSites)
	}
	for i, cs := range tree.Segments[0].CallSites {
		if cs.Measured {
			t.Errorf("site[%d].Measured = true, want false", i)
		}
		if len(cs.ExecutedBodies) != 0 {
			t.Errorf("site[%d].ExecutedBodies not empty", i)
		}
	}
}
