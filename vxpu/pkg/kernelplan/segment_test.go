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
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
	"google.golang.org/protobuf/proto"
)

func createTestArchive(t *testing.T, format, version string, major, minor int, nodesJSON string) string {
	return createTestArchiveWithSignature(t, format, version, major, minor, nodesJSON, `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "out"}}}}
		]
	}`)
}

func createTestArchiveWithSignature(t *testing.T, format, version string, major, minor int, nodesJSON, sigJSON string) string {
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
	var modelJSON string
	if sigJSON != "" {
		modelJSON = fmt.Sprintf(`{
		"schema_version": {"major": %d, "minor": %d},
		"graph_module": {
			"signature": %s,
			"graph": {
				"nodes": %s
			}
		}
	}`, major, minor, sigJSON, nodesJSON)
	} else {
		modelJSON = fmt.Sprintf(`{
		"schema_version": {"major": %d, "minor": %d},
		"graph_module": {
			"graph": {
				"nodes": %s
			}
		}
	}`, major, minor, nodesJSON)
	}
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
			"target": "torch.ops.aten.select.int",
			"outputs": [{"as_tensor": {"name": "aten_select"}}]
		},
		{
			"name": "fp8_linear_0",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"outputs": [{"as_tensor": {"name": "fp8_linear_0"}}],
			"metadata": {
				"nn_module_stack": "L__self__,,Decode;L__self__model,model,Transformer;L__self__model.layers.0.attn.wq_a,model.layers.0.attn.wq_a,Linear"
			}
		},
		{
			"name": "moe_experts_0",
			"target": "torch.ops.vxpu.moe_experts.default",
			"outputs": [{"as_tensor": {"name": "moe_experts_0"}}],
			"metadata": {
				"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.ffn,model.layers.0.ffn,MoE"
			}
		},
		{
			"name": "fp8_linear_1",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"outputs": [{"as_tensor": {"name": "fp8_linear_1"}}],
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
	if !strings.Contains(err.Error(), "pt1") || !strings.Contains(err.Error(), "pt2") {
		t.Errorf("error %q should name both the refused format pt1 and expected format pt2", err.Error())
	}
}

func TestReadSegments_RefuseUnsupportedArchiveVersion(t *testing.T) {
	dir := createTestArchive(t, "pt2", "1", 8, 20, "[]")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for unsupported archive version 1, got nil")
	}
	if !strings.Contains(err.Error(), "1") || !strings.Contains(err.Error(), "0") {
		t.Errorf("error %q should name both the refused archive version 1 and expected version 0", err.Error())
	}
}

func TestReadSegments_RefuseUnsupportedSchemaVersion(t *testing.T) {
	dir := createTestArchive(t, "pt2", "0", 9, 0, "[]")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for unsupported schema version 9.0, got nil")
	}
	if !strings.Contains(err.Error(), "9.0") || !strings.Contains(err.Error(), "8.20") {
		t.Errorf("error %q should name both the refused schema version 9.0 and expected version 8.20", err.Error())
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
			"outputs": [{"as_tensor": {"name": "hop_node"}}],
			"inputs": [
				{
					"name": "subgraph",
					"arg": {
						"as_graph": {
							"graph": {
								"nodes": [
									{"name": "sub_op", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "sub_op"}}]}
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
			"outputs": [{"as_tensor": {"name": "hop_node"}}],
			"inputs": [
				{
					"name": "subgraph",
					"arg": {
						"as_graph": {
							"graph": {
								"nodes": [
									{"name": "sub_op1", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "sub_op1"}}]},
									{"name": "sub_host", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "sub_host"}}]},
									{"name": "sub_op2", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "sub_op2"}}]}
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
		{"name": "n1", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "n1"}}]},
		{"name": "h1", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "h1"}}]},
		{"name": "n2", "target": "torch.ops.vxpu.head_linear.default", "outputs": [{"as_tensor": {"name": "n2"}}]}
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

func TestReadSegmentsFromStore_PropagatesFetchErrors(t *testing.T) {
	artDir := createTestArchive(t, "pt2", "0", 8, 20, `[{"name": "n0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "n0"}}]}]`)
	decodeBytes, err := os.ReadFile(filepath.Join(artDir, "decode.pt2"))
	if err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/vxpu-artifacts/m/resolve/v1/decode.pt2":
			_, _ = w.Write(decodeBytes)
		case "/vxpu-artifacts/m/resolve/v1/manifest.json":
			http.Error(w, "store exploded", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	_, err = ReadSegmentsFromStore(t.Context(), weightstore.New(ts.URL), "vxpu-artifacts/m", "v1")
	if err == nil || !strings.Contains(err.Error(), "manifest.json") {
		t.Fatalf("expected manifest.json fetch error, got: %v", err)
	}
}

func assertSegmentReadsWritesInvariants(t *testing.T, segments []*planpb.Segment) {
	t.Helper()
	if len(segments) == 0 {
		t.Fatalf("expected at least one segment, got 0")
	}

	writtenSoFar := make(map[string]bool)

	for i, seg := range segments {
		for _, readVal := range seg.Reads {
			isStepInput := readVal.Kind == planpb.ValueKind_VALUE_KIND_STEP_INPUT
			isResident := readVal.Kind == planpb.ValueKind_VALUE_KIND_RESIDENT
			isWrittenEarlier := writtenSoFar[readVal.Name]

			if !isStepInput && !isResident && !isWrittenEarlier {
				t.Errorf("segment %s (%d) reads %q (kind: %v) which is not a step input, not resident, and was not written by an earlier segment (written so far: %v)",
					seg.Name, i, readVal.Name, readVal.Kind, writtenSoFar)
			}
			if isResident && readVal.Resident == "" {
				t.Errorf("segment %s (%d) resident read %q has empty Resident slot name", seg.Name, i, readVal.Name)
			}
			if !isResident && readVal.Resident != "" {
				t.Errorf("segment %s (%d) non-resident read %q (kind: %v) has unexpected Resident %q", seg.Name, i, readVal.Name, readVal.Kind, readVal.Resident)
			}
		}

		for _, writeVal := range seg.Writes {
			isResident := writeVal.Kind == planpb.ValueKind_VALUE_KIND_RESIDENT
			if isResident && writeVal.Resident == "" {
				t.Errorf("segment %s (%d) resident write %q has empty Resident slot name", seg.Name, i, writeVal.Name)
			}
			if !isResident && writeVal.Resident != "" {
				t.Errorf("segment %s (%d) non-resident write %q (kind: %v) has unexpected Resident %q", seg.Name, i, writeVal.Name, writeVal.Kind, writeVal.Resident)
			}
			writtenSoFar[writeVal.Name] = true
		}
	}

	lastSeg := segments[len(segments)-1]
	hasStepOutput := false
	for _, writeVal := range lastSeg.Writes {
		if writeVal.Kind == planpb.ValueKind_VALUE_KIND_STEP_OUTPUT {
			hasStepOutput = true
			break
		}
	}
	if !hasStepOutput {
		t.Errorf("last segment %s does not write any step outputs (writes: %v)", lastSeg.Name, lastSeg.Writes)
	}
}

func getTinyLlamaArtifactDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("TINY_LLAMA_ARTIFACT"); dir != "" {
		if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err == nil {
			return dir
		}
	}
	if matches, err := filepath.Glob(filepath.Join(os.TempDir(), "tiny-llama-artifact-*", "decode.pt2")); err == nil && len(matches) > 0 {
		return filepath.Dir(matches[len(matches)-1])
	}
	pyBin := "python3"
	root := findGitRoot()
	if root != "" {
		venvPy := filepath.Join(root, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			pyBin = venvPy
		}
	}
	stagingDir, err := os.MkdirTemp(os.TempDir(), "staging-tiny-llama-*")
	if err != nil {
		t.Skipf("cannot create staging temp dir: %v", err)
	}
	defer func() {
		_ = os.RemoveAll(stagingDir)
	}()

	cmd := exec.Command(pyBin, "-c", fmt.Sprintf("import sys; from vxpu.export import export_artifact; export_artifact('trl-internal-testing/tiny-random-LlamaForCausalLM', %q, max_cache_len=64)", stagingDir))
	if root != "" {
		cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(root, "vxpu/python"))
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Skipf("tiny Llama artifact unavailable: %v (stderr: %s)", err, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(stagingDir, "decode.pt2")); err != nil {
		t.Skipf("decode.pt2 not found in generated artifact dir %s: %v", stagingDir, err)
	}

	finalDir, err := os.MkdirTemp(os.TempDir(), "tiny-llama-artifact-*")
	if err != nil {
		return stagingDir
	}
	_ = os.Remove(finalDir)
	if err := os.Rename(stagingDir, finalDir); err != nil {
		return stagingDir
	}
	return finalDir
}

func TestReadSegments_ReadsWrites_TinyDeepSeek(t *testing.T) {
	artDir := getTinyDeepSeekArtifactDir(t)
	segments, err := ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}
	assertSegmentReadsWritesInvariants(t, segments)
}

func TestReadSegments_ReadsWrites_TinyLlama(t *testing.T) {
	artDir := getTinyLlamaArtifactDir(t)
	segments, err := ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}
	assertSegmentReadsWritesInvariants(t, segments)
}

func TestReadSegments_RefuseUndeclaredInput(t *testing.T) {
	nodes := `[
		{
			"name": "op0",
			"target": "torch.ops.aten.select.int",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "undeclared_value_xyz"}}}
			],
			"outputs": [{"as_tensor": {"name": "op0"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "op0"}}}}
		]
	}`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for undeclared input, got nil")
	}
	if !strings.Contains(err.Error(), "undeclared_value_xyz") {
		t.Fatalf("expected error naming undeclared_value_xyz, got: %v", err)
	}
}

func TestReadSegments_RefuseMissingSignature(t *testing.T) {
	nodes := `[
		{"name": "op0", "target": "torch.ops.aten.select.int", "outputs": [{"as_tensor": {"name": "op0"}}]}
	]`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, "")
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for missing signature, got nil")
	}
	if !strings.Contains(err.Error(), "signature") {
		t.Fatalf("expected error mentioning signature, got: %v", err)
	}
}

func TestReadSegments_RefuseMissingOutputs(t *testing.T) {
	nodes := `[
		{"name": "op_no_outputs", "target": "torch.ops.aten.select.int"}
	]`
	dir := createTestArchive(t, "pt2", "0", 8, 20, nodes)
	_, err := ReadSegments(dir)
	if err == nil {
		t.Fatal("expected error for node missing outputs, got nil")
	}
	if !strings.Contains(err.Error(), "op_no_outputs") || !strings.Contains(err.Error(), "outputs") {
		t.Fatalf("expected error naming op_no_outputs and outputs, got: %v", err)
	}
}

func TestReadSegments_VoidNodeEmptyOutputs(t *testing.T) {
	nodes := `[
		{
			"name": "assert_meta",
			"target": "torch.ops.aten._assert_tensor_metadata.default",
			"inputs": [{"name": "a", "arg": {"as_tensor": {"name": "input_ids"}}}],
			"outputs": []
		},
		{
			"name": "op_head",
			"target": "torch.ops.vxpu.head_linear.default",
			"inputs": [{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}],
			"outputs": [{"as_tensor": {"name": "op_head"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "op_head"}}}}
		]
	}`
	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed on void node: %v", err)
	}
	assertSegmentReadsWritesInvariants(t, segments)
}

func TestReadSegments_BufferMutation(t *testing.T) {
	nodes := `[
		{
			"name": "op0_update",
			"target": "torch.ops.aten.index_copy.default",
			"inputs": [
				{"name": "cache", "arg": {"as_tensor": {"name": "b_cache"}}},
				{"name": "val", "arg": {"as_tensor": {"name": "input_ids"}}}
			],
			"outputs": [{"as_tensor": {"name": "op0_update"}}]
		},
		{
			"name": "host_op",
			"target": "torch.ops.vxpu.moe_experts.default",
			"inputs": [
				{"name": "x", "arg": {"as_tensor": {"name": "op0_update"}}}
			],
			"outputs": [{"as_tensor": {"name": "host_op"}}]
		},
		{
			"name": "op2_head",
			"target": "torch.ops.vxpu.head_linear.default",
			"inputs": [
				{"name": "x", "arg": {"as_tensor": {"name": "host_op"}}}
			],
			"outputs": [{"as_tensor": {"name": "op2_head"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}},
			{"buffer": {"arg": {"name": "b_cache"}, "buffer_name": "model.layers.0.cache"}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "op2_head"}}}},
			{"buffer_mutation": {"arg": {"name": "op0_update"}, "buffer_name": "model.layers.0.cache"}}
		]
	}`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	if len(segments) != 3 {
		t.Fatalf("got %d segments, want 3", len(segments))
	}
	assertSegmentReadsWritesInvariants(t, segments)

	// submod_0 reads b_cache using the same resident slot name (model.layers.0.cache).
	foundCacheRead := false
	for _, r := range segments[0].Reads {
		if r.Name == "b_cache" {
			foundCacheRead = true
			if r.Kind != planpb.ValueKind_VALUE_KIND_RESIDENT || r.Resident != "model.layers.0.cache" {
				t.Errorf("submod_0 read b_cache = %+v, want RESIDENT with Resident=model.layers.0.cache", r)
			}
		}
	}
	if !foundCacheRead {
		t.Errorf("submod_0 reads missing b_cache: %+v", segments[0].Reads)
	}

	// submod_0 produces op0_update which is a buffer mutation for model.layers.0.cache.
	// It must be written by submod_0 as VALUE_KIND_RESIDENT naming both the output tensor
	// (Name: op0_update) and the resident slot it replaces (Resident: model.layers.0.cache).
	foundMutation := false
	for _, w := range segments[0].Writes {
		if w.Name == "op0_update" && w.Resident == "model.layers.0.cache" {
			foundMutation = true
			if w.Kind != planpb.ValueKind_VALUE_KIND_RESIDENT {
				t.Errorf("op0_update kind = %v, want VALUE_KIND_RESIDENT", w.Kind)
			}
		}
	}
	if !foundMutation {
		t.Errorf("submod_0 writes missing op0_update -> model.layers.0.cache: %+v", segments[0].Writes)
	}

	// submod_1 reads op0_update in the same step and sees the new state; it is marked
	// consistently as VALUE_KIND_RESIDENT with the same Resident slot name.
	if len(segments[1].Reads) != 1 || segments[1].Reads[0].Name != "op0_update" ||
		segments[1].Reads[0].Kind != planpb.ValueKind_VALUE_KIND_RESIDENT ||
		segments[1].Reads[0].Resident != "model.layers.0.cache" {
		t.Errorf("submod_1 reads = %+v, want [op0_update RESIDENT model.layers.0.cache]", segments[1].Reads)
	}

	// The buffer mutation must NOT be duplicated in the last segment's writes.
	lastSeg := segments[len(segments)-1]
	for _, w := range lastSeg.Writes {
		if w.Resident == "model.layers.0.cache" || w.Name == "op0_update" {
			t.Errorf("last segment duplicated buffer mutation in writes: %+v", lastSeg.Writes)
		}
	}
}

func TestReadSegments_MultiOutputNode(t *testing.T) {
	nodes := `[
		{
			"name": "host_split",
			"target": "torch.ops.vxpu.moe_experts.default",
			"inputs": [
				{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}
			],
			"outputs": [
				{"as_tensor": {"name": "out_expert_weights"}},
				{"as_tensor": {"name": "out_expert_indices"}}
			]
		},
		{
			"name": "op2_linear",
			"target": "torch.ops.vxpu.head_linear.default",
			"inputs": [
				{"name": "weights", "arg": {"as_tensor": {"name": "out_expert_weights"}}},
				{"name": "indices", "arg": {"as_tensor": {"name": "out_expert_indices"}}}
			],
			"outputs": [{"as_tensor": {"name": "op2_linear"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "op2_linear"}}}}
		]
	}`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	if len(segments) != 2 {
		t.Fatalf("got %d segments, want 2", len(segments))
	}

	// submod_0 (host) produces both out_expert_weights and out_expert_indices.
	// Both must be written as intermediate values by submod_0.
	writes := make(map[string]planpb.ValueKind)
	for _, w := range segments[0].Writes {
		writes[w.Name] = w.Kind
	}
	if writes["out_expert_weights"] != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("out_expert_weights write kind = %v, want VALUE_KIND_INTERMEDIATE", writes["out_expert_weights"])
	}
	if writes["out_expert_indices"] != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("out_expert_indices write kind = %v, want VALUE_KIND_INTERMEDIATE", writes["out_expert_indices"])
	}

	// submod_1 reads both out_expert_weights and out_expert_indices as intermediate values.
	reads := make(map[string]planpb.ValueKind)
	for _, r := range segments[1].Reads {
		reads[r.Name] = r.Kind
	}
	if reads["out_expert_weights"] != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("out_expert_weights read kind = %v, want VALUE_KIND_INTERMEDIATE", reads["out_expert_weights"])
	}
	if reads["out_expert_indices"] != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("out_expert_indices read kind = %v, want VALUE_KIND_INTERMEDIATE", reads["out_expert_indices"])
	}
}

func TestReadSegments_ReadsWritesSynthetic_MultiSegment(t *testing.T) {
	nodes := `[
		{
			"name": "op0_select",
			"target": "torch.ops.aten.select.int",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "input_ids"}}},
				{"name": "dim", "arg": {"as_int": 0}}
			],
			"outputs": [{"as_tensor": {"name": "op0_select"}}]
		},
		{
			"name": "op0_linear",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "op0_select"}}},
				{"name": "weight", "arg": {"as_tensor": {"name": "p_model_layers_0_attn_wq_a_weight"}}}
			],
			"outputs": [{"as_tensor": {"name": "op0_linear"}}]
		},
		{
			"name": "host_experts",
			"target": "torch.ops.vxpu.moe_experts.default",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "op0_linear"}}},
				{"name": "cache", "arg": {"as_tensor": {"name": "b_model_layers_0_cache"}}}
			],
			"outputs": [{"as_tensor": {"name": "host_experts"}}]
		},
		{
			"name": "op2_head",
			"target": "torch.ops.vxpu.head_linear.default",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "host_experts"}}},
				{"name": "weight", "arg": {"as_tensor": {"name": "p_model_head_weight"}}}
			],
			"outputs": [{"as_tensor": {"name": "op2_head"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}},
			{"parameter": {"arg": {"name": "p_model_layers_0_attn_wq_a_weight"}, "parameter_name": "model.layers.0.attn.wq_a.weight"}},
			{"buffer": {"arg": {"name": "b_model_layers_0_cache"}, "buffer_name": "model.layers.0.cache"}},
			{"parameter": {"arg": {"name": "p_model_head_weight"}, "parameter_name": "model.head.weight"}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "op2_head"}}}}
		]
	}`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	if len(segments) != 3 {
		t.Fatalf("got %d segments, want 3", len(segments))
	}

	// Verify invariants
	assertSegmentReadsWritesInvariants(t, segments)

	// In submod_0:
	// reads: input_ids (STEP_INPUT), p_model_layers_0_attn_wq_a_weight (RESIDENT: model.layers.0.attn.wq_a.weight)
	// writes: op0_linear (INTERMEDIATE)
	if len(segments[0].Reads) != 2 {
		t.Errorf("submod_0 reads count = %d, want 2", len(segments[0].Reads))
	} else {
		if segments[0].Reads[0].Name != "input_ids" || segments[0].Reads[0].Kind != planpb.ValueKind_VALUE_KIND_STEP_INPUT {
			t.Errorf("submod_0 reads[0] = %+v, want input_ids (STEP_INPUT)", segments[0].Reads[0])
		}
		if segments[0].Reads[1].Name != "p_model_layers_0_attn_wq_a_weight" || segments[0].Reads[1].Kind != planpb.ValueKind_VALUE_KIND_RESIDENT || segments[0].Reads[1].Resident != "model.layers.0.attn.wq_a.weight" {
			t.Errorf("submod_0 reads[1] = %+v, want p_model_layers_0_attn_wq_a_weight (RESIDENT: model.layers.0.attn.wq_a.weight)", segments[0].Reads[1])
		}
	}
	if len(segments[0].Writes) != 1 || segments[0].Writes[0].Name != "op0_linear" || segments[0].Writes[0].Kind != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("submod_0 writes = %+v, want [op0_linear (INTERMEDIATE)]", segments[0].Writes)
	}

	// In submod_1 (host):
	// reads: op0_linear (INTERMEDIATE), b_model_layers_0_cache (RESIDENT: model.layers.0.cache)
	// writes: host_experts (INTERMEDIATE)
	if len(segments[1].Reads) != 2 {
		t.Errorf("submod_1 reads count = %d, want 2", len(segments[1].Reads))
	} else {
		if segments[1].Reads[0].Name != "op0_linear" || segments[1].Reads[0].Kind != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
			t.Errorf("submod_1 reads[0] = %+v, want op0_linear (INTERMEDIATE)", segments[1].Reads[0])
		}
		if segments[1].Reads[1].Name != "b_model_layers_0_cache" || segments[1].Reads[1].Kind != planpb.ValueKind_VALUE_KIND_RESIDENT || segments[1].Reads[1].Resident != "model.layers.0.cache" {
			t.Errorf("submod_1 reads[1] = %+v, want b_model_layers_0_cache (RESIDENT: model.layers.0.cache)", segments[1].Reads[1])
		}
	}
	if len(segments[1].Writes) != 1 || segments[1].Writes[0].Name != "host_experts" || segments[1].Writes[0].Kind != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
		t.Errorf("submod_1 writes = %+v, want [host_experts (INTERMEDIATE)]", segments[1].Writes)
	}

	// In submod_2 (dense):
	// reads: host_experts (INTERMEDIATE), p_model_head_weight (RESIDENT: model.head.weight)
	// writes: op2_head (STEP_OUTPUT)
	if len(segments[2].Reads) != 2 {
		t.Errorf("submod_2 reads count = %d, want 2", len(segments[2].Reads))
	} else {
		if segments[2].Reads[0].Name != "host_experts" || segments[2].Reads[0].Kind != planpb.ValueKind_VALUE_KIND_INTERMEDIATE {
			t.Errorf("submod_2 reads[0] = %+v, want host_experts (INTERMEDIATE)", segments[2].Reads[0])
		}
		if segments[2].Reads[1].Name != "p_model_head_weight" || segments[2].Reads[1].Kind != planpb.ValueKind_VALUE_KIND_RESIDENT || segments[2].Reads[1].Resident != "model.head.weight" {
			t.Errorf("submod_2 reads[1] = %+v, want p_model_head_weight (RESIDENT: model.head.weight)", segments[2].Reads[1])
		}
	}
	if len(segments[2].Writes) != 1 || segments[2].Writes[0].Name != "op2_head" || segments[2].Writes[0].Kind != planpb.ValueKind_VALUE_KIND_STEP_OUTPUT {
		t.Errorf("submod_2 writes = %+v, want [op2_head (STEP_OUTPUT)]", segments[2].Writes)
	}
}

func TestReadSegments_ReadsWritesSynthetic_SingleSegment(t *testing.T) {
	nodes := `[
		{
			"name": "op0_linear",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "input_ids"}}},
				{"name": "weight", "arg": {"as_tensor": {"name": "p_w"}}}
			],
			"outputs": [{"as_tensor": {"name": "op0_linear"}}]
		},
		{
			"name": "head",
			"target": "torch.ops.vxpu.head_linear.default",
			"inputs": [
				{"name": "input", "arg": {"as_tensor": {"name": "op0_linear"}}},
				{"name": "weight", "arg": {"as_tensor": {"name": "p_head"}}}
			],
			"outputs": [{"as_tensor": {"name": "head"}}]
		}
	]`
	sig := `{
		"input_specs": [
			{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}},
			{"parameter": {"arg": {"name": "p_w"}, "parameter_name": "model.w"}},
			{"parameter": {"arg": {"name": "p_head"}, "parameter_name": "model.head"}}
		],
		"output_specs": [
			{"user_output": {"arg": {"as_tensor": {"name": "head"}}}}
		]
	}`

	dir := createTestArchiveWithSignature(t, "pt2", "0", 8, 20, nodes, sig)
	segments, err := ReadSegments(dir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	if len(segments) != 1 {
		t.Fatalf("got %d segments, want 1", len(segments))
	}

	assertSegmentReadsWritesInvariants(t, segments)

	if len(segments[0].Reads) != 3 {
		t.Errorf("reads count = %d, want 3", len(segments[0].Reads))
	}
	if len(segments[0].Writes) != 1 || segments[0].Writes[0].Name != "head" || segments[0].Writes[0].Kind != planpb.ValueKind_VALUE_KIND_STEP_OUTPUT {
		t.Errorf("writes = %+v, want [head (STEP_OUTPUT)]", segments[0].Writes)
	}
}

func TestAttachPlan_SegmentTotalMeasuredTimeFallback(t *testing.T) {
	tree := &planpb.ModelTree{
		ArtifactMd5: "art1",
		Segments: []*planpb.Segment{
			{
				Name: "submod_0",
				Host: false,
				CallSites: []*planpb.CallSite{
					{Node: "fp8_linear_0", Op: "vxpu::fp8_linear"},
				},
			},
			{
				Name: "submod_1",
				Host: true,
				CallSites: []*planpb.CallSite{
					{Node: "moe_experts", Op: "vxpu::moe_experts"},
				},
			},
		},
	}
	record := &planpb.PlanFile{
		ArtifactMd5: "art1",
		Sites: []*planpb.PlanSite{
			{Id: "fp8_linear_0"},
			{
				Id:             "moe_experts",
				Measured:       true,
				ExecutedBodies: []*planpb.ExecutedBody{{Body: "cpu-fused", RowCounts: []int32{1}}},
				Stats:          &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(42.5)},
			},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Host: false, TotalMeasuredTimeUs: proto.Float64(900.0)},
			{Name: "submod_1", Host: true, TotalMeasuredTimeUs: proto.Float64(42.5)},
		},
	}

	if err := AttachPlan(tree, "", nil, record); err != nil {
		t.Fatalf("AttachPlan failed: %v", err)
	}
	if tree.Segments[0].TotalMeasuredTimeUs == nil || tree.Segments[0].GetTotalMeasuredTimeUs() != 900.0 {
		t.Errorf("submod_0 TotalMeasuredTimeUs = %v, want 900.0", tree.Segments[0].TotalMeasuredTimeUs)
	}
	if tree.Segments[1].TotalMeasuredTimeUs == nil || tree.Segments[1].GetTotalMeasuredTimeUs() != 42.5 {
		t.Errorf("submod_1 TotalMeasuredTimeUs = %v, want 42.5", tree.Segments[1].TotalMeasuredTimeUs)
	}
	if tree.UnmeasuredCallSites != 1 {
		t.Errorf("UnmeasuredCallSites = %d, want 1", tree.UnmeasuredCallSites)
	}
}
