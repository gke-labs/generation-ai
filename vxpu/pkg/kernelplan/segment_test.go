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
