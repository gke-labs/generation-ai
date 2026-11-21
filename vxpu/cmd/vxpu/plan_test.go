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
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

func TestCmdPlanStep(t *testing.T) {
	tmpDir := t.TempDir()
	graphPath := filepath.Join(tmpDir, "stepgraph.json")
	planPath := filepath.Join(tmpDir, "stepplan.pb")
	serializedPlanPath := filepath.Join(tmpDir, "stepplan_serialized.pb")
	ctx := t.Context()

	smallGraphJSON := `{
  "nodes": [
    {
      "name": "n0",
      "execution": "EXECUTION_GRAPH",
      "kind": "dense",
      "reads": [{"input": "x"}]
    },
    {
      "name": "n1",
      "execution": "EXECUTION_GRAPH",
      "kind": "dense",
      "reads": [{"edge": {"node": "n0"}}]
    },
    {
      "name": "n2",
      "execution": "EXECUTION_GRAPH",
      "kind": "dense",
      "reads": [{"input": "y"}]
    },
    {
      "name": "n3",
      "execution": "EXECUTION_EXCHANGE",
      "kind": "experts",
      "host": {"collection": "experts", "index": 0},
      "answer": "ANSWER_GPU_POOL",
      "reads": [{"host_mapping": "experts.0"}]
    },
    {
      "name": "n4",
      "execution": "EXECUTION_EXCHANGE",
      "kind": "experts",
      "host": {"collection": "experts", "index": 1},
      "answer": "ANSWER_CPU",
      "reads": [{"host_mapping": "experts.1"}]
    }
  ],
  "host_mappings": [
    {"name": "experts.0", "file_backed": true},
    {"name": "experts.1", "file_backed": true},
    {"name": "engram.0", "file_backed": false}
  ],
  "inputs": ["x", "y"]
}`
	if err := os.WriteFile(graphPath, []byte(smallGraphJSON), 0644); err != nil {
		t.Fatalf("WriteFile stepgraph failed: %v", err)
	}

	if err := cmdPlan(ctx, []string{"step", "--graph", graphPath, "--out", planPath}); err != nil {
		t.Fatalf("cmdPlan step failed: %v", err)
	}

	plan, err := stepplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan stepplan failed: %v", err)
	}

	if len(plan.Nodes) != 5 {
		t.Fatalf("got %d nodes, want 5", len(plan.Nodes))
	}

	// Verify release_host_mappings contains experts.0 and not experts.1 or engram.0
	if len(plan.ReleaseHostMappings) != 1 || plan.ReleaseHostMappings[0] != "experts.0" {
		t.Errorf("plan.ReleaseHostMappings = %v, want [\"experts.0\"]", plan.ReleaseHostMappings)
	}

	// n0 is on main, n1 reads n0 (stays on main), n2 reads y (parallel stream)
	if plan.Nodes[0].Stream != stepplan.StreamMain {
		t.Errorf("node n0 stream = %v, want StreamMain", plan.Nodes[0].Stream)
	}
	if plan.Nodes[1].Stream != stepplan.StreamMain {
		t.Errorf("node n1 stream = %v, want StreamMain", plan.Nodes[1].Stream)
	}
	if plan.Nodes[2].Stream != stepplan.StreamParallel {
		t.Errorf("node n2 stream = %v, want StreamParallel", plan.Nodes[2].Stream)
	}

	// Test with --serialize
	if err := cmdPlan(ctx, []string{"step", "--graph", graphPath, "--out", serializedPlanPath, "--serialize"}); err != nil {
		t.Fatalf("cmdPlan step --serialize failed: %v", err)
	}

	serPlan, err := stepplan.ReadPlan(serializedPlanPath)
	if err != nil {
		t.Fatalf("ReadPlan serialized stepplan failed: %v", err)
	}

	for _, n := range serPlan.Nodes {
		if n.Stream != stepplan.StreamMain {
			t.Errorf("serialized node %s stream = %v, want StreamMain", n.Name, n.Stream)
		}
	}
	if len(serPlan.Nodes[1].Waits) != 1 || serPlan.Nodes[1].Waits[0] != "n0" {
		t.Errorf("serialized node n1 waits = %v, want [n0]", serPlan.Nodes[1].Waits)
	}
	if len(serPlan.Nodes[2].Waits) != 1 || serPlan.Nodes[2].Waits[0] != "n1" {
		t.Errorf("serialized node n2 waits = %v, want [n1]", serPlan.Nodes[2].Waits)
	}

	// Test with --json
	jsonPlanPath := filepath.Join(tmpDir, "stepplan.json")
	if err := cmdPlan(ctx, []string{"step", "--graph", graphPath, "--out", jsonPlanPath, "--json"}); err != nil {
		t.Fatalf("cmdPlan step --json failed: %v", err)
	}
	jsonData, err := os.ReadFile(jsonPlanPath)
	if err != nil {
		t.Fatalf("ReadFile json stepplan failed: %v", err)
	}
	var jsonPlan planpb.StepPlan
	if err := protojson.Unmarshal(jsonData, &jsonPlan); err != nil {
		t.Fatalf("protojson.Unmarshal json stepplan failed: %v", err)
	}
	if len(jsonPlan.Nodes) != 5 {
		t.Errorf("jsonPlan.Nodes = %d, want 5", len(jsonPlan.Nodes))
	}
}

func TestCmdPlanEmit(t *testing.T) {
	tmpDir := t.TempDir()
	artDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	planPath := filepath.Join(tmpDir, "plan.pb")
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath, "--rows", "1,5,8", "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	plan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan tree_commit = %q, want commit123", plan.TreeCommit)
	}
	if len(plan.Sites) == 0 {
		t.Error("expected at least one site in plan")
	}
}

func TestCmdPlanErrors(t *testing.T) {
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{}); err == nil {
		t.Error("expected error for empty plan args, got nil")
	}
	if err := cmdPlan(ctx, []string{"unknown"}); err == nil {
		t.Error("expected error for unknown plan subcommand, got nil")
	}
	if err := cmdPlan(ctx, []string{"step"}); err == nil {
		t.Error("expected error for missing --graph in step, got nil")
	}
	emptyGraphPath := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(emptyGraphPath, []byte("{}"), 0644); err != nil {
		t.Fatalf("WriteFile empty graph failed: %v", err)
	}
	if err := cmdPlan(ctx, []string{"step", "--graph", emptyGraphPath}); err == nil {
		t.Error("expected error for empty step graph description, got nil")
	}
	if err := cmdPlan(ctx, []string{"emit"}); err == nil {
		t.Error("expected error for missing --artifact, got nil")
	}
	if err := cmdPlan(ctx, []string{"emit", "--artifact", "art", "--rows", "abc", "--tree-commit", "commit123"}); err == nil {
		t.Error("expected error for invalid --rows, got nil")
	}
	if err := cmdPlan(ctx, []string{"emit", "--artifact", "art", "--rows", ",", "--tree-commit", "commit123"}); err == nil {
		t.Error("expected error for comma-only --rows, got nil")
	}
	if err := cmdPlan(ctx, []string{"emit", "--artifact", "art", "--tree-commit", "unknown"}); err == nil {
		t.Error("expected error for unknown --tree-commit, got nil")
	}
}

func TestCmdPlanShowAndStepPlan(t *testing.T) {
	ctx := t.Context()
	dir := t.TempDir()
	planPath := filepath.Join(dir, "stepplan.json")
	planJSON := `{
  "nodes": [
    {
      "name": "n0",
      "stream": "STREAM_MAIN",
      "stats": {
        "device_time_ms": 0.052,
        "sample_count": 8
      }
    },
    {
      "name": "n1",
      "stream": "STREAM_PARALLEL",
      "waits": ["n0"]
    }
  ]
}`
	if err := os.WriteFile(planPath, []byte(planJSON), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	graphPath := filepath.Join(dir, "stepgraph.json")
	graphJSON := `{
  "nodes": [
    {
      "name": "n0",
      "execution": "EXECUTION_GRAPH",
      "kind": "dense"
    },
    {
      "name": "n1",
      "execution": "EXECUTION_EXCHANGE",
      "kind": "experts",
      "host": {"collection": "experts"}
    }
  ]
}`
	if err := os.WriteFile(graphPath, []byte(graphJSON), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	// Test vxpu plan show --type stepplan <file>
	if err := cmdPlan(ctx, []string{"show", "--type", "stepplan", planPath}); err != nil {
		t.Fatalf("cmdPlan show failed: %v", err)
	}

	// Test vxpu plan show --type stepplan --graph <file> <plan>
	if err := cmdPlan(ctx, []string{"show", "--type", "stepplan", "--graph", graphPath, planPath}); err != nil {
		t.Fatalf("cmdPlan show with --graph failed: %v", err)
	}

	// Test vxpu plan show --type stepplan --json <file>
	if err := cmdPlan(ctx, []string{"show", "--type", "stepplan", "--json", planPath}); err != nil {
		t.Fatalf("cmdPlan show --json failed: %v", err)
	}

	// Test vxpu plan show error without --type
	if err := cmdPlan(ctx, []string{"show", planPath}); err == nil {
		t.Error("expected error for show without --type, got nil")
	}

	// Test vxpu plan show error without target arg
	if err := cmdPlan(ctx, []string{"show", "--type", "stepplan"}); err == nil {
		t.Error("expected error for show without args, got nil")
	}

	// Test vxpu plan step requires --graph
	if err := cmdPlan(ctx, []string{"step", planPath}); err == nil {
		t.Error("expected error for step without --graph, got nil")
	}
}

func captureStdout(t *testing.T, fn func() error) (string, error) {
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	defer func() {
		os.Stdout = orig
	}()

	runErr := fn()
	_ = w.Close()

	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	return buf.String(), runErr
}

func TestCmdPlanShowAllDocumentTypesJSON(t *testing.T) {
	ctx := t.Context()
	dir := t.TempDir()

	docs := []struct {
		name    string
		docType string
		msg     proto.Message
		newMsg  func() proto.Message
	}{
		{
			name:    "stepplan",
			docType: "stepplan",
			msg: &planpb.StepPlan{
				Nodes: []*planpb.NodePlan{
					{Name: "n0", Stream: planpb.Stream_STREAM_MAIN},
				},
			},
			newMsg: func() proto.Message { return new(planpb.StepPlan) },
		},
		{
			name:    "stepgraph",
			docType: "stepgraph",
			msg: &planpb.StepGraphDescription{
				Nodes: []*planpb.StepNode{
					{Name: "n0", Kind: "dense", Execution: planpb.Execution_EXECUTION_GRAPH},
				},
			},
			newMsg: func() proto.Message { return new(planpb.StepGraphDescription) },
		},
		{
			name:    "plan",
			docType: "plan",
			msg: &planpb.PlanFile{
				ArtifactMd5: "md5_123",
				TreeCommit:  "commit_456",
			},
			newMsg: func() proto.Message { return new(planpb.PlanFile) },
		},
		{
			name:    "kernel_record",
			docType: "record",
			msg: &planpb.KernelRecord{
				Device: "cpu",
				Sites: []*planpb.SiteRecord{
					{Op: "vxpu::fp8_linear", Node: "fp8_0"},
				},
			},
			newMsg: func() proto.Message { return new(planpb.KernelRecord) },
		},
		{
			name:    "kernel_catalog",
			docType: "catalog",
			msg: &planpb.KernelCatalog{
				Bodies: []*planpb.KernelBody{
					{Name: "ref_linear", Op: "vxpu::fp8_linear"},
				},
			},
			newMsg: func() proto.Message { return new(planpb.KernelCatalog) },
		},
	}

	for _, tc := range docs {
		t.Run(tc.name+"_binary", func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".pb")
			binData, err := proto.Marshal(tc.msg)
			if err != nil {
				t.Fatalf("proto.Marshal %s: %v", tc.name, err)
			}
			if err := os.WriteFile(path, binData, 0644); err != nil {
				t.Fatalf("WriteFile %s: %v", tc.name, err)
			}

			out, err := captureStdout(t, func() error {
				return cmdPlan(ctx, []string{"show", "--type", tc.docType, "--json", path})
			})
			if err != nil {
				t.Fatalf("cmdPlan show --json %s failed: %v", tc.name, err)
			}

			got := tc.newMsg()
			if err := protojson.Unmarshal([]byte(out), got); err != nil {
				t.Fatalf("protojson.Unmarshal %s output failed: %v\nOutput: %s", tc.name, err, out)
			}
			if !proto.Equal(got, tc.msg) {
				t.Errorf("plan show --json %s mismatch:\ngot:  %v\nwant: %v", tc.name, got, tc.msg)
			}
		})

		t.Run(tc.name+"_json", func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".json")
			jsonData, err := kernelplan.MarshalAsJSON(tc.msg)
			if err != nil {
				t.Fatalf("MarshalAsJSON %s: %v", tc.name, err)
			}
			if err := os.WriteFile(path, jsonData, 0644); err != nil {
				t.Fatalf("WriteFile %s: %v", tc.name, err)
			}

			out, err := captureStdout(t, func() error {
				return cmdPlan(ctx, []string{"show", "--type", tc.docType, "--json", path})
			})
			if err != nil {
				t.Fatalf("cmdPlan show --json %s failed: %v", tc.name, err)
			}

			got := tc.newMsg()
			if err := protojson.Unmarshal([]byte(out), got); err != nil {
				t.Fatalf("protojson.Unmarshal %s output failed: %v\nOutput: %s", tc.name, err, out)
			}
			if !proto.Equal(got, tc.msg) {
				t.Errorf("plan show --json %s mismatch:\ngot:  %v\nwant: %v", tc.name, got, tc.msg)
			}
		})
	}

	// KernelTable is always JSON, whether with .json extension or without extension
	tableMsg := &planpb.KernelTable{
		Gpu: proto.String("cpu"),
		Sites: []*planpb.SiteEntry{
			{Op: "vxpu::fp8_linear", Impl: "ref_linear"},
		},
	}
	tableJSON, err := kernelplan.MarshalAsJSON(tableMsg)
	if err != nil {
		t.Fatalf("MarshalAsJSON table: %v", err)
	}

	for _, ext := range []string{".json", "_noext"} {
		t.Run("kernel_table"+ext, func(t *testing.T) {
			path := filepath.Join(dir, "kernel_table"+ext)
			if err := os.WriteFile(path, tableJSON, 0644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}
			out, err := captureStdout(t, func() error {
				return cmdPlan(ctx, []string{"show", "--type", "table", "--json", path})
			})
			if err != nil {
				t.Fatalf("cmdPlan show --type table failed: %v", err)
			}
			got := new(planpb.KernelTable)
			if err := protojson.Unmarshal([]byte(out), got); err != nil {
				t.Fatalf("protojson.Unmarshal table output failed: %v", err)
			}
			if !proto.Equal(got, tableMsg) {
				t.Errorf("table mismatch:\ngot:  %v\nwant: %v", got, tableMsg)
			}
		})
	}
}

func TestOutputExtensionDecidesFormat(t *testing.T) {
	tmpDir := t.TempDir()
	artDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	ctx := t.Context()

	// 1. plan emit writing to .json produces JSON
	jsonPath := filepath.Join(tmpDir, "emitted_plan.json")
	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", jsonPath, "--rows", "1", "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("emit to .json failed: %v", err)
	}
	rawJSON, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("ReadFile json: %v", err)
	}
	var pFromJSON planpb.PlanFile
	if err := protojson.Unmarshal(rawJSON, &pFromJSON); err != nil {
		t.Fatalf("emitted .json file is not valid JSON: %v", err)
	}

	// 2. plan emit writing to .pb produces binary
	pbPath := filepath.Join(tmpDir, "emitted_plan.pb")
	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", pbPath, "--rows", "1", "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("emit to .pb failed: %v", err)
	}
	rawPB, err := os.ReadFile(pbPath)
	if err != nil {
		t.Fatalf("ReadFile pb: %v", err)
	}
	var pFromPB planpb.PlanFile
	if err := proto.Unmarshal(rawPB, &pFromPB); err != nil {
		t.Fatalf("emitted .pb file is not valid binary: %v", err)
	}

	// 3. plan emit with conflicting --json and .pb extension is refused
	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--json", "--out", pbPath, "--rows", "1", "--tree-commit", "commit123"}); err == nil {
		t.Error("expected error for --json with .pb output extension, got nil")
	}

	// 4. plan step with conflicting --json and .pb extension is refused
	graphPath := filepath.Join(tmpDir, "stepgraph.json")
	if err := os.WriteFile(graphPath, []byte(`{"nodes":[{"name":"n0"}]}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	stepPB := filepath.Join(tmpDir, "stepplan.pb")
	if err := cmdPlan(ctx, []string{"step", "--graph", graphPath, "--json", "--out", stepPB}); err == nil {
		t.Error("expected error for step --json with .pb output extension, got nil")
	}
}

func TestCmdPlanEmitBinarySitesAndCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	artDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")

	var kr planpb.KernelRecord
	if err := kernelplan.ReadJSON(filepath.Join(fixtureDir, "kernel_record.json"), &kr); err != nil {
		t.Fatalf("ReadJSON kernel_record.json: %v", err)
	}
	sitesData, err := proto.Marshal(&kr)
	if err != nil {
		t.Fatalf("proto.Marshal sites: %v", err)
	}
	sitesPath := filepath.Join(tmpDir, "sites.pb")
	if err := os.WriteFile(sitesPath, sitesData, 0644); err != nil {
		t.Fatalf("WriteFile sites.pb: %v", err)
	}

	var cat planpb.KernelCatalog
	if err := kernelplan.ReadJSON(filepath.Join(fixtureDir, "kernel_catalog.json"), &cat); err != nil {
		t.Fatalf("ReadJSON kernel_catalog.json: %v", err)
	}
	catData, err := proto.Marshal(&cat)
	if err != nil {
		t.Fatalf("proto.Marshal catalog: %v", err)
	}
	catPath := filepath.Join(tmpDir, "catalog.pb")
	if err := os.WriteFile(catPath, catData, 0644); err != nil {
		t.Fatalf("WriteFile catalog.pb: %v", err)
	}

	planPath := filepath.Join(tmpDir, "plan.pb")
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--sites", sitesPath, "--catalog", catPath, "--out", planPath, "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("cmdPlan emit failed with binary sites and catalog: %v", err)
	}

	plan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan tree_commit = %q, want commit123", plan.TreeCommit)
	}
	if len(plan.Sites) != 87 {
		t.Errorf("plan.Sites count = %d, want 87", len(plan.Sites))
	}
}

func TestCmdPlanEmitJSON(t *testing.T) {
	tmpDir := t.TempDir()
	artDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	planPath := filepath.Join(tmpDir, "plan.json")
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath, "--json", "--rows", "1,5", "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("cmdPlan emit --json failed: %v", err)
	}

	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("ReadFile plan failed: %v", err)
	}

	var plan planpb.PlanFile
	if err := protojson.Unmarshal(planBytes, &plan); err != nil {
		t.Fatalf("protojson.Unmarshal plan failed: %v", err)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan tree_commit = %q, want commit123", plan.TreeCommit)
	}
}

func TestCmdPlanSegments(t *testing.T) {
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"segments"}); err == nil {
		t.Error("expected error for missing --artifact, got nil")
	}

	tmpDir := t.TempDir()
	pt2Path := filepath.Join(tmpDir, "decode.pt2")

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)
	wFormat, _ := zw.Create("decode/archive_format")
	wFormat.Write([]byte("pt2"))
	wVer, _ := zw.Create("decode/archive_version")
	wVer.Write([]byte("0"))
	wModel, _ := zw.Create("decode/models/model.json")
	wModel.Write([]byte(`{
		"schema_version": {"major": 8, "minor": 20},
		"graph_module": {
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.w,m.w,L"}},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.e,m.e,MoE"}}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	// Test output to JSON
	outJSON := filepath.Join(tmpDir, "segments.json")
	if err := cmdPlan(ctx, []string{"segments", "--artifact", tmpDir, "--out", outJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan segments --json failed: %v", err)
	}
	jsonBytes, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read segments.json failed: %v", err)
	}
	var segs []*planpb.Segment
	if err := json.Unmarshal(jsonBytes, &segs); err != nil {
		t.Fatalf("unmarshal segments.json failed: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("got %d segments, want 2", len(segs))
	}
	if segs[0].Name != "submod_0" || segs[0].Host {
		t.Errorf("segs[0] = %+v, want submod_0 (dense)", segs[0])
	}
	if segs[1].Name != "submod_1" || !segs[1].Host {
		t.Errorf("segs[1] = %+v, want submod_1 (host)", segs[1])
	}

	// Test output to text
	outTxt := filepath.Join(tmpDir, "segments.txt")
	if err := cmdPlan(ctx, []string{"segments", "--artifact", tmpDir, "--out", outTxt}); err != nil {
		t.Fatalf("cmdPlan segments text failed: %v", err)
	}
	txtBytes, err := os.ReadFile(outTxt)
	if err != nil {
		t.Fatalf("read segments.txt failed: %v", err)
	}
	txt := string(txtBytes)
	if !strings.Contains(txt, "submod_0 (dense)") || !strings.Contains(txt, "submod_1 (host)") {
		t.Errorf("segments text output missing expected headers: %s", txt)
	}

	// Test --host-ops output
	outHostOps := filepath.Join(tmpDir, "host_ops.json")
	if err := cmdPlan(ctx, []string{"segments", "--host-ops", "--out", outHostOps, "--json"}); err != nil {
		t.Fatalf("cmdPlan segments --host-ops failed: %v", err)
	}
	hostOpsBytes, err := os.ReadFile(outHostOps)
	if err != nil {
		t.Fatalf("read host_ops.json failed: %v", err)
	}
	var hostOps []string
	if err := json.Unmarshal(hostOpsBytes, &hostOps); err != nil {
		t.Fatalf("unmarshal host_ops.json failed: %v", err)
	}
	if len(hostOps) != 2 || hostOps[0] != "vxpu::engram_rows" || hostOps[1] != "vxpu::moe_experts" {
		t.Errorf("hostOps = %v, want [vxpu::engram_rows, vxpu::moe_experts]", hostOps)
	}
}

func verifyTreeHTML(t *testing.T, html string, modelName string, segs []*planpb.Segment) {
	t.Helper()
	if strings.Contains(html, "http://") || strings.Contains(html, "https://") {
		t.Errorf("HTML page references external resource (contains http:// or https://)")
	}

	if !strings.Contains(html, modelName) {
		t.Errorf("HTML page does not contain model name %q", modelName)
	}

	totalCallSites := 0
	for _, s := range segs {
		totalCallSites += len(s.CallSites)
	}

	segCountStr := fmt.Sprintf("%d</span> Segments", len(segs))
	if !strings.Contains(html, segCountStr) {
		t.Errorf("HTML page does not state segment count (%d)", len(segs))
	}
	callSiteCountStr := fmt.Sprintf("%d</span> CallSites", totalCallSites)
	if !strings.Contains(html, callSiteCountStr) {
		t.Errorf("HTML page does not state call site count (%d)", totalCallSites)
	}

	lastIdx := -1
	for _, s := range segs {
		idx := strings.Index(html[lastIdx+1:], s.Name)
		if idx == -1 {
			t.Fatalf("segment %s missing or out of order in HTML", s.Name)
		}
		absIdx := lastIdx + 1 + idx

		sumHeader := fmt.Sprintf(`class="segment-name">%s</span>`, s.Name)
		if strings.Count(html, sumHeader) != 1 {
			t.Errorf("segment %s header appears %d times, want 1", s.Name, strings.Count(html, sumHeader))
		}

		if s.Host {
			hostBadge := `<span class="badge badge-host">Host Segment</span>`
			segStart := strings.Index(html[absIdx:], "<div class=\"callsites-box\">")
			if segStart == -1 {
				segStart = len(html) - absIdx
			}
			segHeader := html[absIdx : absIdx+segStart]
			if !strings.Contains(segHeader, hostBadge) {
				t.Errorf("segment %s is host op but not marked as host in header: %s", s.Name, segHeader)
			}
		} else {
			segBadge := `<span class="badge badge-segment">Segment</span>`
			segStart := strings.Index(html[absIdx:], "<div class=\"callsites-box\">")
			if segStart == -1 {
				segStart = len(html) - absIdx
			}
			segHeader := html[absIdx : absIdx+segStart]
			if !strings.Contains(segHeader, segBadge) {
				t.Errorf("segment %s is ordinary segment but missing badge in header: %s", s.Name, segHeader)
			}
		}

		lastSiteIdx := absIdx
		for _, cs := range s.CallSites {
			opIdx := strings.Index(html[lastSiteIdx:], cs.Op)
			if opIdx == -1 {
				t.Errorf("call site op %s for segment %s not found in order", cs.Op, s.Name)
				continue
			}
			afterOpIdx := lastSiteIdx + opIdx + len(cs.Op)
			nodeIdx := strings.Index(html[afterOpIdx:], cs.Node)
			if nodeIdx == -1 {
				t.Errorf("call site node %s for segment %s not found in order", cs.Node, s.Name)
				continue
			}
			nodeRow := html[afterOpIdx : afterOpIdx+nodeIdx+len(cs.Node)+10]
			if !strings.Contains(nodeRow, "node") {
				t.Errorf("call site node %s row does not label node: %s", cs.Node, nodeRow)
			}
			afterNodeIdx := afterOpIdx + nodeIdx + len(cs.Node)
			if cs.Site != "" {
				siteIdx := strings.Index(html[afterNodeIdx:], cs.Site)
				if siteIdx == -1 {
					t.Errorf("call site site %s for segment %s not found in order", cs.Site, s.Name)
				} else {
					siteRow := html[afterNodeIdx : afterNodeIdx+siteIdx+len(cs.Site)+10]
					if !strings.Contains(siteRow, "site") {
						t.Errorf("call site site %s row does not label site: %s", cs.Site, siteRow)
					}
				}
			}
			lastSiteIdx = afterNodeIdx
		}

		lastIdx = absIdx
	}
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
	if dir := os.Getenv("TINY_DEEPSEEK_ARTIFACT"); dir != "" {
		if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err == nil {
			return dir
		}
	}
	pyBin := "python3"
	if root := findGitRoot(); root != "" {
		venvPy := filepath.Join(root, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			pyBin = venvPy
		}
	}
	cmd := exec.Command(pyBin, "-c", "import sys; from vxpu.fixtures import tiny_deepseek_v41; sys.stdout.write(tiny_deepseek_v41.get_artifact_dir(max_cache_len=64))")
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

func TestCmdPlanTree_Synthetic(t *testing.T) {
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"tree"}); err == nil {
		t.Error("expected error for missing --artifact, got nil")
	}

	tmpDir := t.TempDir()
	pt2Path := filepath.Join(tmpDir, "decode.pt2")

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)
	wFormat, _ := zw.Create("decode/archive_format")
	wFormat.Write([]byte("pt2"))
	wVer, _ := zw.Create("decode/archive_version")
	wVer.Write([]byte("0"))
	wModel, _ := zw.Create("decode/models/model.json")
	wModel.Write([]byte(`{
		"schema_version": {"major": 8, "minor": 20},
		"graph_module": {
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.w,m.w,L"}},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.e,m.e,MoE"}}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	segs, err := kernelplan.ReadSegments(tmpDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	// Test output to HTML
	outHTML := filepath.Join(tmpDir, "tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--out", outHTML, "--model", "test-model"}); err != nil {
		t.Fatalf("cmdPlan tree HTML failed: %v", err)
	}
	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("read tree.html failed: %v", err)
	}
	html := string(htmlBytes)
	verifyTreeHTML(t, html, "test-model", segs)

	// Test output to JSON
	outJSON := filepath.Join(tmpDir, "tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--out", outJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree JSON failed: %v", err)
	}
	jsonBytes, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read tree.json failed: %v", err)
	}
	var tree planpb.ModelTree
	if err := protojson.Unmarshal(jsonBytes, &tree); err != nil {
		t.Fatalf("unmarshal tree.json failed: %v", err)
	}
	if tree.Model != filepath.Base(tmpDir) {
		t.Errorf("tree.Model = %q, want %q", tree.Model, filepath.Base(tmpDir))
	}
	if len(tree.Segments) != 2 {
		t.Fatalf("got %d segments in tree, want 2", len(tree.Segments))
	}
}

func verifyPlanTreeHTML(t *testing.T, html string, segs []*planpb.Segment, plan *planpb.PlanFile, uncoveredNode string) {
	t.Helper()
	if strings.Contains(html, "http://") || strings.Contains(html, "https://") {
		t.Errorf("HTML page references external resource (contains http:// or https://)")
	}
	if !strings.Contains(html, "EXPLAIN") {
		t.Errorf("HTML page missing EXPLAIN label in header")
	}

	planBodies := make(map[string]string)
	for _, s := range plan.Sites {
		if s.Body != "" {
			planBodies[s.Id] = s.Body
		}
	}

	var foundUncovered bool
	for _, s := range segs {
		for _, cs := range s.CallSites {
			siteAnchor := fmt.Sprintf(`id="site-%s"`, cs.Node)
			nodeIdx := strings.Index(html, siteAnchor)
			if nodeIdx == -1 {
				t.Errorf("call site %s not found in HTML", cs.Node)
				continue
			}
			endLi := strings.Index(html[nodeIdx:], "</li>")
			if endLi == -1 {
				endLi = len(html) - nodeIdx
			}
			rowHTML := html[nodeIdx : nodeIdx+endLi]

			if cs.Node == uncoveredNode {
				foundUncovered = true
				if !strings.Contains(rowHTML, "not covered in plan") {
					t.Errorf("uncovered node %s does not display 'not covered in plan', got row: %s", cs.Node, rowHTML)
				}
				continue
			}

			if wantBody, ok := planBodies[cs.Node]; ok {
				bodyLabel := fmt.Sprintf("body</span>%s", wantBody)
				if !strings.Contains(rowHTML, bodyLabel) {
					t.Errorf("covered node %s does not carry body %q, got row: %s", cs.Node, wantBody, rowHTML)
				}
			}
		}
	}

	if uncoveredNode != "" && !foundUncovered {
		t.Errorf("uncovered node %s was not found in graph call sites", uncoveredNode)
	}

	if !strings.Contains(html, "Configuration Knobs [plan]") {
		t.Errorf("HTML page missing Configuration Knobs [plan] section")
	}
	if !strings.Contains(html, "[plan knob]") {
		t.Errorf("HTML page knobs not labelled with [plan knob]")
	}
	if !strings.Contains(html, "Unmatched Plan Sites") {
		t.Errorf("HTML page missing Unmatched Plan Sites section")
	}
	if !strings.Contains(html, "<td>default</td>") {
		t.Errorf("HTML page missing default strategy in unmatched plan sites table")
	}
}

func verifyRecordTreeHTML(t *testing.T, html string, segs []*planpb.Segment, record *planpb.PlanFile, expectedTotalUs float64, expectedSegTotals map[string]float64) {
	t.Helper()
	if strings.Contains(html, "http://") || strings.Contains(html, "https://") {
		t.Errorf("HTML page references external resource (contains http:// or https://)")
	}
	if !strings.Contains(html, "EXPLAIN ANALYZE") {
		t.Errorf("HTML page missing EXPLAIN ANALYZE label in header")
	}

	for _, s := range record.Sites {
		if s.Stats != nil && s.Stats.MeasuredUs != nil {
			siteAnchor := fmt.Sprintf(`id="site-%s"`, s.Id)
			nodeIdx := strings.Index(html, siteAnchor)
			if nodeIdx == -1 {
				continue
			}
			endLi := strings.Index(html[nodeIdx:], "</li>")
			if endLi == -1 {
				endLi = len(html) - nodeIdx
			}
			rowHTML := html[nodeIdx : nodeIdx+endLi]
			measStr := fmt.Sprintf("%.1f µs [measured]", s.Stats.GetMeasuredUs())
			if !strings.Contains(rowHTML, measStr) {
				t.Errorf("site %s missing measured figure %q in row: %s", s.Id, measStr, rowHTML)
			}
		}
	}

	for segName, segTotal := range expectedSegTotals {
		segAnchor := fmt.Sprintf(`id="seg-%s"`, segName)
		segIdx := strings.Index(html, segAnchor)
		if segIdx == -1 {
			t.Errorf("segment %s not found in HTML", segName)
			continue
		}
		endSummary := strings.Index(html[segIdx:], "</summary>")
		if endSummary == -1 {
			endSummary = len(html) - segIdx
		}
		summaryHTML := html[segIdx : segIdx+endSummary]
		segTotalStr := fmt.Sprintf("%.1f µs [measured]", segTotal)
		if !strings.Contains(summaryHTML, segTotalStr) {
			t.Errorf("segment %s missing total measured time %q in summary: %s", segName, segTotalStr, summaryHTML)
		}
	}

	totalUsStr := fmt.Sprintf("%.1f µs</span> Total Site Measured Time [measured]", expectedTotalUs)
	if !strings.Contains(html, totalUsStr) {
		t.Errorf("HTML page does not state total site measured time matching record (%s)", totalUsStr)
	}

	if !strings.Contains(html, "Slowest Call Sites") {
		t.Errorf("HTML page missing Slowest Call Sites section")
	}
	ruleStr := "The up to five call sites with the highest measured execution time, ordered descending by measured time (breaking ties by node identifier)."
	if !strings.Contains(html, ruleStr) {
		t.Errorf("HTML page missing slowest call sites selection rule %q", ruleStr)
	}

	if record.Stats != nil && record.Stats.MedianStepMs != nil {
		medianStr := fmt.Sprintf("%.2f ms</span> Median Step [measured]", record.Stats.GetMedianStepMs())
		if !strings.Contains(html, medianStr) {
			t.Errorf("HTML page does not state median step matching record (%s)", medianStr)
		}
	}

	if record.Stats != nil && record.Stats.TokensAStreamAStep != nil {
		tokStr := fmt.Sprintf("%.2f tokens/stream/step [measured]", record.Stats.GetTokensAStreamAStep())
		if !strings.Contains(html, tokStr) {
			t.Errorf("HTML page does not state tokens a stream a step matching record (%s)", tokStr)
		}
	}

	if record.Stats != nil && record.Stats.AcceptedDraftsAStep != nil {
		draftStr := fmt.Sprintf("%.2f tokens/step [measured]", record.Stats.GetAcceptedDraftsAStep())
		if !strings.Contains(html, draftStr) {
			t.Errorf("HTML page does not state accepted drafts matching record (%s)", draftStr)
		}
	}

	if record.Stats != nil && len(record.Stats.PerCardPeakMemory) > 0 {
		memStr := fmt.Sprintf("%d bytes) [measured]", record.Stats.PerCardPeakMemory[0])
		if !strings.Contains(html, memStr) {
			t.Errorf("HTML page does not state peak memory matching record (%s)", memStr)
		}
	}

	if record.Stats != nil && len(record.Stats.Checks) > 0 {
		for chk, pass := range record.Stats.Checks {
			wantChk := fmt.Sprintf("%s:</span> <span class=\"badge %s\">%s</span>", chk, map[bool]string{true: "badge-pass", false: "badge-fail"}[pass], map[bool]string{true: "PASS", false: "FAIL"}[pass])
			if !strings.Contains(html, wantChk) {
				t.Errorf("HTML page missing check result %q", wantChk)
			}
		}
	}
}

func testChecksumRefusal(t *testing.T, ctx context.Context, artifactDir string, plan *planpb.PlanFile, tmpDir string) {
	t.Helper()
	badPlan := proto.Clone(plan).(*planpb.PlanFile)
	badPlan.ArtifactMd5 = "deadbeef00000000deadbeef00000000"
	badPath := filepath.Join(tmpDir, "bad_checksum_plan.json")
	badData, err := kernelplan.MarshalAsJSON(badPlan)
	if err != nil {
		t.Fatalf("MarshalAsJSON bad plan: %v", err)
	}
	if err := os.WriteFile(badPath, badData, 0644); err != nil {
		t.Fatalf("write bad plan: %v", err)
	}

	badOut := filepath.Join(tmpDir, "bad.html")
	err = cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", badPath, "--out", badOut})
	if err == nil {
		t.Fatalf("expected error for mismatched artifact checksum, got nil")
	}
	if !strings.Contains(err.Error(), "checksum") || !strings.Contains(err.Error(), "deadbeef") {
		t.Errorf("error %q does not state checksum mismatch reason", err.Error())
	}
	if _, statErr := os.Stat(badOut); statErr == nil {
		t.Errorf("output file %s was created despite checksum mismatch", badOut)
	}
}

func TestCmdPlanTree_TinyDeepSeek(t *testing.T) {
	ctx := t.Context()
	artifactDir := getTinyDeepSeekArtifactDir(t)

	segs, err := kernelplan.ReadSegments(artifactDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	tmpDir := t.TempDir()

	// 1. Structure only
	outHTML := filepath.Join(tmpDir, "tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--out", outHTML}); err != nil {
		t.Fatalf("cmdPlan tree HTML failed: %v", err)
	}

	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("read tree.html failed: %v", err)
	}
	html := string(htmlBytes)
	verifyTreeHTML(t, html, filepath.Base(artifactDir), segs)

	outJSON := filepath.Join(tmpDir, "tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--out", outJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree JSON failed: %v", err)
	}

	jsonBytes, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read tree.json failed: %v", err)
	}
	var tree planpb.ModelTree
	if err := protojson.Unmarshal(jsonBytes, &tree); err != nil {
		t.Fatalf("protojson.Unmarshal tree.json failed: %v", err)
	}
	if len(tree.Segments) != len(segs) {
		t.Errorf("got %d segments in tree, want %d", len(tree.Segments), len(segs))
	}
	if tree.HasPlan {
		t.Errorf("tree.HasPlan = true, want false")
	}
	if tree.HasMeasurements {
		t.Errorf("tree.HasMeasurements = true, want false")
	}
	if tree.TotalCallSites == 0 {
		t.Errorf("tree.TotalCallSites = 0, want > 0")
	}

	// 2. Emit plan with vxpu plan emit
	fixtureDir := filepath.Join(findGitRoot(), "vxpu", "pkg", "kernelplan", "testdata")
	planPath := filepath.Join(tmpDir, "emitted_plan.json")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
		"--sites", filepath.Join(fixtureDir, "kernel_record.json"),
		"--catalog", filepath.Join(fixtureDir, "kernel_catalog.json"),
		"--table", filepath.Join(fixtureDir, "kernel_table.json"),
		"--out", planPath,
		"--json",
	}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	emittedPlan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("read emitted plan: %v", err)
	}
	if len(emittedPlan.Sites) == 0 {
		t.Fatalf("emitted plan has 0 sites")
	}

	// Create partial plan to test uncovered site display
	graphCallSiteNodes := make(map[string]bool)
	for _, seg := range segs {
		for _, cs := range seg.CallSites {
			graphCallSiteNodes[cs.Node] = true
		}
	}

	partialPlan := proto.Clone(emittedPlan).(*planpb.PlanFile)
	var omittedSiteID string
	var remainingSites []*planpb.PlanSite
	for _, s := range partialPlan.Sites {
		if omittedSiteID == "" && graphCallSiteNodes[s.Id] {
			omittedSiteID = s.Id
			continue
		}
		remainingSites = append(remainingSites, s)
	}
	if omittedSiteID == "" {
		t.Fatalf("could not find a graph call site to omit from plan")
	}
	partialPlan.Sites = remainingSites
	partialPlanPath := filepath.Join(tmpDir, "partial_plan.json")
	partialData, err := kernelplan.MarshalAsJSON(partialPlan)
	if err != nil {
		t.Fatalf("marshal partial plan: %v", err)
	}
	if err := os.WriteFile(partialPlanPath, partialData, 0644); err != nil {
		t.Fatalf("write partial plan: %v", err)
	}

	outPlanHTML := filepath.Join(tmpDir, "plan_tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", partialPlanPath, "--out", outPlanHTML}); err != nil {
		t.Fatalf("cmdPlan tree with plan failed: %v", err)
	}
	planHTMLBytes, err := os.ReadFile(outPlanHTML)
	if err != nil {
		t.Fatalf("read plan_tree.html failed: %v", err)
	}
	planHTML := string(planHTMLBytes)
	verifyPlanTreeHTML(t, planHTML, segs, partialPlan, omittedSiteID)

	outPlanJSON := filepath.Join(tmpDir, "plan_tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", partialPlanPath, "--out", outPlanJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree with plan JSON failed: %v", err)
	}
	planJSONBytes, err := os.ReadFile(outPlanJSON)
	if err != nil {
		t.Fatalf("read plan_tree.json failed: %v", err)
	}
	var planTree planpb.ModelTree
	if err := protojson.Unmarshal(planJSONBytes, &planTree); err != nil {
		t.Fatalf("protojson.Unmarshal plan_tree.json failed: %v", err)
	}
	if planTree.ArtifactMd5 != partialPlan.ArtifactMd5 {
		t.Errorf("planTree.ArtifactMd5 = %v, want %v", planTree.ArtifactMd5, partialPlan.ArtifactMd5)
	}
	if len(planTree.UnmatchedPlanSites) == 0 {
		t.Errorf("planTree has no unmatched plan sites")
	}
	if !planTree.HasPlan {
		t.Errorf("planTree.HasPlan = false, want true")
	}
	if planTree.HasMeasurements {
		t.Errorf("planTree.HasMeasurements = true, want false")
	}
	if planTree.TotalCallSites == 0 {
		t.Errorf("planTree.TotalCallSites = 0, want > 0")
	}

	// 3. Run record with measurements (EXPLAIN ANALYZE)
	runRecord := proto.Clone(emittedPlan).(*planpb.PlanFile)
	runRecord.Stats = &planpb.PlanExecutionStats{
		PlanHash:            proto.String("hash789"),
		MedianStepMs:        proto.Float64(15.5),
		TokensAStreamAStep:  proto.Float64(1.0),
		AcceptedDraftsAStep: proto.Float64(0.75),
		PerCardPeakMemory:   []int64{524288000},
		PerClassDeviceTimes: map[string]float64{"dense": 12.3},
		Checks:              map[string]bool{"bounds": false, "exactness": true},
		Knobs: &planpb.Knobs{
			GEMV_ROWS_MAX: proto.Int32(32),
			FP4_NATIVE:    proto.Bool(true),
		},
	}
	var expectedTotalUs float64
	expectedSegTotals := make(map[string]float64)
	siteIdx := 0
	sitesMap := make(map[string]*planpb.PlanSite, len(runRecord.Sites))
	for _, s := range runRecord.Sites {
		sitesMap[s.Id] = s
	}
	var measuredCount int32
	var differingSiteID string
	for _, seg := range segs {
		for _, cs := range seg.CallSites {
			if s, ok := sitesMap[cs.Node]; ok {
				us := float64((siteIdx%10)+1)*10.0 + 2.5
				s.Stats = &planpb.SiteExecutionStats{
					MeasuredUs: proto.Float64(us),
				}
				if differingSiteID == "" && s.Body != "" {
					differingSiteID = s.Id
					s.Body = "gpu-fp8-gemv"
				}
				expectedTotalUs += us
				expectedSegTotals[seg.Name] += us
				measuredCount++
				siteIdx++
			}
		}
	}
	recordPath := filepath.Join(tmpDir, "run_record.json")
	recordData, err := kernelplan.MarshalAsJSON(runRecord)
	if err != nil {
		t.Fatalf("marshal run record: %v", err)
	}
	if err := os.WriteFile(recordPath, recordData, 0644); err != nil {
		t.Fatalf("write run record: %v", err)
	}

	outRecordHTML := filepath.Join(tmpDir, "record_tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", planPath, "--record", recordPath, "--out", outRecordHTML}); err != nil {
		t.Fatalf("cmdPlan tree with record failed: %v", err)
	}
	recordHTMLBytes, err := os.ReadFile(outRecordHTML)
	if err != nil {
		t.Fatalf("read record_tree.html failed: %v", err)
	}
	recordHTML := string(recordHTMLBytes)
	verifyRecordTreeHTML(t, recordHTML, segs, runRecord, expectedTotalUs, expectedSegTotals)

	if !strings.Contains(recordHTML, "1</span> Differing Call Sites") {
		t.Errorf("recordHTML missing Differing Call Sites count in header")
	}
	if !strings.Contains(recordHTML, `<span class="badge badge-diff"><span class="badge-label">run</span>gpu-fp8-gemv</span>`) {
		t.Errorf("recordHTML missing differing run body badge")
	}

	outRecordJSON := filepath.Join(tmpDir, "record_tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", planPath, "--record", recordPath, "--out", outRecordJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree with record JSON failed: %v", err)
	}
	recordJSONBytes, err := os.ReadFile(outRecordJSON)
	if err != nil {
		t.Fatalf("read record_tree.json failed: %v", err)
	}
	var recordTree planpb.ModelTree
	if err := protojson.Unmarshal(recordJSONBytes, &recordTree); err != nil {
		t.Fatalf("protojson.Unmarshal record_tree.json failed: %v", err)
	}
	if recordTree.Stats == nil || recordTree.Stats.GetMedianStepMs() != 15.5 {
		t.Errorf("recordTree.Stats.MedianStepMs = %v, want 15.5", recordTree.GetStats().GetMedianStepMs())
	}
	if recordTree.GetTotalMeasuredTimeUs() != expectedTotalUs {
		t.Errorf("recordTree.TotalMeasuredTimeUs = %v, want %v", recordTree.GetTotalMeasuredTimeUs(), expectedTotalUs)
	}
	if recordTree.MeasuredCallSites != measuredCount {
		t.Errorf("recordTree.MeasuredCallSites = %d, want %d", recordTree.MeasuredCallSites, measuredCount)
	}
	if recordTree.DifferingCallSites != 1 {
		t.Errorf("recordTree.DifferingCallSites = %d, want 1", recordTree.DifferingCallSites)
	}
	if !recordTree.HasPlan {
		t.Errorf("recordTree.HasPlan = false, want true")
	}
	if !recordTree.HasMeasurements {
		t.Errorf("recordTree.HasMeasurements = false, want true")
	}
	if recordTree.TotalCallSites == 0 {
		t.Errorf("recordTree.TotalCallSites = 0, want > 0")
	}
	var foundBoth bool
	for _, seg := range recordTree.Segments {
		for _, cs := range seg.CallSites {
			if cs.PlanSite != nil && cs.Stats != nil && cs.Stats.MeasuredUs != nil {
				foundBoth = true
				if cs.PlanSite.Body == "" {
					t.Errorf("expected plan site body to be populated on cs %s", cs.Node)
				}
			}
		}
	}
	if !foundBoth {
		t.Errorf("expected to find call sites with both plan choices and measurements")
	}
	if len(recordTree.SlowestCallSites) != 5 {
		t.Fatalf("got %d slowest call sites, want 5", len(recordTree.SlowestCallSites))
	}
	for i, sc := range recordTree.SlowestCallSites {
		if sc.Rank != int32(i+1) {
			t.Errorf("slowest call site [%d] rank = %d, want %d", i, sc.Rank, i+1)
		}
		wantPct := (sc.MeasuredTimeUs / expectedTotalUs) * 100.0
		if fmt.Sprintf("%.1f", sc.Percent) != fmt.Sprintf("%.1f", wantPct) {
			t.Errorf("slowest call site [%d] percent = %.1f, want %.1f", i, sc.Percent, wantPct)
		}
	}
	for _, seg := range recordTree.Segments {
		wantSegTotal := expectedSegTotals[seg.Name]
		if wantSegTotal > 0 {
			if seg.TotalMeasuredTimeUs == nil {
				t.Errorf("segment %s missing TotalMeasuredTimeUs in JSON", seg.Name)
			} else if fmt.Sprintf("%.1f", seg.GetTotalMeasuredTimeUs()) != fmt.Sprintf("%.1f", wantSegTotal) {
				t.Errorf("segment %s TotalMeasuredTimeUs = %v, want %v", seg.Name, seg.GetTotalMeasuredTimeUs(), wantSegTotal)
			}
		}
	}

	// 4. Test record with per-site measurements and no top-level stats (must not panic with nil pointer)
	siteOnlyRecord := proto.Clone(runRecord).(*planpb.PlanFile)
	siteOnlyRecord.Stats = nil
	siteOnlyPath := filepath.Join(tmpDir, "site_only_record.json")
	siteOnlyData, err := kernelplan.MarshalAsJSON(siteOnlyRecord)
	if err != nil {
		t.Fatalf("marshal site-only record: %v", err)
	}
	if err := os.WriteFile(siteOnlyPath, siteOnlyData, 0644); err != nil {
		t.Fatalf("write site-only record: %v", err)
	}
	outSiteOnlyHTML := filepath.Join(tmpDir, "site_only_tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--record", siteOnlyPath, "--out", outSiteOnlyHTML}); err != nil {
		t.Fatalf("cmdPlan tree with site-only record failed: %v", err)
	}
	siteOnlyHTMLBytes, err := os.ReadFile(outSiteOnlyHTML)
	if err != nil {
		t.Fatalf("read site_only_tree.html failed: %v", err)
	}
	siteOnlyHTML := string(siteOnlyHTMLBytes)
	if !strings.Contains(siteOnlyHTML, "MODEL MEASUREMENTS (NO PLAN)") {
		t.Errorf("site-only record tree HTML missing MODEL MEASUREMENTS (NO PLAN)")
	}
	if !strings.Contains(siteOnlyHTML, "(MEASUREMENTS, NO PLAN)</title>") {
		t.Errorf("site-only record tree HTML missing (MEASUREMENTS, NO PLAN) in title")
	}
	if !strings.Contains(siteOnlyHTML, "Run Measurement Details") {
		t.Errorf("site-only record tree HTML missing Run Measurement Details")
	}
	if !strings.Contains(siteOnlyHTML, "[measured]") {
		t.Errorf("site-only record tree HTML missing [measured]")
	}

	// 5. Test record whose sites do not match the graph (unmatched measurements shown with count, not dropped)
	unmatchedRecord := proto.Clone(emittedPlan).(*planpb.PlanFile)
	unmatchedRecord.Sites = []*planpb.PlanSite{
		{
			Id:    "unmatched_node_alpha",
			Kind:  planpb.SiteKind_SITE_KIND_DENSE_GEMV,
			Body:  "ref_alpha",
			Stats: &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(77.5)},
		},
		{
			Id:    "unmatched_node_beta",
			Kind:  planpb.SiteKind_SITE_KIND_ATTENTION,
			Body:  "ref_beta",
			Stats: &planpb.SiteExecutionStats{MeasuredUs: proto.Float64(88.5)},
		},
	}
	unmatchedRecordPath := filepath.Join(tmpDir, "unmatched_record.json")
	unmatchedRecordData, err := kernelplan.MarshalAsJSON(unmatchedRecord)
	if err != nil {
		t.Fatalf("marshal unmatched record: %v", err)
	}
	if err := os.WriteFile(unmatchedRecordPath, unmatchedRecordData, 0644); err != nil {
		t.Fatalf("write unmatched record: %v", err)
	}

	outUnmatchedHTML := filepath.Join(tmpDir, "unmatched_tree.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--record", unmatchedRecordPath, "--out", outUnmatchedHTML}); err != nil {
		t.Fatalf("cmdPlan tree with unmatched record failed: %v", err)
	}
	unmatchedHTMLBytes, err := os.ReadFile(outUnmatchedHTML)
	if err != nil {
		t.Fatalf("read unmatched_tree.html failed: %v", err)
	}
	unmatchedHTML := string(unmatchedHTMLBytes)
	if !strings.Contains(unmatchedHTML, "2</span> Unmatched Measurements") {
		t.Errorf("unmatched HTML page missing count of unmatched measurements (2)")
	}
	if strings.Contains(unmatchedHTML, "0.0 µs") || strings.Contains(unmatchedHTML, "Total Site Measured Time") {
		t.Errorf("unmatched HTML page should not show 0.0 µs total measured time when no call site was measured")
	}
	if strings.Contains(unmatchedHTML, "Unmatched Plan Sites") {
		t.Errorf("unmatched HTML page should not show Unmatched Plan Sites when no plan was provided")
	}
	if !strings.Contains(unmatchedHTML, "unmatched_node_alpha") || !strings.Contains(unmatchedHTML, "77.5 µs [measured]") {
		t.Errorf("unmatched HTML page missing unmatched measurement for unmatched_node_alpha")
	}
	if !strings.Contains(unmatchedHTML, "unmatched_node_beta") || !strings.Contains(unmatchedHTML, "88.5 µs [measured]") {
		t.Errorf("unmatched HTML page missing unmatched measurement for unmatched_node_beta")
	}

	outUnmatchedJSON := filepath.Join(tmpDir, "unmatched_tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--record", unmatchedRecordPath, "--out", outUnmatchedJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree with unmatched record JSON failed: %v", err)
	}
	unmatchedJSONBytes, err := os.ReadFile(outUnmatchedJSON)
	if err != nil {
		t.Fatalf("read unmatched_tree.json failed: %v", err)
	}
	var unmatchedTree planpb.ModelTree
	if err := protojson.Unmarshal(unmatchedJSONBytes, &unmatchedTree); err != nil {
		t.Fatalf("protojson.Unmarshal unmatched_tree.json failed: %v", err)
	}
	if unmatchedTree.TotalMeasuredTimeUs != nil {
		t.Errorf("unmatchedTree.TotalMeasuredTimeUs = %v, want nil (unset)", *unmatchedTree.TotalMeasuredTimeUs)
	}
	if len(unmatchedTree.UnmatchedMeasurements) != 2 {
		t.Errorf("got %d unmatched measurements in tree, want 2", len(unmatchedTree.UnmatchedMeasurements))
	}
	if len(unmatchedTree.UnmatchedPlanSites) != 0 {
		t.Errorf("got %d unmatched plan sites in tree, want 0", len(unmatchedTree.UnmatchedPlanSites))
	}
	if unmatchedTree.MeasuredCallSites != 0 {
		t.Errorf("unmatchedTree.MeasuredCallSites = %d, want 0", unmatchedTree.MeasuredCallSites)
	}
	if !unmatchedTree.HasMeasurements {
		t.Errorf("unmatchedTree.HasMeasurements = false, want true")
	}
	if unmatchedTree.HasPlan {
		t.Errorf("unmatchedTree.HasPlan = true, want false")
	}

	// 6. Checksum refusal test
	testChecksumRefusal(t, ctx, artifactDir, emittedPlan, tmpDir)

	// 7. Upload CI artifacts
	if artifactsDir := os.Getenv("ARTIFACTS"); artifactsDir != "" {
		if err := os.MkdirAll(artifactsDir, 0755); err != nil {
			t.Fatalf("create ARTIFACTS directory failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tiny-deepseek-tree.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write tiny-deepseek-tree.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tree.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write tree.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tiny-deepseek-plan-tree.html"), planHTMLBytes, 0644); err != nil {
			t.Fatalf("write tiny-deepseek-plan-tree.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tiny-deepseek-record-tree.html"), recordHTMLBytes, 0644); err != nil {
			t.Fatalf("write tiny-deepseek-record-tree.html failed: %v", err)
		}
	}
}

func verifyGraphHTML(t *testing.T, html string, plan *planpb.StepPlan) {
	t.Helper()
	if strings.Contains(html, "http://") || strings.Contains(html, "https://") {
		t.Errorf("HTML page references external resource (contains http:// or https://)")
	}

	// 1. Every node appears once
	for _, n := range plan.Nodes {
		nodeID := fmt.Sprintf(`id="node-%s"`, n.Name)
		count := strings.Count(html, nodeID)
		if count != 1 {
			t.Errorf("node %s appears %d times in HTML, want exactly 1", n.Name, count)
		}
	}

	// 2. Each node is under its stream
	for _, n := range plan.Nodes {
		streamName, err := stepplan.FormatStream(n.Stream)
		if err != nil {
			streamName = "unspecified"
		}
		streamContainer := fmt.Sprintf(`id="stream-%s"`, streamName)
		streamIdx := strings.Index(html, streamContainer)
		nodeAttr := fmt.Sprintf(`id="node-%s"`, n.Name)
		nodeIdx := strings.Index(html, nodeAttr)
		if streamIdx == -1 {
			t.Errorf("stream container %s not found in HTML", streamContainer)
		} else if nodeIdx == -1 {
			t.Errorf("node %s not found in HTML", n.Name)
		} else if nodeIdx < streamIdx {
			t.Errorf("node %s appears before stream container %s", n.Name, streamName)
		}
	}

	// 3. Every wait drawn as an edge
	for _, n := range plan.Nodes {
		for _, w := range n.Waits {
			edgeAttr := fmt.Sprintf(`data-from="%s" data-to="%s"`, w, n.Name)
			if !strings.Contains(html, edgeAttr) {
				t.Errorf("wait dependency edge %s -> %s not found in HTML", w, n.Name)
			}
		}
	}

	// 4. Header states node count
	nodeCountStr := fmt.Sprintf(`%d</span> Nodes`, len(plan.Nodes))
	if !strings.Contains(html, nodeCountStr) {
		t.Errorf("HTML page does not state node count (%d)", len(plan.Nodes))
	}
}

func TestCmdPlanGraph_TinyDeepSeek(t *testing.T) {
	ctx := t.Context()
	root := findGitRoot()
	if root == "" {
		t.Fatalf("findGitRoot failed")
	}

	fixturePath := filepath.Join(root, "vxpu", "pkg", "stepplan", "testdata", "stepgraph_chain.json")
	desc, err := stepplan.ReadDescription(fixturePath)
	if err != nil {
		t.Fatalf("ReadDescription failed: %v", err)
	}

	plan, err := stepplan.Plan(desc, stepplan.Options{})
	if err != nil {
		t.Fatalf("stepplan.Plan failed: %v", err)
	}

	tmpDir := t.TempDir()
	planPath := filepath.Join(tmpDir, "stepplan.json")
	planData, err := stepplan.MarshalAsJSON(plan)
	if err != nil {
		t.Fatalf("MarshalAsJSON plan failed: %v", err)
	}
	if err := os.WriteFile(planPath, planData, 0644); err != nil {
		t.Fatalf("write plan file failed: %v", err)
	}

	// 1. Unmeasured step plan HTML (EXPLAIN)
	outHTML := filepath.Join(tmpDir, "graph.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "--out", outHTML}); err != nil {
		t.Fatalf("cmdPlan graph HTML failed: %v", err)
	}
	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("read graph.html failed: %v", err)
	}
	html := string(htmlBytes)
	verifyGraphHTML(t, html, plan)

	// Check that unmeasured plan does not show 0 or 0.000 ms
	if strings.Contains(html, "0.000 ms") || strings.Contains(html, "0.0 ms") {
		t.Errorf("unmeasured HTML page shows 0.000 ms: %s", html)
	}
	if !strings.Contains(html, "not measured") {
		t.Errorf("unmeasured HTML page does not display 'not measured' for unmeasured nodes")
	}
	if !strings.Contains(html, "0</span> Measured Nodes") {
		t.Errorf("unmeasured HTML page does not state 0 Measured Nodes")
	}
	if !strings.Contains(html, "(EXPLAIN)</title>") {
		t.Errorf("unmeasured HTML page title does not state (EXPLAIN)")
	}
	if !strings.Contains(html, "crossings not known") {
		t.Errorf("unmeasured HTML page does not state 'crossings not known' when --graph is missing")
	}
	if strings.Contains(html, "0 Card Crossings") {
		t.Errorf("unmeasured HTML page should not state '0 Card Crossings' when --graph is missing")
	}

	// 2. Unmeasured step plan JSON
	outJSON := filepath.Join(tmpDir, "graph.json")
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "--out", outJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan graph JSON failed: %v", err)
	}
	jsonBytes, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read graph.json failed: %v", err)
	}
	var graph planpb.StepPlanGraph
	if err := protojson.Unmarshal(jsonBytes, &graph); err != nil {
		t.Fatalf("protojson.Unmarshal graph.json failed: %v", err)
	}
	if len(graph.Nodes) != 11 {
		t.Errorf("len(graph.Nodes) = %d, want 11", len(graph.Nodes))
	}
	if len(graph.Layers) != 11 {
		t.Errorf("len(graph.Layers) = %d, want 11", len(graph.Layers))
	}
	if graph.TotalCrossings != nil {
		t.Errorf("graph.TotalCrossings = %v, want nil without --graph", *graph.TotalCrossings)
	}
	if graph.TotalMeasuredTimeMs != nil {
		t.Errorf("graph.TotalMeasuredTimeMs = %v, want nil", *graph.TotalMeasuredTimeMs)
	}
	if graph.MeasuredNodes != 0 {
		t.Errorf("graph.MeasuredNodes = %d, want 0", graph.MeasuredNodes)
	}

	// Verify extra positional arguments are rejected
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "extra.json"}); err == nil {
		t.Errorf("expected error for extra positional argument after --plan")
	}
	if err := cmdPlan(ctx, []string{"graph", planPath, "extra.json"}); err == nil {
		t.Errorf("expected error for multiple positional arguments")
	}

	// 3. Graph with description attached
	outWithDescHTML := filepath.Join(tmpDir, "graph_desc.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "--graph", fixturePath, "--out", outWithDescHTML}); err != nil {
		t.Fatalf("cmdPlan graph with desc failed: %v", err)
	}
	descHTMLBytes, err := os.ReadFile(outWithDescHTML)
	if err != nil {
		t.Fatalf("read graph_desc.html failed: %v", err)
	}
	descHTML := string(descHTMLBytes)
	verifyGraphHTML(t, descHTML, plan)
	if !strings.Contains(descHTML, "dense") || !strings.Contains(descHTML, "experts") {
		t.Errorf("graph with description does not show layer kinds")
	}
	if !strings.Contains(descHTML, "0</span> Card Crossings") {
		t.Errorf("graph with description does not state 0 Card Crossings")
	}

	// 4. Measured step plan (EXPLAIN ANALYZE)
	measuredPlan := proto.Clone(plan).(*planpb.StepPlan)
	measuredPlan.Nodes[0].Stats = &planpb.NodeExecutionStats{
		DeviceTimeMs: proto.Float64(0.078),
		SampleCount:  proto.Int32(10),
	}
	measuredPlan.Nodes[1].Stats = &planpb.NodeExecutionStats{
		DeviceTimeMs: proto.Float64(0.077),
		SampleCount:  proto.Int32(10),
	}
	measuredPlanPath := filepath.Join(tmpDir, "measured_plan.json")
	measuredPlanData, err := stepplan.MarshalAsJSON(measuredPlan)
	if err != nil {
		t.Fatalf("MarshalAsJSON measured plan failed: %v", err)
	}
	if err := os.WriteFile(measuredPlanPath, measuredPlanData, 0644); err != nil {
		t.Fatalf("write measured plan file failed: %v", err)
	}

	measuredHTMLPath := filepath.Join(tmpDir, "measured_graph.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", measuredPlanPath, "--out", measuredHTMLPath}); err != nil {
		t.Fatalf("cmdPlan graph measured failed: %v", err)
	}
	measuredHTMLBytes, err := os.ReadFile(measuredHTMLPath)
	if err != nil {
		t.Fatalf("read measured_graph.html failed: %v", err)
	}
	measuredHTML := string(measuredHTMLBytes)
	verifyGraphHTML(t, measuredHTML, measuredPlan)

	if !strings.Contains(measuredHTML, "(EXPLAIN ANALYZE)</title>") {
		t.Errorf("measured HTML page title does not state (EXPLAIN ANALYZE)")
	}
	if !strings.Contains(measuredHTML, "0.078 ms") || !strings.Contains(measuredHTML, "[measured]") {
		t.Errorf("measured HTML page does not display 0.078 ms [measured]")
	}
	if !strings.Contains(measuredHTML, "2</span> Measured Nodes") {
		t.Errorf("measured HTML page does not state 2 Measured Nodes")
	}

	// 5. Unmatched wait dependency (node references missing producer)
	unmatchedPlan := proto.Clone(plan).(*planpb.StepPlan)
	unmatchedPlan.Nodes[1].Waits = append(unmatchedPlan.Nodes[1].Waits, "ghost_node")
	unmatchedPlanPath := filepath.Join(tmpDir, "unmatched_plan.json")
	unmatchedPlanData, err := stepplan.MarshalAsJSON(unmatchedPlan)
	if err != nil {
		t.Fatalf("MarshalAsJSON unmatched plan failed: %v", err)
	}
	if err := os.WriteFile(unmatchedPlanPath, unmatchedPlanData, 0644); err != nil {
		t.Fatalf("write unmatched plan file failed: %v", err)
	}

	unmatchedHTMLPath := filepath.Join(tmpDir, "unmatched_graph.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", unmatchedPlanPath, "--out", unmatchedHTMLPath}); err != nil {
		t.Fatalf("cmdPlan graph unmatched failed: %v", err)
	}
	unmatchedHTMLBytes, err := os.ReadFile(unmatchedHTMLPath)
	if err != nil {
		t.Fatalf("read unmatched_graph.html failed: %v", err)
	}
	unmatchedHTML := string(unmatchedHTMLBytes)
	if !strings.Contains(unmatchedHTML, "1</span> Unmatched Waits") {
		t.Errorf("unmatched HTML does not state 1 Unmatched Waits")
	}
	if !strings.Contains(unmatchedHTML, "ghost_node") {
		t.Errorf("unmatched HTML does not mention ghost_node")
	}

	// 6. Multi-stream and card crossings
	branchedPlan := proto.Clone(plan).(*planpb.StepPlan)
	branchedPlan.Nodes[1].Stream = stepplan.StreamParallel
	branchedPlanPath := filepath.Join(tmpDir, "branched_plan.json")
	branchedPlanData, err := stepplan.MarshalAsJSON(branchedPlan)
	if err != nil {
		t.Fatalf("MarshalAsJSON branched plan failed: %v", err)
	}
	if err := os.WriteFile(branchedPlanPath, branchedPlanData, 0644); err != nil {
		t.Fatalf("write branched plan file failed: %v", err)
	}

	descWithPeer := proto.Clone(desc).(*planpb.StepGraphDescription)
	descWithPeer.Nodes[1].Answer = stepplan.AnswerPeerGPU
	descPeerPath := filepath.Join(tmpDir, "desc_peer.json")
	descPeerData, err := protojson.Marshal(descWithPeer)
	if err != nil {
		t.Fatalf("marshal descWithPeer failed: %v", err)
	}
	if err := os.WriteFile(descPeerPath, descPeerData, 0644); err != nil {
		t.Fatalf("write desc_peer.json failed: %v", err)
	}

	branchedHTMLPath := filepath.Join(tmpDir, "branched_graph.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", branchedPlanPath, "--graph", descPeerPath, "--out", branchedHTMLPath}); err != nil {
		t.Fatalf("cmdPlan graph branched failed: %v", err)
	}
	branchedHTMLBytes, err := os.ReadFile(branchedHTMLPath)
	if err != nil {
		t.Fatalf("read branched_graph.html failed: %v", err)
	}
	branchedHTML := string(branchedHTMLBytes)
	verifyGraphHTML(t, branchedHTML, branchedPlan)
	if !strings.Contains(branchedHTML, "Parallel Stream") {
		t.Errorf("branched HTML does not display Parallel Stream column")
	}
	if !strings.Contains(branchedHTML, "edge-crosses-cards") {
		t.Errorf("branched HTML does not mark cross-card edge")
	}
	if !strings.Contains(branchedHTML, "peer GPU") {
		t.Errorf("branched HTML does not mark peer GPU answer")
	}
	if !strings.Contains(branchedHTML, "2</span> Card Crossings") {
		t.Errorf("branched HTML does not state 2 Card Crossings")
	}

	// 7. Upload CI artifacts
	if artifactsDir := os.Getenv("ARTIFACTS"); artifactsDir != "" {
		if err := os.MkdirAll(artifactsDir, 0755); err != nil {
			t.Fatalf("create ARTIFACTS directory failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tiny-deepseek-step-graph.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write tiny-deepseek-step-graph.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "stepgraph.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write stepgraph.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "tiny-deepseek-measured-step-graph.html"), measuredHTMLBytes, 0644); err != nil {
			t.Fatalf("write tiny-deepseek-measured-step-graph.html failed: %v", err)
		}
	}
}
