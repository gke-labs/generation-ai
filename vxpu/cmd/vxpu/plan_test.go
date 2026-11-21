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

	// 3. Checksum refusal test
	testChecksumRefusal(t, ctx, artifactDir, emittedPlan, tmpDir)

	// 4. Upload CI artifacts
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
	}
}
