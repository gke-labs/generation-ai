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
	"bytes"
	"io"
	"os"
	"path/filepath"
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
