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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
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

func TestCmdPlanEmitBinaryCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")

	var cat planpb.KernelCatalog
	catFile := filepath.Join("..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
	if err := kernelplan.ReadJSON(catFile, &cat); err != nil {
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

	if err := cmdPlan(ctx, []string{"emit", "--artifact", fixtureDir, "--catalog", catPath, "--out", planPath, "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("cmdPlan emit failed with binary catalog: %v", err)
	}

	plan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}
	if plan.TreeCommit != "commit123" {
		t.Errorf("plan tree_commit = %q, want commit123", plan.TreeCommit)
	}
}

func TestCmdPlanEmit_TinyDeepSeek(t *testing.T) {
	artifactDir := getTinyDeepSeekArtifactDir(t)
	tmpDir := t.TempDir()
	planPath := filepath.Join(tmpDir, "plan.json")
	treeHTMLPath := filepath.Join(tmpDir, "tree.html")
	ctx := t.Context()

	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	catPath := filepath.Join("..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
	tablePath := filepath.Join(fixtureDir, "kernel_table.json")

	// 1. Emit plan with no description from Python
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--catalog", catPath,
		"--table", tablePath,
		"--out", planPath,
		"--json",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	// 2. Read graph call sites and emitted plan
	graphCallSites, err := kernelplan.ReadCallSites(artifactDir)
	if err != nil {
		t.Fatalf("ReadCallSites failed: %v", err)
	}
	plan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}

	// 3. Verify call-site entries match the graph's call sites one for one, by node name,
	// and that macroscopic family sites are preserved.
	if len(plan.Sites) != 141 {
		t.Errorf("plan.Sites count = %d, want 141 (74 artifact family sites + 67 call sites)", len(plan.Sites))
	}

	var callSiteEntries []*planpb.PlanSite
	for _, ps := range plan.Sites {
		if ps.Body != "" {
			callSiteEntries = append(callSiteEntries, ps)
		}
	}
	if len(callSiteEntries) != len(graphCallSites) {
		t.Fatalf("callSiteEntries count = %d, graph call sites count = %d; want exact match", len(callSiteEntries), len(graphCallSites))
	}
	t.Logf("Call site counts match: graph=%d, plan call sites=%d", len(graphCallSites), len(callSiteEntries))

	graphNodeSet := make(map[string]bool, len(graphCallSites))
	for _, cs := range graphCallSites {
		graphNodeSet[cs.Node] = true
	}
	planIdSet := make(map[string]bool, len(callSiteEntries))
	for _, ps := range callSiteEntries {
		planIdSet[ps.Id] = true
	}

	// Verify none is missing on either side
	for _, cs := range graphCallSites {
		if !planIdSet[cs.Node] {
			t.Errorf("graph call site %q is missing in emitted plan", cs.Node)
		}
	}
	for _, ps := range callSiteEntries {
		if !graphNodeSet[ps.Id] {
			t.Errorf("plan site %q is missing in graph call sites", ps.Id)
		}
	}

	for i := range callSiteEntries {
		if callSiteEntries[i].Id != graphCallSites[i].Node {
			t.Errorf("site[%d] node name mismatch: plan=%q, graph=%q", i, callSiteEntries[i].Id, graphCallSites[i].Node)
		}
	}

	// 4. Compare chosen bodies with existing golden plan
	goldenPlanPath := filepath.Join(fixtureDir, "golden_call_sites_plan.json")
	goldenPlan, err := kernelplan.ReadPlan(goldenPlanPath)
	if err != nil {
		t.Fatalf("ReadPlan golden failed: %v", err)
	}
	if len(goldenPlan.Sites) != len(callSiteEntries) {
		t.Fatalf("golden sites count = %d, plan sites count = %d", len(goldenPlan.Sites), len(callSiteEntries))
	}
	for i := range callSiteEntries {
		if callSiteEntries[i].Id != goldenPlan.Sites[i].Id {
			t.Errorf("site[%d] id = %q, want %q", i, callSiteEntries[i].Id, goldenPlan.Sites[i].Id)
		}
		if callSiteEntries[i].Body != goldenPlan.Sites[i].Body {
			t.Errorf("site[%d] %q body = %q, want golden body %q", i, callSiteEntries[i].Id, callSiteEntries[i].Body, goldenPlan.Sites[i].Body)
		}
		if callSiteEntries[i].Kind != goldenPlan.Sites[i].Kind {
			t.Errorf("site[%d] %q kind = %v, want golden kind %v", i, callSiteEntries[i].Id, callSiteEntries[i].Kind, goldenPlan.Sites[i].Kind)
		}
		if goldenPlan.Sites[i].Strategy != "" {
			t.Errorf("golden site[%d] %q strategy = %q, want empty", i, goldenPlan.Sites[i].Id, goldenPlan.Sites[i].Strategy)
		}
	}

	deepseekSegs, err := kernelplan.ReadSegments(artifactDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}
	assertSegmentReadsWritesInvariants(t, deepseekSegs)

	cpuMK, err := kernelplan.MachineKindFromName("c3d-standard-60")
	if err != nil {
		t.Fatalf("MachineKindFromName CPU: %v", err)
	}
	wantPackagedSegments := []string{"submod_0", "submod_2", "submod_4", "submod_6", "submod_8", "submod_10"}
	cpuReqPkgs, err := kernelplan.RequiredPackages(plan, deepseekSegs, cpuMK)
	if err != nil {
		t.Fatalf("RequiredPackages CPU: %v", err)
	}
	cpuPkgNames := make([]string, len(cpuReqPkgs))
	cpuReqByName := make(map[string]string, len(cpuReqPkgs))
	for i, p := range cpuReqPkgs {
		cpuPkgNames[i] = p.Name
		cpuReqByName[p.Name] = p.Key
	}
	if !slices.Equal(cpuPkgNames, wantPackagedSegments) {
		t.Errorf("CPU required packages = %v, want %v", cpuPkgNames, wantPackagedSegments)
	}
	for _, seg := range plan.Segments {
		if kernelplan.SaysPackage(seg.Run) {
			want := cpuReqByName[seg.Name]
			if want == "" {
				t.Errorf("CPU packaged segment %q missing from RequiredPackages", seg.Name)
			} else if seg.PackageKey != want {
				t.Errorf("CPU segment %q package_key = %q, want RequiredPackages key %q", seg.Name, seg.PackageKey, want)
			}
		} else if seg.PackageKey != "" {
			t.Errorf("CPU non-packaged segment %q (%s) has package_key = %q, want empty", seg.Name, seg.Run, seg.PackageKey)
		}
	}

	// 5. Verify planning for GPU machine with --rows (default catalog, no table override)
	gpuPlanPath := filepath.Join(tmpDir, "gpu_plan.json")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--catalog", catPath,
		"--machine", "g2-standard-48",
		"--rows", "1",
		"--out", gpuPlanPath,
		"--json",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit on gpu machine failed: %v", err)
	}
	gpuPlan, err := kernelplan.ReadPlan(gpuPlanPath)
	if err != nil {
		t.Fatalf("ReadPlan gpuPlan failed: %v", err)
	}
	gpuMK, err := kernelplan.MachineKindFromName("g2-standard-48")
	if err != nil {
		t.Fatalf("MachineKindFromName GPU: %v", err)
	}
	gpuSegs, err := kernelplan.ReadSegments(artifactDir)
	if err != nil {
		t.Fatalf("ReadSegments GPU: %v", err)
	}
	gpuReqPkgs, err := kernelplan.RequiredPackages(gpuPlan, gpuSegs, gpuMK)
	if err != nil {
		t.Fatalf("RequiredPackages GPU: %v", err)
	}
	gpuPkgNames := make([]string, len(gpuReqPkgs))
	gpuReqByName := make(map[string]string, len(gpuReqPkgs))
	for i, p := range gpuReqPkgs {
		gpuPkgNames[i] = p.Name
		gpuReqByName[p.Name] = p.Key
	}
	if !slices.Equal(gpuPkgNames, wantPackagedSegments) {
		t.Errorf("GPU required packages = %v, want %v", gpuPkgNames, wantPackagedSegments)
	}
	for _, seg := range gpuPlan.Segments {
		if kernelplan.SaysPackage(seg.Run) {
			want := gpuReqByName[seg.Name]
			if want == "" {
				t.Errorf("GPU packaged segment %q missing from RequiredPackages", seg.Name)
			} else if seg.PackageKey != want {
				t.Errorf("GPU segment %q package_key = %q, want RequiredPackages key %q", seg.Name, seg.PackageKey, want)
			}
			if cpuKey := cpuReqByName[seg.Name]; cpuKey != "" && seg.PackageKey == cpuKey {
				t.Errorf("GPU segment %q package_key %q unexpectedly equals CPU key", seg.Name, seg.PackageKey)
			}
		} else if seg.PackageKey != "" {
			t.Errorf("GPU non-packaged segment %q (%s) has package_key = %q, want empty", seg.Name, seg.Run, seg.PackageKey)
		}
	}

	// Verify plan identity changes when a package key changes and not otherwise.
	cpuPlanOtherCommitPath := filepath.Join(tmpDir, "cpu_plan_other_commit.json")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--catalog", catPath,
		"--table", tablePath,
		"--out", cpuPlanOtherCommitPath,
		"--json",
		"--tree-commit", "fedcba9876543210fedcba9876543210fedcba98",
	}); err != nil {
		t.Fatalf("cmdPlan emit with other commit failed: %v", err)
	}
	cpuPlanOtherCommit, err := kernelplan.ReadPlan(cpuPlanOtherCommitPath)
	if err != nil {
		t.Fatalf("ReadPlan cpuPlanOtherCommit failed: %v", err)
	}
	if cpuPlanOtherCommit.PlanHash == plan.PlanHash {
		t.Errorf("plan_hash did not change when package keys changed with software version: %s", plan.PlanHash)
	}

	// Verify a plan with a package key on a plain-torch segment is refused.
	badPlan := proto.Clone(plan).(*planpb.PlanFile)
	for _, seg := range badPlan.Segments {
		if !kernelplan.SaysPackage(seg.Run) {
			seg.PackageKey = "0123456789abcdef0123456789abcdef"
			break
		}
	}
	badPlanBytes, err := kernelplan.MarshalAsJSON(badPlan)
	if err != nil {
		t.Fatalf("MarshalAsJSON badPlan: %v", err)
	}
	badPlanPath := filepath.Join(tmpDir, "bad_deepseek_plan.json")
	if err := os.WriteFile(badPlanPath, badPlanBytes, 0o644); err != nil {
		t.Fatalf("WriteFile badPlan: %v", err)
	}
	if _, err := kernelplan.ReadPlan(badPlanPath); err == nil {
		t.Error("ReadPlan accepted tiny DeepSeek plan with package key on non-packaged segment, want error")
	}
	if _, err := kernelplan.RequiredPackages(badPlan, deepseekSegs, cpuMK); err == nil {
		t.Error("RequiredPackages accepted tiny DeepSeek plan with package key on non-packaged segment, want error")
	}

	var gpuCallSiteEntries []*planpb.PlanSite
	for _, ps := range gpuPlan.Sites {
		if ps.Body != "" {
			gpuCallSiteEntries = append(gpuCallSiteEntries, ps)
		}
	}
	gpuGoldenPlanPath := filepath.Join(fixtureDir, "golden_call_sites_plan_gpu.json")
	gpuGoldenPlan, err := kernelplan.ReadPlan(gpuGoldenPlanPath)
	if err != nil {
		t.Fatalf("ReadPlan golden GPU failed: %v", err)
	}
	if len(gpuGoldenPlan.Sites) != len(gpuCallSiteEntries) {
		t.Fatalf("golden GPU sites count = %d, plan sites count = %d", len(gpuGoldenPlan.Sites), len(gpuCallSiteEntries))
	}
	for i := range gpuCallSiteEntries {
		if gpuCallSiteEntries[i].Id != gpuGoldenPlan.Sites[i].Id {
			t.Errorf("gpu site[%d] id = %q, want %q", i, gpuCallSiteEntries[i].Id, gpuGoldenPlan.Sites[i].Id)
		}
		if gpuCallSiteEntries[i].Body != gpuGoldenPlan.Sites[i].Body {
			t.Errorf("gpu site[%d] %q body = %q, want golden body %q", i, gpuCallSiteEntries[i].Id, gpuCallSiteEntries[i].Body, gpuGoldenPlan.Sites[i].Body)
		}
		if gpuCallSiteEntries[i].Kind != gpuGoldenPlan.Sites[i].Kind {
			t.Errorf("gpu site[%d] %q kind = %v, want golden kind %v", i, gpuCallSiteEntries[i].Id, gpuCallSiteEntries[i].Kind, gpuGoldenPlan.Sites[i].Kind)
		}
		if gpuGoldenPlan.Sites[i].Strategy != "" {
			t.Errorf("golden GPU site[%d] %q strategy = %q, want empty", i, gpuGoldenPlan.Sites[i].Id, gpuGoldenPlan.Sites[i].Strategy)
		}
	}
	for _, ps := range gpuPlan.Sites {
		switch {
		case strings.HasPrefix(ps.Id, "fp8_linear"):
			if ps.Body != "gpu-fp8-gemv" {
				t.Errorf("cuda site %q body = %q, want gpu-fp8-gemv", ps.Id, ps.Body)
			}
		case strings.HasPrefix(ps.Id, "hc_sinkhorn"):
			if ps.Body != "gpu-triton" {
				t.Errorf("cuda site %q body = %q, want gpu-triton", ps.Id, ps.Body)
			}
		case strings.HasPrefix(ps.Id, "rotary"):
			if ps.Body != "gpu-triton" {
				t.Errorf("cuda site %q body = %q, want gpu-triton", ps.Id, ps.Body)
			}
		case ps.Id == "head_linear":
			if ps.Body != "gpu-triton-bf16" {
				t.Errorf("cuda site %q body = %q, want gpu-triton-bf16", ps.Id, ps.Body)
			}
		}
	}

	// 6. Verify the tree page on that emitted plan shows a body on every call site
	// and the package key on every packaged segment.
	if err := cmdPlan(ctx, []string{
		"tree",
		"--artifact", artifactDir,
		"--plan", planPath,
		"--out", treeHTMLPath,
	}); err != nil {
		t.Fatalf("cmdPlan tree failed: %v", err)
	}
	htmlBytes, err := os.ReadFile(treeHTMLPath)
	if err != nil {
		t.Fatalf("ReadFile tree.html: %v", err)
	}
	htmlStr := string(htmlBytes)
	if !strings.Contains(htmlStr, "badge-body") {
		t.Error("tree.html missing badge-body badge")
	}
	for _, p := range cpuReqPkgs {
		if !strings.Contains(htmlStr, p.Key) {
			t.Errorf("tree.html missing package key %q for segment %q", p.Key, p.Name)
		}
	}
	// Verify every call site's body is present in the rendered HTML
	for _, ps := range callSiteEntries {
		badgeSnippet := fmt.Sprintf("<span class=\"badge badge-body\"><span class=\"badge-label\">body</span>%s</span>", ps.Body)
		if !strings.Contains(htmlStr, badgeSnippet) {
			t.Errorf("tree.html does not contain body badge for body %q", ps.Body)
			break
		}
	}

	// 7. Verify JoinPlan refuses unmatched on either side by name
	segments, err := kernelplan.ReadSegments(artifactDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}
	if err := kernelplan.JoinPlan(segments, plan); err != nil {
		t.Fatalf("JoinPlan failed on complete plan: %v", err)
	}

	lastSite := plan.Sites[len(plan.Sites)-1]
	brokenPlan := &planpb.PlanFile{
		ArtifactMd5: plan.ArtifactMd5,
		TreeCommit:  plan.TreeCommit,
		Sites:       plan.Sites[:len(plan.Sites)-1],
	}
	if err := kernelplan.JoinPlan(segments, brokenPlan); err == nil || !strings.Contains(err.Error(), lastSite.Id) {
		t.Errorf("JoinPlan with missing site error = %v, want substring %q", err, lastSite.Id)
	}

	extraPlan := &planpb.PlanFile{
		ArtifactMd5: plan.ArtifactMd5,
		TreeCommit:  plan.TreeCommit,
		Sites:       append(slices.Clone(plan.Sites), &planpb.PlanSite{Id: "extra_site", Body: "ref"}),
	}
	if err := kernelplan.JoinPlan(segments, extraPlan); err == nil || !strings.Contains(err.Error(), "extra_site") {
		t.Errorf("JoinPlan with extra site error = %v, want substring %q", err, "extra_site")
	}
}

func TestCmdPlanEmit_TinyLlama(t *testing.T) {
	ctx := t.Context()
	artifactDir := getTinyLlamaArtifactDir(t)
	tmpDir := t.TempDir()
	planPath := filepath.Join(tmpDir, "plan.json")

	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--rows", "1",
		"--out", planPath,
		"--json",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	segments, err := kernelplan.ReadSegments(artifactDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}
	assertSegmentReadsWritesInvariants(t, segments)
}

func TestCmdPlanSegments_ReadsWritesSynthetic(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	// Write minimal manifest.json
	manifestJSON := `{
		"config": {"model_type": "test_model"},
		"tensors": {
			"model.layers.0.weight": {"dtype": "BF16", "shape": [128, 128]}
		}
	}`
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), []byte(manifestJSON), 0644); err != nil {
		t.Fatalf("write manifest.json: %v", err)
	}

	// Write minimal binding.json
	bindingJSON := `{
		"bound": {
			"model.layers.0.weight": {"dtype": "BF16", "shape": [128, 128]}
		}
	}`
	if err := os.WriteFile(filepath.Join(tmpDir, "binding.json"), []byte(bindingJSON), 0644); err != nil {
		t.Fatalf("write binding.json: %v", err)
	}

	// Write decode.pt2
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "op2"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}], "outputs": [{"as_tensor": {"name": "op1"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.w,m.w,L"}},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "op1"}}}], "outputs": [{"as_tensor": {"name": "op2"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.e,m.e,MoE"}}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	segsPath := filepath.Join(tmpDir, "segments.json")
	if err := cmdPlan(ctx, []string{
		"segments",
		"--artifact", tmpDir,
		"--out", segsPath,
		"--json",
	}); err != nil {
		t.Fatalf("cmdPlan segments failed: %v", err)
	}

	segsBytes, err := os.ReadFile(segsPath)
	if err != nil {
		t.Fatalf("ReadFile segments failed: %v", err)
	}
	var segs []*planpb.Segment
	if err := json.Unmarshal(segsBytes, &segs); err != nil {
		t.Fatalf("Unmarshal segments failed: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("segments count = %d, want 2", len(segs))
	}
	assertSegmentReadsWritesInvariants(t, segs)
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "op2"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}], "outputs": [{"as_tensor": {"name": "op1"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.w,m.w,L"}},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "op1"}}}], "outputs": [{"as_tensor": {"name": "op2"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.e,m.e,MoE"}}
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
	assertSegmentReadsWritesInvariants(t, segs)
	if len(segs[0].Reads) == 0 || segs[0].Reads[0].Name != "input_ids" {
		t.Errorf("segs[0].Reads = %+v, want [input_ids]", segs[0].Reads)
	}
	if len(segs[0].Writes) == 0 || segs[0].Writes[0].Name != "op1" {
		t.Errorf("segs[0].Writes = %+v, want [op1]", segs[0].Writes)
	}
	if len(segs[1].Reads) == 0 || segs[1].Reads[0].Name != "op1" {
		t.Errorf("segs[1].Reads = %+v, want [op1]", segs[1].Reads)
	}
	if len(segs[1].Writes) == 0 || segs[1].Writes[0].Name != "op2" {
		t.Errorf("segs[1].Writes = %+v, want [op2]", segs[1].Writes)
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
	if !strings.Contains(txt, "reads: input_ids") || !strings.Contains(txt, "writes: op1") {
		t.Errorf("segments text output missing reads/writes for submod_0: %s", txt)
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

var (
	tinyDeepSeekArtifactOnce sync.Once
	tinyDeepSeekArtifactDir  string
	tinyDeepSeekArtifactErr  string
)

func getTinyDeepSeekArtifactDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("TINY_DEEPSEEK_ARTIFACT"); dir != "" {
		if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err == nil {
			return dir
		}
	}
	tinyDeepSeekArtifactOnce.Do(func() {
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
			tinyDeepSeekArtifactErr = fmt.Sprintf("tiny DeepSeek artifact unavailable: %v (stderr: %s)", err, stderr.String())
			return
		}
		dir := strings.TrimSpace(stdout.String())
		if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err != nil {
			tinyDeepSeekArtifactErr = fmt.Sprintf("decode.pt2 not found in generated artifact dir %s: %v", dir, err)
			return
		}
		tinyDeepSeekArtifactDir = dir
	})
	if tinyDeepSeekArtifactErr != "" {
		t.Skip(tinyDeepSeekArtifactErr)
	}
	return tinyDeepSeekArtifactDir
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
		}

		for _, writeVal := range seg.Writes {
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "op2"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}], "outputs": [{"as_tensor": {"name": "op1"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.w,m.w,L"}},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "op1"}}}], "outputs": [{"as_tensor": {"name": "op2"}}], "metadata": {"nn_module_stack": "L__self__,,D;L__self__m,m,T;L__self__m.e,m.e,MoE"}}
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
	if !strings.Contains(html, "Reads:") || !strings.Contains(html, "Writes:") {
		t.Errorf("expected tree.html to show Reads and Writes, got:\n%s", html)
	}

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
	if len(tree.Segments[0].Reads) == 0 || tree.Segments[0].Reads[0].Name != "input_ids" {
		t.Errorf("tree.Segments[0].Reads = %+v, want [input_ids]", tree.Segments[0].Reads)
	}
	if len(tree.Segments[0].Writes) == 0 || tree.Segments[0].Writes[0].Name != "op1" {
		t.Errorf("tree.Segments[0].Writes = %+v, want [op1]", tree.Segments[0].Writes)
	}
	if len(tree.Segments[1].Reads) == 0 || tree.Segments[1].Reads[0].Name != "op1" {
		t.Errorf("tree.Segments[1].Reads = %+v, want [op1]", tree.Segments[1].Reads)
	}
	if len(tree.Segments[1].Writes) == 0 || tree.Segments[1].Writes[0].Name != "op2" {
		t.Errorf("tree.Segments[1].Writes = %+v, want [op2]", tree.Segments[1].Writes)
	}
	assertSegmentReadsWritesInvariants(t, tree.Segments)
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
			} else {
				if !strings.Contains(rowHTML, "default implementation") {
					t.Errorf("covered node %s with empty body does not carry 'default implementation', got row: %s", cs.Node, rowHTML)
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

	// 2. Emit plan with vxpu plan emit without --sites
	fixtureDir := filepath.Join(findGitRoot(), "vxpu", "pkg", "kernelplan", "testdata")
	catalogDir := filepath.Join(findGitRoot(), "vxpu", "pkg", "kernelplan", "catalog")
	planPath := filepath.Join(tmpDir, "emitted_plan.json")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artifactDir,
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
		"--catalog", filepath.Join(catalogDir, "kernel_catalog.json"),
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
					s.ExecutedBodies = []*planpb.ExecutedBody{{Body: "gpu-fp8-gemv"}}
				} else if s.Body != "" {
					s.ExecutedBodies = []*planpb.ExecutedBody{{Body: s.Body}}
				} else {
					s.ExecutedBodies = []*planpb.ExecutedBody{{Body: ""}}
				}
				s.Body = ""
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

	// 5b. When both plan and record are provided, unmatched measurements are retained in their own section
	outBothUnmatchedJSON := filepath.Join(tmpDir, "both_unmatched_tree.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", artifactDir, "--plan", planPath, "--record", unmatchedRecordPath, "--out", outBothUnmatchedJSON, "--json"}); err != nil {
		t.Fatalf("cmdPlan tree with plan and unmatched record failed: %v", err)
	}
	bothUnmatchedJSONBytes, err := os.ReadFile(outBothUnmatchedJSON)
	if err != nil {
		t.Fatalf("read both_unmatched_tree.json failed: %v", err)
	}
	var bothUnmatchedTree planpb.ModelTree
	if err := protojson.Unmarshal(bothUnmatchedJSONBytes, &bothUnmatchedTree); err != nil {
		t.Fatalf("protojson.Unmarshal both_unmatched_tree.json failed: %v", err)
	}
	if len(bothUnmatchedTree.UnmatchedMeasurements) != 2 {
		t.Errorf("got %d unmatched measurements in tree, want 2", len(bothUnmatchedTree.UnmatchedMeasurements))
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
	if strings.Contains(unmatchedHTML, "http://") || strings.Contains(unmatchedHTML, "https://") {
		t.Errorf("unmatched HTML page references external resource (contains http:// or https://)")
	}
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

func make85NodePlanFixture() (*planpb.StepPlan, *planpb.StepGraphDescription) {
	var (
		descNodes []*planpb.StepNode
		planNodes []*planpb.NodePlan
	)

	// Layer 0
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_0", Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_0", Stream: stepplan.StreamParallel, Waits: nil,
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "experts_0", Kind: "experts", Execution: stepplan.ExecutionExchange,
		Host: &planpb.HostCall{Collection: "experts", Index: 0},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "experts_0", Stream: stepplan.StreamMain, Waits: []string{"dense_0"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_1", Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_1", Stream: stepplan.StreamParallel, Waits: []string{"dense_0", "experts_0"},
	})

	// Layer 1
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "engram_1", Kind: "engram", Execution: stepplan.ExecutionExchange,
		Host: &planpb.HostCall{Collection: "engram", Index: 1},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "engram_1", Stream: stepplan.StreamMain, Waits: []string{"dense_1"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_2", Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_2", Stream: stepplan.StreamParallel, Waits: []string{"dense_1", "engram_1", "dense_0"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "experts_1", Kind: "experts", Execution: stepplan.ExecutionExchange,
		Host: &planpb.HostCall{Collection: "experts", Index: 1},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "experts_1", Stream: stepplan.StreamMain, Waits: []string{"dense_2"},
	})

	denseCount := 3
	lastDense := "dense_2"
	secondLastDense := "dense_1"

	// Layers 2 to 39: 38 layers
	for l := 2; l < 40; l++ {
		if l == 12 {
			engramName := fmt.Sprintf("engram_%d", l)
			descNodes = append(descNodes, &planpb.StepNode{
				Name: engramName, Kind: "engram", Execution: stepplan.ExecutionExchange,
				Host: &planpb.HostCall{Collection: "engram", Index: int32(l)},
			})
			planNodes = append(planNodes, &planpb.NodePlan{
				Name: engramName, Stream: stepplan.StreamMain, Waits: []string{lastDense},
			})

			denseName := fmt.Sprintf("dense_%d", denseCount)
			denseCount++
			descNodes = append(descNodes, &planpb.StepNode{
				Name: denseName, Kind: "dense", Execution: stepplan.ExecutionGraph,
			})
			planNodes = append(planNodes, &planpb.NodePlan{
				Name: denseName, Stream: stepplan.StreamParallel, Waits: []string{lastDense, engramName, secondLastDense},
			})
			secondLastDense = lastDense
			lastDense = denseName
		}

		expName := fmt.Sprintf("experts_%d", l)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: expName, Kind: "experts", Execution: stepplan.ExecutionExchange,
			Host: &planpb.HostCall{Collection: "experts", Index: int32(l)},
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: expName, Stream: stepplan.StreamMain, Waits: []string{lastDense},
		})

		denseName := fmt.Sprintf("dense_%d", denseCount)
		denseCount++
		descNodes = append(descNodes, &planpb.StepNode{
			Name: denseName, Kind: "dense", Execution: stepplan.ExecutionGraph,
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: denseName, Stream: stepplan.StreamParallel, Waits: []string{lastDense, expName, secondLastDense},
		})
		secondLastDense = lastDense
		lastDense = denseName
	}

	// Final norm and lm_head dense node
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_head", Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_head", Stream: stepplan.StreamParallel, Waits: []string{lastDense},
	})

	return &planpb.StepPlan{Nodes: planNodes}, &planpb.StepGraphDescription{Nodes: descNodes}
}

func TestCmdPlanGraph_85NodeTwoStream(t *testing.T) {
	ctx := t.Context()
	plan, desc := make85NodePlanFixture()
	if len(plan.Nodes) != 85 {
		t.Fatalf("expected 85 nodes, got %d", len(plan.Nodes))
	}

	tmpDir := t.TempDir()
	planPath := filepath.Join(tmpDir, "stepplan_85.json")
	planData, err := stepplan.MarshalAsJSON(plan)
	if err != nil {
		t.Fatalf("MarshalAsJSON plan failed: %v", err)
	}
	if err := os.WriteFile(planPath, planData, 0644); err != nil {
		t.Fatalf("write plan file failed: %v", err)
	}

	descPath := filepath.Join(tmpDir, "stepgraph_85.json")
	descData, err := kernelplan.MarshalAsJSON(desc)
	if err != nil {
		t.Fatalf("MarshalAsJSON desc failed: %v", err)
	}
	if err := os.WriteFile(descPath, descData, 0644); err != nil {
		t.Fatalf("write desc file failed: %v", err)
	}

	outHTML := filepath.Join(tmpDir, "graph_85.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "--graph", descPath, "--out", outHTML}); err != nil {
		t.Fatalf("cmdPlan graph 85 HTML failed: %v", err)
	}
	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("read graph_85.html failed: %v", err)
	}
	html := string(htmlBytes)
	verifyGraphHTML(t, html, plan)

	if !strings.Contains(html, "85</span> Nodes") {
		t.Errorf("HTML missing 85 Nodes count")
	}
	if !strings.Contains(html, "Parallel Stream") {
		t.Errorf("HTML missing Parallel Stream column")
	}
	if !strings.Contains(html, "Edge Crossings") {
		t.Errorf("HTML missing Edge Crossings count")
	}

	// Verify that with script disabled, every edge path has d="M ..." rendered
	pathMatches := strings.Count(html, `<path class="edge-path"`)
	if pathMatches != 165 {
		t.Errorf("expected 165 edge paths, got %d", pathMatches)
	}
	for _, n := range plan.Nodes {
		for _, w := range n.Waits {
			edgeAttr := fmt.Sprintf(`class="edge-path" data-from="%s" data-to="%s" d="M `, w, n.Name)
			if !strings.Contains(html, edgeAttr) {
				t.Errorf("HTML missing rendered edge path with d attribute for %s -> %s", w, n.Name)
			}
		}
	}

	// Upload CI artifact
	if artifactsDir := os.Getenv("ARTIFACTS"); artifactsDir != "" {
		if err := os.MkdirAll(artifactsDir, 0755); err != nil {
			t.Fatalf("create ARTIFACTS directory failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "deepseek-85-step-graph.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write deepseek-85-step-graph.html failed: %v", err)
		}
		if err := os.WriteFile(filepath.Join(artifactsDir, "step-graph-85.html"), htmlBytes, 0644); err != nil {
			t.Fatalf("write step-graph-85.html failed: %v", err)
		}
	}
}

func TestCmdPlanGraph_MultiNodeAndLongWaits(t *testing.T) {
	ctx := t.Context()
	var (
		descNodes []*planpb.StepNode
		planNodes []*planpb.NodePlan
	)

	// Depth 0 root node
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "root_op", Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "root_op", Stream: stepplan.StreamMain, Waits: nil,
	})

	var allWaits []string
	allWaits = append(allWaits, "root_op")

	// 5 main stream nodes at depth 1
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("long_named_main_stream_producer_module_op_layer_12_part_%d", i)
		allWaits = append(allWaits, name)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: name, Kind: "experts", Execution: stepplan.ExecutionExchange,
			Host: &planpb.HostCall{Collection: "experts", Index: int32(i)},
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: name, Stream: stepplan.StreamMain, Waits: []string{"root_op"},
		})
	}

	// 5 parallel stream nodes at depth 1
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("long_named_parallel_stream_producer_kernel_projection_layer_12_part_%d", i)
		allWaits = append(allWaits, name)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: name, Kind: "dense", Execution: stepplan.ExecutionGraph,
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: name, Stream: stepplan.StreamParallel, Waits: []string{"root_op"},
		})
	}

	// Consumer node at depth 2 with 11 long-named waits
	consumerName := "extremely_long_named_consumer_kernel_operation_that_receives_eleven_dependencies_from_both_streams"
	descNodes = append(descNodes, &planpb.StepNode{
		Name: consumerName, Kind: "dense", Execution: stepplan.ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: consumerName, Stream: stepplan.StreamParallel, Waits: allWaits,
	})

	plan := &planpb.StepPlan{Nodes: planNodes}
	desc := &planpb.StepGraphDescription{Nodes: descNodes}

	tmpDir := t.TempDir()
	planPath := filepath.Join(tmpDir, "stepplan_wide.json")
	planData, err := stepplan.MarshalAsJSON(plan)
	if err != nil {
		t.Fatalf("MarshalAsJSON plan failed: %v", err)
	}
	if err := os.WriteFile(planPath, planData, 0644); err != nil {
		t.Fatalf("write plan file failed: %v", err)
	}

	descPath := filepath.Join(tmpDir, "stepgraph_wide.json")
	descData, err := kernelplan.MarshalAsJSON(desc)
	if err != nil {
		t.Fatalf("MarshalAsJSON desc failed: %v", err)
	}
	if err := os.WriteFile(descPath, descData, 0644); err != nil {
		t.Fatalf("write desc file failed: %v", err)
	}

	outHTML := filepath.Join(tmpDir, "graph_wide.html")
	if err := cmdPlan(ctx, []string{"graph", "--plan", planPath, "--graph", descPath, "--out", outHTML}); err != nil {
		t.Fatalf("cmdPlan graph wide HTML failed: %v", err)
	}
	htmlBytes, err := os.ReadFile(outHTML)
	if err != nil {
		t.Fatalf("read graph_wide.html failed: %v", err)
	}
	html := string(htmlBytes)
	verifyGraphHTML(t, html, plan)

	// Verify that the consumer node shows capped chips and +9 more with tooltip
	if !strings.Contains(html, `+9 more`) {
		t.Errorf("expected HTML to contain '+9 more' wait chip overflow counter")
	}
	if !strings.Contains(html, `title="waits: `) {
		t.Errorf("expected HTML to contain title tooltip with full waits list")
	}
	if !strings.Contains(html, fmt.Sprintf(`title="%s"`, consumerName)) {
		t.Errorf("expected HTML to contain title attribute with full consumer node name")
	}
}

func createArtifactWithNodes(t *testing.T, dir string, nodesJSON []string) {
	t.Helper()
	pt2Path := filepath.Join(dir, "decode.pt2")
	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)
	wFormat, _ := zw.Create("decode/archive_format")
	wFormat.Write([]byte("pt2"))
	wVer, _ := zw.Create("decode/archive_version")
	wVer.Write([]byte("0"))
	wModel, _ := zw.Create("decode/models/model.json")
	wModel.Write([]byte(fmt.Sprintf(`{
		"schema_version": {"major": 8, "minor": 20},
		"graph_module": {
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "out"}}}}
				]
			},
			"graph": {
				"nodes": [%s]
			}
		}
	}`, strings.Join(nodesJSON, ","))))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}
}

func fakeModelstoreServer(t *testing.T) *httptest.Server {
	t.Helper()
	blobs := map[string][]byte{}
	manifests := map[string][]byte{}
	files := map[string][]byte{}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if sha, ok := strings.CutPrefix(r.URL.Path, "/blobs/"); ok {
			if data, ok := blobs["sha256:"+sha]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
				return
			}
			http.NotFound(w, r)
			return
		}
		if strings.Contains(r.URL.Path, "/resolve/") {
			key := r.URL.Path
			if r.Method == http.MethodPut {
				data, _ := io.ReadAll(r.Body)
				files[key] = data
				w.WriteHeader(http.StatusOK)
			} else if data, ok := files[key]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
			return
		}
		path := strings.TrimPrefix(r.URL.Path, "/v2/")
		switch {
		case r.Method == http.MethodPost && strings.Contains(path, "/blobs/uploads/"):
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			blobs[digest] = data
			w.WriteHeader(http.StatusCreated)
		case (r.Method == http.MethodGet || r.Method == http.MethodHead) && strings.Contains(path, "/blobs/"):
			digest := path[strings.LastIndex(path, "/")+1:]
			if data, ok := blobs[digest]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
		case strings.Contains(path, "/manifests/"):
			i := strings.Index(path, "/manifests/")
			key := path[:i] + ":" + path[i+len("/manifests/"):]
			if r.Method == http.MethodPut {
				data, _ := io.ReadAll(r.Body)
				manifests[key] = data
				w.WriteHeader(http.StatusCreated)
			} else if data, ok := manifests[key]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
		case strings.HasSuffix(path, "/tags/list"):
			name := strings.TrimSuffix(path, "/tags/list")
			var tags []string
			for k := range manifests {
				if n, tag, _ := strings.Cut(k, ":"); n == name {
					tags = append(tags, tag)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"name": name, "tags": tags})
		default:
			http.NotFound(w, r)
		}
	}))
}

func fakePutArtifact(t *testing.T, storeURL, repo, tag, artDir string) string {
	t.Helper()
	var layers []weightstore.Layer
	for _, name := range []string{"manifest.json", "binding.json", "decode.pt2"} {
		data, err := os.ReadFile(filepath.Join(artDir, name))
		if err != nil {
			if os.IsNotExist(err) && name != "decode.pt2" {
				continue
			}
			t.Fatalf("read %s: %v", name, err)
		}
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, fmt.Sprintf("%s/v2/%s/blobs/uploads/?digest=%s", storeURL, repo, digest), bytes.NewReader(data))
		if err != nil {
			t.Fatalf("upload req %s: %v", name, err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusCreated {
			t.Fatalf("upload blob %s: %v", name, err)
		}
		_ = resp.Body.Close()

		reqResolve, err := http.NewRequestWithContext(t.Context(), http.MethodPut, fmt.Sprintf("%s/%s/resolve/%s/%s", storeURL, repo, tag, name), bytes.NewReader(data))
		if err != nil {
			t.Fatalf("resolve req %s: %v", name, err)
		}
		rResp, err := http.DefaultClient.Do(reqResolve)
		if err != nil || rResp.StatusCode != http.StatusOK {
			t.Fatalf("put resolve %s: %v", name, err)
		}
		_ = rResp.Body.Close()

		layers = append(layers, weightstore.Layer{
			MediaType:   "application/octet-stream",
			Digest:      digest,
			Size:        int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": name},
		})
	}
	body, err := json.Marshal(weightstore.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.oci.image.manifest.v1+json",
		Layers:        layers,
	})
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", storeURL, repo, tag), bytes.NewReader(body))
	if err != nil {
		t.Fatalf("manifest req: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusCreated {
		t.Fatalf("put manifest: %v", err)
	}
	_ = resp.Body.Close()
	return repo + ":" + tag
}

func fakePutPackageManifest(t *testing.T, storeURL, model, key string) {
	t.Helper()
	repo := "vxpu-packages/" + strings.ToLower(model)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", storeURL, repo, key), strings.NewReader("{}"))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("put package manifest: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		t.Fatalf("put package manifest status: %s", resp.Status)
	}
}

func TestCmdPlanPackagesWithFailures(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
		`{"name": "h1", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "h1"}}]}`,
		`{"name": "s2", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s2"}}]}`,
		`{"name": "h3", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "h3"}}]}`,
		`{"name": "s4", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s4"}}]}`,
		`{"name": "h5", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "h5"}}]}`,
		`{"name": "s6", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s6"}}]}`,
	})

	ts := fakeModelstoreServer(t)
	defer ts.Close()

	plan := &planpb.PlanFile{
		TreeCommit:  "commit123",
		ArtifactMd5: "artifactmd5",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "ref_linear"},
			{Id: "h1", Body: "ref_linear"},
			{Id: "s2", Body: "ref_linear"},
			{Id: "h3", Body: "ref_linear"},
			{Id: "s4", Body: "ref_linear"},
			{Id: "h5", Body: "ref_linear"},
			{Id: "s6", Body: "ref_linear"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
			{Name: "submod_1", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH, Host: true},
			{Name: "submod_2", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
			{Name: "submod_3", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH, Host: true},
			{Name: "submod_4", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
			{Name: "submod_5", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH, Host: true},
			{Name: "submod_6", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("proto.Marshal plan: %v", err)
	}
	planPath := filepath.Join(tmpDir, "plan.pb")
	if err := os.WriteFile(planPath, planBytes, 0644); err != nil {
		t.Fatalf("WriteFile plan: %v", err)
	}

	targetMachine := "c3d-standard-60"
	mk, err := kernelplan.MachineKindFromName(targetMachine)
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}

	segments, err := kernelplan.ReadSegments(tmpDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	pkgs, err := kernelplan.RequiredPackages(plan, segments, mk)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}
	var submod6Pkg *planpb.Package
	for _, p := range pkgs {
		if p.Name == "submod_6" {
			submod6Pkg = p
			break
		}
	}
	if submod6Pkg == nil {
		t.Fatal("submod_6 package not found")
	}

	failureReport := &planpb.PackageReport{
		Failed: []*planpb.CompileFailure{
			{
				Package:         submod6Pkg,
				MachineKind:     mk,
				SoftwareVersion: "commit123",
				Reason:          "Float8_e8m0fnu in namespace at does not name a type",
			},
		},
		BodyOutcomes: []*planpb.BodyCompileOutcome{
			{
				Op:              "vxpu::fp8_linear",
				Body:            "ref_linear",
				MachineKind:     mk,
				SoftwareVersion: "commit123",
				Compiled:        false,
				Reason:          "Float8_e8m0fnu in namespace at does not name a type",
			},
		},
	}
	failuresJSON, err := protojson.Marshal(failureReport)
	if err != nil {
		t.Fatalf("protojson.Marshal failureReport: %v", err)
	}
	failuresPath := filepath.Join(tmpDir, "failures.json")
	if err := os.WriteFile(failuresPath, failuresJSON, 0644); err != nil {
		t.Fatalf("WriteFile failures: %v", err)
	}

	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "v1", tmpDir)

	outReportPath := filepath.Join(tmpDir, "report.json")
	cmdArgs := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--report", failuresPath,
		"--out", outReportPath,
		"--json",
	}

	err = cmdPlan(ctx, cmdArgs)
	if err == nil {
		t.Fatal("expected non-zero exit (error) when a required package failed, got nil")
	}
	if !strings.Contains(err.Error(), "submod_6") {
		t.Errorf("expected error to name submod_6, got: %v", err)
	}

	reportData, err := os.ReadFile(outReportPath)
	if err != nil {
		t.Fatalf("ReadFile out report: %v", err)
	}
	var rep planpb.PackageReport
	if err := protojson.Unmarshal(reportData, &rep); err != nil {
		t.Fatalf("protojson.Unmarshal out report: %v", err)
	}
	if len(rep.Failed) != 1 {
		t.Fatalf("expected 1 failure in report, got %d", len(rep.Failed))
	}
	if rep.Failed[0].Package.Name != "submod_6" {
		t.Errorf("rep.Failed[0].Package.Name = %q, want submod_6", rep.Failed[0].Package.Name)
	}
	if rep.Failed[0].MachineKind != mk {
		t.Errorf("rep.Failed[0].MachineKind = %q, want %q", rep.Failed[0].MachineKind, mk)
	}
	if len(rep.BodyOutcomes) != 1 || rep.BodyOutcomes[0].Body != "ref_linear" || rep.BodyOutcomes[0].Compiled {
		t.Errorf("rep.BodyOutcomes = %+v, want 1 item with ref_linear compiled=false", rep.BodyOutcomes)
	}

	// Second request without --report reads from the store, does not ask again in Missing,
	// and exits non-zero because a required package is known to fail.
	outReport2Path := filepath.Join(tmpDir, "report2.json")
	cmdArgs2 := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outReport2Path,
		"--json",
	}
	err2 := cmdPlan(ctx, cmdArgs2)
	if err2 == nil || !strings.Contains(err2.Error(), "submod_6") {
		t.Fatalf("expected error when a required package is recorded as failing, got: %v", err2)
	}
	report2Data, err := os.ReadFile(outReport2Path)
	if err != nil {
		t.Fatalf("ReadFile out report 2: %v", err)
	}
	var rep2 planpb.PackageReport
	if err := protojson.Unmarshal(report2Data, &rep2); err != nil {
		t.Fatalf("protojson.Unmarshal out report 2: %v", err)
	}
	if len(rep2.Failed) != 1 || rep2.Failed[0].Package.Name != "submod_6" {
		t.Errorf("rep2.Failed = %+v, want submod_6", rep2.Failed)
	}
	if len(rep2.BodyOutcomes) != 1 || rep2.BodyOutcomes[0].Body != "ref_linear" {
		t.Errorf("rep2.BodyOutcomes = %+v, want 1 item with ref_linear", rep2.BodyOutcomes)
	}
	for _, m := range rep2.Missing {
		if m.Name == "submod_6" {
			t.Errorf("second request should not ask for submod_6 in missing")
		}
	}

	// Clearing an unknown package name is refused with a clear error.
	cmdArgsUnknown := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--clear", "submod_99",
	}
	errUnknown := cmdPlan(ctx, cmdArgsUnknown)
	if errUnknown == nil || !strings.Contains(errUnknown.Error(), `"submod_99" is not named by the plan`) {
		t.Fatalf("expected error for unknown clear package name, got: %v", errUnknown)
	}

	// Third request with --clear clears the failure, so it is asked again in Missing and exits 0.
	outReport3Path := filepath.Join(tmpDir, "report3.json")
	cmdArgs3 := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--clear", "submod_6",
		"--out", outReport3Path,
		"--json",
	}
	if err := cmdPlan(ctx, cmdArgs3); err != nil {
		t.Fatalf("clear plan packages request failed: %v", err)
	}
	report3Data, err := os.ReadFile(outReport3Path)
	if err != nil {
		t.Fatalf("ReadFile out report 3: %v", err)
	}
	var rep3 planpb.PackageReport
	if err := protojson.Unmarshal(report3Data, &rep3); err != nil {
		t.Fatalf("protojson.Unmarshal out report 3: %v", err)
	}
	if len(rep3.Failed) != 0 {
		t.Errorf("rep3.Failed = %+v, want 0 after clear", rep3.Failed)
	}
	foundInMissing := false
	for _, m := range rep3.Missing {
		if m.Name == "submod_6" {
			foundInMissing = true
			break
		}
	}
	if !foundInMissing {
		t.Errorf("cleared request should ask for submod_6 in missing")
	}
	// Body outcome remains in store (not cleared by clearing package).
	if len(rep3.BodyOutcomes) != 1 || rep3.BodyOutcomes[0].Body != "ref_linear" {
		t.Errorf("rep3.BodyOutcomes = %+v, want 1 item with ref_linear retained", rep3.BodyOutcomes)
	}
}

func TestCmdPlanPackagesZeroFailures(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
	})

	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "v1", tmpDir)

	plan := &planpb.PlanFile{
		TreeCommit:  "commit123",
		ArtifactMd5: "artifactmd5",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "ref_linear"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("proto.Marshal plan: %v", err)
	}
	planPath := filepath.Join(tmpDir, "plan.pb")
	if err := os.WriteFile(planPath, planBytes, 0644); err != nil {
		t.Fatalf("WriteFile plan: %v", err)
	}

	cleanReport := &planpb.PackageReport{
		Required: []*planpb.Package{
			{Name: "submod_0", Key: "key0"},
		},
		Present: []*planpb.Package{
			{Name: "submod_0", Key: "key0"},
		},
		Failed: []*planpb.CompileFailure{},
	}

	// 1. Test clean report in JSON
	jsonBytes, err := protojson.Marshal(cleanReport)
	if err != nil {
		t.Fatalf("protojson.Marshal cleanReport: %v", err)
	}
	jsonReportPath := filepath.Join(tmpDir, "clean_report.json")
	if err := os.WriteFile(jsonReportPath, jsonBytes, 0644); err != nil {
		t.Fatalf("WriteFile json clean report: %v", err)
	}

	cmdArgsJSON := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--report", jsonReportPath,
	}
	if err := cmdPlan(ctx, cmdArgsJSON); err != nil {
		t.Fatalf("cmdPlan with clean JSON report failed: %v", err)
	}

	// 2. Test clean report in binary protobuf
	pbBytes, err := proto.Marshal(cleanReport)
	if err != nil {
		t.Fatalf("proto.Marshal cleanReport: %v", err)
	}
	pbReportPath := filepath.Join(tmpDir, "clean_report.pb")
	if err := os.WriteFile(pbReportPath, pbBytes, 0644); err != nil {
		t.Fatalf("WriteFile pb clean report: %v", err)
	}

	cmdArgsPB := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--report", pbReportPath,
	}
	if err := cmdPlan(ctx, cmdArgsPB); err != nil {
		t.Fatalf("cmdPlan with clean binary protobuf report failed: %v", err)
	}
}

func TestCmdPlanPackages(t *testing.T) {
	ctx := t.Context()

	// Error when required flags are missing
	if err := cmdPlan(ctx, []string{"packages"}); err == nil {
		t.Error("expected error for missing flags, got nil")
	}

	tmpDir := t.TempDir()

	// Create test artifact archive with 1 dense segment (submod_0) and 1 host segment (submod_1)
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "out"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "op1", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "op1"}}]},
					{"name": "op2", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "op2"}}]}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	// Create test plan file
	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites: []*planpb.PlanSite{
			{
				Id:   "op1",
				Body: "reference",
			},
			{
				Id:   "op2",
				Body: "reference",
			},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
			{Name: "submod_1", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH, Host: true},
		},
	}
	planBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	planPath := filepath.Join(tmpDir, "plan.pb")
	if err := os.WriteFile(planPath, planBytes, 0644); err != nil {
		t.Fatalf("write plan.pb: %v", err)
	}

	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "v1", tmpDir)

	// Missing --artifact
	if err := cmdPlan(ctx, []string{"packages", "--plan", planPath, "--machine", "g2-standard-48", "--model", "test-model", "--modelstore", ts.URL}); err == nil {
		t.Error("expected error for missing --artifact, got nil")
	}

	// Unknown flag --segments fails
	if err := cmdPlan(ctx, []string{"packages", "--plan", planPath, "--artifact", artRef, "--machine", "g2-standard-48", "--model", "test-model", "--modelstore", ts.URL, "--segments", "submod_0"}); err == nil {
		t.Error("expected error for --segments flag, got nil")
	}

	// Plan site absent from graph is refused, naming the site
	planGhost := proto.Clone(plan).(*planpb.PlanFile)
	planGhost.Sites = append(planGhost.Sites, &planpb.PlanSite{Id: "ghost_site", Body: "reference"})
	planGhostBytes, _ := proto.Marshal(planGhost)
	planGhostPath := filepath.Join(tmpDir, "plan_ghost.pb")
	os.WriteFile(planGhostPath, planGhostBytes, 0644)
	errGhost := cmdPlan(ctx, []string{"packages", "--plan", planGhostPath, "--artifact", artRef, "--machine", "g2-standard-48", "--model", "test-model", "--modelstore", ts.URL})
	if errGhost == nil || !strings.Contains(errGhost.Error(), `"ghost_site"`) {
		t.Errorf("expected error naming ghost_site, got: %v", errGhost)
	}

	// Graph call site absent from plan is refused, naming the site
	planMissing := &planpb.PlanFile{
		ArtifactMd5: plan.ArtifactMd5,
		TreeCommit:  plan.TreeCommit,
		Sites:       []*planpb.PlanSite{{Id: "op1", Body: "reference"}}, // missing op2
		Segments:    plan.Segments,
	}
	planMissingBytes, _ := proto.Marshal(planMissing)
	planMissingPath := filepath.Join(tmpDir, "plan_missing.pb")
	os.WriteFile(planMissingPath, planMissingBytes, 0644)
	errMissing := cmdPlan(ctx, []string{"packages", "--plan", planMissingPath, "--artifact", artRef, "--machine", "g2-standard-48", "--model", "test-model", "--modelstore", ts.URL})
	if errMissing == nil || !strings.Contains(errMissing.Error(), `"op2"`) {
		t.Errorf("expected error naming op2, got: %v", errMissing)
	}

	// Run with --json output
	outJSON := filepath.Join(tmpDir, "packages.json")
	if err := cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "g2-standard-48",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outJSON,
		"--json",
	}); err != nil {
		t.Fatalf("cmdPlan packages failed: %v", err)
	}

	jsonBytes, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read packages.json failed: %v", err)
	}
	var report planpb.PackageReport
	if err := protojson.Unmarshal(jsonBytes, &report); err != nil {
		t.Fatalf("unmarshal package report failed: %v", err)
	}

	// Dense submod_0 produces a package; host op submod_1 produces no package.
	if len(report.Required) != 1 {
		t.Fatalf("got %d required packages, want 1", len(report.Required))
	}
	if report.Required[0].Name != "submod_0" {
		t.Errorf("got package name %q, want submod_0", report.Required[0].Name)
	}
	if len(report.Required[0].Key) != 32 {
		t.Errorf("got key length %d, want 32 (%q)", len(report.Required[0].Key), report.Required[0].Key)
	}
	if len(report.Missing) != 1 || report.Missing[0].Name != "submod_0" {
		t.Errorf("missing packages = %+v, want [submod_0]", report.Missing)
	}

	// Run with text output
	outTxt := filepath.Join(tmpDir, "packages.txt")
	if err := cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "g2-standard-48",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outTxt,
	}); err != nil {
		t.Fatalf("cmdPlan packages text failed: %v", err)
	}
	txtBytes, err := os.ReadFile(outTxt)
	if err != nil {
		t.Fatalf("read packages.txt failed: %v", err)
	}
	txt := string(txtBytes)
	if !strings.Contains(txt, "Required: 1") || !strings.Contains(txt, "submod_0:") {
		t.Errorf("unexpected text output: %s", txt)
	}
}

func TestPassThroughPlan_NoOpsModel(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	// 1. Create artifact with decode.pt2 containing standard torch ops (no vxpu ops)
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "add_1"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "linear_1", "target": "torch.ops.aten.linear.default", "outputs": [{"as_tensor": {"name": "linear_1"}}]},
					{"name": "add_1", "target": "torch.ops.aten.add.Tensor", "outputs": [{"as_tensor": {"name": "add_1"}}]}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	manifestPath := filepath.Join(tmpDir, "manifest.json")
	if err := os.WriteFile(manifestPath, []byte(`{"files": {}, "source": {"repo": "test/model"}}`), 0644); err != nil {
		t.Fatalf("write manifest.json: %v", err)
	}
	bindingPath := filepath.Join(tmpDir, "binding.json")
	if err := os.WriteFile(bindingPath, []byte(`{"bound": {"model.weight": {"shape": [64, 64], "dtype": "float32"}}}`), 0644); err != nil {
		t.Fatalf("write binding.json: %v", err)
	}

	planPath := filepath.Join(tmpDir, "plan.pb")

	// 2. vxpu plan emit emits a pass-through plan (plain torch on every segment, no call site bodies)
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", planPath,
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	plan, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("read plan failed: %v", err)
	}
	if len(plan.Segments) != 1 {
		t.Fatalf("expected 1 segment in plan, got %d", len(plan.Segments))
	}
	if plan.Segments[0].Run != planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH {
		t.Errorf("expected segment run plain torch, got %v", plan.Segments[0].Run)
	}
	if plan.Segments[0].PackageKey != "" {
		t.Errorf("expected empty package key on plain torch segment, got %q", plan.Segments[0].PackageKey)
	}
	if plan.PlanHash == "" {
		t.Fatal("expected emitted pass-through plan to carry plan_hash, got empty")
	}
	wantHash, err := kernelplan.PlanHash(plan)
	if err != nil {
		t.Fatalf("PlanHash failed: %v", err)
	}
	if plan.PlanHash != wantHash {
		t.Errorf("plan.PlanHash = %q, want %q", plan.PlanHash, wantHash)
	}
	// With no packaged segments, changing build provenance (tree commit) must not alter plan identity.
	otherCommitPlanPath := filepath.Join(tmpDir, "plan_other_commit.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", otherCommitPlanPath,
		"--tree-commit", "fedcba9876543210fedcba9876543210fedcba98",
	}); err != nil {
		t.Fatalf("cmdPlan emit with other commit failed: %v", err)
	}
	otherCommitPlan, err := kernelplan.ReadPlan(otherCommitPlanPath)
	if err != nil {
		t.Fatalf("read other commit plan failed: %v", err)
	}
	if otherCommitPlan.PlanHash != plan.PlanHash {
		t.Errorf("pass-through plan_hash changed across tree commits: got %q, want %q", otherCommitPlan.PlanHash, plan.PlanHash)
	}
	if len(plan.Sites) == 0 {
		t.Fatalf("expected macroscopic family sites from binding, got 0")
	}
	// A pass-through plan has no substituted bodies for any site:
	for _, site := range plan.Sites {
		if site.Body != "" {
			t.Errorf("expected body unset in pass-through plan, got %q on site %q", site.Body, site.Id)
		}
		if site.Strategy != "default" && site.Strategy != "" {
			t.Errorf("expected default strategy, got %q on site %q", site.Strategy, site.Id)
		}
	}
	if plan.Knobs != nil {
		t.Errorf("expected no knobs in pass-through plan, got: %v", plan.Knobs)
	}

	// 3. vxpu plan packages asks for nothing (Required: 0, Missing: 0)
	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "v1", tmpDir)

	outTxt := filepath.Join(tmpDir, "packages.txt")
	if err := cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outTxt,
	}); err != nil {
		t.Fatalf("cmdPlan packages failed: %v", err)
	}
	txtBytes, err := os.ReadFile(outTxt)
	if err != nil {
		t.Fatalf("read packages.txt: %v", err)
	}
	txt := string(txtBytes)
	if !strings.Contains(txt, "Required: 0") || !strings.Contains(txt, "Missing: 0") {
		t.Errorf("packages output = %q, want Required: 0 and Missing: 0", txt)
	}

	// 4. vxpu plan tree renders, shows plain torch badge, and states that every call site runs the engine's own implementation
	treePath := filepath.Join(tmpDir, "tree.html")
	if err := cmdPlan(ctx, []string{
		"tree",
		"--artifact", tmpDir,
		"--plan", planPath,
		"--out", treePath,
	}); err != nil {
		t.Fatalf("cmdPlan tree failed: %v", err)
	}
	treeBytes, err := os.ReadFile(treePath)
	if err != nil {
		t.Fatalf("read tree.html: %v", err)
	}
	treeHTMLStr := string(treeBytes)
	if !strings.Contains(treeHTMLStr, "Every call site runs the engine's own implementation") {
		t.Errorf("tree.html should state that every call site runs the engine's own implementation, got:\n%s", treeHTMLStr)
	}
	if !strings.Contains(treeHTMLStr, "plain torch") {
		t.Errorf("tree.html should display plain torch badge, got:\n%s", treeHTMLStr)
	}
	if strings.Contains(treeHTMLStr, "<details class=\"unmatched-section\">") {
		t.Errorf("tree.html should not render the unmatched-section for pass-through plan, got:\n%s", treeHTMLStr)
	}
}

func TestPlanTree_GraphWithOpsDefaultSitesMatchNothing(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	// Create artifact with decode.pt2 containing a custom vxpu op (so graph has our ops)
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "fp8_linear_0"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "fp8_linear_0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "fp8_linear_0"}}]}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}

	manifestPath := filepath.Join(tmpDir, "manifest.json")
	if err := os.WriteFile(manifestPath, []byte(`{"files": {}, "source": {"repo": "test/model"}}`), 0644); err != nil {
		t.Fatalf("write manifest.json: %v", err)
	}
	bindingPath := filepath.Join(tmpDir, "binding.json")
	if err := os.WriteFile(bindingPath, []byte(`{"bound": {}}`), 0644); err != nil {
		t.Fatalf("write binding.json: %v", err)
	}

	// Create a plan with default sites (body unset, no knobs) that match nothing in the graph
	planPath := filepath.Join(tmpDir, "plan.pb")
	md5Val, err := kernelplan.ArtifactMD5(tmpDir)
	if err != nil {
		t.Fatalf("ArtifactMD5: %v", err)
	}
	plan := &planpb.PlanFile{
		ArtifactMd5: md5Val,
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites: []*planpb.PlanSite{
			{
				Id:       "unmatched_site_0",
				Strategy: "default",
			},
		},
	}
	planBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	if err := os.WriteFile(planPath, planBytes, 0644); err != nil {
		t.Fatalf("write plan.pb: %v", err)
	}

	treePath := filepath.Join(tmpDir, "tree.html")
	if err := cmdPlan(ctx, []string{
		"tree",
		"--artifact", tmpDir,
		"--plan", planPath,
		"--out", treePath,
	}); err != nil {
		t.Fatalf("cmdPlan tree failed: %v", err)
	}
	treeBytes, err := os.ReadFile(treePath)
	if err != nil {
		t.Fatalf("read tree.html: %v", err)
	}
	treeHTML := string(treeBytes)

	// Since the graph contains custom ops (fp8_linear_0), this is NOT a pass-through plan:
	// it must not show the pass-through note, but must show the unmatched note and section.
	if strings.Contains(treeHTML, "Pass-through plan: Every call site runs the engine's own implementation") {
		t.Errorf("tree.html should NOT display pass-through note when graph has custom ops:\n%s", treeHTML)
	}
	if !strings.Contains(treeHTML, "The provided plan contains no call-site entries matching the exported graph") {
		t.Errorf("tree.html should display unmatched call sites note when plan sites match nothing:\n%s", treeHTML)
	}
	if !strings.Contains(treeHTML, `<details class="unmatched-section">`) {
		t.Errorf("tree.html should render the unmatched-section when plan sites match nothing:\n%s", treeHTML)
	}
}

func TestCmdPlanPackages_FlagValidation(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
	})
	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "commit123",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "reference"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, _ := proto.Marshal(plan)
	planPath := filepath.Join(tmpDir, "plan.pb")
	_ = os.WriteFile(planPath, planBytes, 0644)

	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "v1", tmpDir)

	// 1. --build without --agent fails
	err := cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--build",
	})
	if err == nil || !strings.Contains(err.Error(), "--agent is required when --build is set") {
		t.Fatalf("expected error about --agent required, got: %v", err)
	}

	// 2. --agent without --build fails
	err = cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--agent", "127.0.0.1:50051",
	})
	if err == nil || !strings.Contains(err.Error(), "--build is required when --agent is set") {
		t.Fatalf("expected error about --build required, got: %v", err)
	}

	// 3. Plan with no tree commit (software version) fails when --build is set
	noVersionPlan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "reference"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	noVersionBytes, _ := proto.Marshal(noVersionPlan)
	noVersionPath := filepath.Join(tmpDir, "no_version_plan.pb")
	_ = os.WriteFile(noVersionPath, noVersionBytes, 0644)

	err = cmdPlan(ctx, []string{
		"packages",
		"--plan", noVersionPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--build",
		"--agent", "127.0.0.1:50051",
	})
	if err == nil || !strings.Contains(err.Error(), "software version is required") {
		t.Fatalf("expected error about missing software version, got: %v", err)
	}

	// 4. Local directory path is refused as --artifact (must be repository:tag).
	err = cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", tmpDir,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
	})
	if err == nil || !strings.Contains(err.Error(), "invalid artifact reference") {
		t.Fatalf("expected invalid artifact reference error for local path, got: %v", err)
	}
}

type mockCompilerService struct {
	pb.UnimplementedCompilerServer
	compileFunc func(req *pb.CompileRequest) (*pb.CompileResponse, error)
}

func (m *mockCompilerService) Compile(ctx context.Context, req *pb.CompileRequest) (*pb.CompileResponse, error) {
	if m.compileFunc != nil {
		return m.compileFunc(req)
	}
	return nil, fmt.Errorf("compile not implemented")
}

type mockAgentServer struct {
	pb.UnimplementedAgentServer
	processes map[string]*pb.ProcessStatus
}

func (m *mockAgentServer) StartProcess(ctx context.Context, req *pb.StartProcessRequest) (*pb.StartProcessResponse, error) {
	st := m.processes[req.Process]
	if st == nil {
		st = &pb.ProcessStatus{
			Id:    req.Id,
			State: "running",
		}
	}
	clone := proto.Clone(st).(*pb.ProcessStatus)
	clone.Id = req.Id
	m.processes[req.Id] = clone
	return &pb.StartProcessResponse{Status: clone}, nil
}

func (m *mockAgentServer) GetProcess(ctx context.Context, req *pb.GetProcessRequest) (*pb.GetProcessResponse, error) {
	st, ok := m.processes[req.Id]
	if !ok {
		return nil, fmt.Errorf("process %s not found", req.Id)
	}
	return &pb.GetProcessResponse{Status: st}, nil
}

func (m *mockAgentServer) StopProcess(ctx context.Context, req *pb.StopProcessRequest) (*pb.StopProcessResponse, error) {
	delete(m.processes, req.Id)
	return &pb.StopProcessResponse{}, nil
}

func startMockCompilerAgent(t *testing.T, srv pb.CompilerServer) (string, func()) {
	t.Helper()
	compLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen compiler: %v", err)
	}
	compGrpc := grpc.NewServer()
	pb.RegisterCompilerServer(compGrpc, srv)
	hs := health.NewServer()
	hs.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(compGrpc, hs)
	go func() {
		_ = compGrpc.Serve(compLis)
	}()

	mockAgent := &mockAgentServer{
		processes: map[string]*pb.ProcessStatus{
			"compiler": {
				Endpoint: compLis.Addr().String(),
				State:    "running",
			},
		},
	}
	agentLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen agent: %v", err)
	}
	agentGrpc := grpc.NewServer()
	pb.RegisterAgentServer(agentGrpc, mockAgent)
	go func() {
		_ = agentGrpc.Serve(agentLis)
	}()

	cleanup := func() {
		agentGrpc.Stop()
		_ = agentLis.Close()
		compGrpc.Stop()
		_ = compLis.Close()
	}
	return agentLis.Addr().String(), cleanup
}

func TestCmdPlanPackages_BuildThroughMockAgent(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
	})
	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "commit123",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "reference"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, _ := proto.Marshal(plan)
	planPath := filepath.Join(tmpDir, "plan.pb")
	_ = os.WriteFile(planPath, planBytes, 0644)

	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "commit123", tmpDir)

	compSrv := &mockCompilerService{
		compileFunc: func(req *pb.CompileRequest) (*pb.CompileResponse, error) {
			if len(req.Packages) == 0 {
				return nil, fmt.Errorf("no packages requested")
			}
			if req.Artifact != artRef {
				return nil, fmt.Errorf("expected artifact reference %q, got: %q", artRef, req.Artifact)
			}
			if req.Plan == nil || req.Plan.GetTreeCommit() != "commit123" {
				return nil, fmt.Errorf("expected plan message with TreeCommit commit123, got: %v", req.Plan)
			}
			var present []*planpb.Package
			for _, p := range req.Packages {
				present = append(present, p)
				fakePutPackageManifest(t, ts.URL, req.Model, p.Key)
			}
			return &pb.CompileResponse{
				Report: &planpb.PackageReport{
					Present:     present,
					MachineKind: req.MachineKind,
				},
			}, nil
		},
	}
	agentAddr, cleanup := startMockCompilerAgent(t, compSrv)
	defer cleanup()

	// 1. Build missing packages using store artifact reference
	outJSON := filepath.Join(tmpDir, "report1.json")
	cmdArgs := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--build",
		"--agent", agentAddr,
		"--out", outJSON,
		"--json",
	}
	if err := cmdPlan(ctx, cmdArgs); err != nil {
		t.Fatalf("cmdPlan packages with --build failed: %v", err)
	}

	data1, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read report1.json: %v", err)
	}
	var rep1 planpb.PackageReport
	if err := protojson.Unmarshal(data1, &rep1); err != nil {
		t.Fatalf("unmarshal report1: %v", err)
	}
	if len(rep1.Present) != 1 || rep1.Present[0].Name != "submod_0" {
		t.Fatalf("expected 1 present package (submod_0), got: %+v", rep1.Present)
	}
	if len(rep1.Missing) != 0 {
		t.Fatalf("expected 0 missing after build, got: %+v", rep1.Missing)
	}

	// 2. Second run without --build reports package present in store
	outJSON2 := filepath.Join(tmpDir, "report2.json")
	cmdArgs2 := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outJSON2,
		"--json",
	}
	if err := cmdPlan(ctx, cmdArgs2); err != nil {
		t.Fatalf("second cmdPlan packages run failed: %v", err)
	}

	data2, err := os.ReadFile(outJSON2)
	if err != nil {
		t.Fatalf("read report2.json: %v", err)
	}
	var rep2 planpb.PackageReport
	if err := protojson.Unmarshal(data2, &rep2); err != nil {
		t.Fatalf("unmarshal report2: %v", err)
	}
	if len(rep2.Present) != 1 || rep2.Present[0].Name != "submod_0" {
		t.Fatalf("expected 1 present package (submod_0) in second run, got: %+v", rep2.Present)
	}
	if len(rep2.Missing) != 0 {
		t.Fatalf("expected 0 missing in second run, got: %+v", rep2.Missing)
	}
}

func TestCmdPlanPackages_MissingPackageFileFails(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
	})
	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "commit123",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "reference"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, _ := proto.Marshal(plan)
	planPath := filepath.Join(tmpDir, "plan.pb")
	_ = os.WriteFile(planPath, planBytes, 0644)

	ts := fakeModelstoreServer(t)
	defer ts.Close()
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "commit123", tmpDir)

	// Compiler reports present, but package was not written to modelstore.
	compSrv := &mockCompilerService{
		compileFunc: func(req *pb.CompileRequest) (*pb.CompileResponse, error) {
			return &pb.CompileResponse{
				Report: &planpb.PackageReport{
					Present:     req.Packages,
					MachineKind: req.MachineKind,
				},
			}, nil
		},
	}
	agentAddr, cleanup := startMockCompilerAgent(t, compSrv)
	defer cleanup()

	cmdArgs := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--build",
		"--agent", agentAddr,
	}
	err := cmdPlan(ctx, cmdArgs)
	if err == nil {
		t.Fatal("expected error when compiler reports present but package is missing from store, got nil")
	}
}

func startRealAgentServer(t *testing.T, root, pyBin string, store *weightstore.Client, customProcs map[string]agent.ProcessConfig, softwareVersion string) (string, func()) {
	t.Helper()
	workDir := t.TempDir()
	procs := customProcs
	if procs == nil {
		procs = agent.DefaultProcesses()
		cfg := procs["compiler"]
		cfg.Binary = pyBin
		if root != "" {
			cfg.Env = append(cfg.Env, "PYTHONPATH="+filepath.Join(root, "vxpu/python"))
			venvBin := filepath.Join(root, ".venv/bin")
			if _, err := os.Stat(venvBin); err == nil {
				cfg.Env = append(cfg.Env, "PATH="+venvBin+":"+os.Getenv("PATH"))
			}
		}
		procs["compiler"] = cfg
	}

	srv := agent.NewServer("", workDir, store)
	srv.Host = "127.0.0.1"
	srv.SoftwareVersion = softwareVersion
	srv.Processes = procs

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}
	grpcSrv := grpc.NewServer()
	pb.RegisterAgentServer(grpcSrv, srv)
	go func() {
		_ = grpcSrv.Serve(lis)
	}()

	cleanup := func() {
		srv.Stop()
		grpcSrv.Stop()
		_ = lis.Close()
	}
	return lis.Addr().String(), cleanup
}

func isCUDASupported(pyBin string) bool {
	cmd := exec.Command(pyBin, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)")
	return cmd.Run() == nil
}

func findPythonBin(root string) string {
	if root != "" {
		venvPy := filepath.Join(root, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			return venvPy
		}
	}
	if py, err := exec.LookPath("python3"); err == nil {
		return py
	}
	return "python3"
}

func TestCmdPlanPackages_BuildThroughService(t *testing.T) {
	root := findGitRoot()
	pyBin := findPythonBin(root)

	if root != "" {
		submoduleCheck := filepath.Join(root, "vxpu/third_party/deepseek_v41/inference/model.py")
		if _, err := os.Stat(submoduleCheck); err != nil {
			if os.Getenv("VXPU_REQUIRE_MODEL_SOURCE") == "1" {
				t.Fatalf("deepseek_v41 model source is absent (expected at %s)", submoduleCheck)
			}
			t.Skipf("deepseek_v41 model source is absent (expected at %s)", submoduleCheck)
		}
	}

	ts := fakeModelstoreServer(t)
	defer ts.Close()

	agentStore := weightstore.New(ts.URL)
	agentAddr, cleanup := startRealAgentServer(t, root, pyBin, agentStore, nil, "test-commit-123")
	defer cleanup()

	ctx := t.Context()
	artDir := getTinyDeepSeekArtifactDir(t)
	tmpDir := t.TempDir()

	targetMachine := "c3d-standard-60"
	if isCUDASupported(pyBin) {
		targetMachine = "g2-standard-48"
	}

	mach, err := machines.Lookup(targetMachine)
	if err != nil {
		t.Fatalf("Lookup machine: %v", err)
	}
	mk, err := kernelplan.MachineKind(mach)
	if err != nil {
		t.Fatalf("MachineKind: %v", err)
	}

	// Emit plan for tiny DeepSeek
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	catPath := filepath.Join("..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
	tablePath := filepath.Join(fixtureDir, "kernel_table.json")
	planPath := filepath.Join(tmpDir, "plan.pb")

	emitArgs := []string{
		"emit",
		"--artifact", artDir,
		"--catalog", catPath,
		"--machine", targetMachine,
		"--out", planPath,
		"--tree-commit", "test-commit-123",
	}
	if targetMachine == "c3d-standard-60" {
		emitArgs = append(emitArgs, "--table", tablePath)
	} else {
		emitArgs = append(emitArgs, "--rows", "1")
	}
	if err := cmdPlan(ctx, emitArgs); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	pf, err := kernelplan.ReadPlan(planPath)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}

	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments failed: %v", err)
	}

	reqPkgs, err := kernelplan.RequiredPackages(pf, segments, mk)
	if err != nil {
		t.Fatalf("RequiredPackages failed: %v", err)
	}

	// Pre-populate all packages except submod_2 in modelstore, so only submod_2 is missing and built.
	for _, p := range reqPkgs {
		if p.Name != "submod_2" {
			fakePutPackageManifest(t, ts.URL, "test-model", p.Key)
		}
	}

	// Upload artifact to modelstore; the command uses tmpDir and the agent uses its own
	// separate temporary root in startRealAgentServer, sharing no directory on disk.
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("test-model"), "test-tag", artDir)

	// 1. First run with --build and --agent
	outJSON := filepath.Join(tmpDir, "report1.json")
	cmdArgs := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", targetMachine,
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--build",
		"--agent", agentAddr,
		"--out", outJSON,
		"--json",
	}
	if err := cmdPlan(ctx, cmdArgs); err != nil {
		t.Fatalf("cmdPlan packages with --build failed: %v", err)
	}

	data1, err := os.ReadFile(outJSON)
	if err != nil {
		t.Fatalf("read report1.json: %v", err)
	}
	var rep1 planpb.PackageReport
	if err := protojson.Unmarshal(data1, &rep1); err != nil {
		t.Fatalf("unmarshal report1: %v", err)
	}
	if len(rep1.Failed) != 0 {
		t.Fatalf("expected 0 failures in first run, got %d (%v)", len(rep1.Failed), rep1.Failed)
	}
	if len(rep1.Missing) != 0 {
		t.Fatalf("expected 0 missing in first run after build, got %d (%v)", len(rep1.Missing), rep1.Missing)
	}

	// 2. Second run without --build: reports all present in store
	outJSON2 := filepath.Join(tmpDir, "report2.json")
	cmdArgs2 := []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", targetMachine,
		"--model", "test-model",
		"--modelstore", ts.URL,
		"--out", outJSON2,
		"--json",
	}
	if err := cmdPlan(ctx, cmdArgs2); err != nil {
		t.Fatalf("second cmdPlan packages failed: %v", err)
	}

	data2, err := os.ReadFile(outJSON2)
	if err != nil {
		t.Fatalf("read report2.json: %v", err)
	}
	var rep2 planpb.PackageReport
	if err := protojson.Unmarshal(data2, &rep2); err != nil {
		t.Fatalf("unmarshal report2: %v", err)
	}
	if len(rep2.Failed) != 0 {
		t.Fatalf("expected 0 failures in second run, got %d (%v)", len(rep2.Failed), rep2.Failed)
	}
	if len(rep2.Missing) != 0 {
		t.Fatalf("expected 0 missing in second run, got %d (%v)", len(rep2.Missing), rep2.Missing)
	}
	if len(rep2.Present) != len(reqPkgs) {
		t.Fatalf("expected all %d packages present in store, got %d", len(reqPkgs), len(rep2.Present))
	}
}

func TestPlanEmit_SegmentRunModes(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	// Create artifact with custom ops and a host op:
	// submod_0: dense (fp8_linear)
	// submod_1: host (moe_experts)
	// submod_2: dense (fp8_linear)
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
			"signature": {
				"input_specs": [
					{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}
				],
				"output_specs": [
					{"user_output": {"arg": {"as_tensor": {"name": "op2"}}}}
				]
			},
			"graph": {
				"nodes": [
					{"name": "op0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "op0"}}]},
					{"name": "moe0", "target": "torch.ops.vxpu.moe_experts.default", "outputs": [{"as_tensor": {"name": "moe0"}}]},
					{"name": "op2", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "op2"}}]}
				]
			}
		}
	}`))
	zw.Close()
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), []byte(`{"files": {}, "source": {"repo": "test/model"}}`), 0644); err != nil {
		t.Fatalf("write manifest.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "binding.json"), []byte(`{"bound": {"model.weight": {"shape": [64, 64], "dtype": "float32"}}}`), 0644); err != nil {
		t.Fatalf("write binding.json: %v", err)
	}

	// 1. Emit on CPU: dense segments get SEGMENT_RUN_PACKAGE, host segment gets SEGMENT_RUN_PLAIN_TORCH
	cpuPlanPath := filepath.Join(tmpDir, "cpu_plan.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", cpuPlanPath,
		"--machine", "c3d-standard-60",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit CPU failed: %v", err)
	}
	cpuPlan, err := kernelplan.ReadPlan(cpuPlanPath)
	if err != nil {
		t.Fatalf("read CPU plan: %v", err)
	}
	if len(cpuPlan.Segments) != 3 {
		t.Fatalf("got %d segments, want 3", len(cpuPlan.Segments))
	}
	if cpuPlan.Segments[0].Run != planpb.SegmentRun_SEGMENT_RUN_PACKAGE {
		t.Errorf("submod_0 on CPU: run = %v, want SEGMENT_RUN_PACKAGE", cpuPlan.Segments[0].Run)
	}
	if cpuPlan.Segments[1].Run != planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH {
		t.Errorf("submod_1 on CPU: run = %v, want SEGMENT_RUN_PLAIN_TORCH", cpuPlan.Segments[1].Run)
	}
	if cpuPlan.Segments[2].Run != planpb.SegmentRun_SEGMENT_RUN_PACKAGE {
		t.Errorf("submod_2 on CPU: run = %v, want SEGMENT_RUN_PACKAGE", cpuPlan.Segments[2].Run)
	}

	cpuPkgs, err := kernelplan.RequiredPackages(cpuPlan, cpuPlan.Segments, "cpu:amd epyc 9b14:avx512")
	if err != nil {
		t.Fatalf("RequiredPackages on CPU plan: %v", err)
	}
	if len(cpuPkgs) != 2 {
		t.Fatalf("expected 2 packages for CPU plan, got %d", len(cpuPkgs))
	}
	if cpuPlan.Segments[0].PackageKey != cpuPkgs[0].Key {
		t.Errorf("submod_0 on CPU: package_key = %q, want %q", cpuPlan.Segments[0].PackageKey, cpuPkgs[0].Key)
	}
	if cpuPlan.Segments[1].PackageKey != "" {
		t.Errorf("submod_1 on CPU (plain torch): package_key = %q, want empty", cpuPlan.Segments[1].PackageKey)
	}
	if cpuPlan.Segments[2].PackageKey != cpuPkgs[1].Key {
		t.Errorf("submod_2 on CPU: package_key = %q, want %q", cpuPlan.Segments[2].PackageKey, cpuPkgs[1].Key)
	}

	// 2. Emit on GPU (L4): dense segments get SEGMENT_RUN_CAPTURED, host segment gets SEGMENT_RUN_PLAIN_TORCH
	gpuPlanPath := filepath.Join(tmpDir, "gpu_plan.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", gpuPlanPath,
		"--machine", "g2-standard-48",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit GPU failed: %v", err)
	}
	gpuPlan, err := kernelplan.ReadPlan(gpuPlanPath)
	if err != nil {
		t.Fatalf("read GPU plan: %v", err)
	}
	if len(gpuPlan.Segments) != 3 {
		t.Fatalf("got %d segments, want 3", len(gpuPlan.Segments))
	}
	if gpuPlan.Segments[0].Run != planpb.SegmentRun_SEGMENT_RUN_CAPTURED {
		t.Errorf("submod_0 on GPU: run = %v, want SEGMENT_RUN_CAPTURED", gpuPlan.Segments[0].Run)
	}
	if gpuPlan.Segments[1].Run != planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH {
		t.Errorf("submod_1 on GPU: run = %v, want SEGMENT_RUN_PLAIN_TORCH", gpuPlan.Segments[1].Run)
	}
	if gpuPlan.Segments[2].Run != planpb.SegmentRun_SEGMENT_RUN_CAPTURED {
		t.Errorf("submod_2 on GPU: run = %v, want SEGMENT_RUN_CAPTURED", gpuPlan.Segments[2].Run)
	}

	gpuPkgs, err := kernelplan.RequiredPackages(gpuPlan, gpuPlan.Segments, "gpu:nvidia l4")
	if err != nil {
		t.Fatalf("RequiredPackages on GPU plan: %v", err)
	}
	if len(gpuPkgs) != 2 {
		t.Fatalf("expected 2 packages for GPU plan, got %d", len(gpuPkgs))
	}
	if gpuPlan.Segments[0].PackageKey != gpuPkgs[0].Key {
		t.Errorf("submod_0 on GPU: package_key = %q, want %q", gpuPlan.Segments[0].PackageKey, gpuPkgs[0].Key)
	}
	if gpuPlan.Segments[1].PackageKey != "" {
		t.Errorf("submod_1 on GPU (plain torch): package_key = %q, want empty", gpuPlan.Segments[1].PackageKey)
	}
	if gpuPlan.Segments[2].PackageKey != gpuPkgs[1].Key {
		t.Errorf("submod_2 on GPU: package_key = %q, want %q", gpuPlan.Segments[2].PackageKey, gpuPkgs[1].Key)
	}
	if gpuPlan.Segments[0].PackageKey == cpuPlan.Segments[0].PackageKey {
		t.Errorf("submod_0 GPU package_key %q unexpectedly equals CPU package_key", gpuPlan.Segments[0].PackageKey)
	}

	// 3. Plan identity through emit: changes when a key changes and not otherwise.
	cpuPlanRepeatPath := filepath.Join(tmpDir, "cpu_plan_repeat.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", cpuPlanRepeatPath,
		"--machine", "c3d-standard-60",
		"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
	}); err != nil {
		t.Fatalf("cmdPlan emit CPU repeat failed: %v", err)
	}
	cpuPlanRepeat, err := kernelplan.ReadPlan(cpuPlanRepeatPath)
	if err != nil {
		t.Fatalf("read repeated CPU plan: %v", err)
	}
	if cpuPlanRepeat.PlanHash == "" || cpuPlanRepeat.PlanHash != cpuPlan.PlanHash {
		t.Errorf("repeated emit plan_hash = %q, want %q", cpuPlanRepeat.PlanHash, cpuPlan.PlanHash)
	}

	// Execution measurements and pre-existing PlanHash do not change plan identity.
	cpuPlanMeasured := proto.Clone(cpuPlan).(*planpb.PlanFile)
	cpuPlanMeasured.Stats = &planpb.PlanExecutionStats{PlanHash: proto.String("ignored")}
	cpuPlanMeasured.Segments[0].TotalMeasuredTimeUs = proto.Float64(42.5)
	hMeasured, err := kernelplan.PlanHash(cpuPlanMeasured)
	if err != nil {
		t.Fatalf("PlanHash measured: %v", err)
	}
	if hMeasured != cpuPlan.PlanHash {
		t.Errorf("measured plan_hash = %q, want %q", hMeasured, cpuPlan.PlanHash)
	}

	// Changing a single packaged segment's key changes plan identity.
	cpuPlanMutatedKey := proto.Clone(cpuPlan).(*planpb.PlanFile)
	cpuPlanMutatedKey.Segments[0].PackageKey = "ffeeddccbbaa99887766554433221100"
	hMutatedKey, err := kernelplan.PlanHash(cpuPlanMutatedKey)
	if err != nil {
		t.Fatalf("PlanHash mutated key: %v", err)
	}
	if hMutatedKey == cpuPlan.PlanHash {
		t.Errorf("mutating segment package_key did not change plan_hash: %s", hMutatedKey)
	}

	// Emitting for a different software version changes package keys and therefore plan_hash.
	cpuPlanOtherCommitPath := filepath.Join(tmpDir, "cpu_plan_other_commit.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", cpuPlanOtherCommitPath,
		"--machine", "c3d-standard-60",
		"--tree-commit", "fedcba9876543210fedcba9876543210fedcba98",
	}); err != nil {
		t.Fatalf("cmdPlan emit CPU other commit failed: %v", err)
	}
	cpuPlanOtherCommit, err := kernelplan.ReadPlan(cpuPlanOtherCommitPath)
	if err != nil {
		t.Fatalf("read other commit CPU plan: %v", err)
	}
	if cpuPlanOtherCommit.Segments[0].PackageKey == cpuPlan.Segments[0].PackageKey {
		t.Errorf("changing tree commit did not change packaged segment key: %q", cpuPlan.Segments[0].PackageKey)
	}
	if cpuPlanOtherCommit.PlanHash == cpuPlan.PlanHash {
		t.Errorf("changing packaged segment keys via tree commit did not change plan_hash: %q", cpuPlan.PlanHash)
	}

	// 4. Tree view displays package and captured badges along with package keys (HTML and JSON)
	treeCPUPath := filepath.Join(tmpDir, "tree_cpu.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--plan", cpuPlanPath, "--out", treeCPUPath}); err != nil {
		t.Fatalf("cmdPlan tree CPU: %v", err)
	}
	cpuHTML, _ := os.ReadFile(treeCPUPath)
	if !strings.Contains(string(cpuHTML), "package") {
		t.Errorf("tree_cpu.html should show package badge, got:\n%s", string(cpuHTML))
	}
	for _, pkg := range cpuPkgs {
		if !strings.Contains(string(cpuHTML), pkg.Key) {
			t.Errorf("tree_cpu.html missing package key %q for %s", pkg.Key, pkg.Name)
		}
	}

	treeGPUPath := filepath.Join(tmpDir, "tree_gpu.html")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--plan", gpuPlanPath, "--out", treeGPUPath}); err != nil {
		t.Fatalf("cmdPlan tree GPU: %v", err)
	}
	gpuHTML, _ := os.ReadFile(treeGPUPath)
	if !strings.Contains(string(gpuHTML), "captured") {
		t.Errorf("tree_gpu.html should show captured badge, got:\n%s", string(gpuHTML))
	}
	for _, pkg := range gpuPkgs {
		if !strings.Contains(string(gpuHTML), pkg.Key) {
			t.Errorf("tree_gpu.html missing package key %q for %s", pkg.Key, pkg.Name)
		}
	}

	treeGPUJSONPath := filepath.Join(tmpDir, "tree_gpu.json")
	if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--plan", gpuPlanPath, "--json", "--out", treeGPUJSONPath}); err != nil {
		t.Fatalf("cmdPlan tree GPU JSON: %v", err)
	}
	treeJSONBytes, err := os.ReadFile(treeGPUJSONPath)
	if err != nil {
		t.Fatalf("read tree_gpu.json: %v", err)
	}
	var treeMsg planpb.ModelTree
	if err := protojson.Unmarshal(treeJSONBytes, &treeMsg); err != nil {
		t.Fatalf("unmarshal tree_gpu.json: %v", err)
	}
	if len(treeMsg.Segments) != 3 {
		t.Fatalf("tree_gpu.json segments = %d, want 3", len(treeMsg.Segments))
	}
	if treeMsg.Segments[0].PackageKey != gpuPkgs[0].Key || treeMsg.Segments[1].PackageKey != "" || treeMsg.Segments[2].PackageKey != gpuPkgs[1].Key {
		t.Errorf("tree_gpu.json segment package keys = [%q, %q, %q], want [%q, \"\", %q]",
			treeMsg.Segments[0].PackageKey, treeMsg.Segments[1].PackageKey, treeMsg.Segments[2].PackageKey,
			gpuPkgs[0].Key, gpuPkgs[1].Key)
	}

	// 5. A plan with a package key on a segment that does not run from a package is refused
	// (tested both in memory and via CLI commands).
	for _, tc := range []struct {
		name string
		run  planpb.SegmentRun
	}{
		{"plain_torch", planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH},
		{"jit", planpb.SegmentRun_SEGMENT_RUN_JIT},
		{"unspecified", planpb.SegmentRun_SEGMENT_RUN_UNSPECIFIED},
	} {
		badPlan := proto.Clone(cpuPlan).(*planpb.PlanFile)
		badPlan.Segments[1].Run = tc.run
		badPlan.Segments[1].PackageKey = "0123456789abcdef0123456789abcdef"

		segs, err := kernelplan.ReadSegments(tmpDir)
		if err != nil {
			t.Fatalf("ReadSegments (%s): %v", tc.name, err)
		}
		if _, err := kernelplan.RequiredPackages(badPlan, segs, "cpu:amd epyc 9b14:avx512"); err == nil {
			t.Errorf("%s: RequiredPackages accepted key on non-packaged segment", tc.name)
		}
		if err := kernelplan.JoinPlan(segs, badPlan); err == nil {
			t.Errorf("%s: JoinPlan accepted key on non-packaged segment", tc.name)
		}
		if err := kernelplan.EnsurePlanIdentity(badPlan); err == nil {
			t.Errorf("%s: EnsurePlanIdentity accepted key on non-packaged segment", tc.name)
		}

		badPlanBytes, err := kernelplan.MarshalAsJSON(badPlan)
		if err != nil {
			t.Fatalf("MarshalAsJSON badPlan (%s): %v", tc.name, err)
		}
		badPlanPath := filepath.Join(tmpDir, "bad_plan_"+tc.name+".json")
		if err := os.WriteFile(badPlanPath, badPlanBytes, 0o644); err != nil {
			t.Fatalf("write bad_plan (%s): %v", tc.name, err)
		}
		if _, err := kernelplan.ReadPlan(badPlanPath); err == nil {
			t.Errorf("%s: ReadPlan accepted key on non-packaged segment", tc.name)
		}
		if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--plan", badPlanPath, "--out", filepath.Join(tmpDir, "bad_tree_"+tc.name+".html")}); err == nil {
			t.Errorf("%s: cmdPlan tree accepted key on non-packaged segment", tc.name)
		}
	}
}

func TestCmdPlanPackages_StagingFailureNamesArtifact(t *testing.T) {
	root := findGitRoot()
	pyBin := findPythonBin(root)

	// Command sees the artifact in cmdStore, while the agent points at a store that does not
	// hold the artifact, exercising the agent staging failure path end-to-end.
	cmdStore := fakeModelstoreServer(t)
	defer cmdStore.Close()
	agentFailingStore := httptest.NewServer(http.NotFoundHandler())
	defer agentFailingStore.Close()

	agentAddr, cleanup := startRealAgentServer(t, root, pyBin, weightstore.New(agentFailingStore.URL), nil, "commit123")
	defer cleanup()

	ctx := t.Context()
	tmpDir := t.TempDir()
	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "s0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "s0"}}]}`,
	})
	artRef := fakePutArtifact(t, cmdStore.URL, weightstore.ArtifactRepository("test-model"), "missing-on-agent", tmpDir)

	plan := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "commit123",
		Sites: []*planpb.PlanSite{
			{Id: "s0", Body: "reference"},
		},
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE},
		},
	}
	planBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	planPath := filepath.Join(tmpDir, "plan.pb")
	if err := os.WriteFile(planPath, planBytes, 0o644); err != nil {
		t.Fatalf("write plan: %v", err)
	}

	err = cmdPlan(ctx, []string{
		"packages",
		"--plan", planPath,
		"--artifact", artRef,
		"--machine", "c3d-standard-60",
		"--model", "test-model",
		"--modelstore", cmdStore.URL,
		"--build",
		"--agent", agentAddr,
	})
	if err == nil {
		t.Fatal("expected staging failure error, got nil")
	}
	if !strings.Contains(err.Error(), artRef) || !strings.Contains(err.Error(), "staging artifact") {
		t.Errorf("expected error to report staging failure naming %q, got: %v", artRef, err)
	}
}

func TestCmdPlanEmit_GoalLlama(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	// Build a Llama-like fixture with no custom vxpu operations.
	createArtifactWithNodes(t, tmpDir, []string{
		`{"name": "linear_q", "target": "torch.ops.aten.linear.default", "inputs": [{"arg": {"as_tensor": {"name": "input_ids"}}}], "outputs": [{"as_tensor": {"name": "q"}}], "metadata": {"nn_module_stack": "L['self'],model.layers.0.self_attn.q_proj,Linear"}}`,
		`{"name": "linear_k", "target": "torch.ops.aten.linear.default", "inputs": [{"arg": {"as_tensor": {"name": "q"}}}], "outputs": [{"as_tensor": {"name": "k"}}], "metadata": {"nn_module_stack": "L['self'],model.layers.0.self_attn.k_proj,Linear"}}`,
		`{"name": "add_res", "target": "torch.ops.aten.add.Tensor", "inputs": [{"arg": {"as_tensor": {"name": "k"}}}], "outputs": [{"as_tensor": {"name": "out"}}], "metadata": {"nn_module_stack": "L['self'],model.layers.0,LlamaDecoderLayer"}}`,
	})
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), []byte(`{"files": {}, "source": {"repo": "trl-internal-testing/tiny-random-LlamaForCausalLM"}}`), 0644); err != nil {
		t.Fatalf("write manifest.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "binding.json"), []byte(`{"bound": {"model.layers.0.self_attn.q_proj.weight": {"shape": [64, 64], "dtype": "float32"}}}`), 0644); err != nil {
		t.Fatalf("write binding.json: %v", err)
	}

	const commit = "0123456789abcdef0123456789abcdef01234567"
	emitPlan := func(name, machine, goal string) (*planpb.PlanFile, string) {
		t.Helper()
		planPath := filepath.Join(tmpDir, name+".pb")
		args := []string{
			"emit",
			"--artifact", tmpDir,
			"--out", planPath,
			"--machine", machine,
			"--tree-commit", commit,
		}
		if goal != "" {
			args = append(args, "--goal", goal)
		}
		if err := cmdPlan(ctx, args); err != nil {
			t.Fatalf("cmdPlan emit (%s): %v", name, err)
		}
		p, err := kernelplan.ReadPlan(planPath)
		if err != nil {
			t.Fatalf("ReadPlan (%s): %v", name, err)
		}
		if p.PlanHash == "" {
			t.Fatalf("emitted plan (%s) has empty plan_hash", name)
		}
		return p, planPath
	}

	// Emit all 4 combinations: {pass-through, speed} x {CPU, GPU}.
	ptCPU, ptCPUPath := emitPlan("pt_cpu", "c3d-standard-60", "pass-through")
	speedCPU, speedCPUPath := emitPlan("speed_cpu", "c3d-standard-60", "speed")
	ptGPU, _ := emitPlan("pt_gpu", "g2-standard-48", "pass-through")
	speedGPU, speedGPUPath := emitPlan("speed_gpu", "g2-standard-48", "speed")

	// Verify goals recorded on plans.
	if ptCPU.Goal != planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH {
		t.Errorf("ptCPU.Goal = %v, want PLAN_GOAL_PASS_THROUGH", ptCPU.Goal)
	}
	if speedCPU.Goal != planpb.PlanGoal_PLAN_GOAL_SPEED {
		t.Errorf("speedCPU.Goal = %v, want PLAN_GOAL_SPEED", speedCPU.Goal)
	}
	if ptGPU.Goal != planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH {
		t.Errorf("ptGPU.Goal = %v, want PLAN_GOAL_PASS_THROUGH", ptGPU.Goal)
	}
	if speedGPU.Goal != planpb.PlanGoal_PLAN_GOAL_SPEED {
		t.Errorf("speedGPU.Goal = %v, want PLAN_GOAL_SPEED", speedGPU.Goal)
	}

	// Verify segment runtimes.
	checkRuns := func(label string, p *planpb.PlanFile, want planpb.SegmentRun) int {
		t.Helper()
		nonHost := 0
		for _, seg := range p.Segments {
			if seg.Host {
				if seg.Run != planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH {
					t.Errorf("%s host segment %s run = %v, want SEGMENT_RUN_PLAIN_TORCH", label, seg.Name, seg.Run)
				}
				continue
			}
			nonHost++
			if seg.Run != want {
				t.Errorf("%s segment %s run = %v, want %v", label, seg.Name, seg.Run, want)
			}
		}
		if nonHost == 0 {
			t.Fatalf("%s expected at least one non-host segment", label)
		}
		return nonHost
	}
	checkRuns("ptCPU", ptCPU, planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH)
	nonHostCPU := checkRuns("speedCPU", speedCPU, planpb.SegmentRun_SEGMENT_RUN_PACKAGE)
	checkRuns("ptGPU", ptGPU, planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH)
	nonHostGPU := checkRuns("speedGPU", speedGPU, planpb.SegmentRun_SEGMENT_RUN_CAPTURED)

	// Verify required packages: none for pass-through, one per non-host segment for speed.
	for _, tc := range []struct {
		name        string
		plan        *planpb.PlanFile
		machineKind string
		wantPkgs    int
	}{
		{"ptCPU", ptCPU, "cpu:amd epyc 9b14:avx512", 0},
		{"speedCPU", speedCPU, "cpu:amd epyc 9b14:avx512", nonHostCPU},
		{"ptGPU", ptGPU, "gpu:nvidia l4", 0},
		{"speedGPU", speedGPU, "gpu:nvidia l4", nonHostGPU},
	} {
		pkgs, err := kernelplan.RequiredPackages(tc.plan, tc.plan.Segments, tc.machineKind)
		if err != nil {
			t.Fatalf("RequiredPackages (%s): %v", tc.name, err)
		}
		if len(pkgs) != tc.wantPkgs {
			t.Errorf("RequiredPackages (%s) = %d, want %d", tc.name, len(pkgs), tc.wantPkgs)
		}
	}

	// Verify emitted plan_hash identities differ between pass-through and speed, and between CPU and GPU speed.
	if ptCPU.PlanHash == speedCPU.PlanHash {
		t.Errorf("ptCPU and speedCPU have identical plan_hash %s", ptCPU.PlanHash)
	}
	if ptGPU.PlanHash == speedGPU.PlanHash {
		t.Errorf("ptGPU and speedGPU have identical plan_hash %s", ptGPU.PlanHash)
	}
	if speedCPU.PlanHash == speedGPU.PlanHash {
		t.Errorf("speedCPU and speedGPU have identical plan_hash %s", speedCPU.PlanHash)
	}

	// Verify tree page displays the plan goal for pass-through and speed,
	// and does not render bogus unmatched plan sites or the obsolete --sites note
	// for a model with no custom vxpu operations.
	for _, tc := range []struct {
		name            string
		planPath        string
		wantGoal        string
		wantRun         string
		wantPassThrough bool
	}{
		{"pt_cpu", ptCPUPath, "pass-through", "plain torch", true},
		{"speed_cpu", speedCPUPath, "speed", "package", false},
		{"speed_gpu", speedGPUPath, "speed", "captured", false},
	} {
		treePath := filepath.Join(tmpDir, "tree_"+tc.name+".html")
		if err := cmdPlan(ctx, []string{"tree", "--artifact", tmpDir, "--plan", tc.planPath, "--out", treePath}); err != nil {
			t.Fatalf("cmdPlan tree (%s): %v", tc.name, err)
		}
		htmlBytes, err := os.ReadFile(treePath)
		if err != nil {
			t.Fatalf("read tree html (%s): %v", tc.name, err)
		}
		html := string(htmlBytes)
		if !strings.Contains(html, "Goal:") || !strings.Contains(html, tc.wantGoal) {
			t.Errorf("tree (%s) missing Goal %q", tc.name, tc.wantGoal)
		}
		if !strings.Contains(html, tc.wantRun) {
			t.Errorf("tree (%s) missing segment run badge %q", tc.name, tc.wantRun)
		}
		if strings.Contains(html, "Unmatched Plan Sites") {
			t.Errorf("tree (%s) should not show Unmatched Plan Sites for a model with no custom ops", tc.name)
		}
		if strings.Contains(html, "contains no call-site entries matching the exported graph") {
			t.Errorf("tree (%s) should not show unmatched call-sites warning note for a model with no custom ops", tc.name)
		}
		if strings.Contains(html, `<details class="unmatched-section">`) {
			t.Errorf("tree (%s) should not render unmatched-section for a model with no custom ops", tc.name)
		}
		if gotPT := strings.Contains(html, "Pass-through plan:"); gotPT != tc.wantPassThrough {
			t.Errorf("tree (%s) Pass-through plan note present = %v, want %v", tc.name, gotPT, tc.wantPassThrough)
		}
	}

	// Verify invalid goal is refused.
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--out", filepath.Join(tmpDir, "bad.pb"),
		"--goal", "bogus",
		"--tree-commit", commit,
	}); err == nil {
		t.Fatal("cmdPlan emit accepted invalid --goal, want error")
	}
}

func TestCmdPlanEmit_DefaultGoalMatchesPassThroughAndOmitsGoal(t *testing.T) {
	ctx := t.Context()
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")

	t.Run("default_omits_goal", func(t *testing.T) {
		tmpDir := t.TempDir()
		defaultOut := filepath.Join(tmpDir, "default.json")
		ptOut := filepath.Join(tmpDir, "pt.json")
		for _, args := range [][]string{
			{"emit", "--artifact", fixtureDir, "--out", defaultOut, "--json", "--tree-commit", "0123456789abcdef0123456789abcdef01234567"},
			{"emit", "--artifact", fixtureDir, "--goal", "pass-through", "--out", ptOut, "--json", "--tree-commit", "0123456789abcdef0123456789abcdef01234567"},
		} {
			if err := cmdPlan(ctx, args); err != nil {
				t.Fatalf("cmdPlan emit: %v", err)
			}
		}
		defBytes, err := os.ReadFile(defaultOut)
		if err != nil {
			t.Fatalf("read default.json: %v", err)
		}
		ptBytes, err := os.ReadFile(ptOut)
		if err != nil {
			t.Fatalf("read pt.json: %v", err)
		}
		if !bytes.Equal(defBytes, ptBytes) {
			t.Errorf("default emit differs from explicit --goal=pass-through")
		}
		if strings.Contains(string(defBytes), `"goal"`) {
			t.Errorf("default emit should omit zero-value goal field to stay byte-identical to main")
		}
	})

	t.Run("tiny_deepseek", func(t *testing.T) {
		artifactDir := getTinyDeepSeekArtifactDir(t)
		tmpDir := t.TempDir()
		catPath := filepath.Join("..", "..", "pkg", "kernelplan", "catalog", "kernel_catalog.json")
		tablePath := filepath.Join(fixtureDir, "kernel_table.json")
		defaultOut := filepath.Join(tmpDir, "deepseek_default.json")
		ptOut := filepath.Join(tmpDir, "deepseek_pt.json")
		for _, args := range [][]string{
			{"emit", "--artifact", artifactDir, "--catalog", catPath, "--table", tablePath, "--out", defaultOut, "--json", "--tree-commit", "0123456789abcdef0123456789abcdef01234567"},
			{"emit", "--artifact", artifactDir, "--catalog", catPath, "--table", tablePath, "--goal", "pass-through", "--out", ptOut, "--json", "--tree-commit", "0123456789abcdef0123456789abcdef01234567"},
		} {
			if err := cmdPlan(ctx, args); err != nil {
				t.Fatalf("cmdPlan emit tiny DeepSeek: %v", err)
			}
		}
		defBytes, err := os.ReadFile(defaultOut)
		if err != nil {
			t.Fatalf("read defaultOut: %v", err)
		}
		ptBytes, err := os.ReadFile(ptOut)
		if err != nil {
			t.Fatalf("read ptOut: %v", err)
		}
		if !bytes.Equal(defBytes, ptBytes) {
			t.Errorf("tiny DeepSeek default emit differs from explicit --goal=pass-through")
		}
		if strings.Contains(string(defBytes), `"goal"`) {
			t.Errorf("tiny DeepSeek default emit should omit zero-value goal field to stay byte-identical to main")
		}
	})
}

func countingModelstoreServer(t *testing.T, reqCount *atomic.Int64) *httptest.Server {
	t.Helper()
	blobs := map[string][]byte{}
	manifests := map[string][]byte{}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqCount.Add(1)
		path := strings.TrimPrefix(r.URL.Path, "/v2/")
		switch {
		case r.Method == http.MethodPost && strings.Contains(path, "/blobs/uploads/"):
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			blobs[digest] = data
			w.WriteHeader(http.StatusCreated)
		case r.Method == http.MethodGet && strings.Contains(path, "/blobs/"):
			digest := path[strings.LastIndex(path, "/")+1:]
			if data, ok := blobs[digest]; ok {
				_, _ = w.Write(data)
			} else {
				http.NotFound(w, r)
			}
		case strings.Contains(path, "/manifests/"):
			i := strings.Index(path, "/manifests/")
			key := path[:i] + ":" + path[i+len("/manifests/"):]
			if r.Method == http.MethodPut {
				data, _ := io.ReadAll(r.Body)
				manifests[key] = data
				w.WriteHeader(http.StatusCreated)
			} else if data, ok := manifests[key]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
		case strings.HasSuffix(path, "/tags/list"):
			name := strings.TrimSuffix(path, "/tags/list")
			var tags []string
			for k := range manifests {
				if n, tag, _ := strings.Cut(k, ":"); n == name {
					tags = append(tags, tag)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"name": name, "tags": tags})
		default:
			http.NotFound(w, r)
		}
	}))
}

func createPlanEmitTestArtifact(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	for _, name := range []string{"manifest.json", "binding.json"} {
		data, err := os.ReadFile(filepath.Join(fixtureDir, name))
		if err != nil {
			t.Fatalf("read fixture %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	createArtifactWithNodes(t, dir, []string{
		`{"name": "fp8_linear_0", "target": "torch.ops.vxpu.fp8_linear.default", "outputs": [{"as_tensor": {"name": "fp8_linear_0"}}]}`,
		`{"name": "rotary_0", "target": "torch.ops.vxpu.rotary.default", "outputs": [{"as_tensor": {"name": "rotary_0"}}]}`,
	})
	return dir
}

func TestCmdPlanEmit_ByteIdenticalAndStoreRequestCount(t *testing.T) {
	ctx := t.Context()
	artDir := createPlanEmitTestArtifact(t)
	tmpDir := t.TempDir()

	outBaseline := filepath.Join(tmpDir, "baseline_plan.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outBaseline,
		"--tree-commit", "commit1",
		"--rows", "1",
	}); err != nil {
		t.Fatalf("emit baseline plan failed: %v", err)
	}

	var reqCount atomic.Int64
	ts := countingModelstoreServer(t, &reqCount)
	defer ts.Close()

	outWithStore := filepath.Join(tmpDir, "with_store_plan.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outWithStore,
		"--tree-commit", "commit1",
		"--rows", "1",
		"--machine", "c3d-standard-60",
		"--modelstore", ts.URL,
	}); err != nil {
		t.Fatalf("emit plan with store failed: %v", err)
	}

	// Exactly 1 request to the store during emit (listing tags in vxpu-body-outcomes).
	if got := reqCount.Load(); got != 1 {
		t.Errorf("got %d store requests during emit, want 1", got)
	}

	b1, err := os.ReadFile(outBaseline)
	if err != nil {
		t.Fatalf("read baseline plan: %v", err)
	}
	b2, err := os.ReadFile(outWithStore)
	if err != nil {
		t.Fatalf("read with_store plan: %v", err)
	}
	if !bytes.Equal(b1, b2) {
		t.Errorf("emitted plans differ between baseline and store with no records")
	}

	baselinePlan, err := kernelplan.ReadPlan(outBaseline)
	if err != nil {
		t.Fatalf("ReadPlan baseline: %v", err)
	}
	if baselinePlan.PlanHash == "" {
		t.Fatal("baseline plan missing plan_hash")
	}
	withStorePlan, err := kernelplan.ReadPlan(outWithStore)
	if err != nil {
		t.Fatalf("ReadPlan with_store: %v", err)
	}
	if withStorePlan.PlanHash != baselinePlan.PlanHash {
		t.Errorf("with_store plan_hash = %q, want baseline %q", withStorePlan.PlanHash, baselinePlan.PlanHash)
	}

	// Recording a body failure changes the chosen body and must produce a different plan identity.
	mk, err := kernelplan.MachineKindFromName("c3d-standard-60")
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}
	store := weightstore.New(ts.URL)
	if err := store.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{
		Op:              "vxpu::fp8_linear",
		Body:            "reference",
		MachineKind:     mk,
		SoftwareVersion: "commit1",
		Compiled:        false,
		Reason:          "compiler cast error",
	}); err != nil {
		t.Fatalf("RecordBodyCompileOutcome: %v", err)
	}

	outAvoided := filepath.Join(tmpDir, "avoided_plan.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outAvoided,
		"--tree-commit", "commit1",
		"--rows", "1",
		"--machine", "c3d-standard-60",
		"--modelstore", ts.URL,
	}); err != nil {
		t.Fatalf("emit plan with recorded failure failed: %v", err)
	}
	avoidedPlan, err := kernelplan.ReadPlan(outAvoided)
	if err != nil {
		t.Fatalf("ReadPlan avoided: %v", err)
	}
	if avoidedPlan.PlanHash == "" {
		t.Fatal("avoided plan missing plan_hash")
	}
	if avoidedPlan.PlanHash == baselinePlan.PlanHash {
		t.Errorf("avoided plan_hash = %q matches baseline %q; want distinct identity", avoidedPlan.PlanHash, baselinePlan.PlanHash)
	}
	if err := kernelplan.EnsurePlanIdentity(avoidedPlan); err != nil {
		t.Errorf("EnsurePlanIdentity(avoidedPlan): %v", err)
	}
}

func TestCmdPlanEmit_PlanAroundFailedBody(t *testing.T) {
	ctx := t.Context()
	artDir := createPlanEmitTestArtifact(t)
	tmpDir := t.TempDir()

	ts := fakeModelstoreServer(t)
	defer ts.Close()

	targetMachine := "c3d-standard-60"
	mk, err := kernelplan.MachineKindFromName(targetMachine)
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}

	treeCommit := "commit1"
	outBaseline := filepath.Join(tmpDir, "baseline.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outBaseline,
		"--tree-commit", treeCommit,
		"--rows", "1",
		"--machine", targetMachine,
	}); err != nil {
		t.Fatalf("emit baseline plan failed: %v", err)
	}

	store := weightstore.New(ts.URL)
	// Record body outcome as not compiling for fp8_linear reference body on CPU.
	if err := store.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{
		Op:              "vxpu::fp8_linear",
		Body:            "reference",
		MachineKind:     mk,
		SoftwareVersion: treeCommit,
		Compiled:        false,
		Reason:          "compiler cast error float8_e8m0fnu",
	}); err != nil {
		t.Fatalf("RecordBodyCompileOutcome: %v", err)
	}

	outPlan := filepath.Join(tmpDir, "plan_avoided.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outPlan,
		"--tree-commit", treeCommit,
		"--rows", "1",
		"--machine", targetMachine,
		"--modelstore", ts.URL,
	}); err != nil {
		t.Fatalf("emit plan failed: %v", err)
	}

	pf, err := kernelplan.ReadPlan(outPlan)
	if err != nil {
		t.Fatalf("ReadPlan failed: %v", err)
	}

	var fp8LinearSites []*planpb.PlanSite
	for _, ps := range pf.Sites {
		if strings.Contains(ps.Id, "fp8_linear") {
			fp8LinearSites = append(fp8LinearSites, ps)
			if ps.Body != "bf16_dequant" {
				t.Errorf("site %q Body = %q, want bf16_dequant", ps.Id, ps.Body)
			}
			if ps.BodyReason == nil || !strings.Contains(*ps.BodyReason, `chosen because preferred body "reference" is recorded as not compiling`) {
				t.Errorf("site %q BodyReason = %v, want failure reason", ps.Id, ps.BodyReason)
			}
		}
	}
	if len(fp8LinearSites) == 0 {
		t.Fatal("no fp8_linear sites found in emitted plan")
	}

	// Verify vxpu plan packages reports 0 failed packages for this plan.
	artRef := fakePutArtifact(t, ts.URL, weightstore.ArtifactRepository("tiny-model"), "v1", artDir)
	outReport := filepath.Join(tmpDir, "packages.json")
	if err := cmdPlan(ctx, []string{
		"packages",
		"--plan", outPlan,
		"--artifact", artRef,
		"--machine", targetMachine,
		"--model", "tiny-model",
		"--modelstore", ts.URL,
		"--out", outReport,
		"--json",
	}); err != nil {
		t.Fatalf("cmdPlan packages failed: %v", err)
	}

	report, err := readPackageReport(outReport)
	if err != nil {
		t.Fatalf("readPackageReport failed: %v", err)
	}
	if len(report.Failed) != 0 {
		t.Errorf("got %d failed packages, want 0", len(report.Failed))
	}

	// Replace outcome with Compiled: true and verify emit treats it identically to not known.
	if err := store.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{
		Op:              "vxpu::fp8_linear",
		Body:            "reference",
		MachineKind:     mk,
		SoftwareVersion: treeCommit,
		Compiled:        true,
	}); err != nil {
		t.Fatalf("RecordBodyCompileOutcome(compiled=true): %v", err)
	}

	outRestored := filepath.Join(tmpDir, "plan_restored.pb")
	if err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outRestored,
		"--tree-commit", treeCommit,
		"--rows", "1",
		"--machine", targetMachine,
		"--modelstore", ts.URL,
	}); err != nil {
		t.Fatalf("emit plan after compiled=true failed: %v", err)
	}
	baseBytes, _ := os.ReadFile(outBaseline)
	restoredBytes, _ := os.ReadFile(outRestored)
	if !bytes.Equal(baseBytes, restoredBytes) {
		t.Errorf("plan after Compiled=true outcome differs from baseline")
	}
}

func TestCmdPlanEmit_RefusalWhenNoAlternative(t *testing.T) {
	ctx := t.Context()
	artDir := createPlanEmitTestArtifact(t)
	tmpDir := t.TempDir()

	ts := fakeModelstoreServer(t)
	defer ts.Close()

	targetMachine := "c3d-standard-60"
	mk, err := kernelplan.MachineKindFromName(targetMachine)
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}

	treeCommit := "commit1"
	store := weightstore.New(ts.URL)
	// Record body outcome as not compiling for rotary reference body (only body available on CPU).
	if err := store.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{
		Op:              "vxpu::rotary",
		Body:            "reference",
		MachineKind:     mk,
		SoftwareVersion: treeCommit,
		Compiled:        false,
		Reason:          "unsupported architecture",
	}); err != nil {
		t.Fatalf("RecordBodyCompileOutcome: %v", err)
	}

	outPlan := filepath.Join(tmpDir, "plan.pb")
	err = cmdPlan(ctx, []string{
		"emit",
		"--artifact", artDir,
		"--out", outPlan,
		"--tree-commit", treeCommit,
		"--rows", "1",
		"--machine", targetMachine,
		"--modelstore", ts.URL,
	})
	if err == nil {
		t.Fatal("expected emit to fail when no alternative body available, got nil")
	}

	errMsg := err.Error()
	wantSubstr := fmt.Sprintf(`site "rotary_0": preferred body "reference" is recorded as not compiling on %q at %q (unsupported architecture); no alternative body available`, mk, treeCommit)
	if !strings.Contains(errMsg, wantSubstr) {
		t.Errorf("error = %q, want substring %q", errMsg, wantSubstr)
	}
}

func TestCmdPlanEmit_RefuseStoreWithoutMachine(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", tmpDir,
		"--modelstore", "http://127.0.0.1:9999",
	})
	if err == nil {
		t.Fatal("expected error when --modelstore passed without --machine, got nil")
	}
	wantSubstr := "--machine is required when querying compile outcomes from store"
	if !strings.Contains(err.Error(), wantSubstr) {
		t.Errorf("error = %q, want substring %q", err.Error(), wantSubstr)
	}
}

func TestCmdPlanEmit_UnreachableStoreOnManifestOnlyArtifact(t *testing.T) {
	ctx := t.Context()
	fixtureDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	tmpDir := t.TempDir()

	// Closed test server guarantees an immediately unreachable store URL.
	ts := httptest.NewServer(http.NotFoundHandler())
	unreachableURL := ts.URL
	ts.Close()

	err := cmdPlan(ctx, []string{
		"emit",
		"--artifact", fixtureDir,
		"--out", filepath.Join(tmpDir, "plan.pb"),
		"--tree-commit", "commit1",
		"--machine", "c3d-standard-60",
		"--modelstore", unreachableURL,
	})
	if err == nil {
		t.Fatal("expected error when --modelstore is unreachable on manifest-only artifact, got nil")
	}
}
