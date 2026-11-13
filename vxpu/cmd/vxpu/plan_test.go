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
	"os"
	"path/filepath"
	"testing"

	planv1alpha1 "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestCmdPlanStep(t *testing.T) {
	tmpDir := t.TempDir()
	graphPath := filepath.Join(tmpDir, "stepgraph.json")
	planPath := filepath.Join(tmpDir, "stepplan.json")
	serializedPlanPath := filepath.Join(tmpDir, "stepplan_serialized.json")
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
    }
  ],
  "inputs": ["x", "y"]
}`
	if err := os.WriteFile(graphPath, []byte(smallGraphJSON), 0644); err != nil {
		t.Fatalf("WriteFile stepgraph failed: %v", err)
	}

	if err := cmdPlan(ctx, []string{"step", "--graph", graphPath, "--out", planPath}); err != nil {
		t.Fatalf("cmdPlan step failed: %v", err)
	}

	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("ReadFile stepplan failed: %v", err)
	}

	var plan planv1alpha1.StepPlan
	if err := protojson.Unmarshal(data, &plan); err != nil {
		t.Fatalf("Unmarshal stepplan failed: %v", err)
	}

	if len(plan.Nodes) != 3 {
		t.Fatalf("got %d nodes, want 3", len(plan.Nodes))
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

	serData, err := os.ReadFile(serializedPlanPath)
	if err != nil {
		t.Fatalf("ReadFile serialized stepplan failed: %v", err)
	}

	var serPlan planv1alpha1.StepPlan
	if err := protojson.Unmarshal(serData, &serPlan); err != nil {
		t.Fatalf("Unmarshal serialized stepplan failed: %v", err)
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
}

func TestCmdPlanEmit(t *testing.T) {
	tmpDir := t.TempDir()
	artDir := filepath.Join("..", "..", "pkg", "kernelplan", "testdata")
	planPath := filepath.Join(tmpDir, "plan.json")
	ctx := t.Context()

	if err := cmdPlan(ctx, []string{"emit", "--artifact", artDir, "--out", planPath, "--rows", "1,5,8", "--tree-commit", "commit123"}); err != nil {
		t.Fatalf("cmdPlan emit failed: %v", err)
	}

	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("ReadFile plan failed: %v", err)
	}

	plan, err := kernelplan.Parse(planBytes)
	if err != nil {
		t.Fatalf("Parse plan failed: %v", err)
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
      "device_time_ms": 0.052,
      "sample_count": 8
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

	// Test vxpu plan show <file>
	if err := cmdPlan(ctx, []string{"show", planPath}); err != nil {
		t.Fatalf("cmdPlan show failed: %v", err)
	}

	// Test vxpu plan show --graph <file> <plan>
	if err := cmdPlan(ctx, []string{"show", "--graph", graphPath, planPath}); err != nil {
		t.Fatalf("cmdPlan show with --graph failed: %v", err)
	}

	// Test vxpu plan show error without plan arg
	if err := cmdPlan(ctx, []string{"show"}); err == nil {
		t.Error("expected error for show without args, got nil")
	}

	// Test vxpu plan step requires --graph
	if err := cmdPlan(ctx, []string{"step", planPath}); err == nil {
		t.Error("expected error for step without --graph, got nil")
	}
}
