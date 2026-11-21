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

package stepplan

import (
	"path/filepath"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/proto"
)

func TestBuildStepPlanGraph_NilPlan(t *testing.T) {
	g := BuildStepPlanGraph(nil, nil)
	if g == nil {
		t.Fatalf("expected non-nil StepPlanGraph for nil input")
	}
	if len(g.Nodes) != 0 {
		t.Errorf("len(g.Nodes) = %d, want 0", len(g.Nodes))
	}
}

func TestBuildStepPlanGraph_TinyDeepSeekChain(t *testing.T) {
	fixturePath := filepath.Join("testdata", "stepgraph_chain.json")
	desc, err := ReadDescription(fixturePath)
	if err != nil {
		t.Fatalf("ReadDescription failed: %v", err)
	}

	plan, err := Plan(desc, Options{})
	if err != nil {
		t.Fatalf("Plan failed: %v", err)
	}

	graph := BuildStepPlanGraph(plan, desc)
	if len(graph.Nodes) != 11 {
		t.Errorf("len(graph.Nodes) = %d, want 11", len(graph.Nodes))
	}
	if len(graph.Streams) != 1 {
		t.Fatalf("len(graph.Streams) = %d, want 1 (main stream only)", len(graph.Streams))
	}
	if graph.Streams[0].Stream != StreamMain {
		t.Errorf("stream = %v, want StreamMain", graph.Streams[0].Stream)
	}
	if len(graph.Streams[0].NodeNames) != 11 {
		t.Errorf("stream node names count = %d, want 11", len(graph.Streams[0].NodeNames))
	}

	// Verify all waits are drawn as edges
	waitCount := 0
	for _, np := range plan.Nodes {
		waitCount += len(np.Waits)
	}
	if int(graph.TotalWaits) != waitCount {
		t.Errorf("graph.TotalWaits = %d, want %d", graph.TotalWaits, waitCount)
	}
	if len(graph.Edges) != waitCount {
		t.Errorf("len(graph.Edges) = %d, want %d", len(graph.Edges), waitCount)
	}

	// Verify depth ordering
	if len(graph.Layers) != 11 {
		t.Errorf("len(graph.Layers) = %d, want 11 (linear chain)", len(graph.Layers))
	}
	for i, l := range graph.Layers {
		if l.Depth != int32(i) {
			t.Errorf("layer %d depth = %d, want %d", i, l.Depth, i)
		}
		if len(l.NodeNames) != 1 {
			t.Errorf("layer %d node names count = %d, want 1", i, len(l.NodeNames))
		}
	}

	// With desc, TotalCrossings is 0; without desc, it is nil
	if graph.TotalCrossings == nil || *graph.TotalCrossings != 0 {
		t.Errorf("graph.TotalCrossings = %v, want 0", graph.TotalCrossings)
	}
	graphNoDesc := BuildStepPlanGraph(plan, nil)
	if graphNoDesc.TotalCrossings != nil {
		t.Errorf("graphNoDesc.TotalCrossings = %v, want nil", *graphNoDesc.TotalCrossings)
	}

	// Without measurements, MeasuredNodes is 0 and TotalMeasuredTimeMs is unset
	if graph.TotalMeasuredTimeMs != nil {
		t.Errorf("graph.TotalMeasuredTimeMs = %v, want nil", *graph.TotalMeasuredTimeMs)
	}
	if graph.MeasuredNodes != 0 {
		t.Errorf("graph.MeasuredNodes = %d, want 0", graph.MeasuredNodes)
	}

	for _, node := range graph.Nodes {
		if node.MeasuredTimeMs != nil {
			t.Errorf("node %s MeasuredTimeMs = %v, want nil", node.Name, *node.MeasuredTimeMs)
		}
	}
}

func TestBuildStepPlanGraph_WithMeasurements(t *testing.T) {
	fixturePath := filepath.Join("testdata", "stepgraph_chain.json")
	desc, err := ReadDescription(fixturePath)
	if err != nil {
		t.Fatalf("ReadDescription failed: %v", err)
	}

	plan, err := Plan(desc, Options{})
	if err != nil {
		t.Fatalf("Plan failed: %v", err)
	}

	// Add measurements to submod_0 and submod_1
	plan.Nodes[0].Stats = &planpb.NodeExecutionStats{
		DeviceTimeMs: proto.Float64(0.050),
		SampleCount:  proto.Int32(5),
	}
	plan.Nodes[1].Stats = &planpb.NodeExecutionStats{
		DeviceTimeMs: proto.Float64(0.120),
		SampleCount:  proto.Int32(5),
	}

	graph := BuildStepPlanGraph(plan, desc)
	if graph.MeasuredNodes != 2 {
		t.Errorf("graph.MeasuredNodes = %d, want 2", graph.MeasuredNodes)
	}
	if graph.TotalMeasuredTimeMs == nil {
		t.Fatalf("graph.TotalMeasuredTimeMs = nil, want 0.170")
	}
	if diff := *graph.TotalMeasuredTimeMs - 0.170; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("graph.TotalMeasuredTimeMs = %f, want 0.170", *graph.TotalMeasuredTimeMs)
	}

	if graph.Nodes[0].MeasuredTimeMs == nil || *graph.Nodes[0].MeasuredTimeMs != 0.050 {
		t.Errorf("node 0 stats mismatch: time=%v", graph.Nodes[0].MeasuredTimeMs)
	}
	if graph.Nodes[1].MeasuredTimeMs == nil || *graph.Nodes[1].MeasuredTimeMs != 0.120 {
		t.Errorf("node 1 stats mismatch: time=%v", graph.Nodes[1].MeasuredTimeMs)
	}
	if graph.Nodes[2].MeasuredTimeMs != nil {
		t.Errorf("node 2 should have no stats: time=%v", graph.Nodes[2].MeasuredTimeMs)
	}
}

func TestBuildStepPlanGraph_UnmatchedWait(t *testing.T) {
	plan := &planpb.StepPlan{
		Nodes: []*planpb.NodePlan{
			{Name: "n0", Stream: StreamMain},
			{Name: "n1", Stream: StreamMain, Waits: []string{"n0", "ghost_node"}},
		},
	}

	graph := BuildStepPlanGraph(plan, nil)
	if len(graph.UnmatchedWaits) != 1 {
		t.Fatalf("len(graph.UnmatchedWaits) = %d, want 1", len(graph.UnmatchedWaits))
	}
	if graph.UnmatchedWaits[0].Node != "n1" || graph.UnmatchedWaits[0].Target != "ghost_node" {
		t.Errorf("UnmatchedWait = %+v, want {Node: n1, Target: ghost_node}", graph.UnmatchedWaits[0])
	}

	// Valid edge n0 -> n1 exists
	if len(graph.Edges) != 1 {
		t.Errorf("len(graph.Edges) = %d, want 1", len(graph.Edges))
	}
	if graph.Edges[0].From != "n0" || graph.Edges[0].To != "n1" {
		t.Errorf("Edge = %+v, want From: n0, To: n1", graph.Edges[0])
	}
	if graph.TotalWaits != 2 {
		t.Errorf("graph.TotalWaits = %d, want 2", graph.TotalWaits)
	}
}

func TestBuildStepPlanGraph_ParallelAndCrossings(t *testing.T) {
	plan := &planpb.StepPlan{
		Nodes: []*planpb.NodePlan{
			{Name: "n0", Stream: StreamMain},
			{Name: "n1", Stream: StreamParallel, Waits: []string{"n0"}},
			{Name: "n2", Stream: StreamMain, Waits: []string{"n0"}},
			{Name: "n3", Stream: StreamMain, Waits: []string{"n1", "n2"}},
		},
	}

	desc := &planpb.StepGraphDescription{
		Nodes: []*planpb.StepNode{
			{Name: "n0", Kind: "dense", Execution: ExecutionGraph},
			{Name: "n1", Kind: "dense", Execution: ExecutionGraph},
			{Name: "n2", Kind: "dense", Execution: ExecutionGraph},
			{Name: "n3", Kind: "experts", Execution: ExecutionExchange, Answer: AnswerPeerGPU},
		},
	}

	graph := BuildStepPlanGraph(plan, desc)
	if len(graph.Streams) != 2 {
		t.Fatalf("len(graph.Streams) = %d, want 2", len(graph.Streams))
	}

	// Verify layers: n0 is depth 0; n1 and n2 are both depth 1; n3 is depth 2
	if len(graph.Layers) != 3 {
		t.Fatalf("len(graph.Layers) = %d, want 3", len(graph.Layers))
	}
	if len(graph.Layers[1].NodeNames) != 2 {
		t.Errorf("layer 1 should have 2 parallel nodes (n1 and n2), got %d", len(graph.Layers[1].NodeNames))
	}

	// Edge n0 -> n1 crosses streams (Main -> Parallel)
	if !graph.Edges[0].CrossesStreams {
		t.Errorf("edge n0 -> n1 CrossesStreams = false, want true")
	}
	if graph.Edges[0].CrossesCards {
		t.Errorf("edge n0 -> n1 CrossesCards = true, want false")
	}

	// Edge n1 -> n3 crosses streams (Parallel -> Main) and crosses cards (connecting to peer GPU node)
	edgeN1N3 := graph.Edges[2]
	if edgeN1N3.From != "n1" || edgeN1N3.To != "n3" {
		t.Fatalf("expected edge n1 -> n3 at index 2, got %+v", edgeN1N3)
	}
	if !edgeN1N3.CrossesStreams {
		t.Errorf("edge n1 -> n3 CrossesStreams = false, want true")
	}
	if !edgeN1N3.CrossesCards {
		t.Errorf("edge n1 -> n3 CrossesCards = false, want true")
	}
	if graph.TotalCrossings == nil || *graph.TotalCrossings != 2 { // edges to n3 from n1 and n2 both cross to peer GPU
		t.Errorf("graph.TotalCrossings = %v, want 2", graph.TotalCrossings)
	}
}
