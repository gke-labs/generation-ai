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
	"fmt"
	"math"
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

	checkEdgeGeometry(t, graph)
}

// checkEdgeGeometry verifies that for every edge in the graph, no edge segment
// intersects any node's bounding box other than at its ends (source and target),
// and verifies that no two node bounding boxes overlap.
func checkEdgeGeometry(t *testing.T, graph *planpb.StepPlanGraph) {
	t.Helper()
	if graph == nil {
		t.Fatal("graph is nil")
	}
	if len(graph.Nodes) == 0 {
		return
	}

	nodeMap := make(map[string]*planpb.StepGraphNode, len(graph.Nodes))
	for _, n := range graph.Nodes {
		nodeMap[n.Name] = n
		if n.Box == nil {
			t.Fatalf("node %s has nil Box", n.Name)
		}
		if n.Box.Width <= 0 || n.Box.Height <= 0 {
			t.Fatalf("node %s has invalid box dimensions: %+v", n.Name, n.Box)
		}
	}

	// Verify that no two node bounding boxes overlap.
	for i := 0; i < len(graph.Nodes); i++ {
		for j := i + 1; j < len(graph.Nodes); j++ {
			n1, n2 := graph.Nodes[i], graph.Nodes[j]
			b1, b2 := n1.Box, n2.Box
			xOverlap := math.Max(b1.X, b2.X) < math.Min(b1.X+b1.Width, b2.X+b2.Width)
			yOverlap := math.Max(b1.Y, b2.Y) < math.Min(b1.Y+b1.Height, b2.Y+b2.Height)
			if xOverlap && yOverlap {
				t.Errorf("nodes %s and %s bounding boxes overlap: [X: %.1f..%.1f, Y: %.1f..%.1f] and [X: %.1f..%.1f, Y: %.1f..%.1f]",
					n1.Name, n2.Name, b1.X, b1.X+b1.Width, b1.Y, b1.Y+b1.Height, b2.X, b2.X+b2.Width, b2.Y, b2.Y+b2.Height)
			}
		}
	}

	for _, e := range graph.Edges {
		if len(e.Points) < 2 {
			t.Errorf("edge %s -> %s has fewer than 2 points (%d)", e.From, e.To, len(e.Points))
			continue
		}
		if e.Path == "" {
			t.Errorf("edge %s -> %s has empty path", e.From, e.To)
		}

		for i := 0; i+1 < len(e.Points); i++ {
			p1 := e.Points[i]
			p2 := e.Points[i+1]

			for _, n := range graph.Nodes {
				b := n.Box
				xMin := b.X
				xMax := b.X + b.Width
				yMin := b.Y
				yMax := b.Y + b.Height

				if n.Name == e.From && i == 0 {
					continue
				}
				if n.Name == e.To && i == len(e.Points)-2 {
					continue
				}

				// Check horizontal segment
				if p1.Y == p2.Y {
					y := p1.Y
					if yMin < y && y < yMax {
						hMinX := math.Min(p1.X, p2.X)
						hMaxX := math.Max(p1.X, p2.X)
						overlapMin := math.Max(xMin, hMinX)
						overlapMax := math.Min(xMax, hMaxX)
						if overlapMin < overlapMax {
							t.Errorf("edge %s -> %s segment %d horizontal line Y=%.1f [X: %.1f..%.1f] intersects node %s box [X: %.1f..%.1f, Y: %.1f..%.1f]",
								e.From, e.To, i, y, hMinX, hMaxX, n.Name, xMin, xMax, yMin, yMax)
						}
					}
				}

				// Check vertical segment
				if p1.X == p2.X {
					x := p1.X
					if xMin < x && x < xMax {
						vMinY := math.Min(p1.Y, p2.Y)
						vMaxY := math.Max(p1.Y, p2.Y)
						overlapMin := math.Max(yMin, vMinY)
						overlapMax := math.Min(yMax, vMaxY)
						if overlapMin < overlapMax {
							t.Errorf("edge %s -> %s segment %d vertical line X=%.1f [Y: %.1f..%.1f] intersects node %s box [X: %.1f..%.1f, Y: %.1f..%.1f]",
								e.From, e.To, i, x, vMinY, vMaxY, n.Name, xMin, xMax, yMin, yMax)
						}
					}
				}
			}
		}
	}
}

func TestStepPlanGraph_GeometryChain(t *testing.T) {
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
	checkEdgeGeometry(t, graph)

	// Verify that edge skipping rows (e.g. submod_0 -> submod_4) has 6 points (routes through gutter)
	for _, e := range graph.Edges {
		if e.From == "submod_0" && e.To == "submod_4" {
			if len(e.Points) != 6 {
				t.Errorf("submod_0 -> submod_4 should route through gutter with 6 points, got %d", len(e.Points))
			}
		}
	}
}

// make85NodeTwoStreamPlan builds an 85-node two-stream DeepSeek step plan
// with 43 dense nodes on STREAM_PARALLEL, 40 experts nodes on STREAM_MAIN,
// and 2 engram nodes on STREAM_MAIN.
func make85NodeTwoStreamPlan() (*planpb.StepPlan, *planpb.StepGraphDescription) {
	var (
		descNodes []*planpb.StepNode
		planNodes []*planpb.NodePlan
	)

	// Layer 0
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_0", Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_0", Stream: StreamParallel, Waits: nil,
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "experts_0", Kind: "experts", Execution: ExecutionExchange,
		Host: &planpb.HostCall{Collection: "experts", Index: 0},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "experts_0", Stream: StreamMain, Waits: []string{"dense_0"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_1", Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_1", Stream: StreamParallel, Waits: []string{"dense_0", "experts_0"},
	})

	// Layer 1
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "engram_1", Kind: "engram", Execution: ExecutionExchange,
		Host: &planpb.HostCall{Collection: "engram", Index: 1},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "engram_1", Stream: StreamMain, Waits: []string{"dense_1"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_2", Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_2", Stream: StreamParallel, Waits: []string{"dense_1", "engram_1", "dense_0"},
	})

	descNodes = append(descNodes, &planpb.StepNode{
		Name: "experts_1", Kind: "experts", Execution: ExecutionExchange,
		Host: &planpb.HostCall{Collection: "experts", Index: 1},
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "experts_1", Stream: StreamMain, Waits: []string{"dense_2"},
	})

	denseCount := 3
	lastDense := "dense_2"
	secondLastDense := "dense_1"

	// Layers 2 to 39: 38 layers
	for l := 2; l < 40; l++ {
		if l == 12 {
			engramName := fmt.Sprintf("engram_%d", l)
			descNodes = append(descNodes, &planpb.StepNode{
				Name: engramName, Kind: "engram", Execution: ExecutionExchange,
				Host: &planpb.HostCall{Collection: "engram", Index: int32(l)},
			})
			planNodes = append(planNodes, &planpb.NodePlan{
				Name: engramName, Stream: StreamMain, Waits: []string{lastDense},
			})

			denseName := fmt.Sprintf("dense_%d", denseCount)
			denseCount++
			descNodes = append(descNodes, &planpb.StepNode{
				Name: denseName, Kind: "dense", Execution: ExecutionGraph,
			})
			planNodes = append(planNodes, &planpb.NodePlan{
				Name: denseName, Stream: StreamParallel, Waits: []string{lastDense, engramName, secondLastDense},
			})
			secondLastDense = lastDense
			lastDense = denseName
		}

		expName := fmt.Sprintf("experts_%d", l)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: expName, Kind: "experts", Execution: ExecutionExchange,
			Host: &planpb.HostCall{Collection: "experts", Index: int32(l)},
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: expName, Stream: StreamMain, Waits: []string{lastDense},
		})

		denseName := fmt.Sprintf("dense_%d", denseCount)
		denseCount++
		descNodes = append(descNodes, &planpb.StepNode{
			Name: denseName, Kind: "dense", Execution: ExecutionGraph,
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: denseName, Stream: StreamParallel, Waits: []string{lastDense, expName, secondLastDense},
		})
		secondLastDense = lastDense
		lastDense = denseName
	}

	// Final norm and lm_head dense node
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "dense_head", Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "dense_head", Stream: StreamParallel, Waits: []string{lastDense},
	})

	return &planpb.StepPlan{Nodes: planNodes}, &planpb.StepGraphDescription{Nodes: descNodes}
}

func TestStepPlanGraph_85NodeTwoStream(t *testing.T) {
	plan, desc := make85NodeTwoStreamPlan()
	if len(plan.Nodes) != 85 {
		t.Fatalf("expected 85 nodes, got %d", len(plan.Nodes))
	}

	graph := BuildStepPlanGraph(plan, desc)
	if len(graph.Nodes) != 85 {
		t.Fatalf("expected 85 nodes in graph, got %d", len(graph.Nodes))
	}
	if len(graph.Streams) != 2 {
		t.Fatalf("expected 2 streams, got %d", len(graph.Streams))
	}

	checkEdgeGeometry(t, graph)

	t.Logf("85-node two-stream graph: %d nodes, %d edges, %d layers, %d edge crossings, width=%.1f, height=%.1f",
		len(graph.Nodes), len(graph.Edges), len(graph.Layers), graph.EdgeCrossings, graph.Width, graph.Height)
}

func TestStepPlanGraph_MultipleNodesAtSameDepthAndLongWaits(t *testing.T) {
	var (
		descNodes []*planpb.StepNode
		planNodes []*planpb.NodePlan
	)

	// Depth 0 root node
	descNodes = append(descNodes, &planpb.StepNode{
		Name: "root_op", Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: "root_op", Stream: StreamMain, Waits: nil,
	})

	var allWaits []string
	allWaits = append(allWaits, "root_op")

	// 5 main stream nodes at depth 1
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("long_named_main_stream_producer_module_op_layer_12_part_%d", i)
		allWaits = append(allWaits, name)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: name, Kind: "experts", Execution: ExecutionExchange,
			Host: &planpb.HostCall{Collection: "experts", Index: int32(i)},
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: name, Stream: StreamMain, Waits: []string{"root_op"},
		})
	}

	// 5 parallel stream nodes at depth 1
	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("long_named_parallel_stream_producer_kernel_projection_layer_12_part_%d", i)
		allWaits = append(allWaits, name)
		descNodes = append(descNodes, &planpb.StepNode{
			Name: name, Kind: "dense", Execution: ExecutionGraph,
		})
		planNodes = append(planNodes, &planpb.NodePlan{
			Name: name, Stream: StreamParallel, Waits: []string{"root_op"},
		})
	}

	// Consumer node at depth 2 with 11 long-named waits
	consumerName := "extremely_long_named_consumer_kernel_operation_that_receives_eleven_dependencies_from_both_streams"
	descNodes = append(descNodes, &planpb.StepNode{
		Name: consumerName, Kind: "dense", Execution: ExecutionGraph,
	})
	planNodes = append(planNodes, &planpb.NodePlan{
		Name: consumerName, Stream: StreamParallel, Waits: allWaits,
	})

	plan := &planpb.StepPlan{Nodes: planNodes}
	desc := &planpb.StepGraphDescription{Nodes: descNodes}

	graph := BuildStepPlanGraph(plan, desc)
	if len(graph.Nodes) != 12 {
		t.Fatalf("expected 12 nodes, got %d", len(graph.Nodes))
	}

	checkEdgeGeometry(t, graph)

	// Check that the 5 main nodes at depth 1 have distinct X coordinates
	mainDepth1X := make(map[float64]bool)
	parallelDepth1X := make(map[float64]bool)
	for _, n := range graph.Nodes {
		if n.Depth == 1 {
			if n.Stream == StreamMain {
				if mainDepth1X[n.Box.X] {
					t.Errorf("duplicate Box.X=%.1f for main stream node %s at depth 1", n.Box.X, n.Name)
				}
				mainDepth1X[n.Box.X] = true
			} else if n.Stream == StreamParallel {
				if parallelDepth1X[n.Box.X] {
					t.Errorf("duplicate Box.X=%.1f for parallel stream node %s at depth 1", n.Box.X, n.Name)
				}
				parallelDepth1X[n.Box.X] = true
			}
		}
	}
	if len(mainDepth1X) != 5 {
		t.Errorf("expected 5 distinct X positions for main stream at depth 1, got %d", len(mainDepth1X))
	}
	if len(parallelDepth1X) != 5 {
		t.Errorf("expected 5 distinct X positions for parallel stream at depth 1, got %d", len(parallelDepth1X))
	}
}
