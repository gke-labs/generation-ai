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
	"slices"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/proto"
)

// BuildStepPlanGraph constructs a StepPlanGraph protobuf message from a StepPlan
// and an optional StepGraphDescription. It computes streams, layers by topological depth,
// edges, wait dependencies, card crossings, execution actuals, and unmatched references.
func BuildStepPlanGraph(plan *planpb.StepPlan, desc *planpb.StepGraphDescription) *planpb.StepPlanGraph {
	if plan == nil {
		return &planpb.StepPlanGraph{}
	}

	planNodes := make(map[string]*planpb.NodePlan, len(plan.Nodes))
	for _, np := range plan.Nodes {
		planNodes[np.Name] = np
	}

	descNodes := make(map[string]*planpb.StepNode)
	if desc != nil {
		for _, dn := range desc.Nodes {
			descNodes[dn.Name] = dn
		}
	}

	depths := make(map[string]int32, len(plan.Nodes))
	for _, np := range plan.Nodes {
		maxD := int32(0)
		for _, w := range np.Waits {
			if d, ok := depths[w]; ok && d+1 > maxD {
				maxD = d + 1
			}
		}
		depths[np.Name] = maxD
	}

	var (
		nodes               []*planpb.StepGraphNode
		measuredNodes       int32
		totalMeasuredTimeMs float64
		maxDepth            int32
	)

	for _, np := range plan.Nodes {
		d := depths[np.Name]
		if d > maxDepth {
			maxDepth = d
		}

		sgn := &planpb.StepGraphNode{
			Name:   np.Name,
			Stream: np.Stream,
			Waits:  np.Waits,
			Depth:  d,
		}

		if np.Stats != nil && np.Stats.DeviceTimeMs != nil {
			sgn.MeasuredTimeMs = np.Stats.DeviceTimeMs
			measuredNodes++
			totalMeasuredTimeMs += *np.Stats.DeviceTimeMs
		}

		if dn, ok := descNodes[np.Name]; ok {
			sgn.Kind = dn.Kind
			sgn.Execution = dn.Execution
			sgn.Answer = dn.Answer
		}

		nodes = append(nodes, sgn)
	}

	var (
		edges          []*planpb.StepGraphEdge
		unmatchedWaits []*planpb.UnmatchedWait
		totalWaits     int32
		totalCrossings int32
	)

	for _, np := range plan.Nodes {
		for _, waitName := range np.Waits {
			totalWaits++
			producer, ok := planNodes[waitName]
			if !ok {
				unmatchedWaits = append(unmatchedWaits, &planpb.UnmatchedWait{
					Node:   np.Name,
					Target: waitName,
				})
				continue
			}

			edgeCrossesCards := false
			consumerDesc := descNodes[np.Name]
			producerDesc := descNodes[waitName]

			if (consumerDesc != nil && (consumerDesc.Answer == AnswerPeerGPU || consumerDesc.Answer == AnswerRemote)) ||
				(producerDesc != nil && (producerDesc.Answer == AnswerPeerGPU || producerDesc.Answer == AnswerRemote)) {
				edgeCrossesCards = true
				totalCrossings++
			}

			edges = append(edges, &planpb.StepGraphEdge{
				From:           waitName,
				To:             np.Name,
				CrossesCards:   edgeCrossesCards,
				CrossesStreams: producer.Stream != np.Stream,
			})
		}
	}

	layerMap := make(map[int32][]string)
	for _, sgn := range nodes {
		layerMap[sgn.Depth] = append(layerMap[sgn.Depth], sgn.Name)
	}
	var layers []*planpb.StepGraphLayer
	for d := int32(0); d <= maxDepth && len(nodes) > 0; d++ {
		if nodeNames, ok := layerMap[d]; ok {
			layers = append(layers, &planpb.StepGraphLayer{
				Depth:     d,
				NodeNames: nodeNames,
			})
		}
	}

	mainGroup := &planpb.StepStreamGroup{Stream: StreamMain}
	parallelGroup := &planpb.StepStreamGroup{Stream: StreamParallel}
	otherGroup := &planpb.StepStreamGroup{Stream: StreamUnspecified}

	for _, sgn := range nodes {
		switch sgn.Stream {
		case StreamMain:
			mainGroup.NodeNames = append(mainGroup.NodeNames, sgn.Name)
		case StreamParallel:
			parallelGroup.NodeNames = append(parallelGroup.NodeNames, sgn.Name)
		default:
			otherGroup.NodeNames = append(otherGroup.NodeNames, sgn.Name)
		}
	}

	var streamGroups []*planpb.StepStreamGroup
	if len(mainGroup.NodeNames) > 0 {
		streamGroups = append(streamGroups, mainGroup)
	}
	if len(parallelGroup.NodeNames) > 0 {
		streamGroups = append(streamGroups, parallelGroup)
	}
	if len(otherGroup.NodeNames) > 0 {
		streamGroups = append(streamGroups, otherGroup)
	}

	var unmatchedNodes []string
	if desc != nil {
		for _, dn := range desc.Nodes {
			if _, ok := planNodes[dn.Name]; !ok {
				unmatchedNodes = append(unmatchedNodes, dn.Name)
			}
		}
		slices.Sort(unmatchedNodes)
	}

	graph := &planpb.StepPlanGraph{
		Nodes:          nodes,
		Layers:         layers,
		Streams:        streamGroups,
		Edges:          edges,
		TotalWaits:     totalWaits,
		MeasuredNodes:  measuredNodes,
		UnmatchedWaits: unmatchedWaits,
		UnmatchedNodes: unmatchedNodes,
	}

	if desc != nil {
		graph.TotalCrossings = proto.Int32(totalCrossings)
	}

	if measuredNodes > 0 {
		graph.TotalMeasuredTimeMs = proto.Float64(totalMeasuredTimeMs)
	}

	layoutStepPlanGraph(graph)

	return graph
}
