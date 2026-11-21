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
	"sort"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

const (
	cardWidth     = 340.0
	cardHeight    = 96.0
	colGap        = 16.0
	rowGap        = 28.0
	rowPitch      = cardHeight + rowGap
	topMargin     = 48.0
	leftMargin    = 24.0
	depthColWidth = 50.0
	laneWidth     = 14.0
	gutterBasePad = 16.0
)

type routeInfo struct {
	edge                 *planpb.StepGraphEdge
	gutter               string
	yExit, yEnter        float64
	exitOffset, enterOff float64
	lane                 int
}

type streamDepthKey struct {
	stream planpb.Stream
	depth  int32
}

// layoutStepPlanGraph computes node placements, stream column positions,
// edge routing paths through dedicated gutters, and edge crossings.
func layoutStepPlanGraph(graph *planpb.StepPlanGraph) {
	if graph == nil || len(graph.Nodes) == 0 {
		if graph != nil {
			graph.Width, graph.Height = 900.0, 200.0
		}
		return
	}

	nodeMap := make(map[string]*planpb.StepGraphNode, len(graph.Nodes))
	nodeSlot := make(map[string]int, len(graph.Nodes))
	streamDepthCount := make(map[streamDepthKey]int)
	maxMainSlots, maxParallelSlots := 1, 1
	hasParallel, maxDepth := false, int32(0)

	for _, n := range graph.Nodes {
		nodeMap[n.Name] = n
		if n.Stream == planpb.Stream_STREAM_PARALLEL {
			hasParallel = true
		}
		if n.Depth > maxDepth {
			maxDepth = n.Depth
		}

		k := streamDepthKey{stream: n.Stream, depth: n.Depth}
		slot := streamDepthCount[k]
		streamDepthCount[k]++
		nodeSlot[n.Name] = slot

		if n.Stream == planpb.Stream_STREAM_MAIN && slot+1 > maxMainSlots {
			maxMainSlots = slot + 1
		} else if n.Stream == planpb.Stream_STREAM_PARALLEL && slot+1 > maxParallelSlots {
			maxParallelSlots = slot + 1
		}
	}

	mainStreamWidth := float64(maxMainSlots)*cardWidth + float64(maxMainSlots-1)*colGap
	parallelStreamWidth := float64(maxParallelSlots)*cardWidth + float64(maxParallelSlots-1)*colGap

	outCount, inCount := make(map[string]int), make(map[string]int)
	var nonDirectRoutes []*routeInfo
	directEdges := make(map[*planpb.StepGraphEdge]bool)

	for _, e := range graph.Edges {
		u, okU := nodeMap[e.From]
		v, okV := nodeMap[e.To]
		if !okU || !okV {
			continue
		}
		if u.Stream == v.Stream && nodeSlot[u.Name] == nodeSlot[v.Name] && v.Depth == u.Depth+1 {
			directEdges[e] = true
			continue
		}

		gutter := "center"
		if !e.CrossesStreams {
			if u.Stream == planpb.Stream_STREAM_PARALLEL {
				gutter = "right"
			} else if hasParallel {
				gutter = "left"
			} else {
				gutter = "right"
			}
		}

		outIdx, inIdx := outCount[e.From], inCount[e.To]
		outCount[e.From]++
		inCount[e.To]++

		exitOff, enterOff := 6.0+float64(outIdx%4)*4.0, 6.0+float64(inIdx%4)*4.0
		if v.Depth == u.Depth+1 {
			exitOff, enterOff = 7.0+float64(outIdx%2)*3.0, 7.0+float64(inIdx%2)*3.0
		}

		nonDirectRoutes = append(nonDirectRoutes, &routeInfo{
			edge:       e,
			gutter:     gutter,
			yExit:      topMargin + float64(u.Depth)*rowPitch + cardHeight + exitOff,
			yEnter:     topMargin + float64(v.Depth)*rowPitch - enterOff,
			exitOffset: exitOff,
			enterOff:   enterOff,
		})
	}

	gutterLanes := make(map[string]int)
	for _, g := range []string{"left", "center", "right"} {
		var gRoutes []*routeInfo
		for _, r := range nonDirectRoutes {
			if r.gutter == g {
				gRoutes = append(gRoutes, r)
			}
		}
		sort.Slice(gRoutes, func(i, j int) bool {
			if gRoutes[i].yExit != gRoutes[j].yExit {
				return gRoutes[i].yExit < gRoutes[j].yExit
			}
			return gRoutes[i].yEnter < gRoutes[j].yEnter
		})

		var laneEnds []float64
		for _, r := range gRoutes {
			assigned := -1
			for k, endY := range laneEnds {
				if r.yExit >= endY {
					assigned, laneEnds[k] = k, r.yEnter
					break
				}
			}
			if assigned == -1 {
				assigned = len(laneEnds)
				laneEnds = append(laneEnds, r.yEnter)
			}
			r.lane = assigned
		}
		gutterLanes[g] = len(laneEnds)
	}

	leftGutterWidth := gutterBasePad + float64(gutterLanes["left"])*laneWidth
	mainColX := leftMargin + depthColWidth + leftGutterWidth

	centerGutterWidth := gutterBasePad + float64(gutterLanes["center"])*laneWidth
	parallelColX := mainColX + mainStreamWidth + centerGutterWidth

	rightGutterWidth := gutterBasePad + float64(gutterLanes["right"])*laneWidth
	rightEdgeX := mainColX + mainStreamWidth
	if hasParallel {
		rightEdgeX = parallelColX + parallelStreamWidth
	}
	totalWidth := math.Max(rightEdgeX+rightGutterWidth+leftMargin, 900.0)

	for _, n := range graph.Nodes {
		slot := nodeSlot[n.Name]
		colX := mainColX + float64(slot)*(cardWidth+colGap)
		if n.Stream == planpb.Stream_STREAM_PARALLEL {
			colX = parallelColX + float64(slot)*(cardWidth+colGap)
		}
		n.Box = &planpb.StepGraphBox{
			X:      colX,
			Y:      topMargin + float64(n.Depth)*rowPitch,
			Width:  cardWidth,
			Height: cardHeight,
		}
	}

	for _, layer := range graph.Layers {
		layer.Y = topMargin + float64(layer.Depth)*rowPitch + (cardHeight-24.0)/2.0
	}

	for _, stream := range graph.Streams {
		if stream.Stream == planpb.Stream_STREAM_MAIN {
			stream.X, stream.Width = mainColX, mainStreamWidth
		} else if stream.Stream == planpb.Stream_STREAM_PARALLEL {
			stream.X, stream.Width = parallelColX, parallelStreamWidth
		}
	}

	for _, e := range graph.Edges {
		if !directEdges[e] {
			continue
		}
		u, v := nodeMap[e.From], nodeMap[e.To]
		if u == nil || v == nil {
			continue
		}
		startX, startY := u.Box.X+u.Box.Width/2.0, u.Box.Y+u.Box.Height
		endX, endY := v.Box.X+v.Box.Width/2.0, v.Box.Y
		e.Points = []*planpb.StepGraphPoint{{X: startX, Y: startY}, {X: endX, Y: endY}}
		e.Path = fmt.Sprintf("M %.1f %.1f L %.1f %.1f", startX, startY, endX, endY)
	}

	for _, r := range nonDirectRoutes {
		u, v := nodeMap[r.edge.From], nodeMap[r.edge.To]
		if u == nil || v == nil {
			continue
		}
		startX, startY := u.Box.X+u.Box.Width/2.0, u.Box.Y+u.Box.Height
		endX, endY := v.Box.X+v.Box.Width/2.0, v.Box.Y

		var gx float64
		switch r.gutter {
		case "left":
			gx = leftMargin + depthColWidth + 8.0 + float64(r.lane)*laneWidth
		case "center":
			gx = mainColX + mainStreamWidth + 8.0 + float64(r.lane)*laneWidth
		default: // "right"
			gx = rightEdgeX + 8.0 + float64(r.lane)*laneWidth
		}

		r.edge.Points = []*planpb.StepGraphPoint{
			{X: startX, Y: startY},
			{X: startX, Y: r.yExit},
			{X: gx, Y: r.yExit},
			{X: gx, Y: r.yEnter},
			{X: endX, Y: r.yEnter},
			{X: endX, Y: endY},
		}
		r.edge.Path = fmt.Sprintf("M %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f",
			startX, startY, startX, r.yExit, gx, r.yExit, gx, r.yEnter, endX, r.yEnter, endX, endY)
	}

	graph.Width = totalWidth
	graph.Height = topMargin + float64(maxDepth+1)*rowPitch + 32.0
	graph.EdgeCrossings = int32(countEdgeCrossings(graph.Edges))
}

// countEdgeCrossings counts geometric intersections between different routed edges.
func countEdgeCrossings(edges []*planpb.StepGraphEdge) int {
	type seg struct {
		x1, y1, x2, y2 float64
		isHoriz        bool
	}

	edgeSegs := make([][]seg, len(edges))
	for idx, e := range edges {
		for i := 0; i+1 < len(e.Points); i++ {
			p1, p2 := e.Points[i], e.Points[i+1]
			edgeSegs[idx] = append(edgeSegs[idx], seg{
				x1: p1.X, y1: p1.Y, x2: p2.X, y2: p2.Y, isHoriz: p1.Y == p2.Y,
			})
		}
	}

	crossings := 0
	for i := 0; i < len(edges); i++ {
		for j := i + 1; j < len(edges); j++ {
			for _, s1 := range edgeSegs[i] {
				for _, s2 := range edgeSegs[j] {
					if s1.isHoriz == s2.isHoriz {
						continue
					}
					h, v := s1, s2
					if !h.isHoriz {
						h, v = v, h
					}
					hMinX, hMaxX := math.Min(h.x1, h.x2), math.Max(h.x1, h.x2)
					vMinY, vMaxY := math.Min(v.y1, v.y2), math.Max(v.y1, v.y2)
					if hMinX < v.x1 && v.x1 < hMaxX && vMinY < h.y1 && h.y1 < vMaxY {
						crossings++
					}
				}
			}
		}
	}
	return crossings
}
