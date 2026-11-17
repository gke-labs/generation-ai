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

package planner

import (
	"fmt"
	"maps"
	"sort"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// Candidate plans (#409): the ways a model can run on a set of machines,
// each predicted by the cost model, ranked. Nothing runs: the planner
// generates the layouts, Predict prices them, and the cheapest feasible
// one wins. Three kinds of layout:
//
//	one-machine     a machine runs the step and holds every expert
//	experts-spread  a machine runs the step (every layer's dense half, on
//	                its GPU if it has one); the experts that do not fit
//	                its RAM are served by other machines' node servers, a
//	                network round trip per remote layer (Choose)
//	layer-ranges    each of k machines runs a contiguous range of whole
//	                layers, dense half and experts on its own GPU and RAM;
//	                a token crosses the network once per range (and back
//	                to the first for the next step), k hops instead of one
//	                per remote layer
//
// The executors can run all three layouts. For layer ranges, each range on
// another machine runs in an engine of its own (VXPU_LAYER_RANGE), which
// serves its range to the first engine (rangeserver.py, #387). A range
// starts only where the step can be cut.

// Node is a machine a candidate can use.
type Node struct {
	// how plans name it: an executor's address, or "g2-standard-48#1"
	Name  string
	Probe machines.Probe
}

// Placement is what one machine does in a candidate.
type Placement struct {
	Node    string `json:"node"`
	Machine string `json:"machine"`
	// Role: "engine" (runs the step's dense half, holds the experts not
	// placed elsewhere), "experts" (a node server for other layers'
	// experts) or "layers" (a range of whole layers).
	Role   string `json:"role"`
	Layers []int  `json:"layers,omitempty"`
	// Switches are the plan switches this machine's engine runs with. For a
	// layer range, they turn on the step graph and place its GPU experts.
	Switches map[string]string `json:"switches,omitempty"`
}

// Candidate is one layout, predicted.
type Candidate struct {
	Kind       string      `json:"kind"`
	Summary    string      `json:"summary"`
	Placements []Placement `json:"placements"`
	Prediction Prediction  `json:"prediction"`
	// Runnable: the executors can run it today; else NotYet says what is
	// missing.
	Runnable bool   `json:"runnable"`
	NotYet   string `json:"not_yet,omitempty"`
	// Infeasible says why it cannot run on these machines at all (it is
	// ranked last, unpriced).
	Infeasible string   `json:"infeasible,omitempty"`
	Reasons    []string `json:"reasons,omitempty"`
	// the switches, for a runnable one
	Plan *Plan `json:"plan,omitempty"`
}

// Candidates are the layouts of a on the fleet, cheapest first (the
// infeasible last). counts are the recorded expert selections, which
// place hot experts on GPUs (nil: no GPU experts).
func Candidates(a Artifact, fleet []Node, counts map[int]map[int]int64) []Candidate {
	var out []Candidate
	tried := map[string]bool{} // machine types tried as the engine
	for i, engine := range fleet {
		if tried[describe(engine.Probe)] {
			continue
		}
		tried[describe(engine.Probe)] = true
		var others []Executor
		for j, n := range fleet {
			if j != i {
				others = append(others, Executor{Address: n.Name, Probe: n.Probe})
			}
		}
		out = append(out, expertsSpread(a, engine, others, counts))
	}
	order := byLayerCost(a, fleet)
	for k := 2; k <= len(order) && k <= len(a.Experts); k++ {
		out = append(out, layerRanges(a, order[:k], counts))
	}
	sort.SliceStable(out, func(i, j int) bool {
		fi, fj := out[i].Infeasible == "", out[j].Infeasible == ""
		if fi != fj {
			return fi
		}
		return out[i].Prediction.MsPerToken < out[j].Prediction.MsPerToken
	})
	return out
}

// CandidatesFor returns the layouts that start on engine, the executor the
// router is loading a model onto, and may use the free executors in others.
// In the experts-spread layout (or one-machine, with no others), engine runs
// the whole step. In layer-ranges layouts, engine runs the first range.
// The cheapest layout comes first.
func CandidatesFor(a Artifact, engine Node, others []Node, counts map[int]map[int]int64, hints map[string]string) []Candidate {
	var executors []Executor
	for _, n := range others {
		executors = append(executors, Executor{Address: n.Name, Probe: n.Probe})
	}
	spread := expertsSpread(a, engine, executors, counts)
	if spread.Plan != nil { // apply the registered plan's switches, as the router does
		maps.Copy(spread.Plan.Switches, hints)
	}
	out := []Candidate{spread}
	order := byLayerCost(a, others)
	for k := 1; k <= len(order) && k < len(a.Experts); k++ {
		out = append(out, layerRanges(a, append([]Node{engine}, order[:k]...), counts))
	}
	sort.SliceStable(out, func(i, j int) bool {
		fi, fj := out[i].Infeasible == "", out[j].Infeasible == ""
		if fi != fj {
			return fi
		}
		return out[i].Prediction.MsPerToken < out[j].Prediction.MsPerToken
	})
	return out
}

func layersOf(a Artifact) []int {
	layers := make([]int, 0, len(a.Experts))
	for layer := range a.Experts {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	return layers
}

func machineName(p machines.Probe) string {
	if m := machines.Match(p); m != nil {
		return m.Name
	}
	return describe(p)
}

func graphSwitches(p machines.Probe) map[string]string {
	if p.GPU != "" && p.GPUCount > 0 {
		// the step graph wherever there is a GPU, with what it requires (plan.py): without
		// the AOTI segments the engine turns it off and runs the dataflow runner in Python
		return map[string]string{"VXPU_STEP_GRAPH": "1", "VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti"}
	}
	return map[string]string{}
}

func expertsSpread(a Artifact, engine Node, others []Executor, counts map[int]map[int]int64) Candidate {
	p := Choose(a, &engine.Probe, others, counts, graphSwitches(engine.Probe))
	maps.Copy(p.Switches, graphSwitches(engine.Probe))
	c := Candidate{Kind: "experts-spread", Runnable: true, Reasons: p.Reasons, Prediction: p.Prediction, Plan: &p}
	remote := map[int]bool{}
	var names []string
	for _, o := range others {
		layers := p.ExpertExecutors[o.Address]
		if len(layers) == 0 {
			continue
		}
		for _, l := range layers {
			remote[l] = true
		}
		c.Placements = append(c.Placements, Placement{Node: o.Address, Machine: machineName(o.Probe), Role: "experts", Layers: layers})
		names = append(names, fmt.Sprintf("%s on %s", span(layers), o.Address))
	}
	var local []int
	for _, l := range layersOf(a) {
		if !remote[l] {
			local = append(local, l)
		}
	}
	c.Placements = append([]Placement{{Node: engine.Name, Machine: machineName(engine.Probe), Role: "engine", Layers: local}}, c.Placements...)
	switch {
	case len(names) == 0:
		c.Kind, c.Summary = "one-machine", engine.Name+" runs everything"
	default:
		c.Summary = fmt.Sprintf("%s runs the step; the experts of layers %s", engine.Name, strings.Join(names, ", "))
	}
	if len(p.Paged) > 0 {
		c.Infeasible = fmt.Sprintf("the experts of %d layers fit no machine's RAM (%s would page from disk, which is not priced)",
			len(p.Paged), span(p.Paged))
	} else if p.Prediction.MsPerToken == 0 {
		c.Infeasible = "not predicted: " + strings.Join(p.Prediction.Basis, "; ")
	}
	return c
}

// byLayerCost is the fleet, the machines that run a layer cheapest first
// (a typical layer, its dense share and its experts, no GPU experts),
// the machine with more RAM first among equals.
func byLayerCost(a Artifact, fleet []Node) []Node {
	layers := layersOf(a)
	if len(layers) == 0 { // a dense model: no layer of experts to price
		return fleet
	}
	cost := map[string]float64{}
	for _, n := range fleet {
		pr := Predict(part(a, layers[:1], len(layers)), Plan{Switches: graphSwitches(n.Probe)}, &n.Probe, nil, nil)
		cost[n.Name] = pr.MsPerToken
		if pr.MsPerToken == 0 {
			cost[n.Name] = 1e9 // not in the catalog: last
		}
	}
	order := append([]Node(nil), fleet...)
	sort.SliceStable(order, func(i, j int) bool {
		ci, cj := cost[order[i].Name], cost[order[j].Name]
		if ci != cj {
			return ci < cj
		}
		return order[i].Probe.RAMBytes > order[j].Probe.RAMBytes
	})
	return order
}

// part is a's share in layers: their experts and n-gram tables, and the
// dense weights pro rata (n of the model's layers have experts).
func part(a Artifact, layers []int, n int) Artifact {
	frac := float64(len(layers)) / float64(n)
	p := Artifact{Experts: map[int]Layer{}, TopK: a.TopK,
		DenseBytes:       int64(frac * float64(a.DenseBytes)),
		DecodeFP8Bytes:   int64(frac * float64(a.DecodeFP8Bytes)),
		DecodeOtherBytes: int64(frac * float64(a.DecodeOtherBytes))}
	in := map[int]bool{}
	for _, l := range layers {
		p.Experts[l] = a.Experts[l]
		in[l] = true
	}
	for _, l := range a.Engram {
		if in[l] {
			p.Engram = append(p.Engram, l)
		}
	}
	p.EngramLayers = len(p.Engram)
	if len(a.Engram) == 0 && in[layers[0]] && layers[0] == layersOf(a)[0] {
		p.EngramLayers = a.EngramLayers // where the layers are not known: with the first range
	}
	return p
}

func layerRanges(a Artifact, nodes []Node, counts map[int]map[int]int64) Candidate {
	layers := layersOf(a)
	c := Candidate{Kind: "layer-ranges", Runnable: true}
	// each machine's capacity, in layers of experts (and, without a GPU, its share of the dense weights)
	perLayer := int64(a.Experts[layers[0]].Experts) * a.Experts[layers[0]].BytesPerExpert
	caps := make([]int, len(nodes))
	total := 0
	for i, n := range nodes {
		budget := ramBudget(n.Probe.RAMBytes)
		if n.Probe.GPU == "" {
			budget -= a.DenseBytes / int64(len(nodes))
		}
		caps[i] = max(0, int(budget/perLayer))
		total += caps[i]
	}
	if total < len(layers) {
		c.Infeasible = fmt.Sprintf("%d machines hold %d of the %d layers' experts in RAM", len(nodes), total, len(layers))
		c.Summary = "layer ranges over " + names(nodes)
		return c
	}
	// contiguous ranges, the cheapest machines filled first (a step costs the sum of its layers'
	// costs, so balance does not matter; the cheapest per layer should hold the most), every
	// machine at least one layer, each range starting where the step can be cut (#387): each
	// machine takes the furthest legal end whose rest still fits, backing up when it doesn't
	counts_ := make([]int, len(nodes))
	var fit func(i, start int) bool
	fit = func(i, start int) bool {
		if i == len(nodes)-1 {
			n := len(layers) - start
			counts_[i] = n
			return n >= 1 && n <= caps[i]
		}
		for e := min(start+caps[i], len(layers)-(len(nodes)-1-i)); e > start; e-- {
			if a.CanStart(layers[e]) {
				counts_[i] = e - start
				if fit(i+1, e) {
					return true
				}
			}
		}
		return false
	}
	if !fit(0, 0) {
		c.Infeasible = fmt.Sprintf("%d machines can't hold the %d layers in ranges that start where the step can be cut",
			len(nodes), len(layers))
		c.Summary = "layer ranges over " + names(nodes)
		return c
	}
	pr := Prediction{Parts: map[string]float64{}}
	start := 0
	var summary []string
	for i, n := range nodes {
		rng := layers[start : start+counts_[i]]
		start += counts_[i]
		sub := part(a, rng, len(layers))
		if n.Probe.GPU != "" {
			if need := int64(denseFP8Factor * float64(sub.DenseBytes)); need > n.Probe.GPUMemoryBytes-gpuHeadroomBytes {
				c.Infeasible = fmt.Sprintf("%s: layers %s need %.1f GB of dense weights on its GPU", n.Name, span(rng), gb(need))
			}
		}
		inRange := map[int]map[int]int64{}
		for _, l := range rng {
			if cc, ok := counts[l]; ok {
				inRange[l] = cc
			}
		}
		plan := Plan{Switches: graphSwitches(n.Probe)}
		placeOnGPUs(&plan, sub, &n.Probe, inRange)
		rp := Predict(sub, plan, &n.Probe, nil, inRange)
		if rp.MsPerToken == 0 {
			c.Infeasible = "not predicted: " + strings.Join(rp.Basis, "; ")
		}
		for k, v := range rp.Parts {
			pr.Parts[k] += v
		}
		pr.Basis = append(pr.Basis, fmt.Sprintf("layers %s on %s (%s): %.1f ms", span(rng), n.Name, machineName(n.Probe), rp.MsPerToken))
		for _, b := range rp.Basis {
			pr.Basis = append(pr.Basis, "  "+b)
		}
		c.Reasons = append(c.Reasons, plan.Reasons...)
		c.Placements = append(c.Placements, Placement{Node: n.Name, Machine: machineName(n.Probe), Role: "layers", Layers: rng,
			Switches: plan.Switches})
		summary = append(summary, fmt.Sprintf("%s on %s", span(rng), n.Name))
	}
	pr.Parts["hops"] = float64(len(nodes)) * remoteMs
	pr.Basis = append(pr.Basis, fmt.Sprintf("hops: %d a token (each range to the next, the last back to the first), %.2f ms each (the wire, as for a remote layer)",
		len(nodes), remoteMs))
	for _, v := range pr.Parts {
		pr.MsPerToken += v
	}
	c.Prediction, c.Summary = pr, "layers "+strings.Join(summary, ", ")
	return c
}

func names(nodes []Node) string {
	var s []string
	for _, n := range nodes {
		s = append(s, n.Name)
	}
	return strings.Join(s, ", ")
}

// span is layers as "a-b" (they are contiguous where it is used) or a list.
func span(layers []int) string {
	if len(layers) == 0 {
		return "none"
	}
	contiguous := true
	for i := 1; i < len(layers); i++ {
		contiguous = contiguous && layers[i] == layers[i-1]+1
	}
	if contiguous {
		if len(layers) == 1 {
			return fmt.Sprint(layers[0])
		}
		return fmt.Sprintf("%d-%d", layers[0], layers[len(layers)-1])
	}
	return strings.Trim(fmt.Sprint(layers), "[]")
}
