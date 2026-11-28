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
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

func fleetOf(t *testing.T, spec ...string) []Node {
	t.Helper()
	var fleet []Node
	for _, name := range spec {
		m, err := machines.Lookup(name)
		if err != nil {
			t.Fatal(err)
		}
		fleet = append(fleet, Node{Name: fmt.Sprintf("%s#%d", name, len(fleet)+1), Probe: machines.ProbeOf(m)})
	}
	return fleet
}

func deepseekFull() Artifact {
	a := deepseek()
	a.DecodeFP8Bytes, a.DecodeOtherBytes, a.TopK, a.EngramLayers, a.Engram = 6_836_715_520, 1_686_144_448, 6, 2, []int{1, 14}
	return a
}

func ranked(t *testing.T, fleet []Node, counts map[int]map[int]int64) []Candidate {
	t.Helper()
	cands := Candidates(deepseekFull(), fleet, counts)
	for i, c := range cands {
		t.Logf("%d. %-15s %7.1f ms  %s %s", i+1, c.Kind, c.Prediction.MsPerToken, c.Summary, c.Infeasible)
	}
	return cands
}

func machinesIn(c Candidate) []string {
	var out []string
	for _, p := range c.Placements {
		out = append(out, p.Machine)
	}
	return out
}

// The L4 run's fleet: a CPU dense half costs far more than the wire it
// would save, so spreading the experts wins, and every layout that puts
// whole layers on a c3d is rejected by a wide margin.
func TestCandidates_L4AndTwoC3d(t *testing.T) {
	cands := ranked(t, fleetOf(t, "g2-standard-24", "c3d-standard-60", "c3d-standard-60"), nil)
	best := cands[0]
	if best.Kind != "experts-spread" || best.Placements[0].Machine != "g2-standard-24" || !best.Runnable {
		t.Fatalf("best: %+v", best)
	}
	for _, c := range cands[1:] {
		if c.Infeasible == "" && c.Prediction.MsPerToken < 5*best.Prediction.MsPerToken {
			t.Errorf("%s (%s) at %.1f ms is not rejected by a wide margin against %.1f", c.Kind, c.Summary,
				c.Prediction.MsPerToken, best.Prediction.MsPerToken)
		}
	}
}

// Two machines with GPUs and the RAM for half the experts each: a range
// of whole layers on each (two hops a token) beats spreading the experts
// (a hop for each of the second machine's layers).
func TestCandidates_TwoG2Standard48(t *testing.T) {
	cands := ranked(t, fleetOf(t, "g2-standard-48", "g2-standard-48"), nil)
	if cands[0].Kind != "layer-ranges" || len(cands[0].Placements) != 2 || !cands[0].Runnable {
		t.Fatalf("best: %+v", cands[0])
	}
	if cands[1].Kind != "experts-spread" || cands[1].Prediction.MsPerToken <= cands[0].Prediction.MsPerToken {
		t.Errorf("runner-up: %+v", cands[1])
	}
}

// With CPU machines on offer as well, the layer ranges stay on the GPU
// machines: adding a c3d's range only adds its CPU dense half.
func TestCandidates_G2sAndC3ds(t *testing.T) {
	cands := ranked(t, fleetOf(t, "g2-standard-48", "g2-standard-48", "c3d-standard-60", "c3d-standard-60"), nil)
	best := cands[0]
	if best.Kind != "layer-ranges" {
		t.Fatalf("best: %+v", best)
	}
	for _, m := range machinesIn(best) {
		if m != "g2-standard-48" {
			t.Errorf("best uses %v, want only the g2-standard-48s", machinesIn(best))
		}
	}
}

// One machine with the GPUs and RAM for everything needs no network: it
// wins, at the hero run's measured 35.9 ms/token within 10%.
func TestCandidates_OneG4Standard96(t *testing.T) {
	cands := ranked(t, fleetOf(t, "g4-standard-96"), nil)
	if cands[0].Kind != "one-machine" || cands[0].Prediction.MsPerToken < 35.9*0.9 || cands[0].Prediction.MsPerToken > 35.9*1.1 {
		t.Fatalf("best: %+v", cands[0])
	}
}

// TestCandidates_DenseModel: a model with no routed experts has layouts
// too (ordering machines by a layer of experts panicked on it).
func TestCandidates_DenseModel(t *testing.T) {
	dense := Artifact{DenseBytes: 16e9, DecodeOtherBytes: 16e9, Experts: map[int]Layer{}}
	if cands := Candidates(dense, fleetOf(t, "g2-standard-48", "g2-standard-48"), nil); len(cands) == 0 {
		t.Fatal("no candidates for a dense model")
	}
}

// DeepSeek-V4.1-Flash can only be cut before layers 1, 2, 8, 14 and 20
// (cuts.py: layers 2, 8, 14 and 20 write compressed KV caches later layers
// read): two ranges split at 20, and every range of every layout starts
// at one of those cuts (3 and 4 ranges fit too, cut at 8, 14 and 20).
func TestCandidates_OnlyAtLegalCuts(t *testing.T) {
	a := deepseekFull()
	a.Cuts = []int{1, 2, 8, 14, 20}
	cands := Candidates(a, fleetOf(t, "g2-standard-48", "g2-standard-48", "c3d-standard-60", "c3d-standard-60"), nil)
	for i, c := range cands {
		t.Logf("%d. %-15s %7.1f ms  %s %s", i+1, c.Kind, c.Prediction.MsPerToken, c.Summary, c.Infeasible)
	}
	best := cands[0]
	if best.Kind != "layer-ranges" || len(best.Placements) != 2 || best.Placements[1].Layers[0] != 20 {
		t.Fatalf("best: %+v", best)
	}
	more := false
	for _, c := range cands {
		if c.Kind != "layer-ranges" || c.Infeasible != "" {
			continue
		}
		more = more || len(c.Placements) > 2
		for _, p := range c.Placements[1:] {
			if !a.CanStart(p.Layers[0]) {
				t.Errorf("%s starts a range at layer %d", c.Summary, p.Layers[0])
			}
		}
	}
	if !more {
		t.Error("no layout of more than 2 ranges fits, though 0-13, 14-19, 20-39 does")
	}
}

// CandidatesFor starts every layout on the executor being loaded: it runs
// the step (experts-spread) or the first range (layer-ranges), each range
// with its own switches.
func TestCandidatesFor_StartOnTheEngine(t *testing.T) {
	a := deepseekFull()
	a.Cuts = []int{1, 2, 8, 14, 20}
	fleet := fleetOf(t, "g2-standard-48", "c3d-standard-60", "g2-standard-48")
	cands := CandidatesFor(a, fleet[0], fleet[1:], skewedCounts(), map[string]string{"VXPU_DATAFLOW": "1"})
	for i, c := range cands {
		t.Logf("%d. %-15s %7.1f ms  %s %s", i+1, c.Kind, c.Prediction.MsPerToken, c.Summary, c.Infeasible)
		if len(c.Placements) > 0 && c.Placements[0].Node != fleet[0].Name {
			t.Errorf("%s starts on %s, want the engine", c.Summary, c.Placements[0].Node)
		}
	}
	best := cands[0]
	if best.Kind != "layer-ranges" || best.Placements[1].Machine != "g2-standard-48" || best.Placements[1].Layers[0] != 20 {
		t.Fatalf("best: %+v", best)
	}
	for _, p := range best.Placements {
		if p.Switches["VXPU_STEP_GRAPH"] != "1" || p.Switches["VXPU_GPU_EXPERTS"] == "" {
			t.Errorf("%s's switches %v: want the step graph and its GPU experts", p.Node, p.Switches)
		}
		if _, has := p.Switches["VXPU_SEGMENT_RUNTIME"]; has {
			t.Errorf("%s's switches %v must not set removed VXPU_SEGMENT_RUNTIME", p.Node, p.Switches)
		}
	}
	for _, c := range cands {
		if c.Kind == "experts-spread" && c.Plan.Switches["VXPU_DATAFLOW"] != "1" {
			t.Errorf("experts-spread lacks the registered plan's switches: %v", c.Plan.Switches)
		}
	}
}
