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
	"encoding/json"
	"math"
	"strings"
	"testing"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// heldPlacement holds the first n[g] experts of every layer on GPU g;
// with flat counts, that is n[g]/384 of each layer's selections.
func heldPlacement(n ...int) string {
	var gpus []map[string]any
	start := 0
	for _, k := range n {
		layers := map[string][]int{}
		for layer := range 40 {
			for e := start; e < start+k; e++ {
				layers[itoa(layer)] = append(layers[itoa(layer)], e)
			}
		}
		gpus = append(gpus, map[string]any{"layers": layers})
		start += k
	}
	data, _ := json.Marshal(map[string]any{"gpus": gpus})
	return string(data)
}

func itoa(i int) string { b, _ := json.Marshal(i); return string(b) }

func flatCounts() map[int]map[int]int64 {
	counts := map[int]map[int]int64{}
	for layer := range 40 {
		counts[layer] = map[int]int64{}
		for e := range 384 {
			counts[layer][e] = 1
		}
	}
	return counts
}

// TestPredict_HeroRun: the model against what the hero run measured
// (DeepSeek-V4.1-Flash on a g4-standard-96, #377-#381): the step graph
// with every expert on the CPU, with 68 GB of experts on GPU0 (71% of the
// selections), with 163 GB on both GPUs (71% + 21%), and node by node.
func TestPredict_HeroRun(t *testing.T) {
	a := deepseek()
	a.DecodeFP8Bytes, a.DecodeOtherBytes, a.TopK, a.EngramLayers = 6_836_715_520, 1_686_144_448, 6, 2
	g4 := &pb.Probe{Device: "cuda", Gpu: "NVIDIA RTX PRO 6000", GpuMemoryBytes: 102_005_473_280, GpuCount: 2}
	graph := map[string]string{"VXPU_STEP_GRAPH": "1"}
	for _, c := range []struct {
		name     string
		switches map[string]string
		measured float64
	}{
		{"step graph, experts on the CPU", graph, 35.9},
		{"node by node, experts on the CPU", map[string]string{}, 47.3},
		{"step graph, 68 GB on GPU0", map[string]string{"VXPU_STEP_GRAPH": "1", "VXPU_GPU_EXPERTS": heldPlacement(273)}, 26.4},
		{"step graph, 163 GB on both GPUs", map[string]string{"VXPU_STEP_GRAPH": "1", "VXPU_GPU_EXPERTS": heldPlacement(273, 81)}, 26.6},
		{"node by node, 163 GB on both GPUs", map[string]string{"VXPU_GPU_EXPERTS": heldPlacement(273, 81)}, 33.45},
	} {
		pr := Predict(a, Plan{Switches: c.switches}, g4, nil, flatCounts())
		t.Logf("%s: predicted %.1f ms/token, measured %.1f", c.name, pr.MsPerToken, c.measured)
		if off := math.Abs(pr.MsPerToken-c.measured) / c.measured; off > 0.10 {
			t.Errorf("%s: predicted %.1f ms/token (%v), measured %.1f: off by %.0f%%", c.name, pr.MsPerToken, pr.Parts, c.measured, 100*off)
		}
	}
}

// TestPredict_L4Run: the model against the L4 run (#409): DeepSeek on a
// g2-standard-24 (two L4s) in the step graph, its own layers 0-8 with
// their hottest experts on the L4s (a ~75% share of their selections,
// most on the second GPU), layers 9-35 and 36-39 on two c3d-standard-60
// node servers. Measured 119.9 ms/token over 10 prompts.
func TestPredict_L4Run(t *testing.T) {
	a := deepseek()
	a.DecodeFP8Bytes, a.DecodeOtherBytes, a.TopK, a.EngramLayers = 6_836_715_520, 1_686_144_448, 6, 2
	g2 := &pb.Probe{Device: "cuda", Gpu: "NVIDIA L4", GpuMemoryBytes: 23_659_151_360, GpuCount: 2, RamBytes: 101e9}
	c3d := pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B14", CpuThreads: 60, RamBytes: 253e9}
	gpus := []map[string]any{{"layers": map[string][]int{}}, {"layers": map[string][]int{}}}
	counts := map[int]map[int]int64{}
	for layer := range 9 {
		counts[layer] = map[int]int64{}
		for e := range 384 {
			counts[layer][e] = 1
		}
		for e := range 91 { // held: 10x the traffic of the rest, 91 x 10 / (91 x 10 + 293) = 76%
			counts[layer][e] = 10
			g := 1
			if e < 2 {
				g = 0
			}
			gpus[g]["layers"].(map[string][]int)[itoa(layer)] = append(gpus[g]["layers"].(map[string][]int)[itoa(layer)], e)
		}
	}
	placement, err := json.Marshal(map[string]any{"gpus": gpus})
	if err != nil {
		t.Fatal(err)
	}
	remote := map[string][]int{"10.0.0.2:50051": {}, "10.0.0.3:50051": {}}
	for layer := 9; layer < 40; layer++ {
		addr := "10.0.0.2:50051"
		if layer >= 36 {
			addr = "10.0.0.3:50051"
		}
		remote[addr] = append(remote[addr], layer)
	}
	plan := Plan{Switches: map[string]string{"VXPU_STEP_GRAPH": "1", "VXPU_GPU_EXPERTS": string(placement)}, ExpertExecutors: remote}
	pr := Predict(a, plan, g2, map[string]*pb.Probe{"10.0.0.2:50051": &c3d, "10.0.0.3:50051": &c3d}, counts)
	t.Logf("predicted %.1f ms/token (%v), measured 119.9", pr.MsPerToken, pr.Parts)
	if off := math.Abs(pr.MsPerToken-119.9) / 119.9; off > 0.10 {
		t.Errorf("predicted %.1f ms/token (%v), measured 119.9: off by %.0f%%", pr.MsPerToken, pr.Parts, 100*off)
	}
}

func TestPredict_NotInTheCatalog(t *testing.T) {
	pr := Predict(deepseek(), Plan{}, &pb.Probe{Cpu: "Intel Xeon Platinum 8281"}, nil, nil)
	if pr.MsPerToken != 0 || len(pr.Basis) != 1 {
		t.Errorf("an unknown machine should not be predicted: %+v", pr)
	}
}

// TestPredict_ExecutorWithoutAGPU: an executor that sees no GPU runs
// everything on its CPU, even on a machine the catalog lists with GPUs,
// and a placement of GPU experts (a registered plan's) does not apply.
// The placement on a CPU machine panicked the router.
func TestPredict_ExecutorWithoutAGPU(t *testing.T) {
	g4cpu := pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B45 (Turin)", CpuThreads: 96, RamBytes: 360 << 30}
	c3d := pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B14", CpuThreads: 60, RamBytes: 240 << 30}
	held := Plan{Switches: map[string]string{"VXPU_STEP_GRAPH": "1", "VXPU_GPU_EXPERTS": heldPlacement(100)}}
	a := deepseek()
	a.TopK, a.DecodeFP8Bytes = 6, 6_836_715_520
	for _, probe := range []*pb.Probe{&g4cpu, &c3d} {
		pr := Predict(a, held, probe, nil, flatCounts())
		if !strings.Contains(strings.Join(pr.Basis, "\n"), "on the CPU") {
			t.Errorf("%s: dense priced off the CPU: %v", probe.Cpu, pr.Basis)
		}
		if pr.Parts["experts local"] == 0 {
			t.Errorf("%s: no expert time: %v", probe.Cpu, pr.Basis)
		}
	}
}

// TestPredict_DenseModel: a model with no routed experts is predicted too.
func TestPredict_DenseModel(t *testing.T) {
	probe := pb.Probe{Device: "cuda", Gpu: "NVIDIA L4", GpuMemoryBytes: 23_580_000_000, GpuCount: 2, RamBytes: 96e9}
	p := Choose(Artifact{DenseBytes: 16e9, DecodeOtherBytes: 16e9, Experts: map[int]Layer{}}, &probe, nil, nil, nil)
	if p.Prediction.MsPerToken <= 0 {
		t.Errorf("no prediction for a dense model: %+v", p.Prediction)
	}
}
