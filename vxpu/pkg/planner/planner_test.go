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
	"strconv"
	"strings"
	"testing"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestReadArtifact(t *testing.T) {
	manifest := `{"tensors": {
		"e0.w1": {"length": 100}, "e1.w1": {"length": 100},
		"e0.w2": {"length": 50},  "e1.w2": {"length": 50},
		"s0": {"length": 7}, "s1": {"length": 7}}}`
	binding := `{"bound": {"embed": {"length": 1000}},
		"stacked": {"bank": {"parts": [["s0"], ["s1"]]}},
		"external": {"experts": {"3": {
			"w1": {"parts": [["e0.w1"], ["e1.w1"]]},
			"w2": {"parts": [["e0.w2"], ["e1.w2"]]}}}}}`
	a, err := ReadArtifact([]byte(manifest), []byte(binding))
	if err != nil {
		t.Fatal(err)
	}
	if a.DenseBytes != 1014 || a.Experts[3] != (Layer{Experts: 2, BytesPerExpert: 150}) {
		t.Errorf("got %+v", a)
	}
	if _, err := ReadArtifact([]byte(`{"tensors": {}}`), []byte(binding)); err == nil {
		t.Error("a binding naming tensors the manifest lacks should be refused")
	}
	empty := `{"external": {"experts": {"3": {"w1": {"parts": []}}}}}`
	if _, err := ReadArtifact([]byte(manifest), []byte(empty)); err == nil {
		t.Error("a bank with no experts should be refused")
	}
}

// deepseek is DeepSeek-V4.1-Flash's inventory: 40 layers of 384 experts
// of 18.8 MB, 9.8 GB of dense weights.
func deepseek() Artifact {
	a := Artifact{DenseBytes: 9_800_000_000, Experts: map[int]Layer{}}
	for layer := range 40 {
		a.Experts[layer] = Layer{Experts: 384, BytesPerExpert: 18_800_000}
	}
	return a
}

// uniform is traffic that selected every expert of every layer, the
// lower ids more.
func uniform(a Artifact) map[int]map[int]int64 {
	counts := map[int]map[int]int64{}
	for layer, l := range a.Experts {
		counts[layer] = map[int]int64{}
		for e := range l.Experts {
			counts[layer][e] = int64(1000 - e)
		}
	}
	return counts
}

func TestChoose_DeepSeekOnTwoRTXPro6000(t *testing.T) {
	a := deepseek()
	g4 := &pb.Probe{Device: "cuda", Gpu: "NVIDIA RTX PRO 6000", GpuMemoryBytes: 102_005_473_280, GpuCount: 2,
		RamBytes: 387_000_000_000}
	p := Choose(a, g4, nil, uniform(a), nil)
	var placement struct {
		Gpus []struct {
			Experts int
			Layers  map[string][]int
		}
	}
	if err := json.Unmarshal([]byte(p.Switches["VXPU_GPU_EXPERTS"]), &placement); err != nil {
		t.Fatalf("placement: %v (plan %+v)", err, p)
	}
	// the budgets the hero run used by hand: about 68 GB on GPU0 (it also
	// holds the dense weights), 95 GB on GPU1
	var held []int
	for _, g := range placement.Gpus {
		held = append(held, g.Experts)
	}
	if len(held) != 2 || held[0] != 3665 || held[1] != 4968 {
		t.Errorf("placed %v experts per GPU, want 3665 (68.9 GB) and 4968 (93.4 GB)", held)
	}
	// the hottest on the engine's GPU: expert 0 of every layer
	if ids := placement.Gpus[0].Layers["7"]; len(ids) == 0 || ids[0] != 0 {
		t.Errorf("GPU0 should hold the hottest experts, layer 7 holds %v", ids)
	}
	// its 387 GB of RAM holds all 290 GB of experts, as on the hero run
	if len(p.Reasons) != 2 || !strings.Contains(p.Reasons[0], "all 40 layers") || !strings.Contains(p.Reasons[1], "8633 of 15360") {
		t.Errorf("reasons: %v", p.Reasons)
	}
}

func TestChoose_ReasonsForTheCPU(t *testing.T) {
	a := deepseek()
	gpu := &pb.Probe{Device: "cuda", Gpu: "NVIDIA L4", GpuMemoryBytes: 24e9, GpuCount: 1, RamBytes: 2e12}
	for _, c := range []struct {
		name     string
		artifact Artifact
		probe    *pb.Probe
		counts   map[int]map[int]int64
		reason   string
	}{
		{"dense model", Artifact{DenseBytes: 1e9}, gpu, uniform(a), "no routed experts"},
		{"undescribed executor", a, nil, uniform(a), "did not describe"},
		{"CPU executor", a, &pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B14", RamBytes: 2e12}, uniform(a), "no GPU"},
		{"no traffic yet", a, gpu, nil, "no recorded selections"},
		{"dense weights fill the GPU", Artifact{DenseBytes: 20e9, Experts: a.Experts}, gpu, uniform(a), "no GPU has room"},
	} {
		p := Choose(c.artifact, c.probe, nil, c.counts, nil)
		if _, placed := p.Switches["VXPU_GPU_EXPERTS"]; placed || !strings.Contains(p.Reasons[len(p.Reasons)-1], c.reason) {
			t.Errorf("%s: got %+v, want no GPU experts and a reason about %q", c.name, p, c.reason)
		}
	}
}

// TestChoose_SpreadsExpertsOverMachines: DeepSeek on hardware we can get.
// A 4x L4 VM (192 GB of RAM) holds half the experts; a CPU VM holds the
// rest and answers the engine's mailboxes; the L4s hold the hottest of
// the engine's own layers.
func TestChoose_SpreadsExpertsOverMachines(t *testing.T) {
	a := deepseek()
	g2 := &pb.Probe{Device: "cuda", Gpu: "NVIDIA L4", GpuMemoryBytes: 23_580_000_000, GpuCount: 4, RamBytes: 192e9}
	cpu := pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B14", CpuThreads: 60, RamBytes: 258e9}
	p := Choose(a, g2, []Executor{{"10.0.0.2:50051", &cpu}, {"10.0.0.3:50051", &cpu}}, uniform(a), nil)
	// the engine's budget: 0.85 x 192 GB - 16 GiB = 146 GB, 20 layers of 7.2 GB
	remote := p.ExpertExecutors["10.0.0.2:50051"]
	if len(remote) != 20 || remote[0] != 20 || remote[19] != 39 || len(p.ExpertExecutors) != 1 {
		t.Errorf("want layers 20-39 on the first CPU VM and nothing on the second, got %v", p.ExpertExecutors)
	}
	var placement struct {
		Gpus []struct{ Layers map[string][]int }
	}
	if err := json.Unmarshal([]byte(p.Switches["VXPU_GPU_EXPERTS"]), &placement); err != nil {
		t.Fatalf("placement: %v (reasons %v)", err, p.Reasons)
	}
	for _, g := range placement.Gpus {
		for key := range g.Layers {
			if layer, _ := strconv.Atoi(key); layer >= 20 {
				t.Errorf("the GPUs hold layer %d, which a CPU VM serves", layer)
			}
		}
	}
	// short of RAM: what fits nowhere stays on the engine, and it says so
	small := pb.Probe{Device: "cpu", Cpu: "AMD EPYC 9B14", RamBytes: 64e9}
	p = Choose(a, g2, []Executor{{"10.0.0.4:50051", &small}}, nil, nil)
	if !strings.Contains(strings.Join(p.Reasons, "\n"), "fit no machine's RAM") {
		t.Errorf("reasons: %v", p.Reasons)
	}
}

// A GPU after the first with less memory than the headroom: its budget
// went negative and place sliced backwards, a panic in the router's
// request handler.
func TestChoose_SmallSecondGPU(t *testing.T) {
	a := deepseek()
	probe := pb.Probe{Cpu: "test", CpuThreads: 8, RamBytes: 64 << 30, Gpu: "small", GpuCount: 2, GpuMemoryBytes: 4 << 30}
	if p := Choose(a, &probe, nil, uniform(a), nil); p.Switches["VXPU_GPU_EXPERTS"] != "" {
		t.Errorf("experts placed on GPUs with no room: %v", p.Reasons)
	}
}

// Experts of no size (a bank whose parts read as empty) divided by zero.
func TestPlace_NoExpertSize(t *testing.T) {
	place(map[int]map[int]int64{0: {0: 1}}, 0, []int64{1 << 30, -1})
}
