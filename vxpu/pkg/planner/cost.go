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
	"fmt"
	"strconv"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// The cost model (#349): a decode step's time, predicted from what it
// reads and where. Decode is memory-bound, so each part costs its bytes
// over the bandwidth that reads them, plus fixed costs per host op.
//
//	dense    the dense weights a step reads, over the engine GPU's GEMV
//	         bandwidth (or its CPU's, without a GPU): the fp8 weights as
//	         they are, since decode's GEMV dequantizes them block by block
//	         as it reads (the bf16 copies serve only prefill's GEMMs)
//	experts  per layer, the top-k experts' bytes: the share its GPUs hold
//	         over GPU bandwidth, the rest over the CPU expert kernel's,
//	         the two in parallel; on another executor, its CPU's
//	host ops a crossing per experts or n-gram layer between the step and
//	         the host; an n-gram lookup; per layer with GPU experts, their
//	         kernels and the combine; a layer on another executor, the
//	         wire (a network round trip, serialization) and its CPU
//	segments without the step graph, Python launching each dense segment
//
// The constants are measured. The hero run (DeepSeek-V4.1-Flash on two
// RTX PRO 6000s, #377-#381) gives 35.9 ms/token with every expert on the
// CPU in the step graph, 26.4 with 68 GB of experts on GPU0, 26.6 with
// 163 GB on both GPUs, and node by node 47.3 and 33.5. The L4 run (two
// L4s, experts layers 9-39 on two c3d-standard-60s, #409) gives 119.9 in
// the step graph, with the step host's timers splitting every hop. The
// model reproduces each within 10% (cost_test.go).
const (
	// gemvEfficiency is the share of a GPU's memory bandwidth decode's
	// GEMVs reach where the catalog has no measurement: 252 of 300 GB/s
	// on an L4, bf16 and fp8 alike. The RTX PRO 6000 reaches less (754
	// GB/s, 47%: the hero run's 8.5 GB of dense weights in 11.3 ms), so
	// this is optimistic for a GPU not yet measured.
	gemvEfficiency = 0.85
	// cpuKernelEfficiency is the share of a CPU's memory bandwidth the fp4
	// expert kernel reaches, where it was not measured on the machine:
	// 209 of 288 GB/s on a g4-standard-96, 121 of 154 on a c3d-standard-60.
	cpuKernelEfficiency = 0.73
	crossingMs          = 0.012 // per host op: the GPU's flag to the step host and the reply's back (#409: GPU events less host time)
	engramMs            = 1.0   // per n-gram lookup: the host-parallel gather of a token's rows (#366; 2.0 ms for 2 on the L4 run)
	gpuExpertsMs        = 0.14  // per layer with GPU experts in the step graph: their kernels and the combine
	gpuExpertsNodesMs   = 0.11  // the same, node by node
	peerGPUMs           = 0.15  // per layer with experts on a GPU other than the engine's
	// remoteMs is the wire per layer on another executor: the network both
	// ways, serialization, the server reading the request (#409: 1.10 ms
	// measured between a g2-standard-24 and a c3d-standard-60 in one zone).
	// It depends on the pair: ping there was 0.70-0.85 ms, but 0.20 ms
	// between the two c3ds; the planner should measure the pair it places.
	remoteMs  = 1.10
	segmentMs = 0.22 // per dense segment without the step graph: (20.7 - 11.3 ms) / 43 segments
)

// Prediction is a plan's predicted decode time, by part.
type Prediction struct {
	MsPerToken float64            `json:"ms_per_token"`
	Parts      map[string]float64 `json:"parts_ms"`
	// Basis lists, one line each, what the prediction rests on: the rates used and where they came from.
	Basis []string `json:"basis"`
}

// Predict predicts a decode step of plan p for artifact a, its engine on
// the machine engine describes and its other executors on the machines
// others describe (by address), the model's selections being counts. It
// predicts nothing (MsPerToken 0) for a machine the catalog lacks.
func Predict(a Artifact, p Plan, engine *pb.Probe, others map[string]*pb.Probe,
	counts map[int]map[int]int64) Prediction {
	pr := Prediction{Parts: map[string]float64{}}
	if engine == nil {
		pr.Basis = append(pr.Basis, "not predicted: the engine did not describe its machine")
		return pr
	}
	m := machines.Match(engine)
	if m == nil {
		pr.Basis = append(pr.Basis, fmt.Sprintf("not predicted: %s is not in the machine catalog", describe(engine)))
		return pr
	}
	graph := p.Switches["VXPU_STEP_GRAPH"] == "1"
	// the catalog machine has a GPU and the executor sees it (a CPU-only
	// executor on a GPU machine runs everything on the CPU)
	gpu := m.GPU != nil && engine.Gpu != "" && engine.GpuCount > 0

	// dense
	if gpu {
		bw := gemvBandwidth(m)
		bytes := a.DecodeFP8Bytes + a.DecodeOtherBytes
		pr.Parts["dense"] = ms(bytes, bw)
		pr.Basis = append(pr.Basis, fmt.Sprintf("dense: %.1f GB a step (the fp8 weights read as they are) at %.0f GB/s on %s",
			gb(bytes), gb(int64(bw)), m.GPU.Name))
	} else {
		bw, basis := cpuDenseBandwidth(m)
		bytes := a.DecodeFP8Bytes + a.DecodeOtherBytes
		pr.Parts["dense"] = ms(bytes, bw)
		pr.Basis = append(pr.Basis, fmt.Sprintf("dense: %.1f GB a step on the CPU at %.1f GB/s (%s)", gb(bytes), bw/1e9, basis))
	}
	hostOps := len(a.Experts) + a.EngramLayers
	pr.Parts["crossings"] = float64(hostOps) * crossingMs
	if a.EngramLayers > 0 {
		pr.Parts["n-gram"] = float64(a.EngramLayers) * engramMs
	}
	if !graph {
		pr.Parts["segments"] = float64(hostOps+1) * segmentMs
	}

	// experts, layer by layer
	held := map[int]map[int]float64{} // layer -> GPU index -> share of its selections
	if gpu {
		held = gpuShares(p.Switches["VXPU_GPU_EXPERTS"], counts)
	}
	remote := map[int]string{}
	for addr, layers := range p.ExpertExecutors {
		for _, layer := range layers {
			remote[layer] = addr
		}
	}
	cpuBW := cpuKernelBandwidth(m)
	for layer, l := range a.Experts {
		perToken := int64(a.TopK) * l.BytesPerExpert
		if addr, ok := remote[layer]; ok {
			om := machines.Match(others[addr])
			if om == nil {
				pr.Basis = append(pr.Basis, fmt.Sprintf("not predicted: %s (%s) is not in the machine catalog", addr, describe(others[addr])))
				return Prediction{Parts: map[string]float64{}, Basis: pr.Basis}
			}
			pr.Parts["experts remote"] += ms(perToken, cpuKernelBandwidth(om)) + remoteMs
			continue
		}
		cpuShare, slowest, peer := 1.0, 0.0, false
		for gpu, share := range held[layer] {
			cpuShare -= share
			slowest = max(slowest, ms(int64(share*float64(perToken)), gemvEfficiency*m.GPU.MemoryBandwidthBytesPerSec))
			peer = peer || gpu > 0
		}
		t := max(ms(int64(cpuShare*float64(perToken)), cpuBW), slowest)
		if len(held[layer]) > 0 {
			t += gpuExpertsNodesMs
			if graph {
				t += gpuExpertsMs - gpuExpertsNodesMs
			}
			if peer {
				t += peerGPUMs
			}
		}
		pr.Parts["experts local"] += t
	}
	pr.Basis = append(pr.Basis, fmt.Sprintf("experts: %d of each layer's selections a token; the engine's CPU at %.0f GB/s",
		a.TopK, gb(int64(cpuBW))))
	for addr := range p.ExpertExecutors {
		if om := machines.Match(others[addr]); om != nil {
			pr.Basis = append(pr.Basis, fmt.Sprintf("experts on %s (%s): its CPU at %.0f GB/s, %.2f ms a layer for the wire",
				addr, om.Name, gb(int64(cpuKernelBandwidth(om))), remoteMs))
		}
	}
	for _, v := range pr.Parts {
		pr.MsPerToken += v
	}
	return pr
}

// gemvBandwidth is what decode's dense GEMVs read at on m's GPU:
// measured, or its memory bandwidth at gemvEfficiency. (The expert
// kernels are not measured on their own: they are taken at
// gemvEfficiency.)
func gemvBandwidth(m *machines.Machine) float64 {
	if v, ok := m.Measured["gpu_gemv_bytes_per_sec"].(float64); ok {
		return v
	}
	return gemvEfficiency * m.GPU.MemoryBandwidthBytesPerSec
}

// cpuDenseBandwidth is what a decode step's dense half runs at on m's
// CPU, in weight bytes a second: measured, or the c3d-standard-60's. On a
// CPU it is not memory-bound (the c3d's: 9.9 GB in 1.49 s, 6.65 GB/s,
// against the 121 GB/s its expert kernel reads at): attention, the
// compressor and the hyper-connections are CPU ops, so scaling another
// CPU's memory bandwidth would not estimate it.
func cpuDenseBandwidth(m *machines.Machine) (float64, string) {
	if v, ok := m.Measured["cpu_dense_bytes_per_sec"].(float64); ok {
		return v, "measured on " + m.Name
	}
	c3d, err := machines.Lookup("c3d-standard-60")
	if err != nil {
		return cpuKernelBandwidth(m), fmt.Sprintf("estimated: the CPU's dense kernels are not measured (%v)", err)
	}
	if v, ok := c3d.Measured["cpu_dense_bytes_per_sec"].(float64); ok {
		return v, "estimated: measured on a c3d-standard-60, not on " + m.Name
	}
	return cpuKernelBandwidth(m), "estimated: the CPU's dense kernels are not measured"
}

// cpuKernelBandwidth is what the fp4 expert kernel reads at on m's CPU:
// measured, or its memory bandwidth at cpuKernelEfficiency.
func cpuKernelBandwidth(m *machines.Machine) float64 {
	if v, ok := m.Measured["cpu_expert_kernel_bytes_per_sec"].(float64); ok {
		return v
	}
	return cpuKernelEfficiency * m.CPU.MemoryBandwidthBytesPerSec
}

// gpuShares is, per layer and GPU, the share of the layer's recorded
// selections the placement holds there.
func gpuShares(placementJSON string, counts map[int]map[int]int64) map[int]map[int]float64 {
	var placement struct {
		Gpus []struct {
			Layers map[string][]int `json:"layers"`
		} `json:"gpus"`
	}
	shares := map[int]map[int]float64{}
	if placementJSON == "" || json.Unmarshal([]byte(placementJSON), &placement) != nil {
		return shares
	}
	for gpu, g := range placement.Gpus {
		for key, ids := range g.Layers {
			layer, err := strconv.Atoi(key)
			if err != nil {
				continue
			}
			var total, sum int64
			for _, n := range counts[layer] {
				total += n
			}
			for _, id := range ids {
				sum += counts[layer][id]
			}
			if total > 0 && sum > 0 {
				if shares[layer] == nil {
					shares[layer] = map[int]float64{}
				}
				shares[layer][gpu] = float64(sum) / float64(total)
			}
		}
	}
	return shares
}

func ms(bytes int64, bytesPerSec float64) float64 {
	if bytes <= 0 || bytesPerSec <= 0 {
		return 0
	}
	return 1e3 * float64(bytes) / bytesPerSec
}
