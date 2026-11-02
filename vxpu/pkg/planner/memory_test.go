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
	"math"
	"os"
	"strings"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// A small model: four layers of 1 GB dense and 10 experts of 100 MB each,
// sources at 0 and 2, a 0.5 GB head; two GPUs of 10 GB with 1 GB headroom.
func memoryFixture() (LayerBytes, *kernelplan.PipelinePlan, MemoryCost) {
	lb := LayerBytes{TopK: 2, HeadBytes: 500e6, KVSourceLayers: []int{0, 2}}
	for l := 0; l < 4; l++ {
		lb.Layers = append(lb.Layers, LayerBytesEntry{Layer: l, DenseBytes: 1e9, Experts: 10, BytesPerExpert: 100e6})
	}
	p := &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, Ranges: []kernelplan.Range{{GPU: 0, Layers: "0:2"}, {GPU: 1, Layers: "2:4"}},
		Pipeline: kernelplan.Pipeline{Batches: 1, Schedule: kernelplan.ScheduleRangeOrder}}
	mc := MemoryCost{TotalGB: 10, HeadroomGB: 1, FixedFirstGB: 1, FixedMiddleGB: 0.5, FixedLastGB: 0.75, PerRowLayerMB: 10,
		ReplicaMBPerRow: 5, ReferenceDenseFactor: 2, DrafterPerRowLayerMB: 20, DrafterFixedLastGB: 1, HostMsPerExpert: 0.1}
	return lb, p, mc
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestPredictMemory_SumsTheTerms(t *testing.T) {
	lb, p, mc := memoryFixture()
	m, err := PredictMemory(lb, p, Workload{Rows: 10, Batches: 2}, nil, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	g0, g1 := m.GPUs[0], m.GPUs[1]
	// gpu 0: dense 2 + experts 2 + fixed (first) 1 + batches 2 x 10 rows x 2 layers x 10 MB = 0.4
	if !near(g0.DenseGB, 2) || !near(g0.ExpertsGB, 2) || !near(g0.FixedGB, 1) || !near(g0.BatchesGB, 0.4) || !near(g0.TotalGB, 5.4) || !g0.Fits {
		t.Fatalf("gpu 0: %+v", g0)
	}
	// gpu 1 (last): the head joins the dense bytes, fixed (last) 0.75; layer 2 is a source: no replica
	if !near(g1.DenseGB, 2.5) || !near(g1.FixedGB, 0.75) || g1.ReplicaGB != 0 || !near(g1.TotalGB, 5.65) || !m.Fits {
		t.Fatalf("gpu 1: %+v, fits %v", g1, m.Fits)
	}
	if !strings.Contains(m.Why, "fit") {
		t.Fatalf("why: %s", m.Why)
	}
}

func TestPredictMemory_RefusesWithTheShortfall(t *testing.T) {
	lb, p, mc := memoryFixture()
	// 40 batches of 10 rows: gpu 0 holds 2+2+1 + 40 x 0.2 = 13 GB against a 9 GB budget
	m, err := PredictMemory(lb, p, Workload{Rows: 10, Batches: 40}, nil, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	if m.Fits || m.GPUs[0].Fits || !near(m.GPUs[0].ShortfallGB, 4) {
		t.Fatalf("expected gpu 0 over by 4 GB: %+v (%s)", m.GPUs[0], m.Why)
	}
	if !strings.Contains(m.Why, "do not fit") || !strings.Contains(m.Why, "gpu 0") {
		t.Fatalf("why: %s", m.Why)
	}
	// the residency lever: 4 GB over, 100 MB an expert, 2 layers: 20 a layer, 40 experts; uniform popularity at 6 distinct: 2 x 6 x 20/10 (capped share) misses
	ec := Evictions(lb, m.GPUs[0], Workload{Rows: 10, Batches: 40, DistinctExperts: 6}, 0, mc)
	if ec.Experts != 40 || ec.PerLayer != 20 || ec.FitsAfterwards {
		t.Fatalf("evictions: %+v", ec)
	}
	ec = Evictions(lb, GPUMemory{GPU: 0, Layers: "0:2", ShortfallGB: 0.5}, Workload{Rows: 10, Batches: 1, DistinctExperts: 6}, 0, mc)
	// 0.5 GB = 5 experts, 3 a layer, 6 in all; share 3/10; misses 2 layers x 6 x 0.3 = 3.6; 0.36 ms a step
	if ec.Experts != 6 || ec.PerLayer != 3 || !ec.FitsAfterwards || !near(ec.MissesPerStep, 3.6) || !near(ec.MsPerStep, 0.36) || !near(ec.HostRAMGB, 0.6) {
		t.Fatalf("evictions: %+v", ec)
	}
	ec = Evictions(lb, GPUMemory{GPU: 0, Layers: "0:2", ShortfallGB: 0.5}, Workload{Rows: 10, Batches: 1, DistinctExperts: 6}, 0.05, mc)
	if !near(ec.MissesPerStep, 0.6) {
		t.Fatalf("a measured cold share of 5%% should give 0.6 misses: %+v", ec)
	}
}

func TestPredictMemory_ReplicasAndReferenceBodies(t *testing.T) {
	lb, p, mc := memoryFixture()
	q := *p
	q.Ranges = []kernelplan.Range{{GPU: 0, Layers: "0:1"}, {GPU: 1, Layers: "1:4"}} // a cut inside the group 0..1
	m, err := PredictMemory(lb, &q, Workload{Rows: 10, Batches: 2}, nil, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	// gpu 1 replicates the group's cache: 2 batches x 10 rows x 5 MB
	if !near(m.GPUs[1].ReplicaGB, 0.1) || len(m.GPUs[1].Notes) != 1 || !strings.Contains(m.GPUs[1].Notes[0], "inside a KV group") {
		t.Fatalf("replica: %+v", m.GPUs[1])
	}
	// 100 rows a batch is over a 64-row limit: every batch after the first carries 2 x the range's dense bytes
	m, err = PredictMemory(lb, p, Workload{Rows: 100, Batches: 3}, nil, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	if !near(m.GPUs[0].ReferenceGB, 2*2*2) || !near(m.GPUs[1].ReferenceGB, 2*2*2.5) {
		t.Fatalf("reference bodies: %+v %+v", m.GPUs[0], m.GPUs[1])
	}
	m, err = PredictMemory(lb, p, Workload{Rows: 100, Batches: 3}, nil, 128, mc)
	if err != nil {
		t.Fatal(err)
	}
	if m.GPUs[0].ReferenceGB != 0 {
		t.Fatalf("under a 128-row limit the reference term should be zero: %+v", m.GPUs[0])
	}
	// the drafter: 20 MB a row and layer on every range, plus 1 GB of stages and graphs on the last
	m, err = PredictMemory(lb, p, Workload{Rows: 10, Batches: 2, Drafter: true}, nil, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	if !near(m.GPUs[0].DrafterGB, 0.8) || !near(m.GPUs[1].DrafterGB, 1.8) {
		t.Fatalf("drafter: %+v %+v", m.GPUs[0], m.GPUs[1])
	}
}

func TestPredictMemory_ResidencyFromAPlacement(t *testing.T) {
	lb, p, mc := memoryFixture()
	doc := map[string]any{"format": "vxpu-placement/2", "gpus": []any{
		map[string]any{"layers": map[string]any{"0": []any{0.0, 1.0, 2.0}, "1": []any{0.0}}},
		map[string]any{"layers": map[string]any{"2": []any{}, "3": []any{0.0, 1.0}}},
	}}
	res, err := ReadResidency(doc)
	if err != nil {
		t.Fatal(err)
	}
	if res[0][0] != 3 || res[0][1] != 1 || res[1][3] != 2 {
		t.Fatalf("residency: %v", res)
	}
	m, err := PredictMemory(lb, p, Workload{Rows: 1, Batches: 1}, res, 64, mc)
	if err != nil {
		t.Fatal(err)
	}
	if m.GPUs[0].Resident != 4 || !near(m.GPUs[0].ExpertsGB, 0.4) || m.GPUs[1].Resident != 2 {
		t.Fatalf("resident experts: %+v %+v", m.GPUs[0], m.GPUs[1])
	}
}

func TestChooseLayout_MemoryRefusesALayout(t *testing.T) {
	lb, p, mc := memoryFixture()
	curve := DistinctCurve{TopK: 2, Experts: 10}
	// 40 rows: one batch of 40 (gpu 0: 5 + 0.8 = 5.8 GB) fits; two batches of 20 fit too (5 + 0.8); under a tiny
	// memory only the layout with the least per-batch memory survives, so make the budget 5.7 GB: 1 x 40 is over
	// (5.8) and 2 x 20 is over as well (5.8); 4 x 10 is 5 + 0.8: all equal here, so check the message path instead
	mc.TotalGB = 6.5 // budget 5.5: every layout is over by 0.3
	c, err := ChooseLayoutWithMemory(lb, p, 40, 2, 0, curve, false, RTXPro6000, 64, LayoutMemory{Cost: mc})
	if err != nil {
		t.Fatal(err)
	}
	for _, cand := range c.Candidates {
		if cand.Fits || cand.Memory.Fits {
			t.Fatalf("a layout over the memory must not fit: %+v", cand.Memory.Why)
		}
	}
	mc.TotalGB = 10
	c, err = ChooseLayoutWithMemory(lb, p, 40, 2, 0, curve, false, RTXPro6000, 64, LayoutMemory{Cost: mc})
	if err != nil {
		t.Fatal(err)
	}
	if !c.Chosen.Fits || !c.Chosen.Memory.Fits {
		t.Fatalf("with 10 GB the choice should fit: %s", c.Why)
	}
}

func TestReadLayerBytes_KVSourceLayers(t *testing.T) {
	manifest := `{"tensors": {}, "config": {"num_experts_per_tok": 2}}`
	binding := `{"bound": {"model.layers.0.attn.w": {"length": 10}}, "state": ["model.layers.0.attn.compress_kv_cache", "model.layers.2.attn.compress_kv_cache", "model.layers.1.attn.window_kv_cache"], "external": {"experts": {}}}`
	lb, err := ReadLayerBytes([]byte(manifest), []byte(binding))
	if err != nil {
		t.Fatal(err)
	}
	if len(lb.KVSourceLayers) != 2 || lb.KVSourceLayers[0] != 0 || lb.KVSourceLayers[1] != 2 {
		t.Fatalf("kv sources: %v", lb.KVSourceLayers)
	}
	if _, err := os.Stat("testdata/deepseek-v41-flash-layers.json"); err != nil {
		t.Skip("no fixture")
	}
	data, _ := os.ReadFile("testdata/deepseek-v41-flash-layers.json")
	fixture, err := ReadLayerBytesJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(fixture.KVSourceLayers) != 4 {
		t.Fatalf("the DeepSeek inventory should name four KV source layers: %v", fixture.KVSourceLayers)
	}
}
