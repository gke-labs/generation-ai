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
	"math"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// MemoryCost is what a GPU's memory holds for a planned shape beyond the
// weights, calibrated per card from measured peaks (nvidia-smi, every 5 s)
// on hero run 10's and 12's pods. Every figure is in GB (1e9 bytes).
type MemoryCost struct {
	TotalGB    float64 // the card's memory as the driver reports it (97,887 MiB on the RTX PRO 6000)
	HeadroomGB float64 // kept free: the capture's transients and the allocator's fragmentation (the OOM of two 66-row batches came 0.6 GB under the total)
	// Fixed memory a GPU holds whatever the shape: the CUDA context, the
	// graphs' nodes and inboxes, the warm-up's buffers. The first range
	// holds more (the embedding, the prefill's inputs, the bf16 weight
	// cache), the last some more (the head's graphs).
	FixedFirstGB, FixedMiddleGB, FixedLastGB float64
	// PerRowLayerMB is what one batch adds per row and per layer of the
	// range: the row's caches (window KV, the shared compressed KV and index
	// keys on their source layers) and the batch's step graph's intermediates.
	PerRowLayerMB float64
	// ReplicaMBPerRow is a shared cache replicated onto a range that reads it
	// from another GPU (a range cut inside a KV group): the compressed KV and
	// the indexer's keys at the cache length, per row.
	ReplicaMBPerRow float64
	// ReferenceDenseFactor: above the kernels' row limit every batch's graph
	// after the first carries the reference bodies' bf16 copies of the range's
	// fp8 dense weights (two bytes per fp8 byte), until the GEMM template exists.
	ReferenceDenseFactor float64
	// DrafterPerRowLayerMB is what a drafting batch's verify step (five token
	// rows a row) adds per row and layer of the range beyond a plain batch;
	// DrafterFixedLastGB is what the drafter's own stages and graphs add on
	// the last range whatever the rows (beyond their experts, which the
	// drafter artifact's inventory counts with the layers).
	DrafterPerRowLayerMB float64
	DrafterFixedLastGB   float64
	// HostMsPerExpert is one expert read through the host path in a step
	// (measured Oct 29 2026 on a g4-standard-192: every expert on the CPU
	// cost 39.7 ms a token against 24.2 with all on the GPUs, 240 reads).
	HostMsPerExpert float64
}

// RTXPro6000Memory is the calibration of Nov 1 2026 (pod C, four RTX PRO
// 6000 Blackwell Server Edition): with the weights and the per-batch term
// subtracted, one batch of 33 rows left 11.3 GB on the first range, 3.0 and
// 2.7 on the two middle ones and 4.8 on the last (7.5 in the Nov 2 runs,
// which the constant follows); four batches of 33 added 1.2 to 1.3 GB a
// batch on the 12-layer ranges (3.1 to 3.3 MB a row and layer), 0.5 on the
// 8-layer first range and 0.8 on the last. Checked against four batches of
// 33 on both layouts measured Nov 2: within 1.1 GB on every GPU.
var RTXPro6000Memory = MemoryCost{
	TotalGB:              102.6,
	HeadroomGB:           2.5,
	FixedFirstGB:         11.3,
	FixedMiddleGB:        3.0,
	FixedLastGB:          7.5,
	PerRowLayerMB:        3.0,
	ReplicaMBPerRow:      2.6,
	ReferenceDenseFactor: 2.0,
	DrafterPerRowLayerMB: 4.3,  // one batch of 33 rows with the drafter: GPU 1 94.8 GB against 93.1 plain (Nov 2)
	DrafterFixedLastGB:   11.0, // the same run: GPU 3 84.8 against 66.0, less the drafter stages' 7.2 GB of experts and the per-row term
	HostMsPerExpert:      0.065,
}

// Residency is how many experts of each layer a GPU keeps resident, by GPU
// then layer. nil means every expert of the range's layers (the placement
// files hero runs 7 to 11 ran with).
type Residency map[int]map[int]int

// GPUMemory is one GPU's ledger for a shape.
type GPUMemory struct {
	GPU         int      `json:"gpu"`
	Layers      string   `json:"layers"`
	DenseGB     float64  `json:"dense_gb"`
	ExpertsGB   float64  `json:"experts_gb"`
	Resident    int      `json:"experts_resident"`
	FixedGB     float64  `json:"fixed_gb"`
	BatchesGB   float64  `json:"batches_gb"`   // the batches' caches and graph intermediates
	ReferenceGB float64  `json:"reference_gb"` // reference bodies' weight copies above the row limit
	ReplicaGB   float64  `json:"replica_gb"`
	DrafterGB   float64  `json:"drafter_gb"`
	TotalGB     float64  `json:"total_gb"`
	BudgetGB    float64  `json:"budget_gb"`
	Fits        bool     `json:"fits"`
	ShortfallGB float64  `json:"shortfall_gb"` // how far over the budget (0 when it fits)
	Notes       []string `json:"notes,omitempty"`
}

// MemoryLedger is the shape's memory on every GPU and whether it fits.
type MemoryLedger struct {
	GPUs []GPUMemory `json:"gpus"`
	Fits bool        `json:"fits"`
	Why  string      `json:"why"`
}

// PredictMemory lays out what each GPU holds for plan p serving workload w
// with residency res (nil: every expert of a range's layers), with batches
// wider than maxRows carrying the reference bodies' copies (0: MaxBatchRows).
func PredictMemory(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, res Residency, maxRows int, mc MemoryCost) (MemoryLedger, error) {
	if err := p.Validate(); err != nil {
		return MemoryLedger{}, err
	}
	if w.Rows < 1 || w.Batches < 1 {
		return MemoryLedger{}, fmt.Errorf("workload: rows %d, batches %d: want at least 1 each", w.Rows, w.Batches)
	}
	if maxRows < 1 {
		maxRows = MaxBatchRows
	}
	byLayer := map[int]LayerBytesEntry{}
	for _, l := range lb.Layers {
		byLayer[l.Layer] = l
	}
	sources := map[int]bool{}
	for _, l := range lb.KVSourceLayers {
		sources[l] = true
	}
	ledger := MemoryLedger{Fits: true}
	budget := mc.TotalGB - mc.HeadroomGB
	last := len(p.Ranges) - 1
	var short []string
	for i, r := range p.Ranges {
		first, end, err := r.Span()
		if err != nil {
			return MemoryLedger{}, err
		}
		g := GPUMemory{GPU: r.GPU, Layers: r.Layers, BudgetGB: budget}
		var dense, experts float64
		layers := 0
		for layer := first; layer < end; layer++ {
			l, ok := byLayer[layer]
			if !ok {
				continue
			}
			layers++
			dense += float64(l.DenseBytes)
			n := l.Experts
			if res != nil {
				if per, ok := res[r.GPU]; ok {
					n = per[layer]
				}
			}
			g.Resident += n
			experts += float64(n) * float64(l.BytesPerExpert)
		}
		switch i {
		case 0:
			g.FixedGB = mc.FixedFirstGB
		case last:
			g.FixedGB = mc.FixedLastGB
			dense += float64(lb.HeadBytes)
			if w.Drafter {
				dense += float64(lb.DrafterDenseBytes)
			}
		default:
			g.FixedGB = mc.FixedMiddleGB
		}
		g.DenseGB = dense / 1e9
		g.ExpertsGB = experts / 1e9
		rows := float64(w.Rows)
		g.BatchesGB = float64(w.Batches) * rows * float64(layers) * mc.PerRowLayerMB / 1e3
		if w.Rows > maxRows && w.Batches > 1 {
			g.ReferenceGB = float64(w.Batches-1) * mc.ReferenceDenseFactor * dense / 1e9
			g.Notes = append(g.Notes, fmt.Sprintf("%d rows a batch is over the kernels' %d-row limit: every batch's graph after the first carries the reference bodies' bf16 copies of the range's dense weights, %.1f GB", w.Rows, maxRows, g.ReferenceGB))
		}
		if len(sources) > 0 && first > 0 && !sources[first] {
			g.ReplicaGB = float64(w.Batches) * rows * mc.ReplicaMBPerRow / 1e3
			g.Notes = append(g.Notes, fmt.Sprintf("the range starts inside a KV group (layer %d is not a source layer): the group's shared caches are replicated here, %.2f GB, and refreshed once a step", first, g.ReplicaGB))
		}
		if w.Drafter {
			g.DrafterGB = float64(w.Batches) * rows * float64(layers) * mc.DrafterPerRowLayerMB / 1e3
			if i == last {
				g.DrafterGB += mc.DrafterFixedLastGB
			}
		}
		g.TotalGB = g.DenseGB + g.ExpertsGB + g.FixedGB + g.BatchesGB + g.ReferenceGB + g.ReplicaGB + g.DrafterGB
		g.Fits = g.TotalGB <= budget
		if !g.Fits {
			g.ShortfallGB = g.TotalGB - budget
			ledger.Fits = false
			short = append(short, fmt.Sprintf("gpu %d (layers %s) %.1f GB, %.1f over its %.1f GB budget", r.GPU, r.Layers, g.TotalGB, g.ShortfallGB, budget))
		}
		ledger.GPUs = append(ledger.GPUs, g)
	}
	if ledger.Fits {
		worst := 0.0
		for _, g := range ledger.GPUs {
			worst = math.Max(worst, g.TotalGB)
		}
		ledger.Why = fmt.Sprintf("%d batches of %d rows fit: the fullest GPU holds %.1f of %.1f GB", w.Batches, w.Rows, worst, budget)
	} else {
		ledger.Why = fmt.Sprintf("%d batches of %d rows do not fit: %s", w.Batches, w.Rows, strings.Join(short, "; "))
	}
	return ledger, nil
}

// String prints the ledger the way vxpu plan pipeline shows it.
func (m MemoryLedger) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "memory: %s\n", m.Why)
	for _, g := range m.GPUs {
		fit := "fits"
		if !g.Fits {
			fit = fmt.Sprintf("OVER by %.1f GB", g.ShortfallGB)
		}
		fmt.Fprintf(&b, "  gpu %d layers %-6s dense %5.2f + experts %6.2f (%5d resident) + fixed %4.1f + batches %5.2f + reference %5.2f + replicas %4.2f + drafter %4.2f = %6.1f of %5.1f GB (%s)\n",
			g.GPU, g.Layers, g.DenseGB, g.ExpertsGB, g.Resident, g.FixedGB, g.BatchesGB, g.ReferenceGB, g.ReplicaGB, g.DrafterGB, g.TotalGB, g.BudgetGB, fit)
		for _, n := range g.Notes {
			fmt.Fprintf(&b, "    %s\n", n)
		}
	}
	return b.String()
}

// EvictionCost is the residency lever priced: the experts a full GPU would
// have to give up to fit, and what their reads through the host path cost
// a step.
type EvictionCost struct {
	GPU            int     `json:"gpu"`
	ShortfallGB    float64 `json:"shortfall_gb"`
	Experts        int     `json:"experts_evicted"`
	PerLayer       int     `json:"per_layer"`
	Layers         int     `json:"layers"`
	ColdShare      float64 `json:"cold_share"` // the share of a layer's selections the evicted experts receive (uniform popularity: their share of the experts)
	MissesPerStep  float64 `json:"misses_per_step"`
	MsPerStep      float64 `json:"ms_per_step"`
	HostRAMGB      float64 `json:"host_ram_gb"`
	Basis          string  `json:"basis"`
	FitsAfterwards bool    `json:"fits_afterwards"`
}

// Evictions prices making GPU g's shape fit by evicting experts from its
// layers, spread evenly: the misses a batch step takes on that range are
// the evicted experts' share of the layer's distinct selections (coldShare
// <= 0: uniform popularity, the evicted count over the layer's experts;
// measured popularity gives a smaller share), each read through the host
// path at mc.HostMsPerExpert.
func Evictions(lb LayerBytes, g GPUMemory, w Workload, coldShare float64, mc MemoryCost) EvictionCost {
	ec := EvictionCost{GPU: g.GPU, ShortfallGB: g.ShortfallGB}
	if g.ShortfallGB <= 0 {
		ec.FitsAfterwards = true
		ec.Basis = "nothing to evict: the shape fits"
		return ec
	}
	r := kernelplan.Range{GPU: g.GPU, Layers: g.Layers}
	first, end, err := r.Span()
	if err != nil {
		ec.Basis = err.Error()
		return ec
	}
	var perExpert float64
	experts := 0
	for _, l := range lb.Layers {
		if l.Layer >= first && l.Layer < end && l.Experts > 0 {
			ec.Layers++
			experts = l.Experts
			perExpert = float64(l.BytesPerExpert)
		}
	}
	if ec.Layers == 0 || perExpert == 0 {
		ec.Basis = "the range holds no experts to evict"
		return ec
	}
	ec.Experts = int(math.Ceil(g.ShortfallGB * 1e9 / perExpert))
	ec.PerLayer = int(math.Ceil(float64(ec.Experts) / float64(ec.Layers)))
	ec.Experts = ec.PerLayer * ec.Layers
	ec.FitsAfterwards = ec.PerLayer < experts
	distinct := w.DistinctExperts
	if distinct <= 0 {
		distinct = float64(lb.TopK)
	}
	share := coldShare
	if share <= 0 {
		share = float64(ec.PerLayer) / float64(experts)
	}
	ec.ColdShare = share
	ec.MissesPerStep = float64(ec.Layers) * distinct * share
	ec.MsPerStep = ec.MissesPerStep * mc.HostMsPerExpert
	ec.HostRAMGB = float64(ec.Experts) * perExpert / 1e9
	ec.Basis = fmt.Sprintf("evict %d experts a layer (%d of %d on %d layers, %.1f GB to host RAM); at %.0f distinct experts a layer the evicted ones take %.0f%% of the selections, %.0f misses a batch step, %.1f ms a step through the host path at %.3f ms an expert",
		ec.PerLayer, ec.Experts, experts*ec.Layers, ec.Layers, ec.HostRAMGB, distinct, 100*share, ec.MissesPerStep, ec.MsPerStep, mc.HostMsPerExpert)
	return ec
}

// ReadResidency turns a placement file (vxpu-placement/2: "gpus", each
// with "layers" {layer: [expert ids]}) into the experts resident per GPU and
// layer; GPUs are numbered in the file's order.
func ReadResidency(doc map[string]any) (Residency, error) {
	gpus, ok := doc["gpus"].([]any)
	if !ok {
		return nil, fmt.Errorf("placement: no \"gpus\" list")
	}
	res := Residency{}
	for g, item := range gpus {
		gpu, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("placement: gpu %d is not an object", g)
		}
		layers, ok := gpu["layers"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("placement: gpu %d has no \"layers\"", g)
		}
		res[g] = map[int]int{}
		for key, ids := range layers {
			var layer int
			if _, err := fmt.Sscanf(key, "%d", &layer); err != nil {
				return nil, fmt.Errorf("placement: gpu %d layer %q: %w", g, key, err)
			}
			list, ok := ids.([]any)
			if !ok {
				return nil, fmt.Errorf("placement: gpu %d layer %d: experts are not a list", g, layer)
			}
			res[g][layer] = len(list)
		}
	}
	return res, nil
}
