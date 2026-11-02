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
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// The pipelined step's cost model (hero run 9). A step across layer
// ranges on several GPUs is, per range, the bytes its GPU reads over the
// GPU's bandwidth plus what the step costs besides bytes; with several
// batches in flight, one per range, the throughput is set by the slowest
// range and a batch's latency by the sum.
//
// Per range k, for a batch of R rows selecting D distinct experts a layer:
//
//	t_k = dense_k / BW                          the range's dense weights, the head on the last range
//	    + D * expertBytes * layers_k / (BW * e)  the experts read once per distinct expert (e: the grouped kernels' share of bandwidth at many pairs)
//	    + (fixed + perRow * R) * layers_k / L    launches, crossings, host nodes and the per-row cost of the kernels, by the range's share of the layers
//
// The constants are calibrated on hero run 8's single-batch table (four
// RTX PRO 6000, the step walking the ranges one GPU at a time): one row
// 13.6 ms, 9 rows with one prompt in every row 18.5, 33 rows 32.8 (the
// same bytes each: 8.5 GB dense, 6 distinct experts a layer, 4.5 GB), so
// fixed + perRow = 4.7 and fixed + 9 perRow = 9.6: perRow 0.61 ms a row,
// fixed 4.1 ms; 33 rows then predicts 33.1 (measured 32.8). With a prompt
// per row the experts dominate: 9 rows select 37 distinct experts a layer
// (measured 35.6 ms), 33 rows 90 to 113 (measured 99.2); the grouped
// expert kernels at those pair counts run at about 87% of the one-row
// kernels' bandwidth (the per-kernel profile at 33 rows: 1,483 µs a layer
// for 2.1 GB), which is e. The model predicts 35.6 within 6% and 99.2
// within 10% (pipeline_test.go).
type PipelineCost struct {
	BandwidthGBs    float64 // the GPU's copy bandwidth; 1,460 on an RTX PRO 6000 (measured)
	FixedMs         float64 // a step's launches, crossings and host nodes at one row
	PerRowMs        float64 // the step's cost per row beyond bytes (kernels at more rows, the boundary copies)
	GroupedEfficacy float64 // the grouped expert kernels' share of bandwidth with a prompt per row
}

// RTXPro6000 is the calibrated cost of hero run 8's machine.
var RTXPro6000 = PipelineCost{BandwidthGBs: 1460, FixedMs: 4.1, PerRowMs: 0.61, GroupedEfficacy: 0.87}

// LayerBytes is what a decode step reads of a model, by layer: the dense
// weights bound into the graphs and the experts held outside them.
type LayerBytes struct {
	Model string `json:"model"`
	// HeadBytes is the output head, read whole once a step by the last range.
	HeadBytes int64 `json:"head_bytes"`
	// DrafterDenseBytes is the drafter's own dense weights (its stages, not
	// the shared head and embedding), on the last range when the drafter runs.
	DrafterDenseBytes int64 `json:"drafter_dense_bytes"`
	TopK              int   `json:"top_k"`
	// KVSourceLayers write the compressed KV cache the layers after them
	// (up to the next source) read: a range may start at one without a
	// replica; a range starting inside a group replicates the group's cache.
	KVSourceLayers []int             `json:"kv_source_layers,omitempty"`
	Layers         []LayerBytesEntry `json:"layers"`
}

// LayerBytesEntry is one layer's bytes.
type LayerBytesEntry struct {
	Layer          int   `json:"layer"`
	DenseBytes     int64 `json:"dense_bytes"`
	Experts        int   `json:"experts"`
	BytesPerExpert int64 `json:"bytes_per_expert"`
}

var layerName = regexp.MustCompile(`^model\.layers\.(\d+)\.`)

// ReadLayerBytes takes the per-layer inventory from an artifact's manifest
// and binding: the bound tensors that are not state, by layer; the head
// (the drafter's, which the model shares) and the drafter's stages; the
// experts per layer from the external banks.
func ReadLayerBytes(manifestJSON, bindingJSON []byte) (LayerBytes, error) {
	var manifest struct {
		Tensors map[string]struct {
			Length int64 `json:"length"`
		} `json:"tensors"`
		Config struct {
			NumExpertsPerTok int `json:"num_experts_per_tok"`
			// the released checkpoints wrap the language model's config in text_config
			TextConfig struct {
				NumExpertsPerTok int `json:"num_experts_per_tok"`
			} `json:"text_config"`
		} `json:"config"`
	}
	var binding struct {
		Bound map[string]struct {
			Length int64 `json:"length"`
		} `json:"bound"`
		State    []string `json:"state"`
		External struct {
			Experts map[string]map[string]struct {
				Parts [][]string `json:"parts"`
			} `json:"experts"`
		} `json:"external"`
	}
	if err := json.Unmarshal(manifestJSON, &manifest); err != nil {
		return LayerBytes{}, fmt.Errorf("manifest: %w", err)
	}
	if err := json.Unmarshal(bindingJSON, &binding); err != nil {
		return LayerBytes{}, fmt.Errorf("binding: %w", err)
	}
	state := map[string]bool{}
	var kvSources []int
	for _, s := range binding.State {
		state[s] = true
		if m := layerName.FindStringSubmatch(s); m != nil && strings.HasSuffix(s, ".attn.compress_kv_cache") {
			layer, err := strconv.Atoi(m[1])
			if err != nil {
				return LayerBytes{}, fmt.Errorf("layer of %q: %w", s, err)
			}
			kvSources = append(kvSources, layer)
		}
	}
	sort.Ints(kvSources)
	dense := map[int]int64{}
	lb := LayerBytes{TopK: manifest.Config.NumExpertsPerTok, KVSourceLayers: kvSources}
	if lb.TopK == 0 {
		lb.TopK = manifest.Config.TextConfig.NumExpertsPerTok
	}
	for name, t := range binding.Bound {
		if state[name] {
			continue
		}
		switch m := layerName.FindStringSubmatch(name); {
		case m != nil:
			layer, err := strconv.Atoi(m[1])
			if err != nil {
				return LayerBytes{}, fmt.Errorf("layer of %q: %w", name, err)
			}
			dense[layer] += t.Length
		case strings.Contains(name, ".embed."): // indexed one row at a time, not read whole
		case strings.Contains(name, ".head."):
			lb.HeadBytes += t.Length
		case strings.Contains(name, ".mtp."):
			lb.DrafterDenseBytes += t.Length
		}
	}
	experts := map[int]LayerBytesEntry{}
	for key, banks := range binding.External.Experts {
		layer, err := strconv.Atoi(key)
		if err != nil {
			return LayerBytes{}, fmt.Errorf("experts of layer %q: %w", key, err)
		}
		e := LayerBytesEntry{Layer: layer}
		var total int64
		for bank, spec := range banks {
			if len(spec.Parts) == 0 {
				return LayerBytes{}, fmt.Errorf("layer %d: bank %s has no experts", layer, bank)
			}
			e.Experts = len(spec.Parts)
			for _, part := range spec.Parts {
				for _, name := range part {
					t, ok := manifest.Tensors[name]
					if !ok {
						return LayerBytes{}, fmt.Errorf("binding names tensor %q, which the manifest lacks", name)
					}
					total += t.Length
				}
			}
		}
		e.BytesPerExpert = total / int64(e.Experts)
		experts[layer] = e
	}
	layers := map[int]bool{}
	for l := range dense {
		layers[l] = true
	}
	for l := range experts {
		layers[l] = true
	}
	for l := range layers {
		e := experts[l]
		e.Layer = l
		e.DenseBytes = dense[l]
		lb.Layers = append(lb.Layers, e)
	}
	sort.Slice(lb.Layers, func(i, j int) bool { return lb.Layers[i].Layer < lb.Layers[j].Layer })
	if len(lb.Layers) == 0 {
		return LayerBytes{}, fmt.Errorf("the binding names no layers")
	}
	return lb, nil
}

// ReadLayerBytesJSON reads the inventory written by vxpu plan pipeline
// --layers-out (or the checked-in fixture of a model).
func ReadLayerBytesJSON(data []byte) (LayerBytes, error) {
	var lb LayerBytes
	if err := json.Unmarshal(data, &lb); err != nil {
		return LayerBytes{}, err
	}
	if len(lb.Layers) == 0 {
		return LayerBytes{}, fmt.Errorf("no layers in the inventory")
	}
	return lb, nil
}

// Workload is what the batches look like.
type Workload struct {
	Rows int // rows per batch
	// DistinctExperts a layer the batch selects: TopK when every row holds
	// the same prompt, up to Rows*TopK with a prompt per row; measured by
	// the engine (VXPU_EXPERT_STATS=1) and carried in the record's stats.
	DistinctExperts float64
	Batches         int // batches in flight (the plan's pipeline)
	Drafter         bool
}

// RangeCost is one range's predicted time for one batch's step.
type RangeCost struct {
	Range    kernelplan.Range `json:"range"`
	Layers   int              `json:"layers"`
	DenseGB  float64          `json:"dense_gb"`
	ExpertGB float64          `json:"expert_gb"`
	BytesMs  float64          `json:"bytes_ms"`
	FixedMs  float64          `json:"fixed_ms"`
	Ms       float64          `json:"ms"`
	HeldGB   float64          `json:"held_gb"` // weights resident on the range's GPU: its dense weights and every expert of its layers
}

// PipelinePrediction is what the model predicts for a plan and a workload.
type PipelinePrediction struct {
	Ranges []RangeCost `json:"ranges"`
	// SingleBatchMs is one batch's step walking the ranges alone (hero 8's measurement).
	SingleBatchMs float64 `json:"single_batch_ms"`
	// PeriodMs is the time between batch steps completing in steady state:
	// the slowest range, or the batch's own cycle when fewer batches than
	// ranges are in flight.
	PeriodMs float64 `json:"period_ms"`
	// LatencyMs is one batch's step in steady state: every stage takes the
	// slowest's time once the pipeline is full.
	LatencyMs      float64  `json:"latency_ms"`
	AggregateTokS  float64  `json:"aggregate_tok_s"`
	PerBatchTokS   float64  `json:"per_batch_tok_s"`
	FloorMs        float64  `json:"floor_ms"`        // one batch's bytes over the aggregate bandwidth of the GPUs
	FloorTokS      float64  `json:"floor_tok_s"`     // rows over the slowest range's bytes alone
	Bottleneck     string   `json:"bottleneck"`      // the range that sets the period
	BalanceWarning string   `json:"balance_warning"` // when the slowest range is more than 15% over the mean
	Basis          []string `json:"basis"`
}

// PredictPipeline predicts the pipelined step of plan p over model lb
// under workload w with the calibrated cost c.
func PredictPipeline(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, c PipelineCost) (PipelinePrediction, error) {
	if err := p.Validate(); err != nil {
		return PipelinePrediction{}, err
	}
	if w.Rows < 1 || w.Batches < 1 {
		return PipelinePrediction{}, fmt.Errorf("workload: rows %d, batches %d: want at least 1 each", w.Rows, w.Batches)
	}
	byLayer := map[int]LayerBytesEntry{}
	for _, l := range lb.Layers {
		byLayer[l.Layer] = l
	}
	backbone := backboneLayers(lb)
	distinct := w.DistinctExperts
	if distinct <= 0 {
		distinct = float64(lb.TopK)
	}
	efficacy := 1.0
	if w.Rows > 1 && distinct > float64(lb.TopK)*1.5 { // the grouped kernels with many pairs
		efficacy = c.GroupedEfficacy
	}
	bw := c.BandwidthGBs * 1e9
	pr := PipelinePrediction{}
	var totalBytes float64
	last := len(p.Ranges) - 1
	for i, r := range p.Ranges {
		first, end, err := r.Span()
		if err != nil {
			return PipelinePrediction{}, err
		}
		rc := RangeCost{Range: r}
		var dense, experts, held float64
		for layer := first; layer < end; layer++ {
			l, ok := byLayer[layer]
			if !ok {
				continue // a drafter stage the plain step does not run (its experts are still held)
			}
			if layer < backbone || w.Drafter {
				rc.Layers++
				dense += float64(l.DenseBytes)
				n := math.Min(distinct, float64(l.Experts))
				experts += n * float64(l.BytesPerExpert)
			}
			held += float64(l.DenseBytes) + float64(l.Experts)*float64(l.BytesPerExpert)
		}
		if i == last {
			dense += float64(lb.HeadBytes)
			held += float64(lb.HeadBytes)
			if w.Drafter {
				dense += float64(lb.DrafterDenseBytes)
			}
		}
		rc.DenseGB = dense / 1e9
		rc.ExpertGB = experts / 1e9
		rc.HeldGB = held / 1e9
		rc.BytesMs = (dense/bw + experts/(bw*efficacy)) * 1e3
		rc.FixedMs = (c.FixedMs + c.PerRowMs*float64(w.Rows)) * float64(rc.Layers) / float64(backbone)
		rc.Ms = rc.BytesMs + rc.FixedMs
		totalBytes += dense + experts
		pr.Ranges = append(pr.Ranges, rc)
	}
	k := float64(len(p.Ranges))
	pr.FloorMs = totalBytes / (bw * k) * 1e3
	var slowestBytes float64
	for _, rc := range pr.Ranges {
		slowestBytes = math.Max(slowestBytes, (rc.DenseGB+rc.ExpertGB)*1e9/bw*1e3)
	}
	pr.FloorTokS = float64(w.Rows) / slowestBytes * 1e3
	summarize(&pr, w.Rows, w.Batches, len(p.Ranges))
	pr.Basis = append(pr.Basis,
		fmt.Sprintf("%d rows a batch, %.0f distinct experts a layer, %d batches in flight over %d ranges", w.Rows, distinct, w.Batches, len(p.Ranges)),
		fmt.Sprintf("bytes at %.0f GB/s a GPU (experts at %.0f%% of it with a prompt per row); fixed %.1f ms a step plus %.2f ms a row, by each range's share of the %d layers",
			c.BandwidthGBs, efficacy*100, c.FixedMs, c.PerRowMs, backbone),
		fmt.Sprintf("one batch alone %.1f ms; in the pipeline a batch step completes every %.1f ms (the slowest range) and takes %.1f ms end to end",
			pr.SingleBatchMs, pr.PeriodMs, pr.LatencyMs))
	return pr, nil
}

// backboneLayers is the number of layers the plain step runs: those with
// dense weights of their own (the drafter's stages have none in the
// inventory; their dense weights are DrafterDenseBytes).
func backboneLayers(lb LayerBytes) int {
	n := 0
	for _, l := range lb.Layers {
		if l.DenseBytes > 0 && l.Layer+1 > n {
			n = l.Layer + 1
		}
	}
	return n
}

// summarize sums the ranges' times into the step's: one batch alone, the
// period between batch steps, a batch's latency, and the rates.
func summarize(pr *PipelinePrediction, rows, batches, ranges int) {
	var sum, slowest float64
	pr.Bottleneck = ""
	for _, rc := range pr.Ranges {
		sum += rc.Ms
		if rc.Ms > slowest {
			slowest = rc.Ms
			pr.Bottleneck = rc.Range.Layers
		}
	}
	pr.SingleBatchMs = sum
	k := float64(ranges)
	n := float64(batches)
	pr.PeriodMs = math.Max(slowest, sum/n)
	pr.LatencyMs = math.Max(sum, math.Min(n, k)*slowest)
	pr.AggregateTokS = float64(rows) / pr.PeriodMs * 1e3
	pr.PerBatchTokS = float64(rows) / pr.LatencyMs * 1e3
	mean := sum / k
	pr.BalanceWarning = ""
	if slowest > mean*1.15 {
		pr.BalanceWarning = fmt.Sprintf("range %s takes %.1f ms, %.0f%% over the mean of %.1f: the period is its time, the other GPUs idle the rest",
			pr.Bottleneck, slowest, (slowest/mean-1)*100, mean)
	}
}

// Balance looks for ranges over the same GPUs that minimize the slowest
// range's time for the workload, cutting only at the layers the step may
// be cut at (cuts; nil: any layer) and holding each range's weights
// within budgetGB of GPU memory. It returns the best ranges found and
// their prediction, or the plan's own when nothing beats it.
func Balance(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, c PipelineCost, cuts []int, budgetGB float64) (*kernelplan.PipelinePlan, PipelinePrediction, error) {
	base, err := PredictPipeline(lb, p, w, c)
	if err != nil {
		return nil, PipelinePrediction{}, err
	}
	total := p.Layers()
	allowed := map[int]bool{}
	if cuts == nil {
		for l := 1; l < total; l++ {
			allowed[l] = true
		}
	} else {
		for _, l := range cuts {
			if l > 0 && l < total {
				allowed[l] = true
			}
		}
	}
	points := make([]int, 0, len(allowed))
	for l := range allowed {
		points = append(points, l)
	}
	sort.Ints(points)
	k := len(p.Ranges)
	best, bestPred := p, base
	var try func(start, gpu int, chosen []kernelplan.Range)
	try = func(start, gpu int, chosen []kernelplan.Range) {
		if gpu == k-1 {
			ranges := append(append([]kernelplan.Range(nil), chosen...), kernelplan.Range{GPU: p.Ranges[gpu].GPU, Layers: fmt.Sprintf("%d:%d", start, total)})
			q := *p
			q.Ranges = ranges
			pred, err := PredictPipeline(lb, &q, w, c)
			if err != nil {
				return
			}
			for _, rc := range pred.Ranges {
				if budgetGB > 0 && rc.HeldGB > budgetGB {
					return
				}
			}
			if pred.PeriodMs < bestPred.PeriodMs*0.999 {
				best, bestPred = &q, pred
			}
			return
		}
		for _, cut := range points {
			if cut <= start {
				continue
			}
			try(cut, gpu+1, append(chosen, kernelplan.Range{GPU: p.Ranges[gpu].GPU, Layers: fmt.Sprintf("%d:%d", start, cut)}))
		}
	}
	try(0, 0, nil)
	if best != p {
		bestPred.Basis = append(bestPred.Basis, fmt.Sprintf("ranges rebalanced from %s: period %.1f ms instead of %.1f", rangesText(p), bestPred.PeriodMs, base.PeriodMs))
	}
	return best, bestPred, nil
}

func rangesText(p *kernelplan.PipelinePlan) string {
	parts := make([]string, 0, len(p.Ranges))
	for _, r := range p.Ranges {
		parts = append(parts, r.Layers)
	}
	return strings.Join(parts, ",")
}
