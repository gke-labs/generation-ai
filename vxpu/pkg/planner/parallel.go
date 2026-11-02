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
	"sort"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// The tensor-parallel step's cost model (hero run 14). With the dense
// weights split across P cards and each layer's experts spread over them,
// one batch's step streams every card at once:
//
//	t = dense / (P * BW * e_dense)                         each card reads a P-th of the dense weights
//	  + sum over layers of maxShare(D, P) * expertBytes     the slowest card's experts a layer (the sum after the
//	      / (BW * e_experts) (+ the pairs' cost)           FFN waits for it), D distinct experts spread over P cards
//	  + reductions * crossing(T)                           two cross-card sums a layer, by the token rows' payload
//	  + tail + rowMs * R / P (+ host gap, + drafter)        the small kernels' launch floor on every card at once
//
// The crossing costs are measured (dev/hero/crossings.py, pod A, Nov 2
// 2026: a captured chain of 86 one-shot all-reduces with a GEMV between
// them costs 10 µs a reduction at one row, 16 at 8 rows, 52 at 33 and
// 194 at 132, bandwidth-bound at about 40 GB/s a card past 33 rows; the
// eager path costs 264 µs a reduction and is unusable). The dense
// efficiency at a quarter of the columns and the ring reduction's curve
// are estimates until the run measures them, and are marked so in the
// basis lines.

// ParallelCost is the tensor-parallel step's calibration for one card type.
type ParallelCost struct {
	// DenseEfficiency is the share of the card's copy bandwidth the dense
	// templates reach with the whole weight (0.90 on the RTX PRO 6000:
	// the ledger's GEMVs at 86 to 98%, the head at 105% of the copy
	// figure); DenseEfficiencyTP the same with a P-th of the columns,
	// where the narrow weights lose programs (estimated 0.75 until
	// measured).
	DenseEfficiency   float64
	DenseEfficiencyTP float64
	// ExpertEfficiency is the fp4 expert kernels' share of bandwidth
	// (0.82 measured, hero run 11).
	ExpertEfficiency float64
	// TailMs is the small kernels' launch floor a step on one card (hero
	// run 11: 876 nodes under 5 µs at 0.75 to 1.5 µs each, 1.48 ms
	// measured); every card runs it, so it does not divide by P. HostMs
	// is the experts host node's share at one row (the routing glue).
	TailMs float64
	HostMs float64
	// RowMs is the step's cost per row beyond bytes on one card: the dense
	// kernels at more rows, 0.50 ms a row measured on the GEMM path from 33
	// to 132 rows of one prompt (hero run 12: 29.0 to 79.3 ms; the pipeline
	// model's 0.61 predates the GEMV sweep and the tiles); under tensor
	// parallelism each card's kernels see a P-th of the columns, so it
	// divides by P (estimated).
	RowMs float64
	// ReductionsPerLayer is the cross-card sums a layer: two (after the
	// attention block and after the FFN).
	ReductionsPerLayer int
	// OneShotUs and RingUs are the measured and estimated cost of one
	// reduction by the payload's token rows (rows times the verify's
	// token rows a row), interpolated log-linearly between the points.
	OneShotUs []CrossingPoint
	RingUs    []CrossingPoint
	// MaxRows is the widest batch the kernel library runs under tensor
	// parallelism (the row-tiled GEMVs of hero run 12 apply to 1,024).
	MaxRows int
}

// CrossingPoint is one measured (token rows, µs a reduction) point.
type CrossingPoint struct {
	TokenRows int
	Us        float64
}

// RTXPro6000Parallel is the calibration for four RTX PRO 6000 on PCIe
// (no NVLink; cards 0-1 and 2-3 share a switch, the other pairs cross the
// sockets at 2.7 µs a flag instead of 2.0).
var RTXPro6000Parallel = ParallelCost{
	DenseEfficiency:    0.90,
	DenseEfficiencyTP:  0.75,
	ExpertEfficiency:   0.82,
	TailMs:             1.5,
	HostMs:             0.3,
	RowMs:              0.50,
	ReductionsPerLayer: 2,
	OneShotUs:          []CrossingPoint{{1, 10}, {8, 16}, {33, 52}, {132, 194}},
	RingUs:             []CrossingPoint{{1, 10}, {8, 16}, {33, 30}, {132, 100}},
	MaxRows:            1024,
}

// CrossingUs is the cost of one reduction of tokenRows rows' payload,
// log-linear between the curve's points and flat past its ends.
func CrossingUs(curve []CrossingPoint, tokenRows float64) float64 {
	if len(curve) == 0 {
		return 0
	}
	if tokenRows <= float64(curve[0].TokenRows) {
		return curve[0].Us
	}
	for i := 1; i < len(curve); i++ {
		a, b := curve[i-1], curve[i]
		if tokenRows <= float64(b.TokenRows) {
			f := math.Log(tokenRows/float64(a.TokenRows)) / math.Log(float64(b.TokenRows)/float64(a.TokenRows))
			return a.Us * math.Pow(b.Us/a.Us, f)
		}
	}
	last := curve[len(curve)-1]
	// past the last point the reduction is bandwidth-bound: linear in the payload
	return last.Us * tokenRows / float64(last.TokenRows)
}

// MaxExpertShare is the experts the fullest of P cards serves when D
// distinct experts a layer are spread over the cards at random: the mean
// share plus about 0.9 standard deviations of a binomial draw (the
// expected maximum of four such draws), 2.45 of 6 at one row on four
// cards, 26 of 90 at 33 rows, D on one card.
func MaxExpertShare(distinct float64, cards int) float64 {
	if cards <= 1 {
		return distinct
	}
	p := float64(cards)
	mean := distinct / p
	sd := math.Sqrt(distinct * (1 / p) * (1 - 1/p))
	share := mean + 0.9*sd
	if share > distinct {
		share = distinct
	}
	if share < 1 && distinct >= 1 {
		share = 1 // one expert is served whole by one card
	}
	return share
}

// PredictParallel predicts one batch's step with the dense layers split
// across the plan's cards and the experts spread over them, for w.Rows
// rows with prompt mix, drafting by spec (Draft 0: plain). The result is
// a SpecPrediction so the chooser can set it beside the layer-sharded
// options; its Pipeline part describes the one range the group runs.
func PredictParallel(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, mix string, spec kernelplan.Speculation,
	c PipelineCost, sc SpecCost, acc Acceptance, pc ParallelCost) (SpecPrediction, error) {
	if err := spec.Validate(); err != nil {
		return SpecPrediction{}, err
	}
	if w.Rows < 1 {
		return SpecPrediction{}, fmt.Errorf("workload: rows %d: want at least 1", w.Rows)
	}
	cards := p.Cards()
	if cards < 1 {
		cards = 1
	}
	if pc.MaxRows > 0 && w.Rows > pc.MaxRows {
		return SpecPrediction{}, fmt.Errorf("%d rows a batch is over the kernels' %d-row limit", w.Rows, pc.MaxRows)
	}
	experts, topK := 0, lb.TopK
	for _, l := range lb.Layers {
		if l.Experts > experts {
			experts = l.Experts
		}
	}
	k := float64(spec.Draft)
	tokens := acc.Tokens(k)
	drafted := k
	var basis []string
	if r, ok := sc.Replayed[ReplayKey(spec)]; ok && k > 0 {
		tokens, drafted = r.TokensPerStep, r.TokenRowsPerStep-1
	} else if spec.Policy == kernelplan.PolicyConfidence && k > 0 {
		accepted := tokens - 1
		wasted := k - accepted
		st, _ := sc.stat(spec.MinConfidence)
		drafted = (1-st.Loss)*accepted + (1-st.Precision)*wasted
		tokens = 1 + (1-st.Loss)*accepted
	}
	tokenRows := float64(w.Rows) * (1 + drafted)
	distinct := DistinctExpertsOfVerify(topK, experts, w.Rows, 1+drafted, mix)
	if k == 0 && w.DistinctExperts > 0 {
		distinct = w.DistinctExperts
	}
	backbone := backboneLayers(lb)
	bw := c.BandwidthGBs * 1e9
	denseEff := pc.DenseEfficiency
	if cards > 1 {
		denseEff = pc.DenseEfficiencyTP
	}
	// bytes: the dense weights and the head (and the drafter's stages when drafting) over every card
	var dense, expertBytes, slowestExpertBytes float64
	layers := 0
	pairsPerLayer := tokenRows * float64(topK)
	for _, l := range lb.Layers {
		if l.Layer >= backbone && k == 0 {
			continue // a drafter stage the plain step does not run
		}
		layers++
		dense += float64(l.DenseBytes)
		d := math.Min(distinct, float64(l.Experts))
		expertBytes += d * float64(l.BytesPerExpert)
		slowestExpertBytes += MaxExpertShare(d, cards) * float64(l.BytesPerExpert)
	}
	dense += float64(lb.HeadBytes)
	if k > 0 {
		dense += float64(lb.DrafterDenseBytes)
	}
	denseMs := dense / (float64(cards) * bw * denseEff) * 1e3
	// experts: the slowest card's share a layer; plain at the kernels' efficiency (grouped past a
	// few rows), a verify at the read peak plus the pairs' cost as the speculation model prices it
	var expertMs, pairMs float64
	if k == 0 {
		eff := pc.ExpertEfficiency
		if w.Rows > 1 && distinct > float64(topK)*1.5 {
			eff *= c.GroupedEfficacy
		}
		expertMs = slowestExpertBytes / (bw * eff) * 1e3
	} else {
		readPeak := sc.ReadPeakGBs
		if readPeak <= 0 {
			readPeak = c.BandwidthGBs
		}
		expertMs = slowestExpertBytes / (readPeak * 1e9) * 1e3
		perCardPairs := pairsPerLayer / float64(cards)
		pairUs := sc.ExpertPairUs
		if sc.ExpertPairSaturation > 0 {
			pairUs /= 1 + perCardPairs/sc.ExpertPairSaturation
		}
		if mix == MixSame {
			pairUs = sc.ExpertPairUsShared
		}
		pairMs = float64(layers) * perCardPairs * pairUs / 1e3
	}
	// crossings: two sums a layer, by the payload's token rows
	reduce := kernelplan.ReduceOneShot
	curve := pc.OneShotUs
	if p.Parallel != nil && p.Parallel.Reduce == kernelplan.ReduceRing {
		reduce, curve = kernelplan.ReduceRing, pc.RingUs
	} else if p.Parallel != nil && p.Parallel.Reduce == "" && CrossingUs(pc.RingUs, tokenRows) < CrossingUs(pc.OneShotUs, tokenRows) {
		reduce, curve = kernelplan.ReduceRing, pc.RingUs // the planner's choice by payload
	}
	crossingMs := 0.0
	reductions := 0
	if cards > 1 {
		reductions = pc.ReductionsPerLayer * layers
		crossingMs = float64(reductions) * CrossingUs(curve, tokenRows) / 1e3
	}
	// the tail on every card, the rows' cost on a P-th of the columns, the host node, the drafter
	fixedMs := pc.TailMs + pc.HostMs + pc.RowMs*float64(w.Rows)/float64(cards)
	hostGap := 0.0
	if k > 0 {
		fixedMs += sc.PerDraftRowMs * float64(w.Rows) * drafted / float64(cards)
		if tokenRows > float64(sc.HostGapRows) && mix != MixSame {
			hostGap = sc.HostGapMs
			fixedMs += hostGap
		}
		fixedMs += sc.DrafterStepMs + sc.DrafterRowMs*float64(w.Rows-1)
	}
	stepMs := denseMs + expertMs + pairMs + crossingMs + fixedMs
	floorMs := (dense + expertBytes) / (float64(cards) * bw) * 1e3

	rc := RangeCost{Range: p.Ranges[0], Layers: layers, DenseGB: dense / 1e9, ExpertGB: expertBytes / 1e9,
		BytesMs: denseMs + expertMs, FixedMs: fixedMs + pairMs + crossingMs, Ms: stepMs,
		HeldGB: (dense + totalExpertBytes(lb)) / 1e9 / float64(cards)}
	pipe := PipelinePrediction{Ranges: []RangeCost{rc}, SingleBatchMs: stepMs, PeriodMs: stepMs, LatencyMs: stepMs,
		FloorMs: floorMs, FloorTokS: float64(w.Rows) / floorMs * 1e3,
		AggregateTokS: float64(w.Rows) / stepMs * 1e3, PerBatchTokS: float64(w.Rows) / stepMs * 1e3,
		Bottleneck: fmt.Sprintf("the %d cards together", cards)}
	pipe.Basis = append(pipe.Basis,
		fmt.Sprintf("%d rows in one batch over %d cards at once (dense weights split %d ways, each layer's experts spread over the cards): %.0f distinct experts a layer, the fullest card serving %.1f of them",
			w.Rows, cards, cards, distinct, MaxExpertShare(math.Min(distinct, float64(experts)), cards)),
		fmt.Sprintf("dense %.1f GB over %d × %.0f GB/s at %.0f%% efficiency: %.1f ms (the efficiency at a %d-th of the columns is estimated until measured); the slowest card's experts %.1f GB: %.1f ms; %d pairs a layer a card: %.1f ms",
			dense/1e9, cards, c.BandwidthGBs, denseEff*100, denseMs, cards, slowestExpertBytes/1e9, expertMs, int(pairsPerLayer/float64(cards)), pairMs),
		fmt.Sprintf("%d %s reductions of %.0f token rows' payload at %.0f µs each: %.1f ms (one-shot measured on pod A; the ring form is an estimate); tail %.1f + host %.1f + %.2f ms a row over %d cards%s: %.1f ms",
			reductions, reduce, tokenRows, CrossingUs(curve, tokenRows), crossingMs, pc.TailMs, pc.HostMs+hostGap, pc.RowMs, cards,
			map[bool]string{true: " + the drafter's step", false: ""}[k > 0], fixedMs),
		fmt.Sprintf("floor %.1f ms (%.1f GB over the %d cards' bandwidth); one batch's step %.1f ms", floorMs, (dense+expertBytes)/1e9, cards, stepMs))
	pr := SpecPrediction{Pipeline: pipe, Speculation: spec, TokensPerStep: tokens, DraftedPerRow: drafted, Distinct: distinct,
		StepMs: stepMs, PeriodMs: stepMs, AggregateTokS: pipe.AggregateTokS * tokens, PerBatchTokS: pipe.PerBatchTokS * tokens, MsPerToken: stepMs / tokens}
	pr.Basis = append(basis, fmt.Sprintf("%.2f tokens a step a row: %.0f tok/s, %.1f ms a token for a row", tokens, pr.AggregateTokS, pr.MsPerToken))
	return pr, nil
}

func totalExpertBytes(lb LayerBytes) float64 {
	var b float64
	for _, l := range lb.Layers {
		b += float64(l.Experts) * float64(l.BytesPerExpert)
	}
	return b
}

// ParallelMemory lays out what each card of the group holds: a P-th of
// every layer's experts and of the dense weights, the caches split by
// heads, the fixed terms (the first card holds the embedding and the
// prefill's inputs, as the first range does today; the others the head's
// share of the last range's), the drafter's share when drafting.
func ParallelMemory(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, mc MemoryCost) MemoryLedger {
	cards := p.Cards()
	gpus := []int{p.Ranges[0].GPU}
	if p.Parallel != nil && len(p.Parallel.GPUs) > 0 {
		gpus = p.Parallel.GPUs
	}
	var dense, experts float64
	layers := 0
	for _, l := range lb.Layers {
		layers++
		dense += float64(l.DenseBytes)
		experts += float64(l.Experts) * float64(l.BytesPerExpert)
	}
	dense += float64(lb.HeadBytes) + float64(lb.DrafterDenseBytes)
	budget := mc.TotalGB - mc.HeadroomGB
	ledger := MemoryLedger{Fits: true}
	worst := 0.0
	for i, gpu := range gpus {
		g := GPUMemory{GPU: gpu, Layers: p.Ranges[0].Layers, BudgetGB: budget}
		g.DenseGB = dense / 1e9 / float64(cards)
		g.ExpertsGB = experts / 1e9 / float64(cards)
		g.Resident = 0
		for _, l := range lb.Layers {
			g.Resident += l.Experts / cards
		}
		if i == 0 {
			g.FixedGB = mc.FixedFirstGB
		} else {
			g.FixedGB = mc.FixedLastGB
		}
		g.BatchesGB = float64(w.Batches) * float64(w.Rows) * float64(layers) * mc.PerRowLayerMB / 1e3 / float64(cards)
		if w.Drafter {
			g.DrafterGB = (float64(w.Batches)*float64(w.Rows)*float64(layers)*mc.DrafterPerRowLayerMB/1e3 + mc.DrafterFixedLastGB) / float64(cards)
		}
		g.TotalGB = g.DenseGB + g.ExpertsGB + g.FixedGB + g.BatchesGB + g.DrafterGB
		g.Fits = g.TotalGB <= budget
		if !g.Fits {
			g.ShortfallGB = g.TotalGB - budget
			ledger.Fits = false
		}
		worst = math.Max(worst, g.TotalGB)
		g.Notes = append(g.Notes, fmt.Sprintf("a %d-th of the experts (%.1f GB) and of the dense weights; the caches split by heads (estimated)", cards, g.ExpertsGB))
		ledger.GPUs = append(ledger.GPUs, g)
	}
	if ledger.Fits {
		ledger.Why = fmt.Sprintf("%d rows over %d cards fit: the fullest card holds %.1f of %.1f GB", w.Rows, cards, worst, budget)
	} else {
		ledger.Why = fmt.Sprintf("%d rows over %d cards do not fit: the fullest card holds %.1f of %.1f GB", w.Rows, cards, worst, budget)
	}
	return ledger
}

// ParallelPlan is the tensor-parallel counterpart of a layer-sharded plan:
// one range over all its layers on the group's first card, the group
// being the plan's GPUs in step order, the dense weights split across them
// and the experts spread over them.
func ParallelPlan(p *kernelplan.PipelinePlan, reduce string) *kernelplan.PipelinePlan {
	var gpus []int
	for _, r := range p.Ranges {
		gpus = append(gpus, r.GPU)
	}
	q := *p
	q.Ranges = []kernelplan.Range{{GPU: gpus[0], Layers: fmt.Sprintf("0:%d", p.Layers())}}
	q.Pipeline = kernelplan.Pipeline{Batches: 1, Schedule: kernelplan.ScheduleRangeOrder}
	q.Experts = ""
	q.Parallel = &kernelplan.Parallel{DenseTP: len(gpus), ExpertEP: len(gpus), GPUs: gpus, Experts: "by-id", Reduce: reduce}
	return &q
}

// ServingOption is an Option with the shape's parallelism named.
type ServingOption struct {
	Option
	Parallel *kernelplan.Parallel `json:"parallel,omitempty"`
	Memory   MemoryLedger         `json:"memory"`
}

// Describe names the option in a line, its parallelism included.
func (o ServingOption) Describe() string {
	if o.Parallel == nil || o.Parallel.DenseTP <= 1 {
		return o.Option.Describe() + ", layers sharded"
	}
	return o.Option.Describe() + fmt.Sprintf(", dense split %d ways and experts over %d cards", o.Parallel.DenseTP, o.Parallel.ExpertEP)
}

// ChooseServing picks the serving shape for a load among the layer-sharded
// shapes (ChooseShape: one batch or one per range, each draft length and
// policy) and the tensor-parallel ones (one batch over every card, the
// same draft lengths and policies), the one the model predicts to yield
// the most tokens a second within the latency bound and the cards'
// memory, and returns it first with every other option after it, best
// first, each rejected one saying why.
func ChooseServing(lb LayerBytes, p *kernelplan.PipelinePlan, load Load, c PipelineCost, sc SpecCost, acc Acceptance,
	pc ParallelCost, mc MemoryCost) ([]ServingOption, error) {
	if mc.TotalGB == 0 {
		mc = RTXPro6000Memory
	}
	sharded, err := ChooseShape(lb, p, load, c, sc, acc)
	if err != nil && len(sharded) == 0 {
		return nil, err
	}
	var options []ServingOption
	for _, o := range sharded {
		o.Rejected = ""
		q := *p
		q.Pipeline.Batches = o.Batches
		mem, err := PredictMemory(lb, &q, Workload{Rows: o.Rows, Batches: o.Batches, Drafter: o.Speculation != nil && o.Speculation.Draft > 0}, nil, pc.MaxRows, mc)
		if err != nil {
			return nil, err
		}
		options = append(options, ServingOption{Option: o, Memory: mem})
	}
	if len(p.Ranges) > 1 {
		tp := ParallelPlan(p, "")
		var specs []*kernelplan.Speculation
		specs = append(specs, nil)
		if load.Drafter {
			for k := 1; k <= load.MaxDraft; k++ {
				specs = append(specs, &kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyFixed})
				if k >= 2 {
					specs = append(specs, &kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyConfidence, MinConfidence: 0.5})
				}
			}
		}
		for _, s := range specs {
			spec := kernelplan.Speculation{Policy: kernelplan.PolicyFixed}
			if s != nil {
				spec = *s
			}
			w := Workload{Rows: load.Rows, Batches: 1, Drafter: spec.Draft > 0}
			pred, err := PredictParallel(lb, tp, w, load.Mix, spec, c, sc, acc, pc)
			if err != nil {
				continue // wider than the library runs under tensor parallelism
			}
			if load.NewTokens > 0 {
				run := acc.Completion(load.Rows, load.NewTokens, int(math.Round(pred.DraftedPerRow)))
				run.TokS = pred.AggregateTokS * run.Share
				pred.FixedLength = &run
			}
			options = append(options, ServingOption{Option: Option{Batches: 1, Rows: load.Rows, Speculation: s, Prediction: pred},
				Parallel: tp.Parallel, Memory: ParallelMemory(lb, tp, w, mc)})
		}
	}
	sort.SliceStable(options, func(i, j int) bool {
		return options[i].Prediction.AggregateTokS > options[j].Prediction.AggregateTokS
	})
	chosen := -1
	for i := range options {
		o := &options[i]
		if load.LatencyMs > 0 && o.Prediction.MsPerToken > load.LatencyMs {
			o.Rejected = fmt.Sprintf("%.1f ms a token, over the %.1f ms bound", o.Prediction.MsPerToken, load.LatencyMs)
			continue
		}
		if !o.Memory.Fits {
			o.Rejected = "over the cards' memory: " + o.Memory.Why
			continue
		}
		if chosen < 0 {
			chosen = i
			continue
		}
		o.Rejected = fmt.Sprintf("%.0f tok/s, %.0f%% under the choice's %.0f", o.Prediction.AggregateTokS,
			(1-o.Prediction.AggregateTokS/options[chosen].Prediction.AggregateTokS)*100, options[chosen].Prediction.AggregateTokS)
	}
	if chosen < 0 {
		return options, fmt.Errorf("no shape serves %d rows within %.1f ms a token and the cards' memory", load.Rows, load.LatencyMs)
	}
	if chosen > 0 {
		options[0], options[chosen] = options[chosen], options[0]
	}
	return options, nil
}
