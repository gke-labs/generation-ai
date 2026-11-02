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
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// The serving shape (hero run 10). Given rows to serve at once and a
// latency budget, the planner lays the rows out as one wide batch or as
// several batches in flight, one per range, predicts each layout with
// the pipeline cost model, drops the layouts over the budget and keeps
// the highest aggregate. Rows in one batch share every expert read and
// batches do not, but one batch streams one GPU at a time while batches
// in flight stream them all; which wins depends on how many distinct
// experts the rows select, so the choice needs the distinct-expert count
// at each candidate width (DistinctCurve).

// DistinctPoint is a measured count of distinct experts a layer at a
// batch width with a prompt per row (VXPU_EXPERT_STATS=1).
type DistinctPoint struct {
	Rows     int
	Distinct float64
}

// DistinctCurve estimates the distinct experts a layer a batch of rows
// selects with a prompt per row, from measured points: a power law
// between neighbouring points (and from one row at top-k to the first
// point), the last point's slope beyond it, never more than the
// rows*topK the rows can select or the experts a layer has. With no
// points every row is taken to select the same experts: top-k.
type DistinctCurve struct {
	Points  []DistinctPoint
	TopK    int
	Experts int
}

// ParseDistinctPoints reads "9=37.3,33=100".
func ParseDistinctPoints(spec string) ([]DistinctPoint, error) {
	var pts []DistinctPoint
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		rows, distinct, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("distinct experts %q: want ROWS=COUNT", item)
		}
		r, err := strconv.Atoi(strings.TrimSpace(rows))
		if err != nil || r < 1 {
			return nil, fmt.Errorf("distinct experts %q: rows must be a positive integer", item)
		}
		d, err := strconv.ParseFloat(strings.TrimSpace(distinct), 64)
		if err != nil || d <= 0 {
			return nil, fmt.Errorf("distinct experts %q: the count must be a positive number", item)
		}
		pts = append(pts, DistinctPoint{Rows: r, Distinct: d})
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i].Rows < pts[j].Rows })
	return pts, nil
}

// At is the curve's estimate for a batch of rows.
func (dc DistinctCurve) At(rows int) float64 {
	topK := float64(dc.TopK)
	bound := math.Min(float64(rows)*topK, float64(dc.Experts))
	if len(dc.Points) == 0 || rows <= 1 {
		return math.Min(topK, bound)
	}
	pts := append([]DistinctPoint{{Rows: 1, Distinct: topK}}, dc.Points...)
	// the segment the width falls in, or the last one extended
	i := len(pts) - 1
	for k := 1; k < len(pts); k++ {
		if rows <= pts[k].Rows {
			i = k
			break
		}
	}
	a, b := pts[i-1], pts[i]
	if a.Rows == b.Rows || a.Distinct <= 0 {
		return math.Min(b.Distinct, bound)
	}
	exp := math.Log(b.Distinct/a.Distinct) / math.Log(float64(b.Rows)/float64(a.Rows))
	est := a.Distinct * math.Pow(float64(rows)/float64(a.Rows), exp)
	return math.Min(est, bound)
}

// ShapeCandidate is one layout of the rows and its prediction.
type ShapeCandidate struct {
	Batches      int                `json:"batches"`
	RowsPerBatch int                `json:"rows_per_batch"`
	Distinct     float64            `json:"distinct_experts"`
	LatencyMs    float64            `json:"latency_ms"`
	AggregateTS  float64            `json:"aggregate_tok_s"` // for the rows asked for, not the padded width
	Fits         bool               `json:"fits"`            // within the latency budget, the kernels' row limit and the GPUs' memory
	Runnable     bool               `json:"runnable"`        // no wider than the kernels' row limit
	Prediction   PipelinePrediction `json:"prediction"`
	Memory       MemoryLedger       `json:"memory"`
}

// LayoutMemory is the memory calibration and residency ChooseLayout prices
// each candidate with; the zero value means the RTX PRO 6000's calibration
// with every expert of a range resident.
type LayoutMemory struct {
	Cost      MemoryCost
	Residency Residency
}

// MaxBatchRows is the widest batch the kernel library runs as planned:
// the one-pass GEMV kernels and the fused bodies apply to at most this
// many rows (VXPU_GEMV_ROWS_MAX), and above it the lowering falls back to
// reference bodies that the cost model does not describe (hero run 10
// measured 209 ms at 66 rows against the model's 104). A wider batch is
// not a candidate until a GEMM template exists.
const MaxBatchRows = 64

// ShapeChoice is the planner's pick and the candidates it was picked from.
type ShapeChoice struct {
	Rows       int              `json:"rows"`
	LatencyMs  float64          `json:"latency_budget_ms"`
	Chosen     ShapeCandidate   `json:"chosen"`
	Candidates []ShapeCandidate `json:"candidates"`
	Why        string           `json:"why"`
}

// ChooseLayout lays rows out as 1..maxBatches batches in flight over plan
// p (rows split evenly, the prediction at the widest batch), predicts each
// with the pipeline model, drops the layouts whose per-batch latency is
// over latencyMs (0: no budget) or whose batch is wider than maxRows (0:
// MaxBatchRows) and returns the highest aggregate among the rest; when
// nothing fits, the lowest latency among the runnable layouts, marked so.
func ChooseLayout(lb LayerBytes, p *kernelplan.PipelinePlan, rows, maxBatches int, latencyMs float64, dc DistinctCurve, drafter bool, c PipelineCost, maxRows int) (ShapeChoice, error) {
	return ChooseLayoutWithMemory(lb, p, rows, maxBatches, latencyMs, dc, drafter, c, maxRows, LayoutMemory{})
}

// ChooseLayoutWithMemory is ChooseLayout with the memory ledger priced per
// candidate: a layout whose batches do not fit the GPUs is not a fit.
func ChooseLayoutWithMemory(lb LayerBytes, p *kernelplan.PipelinePlan, rows, maxBatches int, latencyMs float64, dc DistinctCurve, drafter bool, c PipelineCost, maxRows int, mem LayoutMemory) (ShapeChoice, error) {
	mc := mem.Cost
	if mc.TotalGB == 0 {
		mc = RTXPro6000Memory
	}
	if rows < 1 {
		return ShapeChoice{}, fmt.Errorf("choose shape: rows %d: want at least 1", rows)
	}
	if maxBatches < 1 {
		maxBatches = len(p.Ranges)
	}
	if maxRows < 1 {
		maxRows = MaxBatchRows
	}
	choice := ShapeChoice{Rows: rows, LatencyMs: latencyMs}
	for n := 1; n <= maxBatches && n <= rows; n++ {
		per := (rows + n - 1) / n
		w := Workload{Rows: per, DistinctExperts: dc.At(per), Batches: n, Drafter: drafter}
		q := *p // the candidate's plan: the same ranges with n batches in flight
		q.Pipeline.Batches = n
		pr, err := PredictPipeline(lb, &q, w, c)
		if err != nil {
			return ShapeChoice{}, err
		}
		cand := ShapeCandidate{Batches: n, RowsPerBatch: per, Distinct: w.DistinctExperts, LatencyMs: pr.LatencyMs,
			AggregateTS: pr.AggregateTokS * float64(rows) / float64(n*per), Prediction: pr}
		cand.Runnable = per <= maxRows
		if cand.Memory, err = PredictMemory(lb, &q, w, mem.Residency, maxRows, mc); err != nil {
			return ShapeChoice{}, err
		}
		cand.Fits = cand.Runnable && (latencyMs <= 0 || pr.LatencyMs <= latencyMs) && cand.Memory.Fits
		choice.Candidates = append(choice.Candidates, cand)
	}
	best := -1
	for i, cand := range choice.Candidates {
		if !cand.Fits {
			continue
		}
		if best < 0 || cand.AggregateTS > choice.Candidates[best].AggregateTS {
			best = i
		}
	}
	if best < 0 {
		for i, cand := range choice.Candidates {
			if !cand.Runnable && best >= 0 && choice.Candidates[best].Runnable {
				continue
			}
			if best < 0 || (cand.Runnable && !choice.Candidates[best].Runnable) || cand.LatencyMs < choice.Candidates[best].LatencyMs {
				best = i
			}
		}
		choice.Chosen = choice.Candidates[best]
		if !choice.Chosen.Runnable {
			choice.Why = fmt.Sprintf("no layout of %d rows over %d batches keeps a batch within the kernels' %d-row limit: the narrowest is %d batches of %d rows",
				rows, maxBatches, maxRows, choice.Chosen.Batches, choice.Chosen.RowsPerBatch)
			return choice, nil
		}
		choice.Why = fmt.Sprintf("no layout of %d rows meets %.0f ms: the lowest latency is %d batches of %d rows at %.1f ms, %.0f tok/s",
			rows, latencyMs, choice.Chosen.Batches, choice.Chosen.RowsPerBatch, choice.Chosen.LatencyMs, choice.Chosen.AggregateTS)
		return choice, nil
	}
	choice.Chosen = choice.Candidates[best]
	others := []string{}
	for _, cand := range choice.Candidates {
		if cand.Batches == choice.Chosen.Batches {
			continue
		}
		fit := "over the latency budget"
		if cand.Fits {
			fit = "fits"
		} else if !cand.Runnable {
			fit = fmt.Sprintf("wider than the kernels' %d-row limit", maxRows)
		} else if !cand.Memory.Fits {
			fit = "over the GPUs' memory: " + cand.Memory.Why
		}
		others = append(others, fmt.Sprintf("%d×%d %.0f tok/s at %.1f ms (%s)", cand.Batches, cand.RowsPerBatch, cand.AggregateTS, cand.LatencyMs, fit))
	}
	choice.Why = fmt.Sprintf("%d rows as %d batches of %d: %.0f tok/s at %.1f ms a batch step, %.0f distinct experts a layer",
		rows, choice.Chosen.Batches, choice.Chosen.RowsPerBatch, choice.Chosen.AggregateTS, choice.Chosen.LatencyMs, choice.Chosen.Distinct)
	if len(others) > 0 {
		choice.Why += "; against " + strings.Join(others, ", ")
	}
	return choice, nil
}
