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
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// PhaseNode represents one stamped execution node in the in-graph phase trace.
type PhaseNode struct {
	Name          string   `json:"name"`                      // unique node or site name
	Kind          string   `json:"kind,omitempty"`            // class: dense, collective, experts, island, fused, etc.
	Class         string   `json:"class,omitempty"`           // normalized class if distinct from Kind
	Stream        int      `json:"stream,omitempty"`          // 0 = main stream, >0 = parallel stream
	SideBranch    bool     `json:"side_branch,omitempty"`     // true if node executes on a side branch
	StartNs       int64    `json:"start_ns,omitempty"`        // timestamp of start in nanoseconds
	EndNs         int64    `json:"end_ns,omitempty"`          // timestamp of end in nanoseconds
	DurationUs    float64  `json:"duration_us,omitempty"`     // duration in microseconds
	DurationMs    float64  `json:"duration_ms,omitempty"`     // duration in milliseconds
	OverlappedMs  float64  `json:"overlapped_ms,omitempty"`   // concurrent main stream duration overlapping this node
	Rows          int      `json:"rows,omitempty"`            // token rows this program was executed with
	FusedRegion   string   `json:"fused_region,omitempty"`    // name of fused region this node belongs to
	IsFusedParent bool     `json:"is_fused_parent,omitempty"` // true if this node is the in-place cost of the fused region
	WaitMs        float64  `json:"wait_ms,omitempty"`         // collective wait measured at this row count
	Dependencies  []string `json:"dependencies,omitempty"`    // upstream node dependencies
}

// CostMs returns the measured duration of the node in milliseconds.
func (n *PhaseNode) CostMs() float64 {
	if n.DurationMs > 0 {
		return n.DurationMs
	}
	if n.DurationUs > 0 {
		return n.DurationUs / 1e3
	}
	if n.EndNs > n.StartNs {
		return float64(n.EndNs-n.StartNs) / 1e6
	}
	return 0
}

// EffectiveClass returns the node's classification category.
func (n *PhaseNode) EffectiveClass() string {
	if n.Class != "" {
		return n.Class
	}
	if n.Kind != "" {
		return n.Kind
	}
	return "dense"
}

// Shape describes the serving configuration: row count and stream count.
type Shape struct {
	Name    string `json:"name,omitempty"`
	Rows    int    `json:"rows"`
	Streams int    `json:"streams"`
}

// ParseShape parses a shape name or row/stream specification.
func ParseShape(s string) Shape {
	name := strings.TrimSpace(strings.ToLower(s))
	switch name {
	case "five-row":
		return Shape{Name: "five-row", Rows: 5, Streams: 1}
	case "block-7":
		return Shape{Name: "block-7", Rows: 8, Streams: 1}
	case "8-stream":
		return Shape{Name: "8-stream", Rows: 40, Streams: 8}
	}
	sh := Shape{Name: s, Rows: 1, Streams: 1}
	parts := strings.Split(s, ",")
	for _, p := range parts {
		kv := strings.Split(strings.TrimSpace(p), "=")
		if len(kv) == 2 {
			k, v := strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1])
			val, err := strconv.Atoi(v)
			if err != nil {
				continue
			}
			if k == "rows" {
				sh.Rows = val
			} else if k == "streams" {
				sh.Streams = val
			}
		}
	}
	return sh
}

// CriticalPathNodeSummary records how a single node was evaluated.
type CriticalPathNodeSummary struct {
	Name             string  `json:"name"`
	Class            string  `json:"class"`
	Stream           int     `json:"stream"`
	SideBranch       bool    `json:"side_branch"`
	RawCostMs        float64 `json:"raw_cost_ms"`
	WallCostMs       float64 `json:"wall_cost_ms"`
	OverlappedMs     float64 `json:"overlapped_ms,omitempty"`
	FusedRegion      string  `json:"fused_region,omitempty"`
	FusedPartSkipped bool    `json:"fused_part_skipped,omitempty"`
	CollectiveWaitMs float64 `json:"collective_wait_ms,omitempty"`
}

// CriticalPathPrediction holds the cost model's predictions.
type CriticalPathPrediction struct {
	Shape           string                    `json:"shape"`
	Rows            int                       `json:"rows"`
	Streams         int                       `json:"streams"`
	TreeCommit      string                    `json:"tree_commit,omitempty"`
	PredictedStepMs float64                   `json:"predicted_step_ms"`
	CriticalNodes   []CriticalPathNodeSummary `json:"critical_nodes,omitempty"`
	ByClassMs       map[string]float64        `json:"by_class_ms"`
	SideBranchCost  float64                   `json:"side_branch_cost_ms"`
	HiddenWorkMs    float64                   `json:"hidden_work_ms"`
	FusedSavingsMs  float64                   `json:"fused_savings_ms"`
	CollectiveWait  float64                   `json:"collective_wait_ms"`
}

// CriticalPathCostModel predicts step time per plan and shape from in-graph stamps.
type CriticalPathCostModel struct {
	StrictRowCheck bool
}

// NewCriticalPathCostModel creates a new critical path cost model v0.
func NewCriticalPathCostModel() *CriticalPathCostModel {
	return &CriticalPathCostModel{StrictRowCheck: true}
}

// Predict calculates the predicted step time on the critical path.
func (m *CriticalPathCostModel) Predict(nodes []PhaseNode, shape Shape) (*CriticalPathPrediction, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("cost model: cannot predict step time from empty nodes list")
	}

	pred := &CriticalPathPrediction{
		Shape:     shape.Name,
		Rows:      shape.Rows,
		Streams:   shape.Streams,
		ByClassMs: make(map[string]float64),
	}
	if pred.Shape == "" {
		pred.Shape = fmt.Sprintf("rows=%d,streams=%d", shape.Rows, shape.Streams)
	}

	// 1. Verify row counts if strict checking is enabled.
	if m.StrictRowCheck && shape.Rows > 0 {
		for _, n := range nodes {
			if n.Rows > 0 && n.Rows != shape.Rows {
				return nil, fmt.Errorf("cost model: node %q recorded at %d rows, but plan shape runs %d rows: knobs change sign across row count",
					n.Name, n.Rows, shape.Rows)
			}
		}
	}

	// 2. Identify fused regions.
	// A fused region is costed in place once (not as its parts).
	// We locate parent nodes or the first node of each fused region.
	fusedRegionHandled := make(map[string]bool)
	fusedRegionPartsSum := make(map[string]float64)
	fusedRegionInPlaceCost := make(map[string]float64)

	for _, n := range nodes {
		if n.FusedRegion != "" {
			if n.IsFusedParent {
				fusedRegionInPlaceCost[n.FusedRegion] = n.CostMs()
			} else {
				fusedRegionPartsSum[n.FusedRegion] += n.CostMs()
			}
		}
	}

	// 3. Identify main-stream interval bounds for timestamped timeline evaluation.
	type span struct {
		start int64
		end   int64
	}
	var mainSpans []span
	for _, n := range nodes {
		isSide := n.SideBranch || n.Stream > 0 || n.Kind == "island"
		if !isSide && n.EndNs > n.StartNs {
			mainSpans = append(mainSpans, span{start: n.StartNs, end: n.EndNs})
		}
	}

	// 4. Cost critical path nodes.
	for _, n := range nodes {
		rawCost := n.CostMs()
		class := n.EffectiveClass()
		isSide := n.SideBranch || n.Stream > 0 || n.Kind == "island"

		// Handle fused region membership.
		if n.FusedRegion != "" {
			if !n.IsFusedParent && fusedRegionHandled[n.FusedRegion] {
				// Constituent part of a fused region already costed in place; skip part.
				pred.CriticalNodes = append(pred.CriticalNodes, CriticalPathNodeSummary{
					Name:             n.Name,
					Class:            class,
					Stream:           n.Stream,
					SideBranch:       isSide,
					RawCostMs:        rawCost,
					WallCostMs:       0,
					FusedRegion:      n.FusedRegion,
					FusedPartSkipped: true,
				})
				continue
			}
			if !n.IsFusedParent && !fusedRegionHandled[n.FusedRegion] && fusedRegionInPlaceCost[n.FusedRegion] > 0 {
				// Parent exists elsewhere; skip part.
				pred.CriticalNodes = append(pred.CriticalNodes, CriticalPathNodeSummary{
					Name:             n.Name,
					Class:            class,
					Stream:           n.Stream,
					SideBranch:       isSide,
					RawCostMs:        rawCost,
					WallCostMs:       0,
					FusedRegion:      n.FusedRegion,
					FusedPartSkipped: true,
				})
				continue
			}
			fusedRegionHandled[n.FusedRegion] = true
		}

		var wallCost float64
		var overlappedMs float64

		if isSide {
			// A node on a side branch contributes only what it adds to the wall.
			if n.EndNs > n.StartNs && len(mainSpans) > 0 {
				// Find overlapping main stream work.
				var latestMainEnd int64
				for _, mSpan := range mainSpans {
					if mSpan.start <= n.EndNs && mSpan.end >= n.StartNs {
						if mSpan.end > latestMainEnd {
							latestMainEnd = mSpan.end
						}
					}
				}
				if latestMainEnd >= n.EndNs {
					// Fully overlapped: side branch does not count towards wall.
					wallCost = 0
					overlappedMs = rawCost
				} else if latestMainEnd > n.StartNs {
					// Partially overlapped: adds only the uncovered tail.
					wallAdditionNs := n.EndNs - latestMainEnd
					wallCost = float64(wallAdditionNs) / 1e6
					overlappedMs = rawCost - wallCost
				} else {
					// Not overlapped by main stream.
					wallCost = rawCost
				}
			} else if n.OverlappedMs > 0 {
				wallCost = math.Max(0, rawCost-n.OverlappedMs)
				overlappedMs = n.OverlappedMs
			} else {
				// Marked as side branch without wall addition: hidden work.
				wallCost = 0
				overlappedMs = rawCost
			}
			pred.SideBranchCost += wallCost
			pred.HiddenWorkMs += overlappedMs
		} else {
			// Main stream / critical path node.
			wallCost = rawCost
		}

		// Account for collective wait measured at this row count.
		var nodeCollectiveWait float64
		if n.WaitMs > 0 {
			nodeCollectiveWait = n.WaitMs
			pred.CollectiveWait += n.WaitMs
		}

		pred.PredictedStepMs += wallCost
		pred.ByClassMs[class] += wallCost

		pred.CriticalNodes = append(pred.CriticalNodes, CriticalPathNodeSummary{
			Name:             n.Name,
			Class:            class,
			Stream:           n.Stream,
			SideBranch:       isSide,
			RawCostMs:        rawCost,
			WallCostMs:       wallCost,
			OverlappedMs:     overlappedMs,
			FusedRegion:      n.FusedRegion,
			CollectiveWaitMs: nodeCollectiveWait,
		})
	}

	// Calculate savings from in-place fusion.
	for reg, inPlaceCost := range fusedRegionInPlaceCost {
		if partsCost, ok := fusedRegionPartsSum[reg]; ok && partsCost > inPlaceCost {
			pred.FusedSavingsMs += (partsCost - inPlaceCost)
		}
	}

	return pred, nil
}

// RunRecord represents one run entry in the ledger.
type RunRecord struct {
	Format        string             `json:"format"`
	TreeCommit    string             `json:"tree_commit,omitempty"`
	Commit        string             `json:"commit,omitempty"`
	ArtifactMD5   string             `json:"artifact_md5,omitempty"`
	Artifact      string             `json:"artifact,omitempty"`
	PlanHash      string             `json:"plan_hash,omitempty"`
	Plan          string             `json:"plan,omitempty"`
	Shape         string             `json:"shape,omitempty"`
	Rows          int                `json:"rows,omitempty"`
	Streams       int                `json:"streams,omitempty"`
	Knobs         map[string]any     `json:"knobs,omitempty"`
	StepMs        float64            `json:"step_ms"`
	MedianStepMs  float64            `json:"median_step_ms,omitempty"`
	TokensPerStep float64            `json:"tokens_per_step,omitempty"`
	TokensAStep   float64            `json:"tokens_a_step,omitempty"`
	Checks        any                `json:"checks,omitempty"`
	PeakMemoryMB  map[string]float64 `json:"peak_memory_mb,omitempty"`
	BuildTimeS    float64            `json:"build_time_s,omitempty"`
	Nodes         []PhaseNode        `json:"nodes,omitempty"`
	PhaseTrace    []PhaseNode        `json:"phase_trace,omitempty"`
	ClassTimesMs  map[string]float64 `json:"class_times_ms,omitempty"`
}

// GetStepMs returns the actual measured median step time.
func (r *RunRecord) GetStepMs() float64 {
	if r.MedianStepMs > 0 {
		return r.MedianStepMs
	}
	return r.StepMs
}

// GetCommit returns the tree commit.
func (r *RunRecord) GetCommit() string {
	if r.TreeCommit != "" {
		return r.TreeCommit
	}
	return r.Commit
}

// GetShape returns the shape string.
func (r *RunRecord) GetShape() string {
	if r.Shape != "" {
		return r.Shape
	}
	return fmt.Sprintf("rows=%d,streams=%d", r.Rows, r.Streams)
}

// GetNodes returns the in-graph stamped nodes.
func (r *RunRecord) GetNodes() []PhaseNode {
	if len(r.Nodes) > 0 {
		return r.Nodes
	}
	return r.PhaseTrace
}

// ReadLedger reads run records from a file (supporting JSON Lines and JSON arrays).
func ReadLedger(path string) ([]RunRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read ledger %s: %w", path, err)
	}
	return ParseLedger(data)
}

// ParseLedger parses ledger bytes into RunRecords.
func ParseLedger(data []byte) ([]RunRecord, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}

	// Try JSON array first.
	if trimmed[0] == '[' {
		var records []RunRecord
		if err := json.Unmarshal(trimmed, &records); err != nil {
			return nil, fmt.Errorf("unmarshal JSON array ledger: %w", err)
		}
		return records, nil
	}

	// Try JSON Lines.
	var records []RunRecord
	scanner := bufio.NewScanner(bytes.NewReader(trimmed))
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		var r RunRecord
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, fmt.Errorf("line %d: invalid JSON run record: %w", lineNum, err)
		}
		records = append(records, r)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read ledger lines: %w", err)
	}
	return records, nil
}

// RecordError tracks prediction error for a single record.
type RecordError struct {
	RunIndex     int     `json:"run_index"`
	ActualStepMs float64 `json:"actual_step_ms"`
	PredictedMs  float64 `json:"predicted_ms"`
	ErrorMs      float64 `json:"error_ms"`
	ErrorPct     float64 `json:"error_pct"`
	AbsolutePct  float64 `json:"absolute_pct"`
}

// ErrorDistribution captures statistical distribution of prediction errors.
type ErrorDistribution struct {
	Count     int     `json:"count"`
	MinPct    float64 `json:"min_pct"`
	P25Pct    float64 `json:"p25_pct"`
	MedianPct float64 `json:"median_pct"`
	P75Pct    float64 `json:"p75_pct"`
	MaxPct    float64 `json:"max_pct"`
	MeanPct   float64 `json:"mean_pct"`
	MedianMs  float64 `json:"median_error_ms"`
}

// VerificationResult holds the ledger verification outcome for one tree and shape.
type VerificationResult struct {
	Tree           string            `json:"tree"`
	Shape          string            `json:"shape"`
	Rows           int               `json:"rows"`
	Streams        int               `json:"streams"`
	RecordsCount   int               `json:"records_count"`
	PredictedMs    float64           `json:"predicted_ms"`
	ActualMedianMs float64           `json:"actual_median_ms"`
	BandPct        float64           `json:"band_pct"`
	MedianErrorMs  float64           `json:"median_error_ms"`
	MedianErrorPct float64           `json:"median_error_pct"`
	Distribution   ErrorDistribution `json:"distribution"`
	Accepted       bool              `json:"accepted"`
	Errors         []RecordError     `json:"errors"`
}

// VerifyLedger compares model predictions against run records in the ledger for matching tree and shape.
func VerifyLedger(records []RunRecord, targetTree string, targetShape string, bandPct float64) ([]VerificationResult, error) {
	if len(records) == 0 {
		return nil, fmt.Errorf("ledger verification: no run records provided")
	}

	type groupKey struct {
		tree  string
		shape string
	}
	groups := make(map[groupKey][]RunRecord)

	for _, r := range records {
		tree := r.GetCommit()
		sh := r.GetShape()
		if targetTree != "" && tree != targetTree {
			continue
		}
		if targetShape != "" && sh != targetShape {
			continue
		}
		key := groupKey{tree: tree, shape: sh}
		groups[key] = append(groups[key], r)
	}

	if len(groups) == 0 {
		return nil, fmt.Errorf("ledger verification: no records matched tree %q and shape %q", targetTree, targetShape)
	}

	model := NewCriticalPathCostModel()
	var results []VerificationResult

	// Sort group keys for deterministic output.
	var keys []groupKey
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].tree != keys[j].tree {
			return keys[i].tree < keys[j].tree
		}
		return keys[i].shape < keys[j].shape
	})

	for _, k := range keys {
		recs := groups[k]
		first := recs[0]
		sh := ParseShape(k.shape)
		if first.Rows > 0 {
			sh.Rows = first.Rows
		}
		if first.Streams > 0 {
			sh.Streams = first.Streams
		}

		// Read in-graph stamped nodes from record.
		nodes := first.GetNodes()
		if len(nodes) == 0 {
			return nil, fmt.Errorf("record for %s/%s lacks in-graph stamps (nodes)", k.tree, k.shape)
		}

		pred, err := model.Predict(nodes, sh)
		if err != nil {
			return nil, fmt.Errorf("predict %s/%s: %w", k.tree, k.shape, err)
		}

		// Set default error band per shape if not overridden:
		// 1% for one stream, stated per shape.
		effectiveBand := bandPct
		if effectiveBand <= 0 {
			if sh.Streams <= 1 {
				effectiveBand = 1.0 // 1 percent for 1 stream
			} else {
				effectiveBand = 2.0 // width band
			}
		}

		var actuals []float64
		var errorsList []RecordError
		var absErrorsPct []float64

		for idx, r := range recs {
			actual := r.GetStepMs()
			actuals = append(actuals, actual)
			errMs := pred.PredictedStepMs - actual
			errPct := (errMs / actual) * 100.0
			absPct := math.Abs(errPct)
			absErrorsPct = append(absErrorsPct, absPct)

			errorsList = append(errorsList, RecordError{
				RunIndex:     idx,
				ActualStepMs: actual,
				PredictedMs:  pred.PredictedStepMs,
				ErrorMs:      errMs,
				ErrorPct:     errPct,
				AbsolutePct:  absPct,
			})
		}

		sort.Float64s(actuals)
		sort.Float64s(absErrorsPct)

		actualMedian := computeMedian(actuals)
		dist := computeDistribution(absErrorsPct)
		dist.MedianMs = math.Abs(pred.PredictedStepMs - actualMedian)

		accepted := dist.MedianPct <= effectiveBand

		res := VerificationResult{
			Tree:           k.tree,
			Shape:          k.shape,
			Rows:           sh.Rows,
			Streams:        sh.Streams,
			RecordsCount:   len(recs),
			PredictedMs:    pred.PredictedStepMs,
			ActualMedianMs: actualMedian,
			BandPct:        effectiveBand,
			MedianErrorMs:  dist.MedianMs,
			MedianErrorPct: dist.MedianPct,
			Distribution:   dist,
			Accepted:       accepted,
			Errors:         errorsList,
		}
		results = append(results, res)
	}

	return results, nil
}

func computeMedian(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return vals[n/2]
	}
	return (vals[n/2-1] + vals[n/2]) / 2.0
}

func computeDistribution(vals []float64) ErrorDistribution {
	n := len(vals)
	if n == 0 {
		return ErrorDistribution{}
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return ErrorDistribution{
		Count:     n,
		MinPct:    vals[0],
		P25Pct:    percentile(vals, 0.25),
		MedianPct: computeMedian(vals),
		P75Pct:    percentile(vals, 0.75),
		MaxPct:    vals[n-1],
		MeanPct:   sum / float64(n),
	}
}

func percentile(sortedVals []float64, p float64) float64 {
	n := len(sortedVals)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sortedVals[0]
	}
	idx := p * float64(n-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	weight := idx - float64(lower)
	return sortedVals[lower]*(1-weight) + sortedVals[upper]*weight
}

// FormatVerificationReport prints human-readable verification error distributions.
func FormatVerificationReport(w io.Writer, results []VerificationResult) error {
	if _, err := fmt.Fprintf(w, "Cost Model v0 Verification against Ledger:\n"); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "%-10s %-12s %7s %12s %12s %10s %8s %10s\n",
		"TREE", "SHAPE", "RECORDS", "PREDICTED", "ACTUAL_MED", "MED_ERR%", "BAND%", "STATUS"); err != nil {
		return err
	}
	for _, r := range results {
		status := "ACCEPTED"
		if !r.Accepted {
			status = "REJECTED"
		}
		if _, err := fmt.Fprintf(w, "%-10s %-12s %7d %10.2f ms %10.2f ms %9.2f%% %7.1f%% %10s\n",
			r.Tree, r.Shape, r.RecordsCount, r.PredictedMs, r.ActualMedianMs, r.MedianErrorPct, r.BandPct, status); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "\nError Distributions (|Predicted - Actual| / Actual):\n"); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "%-10s %-12s %8s %8s %8s %8s %8s %8s\n",
		"TREE", "SHAPE", "MIN%", "P25%", "MEDIAN%", "P75%", "MAX%", "MEAN%"); err != nil {
		return err
	}
	for _, r := range results {
		if _, err := fmt.Fprintf(w, "%-10s %-12s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
			r.Tree, r.Shape, r.Distribution.MinPct, r.Distribution.P25Pct, r.Distribution.MedianPct,
			r.Distribution.P75Pct, r.Distribution.MaxPct, r.Distribution.MeanPct); err != nil {
			return err
		}
	}
	return nil
}
