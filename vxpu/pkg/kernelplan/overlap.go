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

package kernelplan

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

const (
	RunRecordFormat = "vxpu-run-record/1"
)

// NodeSpan represents a node's execution interval on a stream from device timestamps.
type NodeSpan struct {
	Name    string   `json:"name"`
	Kind    string   `json:"kind,omitempty"`
	Class   string   `json:"class,omitempty"`
	Branch  string   `json:"branch,omitempty"` // "main" or "par"
	Ops     []string `json:"ops,omitempty"`
	StartNs int64    `json:"start_ns"`
	EndNs   int64    `json:"end_ns"`
}

// HiddenNode represents a node whose span lies entirely inside another branch's span.
type HiddenNode struct {
	Name        string  `json:"name"`
	Class       string  `json:"class"`
	Inside      string  `json:"inside"`
	InsideClass string  `json:"inside_class,omitempty"`
	DurationMs  float64 `json:"duration_ms"`
	StartNs     int64   `json:"start_ns"`
	EndNs       int64   `json:"end_ns"`
}

// ClassOverlap summarizes execution times and side-branch concurrency for one class.
type ClassOverlap struct {
	BusyMs          float64 `json:"busy_ms"`
	CoveredMs       float64 `json:"covered_ms"`
	SideBranchMs    float64 `json:"side_branch_ms"`
	SideBranchShare float64 `json:"side_branch_share"`
}

// ConcurrencyRecord carries step concurrency metrics and per-class breakdown.
type ConcurrencyRecord struct {
	WallMs      float64                 `json:"wall_ms"`
	BusyMs      float64                 `json:"busy_ms"`
	CoveredMs   float64                 `json:"covered_ms"`
	OverlapMs   float64                 `json:"overlap_ms"`
	IdleMs      float64                 `json:"idle_ms"`
	Ratio       float64                 `json:"ratio"`
	Classes     map[string]ClassOverlap `json:"classes"`
	HiddenNodes []HiddenNode            `json:"hidden_nodes"`
	Steps       []StepConcurrency       `json:"steps,omitempty"`
}

// StepConcurrency carries metrics for one step in a multi-step trace.
type StepConcurrency struct {
	Step        int                     `json:"step"`
	WallMs      float64                 `json:"wall_ms"`
	BusyMs      float64                 `json:"busy_ms"`
	CoveredMs   float64                 `json:"covered_ms"`
	OverlapMs   float64                 `json:"overlap_ms"`
	IdleMs      float64                 `json:"idle_ms"`
	Ratio       float64                 `json:"ratio"`
	Classes     map[string]ClassOverlap `json:"classes"`
	HiddenNodes []HiddenNode            `json:"hidden_nodes"`
}

// RunRecord represents an executor run record file (run.json).
type RunRecord struct {
	Format          string              `json:"format"`
	PlanHash        string              `json:"plan_hash,omitempty"`
	TreeCommit      string              `json:"tree_commit,omitempty"`
	ArtifactMD5     string              `json:"artifact_md5,omitempty"`
	Knobs           map[string]any      `json:"knobs,omitempty"`
	MedianStepMs    float64             `json:"median_step_ms,omitempty"`
	StepTokens      float64             `json:"step_tokens,omitempty"`
	Checks          string              `json:"checks,omitempty"`
	PeakMemoryBytes map[string]int64    `json:"peak_memory_bytes,omitempty"`
	BuildTimeMs     float64             `json:"build_time_ms,omitempty"`
	Concurrency     *ConcurrencyRecord  `json:"concurrency,omitempty"`
	Stats           map[string]any      `json:"stats,omitempty"`
	Sites           []Site              `json:"sites,omitempty"`
	Hardware        Hardware            `json:"hardware,omitempty"`
	Spans           []NodeSpan          `json:"spans,omitempty"`
	Steps           [][]json.RawMessage `json:"steps,omitempty"`
}

// ClassifyNode maps node metadata to one of the Hero 15 timeline classes.
func ClassifyNode(name, kind string, ops []string) string {
	nameLower := strings.ToLower(name)
	opsLower := make([]string, len(ops))
	for i, op := range ops {
		opsLower[i] = strings.ToLower(op)
	}

	if kind == "experts" || (strings.Contains(nameLower, "expert") && !strings.Contains(nameLower, "shared")) {
		return "routed_experts"
	}

	if kind == "allreduce" || kind == "allgather" || kind == "collective" || strings.HasPrefix(nameLower, "allreduce") {
		return "sums"
	}
	for _, op := range opsLower {
		if strings.Contains(op, "allreduce") {
			return "sums"
		}
	}

	if strings.Contains(nameLower, "shared_ffn") || strings.Contains(nameLower, "shared_expert") {
		return "shared_expert"
	}
	for _, op := range opsLower {
		if op == "shared_ffn_front" || op == "shared_ffn_mid" || op == "shared_ffn" || op == "shared_ffn_combine" {
			return "shared_expert"
		}
	}

	if strings.HasPrefix(nameLower, "attn_q") || strings.HasPrefix(nameLower, "window_kv") {
		return "attention_projections"
	}
	for _, op := range opsLower {
		if op == "attn_q" || op == "window_kv" || op == "attn_qkv" {
			return "attention_projections"
		}
	}

	if strings.HasPrefix(nameLower, "attn_core") || strings.HasPrefix(nameLower, "attn_out") {
		return "attention_core"
	}
	for _, op := range opsLower {
		if op == "attn_core" || op == "attn_select" || op == "sparse_attn" || op == "attn_out" || op == "rotary" {
			return "attention_core"
		}
	}

	if strings.HasPrefix(nameLower, "ffn_front") {
		return "ffn_front"
	}
	for _, op := range opsLower {
		if op == "ffn_front" || op == "ffn_front_split" {
			return "ffn_front"
		}
	}

	if strings.HasPrefix(nameLower, "hc_front") {
		return "hc_front"
	}
	for _, op := range opsLower {
		if op == "hc_front" || op == "hc_front_split" || op == "hc_sinkhorn" {
			return "hc_front"
		}
	}

	if kind == "engram" {
		return "engram"
	}
	for _, op := range opsLower {
		if strings.Contains(op, "engram") || strings.Contains(op, "wkv") {
			return "engram"
		}
	}

	if kind == "embed_rows" {
		return "embed"
	}
	for _, op := range opsLower {
		if strings.Contains(op, "embed") {
			return "embed"
		}
	}

	return "dense_other"
}

type interval struct {
	start int64
	end   int64
}

func unionIntervalLength(intervals []interval) int64 {
	var valid []interval
	for _, in := range intervals {
		if in.end > in.start {
			valid = append(valid, in)
		}
	}
	if len(valid) == 0 {
		return 0
	}
	sort.Slice(valid, func(i, j int) bool {
		if valid[i].start == valid[j].start {
			return valid[i].end < valid[j].end
		}
		return valid[i].start < valid[j].start
	})
	var total int64
	currS := valid[0].start
	currE := valid[0].end
	for i := 1; i < len(valid); i++ {
		in := valid[i]
		if in.start <= currE {
			if in.end > currE {
				currE = in.end
			}
		} else {
			total += currE - currS
			currS = in.start
			currE = in.end
		}
	}
	total += currE - currS
	return total
}

func mergeIntervals(intervals []interval) []interval {
	var valid []interval
	for _, in := range intervals {
		if in.end > in.start {
			valid = append(valid, in)
		}
	}
	if len(valid) == 0 {
		return nil
	}
	sort.Slice(valid, func(i, j int) bool {
		if valid[i].start == valid[j].start {
			return valid[i].end < valid[j].end
		}
		return valid[i].start < valid[j].start
	})
	var merged []interval
	currS := valid[0].start
	currE := valid[0].end
	for i := 1; i < len(valid); i++ {
		in := valid[i]
		if in.start <= currE {
			if in.end > currE {
				currE = in.end
			}
		} else {
			merged = append(merged, interval{start: currS, end: currE})
			currS = in.start
			currE = in.end
		}
	}
	merged = append(merged, interval{start: currS, end: currE})
	return merged
}

func intervalIntersectionLength(s1, e1 int64, intervals []interval) int64 {
	if e1 <= s1 {
		return 0
	}
	var total int64
	for _, in := range intervals {
		start := s1
		if in.start > start {
			start = in.start
		}
		end := e1
		if in.end < end {
			end = in.end
		}
		if end > start {
			total += end - start
		}
	}
	return total
}

// MeasureOverlap calculates busy, covered, wall, idle, overlap, per-class metrics, and hidden nodes.
func MeasureOverlap(spans []NodeSpan) *ConcurrencyRecord {
	var parsed []NodeSpan
	for _, s := range spans {
		if s.EndNs > s.StartNs {
			cls := s.Class
			if cls == "" {
				cls = ClassifyNode(s.Name, s.Kind, s.Ops)
			}
			s.Class = cls
			parsed = append(parsed, s)
		}
	}

	if len(parsed) == 0 {
		return &ConcurrencyRecord{
			Ratio:   1.0,
			Classes: make(map[string]ClassOverlap),
		}
	}

	// Detect branches if not provided
	hasExplicitBranch := false
	for _, s := range parsed {
		if s.Branch != "" {
			hasExplicitBranch = true
			break
		}
	}
	if !hasExplicitBranch {
		for i := range parsed {
			s1, e1 := parsed[i].StartNs, parsed[i].EndNs
			dur1 := e1 - s1
			for j := range parsed {
				if i != j && parsed[j].StartNs <= s1 && e1 <= parsed[j].EndNs && (parsed[j].EndNs-parsed[j].StartNs > dur1) {
					parsed[i].Branch = "par"
					break
				}
			}
			if parsed[i].Branch == "" {
				parsed[i].Branch = "main"
			}
		}
	} else {
		for i := range parsed {
			if parsed[i].Branch == "" {
				parsed[i].Branch = "main"
			}
		}
	}

	// 1. Busy: sum of all node durations
	var busyNs int64
	var minStart int64 = math.MaxInt64
	var maxEnd int64 = math.MinInt64
	var allIntervals []interval
	var mainIntervals []interval

	for _, s := range parsed {
		dur := s.EndNs - s.StartNs
		busyNs += dur
		if s.StartNs < minStart {
			minStart = s.StartNs
		}
		if s.EndNs > maxEnd {
			maxEnd = s.EndNs
		}
		allIntervals = append(allIntervals, interval{start: s.StartNs, end: s.EndNs})
		if s.Branch != "par" {
			mainIntervals = append(mainIntervals, interval{start: s.StartNs, end: s.EndNs})
		}
	}

	// 2. Covered: union of all node intervals
	coveredNs := unionIntervalLength(allIntervals)

	// 3. Wall: span from minimum start to maximum end
	var wallNs int64
	if maxEnd > minStart {
		wallNs = maxEnd - minStart
	}

	// 4. Overlap and Idle
	overlapNs := busyNs - coveredNs
	if overlapNs < 0 {
		overlapNs = 0
	}
	idleNs := wallNs - coveredNs
	if idleNs < 0 {
		idleNs = 0
	}
	ratio := 1.0
	if coveredNs > 0 {
		ratio = float64(busyNs) / float64(coveredNs)
	}

	// 5. Hidden nodes: nodes whose span lies entirely inside another branch's span
	var hidden []HiddenNode
	for i, n1 := range parsed {
		s1, e1 := n1.StartNs, n1.EndNs
		dur1 := e1 - s1
		for j, n2 := range parsed {
			if i == j {
				continue
			}
			s2, e2 := n2.StartNs, n2.EndNs
			dur2 := e2 - s2
			if s2 <= s1 && e1 <= e2 {
				if dur2 == dur1 && i > j {
					continue
				}
				if n1.Branch == "par" || n2.Branch != n1.Branch || dur2 > dur1 {
					hidden = append(hidden, HiddenNode{
						Name:        n1.Name,
						Class:       n1.Class,
						Inside:      n2.Name,
						InsideClass: n2.Class,
						DurationMs:  round3(float64(dur1) / 1e6),
						StartNs:     s1,
						EndNs:       e1,
					})
					break
				}
			}
		}
	}
	sort.Slice(hidden, func(i, j int) bool {
		return hidden[i].DurationMs > hidden[j].DurationMs
	})

	// 6. Per-class metrics
	mergedMain := mergeIntervals(mainIntervals)
	classNodes := make(map[string][]NodeSpan)
	for _, s := range parsed {
		classNodes[s.Class] = append(classNodes[s.Class], s)
	}

	classes := make(map[string]ClassOverlap)
	for clsName, cSpans := range classNodes {
		var cBusyNs int64
		var cIntervals []interval
		var sideNodes []NodeSpan
		for _, s := range cSpans {
			dur := s.EndNs - s.StartNs
			cBusyNs += dur
			cIntervals = append(cIntervals, interval{start: s.StartNs, end: s.EndNs})
			if s.Branch == "par" {
				sideNodes = append(sideNodes, s)
			}
		}
		cCoveredNs := unionIntervalLength(cIntervals)

		var sideBranchNs int64
		for _, sn := range sideNodes {
			inter := intervalIntersectionLength(sn.StartNs, sn.EndNs, mergedMain)
			if len(mergedMain) > 0 {
				sideBranchNs += inter
			} else {
				sideBranchNs += sn.EndNs - sn.StartNs
			}
		}

		share := 0.0
		if cBusyNs > 0 {
			share = float64(sideBranchNs) / float64(cBusyNs)
		}

		classes[clsName] = ClassOverlap{
			BusyMs:          round3(float64(cBusyNs) / 1e6),
			CoveredMs:       round3(float64(cCoveredNs) / 1e6),
			SideBranchMs:    round3(float64(sideBranchNs) / 1e6),
			SideBranchShare: round3(share),
		}
	}

	return &ConcurrencyRecord{
		WallMs:      round3(float64(wallNs) / 1e6),
		BusyMs:      round3(float64(busyNs) / 1e6),
		CoveredMs:   round3(float64(coveredNs) / 1e6),
		OverlapMs:   round3(float64(overlapNs) / 1e6),
		IdleMs:      round3(float64(idleNs) / 1e6),
		Ratio:       round2(ratio),
		Classes:     classes,
		HiddenNodes: hidden,
	}
}

func round2(x float64) float64 {
	return math.Round(x*100) / 100
}

func round3(x float64) float64 {
	return math.Round(x*1000) / 1000
}

// ReadRunRecord parses a JSON document (file or bytes) as a run record or concurrency record.
func ReadRunRecord(data []byte) (*RunRecord, *ConcurrencyRecord, error) {
	var rec RunRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		// Try parsing as a raw list of spans
		var spans []NodeSpan
		if errSpans := json.Unmarshal(data, &spans); errSpans != nil {
			// Try parsing directly as ConcurrencyRecord
			var direct ConcurrencyRecord
			if errDirect := json.Unmarshal(data, &direct); errDirect != nil {
				return nil, nil, fmt.Errorf("parsing run record: %w", err)
			}
			if direct.BusyMs > 0 && direct.CoveredMs > 0 {
				return &RunRecord{Concurrency: &direct}, &direct, nil
			}
			return nil, nil, fmt.Errorf("concurrency record has no measurements")
		}
		if len(spans) == 0 {
			return nil, nil, fmt.Errorf("no spans found in JSON")
		}
		c := MeasureOverlap(spans)
		return &RunRecord{Concurrency: c, Spans: spans}, c, nil
	}

	if rec.Concurrency != nil && rec.Concurrency.BusyMs > 0 {
		return &rec, rec.Concurrency, nil
	}
	if rec.Stats != nil {
		if raw, ok := rec.Stats["concurrency"]; ok {
			bytes, err := json.Marshal(raw)
			if err != nil {
				return nil, nil, fmt.Errorf("stats.concurrency marshal: %w", err)
			}
			var c ConcurrencyRecord
			if err := json.Unmarshal(bytes, &c); err != nil {
				return nil, nil, fmt.Errorf("stats.concurrency unmarshal: %w", err)
			}
			if c.BusyMs > 0 {
				rec.Concurrency = &c
				return &rec, &c, nil
			}
		}
	}
	if len(rec.Spans) > 0 {
		c := MeasureOverlap(rec.Spans)
		rec.Concurrency = c
		return &rec, c, nil
	}

	var direct ConcurrencyRecord
	if err := json.Unmarshal(data, &direct); err != nil {
		return nil, nil, fmt.Errorf("reading concurrency from JSON: %w", err)
	}
	if direct.BusyMs > 0 && direct.CoveredMs > 0 {
		return &RunRecord{Concurrency: &direct}, &direct, nil
	}

	return nil, nil, fmt.Errorf("no concurrency data or spans found in JSON")
}

// ReadRunRecordFile reads a run record or concurrency JSON from path.
func ReadRunRecordFile(path string) (*RunRecord, *ConcurrencyRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return ReadRunRecord(data)
}

// FormatOverlapTable formats the step concurrency summary, per-class breakdown, and hidden nodes.
func FormatOverlapTable(rec *ConcurrencyRecord) string {
	if rec == nil {
		return "No concurrency data\n"
	}

	var sb strings.Builder
	sb.WriteString("Step concurrency:\n")
	sb.WriteString(fmt.Sprintf("  wall:        %8.2f ms\n", rec.WallMs))
	sb.WriteString(fmt.Sprintf("  busy:        %8.2f ms  (sum of node spans)\n", rec.BusyMs))
	sb.WriteString(fmt.Sprintf("  covered:     %8.2f ms  (union of spans)\n", rec.CoveredMs))
	sb.WriteString(fmt.Sprintf("  overlap:     %8.2f ms  (busy - covered)\n", rec.OverlapMs))
	sb.WriteString(fmt.Sprintf("  idle:        %8.2f ms  (wall - covered)\n", rec.IdleMs))
	sb.WriteString(fmt.Sprintf("  concurrency: %8.2fx (busy / covered)\n\n", rec.Ratio))

	sb.WriteString("Per-class spans:\n")
	sb.WriteString(fmt.Sprintf("  %-24s %10s %12s %16s %12s\n", "class", "busy ms", "covered ms", "side-branch ms", "side share"))

	var classNames []string
	for name := range rec.Classes {
		classNames = append(classNames, name)
	}
	sort.Slice(classNames, func(i, j int) bool {
		// Sort by busy ms descending
		return rec.Classes[classNames[i]].BusyMs > rec.Classes[classNames[j]].BusyMs
	})

	for _, name := range classNames {
		c := rec.Classes[name]
		sb.WriteString(fmt.Sprintf("  %-24s %10.2f %12.2f %16.2f %11.1f%%\n",
			name, c.BusyMs, c.CoveredMs, c.SideBranchMs, c.SideBranchShare*100))
	}

	if len(rec.HiddenNodes) > 0 {
		sb.WriteString("\nTop hidden nodes (span lies inside another branch):\n")
		topCount := len(rec.HiddenNodes)
		if topCount > 10 {
			topCount = 10
		}
		for i := 0; i < topCount; i++ {
			h := rec.HiddenNodes[i]
			sb.WriteString(fmt.Sprintf("  %-28s (%-20s): %6.3f ms inside %s\n",
				h.Name, h.Class, h.DurationMs, h.Inside))
		}
	}

	return sb.String()
}
