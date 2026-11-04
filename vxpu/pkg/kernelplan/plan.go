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
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

// Options tune the policy.
type Options struct {
	// LaunchUs is one kernel launch inside a CUDA graph, credited to a
	// body the compiler fuses into its neighbours (its first operation
	// fuses into what produces its input). 0: no credit. The engine's own
	// costing measures it per card (costs.py: about 2 us); a record's
	// stats may carry it as "launch_us".
	LaunchUs float64
}

// group is every record site that shares an op and a shape key: the unit
// the measurements are pooled over.
type group struct {
	op, shapes string
	sites      []Site
	times      map[string][]float64 // body -> measured microseconds, over the records
	fused      map[string]bool
}

// Plan reads records of one GPU and writes the table for it. Records of
// different GPUs are refused: a table is per card.
func Plan(ctx context.Context, records []*Record, opts Options) (*Table, error) {
	log := klog.FromContext(ctx)
	if len(records) == 0 {
		return nil, fmt.Errorf("no records to plan from")
	}
	gpu := records[0].Hardware.GPU
	for _, r := range records[1:] {
		if r.Hardware.GPU != gpu {
			return nil, fmt.Errorf("records of different GPUs (%q and %q): a table is per card", gpu, r.Hardware.GPU)
		}
	}
	t := &Table{Format: TableFormat, GPU: gpu, Knobs: map[string]any{}}
	launch := opts.LaunchUs
	if launch == 0 {
		launch = statNumber(records, "launch_us")
	}

	groups := map[string]*group{}
	var order []string
	for _, r := range records {
		for _, s := range r.Sites {
			if s.Source == "kept" || s.Chosen == "" {
				continue // not lowered: nothing to choose
			}
			k := s.Op + "\x00" + s.Shapes
			g := groups[k]
			if g == nil {
				g = &group{op: s.Op, shapes: s.Shapes, times: map[string][]float64{}, fused: map[string]bool{}}
				groups[k] = g
				order = append(order, k)
			}
			g.sites = append(g.sites, s)
			for body, us := range s.MeasuredUs {
				g.times[body] = append(g.times[body], us)
			}
			for _, f := range s.Fused {
				g.fused[f] = true
			}
		}
	}
	sort.Strings(order)
	for _, k := range order {
		g := groups[k]
		body, why := g.choose(launch)
		if perSite := g.disagreeing(body); len(perSite) > 0 {
			// the measurement is per shape, but the engine chose differently at
			// some sites (a table given before, say): say so per site rather
			// than hide it under one entry
			for _, s := range g.sites {
				t.Sites = append(t.Sites, Entry{Op: g.op, Site: s.Site, Impl: body, Why: why})
			}
			t.Explain = append(t.Explain, fmt.Sprintf("%s %s: %s; the record ran %s at %s",
				shortOp(g.op), g.shapes, why, strings.Join(perSite, ", "), "some sites, so the table names every site"))
			continue
		}
		t.Sites = append(t.Sites, Entry{Op: g.op, Shapes: g.shapes, Impl: body, Why: why})
		t.Explain = append(t.Explain, fmt.Sprintf("%s %s (%s): %s", shortOp(g.op), g.shapes, plural(len(g.sites), "site"), why))
	}

	// the knobs: native formats from the card, the rest from sweeps, else as the record ran
	if major := capabilityMajor(records[0].Hardware.GPUCapability); major > 0 {
		native := major >= 10 // Blackwell and later take MXFP4 on the tensor cores
		t.Knobs["FP4_NATIVE"] = native
		t.Explain = append(t.Explain, fmt.Sprintf("FP4_NATIVE=%v: compute capability %s", native, records[0].Hardware.GPUCapability))
	}
	sweeps := map[string]map[string]float64{}
	for _, r := range records {
		if raw, ok := r.Stats["sweeps"]; ok {
			var s map[string]map[string]float64
			if err := json.Unmarshal(raw, &s); err != nil {
				return nil, fmt.Errorf("record stats.sweeps: %w", err)
			}
			for knob, values := range s {
				if sweeps[knob] == nil {
					sweeps[knob] = map[string]float64{}
				}
				for v, us := range values {
					sweeps[knob][v] = us
				}
			}
		}
	}
	for knob, values := range sweeps {
		if _, known := knobTypes[knob]; !known {
			log.Info("ignoring a sweep of a knob the engine does not know", "knob", knob)
			continue
		}
		best, bestUs := "", math.Inf(1)
		for v, us := range values {
			if us < bestUs || (us == bestUs && v < best) {
				best, bestUs = v, us
			}
		}
		value, err := knobValue(knob, best)
		if err != nil {
			return nil, fmt.Errorf("record stats.sweeps: %w", err)
		}
		t.Knobs[knob] = value
		t.Explain = append(t.Explain, fmt.Sprintf("%s=%s: the fastest of %d swept values (%.1f us)", knob, best, len(values), bestUs))
	}
	for knob, value := range records[len(records)-1].Knobs {
		if _, set := t.Knobs[knob]; set || value == nil {
			continue
		}
		if _, known := knobTypes[knob]; !known {
			continue
		}
		t.Knobs[knob] = value // as the last record ran: nothing measured says otherwise
	}
	// The choices by rows under expert parallelism (hero run 15, measured on four RTX PRO 6000, TP4 + EP4,
	// rows of different prompts; design.md "the tensor-parallel step folded"): up to five rows the direct
	// expert kernels (each pair reads its expert whole) beat the GEMM form (5 rows 15.1 -> 13.9 ms, 3 rows
	// 12.6 -> 11.9, 2 rows 12.0 -> 10.7) and their K split of 2 takes 10.0 -> 9.5 at one row (4 and 3: 9.6);
	// the MXFP8 recode is 9.3 -> 9.1 at one row, 0 at three and +3 ms at 33. Above five rows the GEMM form
	// reads each distinct expert once (8 rows 17.4 against 17.2 direct, but 13.1 against 18.4 when rows
	// share experts) and the splits do not apply.
	if major := capabilityMajor(records[0].Hardware.GPUCapability); major >= 10 {
		t.KnobsByRows = []RowsKnobs{
			{RowsMax: 1, Knobs: map[string]any{"FP4_DIRECT_MAX": 48, "FP4_GU_KS": 2, "FP4_DN_KS": 2, "MXFP8_RECODE": true},
				Why: "one row: the direct kernels with a K split of 2 (10.0 -> 9.5 ms) and the recode (9.3 -> 9.1)"},
			{RowsMax: 5, Knobs: map[string]any{"FP4_DIRECT_MAX": 48, "FP4_GU_KS": 2, "FP4_DN_KS": 2, "MXFP8_RECODE": false,
				"FP4_DIRECT_GROUP": true},
				Why: "two to five rows: the direct kernels (5 rows 15.1 -> 13.9 ms) grouped by expert (5 rows of one " +
					"prompt 13.1 -> 11.7, of different prompts 13.1 -> 13.0; 2 rows 9.7 -> 9.2 and 9.9 -> 9.9); the recode is 0 here"},
			{RowsMax: 8, Knobs: map[string]any{"FP4_DIRECT_MAX": 48, "FP4_GU_KS": 2, "FP4_DN_KS": 2, "MXFP8_RECODE": false,
				"FP4_DIRECT_GROUP": true},
				Why: "six to eight rows: the grouped direct kernels (8 rows of different prompts 16.0 against the GEMM " +
					"form's 16.8 and the direct kernels' 16.4; one prompt's rows 14.3 against the GEMM form's 12.6, a " +
					"per-workload choice); at 16 rows the GEMM form wins every arm (21.7 against 23.3 grouped, 24.3 direct)"},
		}
		// The grouped direct form (FP4_DIRECT_GROUP, hero run 15): the launch's pairs sorted by held expert in
		// one program and cut into blocks of two, a block of two reading its expert once; a block of one runs
		// the direct kernel's own body, so rows that share nothing pay nothing (A59, pod A: different prompts
		// 9.9 against 9.9 at 2 rows and 13.0 against 13.1 at 5, one prompt's rows 9.2 against 9.7 and 11.7
		// against 13.1, tokens identical to the direct kernels'). The kernel keeps a single row on the direct
		// path whatever the knob; the rule here is the planner's: on from two rows.
		t.Explain = append(t.Explain, "knobs_by_rows: the direct expert kernels up to 48 pairs with a K split of 2 through "+
			"eight rows, grouped by expert from two rows (5 rows of one prompt 13.1 -> 11.7, of different prompts 13.1 -> 13.0; "+
			"8 rows of different prompts 16.0 against the GEMM form's 16.8), the MXFP8 recode at one row (hero run 15's "+
			"ladders); the GEMM form above")
		// The sums' form by rows (hero run 15's many-row account): the reduce-scatter form moves half the one-shot
		// form's bytes over PCIe and wins once the pushes are bandwidth-bound, -0.4 ms a step at 16 rows, -1.9 at
		// 40 and -3.8 at 80, while its extra round trip a site loses 0.2 to 0.5 ms at 5 and 8 rows (pods A and B,
		// different prompts, the tokens the same either way); the crossover lies between 8 and 16 rows and 16 is
		// the measured winning point (a measuring step's record overrides this).
		// The sums' wire dtype is a numerics class apart from the form (hero run 15): ALLREDUCE_PUSH_BF16 rounds
		// every rank's fp32 partial to bf16 before the fp32 sum and is a choice per workload, not a default here:
		// the speculation workstream's 132-prompt prose line drifted acceptance 1.6 percent (1.436 against 1.459
		// a step) for 2 percent of step at one row and 2 to 4 at 40 rows, code gained 9 percent at eight streams.
		// The pull form (ALLREDUCE_PULL, bf16 wire by construction) measured 17.9 / 21.0 us a site at 1 / 5 rows
		// against the fused push's 15.6 / 18.9, so the form by rows stays: the push through 15, scatter from 16.
		t.Explain = append(t.Explain, "ALLREDUCE_PUSH_BF16 off: the bf16 wire is a numerics class per workload "+
			"(prose entries off, code at width on); ALLREDUCE_PULL off: 17.9 / 21.0 us a site at 1 / 5 rows against "+
			"the fused push's 15.6 / 18.9")
		// The expert form by sharing, not rows alone (hero run 15, Nov 4): the GEMM form reads each distinct
		// expert once and pays a fixed cost (one row 10.1 against the direct kernels' 8.7 ms: a schedule
		// launch, a SwiGLU-quant launch and 160 gathers and scatters a step more); the grouped direct
		// kernels read a shared expert once per block of two and cost nothing where rows share nothing. So
		// the crossover is the pairs a distinct expert: 8 rows of one prompt (48 pairs over 6 experts a
		// layer, 8 a distinct expert) read GEMM 12.8 against grouped 14.5; 5 rows of one prompt (5 a
		// distinct expert) grouped 11.7 against GEMM 12.0; 8 rows of different prompts (about 1.2) grouped
		// 16.0 against GEMM 16.8; 16 rows of different prompts GEMM 21.7 against grouped 23.3. When the
		// record carries the distinct experts a layer (stats.distinct_experts, the measuring harness's mean)
		// the table's entry for the record's rows follows that count: the GEMM form from 6 pairs a distinct
		// expert, the grouped kernels below it; one row stays direct and 16 rows and above stay GEMM.
		if rows, distinct := recordRows(records), statFloat(records, "distinct_experts"); rows >= 2 && rows < 16 && distinct > 0 {
			pairs := float64(rows*6) / distinct
			gemm := pairs >= 6
			for i := range t.KnobsByRows {
				e := &t.KnobsByRows[i]
				if e.RowsMax < rows || (i > 0 && t.KnobsByRows[i-1].RowsMax >= rows) {
					continue // not the entry that covers the record's rows
				}
				if gemm {
					for _, k := range []string{"FP4_DIRECT_MAX", "FP4_GU_KS", "FP4_DN_KS", "FP4_DIRECT_GROUP"} {
						delete(e.Knobs, k)
					}
					e.Knobs["FP4_DIRECT_MAX"] = 0
					e.Why = fmt.Sprintf("%d rows at %.1f pairs a distinct expert (measured %.1f distinct a layer): the GEMM form "+
						"(8 rows of one prompt 12.8 against the grouped kernels' 14.5 ms)", rows, pairs, distinct)
				} else {
					e.Why = fmt.Sprintf("%d rows at %.1f pairs a distinct expert (measured %.1f distinct a layer): the grouped direct "+
						"kernels (8 rows of different prompts 16.0 against the GEMM form's 16.8 ms)", rows, pairs, distinct)
				}
				t.Explain = append(t.Explain, "expert form by sharing: "+e.Why)
			}
		}
		// The rule set as it stands at the end of hero run 15's tensor-parallel work (Nov 4), in one place, so the
		// table explains the whole step shape. What the table cannot carry as a knob is a plan option or an
		// engine constant, named here with where it lives.
		t.Explain = append(t.Explain,
			"expert placement: the tensor-parallel experts (every card a quarter-slice of every routed expert, plan "+
				"option expert_tp = 4) are a selection the planner makes where a workload's measurement chose it, not a "+
				"default (the user's decision: the placement gives up strict accuracy against the single-card "+
				"reference, the first of many such approximations the planner will select; the league accepted the "+
				"class on the 132-prompt line, -0.7 percent accepted, +12.4 percent decode): one row 8.2 -> 7.6 ms, 5 "+
				"rows 12.5 -> 11.4, 8 rows 15.3 -> 13.9, 33 rows 31.5 -> 28.9 on different prompts; 8 rows of one "+
				"prompt 11.4 (grouped) and 10.9 (GEMM) against the reference's 11.86 on its own shape",
			"expert form: direct kernels at one row, grouped by expert from two rows, the GEMM form from 16 rows or from "+
				"6 pairs a distinct expert (knobs_by_rows and the record's distinct_experts)",
			"fused push (engine.FUSED_PUSH_MAX_ROWS): on through 16 token rows (batch x step_tokens), off under the "+
				"bf16 wire class; -0.2 to -0.3 ms a step at 5 and 8 rows and -0.1 to -0.2 at 16 under TP4 experts",
			"self-fed step (stepgraph.GRAPH_TOKEN, on): the greedy choice written into the graph's own inputs, one step "+
				"queued ahead of the host; -0.4 to -0.5 ms a step at 1, 5 and 8 rows, tokens identical",
			"ALLREDUCE_PUSH_BF16 stays a class per workload; ALLREDUCE_PULL closed (pathological fp32 reads on the "+
				"step, a tie with the bf16 class against the fused-off step and 0.3 to 0.5 behind the fused-on one)")
		// The dense GEMVs' tiles per shape class from the rotated-weight sweep (Nov 4; gemv-sweep.py: every tile
		// the kernel accepts, in a captured graph over distinct weight copies past the L2, one row, one RTX PRO
		// 6000). Two shapes moved; the others' defaults were the best of 142 to 360 tiles, and the kernel's
		// structure, not its tiles, holds the dense GEMVs at 52 to 62 percent of the floor.
		for _, e := range []BlockEntry{
			{Shapes: "gemv_rows rows=1 n=5120 k=1024 w=fp8 native=1 split=1",
				Blocks: map[string]int{"BN": 128, "BK": 64, "KSPLIT": 4, "warps": 4, "stages": 3},
				Us:     6.8, Bytes: 5406720, FloorUs: 3.4, Efficiency: 0.50,
				Why: "wo_b: 294 tiles, the default (BN 16, BK 256, KSPLIT 2) 8.1 us"},
			{Shapes: "gemv_rows rows=1 n=1024 k=1280 w=fp8 native=1 split=1",
				Blocks: map[string]int{"BN": 32, "BK": 64, "KSPLIT": 4, "warps": 4, "stages": 4},
				Us:     3.3, Bytes: 1351680, FloorUs: 0.8, Efficiency: 0.25,
				Why: "the indexer's wq_b: 142 tiles, the default (BN 16, BK 256, KSPLIT 1) 4.1 us"},
		} {
			have := false
			for _, b := range t.Blocks {
				have = have || b.Shapes == e.Shapes
			}
			if !have {
				t.Blocks = append(t.Blocks, e)
			}
		}
		t.Explain = append(t.Explain, "GEMV tiles from the rotated-weight sweep: wo_b 8.1 -> 6.8 us (BN 128, BK 64, "+
			"KSPLIT 4), the indexer's wq_b 4.1 -> 3.3 (BN 32, BK 64, KSPLIT 4); the other dense shapes keep their defaults "+
			"(the best of every tile tried); 0.1 ms a step")
		if _, set := t.Knobs["ALLREDUCE_SCATTER_MIN_ROWS"]; !set {
			t.Knobs["ALLREDUCE_SCATTER_MIN_ROWS"] = 16
			t.Explain = append(t.Explain, "ALLREDUCE_SCATTER_MIN_ROWS=16: the reduce-scatter sums from 16 rows "+
				"(-0.4 ms a step at 16 rows, -1.9 at 40, -3.8 at 80; +0.2 at 8, +0.5 at 5)")
		}
	}
	log.Info("planned the kernel table", "gpu", gpu, "sites", len(t.Sites), "knobs", len(t.Knobs), "by_rows", len(t.KnobsByRows))
	return t, nil
}

// recordRows is the token rows the records' step ran (stats.rows, the measuring harness's), 0 when unknown.
func recordRows(records []*Record) int {
	return int(statFloat(records, "rows"))
}

// statFloat is a numeric stat from the first record that has it (a bare number, or {"mean": n}), 0 when none.
func statFloat(records []*Record, name string) float64 {
	for _, r := range records {
		raw, ok := r.Stats[name]
		if !ok {
			continue
		}
		var v float64
		if json.Unmarshal(raw, &v) == nil {
			return v
		}
		var m map[string]float64
		if json.Unmarshal(raw, &m) == nil {
			if x, ok := m["mean"]; ok {
				return x
			}
		}
	}
	return 0
}

// choose is the policy for one group: the measured-fastest body, a launch
// credited to a fused one; without measurements, what the record ran (the
// engine's registry order) when the sites agree.
func (g *group) choose(launch float64) (string, string) {
	if len(g.times) > 0 {
		best, bestCost := "", math.Inf(1)
		parts := make([]string, 0, len(g.times))
		for body, samples := range g.times {
			cost := mean(samples)
			if g.fused[body] {
				cost -= launch
			}
			parts = append(parts, fmt.Sprintf("%s %.1f us", body, mean(samples)))
			if cost < bestCost || (cost == bestCost && body < best) {
				best, bestCost = body, cost
			}
		}
		sort.Strings(parts)
		why := "measured: " + strings.Join(parts, ", ")
		if launch > 0 {
			why += fmt.Sprintf(" (a fused body credited %.1f us for the launch it saves)", launch)
		}
		return best, why
	}
	counts := map[string]int{}
	for _, s := range g.sites {
		counts[s.Chosen]++
	}
	best := ""
	for body, n := range counts {
		if best == "" || n > counts[best] || (n == counts[best] && body < best) {
			best = body
		}
	}
	return best, "not measured: as the engine ran it (its registry's order)"
}

// disagreeing lists the sites whose recorded choice is not body.
func (g *group) disagreeing(body string) []string {
	var out []string
	for _, s := range g.sites {
		if s.Chosen != body {
			out = append(out, s.Site+" ran "+s.Chosen)
		}
	}
	return out
}

// Mismatch is a record site that did not run the table's body.
type Mismatch struct {
	Site Site
	Want string
}

func (m Mismatch) String() string {
	where := m.Site.Site
	if where == "" {
		where = m.Site.Shapes
	}
	return fmt.Sprintf("%s at %s ran %s, the table names %s", shortOp(m.Site.Op), where, m.Site.Chosen, m.Want)
}

// Check compares what a record ran against a table: every site the table
// names must have run the named body. A planner whose choices are not
// applied is no planner, and the failure is silent otherwise (#490).
func Check(t *Table, r *Record) []Mismatch {
	var out []Mismatch
	for _, s := range r.Sites {
		e := t.Entry(s.Op, s.Site, s.Shapes)
		if e != nil && e.Impl != s.Chosen {
			out = append(out, Mismatch{Site: s, Want: e.Impl})
		}
	}
	return out
}

func statNumber(records []*Record, name string) float64 {
	for _, r := range records {
		if raw, ok := r.Stats[name]; ok {
			var f float64
			if json.Unmarshal(raw, &f) == nil {
				return f
			}
		}
	}
	return 0
}

// knobValue parses a swept value's text as the knob's type.
func knobValue(knob, text string) (any, error) {
	switch knobTypes[knob] {
	case "bool":
		switch text {
		case "true", "1":
			return true, nil
		case "false", "0":
			return false, nil
		}
		return nil, fmt.Errorf("knob %s: %q is not true or false", knob, text)
	case "int":
		n, err := strconv.Atoi(text)
		if err != nil {
			return nil, fmt.Errorf("knob %s: %w", knob, err)
		}
		return n, nil
	}
	return nil, fmt.Errorf("unknown knob %q", knob)
}

func mean(xs []float64) float64 {
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

func plural(n int, noun string) string {
	if n == 1 {
		return "1 " + noun
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

func shortOp(op string) string {
	_, name, found := strings.Cut(op, "::")
	if !found {
		return op
	}
	return name
}
