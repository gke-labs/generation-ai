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
	"strings"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func oneCardPlan() *kernelplan.PipelinePlan {
	return &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, Ranges: []kernelplan.Range{{GPU: 0, Layers: "0:43"}}, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: 1, Schedule: kernelplan.ScheduleRangeOrder}}
}

// The parallel model with one card is the single-batch step the hero runs
// measured (the layer-sharded step walks the cards one at a time, which
// streams one card's bandwidth as one card does).
func TestPredictParallel_OneCardMatchesTheMeasuredSteps(t *testing.T) {
	lb := drafterLayers(t)
	one := oneCardPlan()
	cases := []struct {
		name     string
		rows     int
		mix      string
		draft    int
		measured float64
	}{
		{"one row (hero 11/12)", 1, MixSame, 0, 13.1},
		{"one row, four drafts (hero 13)", 1, MixSame, 4, 25.3},
		{"8 rows of one prompt (hero 12, nine rows)", 8, MixSame, 0, 17.0},
		{"33 rows of one prompt (hero 12)", 33, MixSame, 0, 29.0},
		{"132 rows of one prompt (hero 12, the GEMM path)", 132, MixSame, 0, 79.3},
		{"132 rows of different prompts (hero 12)", 132, MixDifferent, 0, 171.8},
	}
	for _, c := range cases {
		pr, err := PredictParallel(lb, one, Workload{Rows: c.rows, Batches: 1, Drafter: c.draft > 0}, c.mix, fixed(c.draft),
			RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel)
		if err != nil {
			t.Fatal(err)
		}
		within(t, c.name, pr.StepMs, c.measured, 0.12)
	}
}

func TestCrossingUs_FollowsTheMeasuredCurve(t *testing.T) {
	curve := RTXPro6000Parallel.OneShotUs
	for _, c := range []struct{ rows, us float64 }{{1, 10}, {8, 16}, {33, 52}, {132, 194}} {
		if got := CrossingUs(curve, c.rows); math.Abs(got-c.us) > 1e-9 {
			t.Errorf("%v token rows: %.1f µs, want the measured %.1f", c.rows, got, c.us)
		}
	}
	if got := CrossingUs(curve, 3); got <= 10 || got >= 16 {
		t.Errorf("3 token rows: %.1f µs, want between the 1-row and 8-row points", got)
	}
	if got := CrossingUs(curve, 264); math.Abs(got-388) > 1e-9 {
		t.Errorf("264 token rows: %.1f µs, want the bandwidth-bound 388", got)
	}
	// 86 one-shot reductions a step: 0.9 ms at one row, 16.7 at 132 rows (dev/hero/crossings.py)
	if ms := 86 * CrossingUs(curve, 1) / 1e3; math.Abs(ms-0.86) > 0.01 {
		t.Errorf("86 reductions at one row: %.2f ms", ms)
	}
	if ms := 86 * CrossingUs(curve, 132) / 1e3; math.Abs(ms-16.7) > 0.1 {
		t.Errorf("86 reductions at 132 rows: %.1f ms, want 16.7", ms)
	}
}

func TestMaxExpertShare(t *testing.T) {
	if got := MaxExpertShare(6, 1); got != 6 {
		t.Errorf("one card serves every expert: %.1f", got)
	}
	if got := MaxExpertShare(6, 4); got < 2.3 || got > 2.6 {
		t.Errorf("6 experts over 4 cards: the fullest serves %.2f, want about 2.45", got)
	}
	if got := MaxExpertShare(90, 4); got < 25 || got > 27 {
		t.Errorf("90 experts over 4 cards: %.1f, want about 26", got)
	}
	if got := MaxExpertShare(1, 4); got != 1 {
		t.Errorf("one expert: %.1f, want 1", got)
	}
}

// What the run is judged against: the TP4 + EP4 step at one row and at
// 132 rows, with the estimated terms named in the basis.
func TestPredictParallel_FourCardsPredictions(t *testing.T) {
	lb := drafterLayers(t)
	tp := ParallelPlan(hero8Plan(t, 1), "")
	if tp.Parallel == nil || tp.Parallel.DenseTP != 4 || tp.Parallel.ExpertEP != 4 || len(tp.Ranges) != 1 || tp.Ranges[0].Layers != "0:43" {
		t.Fatalf("parallel plan: %+v %+v", tp.Ranges, tp.Parallel)
	}
	if err := tp.Validate(); err != nil {
		t.Fatal(err)
	}
	one, err := PredictParallel(lb, tp, Workload{Rows: 1, Batches: 1}, MixSame, fixed(0), RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel)
	if err != nil {
		t.Fatal(err)
	}
	if one.StepMs < 5 || one.StepMs > 8 {
		t.Errorf("one row over four cards: %.1f ms, the model's terms put it between 5 and 8", one.StepMs)
	}
	joined := strings.Join(one.Pipeline.Basis, "\n")
	for _, want := range []string{"estimated until measured", "one-shot", "2.5 of them"} {
		if !strings.Contains(joined, want) {
			t.Errorf("basis lacks %q:\n%s", want, joined)
		}
	}
	wide, err := PredictParallel(lb, tp, Workload{Rows: 132, Batches: 1}, MixDifferent, fixed(0), RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel)
	if err != nil {
		t.Fatal(err)
	}
	if wide.StepMs < 45 || wide.StepMs > 75 || !strings.Contains(strings.Join(wide.Pipeline.Basis, "\n"), "ring") {
		t.Errorf("132 different rows over four cards: %.1f ms with %s", wide.StepMs, strings.Join(wide.Pipeline.Basis, " | "))
	}
	if _, err := PredictParallel(lb, tp, Workload{Rows: 2000, Batches: 1}, MixDifferent, fixed(0), RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel); err == nil {
		t.Error("2,000 rows is over the kernels' limit")
	}
}

func TestParallelMemory(t *testing.T) {
	lb := drafterLayers(t)
	tp := ParallelPlan(hero8Plan(t, 1), "")
	m := ParallelMemory(lb, tp, Workload{Rows: 1, Batches: 1}, RTXPro6000Memory)
	if !m.Fits || len(m.GPUs) != 4 || m.GPUs[1].ExpertsGB < 70 || m.GPUs[1].ExpertsGB > 76 {
		t.Fatalf("one row over four cards: %s; card 1 holds %.1f GB of experts", m.Why, m.GPUs[1].ExpertsGB)
	}
	if m.GPUs[0].TotalGB <= m.GPUs[1].TotalGB {
		t.Errorf("the first card holds the embedding and the prefill's inputs: %.1f against %.1f", m.GPUs[0].TotalGB, m.GPUs[1].TotalGB)
	}
	if w := ParallelMemory(lb, tp, Workload{Rows: 132, Batches: 1}, RTXPro6000Memory); !w.Fits {
		t.Errorf("132 plain rows should fit: %s", w.Why)
	}
}

func TestChooseServing_PicksTheCardsTogetherForOneRow(t *testing.T) {
	lb := drafterLayers(t)
	p := hero8Plan(t, 1)
	options, err := ChooseServing(lb, p, Load{Rows: 1, Mix: MixSame, Batches: 1}, RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel, RTXPro6000Memory)
	if err != nil {
		t.Fatal(err)
	}
	best := options[0]
	if best.Parallel == nil || best.Parallel.DenseTP != 4 {
		t.Fatalf("one row: chose %s", best.Describe())
	}
	var sharded *ServingOption
	for i := range options[1:] {
		if options[i+1].Parallel == nil {
			sharded = &options[i+1]
			break
		}
	}
	if sharded == nil || sharded.Rejected == "" || !strings.Contains(sharded.Rejected, "under the choice") {
		t.Fatalf("the layer-sharded option is listed with why: %+v", sharded)
	}
	if best.Prediction.AggregateTokS < 1.5*sharded.Prediction.AggregateTokS {
		t.Errorf("four cards at one row: %.0f against %.0f tok/s on one", best.Prediction.AggregateTokS, sharded.Prediction.AggregateTokS)
	}
	// with the drafter the chooser still lists both shapes, and the bound applies to both
	options, err = ChooseServing(lb, p, Load{Rows: 1, Mix: MixSame, Batches: 1, Drafter: true, MaxDraft: 4, LatencyMs: 1}, RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance, RTXPro6000Parallel, RTXPro6000Memory)
	if err == nil || len(options) == 0 {
		t.Fatal("a 1 ms bound serves nothing")
	}
}
