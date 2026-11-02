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
	"os"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"math"
)

func drafterLayers(t *testing.T) LayerBytes {
	t.Helper()
	data, err := os.ReadFile("testdata/deepseek-v41-flash-drafter-layers.json")
	if err != nil {
		t.Fatal(err)
	}
	lb, err := ReadLayerBytesJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	return lb
}

func fixed(k int) kernelplan.Speculation {
	return kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyFixed}
}

// The measured speculative steps (hero 8's machine; one row at 2 and 5
// drafts from hero 8's m14 tree on Oct 31, the rest from pod hero-d on
// Nov 1): the calibration and its error.
func TestPredictSpeculation_CalibrationOnHero8(t *testing.T) {
	lb := drafterLayers(t)
	p := hero8Plan(t, 1)
	cases := []struct {
		name       string
		rows       int
		mix        string
		draft      int
		measuredMs float64
		tokens     float64
		distinct   float64 // measured, 0 when not
		tol        float64
	}{
		{"one row, 4 drafts", 1, MixSame, 4, 25.8, 2.67, 18.3, 0.10},
		{"one row, 2 drafts", 1, MixSame, 2, 23.9, 2.21, 0, 0.10},
		{"one row, 5 drafts", 1, MixSame, 5, 29.5, 2.46, 0, 0.10},
		{"8 rows, one prompt, 4 drafts", 8, MixSame, 4, 40.4, 2.67, 18.7, 0.10},
		{"8 rows, a prompt per row, 4 drafts", 8, MixDifferent, 4, 95.1, 2.69, 83.2, 0.10},
	}
	for _, c := range cases {
		pr, err := PredictSpeculation(lb, p, Workload{Rows: c.rows, Batches: 1}, c.mix, fixed(c.draft), RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance)
		if err != nil {
			t.Fatal(err)
		}
		within(t, c.name+" (ms)", pr.StepMs, c.measuredMs, c.tol)
		within(t, c.name+" (tokens a step)", pr.TokensPerStep, c.tokens, 0.16)
		if c.distinct > 0 {
			within(t, c.name+" (distinct experts a layer)", pr.Distinct, c.distinct, 0.10)
		}
		t.Logf("%-38s predicted %5.1f ms, measured %5.1f; %.2f tokens a step, measured %.2f; %.0f distinct experts a layer; %.0f tok/s",
			c.name, pr.StepMs, c.measuredMs, pr.TokensPerStep, c.tokens, pr.Distinct, pr.AggregateTokS)
	}
}

func TestDistinctExperts_FollowsTheMeasuredPoints(t *testing.T) {
	for _, c := range []struct {
		n    int
		want float64
	}{{1, 6}, {9, 37.3}, {33, 90.3}} {
		if got := DistinctExperts(6, 384, c.n, MixDifferent); got < c.want*0.999 || got > c.want*1.001 {
			t.Errorf("%d rows: %.1f distinct experts, want the measured %.1f", c.n, got, c.want)
		}
	}
	if got := DistinctExperts(6, 384, 33, MixSame); got != 6 {
		t.Errorf("33 rows of one prompt: %.1f, want 6", got)
	}
	if got := DistinctExperts(6, 384, 132, MixDifferent); got < 90.3 || got > 384 {
		t.Errorf("132 rows: %.1f, want between the 33-row point and the layer's experts", got)
	}
	if got := DistinctExperts(6, 384, 100000, MixDifferent); got != 384 {
		t.Errorf("many rows: %.1f, want every expert", got)
	}
	// the token curve: one row's five token rows read the measured 18.3 (within 10%); a batch of one
	// prompt in every row reads the same; eight different prompts at five tokens read the measured 83.2
	if got := DistinctExpertsOfVerify(6, 384, 1, 5, MixSame); got < 16.5 || got > 20.1 {
		t.Errorf("one row, five token rows: %.1f distinct experts, want the measured 18.3 within 10%%", got)
	}
	if a, b := DistinctExpertsOfVerify(6, 384, 8, 5, MixSame), DistinctExpertsOfVerify(6, 384, 1, 5, MixSame); a != b {
		t.Errorf("eight rows of one prompt at five tokens read %.1f, one row %.1f: want the same", a, b)
	}
	if got := DistinctExpertsOfVerify(6, 384, 8, 5, MixDifferent); got < 75 || got > 91.5 {
		t.Errorf("eight different prompts at five tokens: %.1f, want the measured 83.2 within 10%%", got)
	}
	if got := DistinctExpertsOfVerify(6, 384, 33, 5, MixDifferent); got >= 384 || got <= 90.3 {
		t.Errorf("33 different prompts at five tokens: %.1f, want between the rows' 90.3 and the layer's 384", got)
	}
}

func TestAcceptance_FitAndTokens(t *testing.T) {
	a := FitAcceptance(4, 2.67)
	if a.Alpha < 0.66 || a.Alpha > 0.70 {
		t.Errorf("alpha %.3f fitted to 2.67 tokens at 4 drafts, want about 0.68", a.Alpha)
	}
	if got := DeepSeekV41FlashAcceptance.Tokens(0); got != 1 {
		t.Errorf("no drafts: %.2f tokens, want 1", got)
	}
	if lo, hi := DeepSeekV41FlashAcceptance.Tokens(2), DeepSeekV41FlashAcceptance.Tokens(2.5); hi <= lo || hi >= DeepSeekV41FlashAcceptance.Tokens(3) {
		t.Errorf("fractional drafts interpolate: %.2f at 2, %.2f at 2.5, %.2f at 3", lo, hi, DeepSeekV41FlashAcceptance.Tokens(3))
	}
}

// The choice at each load of hero 10's matrix. The drafter pays at one
// row (few experts read, the verify's token rows add theirs while the step
// is launch-bound); at 33 rows a batch the token rows' own cost nearly
// cancels the tokens they yield, and the model calls two drafts and none
// within about 10% of each other under either prompt mix: the crossover
// hero 10 measures, not one the model can call from this calibration.
func TestChooseShape_Hero10Loads(t *testing.T) {
	lb := drafterLayers(t)
	p := hero8Plan(t, 4)
	loads := []struct {
		load       Load
		wantDraft  bool
		wantBatchN int // 0: either
	}{
		{Load{Rows: 1, Mix: MixSame, Drafter: true, MaxDraft: 5}, true, 1},
		{Load{Rows: 4, Mix: MixDifferent, Drafter: true, MaxDraft: 5}, true, 4},
		{Load{Rows: 132, Mix: MixDifferent, Drafter: false}, false, 4},
	}
	for _, c := range loads {
		options, err := ChooseShape(lb, p, c.load, RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance)
		if err != nil {
			t.Fatal(err)
		}
		best := options[0]
		t.Logf("%d rows %s: %s: %.0f tok/s, %.1f ms a token (%d options)", c.load.Rows, c.load.Mix, best.Describe(), best.Prediction.AggregateTokS, best.Prediction.MsPerToken, len(options))
		for _, o := range options[1:] {
			if o.Speculation == nil || o.Speculation.Policy == kernelplan.PolicyFixed {
				t.Logf("    %-70s %s", o.Describe(), o.Rejected)
			}
		}
		drafts := best.Speculation != nil && best.Speculation.Draft > 0
		if drafts != c.wantDraft {
			t.Errorf("%d rows %s: chose %s; drafting wanted %v", c.load.Rows, c.load.Mix, best.Describe(), c.wantDraft)
		}
		if c.wantBatchN > 0 && best.Batches != c.wantBatchN {
			t.Errorf("%d rows %s: chose %d batches, want %d", c.load.Rows, c.load.Mix, best.Batches, c.wantBatchN)
		}
		for _, o := range options {
			if o.Prediction.AggregateTokS > o.Prediction.Pipeline.FloorTokS*o.Prediction.TokensPerStep*1.001 {
				t.Errorf("%s: predicted %.0f tok/s above its floor %.0f", o.Describe(), o.Prediction.AggregateTokS, o.Prediction.Pipeline.FloorTokS*o.Prediction.TokensPerStep)
			}
		}
	}
	// four batches of 33: the model's call between two drafts and none is within its error
	for _, mix := range []string{MixSame, MixDifferent} {
		options, err := ChooseShape(lb, p, Load{Rows: 132, Mix: mix, Batches: 4, Drafter: true, MaxDraft: 5}, RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance)
		if err != nil {
			t.Fatal(err)
		}
		var plain, two float64
		for _, o := range options {
			switch {
			case o.Speculation == nil:
				plain = o.Prediction.AggregateTokS
			case o.Speculation.Draft == 2 && o.Speculation.Policy == kernelplan.PolicyFixed:
				two = o.Prediction.AggregateTokS
			}
		}
		t.Logf("four batches of 33 %s prompts: %.0f tok/s plain, %.0f with two drafts; the planner picks %s", mix, plain, two, options[0].Describe())
		if two < plain*0.85 || two > plain*1.25 {
			t.Errorf("four batches of 33 %s prompts: two drafts %.0f tok/s against %.0f plain; the calibration should leave these close, or the comment above is stale", mix, two, plain)
		}
	}
	// a latency bound rejects the shapes over it and says so
	options, err := ChooseShape(lb, p, Load{Rows: 132, Mix: MixDifferent, LatencyMs: 20, Drafter: true, MaxDraft: 5}, RTXPro6000, RTXPro6000Spec, DeepSeekV41FlashAcceptance)
	if err == nil {
		t.Logf("132 rows within 20 ms a token: %s at %.1f ms a token", options[0].Describe(), options[0].Prediction.MsPerToken)
		for _, o := range options {
			if o.Rejected == "" && o.Prediction.MsPerToken > 20 {
				t.Errorf("an option over the bound was not rejected: %s", o.Describe())
			}
		}
	} else {
		t.Logf("132 rows within 20 ms a token: %v", err)
	}
}

func TestAcceptance_MeasuredPositions(t *testing.T) {
	// hero 12: the record's per-position shares replace the geometric Alpha
	acc := Acceptance{Alpha: 0.68, Positions: []float64{0.75, 0.5, 0.25, 0.25}}
	if got := acc.Tokens(4); math.Abs(got-2.75) > 1e-9 {
		t.Fatalf("tokens at 4 drafts from the positions: %v, want 2.75", got)
	}
	if got := acc.Tokens(2.5); math.Abs(got-(1+0.75+0.5+0.125)) > 1e-9 {
		t.Fatalf("interpolated tokens at 2.5 drafts: %v", got)
	}
	if got := acc.Tokens(6); math.Abs(got-2.75) > 1e-9 {
		t.Fatalf("drafts past the measured positions add nothing: %v", got)
	}
	record := []byte(`{"stats": {"speculation": {"alpha": 0.4375, "positions": [{"position": 1, "rate": 0.75}, {"position": 2, "rate": 0.5}],
		"thresholds": {"0.5": {"precision": 1.0, "loss": 0.0}}}}}`)
	measured, thresholds, _, err := MeasuredSpeculation(record)
	if err != nil {
		t.Fatal(err)
	}
	if len(measured.Positions) != 2 || measured.Positions[1] != 0.5 || thresholds["0.5"].Precision != 1.0 {
		t.Fatalf("measured: %+v %+v", measured, thresholds)
	}
	sc := RTXPro6000Spec
	sc.Thresholds = thresholds
	if st, ok := sc.stat(0.5); !ok || st.Loss != 0 {
		t.Fatalf("the measured threshold should be used: %+v %v", st, ok)
	}
	if _, ok := sc.stat(0.3); ok {
		t.Fatal("an unmeasured threshold falls back to the estimates")
	}
	if _, _, _, err := MeasuredSpeculation([]byte(`{"stats": {}}`)); err == nil {
		t.Fatal("a record without speculation statistics is an error")
	}
}
