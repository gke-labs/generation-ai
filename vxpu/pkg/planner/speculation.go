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
	"sort"
	"strconv"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// The speculative step's cost model (hero run 10). A step that drafts k
// tokens a row verifies k+1 token rows per row: the dense weights are read
// once as before, the experts once per distinct expert the token rows
// select, and the drafter's own graphs run before the verify. What the
// step yields is the expected accepted length, from the per-position
// acceptance rate; the trade is bytes for distinct experts against tokens
// per step, and it is a different trade at one row (few experts, many to
// add) than at 33 (most of the layer's experts already read).
//
// Measured inputs (hero 8's machine, four RTX PRO 6000, Oct 31 2026):
// one row at 5 tokens a step 25.8 ms and 2.67 tokens a step (40 of 96
// drafts accepted); at 3 tokens 23.9 ms, 2.21 tokens; at 6 tokens 29.1 to
// 30.0 ms, 2.46 tokens (the model's own five-draft block); eight rows at 5
// tokens 42.9 ms with one prompt in every row and 97.1 ms with a prompt per
// row (2.37 and 2.71 tokens a step). The distinct experts a layer a batch
// reads: 6 at one row or one prompt, 37.3 at 9 different prompts, 90.3 at
// 33 (VXPU_EXPERT_STATS=1, hero 9's runs). With five tokens a step (Nov 1,
// pod hero-d, the specm5 artifacts): one row reads 18.3 distinct experts a
// layer in 25.8 ms, eight rows of one prompt 18.7 in 40.4 ms (2.67 tokens
// a step), eight different prompts 83.2 in 95.1 ms (2.69): consecutive
// tokens of a row overlap more than rows of different prompts do and far
// more than independent draws would (29 at one row).

// SpecCost is what a speculative step costs beyond the pipeline model.
type SpecCost struct {
	// DrafterStepMs is the drafter's graphs and the host gaps between the
	// draft and the verify, a step, on the last range (its stages' bytes are
	// in the pipeline model through LayerBytes.DrafterDenseBytes), at one
	// row; DrafterRowMs is what each further row adds (the graph alone
	// replays in 2.5 ms at one row and 3.8 at eight, hero run 13, pod C).
	DrafterStepMs float64
	DrafterRowMs  float64
	// PerDraftRowMs is a step's cost per verify token row beyond the batch's
	// own rows in the dense families: the GEMVs, fronts and attention at
	// more token rows (hero run 13, pod C: 9.5 ms over 32 extra token rows
	// at eight rows, 0.30 ms each). It is less than PipelineCost.PerRowMs,
	// which also carries a row's cache traffic.
	PerDraftRowMs float64
	// ReadPeakGBs is the card's single-tensor read bandwidth, what the
	// grouped expert kernels reach when bytes bound them (1,535 GB/s on the
	// RTX PRO 6000, hero run 11; the pipeline model's copy figure is 1,460).
	ReadPeakGBs float64
	// ExpertPairUs is the grouped expert kernels' time per (token row,
	// expert) pair per layer beyond the bytes: the kernels do a GEMV per
	// pair, so a verify of many token rows pays per pair on top of the
	// distinct experts' bytes. Measured on pod C (hero run 13, the expert
	// family of the step's kernel profile against the distinct experts'
	// bytes at the read peak): one row with four drafts, 30 pairs a layer,
	// 12.6 ms against 8.9 of bytes; eight rows with two drafts, 144 pairs,
	// 50.0 against 33.3; with four drafts, 240 pairs, 63.8 against 41.5:
	// 2.1 to 2.6 µs a pair. Eight rows plain, 48 pairs, run at the bytes
	// (16.7 ms for 25.8 GB), so the term is for the verify's extra pairs.
	ExpertPairUs float64
	// ExpertPairUsShared is the same when the pairs repeat few experts
	// (one prompt in every row: the copies select the same experts and the
	// L2 serves the repeats): eight rows of one prompt with four drafts
	// measured 40.4 ms against 17.0 plain in hero run 10, which the host
	// gap, the dense token rows and the bytes account for without a pair
	// term (estimated from that step, not profiled).
	ExpertPairUsShared float64
	// HostGapMs is what the step's GPUs wait for the experts host node
	// when the verify of different prompts has more than HostGapRows token
	// rows: 10.8 and 10.5 ms at 24 and 40 token rows (the node runs eager
	// fp32 GEMMs and about a thousand small kernels for the routing then),
	// 1.1 ms at 5 token rows and none at 8 rows plain; with one prompt in
	// every row the measured step (40.4 ms at eight rows and four drafts)
	// has no room for it. A step, not a slope: an engine defect to remove,
	// priced here until it is.
	HostGapMs   float64
	HostGapRows int
	// ConfidencePrecision is the share of the drafts the fixed policy would
	// have wasted (rejected) that the confidence policy never issues, and
	// ConfidenceLoss the share of the drafts it would have had accepted that
	// the policy forgoes. Both are estimates until the engine records the
	// per-position confidence histogram (speculation.py's stats).
	ConfidencePrecision float64
	ConfidenceLoss      float64
	// PreferGain is the share of a draft token's new experts the experts
	// policy avoids by preferring, among the drafter's candidates, the
	// continuation whose experts the step already reads. An estimate until
	// measured.
	PreferGain float64
	// Thresholds carries the confidence policy's measured precision and
	// loss per threshold (the record's speculation statistics, hero run
	// 12: the engine replays the confidence head's score beside every
	// draft's acceptance), keyed by the threshold as the record prints it
	// ("0.5"). Set, it replaces the two estimates above for a threshold
	// it names.
	Thresholds map[string]ConfidenceStat
	// Replayed carries, per policy the engine replayed over its recorded
	// steps (speculation.evaluate_all), the tokens and the token rows a
	// step it would have had: exact for that run, so the model uses them
	// in place of its formulas for a policy it finds here.
	Replayed map[string]Replay
}

// Replay is one policy replayed over recorded steps: the mean tokens a
// step (one plus the drafts accepted among those issued) and the mean
// token rows a step (one plus the drafts issued).
type Replay struct {
	TokensPerStep    float64 `json:"tokens_per_step"`
	TokenRowsPerStep float64 `json:"token_rows_per_step"`
}

// ReplayKey names a policy as the record's replay does.
func ReplayKey(spec kernelplan.Speculation) string {
	switch spec.Policy {
	case kernelplan.PolicyConfidence:
		return fmt.Sprintf("confidence/%d/%s", spec.Draft, strconv.FormatFloat(spec.MinConfidence, 'g', -1, 64))
	case kernelplan.PolicyExperts:
		return fmt.Sprintf("experts/%d/%d", spec.Draft, spec.MaxNewExperts)
	}
	return fmt.Sprintf("fixed/%d", spec.Draft)
}

// ConfidenceStat is what a confidence threshold did on recorded steps:
// Precision is the share of the first rejected drafts it would have
// stopped before, Loss the share of the accepted drafts it would have
// forgone.
type ConfidenceStat struct {
	Precision float64 `json:"precision"`
	Loss      float64 `json:"loss"`
}

// stat is the threshold's measured statistic, or the estimates.
func (sc SpecCost) stat(threshold float64) (ConfidenceStat, bool) {
	if s, ok := sc.Thresholds[strconv.FormatFloat(threshold, 'g', -1, 64)]; ok {
		return s, true
	}
	return ConfidenceStat{Precision: sc.ConfidencePrecision, Loss: sc.ConfidenceLoss}, false
}

// RTXPro6000Spec is the speculative step's calibration: with the distinct
// experts of the verify's token rows from the measured token curve
// (DistinctExpertsOfVerify), the drafter's step and the token rows' cost
// fit the five measured cases within 10% (speculation_test.go).
var RTXPro6000Spec = SpecCost{DrafterStepMs: 2.8, DrafterRowMs: 0.19, PerDraftRowMs: 0.28, ReadPeakGBs: 1535, ExpertPairUs: 2.2, ExpertPairUsShared: 0.3,
	HostGapMs: 10.5, HostGapRows: 8, ConfidencePrecision: 0.5, ConfidenceLoss: 0.1, PreferGain: 0.25}

// Acceptance is the drafter's per-position acceptance rate: a draft at
// position i (after i-1 accepted drafts) is accepted with probability
// Alpha, independently, so k drafts yield 1 + Alpha + ... + Alpha^k tokens
// a step in expectation.
type Acceptance struct {
	Alpha float64
	// Positions, when measured, is the share of steps whose first i drafts
	// were all accepted, for i = 1, 2, ... (the record's per-position
	// acceptance under the fixed policy, which issues every position): the
	// expected tokens of k drafts are then 1 plus the first k shares, and
	// Alpha is not used.
	Positions []float64
}

// DeepSeekV41FlashAcceptance is fitted to the measured 2.67 tokens a step
// at 4 drafts (one row, greedy sampling): Alpha 0.68 reproduces 2.14 at 2
// drafts (measured 2.21), 2.46 at 3 (2.37) and 2.82 at 5 (2.46 with the
// model's own five-draft block, whose fifth draft is rarely accepted).
var DeepSeekV41FlashAcceptance = Acceptance{Alpha: 0.68}

// Tokens is the expected tokens a step with k drafts (k may be fractional
// when a policy cuts the drafts: the expectation is interpolated).
func (a Acceptance) Tokens(k float64) float64 {
	if k <= 0 {
		return 1
	}
	whole := math.Floor(k)
	if len(a.Positions) > 0 {
		sum := 1.0
		for i := 0; i < int(whole) && i < len(a.Positions); i++ {
			sum += a.Positions[i]
		}
		if next := int(whole); next < len(a.Positions) {
			sum += (k - whole) * a.Positions[next]
		}
		return sum
	}
	sum := 1.0
	term := 1.0
	for i := 0; i < int(whole); i++ {
		term *= a.Alpha
		sum += term
	}
	return sum + (k-whole)*term*a.Alpha
}

// FitAcceptance finds the Alpha that yields tokens a step at k drafts.
func FitAcceptance(k int, tokens float64) Acceptance {
	lo, hi := 0.0, 1.0
	for i := 0; i < 60; i++ {
		mid := (lo + hi) / 2
		if (Acceptance{Alpha: mid}).Tokens(float64(k)) < tokens {
			lo = mid
		} else {
			hi = mid
		}
	}
	return Acceptance{Alpha: (lo + hi) / 2}
}

// Prompt mixes a load names.
const (
	MixSame      = "same"      // one prompt in every row: the rows select the same experts
	MixDifferent = "different" // a prompt per row
)

// measuredDistinct is the distinct experts a layer a batch of n rows with
// a prompt per row selects, measured on DeepSeek-V4.1-Flash (top-6 of 384)
// with hero 9's 32 prompts: the rows' choices overlap far more than
// independent draws would (33 independent rows would select 155).
var measuredDistinct = [][2]float64{{1, 6}, {9, 37.3}, {33, 90.3}}

// DistinctExperts is the distinct experts a layer that n token rows with a
// prompt per row select, for a model with topK of experts: log-linear
// between the measured points, extended past the last with its slope
// (an estimate the wide batches measure), never more than the layer has.
// With one prompt in every row it is topK: the rows select the same.
func DistinctExperts(topK, experts int, n int, mix string) float64 {
	if n <= 1 || mix == MixSame {
		return float64(topK)
	}
	pts := measuredDistinct
	scale := float64(topK) / pts[0][1]
	x := float64(n)
	for i := 1; i < len(pts); i++ {
		if x <= pts[i][0] || i == len(pts)-1 {
			a, b := pts[i-1], pts[i]
			slope := math.Log(b[1]/a[1]) / math.Log(b[0]/a[0])
			d := a[1] * math.Pow(x/a[0], slope) * scale
			return math.Min(d, float64(experts))
		}
	}
	return float64(topK)
}

// tokenCurve: t consecutive token rows of a row's verify select experts
// as if the row's first token were drawn t^0.65 times from the experts
// it does not yet read. Measured Nov 1 (pod hero-d, VXPU_EXPERT_STATS=1):
// one row at five tokens reads 18.3 distinct experts a layer (the form
// gives 17.0), eight different prompts at five tokens 83.2 (88.8).
const tokenCurve = 0.65

// DistinctExpertsOfVerify is the distinct experts a layer a verify of rows
// rows with tokens token rows each reads: the rows' count D for their
// first tokens (DistinctExperts), grown toward the layer's E experts as
// E - (E - D) (1 - D/E)^(tokens^tokenCurve - 1), so the growth saturates
// where the rows already read most of the layer. With one prompt in every
// row the rows read the same and the growth is one row's.
func DistinctExpertsOfVerify(topK, experts, rows int, tokens float64, mix string) float64 {
	d := DistinctExperts(topK, experts, rows, mix)
	if tokens <= 1 {
		return d
	}
	e := float64(experts)
	return e - (e-d)*math.Pow(1-d/e, math.Pow(tokens, tokenCurve)-1)
}

// SpecPrediction is what the model predicts for a speculating pipeline.
type SpecPrediction struct {
	Pipeline      PipelinePrediction     `json:"pipeline"`
	Speculation   kernelplan.Speculation `json:"speculation"`
	TokensPerStep float64                `json:"tokens_per_step"`  // accepted drafts plus the verified token, per row per step
	DraftedPerRow float64                `json:"drafted_per_row"`  // drafts the policy issues per row per step
	Distinct      float64                `json:"distinct_experts"` // distinct experts a layer the verify reads
	StepMs        float64                `json:"step_ms"`          // one batch's step in steady state, the drafter included
	PeriodMs      float64                `json:"period_ms"`        // between batch steps completing
	AggregateTokS float64                `json:"aggregate_tok_s"`
	PerBatchTokS  float64                `json:"per_batch_tok_s"`
	MsPerToken    float64                `json:"ms_per_token"`           // a row's latency per token: the step over the tokens it yields
	FixedLength   *FixedLengthRun        `json:"fixed_length,omitempty"` // what a run of Load.NewTokens a row measures
	Basis         []string               `json:"basis"`
}

// PredictSpeculation predicts the pipelined step of plan p over model lb
// for batches of w.Rows rows with prompt mix (w.DistinctExperts is
// ignored: the mix and the drafts set it), drafting by spec.
func PredictSpeculation(lb LayerBytes, p *kernelplan.PipelinePlan, w Workload, mix string, spec kernelplan.Speculation,
	c PipelineCost, sc SpecCost, acc Acceptance) (SpecPrediction, error) {
	if err := spec.Validate(); err != nil {
		return SpecPrediction{}, err
	}
	if w.Rows < 1 {
		return SpecPrediction{}, fmt.Errorf("workload: rows %d: want at least 1", w.Rows)
	}
	experts, topK := 0, lb.TopK
	for _, l := range lb.Layers {
		if l.Experts > experts {
			experts = l.Experts
		}
	}
	k := float64(spec.Draft)
	pr := SpecPrediction{Speculation: spec}
	// what the policy issues and what it yields
	tokens := acc.Tokens(k)
	drafted := k
	var basis []string
	if r, ok := sc.Replayed[ReplayKey(spec)]; ok && k > 0 {
		tokens, drafted = r.TokensPerStep, r.TokenRowsPerStep-1
		basis = append(basis, fmt.Sprintf("%s: replayed over the engine's recorded steps, %.2f drafts issued and %.2f tokens a step", ReplayKey(spec), drafted, tokens))
	} else {
		switch spec.Policy {
		case kernelplan.PolicyConfidence:
			if k > 0 {
				accepted := tokens - 1
				wasted := k - accepted
				st, measured := sc.stat(spec.MinConfidence)
				drafted = (1-st.Loss)*accepted + (1-st.Precision)*wasted
				tokens = 1 + (1-st.Loss)*accepted
				source := "estimates until the engine records the confidence histogram"
				if measured {
					source = "measured on the recorded steps"
				}
				basis = append(basis, fmt.Sprintf("confidence policy (threshold %.2f): of the fixed policy's %.2f wasted drafts a row %.0f%% are foreseen and not issued, at %.0f%% of its accepted drafts forgone (%s)",
					spec.MinConfidence, wasted, st.Precision*100, st.Loss*100, source))
			}
		case kernelplan.PolicyExperts:
			if k > 0 {
				rowsDistinct := DistinctExperts(topK, experts, w.Rows, mix)
				newPerToken := float64(topK) * (1 - rowsDistinct/float64(experts)) * (1 - sc.PreferGain)
				allowed := k
				if newPerToken > 0 && float64(spec.MaxNewExperts)/newPerToken < k {
					allowed = float64(spec.MaxNewExperts) / newPerToken
				}
				drafted = allowed
				tokens = acc.Tokens(allowed)
				basis = append(basis, fmt.Sprintf("experts policy (%d candidates, %d new experts a layer a row): a draft token adds about %.1f experts the batch does not read, so %.1f of %d drafts fit the budget (the preference's %.0f%% saving is an estimate until measured)",
					spec.Candidates, spec.MaxNewExperts, newPerToken, allowed, spec.Draft, sc.PreferGain*100))
			}
		}
	}
	// the distinct experts of the verify: the rows' for their first tokens,
	// grown by the token curve, less what the experts policy saves
	tokenRows := float64(w.Rows) * (1 + drafted)
	distinct := DistinctExpertsOfVerify(topK, experts, w.Rows, 1+drafted, mix)
	if spec.Policy == kernelplan.PolicyExperts && drafted > 0 {
		base := DistinctExperts(topK, experts, w.Rows, mix)
		distinct = base + (distinct-base)*(1-sc.PreferGain)
	}
	ww := w
	ww.DistinctExperts = distinct
	ww.Drafter = k > 0
	base, err := PredictPipeline(lb, p, ww, c)
	if err != nil {
		return SpecPrediction{}, err
	}
	if k > 0 {
		backbone := backboneLayers(lb)
		// the experts of the verify: the distinct experts' bytes at the read peak plus the pairs'
		// cost, in place of what PredictPipeline priced for the bytes alone; the host node's gap
		// past HostGapRows token rows; the dense families' token rows; the drafter on the last range
		efficacy := 1.0
		if w.Rows > 1 && distinct > float64(topK)*1.5 {
			efficacy = c.GroupedEfficacy
		}
		pairs := tokenRows * float64(topK)
		pairUs := sc.ExpertPairUs
		if mix == MixSame {
			pairUs = sc.ExpertPairUsShared
		}
		readPeak := sc.ReadPeakGBs
		if readPeak <= 0 {
			readPeak = c.BandwidthGBs
		}
		var pairMsAll, hostAll float64
		for i := range base.Ranges {
			rc := &base.Ranges[i]
			extra := sc.PerDraftRowMs * float64(w.Rows) * drafted * float64(rc.Layers) / float64(backbone)
			if rc.Layers > 0 {
				priced := rc.ExpertGB * 1e9 / (c.BandwidthGBs * 1e9 * efficacy) * 1e3
				bytesMs := rc.ExpertGB * 1e9 / (readPeak * 1e9) * 1e3
				pairMs := float64(rc.Layers) * pairs * pairUs / 1e3
				pairMsAll += pairMs
				extra += bytesMs + pairMs - priced
				if tokenRows > float64(sc.HostGapRows) && mix != MixSame {
					host := sc.HostGapMs * float64(rc.Layers) / float64(backbone)
					hostAll += host
					extra += host
				}
			}
			if i == len(base.Ranges)-1 {
				extra += sc.DrafterStepMs + sc.DrafterRowMs*float64(w.Rows-1)
			}
			rc.FixedMs += extra
			rc.Ms += extra
		}
		summarize(&base, w.Rows, w.Batches, len(p.Ranges))
		basis = append(basis, fmt.Sprintf("%.1f drafts a row verified as %.1f token rows: %.0f distinct experts a layer (the rows curve times the token curve) at the %.0f GB/s read peak plus %.0f pairs a layer at %.1f µs each (%.1f ms), the experts host node's %.1f ms past %d token rows, %.2f ms a token row in the dense families and the drafter's %.1f ms a step on the last range",
			drafted, tokenRows, distinct, readPeak, pairs, pairUs, pairMsAll, hostAll, sc.HostGapRows, sc.PerDraftRowMs, sc.DrafterStepMs+sc.DrafterRowMs*float64(w.Rows-1)))
	}
	pr.Pipeline = base
	pr.TokensPerStep = tokens
	pr.DraftedPerRow = drafted
	pr.Distinct = distinct
	pr.StepMs = base.LatencyMs
	pr.PeriodMs = base.PeriodMs
	pr.AggregateTokS = base.AggregateTokS * tokens
	pr.PerBatchTokS = base.PerBatchTokS * tokens
	pr.MsPerToken = base.LatencyMs / tokens
	pr.Basis = append(basis, fmt.Sprintf("%.2f tokens a step a row (acceptance %.2f a position): %.0f tok/s aggregate, %.1f ms a token for a row", tokens, acc.Alpha, pr.AggregateTokS, pr.MsPerToken))
	return pr, nil
}

// Load is what a caller asks the planner to serve.
type Load struct {
	Rows int    // rows in all
	Mix  string // MixSame or MixDifferent
	// Batches in flight; 0 lets the planner choose between one batch and
	// one per range.
	Batches int
	// LatencyMs bounds a row's time per token (0: none).
	LatencyMs float64
	// Drafter says the artifact has one; MaxDraft is the longest draft to consider.
	Drafter  bool
	MaxDraft int
	// NewTokens, when set, is the length of a fixed-length run (the hero
	// harness decodes 64 or 256 tokens a row): the planner then also
	// predicts what such a run measures, which is below the steady state
	// when the rows of a batch draft, because a row that accepts few
	// drafts takes more steps and the batch steps until its slowest row
	// is done (FixedLengthRun).
	NewTokens int
}

// FixedLengthRun is what a run of a fixed number of new tokens a row
// measures under speculation: the mean row finishes in MeanSteps, the
// slowest of a batch's rows in SlowestSteps, and the batch's tokens over
// its steps are Share of the steady state (every row still drafting).
// Hero run 12 measured 357 tok/s for four batches of eight different
// prompts at two drafts where the steady-state model said 835: 42 steps
// for 64 tokens with a mean of 4.9 rows still active (hero run 13).
type FixedLengthRun struct {
	Tokens       int     `json:"tokens"`
	MeanSteps    float64 `json:"mean_steps"`
	SlowestSteps float64 `json:"slowest_steps"`
	Share        float64 `json:"share_of_steady_state"`
	TokS         float64 `json:"tok_s"` // the aggregate the run measures
}

// stepDistribution is the probability of accepting j of k drafts in a
// step, j = 0..k, from the per-position acceptance (Positions when
// measured, else Alpha^i).
func (a Acceptance) stepDistribution(k int) []float64 {
	survive := make([]float64, k+1) // share of steps whose first i drafts are all accepted
	survive[0] = 1
	for i := 1; i <= k; i++ {
		if len(a.Positions) >= i {
			survive[i] = a.Positions[i-1]
		} else {
			survive[i] = survive[i-1] * a.Alpha
		}
		if survive[i] > survive[i-1] {
			survive[i] = survive[i-1]
		}
	}
	dist := make([]float64, k+1)
	for j := 0; j <= k; j++ {
		next := 0.0
		if j < k {
			next = survive[j+1]
		}
		dist[j] = survive[j] - next
	}
	return dist
}

// Completion is what a fixed-length run of tokens new tokens a row
// measures for a batch of rows rows drafting k a step: the steps a row's
// count reaches tokens in (one plus the accepted drafts a step), exactly
// from the step distribution, and the slowest of rows independent rows.
// Rows are not independent in practice (a prose prompt accepts far fewer
// drafts than a code prompt: hero run 13 measured the slowest of eight
// rows at 42 steps where this gives 27), so the share is a ceiling on
// what such a run measures.
func (a Acceptance) Completion(rows, tokens, k int) FixedLengthRun {
	out := FixedLengthRun{Tokens: tokens}
	if k <= 0 || rows < 1 || tokens < 1 {
		out.MeanSteps, out.SlowestSteps, out.Share = float64(tokens), float64(tokens), 1
		return out
	}
	dist := a.stepDistribution(k)
	// reach[s][t]: probability that after s steps a row has exactly t tokens (capped at tokens)
	maxSteps := tokens
	cur := make([]float64, tokens+1)
	cur[0] = 1
	done := make([]float64, maxSteps+1) // cdf of a row's finishing step
	for s := 1; s <= maxSteps; s++ {
		next := make([]float64, tokens+1)
		for t, pr := range cur {
			if pr == 0 {
				continue
			}
			if t >= tokens {
				next[tokens] += pr
				continue
			}
			for j, pj := range dist {
				nt := t + 1 + j
				if nt > tokens {
					nt = tokens
				}
				next[nt] += pr * pj
			}
		}
		cur = next
		done[s] = cur[tokens]
	}
	mean, slowest := 0.0, 0.0
	for s := 1; s <= maxSteps; s++ {
		mean += float64(s) * (done[s] - done[s-1])
		slowest += float64(s) * (math.Pow(done[s], float64(rows)) - math.Pow(done[s-1], float64(rows)))
	}
	out.MeanSteps, out.SlowestSteps = mean, slowest
	steady := a.Tokens(float64(k))
	if slowest > 0 {
		out.Share = float64(tokens) / slowest / steady
	}
	return out
}

// Option is one serving shape and what the model predicts for it.
type Option struct {
	Batches     int                     `json:"batches"`
	Rows        int                     `json:"rows"` // a batch's rows
	Speculation *kernelplan.Speculation `json:"speculation,omitempty"`
	Prediction  SpecPrediction          `json:"prediction"`
	Rejected    string                  `json:"rejected,omitempty"` // why it was not chosen; empty for the choice
}

// ChooseShape picks, among the shapes the engine has (one wide batch or
// one batch per range; no drafter or 1 to MaxDraft drafts under each
// policy), the one the model predicts to yield the most tokens a second
// within the latency bound, and returns it first with every option after
// it, best first, each rejected one saying why.
func ChooseShape(lb LayerBytes, p *kernelplan.PipelinePlan, load Load, c PipelineCost, sc SpecCost, acc Acceptance) ([]Option, error) {
	if load.Rows < 1 {
		return nil, fmt.Errorf("load: rows %d: want at least 1", load.Rows)
	}
	if load.Mix != MixSame && load.Mix != MixDifferent {
		return nil, fmt.Errorf("load: mix %q: want %q or %q", load.Mix, MixSame, MixDifferent)
	}
	batchChoices := []int{load.Batches}
	if load.Batches == 0 {
		batchChoices = []int{1}
		if n := len(p.Ranges); n > 1 && load.Rows%n == 0 {
			batchChoices = append(batchChoices, n)
		}
	}
	var specs []*kernelplan.Speculation
	specs = append(specs, nil)
	if load.Drafter {
		for k := 1; k <= load.MaxDraft; k++ {
			specs = append(specs, &kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyFixed})
			if k >= 2 {
				specs = append(specs,
					&kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyConfidence, MinConfidence: 0.5},
					&kernelplan.Speculation{Draft: k, Policy: kernelplan.PolicyExperts, MaxNewExperts: 2 * lb.TopK, Candidates: 2})
			}
		}
	}
	var options []Option
	for _, batches := range batchChoices {
		if batches < 1 || load.Rows%batches != 0 {
			continue
		}
		q := *p
		q.Pipeline.Batches = batches
		for _, s := range specs {
			spec := kernelplan.Speculation{Policy: kernelplan.PolicyFixed}
			if s != nil {
				spec = *s
			}
			pred, err := PredictSpeculation(lb, &q, Workload{Rows: load.Rows / batches, Batches: batches}, load.Mix, spec, c, sc, acc)
			if err != nil {
				return nil, err
			}
			if load.NewTokens > 0 {
				run := acc.Completion(load.Rows/batches, load.NewTokens, int(math.Round(pred.DraftedPerRow)))
				run.TokS = pred.AggregateTokS * run.Share
				pred.FixedLength = &run
				if spec.Draft > 0 {
					pred.Basis = append(pred.Basis, fmt.Sprintf("a run of %d tokens a row measures %.0f tok/s, %.0f%% of the steady state: the mean row finishes in %.1f steps, the slowest of %d in %.1f, and the batch steps until the slowest is done",
						load.NewTokens, run.TokS, run.Share*100, run.MeanSteps, load.Rows/batches, run.SlowestSteps))
				}
			}
			options = append(options, Option{Batches: batches, Rows: load.Rows / batches, Speculation: s, Prediction: pred})
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
		if chosen < 0 {
			chosen = i
			continue
		}
		o.Rejected = fmt.Sprintf("%.0f tok/s, %.0f%% under the choice's %.0f", o.Prediction.AggregateTokS,
			(1-o.Prediction.AggregateTokS/options[chosen].Prediction.AggregateTokS)*100, options[chosen].Prediction.AggregateTokS)
	}
	if chosen < 0 {
		return options, fmt.Errorf("no shape serves %d rows within %.1f ms a token: the best is %.1f ms", load.Rows, load.LatencyMs, options[0].Prediction.MsPerToken)
	}
	if chosen > 0 {
		options[0], options[chosen] = options[chosen], options[0]
	}
	return options, nil
}

// Describe names an option's shape in a line.
func (o Option) Describe() string {
	s := fmt.Sprintf("%d batch(es) of %d rows", o.Batches, o.Rows)
	if o.Speculation == nil || o.Speculation.Draft == 0 {
		return s + ", no drafter"
	}
	s += fmt.Sprintf(", %d drafts, %s policy", o.Speculation.Draft, o.Speculation.Policy)
	switch o.Speculation.Policy {
	case kernelplan.PolicyConfidence:
		s += fmt.Sprintf(" (threshold %.2f)", o.Speculation.MinConfidence)
	case kernelplan.PolicyExperts:
		s += fmt.Sprintf(" (%d candidates, %d new experts a layer a row)", o.Speculation.Candidates, o.Speculation.MaxNewExperts)
	}
	return s
}

// MeasuredSpeculation reads the engine's speculation statistics from a
// kernel record (stats.speculation, written by the drafter path: per
// draft position the acceptance share under the fixed policy, and per
// confidence threshold the precision and loss the engine replayed over
// the recorded steps), as the model's inputs: the per-position acceptance
// and the thresholds' statistics. A record without them is an error.
func MeasuredSpeculation(record []byte) (Acceptance, map[string]ConfidenceStat, map[string]Replay, error) {
	var r struct {
		Stats struct {
			Speculation *struct {
				Alpha     float64 `json:"alpha"`
				Positions []struct {
					Position int     `json:"position"`
					Rate     float64 `json:"rate"`
				} `json:"positions"`
				Thresholds map[string]ConfidenceStat `json:"thresholds"`
				Policies   []struct {
					Policy           string  `json:"policy"`
					Draft            int     `json:"draft"`
					MinConfidence    float64 `json:"min_confidence"`
					TokensPerStep    float64 `json:"tokens_per_step"`
					TokenRowsPerStep float64 `json:"token_rows_per_step"`
				} `json:"policies"`
			} `json:"speculation"`
		} `json:"stats"`
	}
	if err := json.Unmarshal(record, &r); err != nil {
		return Acceptance{}, nil, nil, fmt.Errorf("speculation statistics: %w", err)
	}
	s := r.Stats.Speculation
	if s == nil || len(s.Positions) == 0 {
		return Acceptance{}, nil, nil, fmt.Errorf("speculation statistics: the record has none (a drafter run writes stats.speculation)")
	}
	acc := Acceptance{Alpha: s.Alpha, Positions: make([]float64, len(s.Positions))}
	for _, p := range s.Positions {
		if p.Position < 1 || p.Position > len(s.Positions) {
			return Acceptance{}, nil, nil, fmt.Errorf("speculation statistics: position %d of %d", p.Position, len(s.Positions))
		}
		acc.Positions[p.Position-1] = p.Rate
	}
	replays := map[string]Replay{}
	for _, p := range s.Policies {
		key := ReplayKey(kernelplan.Speculation{Draft: p.Draft, Policy: p.Policy, MinConfidence: p.MinConfidence})
		replays[key] = Replay{TokensPerStep: p.TokensPerStep, TokenRowsPerStep: p.TokenRowsPerStep}
	}
	return acc, s.Thresholds, replays, nil
}
