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
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// PlanFormat names the physical plan an engine reads (hero run 9): the
// layer ranges per GPU, the pipeline of batches over them, and the kernel
// table to apply. It is the data the sharding switches
// (VXPU_LAYER_DEVICES, VXPU_GPU_EXPERTS) and the hand placement file were.
const PlanFormat = "vxpu-plan/1"

// Schedule rules the engine knows.
const (
	// ScheduleRangeOrder: batches are admitted round-robin; a batch enters
	// range k when GPU k is free and the batch's range k-1 has finished,
	// so GPU k always works on some batch's range k.
	ScheduleRangeOrder = "range-order"
)

// PipelinePlan is the physical plan of a step across GPUs.
type PipelinePlan struct {
	Format string `json:"format"`
	// GPU names the card the plan was made for; empty: any.
	GPU string `json:"gpu,omitempty"`
	// Ranges are the layer ranges, one per GPU, in step order. A range's
	// layers include the drafter's stages where the model has them (their
	// ids follow the backbone's: 40, 41, 42 for a 40-layer model).
	Ranges []Range `json:"ranges"`
	// Experts says where a range's routed experts live: "with-layers" (every
	// expert of a range's layers on the range's GPU, the only placement
	// today) or the path of a placement file.
	Experts  string   `json:"experts,omitempty"`
	Pipeline Pipeline `json:"pipeline"`
	// Kernels is the kernel table (vxpu-kernel-table/1) to apply: a path, or
	// the table inline; empty lets the engine measure.
	Kernels json.RawMessage `json:"kernels,omitempty"`
	// Speculation is how a step drafts and verifies tokens (hero run 10);
	// nil: one token a step, no drafter.
	Speculation *Speculation `json:"speculation,omitempty"`
	// Parallel is how a range's layers are spread over several cards at
	// once (hero run 14): the dense weights split across the cards and the
	// experts split among them; nil: each range whole on its GPU.
	Parallel *Parallel `json:"parallel,omitempty"`
	// Explain says why, line by line; the engine does not read it.
	Explain []string `json:"explain,omitempty"`
}

// Accept policies the engine knows (vxpu/python/vxpu/speculation.py applies them).
const (
	// PolicyFixed: every step drafts Draft tokens and keeps the drafts that
	// equal the verified choice up to the first that does not.
	PolicyFixed = "fixed"
	// PolicyConfidence: the drafter stops at the first draft whose top
	// probability is under MinConfidence, so a step verifies fewer token rows
	// where the drafter is unsure; acceptance is the fixed policy's.
	PolicyConfidence = "confidence"
	// PolicyExperts: among the drafter's candidate continuations the one whose
	// experts the step already reads is preferred, and the drafts are cut
	// where they would add more than MaxNewExperts experts a layer beyond
	// what the batch reads; acceptance is the fixed policy's.
	PolicyExperts = "experts"
)

// Reduce forms the engine knows for the cross-card sum of a tensor-parallel
// step (hero run 14; dev/hero/crossings.py measured them).
const (
	// ReduceOneShot: every card writes its partial into the other cards'
	// inboxes with peer stores, raises a flag per destination, waits for
	// the others' flags and sums locally. Latency-bound: about 10 µs a
	// reduction at one row inside a graph, 16 at 8 rows, 52 at 33, 194 at
	// 132 (bandwidth-bound there, about 40 GB/s a card).
	ReduceOneShot = "one-shot"
	// ReduceRing: a reduce-scatter and all-gather around the cards, each
	// pushing a quarter of the payload per hop: the bandwidth-optimal form
	// for wide rows, estimated to halve the one-shot cost past 33 rows.
	ReduceRing = "ring"
)

// Parallel is how the plan spreads one range's layers over several cards
// at once. With DenseTP of 4 every dense weight is split four ways
// (columns into a block, rows out of it), each card streams a quarter of
// the dense bytes a step, and the block's partial outputs are summed across
// the cards twice a layer (after the attention and after the FFN). The
// router runs on the summed, replicated activation, so each card knows the
// routing without a message; with ExpertEP equal to DenseTP each card holds
// a share of every layer's experts and runs only the pairs that land on its
// experts, its partial joining the FFN's sum. The ranges of such a plan
// are one range, the whole step, on the group's first card; GPUs lists the
// group.
type Parallel struct {
	// DenseTP is the cards a dense weight is split across: 1 (none), 2 or 4.
	DenseTP int `json:"dense_tp"`
	// ExpertEP is the cards a layer's experts are spread over: 1 (every
	// expert on every card of the group, which rarely fits) or DenseTP.
	ExpertEP int `json:"expert_ep"`
	// GPUs is the group, DenseTP cards, in the order the slices are numbered.
	GPUs []int `json:"gpus,omitempty"`
	// Experts says which card holds which expert: "" or "by-id" (expert e
	// of every layer on the group's card e mod ExpertEP), "by-popularity"
	// (the placement the planner balances from measured popularity, written
	// into the placement file the engine reads), or the path of a placement
	// file.
	Experts string `json:"experts,omitempty"`
	// Reduce is the cross-card sum's form, ReduceOneShot or ReduceRing;
	// empty: the planner's choice by payload (one-shot up to 33 rows a step).
	Reduce string `json:"reduce,omitempty"`
}

// Validate checks a parallel section the engine would refuse.
func (pp *Parallel) Validate(ranges []Range) error {
	switch pp.DenseTP {
	case 1, 2, 4:
	default:
		return fmt.Errorf("parallel dense_tp %d: want 1, 2 or 4", pp.DenseTP)
	}
	if pp.ExpertEP != 1 && pp.ExpertEP != pp.DenseTP {
		return fmt.Errorf("parallel expert_ep %d: want 1 or dense_tp (%d)", pp.ExpertEP, pp.DenseTP)
	}
	if pp.DenseTP > 1 {
		if len(pp.GPUs) != pp.DenseTP {
			return fmt.Errorf("parallel gpus %v: want the %d cards of the group", pp.GPUs, pp.DenseTP)
		}
		seen := map[int]bool{}
		for _, g := range pp.GPUs {
			if g < 0 || seen[g] {
				return fmt.Errorf("parallel gpus %v: want distinct cards", pp.GPUs)
			}
			seen[g] = true
		}
		if len(ranges) != 1 {
			return fmt.Errorf("parallel dense_tp %d with %d ranges: a tensor-parallel plan has one range, the whole step on the group (tensor parallelism inside a pipeline is not planned yet)", pp.DenseTP, len(ranges))
		}
		if ranges[0].GPU != pp.GPUs[0] {
			return fmt.Errorf("parallel gpus %v: the range's gpu %d must be the group's first card", pp.GPUs, ranges[0].GPU)
		}
	}
	switch pp.Experts {
	case "", "by-id", "by-popularity":
	default:
		if !strings.HasSuffix(pp.Experts, ".json") {
			return fmt.Errorf("parallel experts %q: want \"by-id\", \"by-popularity\" or a placement file", pp.Experts)
		}
	}
	switch pp.Reduce {
	case "", ReduceOneShot, ReduceRing:
	default:
		return fmt.Errorf("parallel reduce %q: the engine knows %q and %q", pp.Reduce, ReduceOneShot, ReduceRing)
	}
	return nil
}

// Cards is the number of cards the step streams at once: DenseTP, or 1
// without a parallel section.
func (p *PipelinePlan) Cards() int {
	if p.Parallel == nil || p.Parallel.DenseTP < 1 {
		return 1
	}
	return p.Parallel.DenseTP
}

// Speculation is the plan's draft length and accept policy.
type Speculation struct {
	// Draft is the tokens drafted a step (the verify runs Draft+1 token rows
	// a row); 0 turns the drafter off.
	Draft  int    `json:"draft"`
	Policy string `json:"policy"`
	// MinConfidence is the confidence policy's threshold on the draft's top
	// probability, in (0, 1).
	MinConfidence float64 `json:"min_confidence,omitempty"`
	// MaxNewExperts is the experts policy's budget of experts a layer a row's
	// drafts may add beyond what the batch reads for its first tokens.
	MaxNewExperts int `json:"max_new_experts,omitempty"`
	// Candidates is how many continuations the drafter proposes per position
	// for the experts policy to choose among (1: no choice).
	Candidates int `json:"candidates,omitempty"`
}

// Validate checks a speculation section the engine would refuse.
func (s *Speculation) Validate() error {
	if s.Draft < 0 || s.Draft > 8 {
		return fmt.Errorf("speculation draft %d: want 0 to 8", s.Draft)
	}
	switch s.Policy {
	case PolicyFixed:
	case PolicyConfidence:
		if !(s.MinConfidence > 0 && s.MinConfidence < 1) {
			return fmt.Errorf("speculation policy %q: min_confidence %.3f, want a probability in (0, 1)", s.Policy, s.MinConfidence)
		}
	case PolicyExperts:
		if s.MaxNewExperts < 0 {
			return fmt.Errorf("speculation policy %q: max_new_experts %d, want 0 or more", s.Policy, s.MaxNewExperts)
		}
		if s.Candidates < 1 {
			return fmt.Errorf("speculation policy %q: candidates %d, want 1 or more", s.Policy, s.Candidates)
		}
	default:
		return fmt.Errorf("speculation policy %q: the engine knows %q, %q and %q", s.Policy, PolicyFixed, PolicyConfidence, PolicyExperts)
	}
	return nil
}

// Range is a contiguous span of layers on one GPU: Layers is "first:last+1".
type Range struct {
	GPU    int    `json:"gpu"`
	Layers string `json:"layers"`
}

// Pipeline is how many batches are in flight and how they move.
type Pipeline struct {
	// Batches in flight: 1 is today's single step walking the ranges.
	Batches  int    `json:"batches"`
	Schedule string `json:"schedule"`
}

// Span parses a range's layers, "a:b" meaning a <= layer < b.
func (r Range) Span() (first, end int, err error) {
	parts := strings.SplitN(r.Layers, ":", 2)
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("range %q: want first:end", r.Layers)
	}
	first, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	end, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil || first < 0 || end <= first {
		return 0, 0, fmt.Errorf("range %q: want first:end with 0 <= first < end", r.Layers)
	}
	return first, end, nil
}

// first is the range's first layer, -1 when the span does not parse
// (Validate reports that; callers after it never see -1).
func (r Range) first() int {
	first, _, err := r.Span()
	if err != nil {
		return -1
	}
	return first
}

// Validate checks what the engine would refuse: the format, ranges that
// are contiguous from layer 0 in step order on distinct GPUs, a pipeline
// of at least one batch with a known schedule.
func (p *PipelinePlan) Validate() error {
	if p.Format != PlanFormat {
		return fmt.Errorf("plan format %q, want %q", p.Format, PlanFormat)
	}
	if len(p.Ranges) == 0 {
		return fmt.Errorf("plan has no ranges")
	}
	next := 0
	seen := map[int]bool{}
	for i, r := range p.Ranges {
		first, end, err := r.Span()
		if err != nil {
			return err
		}
		if first != next {
			return fmt.Errorf("range %d (%s) starts at layer %d, the previous one ended at %d: ranges must be contiguous from layer 0",
				i, r.Layers, first, next)
		}
		if seen[r.GPU] {
			return fmt.Errorf("range %d (%s): GPU %d already holds a range", i, r.Layers, r.GPU)
		}
		seen[r.GPU] = true
		next = end
	}
	if p.Pipeline.Batches < 1 {
		return fmt.Errorf("pipeline batches %d: want at least 1", p.Pipeline.Batches)
	}
	if p.Pipeline.Schedule != ScheduleRangeOrder {
		return fmt.Errorf("pipeline schedule %q: the engine knows %q", p.Pipeline.Schedule, ScheduleRangeOrder)
	}
	if p.Experts != "" && p.Experts != "with-layers" && !strings.HasSuffix(p.Experts, ".json") {
		return fmt.Errorf("experts %q: want \"with-layers\" or a placement file", p.Experts)
	}
	if p.Speculation != nil {
		if err := p.Speculation.Validate(); err != nil {
			return err
		}
	}
	if p.Parallel != nil {
		if err := p.Parallel.Validate(p.Ranges); err != nil {
			return err
		}
	}
	return nil
}

// Layers is the number of layers the ranges cover.
func (p *PipelinePlan) Layers() int {
	n := 0
	for _, r := range p.Ranges {
		if _, end, err := r.Span(); err == nil && end > n {
			n = end
		}
	}
	return n
}

// LayerDevices renders the ranges as the engine's sharding switch today
// ("0:8=cuda:0,8:20=cuda:1,..."), for an engine that reads the switch
// rather than the plan.
func (p *PipelinePlan) LayerDevices() string {
	parts := make([]string, 0, len(p.Ranges))
	for _, r := range p.Ranges {
		parts = append(parts, fmt.Sprintf("%s=cuda:%d", r.Layers, r.GPU))
	}
	return strings.Join(parts, ",")
}

// ReadPlan parses and validates a plan.
func ReadPlan(data []byte) (*PipelinePlan, error) {
	var p PipelinePlan
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("plan: %w", err)
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &p, nil
}

// ReadPlanFile reads a plan from path.
func ReadPlanFile(path string) (*PipelinePlan, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ReadPlan(data)
}

// Write renders the plan as the engine reads it, ranges in step order.
func (p *PipelinePlan) Write() ([]byte, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	q := *p
	q.Ranges = append([]Range(nil), p.Ranges...)
	sort.SliceStable(q.Ranges, func(i, j int) bool { return q.Ranges[i].first() < q.Ranges[j].first() })
	return json.MarshalIndent(q, "", " ")
}

// ParseRanges turns "0:8,8:20,20:32,32:43" into ranges on GPUs 0, 1, 2, ...
func ParseRanges(spec string) ([]Range, error) {
	var ranges []Range
	for i, s := range strings.Split(spec, ",") {
		r := Range{GPU: i, Layers: strings.TrimSpace(s)}
		if _, _, err := r.Span(); err != nil {
			return nil, err
		}
		ranges = append(ranges, r)
	}
	return ranges, nil
}
