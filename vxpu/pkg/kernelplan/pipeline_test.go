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
	"strings"
	"testing"
)

func TestPipelinePlan_RoundTripAndSwitch(t *testing.T) {
	ranges, err := ParseRanges("0:8,8:20,20:32,32:43")
	if err != nil {
		t.Fatal(err)
	}
	p := &PipelinePlan{Format: PlanFormat, GPU: "NVIDIA RTX PRO 6000 Blackwell Server Edition", Ranges: ranges,
		Experts: "with-layers", Pipeline: Pipeline{Batches: 4, Schedule: ScheduleRangeOrder}}
	data, err := p.Write()
	if err != nil {
		t.Fatal(err)
	}
	q, err := ReadPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	if q.Layers() != 43 || q.Pipeline.Batches != 4 || len(q.Ranges) != 4 {
		t.Fatalf("round trip lost something: %+v", q)
	}
	if got, want := q.LayerDevices(), "0:8=cuda:0,8:20=cuda:1,20:32=cuda:2,32:43=cuda:3"; got != want {
		t.Fatalf("layer devices %q, want %q", got, want)
	}
	p.Speculation = &Speculation{Draft: 4, Policy: PolicyConfidence, MinConfidence: 0.4}
	if data, err = p.Write(); err != nil {
		t.Fatal(err)
	}
	if q, err = ReadPlan(data); err != nil {
		t.Fatal(err)
	}
	if q.Speculation == nil || q.Speculation.Draft != 4 || q.Speculation.Policy != PolicyConfidence || q.Speculation.MinConfidence != 0.4 {
		t.Fatalf("round trip lost the speculation section: %+v", q.Speculation)
	}
}

func TestPipelinePlan_Refusals(t *testing.T) {
	cases := map[string]string{
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"},{"gpu":1,"layers":"9:20"}],"pipeline":{"batches":1,"schedule":"range-order"}}`:                                   "contiguous",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"},{"gpu":0,"layers":"8:20"}],"pipeline":{"batches":1,"schedule":"range-order"}}`:                                   "already holds",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":0,"schedule":"range-order"}}`:                                                             "at least 1",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":2,"schedule":"lifo"}}`:                                                                    "schedule",
		`{"format":"vxpu-plan/2","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":2,"schedule":"range-order"}}`:                                                             "format",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":2,"schedule":"range-order"},"surprise":1}`:                                                "unknown field",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":1,"schedule":"range-order"},"speculation":{"draft":4,"policy":"lucky"}}`:                  "policy",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":1,"schedule":"range-order"},"speculation":{"draft":4,"policy":"confidence"}}`:             "min_confidence",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":1,"schedule":"range-order"},"speculation":{"draft":9,"policy":"fixed"}}`:                  "0 to 8",
		`{"format":"vxpu-plan/1","ranges":[{"gpu":0,"layers":"0:8"}],"pipeline":{"batches":1,"schedule":"range-order"},"speculation":{"draft":4,"policy":"experts","candidates":0}}`: "candidates",
	}
	for plan, want := range cases {
		_, err := ReadPlan([]byte(plan))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("plan %s: error %v, want one mentioning %q", plan, err, want)
		}
	}
}

func TestValidateDrafterGPU(t *testing.T) {
	plan := &PipelinePlan{Format: PlanFormat, Ranges: []Range{{GPU: 0, Layers: "0:8"}, {GPU: 1, Layers: "8:43"}},
		Pipeline: Pipeline{Batches: 1, Schedule: ScheduleRangeOrder}}
	if err := plan.Validate(); err != nil {
		t.Fatalf("a plan without a drafter field: %v", err)
	}
	plan.Drafter = &Drafter{GPU: 0}
	if err := plan.Validate(); err != nil {
		t.Fatalf("the drafter on a GPU that holds a range: %v", err)
	}
	plan.Drafter = &Drafter{GPU: 3}
	if err := plan.Validate(); err == nil {
		t.Fatalf("the drafter on a GPU without a range should be refused")
	}
}
