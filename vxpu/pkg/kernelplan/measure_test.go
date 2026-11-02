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
	"strings"
	"testing"
)

func TestParseClass(t *testing.T) {
	c, err := ParseClass("gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1 epi=quant groups=8")
	if err != nil {
		t.Fatal(err)
	}
	if c.Template != "gemv_rows" || c.Rows != 1 || c.N != 5120 || c.K != 8192 || c.Fields["epi"] != "quant" || c.field("groups") != 8 {
		t.Errorf("parsed %+v", c)
	}
	for _, bad := range []string{"gemv_rows", "gemv_rows rows=1 n=5120", "gemv_rows rows=x n=1 k=1"} {
		if _, err := ParseClass(bad); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

func TestCandidatesFitTheShapeAndStartWithWhatRan(t *testing.T) {
	c, _ := ParseClass("gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1")
	ran := map[string]int{"BM": 16, "BN": 16, "BK": 256, "KSPLIT": 1, "warps": 4, "stages": 3}
	cands := Candidates(c, ran, 188)
	if len(cands) == 0 || blockKey(cands[0]) != "BK=256 BN=16 KSPLIT=1 stages=3 warps=4" {
		t.Fatalf("the engine's own blocks must come first: %v", cands)
	}
	for _, b := range cands {
		if 5120%b["BN"] != 0 || 8192%b["BK"] != 0 || b["KSPLIT"] != 1 {
			t.Errorf("candidate %v does not fit the class (no split without split=1)", b)
		}
	}
	// a narrow weight the caller splits: the split stops once the programs fill the card twice
	n, _ := ParseClass("gemv_rows rows=1 n=1280 k=5120 w=fp8 native=1 split=1")
	splits := map[int]bool{}
	for _, b := range Candidates(n, nil, 188) {
		splits[b["KSPLIT"]] = true
		if b["KSPLIT"] > 1 && (1280/b["BN"])*(b["KSPLIT"]/2) >= 2*188 {
			t.Errorf("candidate %v splits past a full card", b)
		}
	}
	if !splits[2] || !splits[4] {
		t.Errorf("a narrow weight should try K splits: %v", splits)
	}
	// the quantizing epilogue and the dual GEMV hold whole blocks of 32
	q, _ := ParseClass("gemv_rows rows=1 n=8192 k=32768 w=fp8 native=1 epi=quant groups=8")
	for _, b := range Candidates(q, nil, 188) {
		if b["BN"]%32 != 0 || (8192/8)%b["BN"] != 0 {
			t.Errorf("quantizing epilogue candidate %v", b)
		}
	}
	d, _ := ParseClass("dual_gemv_swiglu rows=1 n=2304 k=5120 w=fp8 native=1")
	for _, b := range Candidates(d, nil, 188) {
		if (b["BN"]%32 != 0 && b["KSPLIT"] == 1) || 2304%b["BN"] != 0 || (5120/b["KSPLIT"])%b["BK"] != 0 {
			t.Errorf("dual candidate %v", b)
		}
	}
	// the rotary epilogue: even BN dividing the head's unrotated part
	p, _ := ParseClass("gemv_pe rows=1 n=32768 k=1280 w=fp8 epi=rotary hd=512 native=1 rd=64")
	for _, b := range Candidates(p, nil, 188) {
		if (512-64)%b["BN"] != 0 || b["KSPLIT"] != 1 || 1280%b["BK"] != 0 {
			t.Errorf("rotary candidate %v", b)
		}
	}
}

type fakeRunner struct {
	us   func(Job) float64
	fail func(Job) bool
	jobs []Job
}

func (f *fakeRunner) Run(ctx context.Context, jobs []Job) ([]Result, error) {
	f.jobs = append(f.jobs, jobs...)
	var out []Result
	for _, j := range jobs {
		r := Result{Shapes: j.Shapes, Blocks: j.Blocks, Bytes: 42_000_000, FloorUs: 28.8}
		if f.fail != nil && f.fail(j) {
			r.Error = "CompilationError: out of shared memory"
		} else {
			r.Us = f.us(j)
			r.Efficiency = r.FloorUs / r.Us
		}
		out = append(out, r)
	}
	return out, nil
}

func TestMeasureChoosesTheFastestAndKeepsFailuresOut(t *testing.T) {
	classes := map[string]ClassRun{
		"gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1":         {Count: 120, Blocks: map[string]int{"BM": 16, "BN": 16, "BK": 256, "KSPLIT": 1, "warps": 4, "stages": 3}},
		"gemv_rows rows=1 n=1280 k=5120 w=fp8 native=1 split=1": {Count: 40, Blocks: map[string]int{"BM": 16, "BN": 16, "BK": 256, "KSPLIT": 2, "warps": 4, "stages": 3}},
	}
	runner := &fakeRunner{
		us: func(j Job) float64 { // BN 64 with 8 warps is fastest, 4 stages better still; a K split of 4 for the narrow one
			us := 40.0 - float64(j.Blocks["BN"])/4 - float64(j.Blocks["warps"]) - float64(j.Blocks["stages"])/10
			if strings.Contains(j.Shapes, "split=1") {
				us = 12.0 - float64(j.Blocks["KSPLIT"])
				if j.Blocks["KSPLIT"] >= 8 {
					us = 20
				}
			}
			return us
		},
		fail: func(j Job) bool { return j.Blocks["BN"] == 64 && j.Blocks["BK"] == 512 },
	}
	entries, explain, err := Measure(t.Context(), classes, runner, MeasureOptions{SMs: 188})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries: %+v", entries)
	}
	byKey := map[string]BlockEntry{}
	for _, e := range entries {
		byKey[e.Shapes] = e
	}
	wide := byKey["gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1"]
	if wide.Blocks["BN"] != 64 || wide.Blocks["warps"] != 8 || wide.Blocks["BK"] == 512 || wide.Blocks["stages"] != 4 {
		t.Errorf("the wide class should take the fastest candidate that ran, with the second pass's stages: %+v", wide)
	}
	narrow := byKey["gemv_rows rows=1 n=1280 k=5120 w=fp8 native=1 split=1"]
	if narrow.Blocks["KSPLIT"] != 4 {
		t.Errorf("the narrow class should take the K split of 4: %+v", narrow)
	}
	if !strings.Contains(wide.Why, "failed") || !strings.Contains(wide.Why, "engine's own blocks") {
		t.Errorf("why: %q", wide.Why)
	}
	if len(explain) != 2 || !strings.Contains(explain[0], "of bandwidth") {
		t.Errorf("explain: %v", explain)
	}
	if runner.jobs[0].Shapes > runner.jobs[len(runner.jobs)-1].Shapes {
		t.Errorf("jobs should come in class order")
	}
	bytesAStep, usAStep := StepFloor(classes, entries)
	if bytesAStep != 42_000_000*160 || usAStep <= 0 {
		t.Errorf("step floor: %d bytes, %.1f us", bytesAStep, usAStep)
	}
}

func TestTableBlocksRoundTripAndCheck(t *testing.T) {
	table := &Table{GPU: "RTX", Sites: []Entry{}, Blocks: []BlockEntry{
		{Shapes: "gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1", Blocks: map[string]int{"BN": 32, "BK": 512, "KSPLIT": 1, "warps": 8, "stages": 3}, Us: 30, Bytes: 42e6, FloorUs: 28.8, Efficiency: 0.96},
	}}
	data, err := table.Write()
	if err != nil {
		t.Fatal(err)
	}
	back, err := ReadTable(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Blocks) != 1 || back.Blocks[0].Blocks["BN"] != 32 {
		t.Fatalf("round trip: %+v", back.Blocks)
	}
	if _, err := ReadTable([]byte(`{"format":"vxpu-kernel-table/1","sites":[],"blocks":[{"shapes":"x","blocks":{"BZ":1}}]}`)); err == nil {
		t.Error("an unknown block name should be refused")
	}
	classes, _ := json.Marshal(map[string]ClassRun{
		"gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1": {Count: 1, Blocks: map[string]int{"BM": 16, "BN": 16, "BK": 256, "KSPLIT": 1, "warps": 4, "stages": 3}},
	})
	record := &Record{Format: RecordFormat, Stats: map[string]json.RawMessage{"gemv_classes": classes}}
	bad, err := CheckBlocks(back, record)
	if err != nil || len(bad) != 1 || !strings.Contains(bad[0].String(), "the table names") {
		t.Errorf("check: %v %v", bad, err)
	}
	applied, _ := json.Marshal(map[string]ClassRun{
		"gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1": {Count: 1, Blocks: map[string]int{"BM": 16, "BN": 32, "BK": 512, "KSPLIT": 1, "warps": 8, "stages": 3}, FromTable: true},
	})
	record.Stats["gemv_classes"] = applied
	if bad, _ := CheckBlocks(back, record); len(bad) != 0 {
		t.Errorf("an applied table should check clean: %v", bad)
	}
	results, err := ParseResults([]byte("[measure] noise\n{\"shapes\":\"a\",\"blocks\":{\"BN\":16},\"us\":1.5}\n"))
	if err != nil || len(results) != 1 || results[0].Us != 1.5 {
		t.Errorf("parse: %v %v", results, err)
	}
}

func TestTheCardsTableReadsAndItsBlocksAreMeasured(t *testing.T) {
	// the table hero 11 measured on an RTX PRO 6000 (vxpu kernels measure over the one-row and
	// nine-row records, on the ledger's site table): every block class carries its measurement
	table, err := ReadTableFile("testdata/table-rtx6000.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(table.Blocks) < 20 || len(table.Sites) == 0 {
		t.Fatalf("%d block classes, %d sites", len(table.Blocks), len(table.Sites))
	}
	for _, b := range table.Blocks {
		if b.Us <= 0 || b.Bytes <= 0 || b.Efficiency <= 0.2 || b.Efficiency > 1.1 {
			t.Errorf("%s: us %.1f bytes %d efficiency %.2f", b.Shapes, b.Us, b.Bytes, b.Efficiency)
		}
		if _, err := ParseClass(b.Shapes); err != nil {
			t.Error(err)
		}
	}
}

// The all-reduce table the measuring step wrote on four RTX PRO 6000 (hero run 14,
// python -m vxpu.allreduce measure): one entry per row count with the form that won (FORM 1
// one-shot, 2 reduce-scatter) and the crossover as the knob the engine reads.
func TestAllReduceTableReads(t *testing.T) {
	table, err := ReadTableFile("testdata/allreduce-rtx6000.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(table.Blocks) != 5 {
		t.Fatalf("blocks: got %d, want 5 row counts", len(table.Blocks))
	}
	forms := map[string]int{}
	for _, b := range table.Blocks {
		if b.Blocks["BLOCK"] < 1 || (b.Blocks["FORM"] != 1 && b.Blocks["FORM"] != 2) {
			t.Fatalf("%s: blocks %v", b.Shapes, b.Blocks)
		}
		forms[b.Shapes] = b.Blocks["FORM"]
	}
	if forms["allreduce rows=1 n=5120 k=4 w=bf16"] != 1 || forms["allreduce rows=132 n=5120 k=4 w=bf16"] != 2 {
		t.Fatalf("the one-shot form should win at one row and the reduce-scatter at 132: %v", forms)
	}
	if v, ok := table.Knobs["ALLREDUCE_SCATTER_MIN_ROWS"]; !ok || v == nil {
		t.Fatalf("the crossover knob is missing: %v", table.Knobs)
	}
}
