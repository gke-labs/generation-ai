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
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestFixtureSideBranchNotCountingAndFusedCountsOnce(t *testing.T) {
	data, err := os.ReadFile("testdata/trace_side_branch_fused.json")
	if err != nil {
		t.Fatalf("read fixture trace: %v", err)
	}

	var trace struct {
		Format  string      `json:"format"`
		Shape   string      `json:"shape"`
		Rows    int         `json:"rows"`
		Streams int         `json:"streams"`
		Nodes   []PhaseNode `json:"nodes"`
	}
	if err := json.Unmarshal(data, &trace); err != nil {
		t.Fatalf("unmarshal fixture trace: %v", err)
	}

	model := NewCriticalPathCostModel()
	pred, err := model.Predict(trace.Nodes, Shape{Name: trace.Shape, Rows: trace.Rows, Streams: trace.Streams})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	// 1. Verify side branch node does not count towards the wall.
	var sideNode *CriticalPathNodeSummary
	for i := range pred.CriticalNodes {
		if pred.CriticalNodes[i].Name == "model.layers.0.compressor_island" {
			sideNode = &pred.CriticalNodes[i]
			break
		}
	}
	if sideNode == nil {
		t.Fatalf("side branch node model.layers.0.compressor_island missing from critical nodes")
	}
	if sideNode.WallCostMs != 0.0 {
		t.Errorf("expected side branch node wall cost 0.0 ms, got %.2f ms", sideNode.WallCostMs)
	}
	if math.Abs(pred.SideBranchCost-0.0) > 1e-6 {
		t.Errorf("expected total side branch wall cost 0.0 ms, got %.2f ms", pred.SideBranchCost)
	}
	if math.Abs(pred.HiddenWorkMs-0.7) > 1e-6 {
		t.Errorf("expected hidden work 0.7 ms, got %.2f ms", pred.HiddenWorkMs)
	}

	// 2. Verify fused region counts once in place and constituent parts are skipped.
	var parentCount, partSkippedCount int
	for _, n := range pred.CriticalNodes {
		if n.FusedRegion == "fused_push_0" {
			if n.WallCostMs > 0 {
				parentCount++
				if math.Abs(n.WallCostMs-0.5) > 1e-6 {
					t.Errorf("expected in-place fused cost 0.5 ms, got %.2f ms", n.WallCostMs)
				}
			}
			if n.FusedPartSkipped {
				partSkippedCount++
			}
		}
	}
	if parentCount != 1 {
		t.Errorf("expected fused region to count exactly once, got %d times", parentCount)
	}
	if partSkippedCount != 3 {
		t.Errorf("expected 3 fused parts skipped, got %d", partSkippedCount)
	}
	if math.Abs(pred.FusedSavingsMs-0.20) > 1e-6 {
		t.Errorf("expected fused savings 0.20 ms, got %.2f ms", pred.FusedSavingsMs)
	}

	// 3. Verify total predicted step time.
	// Expected: 2.0 (dense_input) + 2.0 (attn_q) + 5.0 (moe_experts) + 2.0 (collective_sum) + 0.5 (fused_push) = 11.5 ms.
	expectedTotal := 11.5
	if math.Abs(pred.PredictedStepMs-expectedTotal) > 1e-6 {
		t.Errorf("expected predicted step time %.2f ms, got %.2f ms", expectedTotal, pred.PredictedStepMs)
	}

	// Collective wait check: wait_ms was 1.2 ms on collective_sum.
	if math.Abs(pred.CollectiveWait-1.2) > 1e-6 {
		t.Errorf("expected collective wait 1.2 ms, got %.2f ms", pred.CollectiveWait)
	}
}

func TestHeroPredictionsWithinBand(t *testing.T) {
	records, err := ReadLedger("testdata/hero_ledger.json")
	if err != nil {
		t.Fatalf("read hero ledger: %v", err)
	}
	if len(records) == 0 {
		t.Fatalf("no records in hero ledger")
	}

	results, err := VerifyLedger(records, "", "", 0)
	if err != nil {
		t.Fatalf("verify ledger: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 verification results (five-row, block-7, 8-stream), got %d", len(results))
	}

	expected := map[string]struct {
		rows        int
		streams     int
		predictedMs float64
		actualMedMs float64
		maxBandPct  float64
	}{
		"five-row": {rows: 5, streams: 1, predictedMs: 11.41, actualMedMs: 11.41, maxBandPct: 1.0},
		"block-7":  {rows: 8, streams: 1, predictedMs: 14.14, actualMedMs: 14.14, maxBandPct: 1.0},
		"8-stream": {rows: 40, streams: 8, predictedMs: 25.70, actualMedMs: 25.70, maxBandPct: 2.0},
	}

	for _, r := range results {
		exp, ok := expected[r.Shape]
		if !ok {
			t.Errorf("unexpected shape result %q", r.Shape)
			continue
		}
		if math.Abs(r.PredictedMs-exp.predictedMs) > 0.05 {
			t.Errorf("[%s] expected predicted %.2f ms, got %.2f ms", r.Shape, exp.predictedMs, r.PredictedMs)
		}
		if math.Abs(r.ActualMedianMs-exp.actualMedMs) > 0.05 {
			t.Errorf("[%s] expected actual median %.2f ms, got %.2f ms", r.Shape, exp.actualMedMs, r.ActualMedianMs)
		}
		if r.MedianErrorPct > exp.maxBandPct {
			t.Errorf("[%s] median error %.2f%% exceeds band %.2f%%", r.Shape, r.MedianErrorPct, exp.maxBandPct)
		}
		if !r.Accepted {
			t.Errorf("[%s] expected model to be accepted", r.Shape)
		}
	}
}

func TestStrictRowMismatchRefused(t *testing.T) {
	nodes := []PhaseNode{
		{Name: "dense_node", Kind: "dense", DurationMs: 1.0, Rows: 1},
	}
	model := NewCriticalPathCostModel()
	_, err := model.Predict(nodes, Shape{Name: "five-row", Rows: 5, Streams: 1})
	if err == nil {
		t.Fatalf("expected error when predicting 5-row shape using 1-row node, got nil")
	}
}

func TestFormatVerificationReport(t *testing.T) {
	records, err := ReadLedger("testdata/hero_ledger.json")
	if err != nil {
		t.Fatalf("read hero ledger: %v", err)
	}
	results, err := VerifyLedger(records, "src86", "five-row", 1.0)
	if err != nil {
		t.Fatalf("verify ledger: %v", err)
	}

	var buf bytes.Buffer
	if err := FormatVerificationReport(&buf, results); err != nil {
		t.Fatalf("format report: %v", err)
	}
	out := buf.String()
	if !bytes.Contains([]byte(out), []byte("Cost Model v0 Verification against Ledger:")) {
		t.Errorf("report output missing header: %s", out)
	}
	if !bytes.Contains([]byte(out), []byte("ACCEPTED")) {
		t.Errorf("report output missing ACCEPTED status: %s", out)
	}
}
