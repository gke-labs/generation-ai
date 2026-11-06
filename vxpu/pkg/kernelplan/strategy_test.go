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
	"path/filepath"
	"strings"
	"testing"
)

func TestStrategySelection_GPUFreeAcceptance(t *testing.T) {
	fixturePath := filepath.Join("testdata", "strategy-hero15.json")
	table, err := LoadStrategyTable(fixturePath)
	if err != nil {
		t.Fatalf("LoadStrategyTable(%q): %v", fixturePath, err)
	}

	tests := []struct {
		name                string
		shape               Shape
		allowApproximations bool
		acceptanceBudget    float64
		wantStrategy        string
		wantDefault         bool
		wantReasonSubstring string
		wantRowsCount       int
	}{
		{
			name:                "8 streams with approximation allowed picks bf16 wire",
			shape:               Shape{Rows: 8, Streams: 8},
			allowApproximations: true,
			acceptanceBudget:    0.05,
			wantStrategy:        StrategySumWireBF16,
			wantDefault:         false,
			wantReasonSubstring: "chose bf16",
			wantRowsCount:       2,
		},
		{
			name:                "1 stream with approximation allowed picks fp32 wire",
			shape:               Shape{Rows: 1, Streams: 1},
			allowApproximations: true,
			acceptanceBudget:    0.05,
			wantStrategy:        StrategySumWireFP32,
			wantDefault:         true,
			wantReasonSubstring: "chose fp32",
			wantRowsCount:       2,
		},
		{
			name:                "8 streams exact only (approximations not allowed) picks fp32 wire",
			shape:               Shape{Rows: 8, Streams: 8},
			allowApproximations: false,
			acceptanceBudget:    0.0,
			wantStrategy:        StrategySumWireFP32,
			wantDefault:         true,
			wantReasonSubstring: "chose fp32",
			wantRowsCount:       2,
		},
		{
			name:                "1 stream exact only (approximations not allowed) picks fp32 wire",
			shape:               Shape{Rows: 1, Streams: 1},
			allowApproximations: false,
			acceptanceBudget:    0.0,
			wantStrategy:        StrategySumWireFP32,
			wantDefault:         true,
			wantReasonSubstring: "chose fp32",
			wantRowsCount:       2,
		},
		{
			name:                "unmeasured shape (4 streams) falls back to default fp32 with note",
			shape:               Shape{Rows: 4, Streams: 4},
			allowApproximations: true,
			acceptanceBudget:    0.05,
			wantStrategy:        StrategySumWireFP32,
			wantDefault:         true,
			wantReasonSubstring: "no measured rows for shape (rows=4, streams=4); falling back to default fp32",
			wantRowsCount:       0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := StrategyRequest{
				Shape:               tc.shape,
				AllowApproximations: tc.allowApproximations,
				AcceptanceBudget:    tc.acceptanceBudget,
			}
			choice := ChooseStrategy(table, req, SiteClassSumWire)

			if choice.Strategy != tc.wantStrategy {
				t.Errorf("Strategy = %q, want %q", choice.Strategy, tc.wantStrategy)
			}
			if choice.Default != tc.wantDefault {
				t.Errorf("Default = %v, want %v", choice.Default, tc.wantDefault)
			}
			if !strings.Contains(choice.Reason, tc.wantReasonSubstring) {
				t.Errorf("Reason = %q, want containing %q", choice.Reason, tc.wantReasonSubstring)
			}
			if len(choice.Rows) != tc.wantRowsCount {
				t.Errorf("Rows count = %d, want %d", len(choice.Rows), tc.wantRowsCount)
			}
		})
	}
}

func TestStrategySelection_ApplyStrategiesToPlan(t *testing.T) {
	fixturePath := filepath.Join("testdata", "strategy-hero15.json")
	table, err := LoadStrategyTable(fixturePath)
	if err != nil {
		t.Fatalf("LoadStrategyTable(%q): %v", fixturePath, err)
	}

	plan := &PipelinePlan{
		Format: PlanFormat,
		Ranges: []Range{{GPU: 0, Layers: "0:43"}},
		Pipeline: Pipeline{
			Batches:  1,
			Schedule: ScheduleRangeOrder,
		},
	}

	req := StrategyRequest{
		Shape:               Shape{Rows: 8, Streams: 8},
		AllowApproximations: true,
		AcceptanceBudget:    0.05,
	}

	ApplyStrategies(plan, table, req, SiteClassSumWire)

	if got := plan.Strategies[SiteClassSumWire]; got != StrategySumWireBF16 {
		t.Errorf("plan.Strategies[%q] = %q, want %q", SiteClassSumWire, got, StrategySumWireBF16)
	}
	if len(plan.StrategyChoices) != 1 {
		t.Fatalf("len(plan.StrategyChoices) = %d, want 1", len(plan.StrategyChoices))
	}
	choice := plan.StrategyChoices[0]
	if choice.Strategy != StrategySumWireBF16 {
		t.Errorf("choice.Strategy = %q, want %q", choice.Strategy, StrategySumWireBF16)
	}
	if len(choice.Rows) != 2 {
		t.Errorf("len(choice.Rows) = %d, want 2 (fp32 and bf16 rows it rested on)", len(choice.Rows))
	}

	foundExplain := false
	for _, exp := range plan.Explain {
		if strings.Contains(exp, "strategy sum_wire: bf16") {
			foundExplain = true
			break
		}
	}
	if !foundExplain {
		t.Errorf("plan.Explain did not contain strategy explanation: %v", plan.Explain)
	}
}
