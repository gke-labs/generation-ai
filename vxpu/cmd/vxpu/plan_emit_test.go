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

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func TestPlanEmit_GPUFreeAcceptance(t *testing.T) {
	fixtureTable := filepath.Join("..", "..", "pkg", "kernelplan", "testdata", "strategy-hero15.json")
	if _, err := os.Stat(fixtureTable); err != nil {
		t.Fatalf("fixture table not found at %q: %v", fixtureTable, err)
	}

	tmpDir := t.TempDir()

	// 1. At 8 streams with approximations allowed -> picks bf16
	out8 := filepath.Join(tmpDir, "plan-8s.json")
	err := cmdPlan(t.Context(), []string{
		"emit",
		"--table", fixtureTable,
		"--streams", "8",
		"--allow-class", "approximation",
		"--acceptance-budget", "0.05",
		"--out", out8,
	})
	if err != nil {
		t.Fatalf("cmdPlan emit 8 streams error: %v", err)
	}

	data8, err := os.ReadFile(out8)
	if err != nil {
		t.Fatalf("read %s: %v", out8, err)
	}
	var plan8 kernelplan.PipelinePlan
	if err := json.Unmarshal(data8, &plan8); err != nil {
		t.Fatalf("unmarshal plan8: %v", err)
	}
	if got := plan8.Strategies[kernelplan.SiteClassSumWire]; got != "bf16" {
		t.Errorf("plan8 strategy = %q, want bf16", got)
	}

	// 2. At 1 stream with approximations allowed -> picks fp32
	out1 := filepath.Join(tmpDir, "plan-1s.json")
	err = cmdPlan(t.Context(), []string{
		"emit",
		"--table", fixtureTable,
		"--streams", "1",
		"--allow-class", "approximation",
		"--acceptance-budget", "0.05",
		"--out", out1,
	})
	if err != nil {
		t.Fatalf("cmdPlan emit 1 stream error: %v", err)
	}
	data1, err := os.ReadFile(out1)
	if err != nil {
		t.Fatalf("read %s: %v", out1, err)
	}
	var plan1 kernelplan.PipelinePlan
	if err := json.Unmarshal(data1, &plan1); err != nil {
		t.Fatalf("unmarshal plan1: %v", err)
	}
	if got := plan1.Strategies[kernelplan.SiteClassSumWire]; got != "fp32" {
		t.Errorf("plan1 strategy = %q, want fp32", got)
	}

	// 3. At 8 streams exact only -> picks fp32
	out8Exact := filepath.Join(tmpDir, "plan-8s-exact.json")
	err = cmdPlan(t.Context(), []string{
		"emit",
		"--table", fixtureTable,
		"--streams", "8",
		"--allow-class", "exact",
		"--out", out8Exact,
	})
	if err != nil {
		t.Fatalf("cmdPlan emit 8 streams exact error: %v", err)
	}
	data8Exact, err := os.ReadFile(out8Exact)
	if err != nil {
		t.Fatalf("read %s: %v", out8Exact, err)
	}
	var plan8Exact kernelplan.PipelinePlan
	if err := json.Unmarshal(data8Exact, &plan8Exact); err != nil {
		t.Fatalf("unmarshal plan8Exact: %v", err)
	}
	if got := plan8Exact.Strategies[kernelplan.SiteClassSumWire]; got != "fp32" {
		t.Errorf("plan8Exact strategy = %q, want fp32", got)
	}

	// 4. At unmeasured shape (4 streams) -> falls back to default fp32 with note
	out4 := filepath.Join(tmpDir, "plan-4s.json")
	err = cmdPlan(t.Context(), []string{
		"emit",
		"--table", fixtureTable,
		"--streams", "4",
		"--out", out4,
	})
	if err != nil {
		t.Fatalf("cmdPlan emit 4 streams error: %v", err)
	}
	data4, err := os.ReadFile(out4)
	if err != nil {
		t.Fatalf("read %s: %v", out4, err)
	}
	var plan4 kernelplan.PipelinePlan
	if err := json.Unmarshal(data4, &plan4); err != nil {
		t.Fatalf("unmarshal plan4: %v", err)
	}
	if got := plan4.Strategies[kernelplan.SiteClassSumWire]; got != "fp32" {
		t.Errorf("plan4 strategy = %q, want fp32", got)
	}
	if len(plan4.StrategyChoices) == 0 || !plan4.StrategyChoices[0].Default {
		t.Errorf("plan4 StrategyChoices = %+v, want default=true", plan4.StrategyChoices)
	}
}
