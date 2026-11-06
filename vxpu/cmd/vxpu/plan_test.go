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
	"testing"
)

func TestCmdPlanCostVerification(t *testing.T) {
	ctx := t.Context()

	// Test ledger verification command across all shapes.
	ledgerPath := "../../pkg/planner/testdata/hero_ledger.json"
	if err := cmdPlan(ctx, []string{"cost", "--ledger", ledgerPath}); err != nil {
		t.Fatalf("cmdPlan cost --ledger: %v", err)
	}

	// Test verify alias with filter by tree and shape.
	if err := cmdPlan(ctx, []string{"verify", "--ledger", ledgerPath, "--tree", "src86", "--shape", "five-row"}); err != nil {
		t.Fatalf("cmdPlan verify --tree src86 --shape five-row: %v", err)
	}

	// Test single fixture trace.
	tracePath := "../../pkg/planner/testdata/trace_side_branch_fused.json"
	if err := cmdPlan(ctx, []string{"cost", "--record", tracePath}); err != nil {
		t.Fatalf("cmdPlan cost --record: %v", err)
	}

	// Test with --json output flag.
	if err := cmdPlan(ctx, []string{"cost", "--ledger", ledgerPath, "--json"}); err != nil {
		t.Fatalf("cmdPlan cost --ledger --json: %v", err)
	}
}
