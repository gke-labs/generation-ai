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
	"path/filepath"
	"testing"
)

func TestPlanOverlap(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "pkg", "kernelplan", "testdata", "block7-hero15-run.json")
	ctx := t.Context()

	// Test text table output
	if err := cmdPlan(ctx, []string{"overlap", "--in", fixturePath}); err != nil {
		t.Fatalf("cmdPlan(overlap) failed: %v", err)
	}

	// Test JSON output
	if err := cmdPlan(ctx, []string{"overlap", "--in", fixturePath, "--json"}); err != nil {
		t.Fatalf("cmdPlan(overlap, --json) failed: %v", err)
	}
}

func TestPlanOverlapMissingIn(t *testing.T) {
	ctx := t.Context()
	if err := cmdPlan(ctx, []string{"overlap"}); err == nil {
		t.Fatal("expected error when --in is missing, got nil")
	}
}
