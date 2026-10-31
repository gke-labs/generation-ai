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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTraceSummarizeFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "trace-fixture.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}

	var buf bytes.Buffer
	if err := summarizeTrace(data, &buf, 5); err != nil {
		t.Fatalf("summarizeTrace failed: %v", err)
	}
	output := buf.String()

	// 1. Should report 5 steps.
	if !strings.Contains(output, "5 steps:") {
		t.Errorf("output does not report 5 steps:\n%s", output)
	}

	// 2. Table columns.
	if !strings.Contains(output, "PHASE") || !strings.Contains(output, "MEDIAN") || !strings.Contains(output, "MAX") {
		t.Errorf("output missing table headers:\n%s", output)
	}

	// 3. Wall median should show ~13.61 ms.
	if !strings.Contains(output, "13.61") && !strings.Contains(output, "13.60") {
		t.Errorf("output does not show ~13.6 ms wall median:\n%s", output)
	}

	// 4. Host nodes should show 0.02 ms medians.
	if !strings.Contains(output, "host node 0") || !strings.Contains(output, "host node 1") {
		t.Errorf("output missing host nodes:\n%s", output)
	}
	if !strings.Contains(output, "0.02") {
		t.Errorf("output does not show 0.02 ms host node time:\n%s", output)
	}

	// 5. The summary must name the host node holding the slow step's time.
	// Host node 1 holds 46.22 ms in step 2.
	if !strings.Contains(output, "host node 1") || !strings.Contains(output, "46.22") {
		t.Errorf("output does not name host node 1 holding slow step time (46.22 ms):\n%s", output)
	}

	// 6. Slowest steps list must contain step 2.
	if !strings.Contains(output, "step 2: wall 59.85 ms") {
		t.Errorf("output does not list step 2 in slowest steps:\n%s", output)
	}
}

func TestTraceSummarizeSpeculative(t *testing.T) {
	traceJSON := `{"step": 0, "decoding": 1, "ms": 25.80, "launch0_ms": 0.01, "launched_ms": 0.12, "synced_ms": [25.60], "draft_launch0_ms": 0.005, "draft_launched_ms": 0.025, "draft_checked_ms": 25.65, "draft_host": [{"start_ms": 0.50, "end_ms": 0.52}], "host": [{"start_ms": 1.0, "end_ms": 1.02}]}
{"step": 1, "decoding": 1, "ms": 26.00, "launch0_ms": 0.01, "launched_ms": 0.12, "synced_ms": [25.80], "draft_launch0_ms": 0.005, "draft_launched_ms": 0.025, "draft_checked_ms": 25.85, "draft_host": [{"start_ms": 0.50, "end_ms": 0.53}], "host": [{"start_ms": 1.0, "end_ms": 1.02}]}
`
	var buf bytes.Buffer
	if err := summarizeTrace([]byte(traceJSON), &buf, 5); err != nil {
		t.Fatalf("summarizeTrace failed: %v", err)
	}
	output := buf.String()

	if !strings.Contains(output, "draft launch") {
		t.Errorf("output missing draft launch phase:\n%s", output)
	}
	if !strings.Contains(output, "draft host node 0") {
		t.Errorf("output missing draft host node 0 phase:\n%s", output)
	}
	if !strings.Contains(output, "draft check") {
		t.Errorf("output missing draft check phase:\n%s", output)
	}
}

func TestTraceSummarizeMultiGPU(t *testing.T) {
	traceJSON := `{"step": 0, "decoding": 1, "ms": 13.60, "launch0_ms": 0.01, "launched_ms": 0.11, "synced_ms": [3.45, 6.95, 10.25, 13.50], "host": [{"start_ms": 1.0, "end_ms": 1.02}, {"start_ms": 4.5, "end_ms": 4.52}]}
{"step": 1, "decoding": 1, "ms": 13.62, "launch0_ms": 0.01, "launched_ms": 0.11, "synced_ms": [3.46, 6.96, 10.26, 13.51], "host": [{"start_ms": 1.0, "end_ms": 1.02}, {"start_ms": 4.5, "end_ms": 4.52}]}
`
	var buf bytes.Buffer
	if err := summarizeTrace([]byte(traceJSON), &buf, 5); err != nil {
		t.Fatalf("summarizeTrace failed: %v", err)
	}
	output := buf.String()

	for g := 0; g < 4; g++ {
		gpuName := fmt.Sprintf("gpu %d", g)
		if !strings.Contains(output, gpuName) {
			t.Errorf("output missing %s phase:\n%s", gpuName, output)
		}
	}
}

func TestTraceSummarizeRejectsDroppedBracket(t *testing.T) {
	path := filepath.Join("testdata", "trace-dropped-bracket.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading dropped-bracket file: %v", err)
	}

	var buf bytes.Buffer
	err = summarizeTrace(data, &buf, 5)
	if err == nil {
		t.Fatal("expected error on dropped bracket trace, got nil")
	}
	if !strings.Contains(err.Error(), "line 1:") {
		t.Errorf("error does not mention line 1: %v", err)
	}
}

func TestTraceSummarizeLineNumber(t *testing.T) {
	traceJSON := `{"step": 0, "ms": 13.5}
{"step": 1, "ms": 13.6}
this is not valid json
`
	var buf bytes.Buffer
	err := summarizeTrace([]byte(traceJSON), &buf, 5)
	if err == nil {
		t.Fatal("expected error on malformed line, got nil")
	}
	if !strings.Contains(err.Error(), "line 3:") {
		t.Errorf("expected error on line 3, got: %v", err)
	}
}

func TestTraceSummarizeTopZero(t *testing.T) {
	fixturePath := filepath.Join("testdata", "trace-fixture.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}

	var buf bytes.Buffer
	if err := summarizeTrace(data, &buf, 0); err != nil {
		t.Fatalf("summarizeTrace failed: %v", err)
	}
	output := buf.String()

	if strings.Contains(output, "slowest steps:") {
		t.Errorf("output should not contain slowest steps when top=0:\n%s", output)
	}
}

func TestCmdTraceCLI(t *testing.T) {
	ctx := t.Context()

	// Missing args.
	if err := cmdTrace(ctx, nil); err == nil {
		t.Error("expected error for empty args, got nil")
	}

	// Unknown subcommand.
	if err := cmdTrace(ctx, []string{"unknown"}); err == nil {
		t.Error("expected error for unknown subcommand, got nil")
	}

	// Valid trace summarize run.
	fixturePath := filepath.Join("testdata", "trace-fixture.jsonl")
	if err := cmdTrace(ctx, []string{"summarize", fixturePath}); err != nil {
		t.Errorf("cmdTrace summarize failed: %v", err)
	}

	if err := cmdTrace(ctx, []string{"summarize", "-top", "2", fixturePath}); err != nil {
		t.Errorf("cmdTrace summarize with -top failed: %v", err)
	}
}
