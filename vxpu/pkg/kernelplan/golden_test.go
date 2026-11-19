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

func TestReadGoldenFile(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-deepseek-v41.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	if golden.Format != "vxpu-golden/1" {
		t.Errorf("Format = %q, want vxpu-golden/1", golden.Format)
	}
	if len(golden.Prompts) != 2 {
		t.Fatalf("got %d prompts, want 2", len(golden.Prompts))
	}
	if golden.Digest == "" {
		t.Error("expected non-empty digest")
	}
}

func TestReadGoldenFile_TinyRandomLlama(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-random-llama.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	if golden.Format != "vxpu-golden/1" {
		t.Errorf("Format = %q, want vxpu-golden/1", golden.Format)
	}
	if len(golden.Prompts) != 1 {
		t.Fatalf("got %d prompts, want 1", len(golden.Prompts))
	}
	if golden.Digest == "" {
		t.Error("expected non-empty digest")
	}
}

func TestParseGoldenJSON_InvalidFormat(t *testing.T) {
	_, err := ParseGoldenJSON([]byte(`{"format": "unknown/1"}`))
	if err == nil {
		t.Fatal("expected error for unknown format, got nil")
	}
}

func TestCompareTokens_Matches(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-deepseek-v41.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	produced := [][]int32{
		append([]int32(nil), golden.Prompts[0].Tokens...),
		append([]int32(nil), golden.Prompts[1].Tokens...),
	}

	if err := CompareTokens(golden, produced); err != nil {
		t.Fatalf("CompareTokens failed: %v", err)
	}
}

func TestCompareTokens_OneProducedTokenChangedFails(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-deepseek-v41.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	produced := [][]int32{
		append([]int32(nil), golden.Prompts[0].Tokens...),
		append([]int32(nil), golden.Prompts[1].Tokens...),
	}
	produced[0][3] = 99999

	err = CompareTokens(golden, produced)
	if err == nil {
		t.Fatal("expected CompareTokens to fail on changed token, got nil")
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 3") {
		t.Errorf("expected error to name prompt 0 and position 3, got: %v", err)
	}
}

func TestCompareTokens_LengthMismatchFails(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-deepseek-v41.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	produced := [][]int32{
		append([]int32(nil), golden.Prompts[0].Tokens[:10]...),
		append([]int32(nil), golden.Prompts[1].Tokens...),
	}

	err = CompareTokens(golden, produced)
	if err == nil {
		t.Fatal("expected CompareTokens to fail on truncated tokens, got nil")
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 10") {
		t.Errorf("expected error to name prompt 0 and position 10, got: %v", err)
	}
}

func TestCompareTokens_PromptCountMismatchFails(t *testing.T) {
	goldenPath := filepath.Join("fixtures", "golden", "tiny-deepseek-v41.json")
	golden, err := ReadGoldenFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	produced := [][]int32{
		append([]int32(nil), golden.Prompts[0].Tokens...),
	}

	err = CompareTokens(golden, produced)
	if err == nil {
		t.Fatal("expected CompareTokens to fail on missing prompt, got nil")
	}
	if !strings.Contains(err.Error(), "expected 2 prompts, got 1") {
		t.Errorf("expected error about prompt count, got: %v", err)
	}
}
