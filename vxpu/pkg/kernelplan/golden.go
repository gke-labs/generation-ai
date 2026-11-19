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
	"fmt"
	"os"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/encoding/protojson"
)

// TokenMismatchError is returned by CompareTokens when a produced token sequence differs from golden.
type TokenMismatchError struct {
	// Prompt is the zero-based index of the first prompt whose tokens differed.
	Prompt int
	// Position is the zero-based token position of the first difference.
	Position int
	// Got is the token ID produced at the differing position.
	Got int32
	// Expected is the expected token ID from the golden reference.
	Expected int32
	// GotLength is the total number of tokens produced when sequence lengths differed.
	GotLength int
	// ExpectedLength is the expected number of tokens when sequence lengths differed.
	ExpectedLength int
	// PromptCountMismatch reports whether the count of produced prompts differed from golden.
	PromptCountMismatch bool
	// GotPrompts is the count of produced prompts when prompt counts differed.
	GotPrompts int
	// ExpectedPrompts is the expected count of prompts when prompt counts differed.
	ExpectedPrompts int
}

func (e *TokenMismatchError) Error() string {
	if e.PromptCountMismatch {
		return fmt.Sprintf("golden values: expected %d prompts, got %d", e.ExpectedPrompts, e.GotPrompts)
	}
	if e.GotLength != 0 || e.ExpectedLength != 0 {
		return fmt.Sprintf("golden values: prompt %d differs first at position %d: got %d tokens, expected %d", e.Prompt, e.Position, e.GotLength, e.ExpectedLength)
	}
	return fmt.Sprintf("golden values: prompt %d differs first at position %d: got %d, expected %d", e.Prompt, e.Position, e.Got, e.Expected)
}

// ParseGoldenJSON parses golden tokens JSON data using protojson.
// Unknown fields are refused.
func ParseGoldenJSON(data []byte) (*planpb.GoldenFile, error) {
	var gf planpb.GoldenFile
	if err := (protojson.UnmarshalOptions{DiscardUnknown: false}).Unmarshal(data, &gf); err != nil {
		return nil, fmt.Errorf("unmarshal golden json: %w", err)
	}
	if gf.Format != "vxpu-golden/1" {
		return nil, fmt.Errorf("unknown golden format %q, expected vxpu-golden/1", gf.Format)
	}
	return &gf, nil
}

// ReadGoldenFile reads and parses a golden tokens JSON file using protojson.
// Unknown fields are refused.
func ReadGoldenFile(path string) (*planpb.GoldenFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read golden file %s: %w", path, err)
	}
	gf, err := ParseGoldenJSON(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return gf, nil
}

// WriteGoldenJSON serializes a GoldenFile message to formatted JSON using protojson.
func WriteGoldenJSON(gf *planpb.GoldenFile) ([]byte, error) {
	return (protojson.MarshalOptions{
		Multiline:     true,
		Indent:        " ",
		UseProtoNames: true,
	}).Marshal(gf)
}

// WriteGoldenFile writes a GoldenFile message as formatted JSON using protojson.
func WriteGoldenFile(path string, gf *planpb.GoldenFile) error {
	data, err := WriteGoldenJSON(gf)
	if err != nil {
		return fmt.Errorf("marshal golden file %s: %w", path, err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write golden file %s: %w", path, err)
	}
	return nil
}

// CompareTokens compares produced token sequences against the expected sequences in golden.
// If any prompt differs, it returns a TokenMismatchError naming the prompt and first differing position.
func CompareTokens(golden *planpb.GoldenFile, producedTokens [][]int32) error {
	if len(producedTokens) != len(golden.Prompts) {
		return &TokenMismatchError{
			PromptCountMismatch: true,
			GotPrompts:          len(producedTokens),
			ExpectedPrompts:     len(golden.Prompts),
		}
	}
	for i, prompt := range golden.Prompts {
		got := producedTokens[i]
		want := prompt.Tokens
		minLen := len(got)
		if len(want) < minLen {
			minLen = len(want)
		}
		for pos := 0; pos < minLen; pos++ {
			if got[pos] != want[pos] {
				return &TokenMismatchError{
					Prompt:   i,
					Position: pos,
					Got:      got[pos],
					Expected: want[pos],
				}
			}
		}
		if len(got) != len(want) {
			return &TokenMismatchError{
				Prompt:         i,
				Position:       minLen,
				GotLength:      len(got),
				ExpectedLength: len(want),
			}
		}
	}
	return nil
}
