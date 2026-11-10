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

package capacity

import (
	"testing"
)

func TestDeterministicName(t *testing.T) {
	tests := []struct {
		base string
		zone string
		want string
	}{
		{"hero-race", "us-central1-b", "hero-race-us-central1-b"},
		{"hero-race-", "us-central1-b", "hero-race-us-central1-b"},
		{"", "us-central1-b", "vxpu-race-us-central1-b"},
	}
	for _, tc := range tests {
		got := DeterministicName(tc.base, tc.zone)
		if got != tc.want {
			t.Errorf("DeterministicName(%q, %q) = %q, want %q", tc.base, tc.zone, got, tc.want)
		}
	}
}
