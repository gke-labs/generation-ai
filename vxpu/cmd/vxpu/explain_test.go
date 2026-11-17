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

import "testing"

func TestParseFleet(t *testing.T) {
	fleet, err := parseFleet("g2-standard-48:2, c3d-standard-60")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"g2-standard-48#1", "g2-standard-48#2", "c3d-standard-60#3"}
	if len(fleet) != len(want) {
		t.Fatalf("got %d machines, want %d", len(fleet), len(want))
	}
	for i, n := range fleet {
		if n.Name != want[i] {
			t.Errorf("machine %d is %q, want %q", i, n.Name, want[i])
		}
	}
	if fleet[0].Probe.GpuCount != 4 || fleet[2].Probe.Gpu != "" {
		t.Errorf("probes: %+v, %+v", fleet[0].Probe, fleet[2].Probe)
	}
	for _, bad := range []string{"no-such-machine", "g2-standard-48:0", "g2-standard-48:x"} {
		if _, err := parseFleet(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
}
