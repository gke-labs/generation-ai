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

package machines

import "testing"

// TestCatalog checks the embedded catalog's entries are complete and in
// the units the planner assumes (bytes, bytes/s, counts).
func TestCatalog(t *testing.T) {
	for _, name := range Names() {
		m, err := Lookup(name)
		if err != nil {
			t.Fatal(err)
		}
		bad := func(what string, v float64, min float64) {
			if v < min {
				t.Errorf("%s: %s = %g, want at least %g", name, what, v, min)
			}
		}
		if m.Name != name || m.CPU.Model == "" || m.CPU.Cores <= 0 || m.CPU.Threads < m.CPU.Cores {
			t.Errorf("%s: name or CPU incomplete: %+v", name, m)
		}
		if (m.GPU == nil) != (m.GPUCount == 0) {
			t.Errorf("%s: gpu and gpu_count disagree", name)
		}
		bad("ram_bytes", float64(m.RAMBytes), 1<<30)
		if m.CPU.MemoryBandwidthBytesPerSec != 0 {
			bad("cpu memory bandwidth", m.CPU.MemoryBandwidthBytesPerSec, 1e9)
		}
		if m.GPU != nil {
			bad("gpu memory_bytes", float64(m.GPU.MemoryBytes), 1<<30)
			bad("gpu memory bandwidth", m.GPU.MemoryBandwidthBytesPerSec, 1e9)
		}
		if m.Storage != nil {
			bad("storage read bandwidth", m.Storage.ReadBandwidthBytesPerSec, 1e6)
		}
		for k, v := range m.Measured {
			if f, ok := v.(float64); ok && f <= 0 {
				t.Errorf("%s: measured %s = %g, want positive", name, k, f)
			}
		}
	}
	for _, alias := range []string{"mac", "A2", "a3-highgpu"} {
		if _, err := Lookup(alias); err != nil {
			t.Errorf("alias %q: %v", alias, err)
		}
	}
	if _, err := Lookup("no-such-machine"); err == nil {
		t.Error("an unknown machine should not be found")
	}
}

func TestMatch(t *testing.T) {
	for _, c := range []struct {
		probe Probe
		want  string
	}{
		{Probe{CPU: "AMD EPYC 9B14 64-Core Processor", CPUThreads: 60}, "c3d-standard-60"},
		{Probe{CPU: "AMD EPYC 9B14 64-Core Processor", CPUThreads: 180}, "c3d-standard-180"},
		{Probe{CPU: "AMD EPYC 9B14 64-Core Processor", CPUThreads: 180, RAMBytes: 1400 << 30}, "c3d-highmem-180"},
		{Probe{CPU: "AMD EPYC 9B45 48-Core Processor", CPUThreads: 48}, "g4-standard-48"},
		{Probe{CPU: "AMD EPYC 9B45 48-Core Processor", CPUThreads: 96}, "g4-standard-96"},
		{Probe{CPU: "Apple M4 Pro", CPUThreads: 14}, "mac-m4-pro"},
		{Probe{CPU: "AMD EPYC 7B13 64-Core Processor", CPUThreads: 32}, ""}, // Milan: not in the catalog
		{Probe{CPU: "Apple M1", CPUThreads: 8}, ""},
		{Probe{CPU: "Intel(R) Xeon(R) Platinum 8281 CPU @ 2.70GHz", CPUThreads: 32}, ""},
		{Probe{GPU: "NVIDIA RTX PRO 6000", GPUCount: 2}, "g4-standard-96"},
		{Probe{GPU: "NVIDIA RTX PRO 6000", GPUCount: 1}, "g4-standard-48"},
		{Probe{GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8}, "a3-highgpu-8g"},
		{Probe{GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1}, ""},
		{Probe{GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 1}, "a2-highgpu-1g"},
		{Probe{GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 8}, ""},
		{Probe{GPU: "NVIDIA L4", GPUCount: 2, CPU: "AMD EPYC 9B45"}, "g2-standard-24"}, // a GPU decides, not the CPU
		{Probe{GPU: "NVIDIA L4", GPUCount: 4, GPUMemoryBytes: 23_659_151_360}, "g2-standard-48"},
		{Probe{GPU: "NVIDIA L4", GPUCount: 1}, ""},
		{Probe{GPU: "NVIDIA RTX A6000", GPUCount: 1}, ""},
		// the memory tells apart siblings the name matches
		{Probe{GPU: "NVIDIA RTX PRO 6000 Blackwell Server Edition", GPUCount: 2, GPUMemoryBytes: 102_005_473_280}, "g4-standard-96"},
		{Probe{GPU: "NVIDIA A100-SXM4-80GB", GPUCount: 1, GPUMemoryBytes: 85_100_000_000}, ""},
		{Probe{GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 1, GPUMemoryBytes: 42_500_000_000}, "a2-highgpu-1g"},
	} {
		got := ""
		if m := Match(c.probe); m != nil {
			got = m.Name
		}
		if got != c.want {
			t.Errorf("Match(%+v) = %q, want %q", c.probe, got, c.want)
		}
	}
}

func TestGPUMemoryBandwidthGBs(t *testing.T) {
	for _, c := range []struct {
		machine string
		want    float64
	}{
		{"g2-standard-24", 300},
		{"c3d-standard-60", 0},
	} {
		m, err := Lookup(c.machine)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", c.machine, err)
		}
		if got := m.GPUMemoryBandwidthGBs(); got != c.want {
			t.Errorf("%s: GPUMemoryBandwidthGBs() = %g, want %g", c.machine, got, c.want)
		}
	}
}

func TestLookupCard(t *testing.T) {
	for _, c := range []struct {
		card    string
		want    float64
		wantErr bool
	}{
		{"NVIDIA L4", 300, false},
		{"nvidia l4", 300, false},
		{"L4", 300, false},
		{"NVIDIA A100-SXM4-40GB", 1555, false},
		{"a100-40gb", 1555, false},
		{"a100-sxm4-40gb", 1555, false},
		{"Apple M4 Pro GPU", 273, false},
		{"m4 pro", 273, false},
		{"NVIDIA RTX PRO 6000", 1597, false},
		{"NVIDIA RTX PRO 6000 Blackwell Server Edition", 1597, false},
		{"rtx pro 6000 blackwell server edition", 1597, false},
		{"a100", 0, true},
		{"NVIDIA L40S", 0, true},
		{"NVIDIA L40", 0, true},
		{"nvidia", 0, true},
		{"", 0, true},
		{"   ", 0, true},
		{"nonexistent-gpu", 0, true},
	} {
		got, err := LookupCard(c.card)
		if (err != nil) != c.wantErr {
			t.Errorf("LookupCard(%q) err = %v, wantErr = %v", c.card, err, c.wantErr)
		}
		if got != c.want {
			t.Errorf("LookupCard(%q) = %g, want %g", c.card, got, c.want)
		}
	}
}
