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

import (
	"strings"
	"testing"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

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

		// Every catalog machine must match its own ProbeOf (Item 3).
		probe := ProbeOf(m)
		if got := Match(probe); got == nil || got.Name != m.Name {
			t.Errorf("%s: Match(ProbeOf(m)) = %v, want %s", name, got, m.Name)
		}
		if diffs := Compare(m, probe); len(diffs) > 0 {
			t.Errorf("%s: Compare(m, ProbeOf(m)) reported differences: %v", name, diffs)
		}
		if got := MatchExact(probe); got == nil || got.Name != m.Name {
			t.Errorf("%s: MatchExact(ProbeOf(m)) = %v, want %s", name, got, m.Name)
		}
		if _, err := Check(name, probe); err != nil {
			t.Errorf("%s: Check(name, ProbeOf(m)) failed: %v", name, err)
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
		probe *pb.Probe
		want  string
	}{
		{&pb.Probe{Cpu: "AMD EPYC 9B14 64-Core Processor", CpuThreads: 60}, "c3d-standard-60"},
		{&pb.Probe{Cpu: "AMD EPYC 9B14 64-Core Processor", CpuThreads: 180}, "c3d-standard-180"},
		{&pb.Probe{Cpu: "AMD EPYC 9B14 64-Core Processor", CpuThreads: 180, RamBytes: 1400 << 30}, "c3d-highmem-180"},
		{&pb.Probe{Cpu: "AMD EPYC 9B45 48-Core Processor", CpuThreads: 48}, "g4-standard-48"},
		{&pb.Probe{Cpu: "AMD EPYC 9B45 48-Core Processor", CpuThreads: 96}, "g4-standard-96"},
		{&pb.Probe{Cpu: "Apple M4 Pro", CpuThreads: 14}, "mac-m4-pro"},
		{&pb.Probe{Cpu: "AMD EPYC 7B13 64-Core Processor", CpuThreads: 32}, ""}, // Milan: not in the catalog
		{&pb.Probe{Cpu: "Apple M1", CpuThreads: 8}, ""},
		{&pb.Probe{Cpu: "Intel(R) Xeon(R) Platinum 8281 CPU @ 2.70GHz", CpuThreads: 32}, ""},
		{&pb.Probe{Gpu: "NVIDIA RTX PRO 6000", GpuCount: 2}, "g4-standard-96"},
		{&pb.Probe{Gpu: "NVIDIA RTX PRO 6000", GpuCount: 1}, "g4-standard-48"},
		{&pb.Probe{Gpu: "NVIDIA H100 80GB HBM3", GpuCount: 8}, "a3-highgpu-8g"},
		{&pb.Probe{Gpu: "NVIDIA H100 80GB HBM3", GpuCount: 1}, ""},
		{&pb.Probe{Gpu: "NVIDIA A100-SXM4-40GB", GpuCount: 1}, "a2-highgpu-1g"},
		{&pb.Probe{Gpu: "NVIDIA A100-SXM4-40GB", GpuCount: 8}, ""},
		{&pb.Probe{Gpu: "NVIDIA L4", GpuCount: 2, Cpu: "AMD EPYC 9B45"}, "g2-standard-24"}, // a GPU decides, not the CPU
		{&pb.Probe{Gpu: "NVIDIA L4", GpuCount: 4, GpuMemoryBytes: 23_659_151_360}, "g2-standard-48"},
		{&pb.Probe{Gpu: "NVIDIA L4", GpuCount: 1}, ""},
		{&pb.Probe{Gpu: "NVIDIA RTX A6000", GpuCount: 1}, ""},
		// the memory tells apart siblings the name matches
		{&pb.Probe{Gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition", GpuCount: 2, GpuMemoryBytes: 102_005_473_280}, "g4-standard-96"},
		{&pb.Probe{Gpu: "NVIDIA A100-SXM4-80GB", GpuCount: 1, GpuMemoryBytes: 85_100_000_000}, ""},
		{&pb.Probe{Gpu: "NVIDIA A100-SXM4-40GB", GpuCount: 1, GpuMemoryBytes: 42_500_000_000}, "a2-highgpu-1g"},
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

// TestCheck tests the coordinator matching an agent's reported probe facts
// against the machine catalog, refusing one that differs from what was planned
// with each difference named using the reported values, and reporting unknown
// machines with facts instead of matching to the nearest entry (#663).
func TestCheck(t *testing.T) {
	// 1. Matching a known machine in the catalog.
	g4Probe := &pb.Probe{
		Cpu:            "AMD EPYC 9B45",
		CpuThreads:     48,
		RamBytes:       193273528320,
		Gpu:            "NVIDIA RTX PRO 6000",
		GpuCount:       1,
		StorageDevices: 4,
	}

	m := MatchExact(g4Probe)
	if m == nil || m.Name != "g4-standard-48" {
		t.Fatalf("MatchExact(g4Probe) = %v, want g4-standard-48", m)
	}

	// 2. Successful check when machine matches the plan.
	checked, err := Check("g4-standard-48", g4Probe)
	if err != nil {
		t.Fatalf("Check(g4-standard-48, g4Probe): %v", err)
	}
	if checked.Name != "g4-standard-48" {
		t.Errorf("checked machine: got %q, want g4-standard-48", checked.Name)
	}

	// 3. Coordinator refuses a machine that differs from what was planned, naming each difference.
	_, err = Check("g4-standard-96", g4Probe)
	if err == nil {
		t.Fatal("Check(g4-standard-96, g4Probe): expected error refusing differing machine, got nil")
	}
	errStr := err.Error()
	for _, want := range []string{
		"planned for g4-standard-96",
		"got g4-standard-48",
		"cards: expected 2x NVIDIA RTX PRO 6000, got 1x NVIDIA RTX PRO 6000",
		"CPU threads: expected 96, got 48",
		"RAM: expected 387 GB, got 193 GB",
	} {
		if !strings.Contains(errStr, want) {
			t.Errorf("error message %q missing expected part %q", errStr, want)
		}
	}

	// 4. EPYC 9B14 with 180 threads and 8 GB of RAM must not match c3d-standard-180 (Item 7).
	epycSmallRAM := &pb.Probe{
		Cpu:        "AMD EPYC 9B14 64-Core Processor",
		CpuThreads: 180,
		RamBytes:   8e9,
	}
	if got := MatchExact(epycSmallRAM); got != nil {
		t.Errorf("MatchExact(epycSmallRAM) = %v, want nil", got)
	}
	if _, err := Check("c3d-standard-180", epycSmallRAM); err == nil {
		t.Error("Check(c3d-standard-180, epycSmallRAM) expected error for 8 GB RAM, got nil")
	} else if !strings.Contains(err.Error(), "got 8 GB") {
		t.Errorf("Check error %q missing RAM difference", err.Error())
	}

	// 5. Four L4 cards with threads unreported and 1 GiB of RAM must not match g2-standard-48 (Item 7).
	l4SmallRAM := &pb.Probe{
		Gpu:      "NVIDIA L4",
		GpuCount: 4,
		RamBytes: 1e9,
	}
	if got := MatchExact(l4SmallRAM); got != nil {
		t.Errorf("MatchExact(l4SmallRAM) = %v, want nil", got)
	}
	if _, err := Check("g2-standard-48", l4SmallRAM); err == nil {
		t.Error("Check(g2-standard-48, l4SmallRAM) expected error, got nil")
	} else if !strings.Contains(err.Error(), "CPU threads: not reported") || !strings.Contains(err.Error(), "RAM: expected 206 GB, got 1 GB") {
		t.Errorf("Check error %q missing threads/RAM differences", err.Error())
	}

	// 6. Two NVIDIA L40S cards must not match g2-standard-24 (Item 7).
	l40sProbe := &pb.Probe{
		Gpu:        "NVIDIA L40S",
		GpuCount:   2,
		CpuThreads: 24,
		RamBytes:   101236568064,
	}
	if got := MatchExact(l40sProbe); got != nil {
		t.Errorf("MatchExact(l40sProbe) = %v, want nil", got)
	}
	if _, err := Check("g2-standard-24", l40sProbe); err == nil {
		t.Error("Check(g2-standard-24, l40sProbe) expected error for L40S, got nil")
	} else if !strings.Contains(err.Error(), "card model: expected NVIDIA L4, got NVIDIA L40S") {
		t.Errorf("Check error %q missing card model difference", err.Error())
	}

	// 7. Pod on g4-standard-48 node with no GPU must not match g4-standard-48 (kube-agents-bot finding).
	podNoGPU := &pb.Probe{
		Cpu:        "AMD EPYC 9B45 (Turin)",
		CpuThreads: 48,
		RamBytes:   193273528320,
		GpuCount:   0,
	}
	if got := MatchExact(podNoGPU); got != nil {
		t.Errorf("MatchExact(podNoGPU) = %v, want nil", got)
	}
	if _, err := Check("g4-standard-48", podNoGPU); err == nil {
		t.Error("Check(g4-standard-48, podNoGPU) expected error for missing GPU, got nil")
	} else if !strings.Contains(err.Error(), "cards: expected 1x NVIDIA RTX PRO 6000, got no GPU") {
		t.Errorf("Check error %q missing GPU difference", err.Error())
	}

	// 8. Pod with a smaller share (1x L4, 8 threads, 32 GB RAM) reported as unknown machine with facts.
	podShareProbe := &pb.Probe{
		Cpu:        "Intel(R) Xeon(R) CPU @ 2.20GHz",
		CpuThreads: 8,
		RamBytes:   32e9,
		Gpu:        "NVIDIA L4",
		GpuCount:   1,
	}
	if got := MatchExact(podShareProbe); got != nil {
		t.Errorf("MatchExact(podShareProbe) = %v, want nil", got)
	}
	if _, err := Check("g2-standard-24", podShareProbe); err == nil {
		t.Fatal("Check(g2-standard-24, podShareProbe): expected error, got nil")
	} else {
		errStr := err.Error()
		for _, want := range []string{
			"unknown machine (1x NVIDIA L4, Intel(R) Xeon(R) CPU @ 2.20GHz (8 threads), 32 GB RAM)",
			"cards: expected 2x NVIDIA L4, got 1x NVIDIA L4",
			"CPU threads: expected 24, got 8",
			"RAM: expected 101 GB, got 32 GB",
		} {
			if !strings.Contains(errStr, want) {
				t.Errorf("Check error %q missing %q", errStr, want)
			}
		}
	}

	// 9. Unread placement fact (such as zone in a pod) does not refuse matching hardware (Item 4).
	unreadProbe := &pb.Probe{
		Cpu:            "AMD EPYC 9B45",
		CpuThreads:     48,
		RamBytes:       193273528320,
		Gpu:            "NVIDIA RTX PRO 6000",
		GpuCount:       1,
		StorageDevices: 4,
		Unread:         []string{"zone: metadata server unreachable"},
	}
	if got := MatchExact(unreadProbe); got == nil || got.Name != "g4-standard-48" {
		t.Errorf("MatchExact(unreadProbe) = %v, want g4-standard-48", got)
	}
	if _, err := Check("g4-standard-48", unreadProbe); err != nil {
		t.Errorf("Check with unread zone fact should accept matching hardware, got: %v", err)
	}
}
