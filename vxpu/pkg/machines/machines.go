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

// Package machines is the coordinator's machine database (#394). It has
// three parts:
//   - the catalog of machine types vXPU knows, one YAML file per type in
//     catalog/. Each file lists the machine's CPUs, GPUs, memory and
//     bandwidths, local storage, network, prices, and what we measured on it;
//   - Probe, what an executor reports about the machine it runs on;
//   - Match, which finds the catalog machine a probe describes.
//
// The planner uses it to price layouts (#349).
package machines

import (
	"embed"
	"fmt"
	"path"
	"sort"
	"strings"

	"sigs.k8s.io/yaml"
)

// ComputeByDtype is peak throughput by precision, in FLOPS.
type ComputeByDtype struct {
	FP32 float64 `json:"fp32"`
	BF16 float64 `json:"bf16"`
	FP8  float64 `json:"fp8"`
}

// GPU is one accelerator device.
type GPU struct {
	Name                       string          `json:"name"`
	MemoryBytes                int64           `json:"memory_bytes"`
	MemoryBandwidthBytesPerSec float64         `json:"memory_bandwidth_bytes_per_sec,omitempty"`
	PeakComputeFLOPS           *ComputeByDtype `json:"peak_compute_flops,omitempty"`
	SMCount                    int             `json:"sm_count,omitempty"`
	PCIeGen                    int             `json:"pcie_gen,omitempty"`
	PCIeLanes                  int             `json:"pcie_lanes,omitempty"`
	PCIeBandwidthBytesPerSec   float64         `json:"pcie_bandwidth_bytes_per_sec,omitempty"` // host<->device, each way
}

// CPU is the host processor.
type CPU struct {
	Model                      string           `json:"model"`
	Cores                      int              `json:"cores"`
	Threads                    int              `json:"threads"`
	SIMD                       string           `json:"simd"`
	MemoryBandwidthBytesPerSec float64          `json:"memory_bandwidth_bytes_per_sec,omitempty"`
	CacheSizes                 map[string]int64 `json:"cache_sizes,omitempty"`
}

// LocalStorage is the machine's local NVMe, all devices together.
type LocalStorage struct {
	Devices                   int     `json:"devices"`
	CapacityBytes             int64   `json:"capacity_bytes"`
	ReadBandwidthBytesPerSec  float64 `json:"read_bandwidth_bytes_per_sec"`
	WriteBandwidthBytesPerSec float64 `json:"write_bandwidth_bytes_per_sec"`
	ReadIOPS                  float64 `json:"read_iops"`
	WriteIOPS                 float64 `json:"write_iops"`
}

// Network is the machine's network interface.
type Network struct {
	BandwidthBytesPerSec float64 `json:"bandwidth_bytes_per_sec"`
}

// Price is the hourly price.
type Price struct {
	OnDemandUSDPerHour float64 `json:"on_demand_usd_per_hour"`
	SpotUSDPerHour     float64 `json:"spot_usd_per_hour"`
}

// Machine is one machine type.
type Machine struct {
	Name        string         `json:"name"`
	Aliases     []string       `json:"aliases,omitempty"` // other names Lookup accepts
	CPU         CPU            `json:"cpu"`
	RAMBytes    int64          `json:"ram_bytes"`
	GPU         *GPU           `json:"gpu,omitempty"` // per device
	GPUCount    int            `json:"gpu_count"`
	Storage     *LocalStorage  `json:"storage,omitempty"`
	Network     *Network       `json:"network,omitempty"`
	Price       *Price         `json:"price,omitempty"`
	Measured    map[string]any `json:"measured,omitempty"` // what we measured on it
	Description string         `json:"description,omitempty"`
}

// GPUMemoryBandwidthGBs returns the GPU's memory bandwidth in GB/s, or 0
// when the machine has no GPU.
func (m *Machine) GPUMemoryBandwidthGBs() float64 {
	if m == nil || m.GPU == nil {
		return 0
	}
	return m.GPU.MemoryBandwidthBytesPerSec / 1e9
}

// The catalog keeps one YAML file per machine type, named for the type, so
// a person can read and review each machine on its own. Sizes are in bytes
// and rates in bytes per second. The files write round numbers in
// exponent form, such as 24e9, when that is exact.
//
//go:embed catalog/*.yaml
var catalogFiles embed.FS

var catalog = struct {
	Aliases  map[string]string
	Machines map[string]*Machine
}{map[string]string{}, map[string]*Machine{}}

func init() {
	files, err := catalogFiles.ReadDir("catalog")
	if err != nil {
		panic(fmt.Sprintf("machines: catalog: %v", err))
	}
	for _, f := range files {
		data, err := catalogFiles.ReadFile(path.Join("catalog", f.Name()))
		if err != nil {
			panic(fmt.Sprintf("machines: catalog/%s: %v", f.Name(), err))
		}
		m := &Machine{}
		if err := yaml.UnmarshalStrict(data, m); err != nil {
			panic(fmt.Sprintf("machines: catalog/%s: %v", f.Name(), err))
		}
		if want := strings.TrimSuffix(f.Name(), ".yaml"); m.Name != want {
			panic(fmt.Sprintf("machines: catalog/%s names machine %q, want %q", f.Name(), m.Name, want))
		}
		catalog.Machines[m.Name] = m
		for _, alias := range m.Aliases {
			if other, ok := catalog.Aliases[alias]; ok {
				panic(fmt.Sprintf("machines: alias %q is both %s's and %s's", alias, other, m.Name))
			}
			catalog.Aliases[alias] = m.Name
		}
	}
	for alias, name := range catalog.Aliases { // Lookup checks aliases first, so one would hide a machine of that name
		if _, ok := catalog.Machines[alias]; ok {
			panic(fmt.Sprintf("machines: %s's alias %q is another machine's name", name, alias))
		}
	}
}

// Names lists the catalog's machines, sorted.
func Names() []string {
	var names []string
	for name := range catalog.Machines {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Lookup finds a machine by name or alias, in any case.
func Lookup(name string) (*Machine, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if alias, ok := catalog.Aliases[key]; ok {
		key = alias
	}
	if m, ok := catalog.Machines[key]; ok {
		return m, nil
	}
	return nil, fmt.Errorf("unknown machine %q; known: %s", name, strings.Join(Names(), ", "))
}

// cardAliases maps common short names and driver strings to catalog GPU names.
var cardAliases = map[string]string{
	"l4":             "nvidia l4",
	"a100-40gb":      "nvidia a100-sxm4-40gb",
	"a100-sxm4-40gb": "nvidia a100-sxm4-40gb",
	"h100":           "nvidia h100 80gb hbm3",
	"m4 pro":         "apple m4 pro gpu",
	"m4-pro":         "apple m4 pro gpu",
	"nvidia rtx pro 6000 blackwell server edition": "nvidia rtx pro 6000",
	"rtx pro 6000 blackwell server edition":        "nvidia rtx pro 6000",
}

// LookupCard finds the memory bandwidth in GB/s for a GPU card by name.
func LookupCard(name string) (float64, error) {
	target := strings.ToLower(strings.TrimSpace(name))
	if target == "" {
		return 0, fmt.Errorf("empty card name")
	}
	if target == "a100" {
		return 0, fmt.Errorf("ambiguous card %q: specify 40gb or 80gb (e.g. \"a100-40gb\")", name)
	}
	if alias, ok := cardAliases[target]; ok {
		target = alias
	}
	for _, m := range catalog.Machines {
		if m.GPU != nil && strings.EqualFold(m.GPU.Name, target) {
			return m.GPUMemoryBandwidthGBs(), nil
		}
	}
	return 0, fmt.Errorf("unknown card %q in machine catalog", name)
}

// Probe is what an executor reports about the machine it runs on: the
// executor's plan.probe_hardware, which reads only what the machine
// says about itself (no bandwidths, no prices).
type Probe struct {
	Device         string `json:"device"`
	CPU            string `json:"cpu"`
	CPUThreads     int    `json:"cpu_threads"`
	SIMD           string `json:"simd"`
	RAMBytes       int64  `json:"ram_bytes,omitempty"`
	GPU            string `json:"gpu,omitempty"`
	GPUMemoryBytes int64  `json:"gpu_memory_bytes,omitempty"`
	GPUCount       int    `json:"gpu_count,omitempty"`
}

// ProbeOf is what an executor on m would report (Match(ProbeOf(m)) is m):
// for planning on machines not yet running (vxpu explain --machines).
func ProbeOf(m *Machine) Probe {
	p := Probe{Device: "cpu", CPU: m.CPU.Model, CPUThreads: m.CPU.Threads, SIMD: m.CPU.SIMD, RAMBytes: m.RAMBytes}
	if m.GPU != nil && m.GPUCount > 0 {
		p.Device, p.GPU, p.GPUMemoryBytes, p.GPUCount = "cuda", m.GPU.Name, m.GPU.MemoryBytes, m.GPUCount
	}
	return p
}

// Match finds the catalog machine a probe describes, or nil. An unknown
// accelerator or host CPU does not borrow another machine's numbers: a
// GPU must also have the catalog's memory (within 5%), which tells apart
// the siblings a name matches (an A100 80GB from an A100 40GB, an RTX
// PRO 6000 Workstation from the server edition).
func Match(p Probe) *Machine {
	at := func(name string) *Machine { return catalog.Machines[name] }
	if p.GPU != "" {
		m := matchGPU(p, at)
		if m == nil || m.GPU == nil || p.GPUMemoryBytes == 0 {
			return m
		}
		if diff := float64(p.GPUMemoryBytes-m.GPU.MemoryBytes) / float64(m.GPU.MemoryBytes); diff < -0.05 || diff > 0.05 {
			return nil
		}
		return m
	}
	return matchCPU(p, at)
}

func matchGPU(p Probe, at func(string) *Machine) *Machine {
	gpu := strings.ToLower(p.GPU)
	switch {
	case strings.Contains(gpu, "rtx pro 6000"): // not an RTX A6000, not an L4
		if p.GPUCount <= 1 {
			return at("g4-standard-48")
		}
		return at("g4-standard-96")
	case strings.Contains(gpu, "h100") && p.GPUCount == 8:
		return at("a3-highgpu-8g")
	case strings.Contains(gpu, "a100") && p.GPUCount <= 1:
		return at("a2-highgpu-1g")
	case strings.Contains(gpu, "l4") && p.GPUCount == 2:
		return at("g2-standard-24")
	case strings.Contains(gpu, "l4") && p.GPUCount == 4:
		return at("g2-standard-48")
	case isM4Pro(gpu):
		return at("mac-m4-pro")
	}
	return nil
}

func matchCPU(p Probe, at func(string) *Machine) *Machine {
	cpu := strings.ToLower(p.CPU)
	switch {
	case isM4Pro(cpu):
		return at("mac-m4-pro")
	case strings.Contains(cpu, "9b14") || strings.Contains(cpu, "genoa"):
		if p.RAMBytes >= 1000<<30 {
			return at("c3d-highmem-180")
		}
		if p.CPUThreads >= 120 {
			return at("c3d-standard-180")
		}
		return at("c3d-standard-60")
	case strings.Contains(cpu, "9b45") || strings.Contains(cpu, "turin") || strings.Contains(cpu, "9005"):
		if p.CPUThreads >= 72 {
			return at("g4-standard-96")
		}
		return at("g4-standard-48")
	}
	return nil
}

func isM4Pro(s string) bool {
	return strings.Contains(s, "m4 pro") || (strings.Contains(s, "apple") && strings.Contains(s, "m4"))
}

// AcceleratorMemoryGB is the on-board memory of a GKE accelerator, by
// its node label. The router lets VXPU_ACCELERATOR_MEMORY_GIB stand in
// for an unlisted (or MIG-partitioned) one.
func AcceleratorMemoryGB(label string) (float64, bool) {
	gb, ok := acceleratorMemoryGB[label]
	return gb, ok
}

var acceleratorMemoryGB = map[string]float64{
	"nvidia-tesla-k80":      12,
	"nvidia-tesla-p4":       8,
	"nvidia-tesla-p100":     16,
	"nvidia-tesla-v100":     16,
	"nvidia-tesla-t4":       16,
	"nvidia-l4":             24,
	"nvidia-tesla-a100":     40,
	"nvidia-a100-80gb":      80,
	"nvidia-h100-80gb":      80,
	"nvidia-h100-mega-80gb": 80,
	"nvidia-h200-141gb":     141,
	"nvidia-b200":           180,
	"nvidia-rtx-pro-6000":   96,
}
