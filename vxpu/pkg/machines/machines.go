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

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// ComputeByDtype is peak throughput by precision, in FLOPS.
type ComputeByDtype struct {
	FP32 float64 `json:"fp32"`
	BF16 float64 `json:"bf16"`
	FP8  float64 `json:"fp8"`
}

// MeasuredRead is the card's measured memory read rate and its provenance.
type MeasuredRead struct {
	// BytesPerSec is the measured sequential memory read rate in bytes per second.
	BytesPerSec float64 `json:"bytes_per_sec"`
	// When is the date the measurement was recorded.
	When string `json:"when,omitempty"`
	// Where is the machine type or environment where the measurement was taken.
	Where string `json:"where,omitempty"`
	// How is the command or method used to perform the measurement.
	How string `json:"how,omitempty"`
}

// GPU is one accelerator device.
type GPU struct {
	Name                       string          `json:"name"`
	MemoryBytes                int64           `json:"memory_bytes"`
	MemoryBandwidthBytesPerSec float64         `json:"memory_bandwidth_bytes_per_sec,omitempty"`
	MeasuredRead               *MeasuredRead   `json:"measured_read,omitempty"`
	PeakComputeFLOPS           *ComputeByDtype `json:"peak_compute_flops,omitempty"`
	SMCount                    int             `json:"sm_count,omitempty"`
	PCIeGen                    int             `json:"pcie_gen,omitempty"`
	PCIeLanes                  int             `json:"pcie_lanes,omitempty"`
	// PCIeBandwidthBytesPerSec is the PCIe bandwidth between host and device in each direction.
	PCIeBandwidthBytesPerSec float64 `json:"pcie_bandwidth_bytes_per_sec,omitempty"`
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
	Name string `json:"name"`
	// Aliases lists other names that Lookup accepts for this machine type.
	Aliases  []string `json:"aliases,omitempty"`
	CPU      CPU      `json:"cpu"`
	RAMBytes int64    `json:"ram_bytes"`
	// GPU is the configuration of each GPU.
	GPU      *GPU          `json:"gpu,omitempty"`
	GPUCount int           `json:"gpu_count"`
	Storage  *LocalStorage `json:"storage,omitempty"`
	Network  *Network      `json:"network,omitempty"`
	Price    *Price        `json:"price,omitempty"`
	// Measured maps benchmark metric names, such as kernel bandwidths or usable memory, to values recorded on this machine.
	Measured    map[string]any `json:"measured,omitempty"`
	Description string         `json:"description,omitempty"`
}

// GPUMemoryBandwidthGBs returns the GPU's nominal memory bandwidth in GB/s,
// or 0 when the machine has no GPU.
func (m *Machine) GPUMemoryBandwidthGBs() float64 {
	if m == nil || m.GPU == nil {
		return 0
	}
	return m.GPU.MemoryBandwidthBytesPerSec / 1e9
}

// GPUMeasuredReadRateGBs returns the GPU's measured read rate in GB/s
// (or the nominal memory bandwidth when unmeasured) and whether it was measured.
// It returns (0, false) when the machine has no GPU.
func (m *Machine) GPUMeasuredReadRateGBs() (float64, bool) {
	if m == nil || m.GPU == nil {
		return 0, false
	}
	if m.GPU.MeasuredRead != nil && m.GPU.MeasuredRead.BytesPerSec > 0 {
		return m.GPU.MeasuredRead.BytesPerSec / 1e9, true
	}
	return m.GPUMemoryBandwidthGBs(), false
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

// CardReadRate is a card's read rate for computing its floor time, and whether it was measured.
type CardReadRate struct {
	// RateGBs is the card's memory read rate in GB/s.
	RateGBs float64
	// Measured indicates whether RateGBs was measured or taken from nominal bandwidth.
	Measured bool
}

// LookupCardReadRate finds the memory read rate in GB/s for a GPU card by name,
// returning the measured read rate when available and the nominal bandwidth otherwise.
func LookupCardReadRate(name string) (CardReadRate, error) {
	target := strings.ToLower(strings.TrimSpace(name))
	if target == "" {
		return CardReadRate{}, fmt.Errorf("empty card name")
	}
	if target == "a100" {
		return CardReadRate{}, fmt.Errorf("ambiguous card %q: specify 40gb or 80gb (e.g. \"a100-40gb\")", name)
	}
	if alias, ok := cardAliases[target]; ok {
		target = alias
	}
	for _, m := range catalog.Machines {
		if m.GPU != nil && strings.EqualFold(m.GPU.Name, target) {
			rate, measured := m.GPUMeasuredReadRateGBs()
			return CardReadRate{RateGBs: rate, Measured: measured}, nil
		}
	}
	return CardReadRate{}, fmt.Errorf("unknown card %q in machine catalog", name)
}

// ProbeOf is what an executor on m would report (Match(ProbeOf(m)) is m):
// for planning on machines not yet running (vxpu explain --machines).
func ProbeOf(m *Machine) *pb.Probe {
	p := &pb.Probe{Device: "cpu", Cpu: m.CPU.Model, CpuThreads: int32(m.CPU.Threads), Simd: m.CPU.SIMD, RamBytes: m.RAMBytes}
	if m.GPU != nil && m.GPUCount > 0 {
		p.Device, p.Gpu, p.GpuMemoryBytes, p.GpuCount = "cuda", m.GPU.Name, m.GPU.MemoryBytes, int32(m.GPUCount)
	}
	if m.Storage != nil && m.Storage.Devices > 0 {
		p.StorageDevices = int32(m.Storage.Devices)
		p.StorageBytes = m.Storage.CapacityBytes
	}
	return p
}

// Match finds the catalog machine a probe describes, or nil. An unknown
// accelerator or host CPU does not borrow another machine's numbers: a
// GPU must also have the catalog's memory (within 5%), which tells apart
// the siblings a name matches (an A100 80GB from an A100 40GB, an RTX
// PRO 6000 Workstation from the server edition).
func Match(p *pb.Probe) *Machine {
	if p == nil {
		return nil
	}
	at := func(name string) *Machine { return catalog.Machines[name] }
	if p.Gpu != "" {
		m := matchGPU(p, at)
		if m == nil || m.GPU == nil || p.GpuMemoryBytes == 0 {
			return m
		}
		if diff := float64(p.GpuMemoryBytes-m.GPU.MemoryBytes) / float64(m.GPU.MemoryBytes); diff < -0.05 || diff > 0.05 {
			return nil
		}
		return m
	}
	return matchCPU(p, at)
}

func matchGPU(p *pb.Probe, at func(string) *Machine) *Machine {
	gpu := strings.ToLower(p.Gpu)
	switch {
	case strings.Contains(gpu, "rtx pro 6000"): // not an RTX A6000, not an L4
		if p.GpuCount <= 1 {
			return at("g4-standard-48")
		}
		return at("g4-standard-96")
	case strings.Contains(gpu, "h100") && p.GpuCount == 8:
		return at("a3-highgpu-8g")
	case strings.Contains(gpu, "a100") && p.GpuCount <= 1:
		return at("a2-highgpu-1g")
	case strings.Contains(gpu, "l4") && p.GpuCount == 2:
		return at("g2-standard-24")
	case strings.Contains(gpu, "l4") && p.GpuCount == 4:
		return at("g2-standard-48")
	case isM4Pro(gpu):
		return at("mac-m4-pro")
	}
	return nil
}

func matchCPU(p *pb.Probe, at func(string) *Machine) *Machine {
	cpu := strings.ToLower(p.Cpu)
	switch {
	case isM4Pro(cpu):
		return at("mac-m4-pro")
	case strings.Contains(cpu, "9b14") || strings.Contains(cpu, "genoa"):
		if p.RamBytes >= 1000<<30 {
			return at("c3d-highmem-180")
		}
		if p.CpuThreads >= 120 {
			return at("c3d-standard-180")
		}
		return at("c3d-standard-60")
	case strings.Contains(cpu, "9b45") || strings.Contains(cpu, "turin") || strings.Contains(cpu, "9005"):
		if p.CpuThreads >= 72 {
			return at("g4-standard-96")
		}
		return at("g4-standard-48")
	}
	return nil
}

// Compare compares the reported probe facts against a catalog machine,
// returning a list of differences naming the expected and reported values.
// It returns nil if the probe matches m.
func Compare(m *Machine, p *pb.Probe) []string {
	if p == nil {
		return []string{"probe is nil"}
	}
	var diffs []string

	// Cards
	if int32(m.GPUCount) != p.GpuCount {
		diffs = append(diffs, fmt.Sprintf("cards: expected %s, got %s", describeGPU(m), describeProbeGPU(p)))
	} else if m.GPUCount > 0 {
		if !cardMatches(m.GPU.Name, p.Gpu) {
			diffs = append(diffs, fmt.Sprintf("card model: expected %s, got %s", m.GPU.Name, p.Gpu))
		}
		if p.GpuMemoryBytes > 0 && m.GPU.MemoryBytes > 0 {
			diff := float64(p.GpuMemoryBytes-m.GPU.MemoryBytes) / float64(m.GPU.MemoryBytes)
			if diff < -0.05 || diff > 0.05 {
				diffs = append(diffs, fmt.Sprintf("card memory: expected %.0f GB, got %.0f GB",
					float64(m.GPU.MemoryBytes)/1e9, float64(p.GpuMemoryBytes)/1e9))
			}
		}
	}

	// CPU Threads
	if p.CpuThreads == 0 {
		diffs = append(diffs, "CPU threads: not reported")
	} else if int(p.CpuThreads) != m.CPU.Threads {
		diffs = append(diffs, fmt.Sprintf("CPU threads: expected %d, got %d", m.CPU.Threads, p.CpuThreads))
	}

	// RAM
	if p.RamBytes == 0 {
		diffs = append(diffs, "RAM: not reported")
	} else {
		diff := float64(p.RamBytes-m.RAMBytes) / float64(m.RAMBytes)
		if diff < -0.10 || diff > 0.10 {
			diffs = append(diffs, fmt.Sprintf("RAM: expected %.0f GB, got %.0f GB",
				float64(m.RAMBytes)/1e9, float64(p.RamBytes)/1e9))
		}
	}

	// Local Disks
	if m.Storage != nil && m.Storage.Devices > 0 && m.Name != "mac-m4-pro" {
		if int(p.StorageDevices) != m.Storage.Devices {
			diffs = append(diffs, fmt.Sprintf("local disks: expected %d devices, got %d",
				m.Storage.Devices, p.StorageDevices))
		}
	}

	return diffs
}

// MatchExact finds the catalog machine that matches the probe's facts
// fact by fact. Unlike Match, which allows partial probes to identify
// a host type for planning, MatchExact requires all reported hardware
// facts (cards, CPU threads, RAM within 10%) to match the entry,
// returning nil if the probe describes a machine the database does not know.
func MatchExact(p *pb.Probe) *Machine {
	for _, name := range Names() {
		m, err := Lookup(name)
		if err != nil {
			continue
		}
		if len(Compare(m, p)) == 0 {
			return m
		}
	}
	return nil
}

// Check compares the facts a probe describes against what the plan
// expected. If the machine is unknown in the catalog or differs from
// what was planned, it returns an error naming each difference using
// the reported values.
func Check(planned string, p *pb.Probe) (*Machine, error) {
	want, err := Lookup(planned)
	if err != nil {
		return nil, fmt.Errorf("planned machine: %w", err)
	}
	diffs := Compare(want, p)
	if len(diffs) > 0 {
		actual := MatchExact(p)
		if actual != nil {
			return nil, fmt.Errorf("planned for %s, got %s: %s", want.Name, actual.Name, strings.Join(diffs, "; "))
		}
		facts := FormatFacts(p)
		if facts == "" {
			facts = "no hardware facts"
		}
		return nil, fmt.Errorf("planned for %s, but machine does not match: unknown machine (%s): %s",
			want.Name, facts, strings.Join(diffs, "; "))
	}
	return want, nil
}

func cardMatches(want, got string) bool {
	wTrim, gTrim := strings.TrimSpace(want), strings.TrimSpace(got)
	if strings.EqualFold(wTrim, gTrim) {
		return true
	}
	cWant, okWant := cardAliases[strings.ToLower(wTrim)]
	cGot, okGot := cardAliases[strings.ToLower(gTrim)]
	return okWant && okGot && cWant == cGot
}

// FormatFacts formats a probe's facts for error and status messages.
func FormatFacts(p *pb.Probe) string {
	if p == nil {
		return ""
	}
	var parts []string
	if p.GpuCount > 0 {
		parts = append(parts, describeProbeGPU(p))
	}
	if p.Cpu != "" {
		if p.CpuThreads > 0 {
			parts = append(parts, fmt.Sprintf("%s (%d threads)", p.Cpu, p.CpuThreads))
		} else {
			parts = append(parts, p.Cpu)
		}
	} else if p.CpuThreads > 0 {
		parts = append(parts, fmt.Sprintf("%d CPU threads", p.CpuThreads))
	}
	if p.RamBytes > 0 {
		parts = append(parts, fmt.Sprintf("%.0f GB RAM", float64(p.RamBytes)/1e9))
	}
	if p.StorageDevices > 0 {
		parts = append(parts, fmt.Sprintf("%d local disks", p.StorageDevices))
	}
	return strings.Join(parts, ", ")
}

func describeGPU(m *Machine) string {
	if m.GPU == nil || m.GPUCount == 0 {
		return "no GPU"
	}
	return fmt.Sprintf("%dx %s", m.GPUCount, m.GPU.Name)
}

func describeProbeGPU(p *pb.Probe) string {
	if p == nil || p.GpuCount == 0 {
		return "no GPU"
	}
	if p.Gpu == "" {
		return fmt.Sprintf("%d GPUs", p.GpuCount)
	}
	return fmt.Sprintf("%dx %s", p.GpuCount, p.Gpu)
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
