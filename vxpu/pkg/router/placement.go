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

package router

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"

	"k8s.io/apimachinery/pkg/api/resource"

	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

const (
	gib = int64(1) << 30

	// acceleratorFitFraction is the share of accelerator memory the
	// resident weights may occupy; the remainder is KV cache,
	// activations, and compile workspace.
	acceleratorFitFraction = 0.85

	// defaultMaxExecutorCPUs caps the CPU request of a CPU-placed
	// executor. Decode on CPU is memory-bandwidth bound, so more cores
	// help only up to the socket's bandwidth. Override with
	// VXPU_CPU_EXECUTOR_MAX_CPUS.
	defaultMaxExecutorCPUs = 30
)

// acceleratorMemoryGB maps GKE accelerator labels to on-board memory.
// Override for an unlisted (or MIG-partitioned) accelerator with
// VXPU_ACCELERATOR_MEMORY_GIB.
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

// Placement is the router's decision of where an executor runs and
// how big it is. The artifact is hardware-agnostic; the router sizes
// the pod from the manifest's weight bytes and falls back to a CPU
// node when the model cannot be resident on the configured
// accelerator.
type Placement struct {
	// Accelerator is the GKE accelerator label, or "" for CPU.
	Accelerator string
	// WeightBytes is the total size of the weights the executor will
	// rehydrate (0 when unknown).
	WeightBytes int64
	Reason      string

	CPU         resource.Quantity
	Memory      resource.Quantity
	MemoryLimit resource.Quantity
	// Threads is the CPU thread count hint for a CPU executor
	// (OMP_NUM_THREADS); 0 for accelerator placements.
	Threads int
}

func (p Placement) IsAccelerator() bool { return p.Accelerator != "" }

// acceleratorMemoryBytes returns the accelerator's memory, or 0 if
// unknown.
func acceleratorMemoryBytes(accelerator string) int64 {
	if v := os.Getenv("VXPU_ACCELERATOR_MEMORY_GIB"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			return int64(f * float64(gib))
		}
	}
	if gb, ok := acceleratorMemoryGB[accelerator]; ok {
		return int64(gb * 1e9)
	}
	return 0
}

func maxExecutorCPUs() int {
	if v := os.Getenv("VXPU_CPU_EXECUTOR_MAX_CPUS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultMaxExecutorCPUs
}

// weightBytes sums the sizes of the files a manifest references.
func weightBytes(manifestJSON string) (int64, error) {
	var manifest v1alpha1.Manifest
	if err := json.Unmarshal([]byte(manifestJSON), &manifest); err != nil {
		return 0, fmt.Errorf("parsing manifest: %w", err)
	}
	var total int64
	for _, f := range manifest.Files {
		total += f.Size
	}
	return total, nil
}

// PlanPlacement decides where an executor for a model of weightBytes
// runs when the router is configured for accelerator (possibly "" or
// "none").
//
// The weights go to the accelerator when they fit within
// acceleratorFitFraction of its memory (an accelerator of unknown
// size is assumed to fit, preserving explicit operator intent).
// Otherwise the executor is a CPU pod sized from the weights: memory
// for the resident bf16 tensors plus headroom, and CPUs proportional
// to the weights up to maxExecutorCPUs.
func PlanPlacement(accelerator string, weightBytes int64) Placement {
	if accelerator != "" && accelerator != "none" {
		capacity := acceleratorMemoryBytes(accelerator)
		budget := int64(float64(capacity) * acceleratorFitFraction)
		if capacity == 0 || weightBytes <= budget {
			reason := "accelerator memory unknown; assuming the model fits"
			if capacity != 0 {
				reason = fmt.Sprintf("%.1f GB of weights fit the %s budget of %.1f GB",
					float64(weightBytes)/1e9, accelerator, float64(budget)/1e9)
			}
			return Placement{
				Accelerator: accelerator,
				WeightBytes: weightBytes,
				Reason:      reason,
				CPU:         resource.MustParse("4"),
				Memory:      resource.MustParse("20Gi"),
				MemoryLimit: resource.MustParse("24Gi"),
			}
		}
		return cpuPlacement(weightBytes, fmt.Sprintf(
			"%.1f GB of weights exceed the %s budget of %.1f GB; placing on CPU",
			float64(weightBytes)/1e9, accelerator, float64(budget)/1e9))
	}
	return cpuPlacement(weightBytes, "no accelerator configured")
}

func cpuPlacement(weightBytes int64, reason string) Placement {
	request := weightBytes + weightBytes/4 + gib
	limit := weightBytes + weightBytes/2 + 2*gib

	// One CPU per 2 GiB of weights, at least half a core, capped.
	cpus := float64(weightBytes) / float64(2*gib)
	if cpus < 0.5 {
		cpus = 0.5
	} else {
		cpus = math.Ceil(cpus)
	}
	cpus = math.Min(cpus, float64(maxExecutorCPUs()))
	threads := int(math.Max(1, math.Floor(cpus)))

	return Placement{
		Accelerator: "",
		WeightBytes: weightBytes,
		Reason:      reason,
		CPU:         *resource.NewMilliQuantity(int64(cpus*1000), resource.DecimalSI),
		Memory:      *resource.NewQuantity(request, resource.BinarySI),
		MemoryLimit: *resource.NewQuantity(limit, resource.BinarySI),
		Threads:     threads,
	}
}
