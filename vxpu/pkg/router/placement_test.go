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
	"testing"
)

func TestPlanPlacement_FitsAccelerator(t *testing.T) {
	// Gemma-4-E4B: ~16 GB bf16 fits an L4 (24 GB * 0.85 = 20.4 GB).
	p := PlanPlacement("nvidia-l4", 16_000_000_000)
	if !p.IsAccelerator() || p.Accelerator != "nvidia-l4" {
		t.Fatalf("expected L4 placement, got %+v", p)
	}
	if p.Threads != 0 {
		t.Errorf("accelerator placement should not set threads, got %d", p.Threads)
	}
}

func TestPlanPlacement_SpillsToCPU(t *testing.T) {
	// Gemma-4-31B: 62.6 GB bf16 cannot be resident on a 24 GB L4.
	weights := int64(62_578_686_256)
	p := PlanPlacement("nvidia-l4", weights)
	if p.IsAccelerator() {
		t.Fatalf("expected CPU placement, got %+v", p)
	}
	minMem := weights + weights/4
	if p.Memory.Value() < minMem {
		t.Errorf("memory request %d below weights+25%% (%d)", p.Memory.Value(), minMem)
	}
	if p.MemoryLimit.Value() <= p.Memory.Value() {
		t.Errorf("memory limit %d should exceed request %d", p.MemoryLimit.Value(), p.Memory.Value())
	}
	if p.CPU.MilliValue() != int64(defaultMaxExecutorCPUs*1000) {
		t.Errorf("expected CPU request capped at %d, got %s", defaultMaxExecutorCPUs, p.CPU.String())
	}
	if p.Threads != defaultMaxExecutorCPUs {
		t.Errorf("expected %d threads, got %d", defaultMaxExecutorCPUs, p.Threads)
	}
}

func TestPlanPlacement_UnknownAcceleratorAssumesFit(t *testing.T) {
	p := PlanPlacement("nvidia-future-9000", 500_000_000_000)
	if !p.IsAccelerator() {
		t.Fatalf("unknown accelerator should be trusted, got %+v", p)
	}
}

func TestPlanPlacement_NoAcceleratorSmallModel(t *testing.T) {
	// A tiny test model on kind: stays well under 2 GiB so CI can
	// schedule it.
	p := PlanPlacement("none", 270_000_000)
	if p.IsAccelerator() {
		t.Fatalf("expected CPU placement, got %+v", p)
	}
	if p.Memory.Value() > 2*gib {
		t.Errorf("memory request %s too large for a 270 MB model", p.Memory.String())
	}
	if p.CPU.MilliValue() != 500 {
		t.Errorf("expected 500m CPU, got %s", p.CPU.String())
	}
	if p.Threads != 1 {
		t.Errorf("expected 1 thread, got %d", p.Threads)
	}
}

func TestPlanPlacement_EnvOverrides(t *testing.T) {
	t.Setenv("VXPU_ACCELERATOR_MEMORY_GIB", "96")
	p := PlanPlacement("nvidia-l4", 62_578_686_256)
	if !p.IsAccelerator() {
		t.Fatalf("with 96 GiB override the 31B should fit, got %+v", p)
	}

	t.Setenv("VXPU_ACCELERATOR_MEMORY_GIB", "")
	t.Setenv("VXPU_CPU_EXECUTOR_MAX_CPUS", "8")
	p = PlanPlacement("nvidia-l4", 62_578_686_256)
	if p.CPU.MilliValue() != 8000 || p.Threads != 8 {
		t.Errorf("expected 8 CPUs, got %s / %d threads", p.CPU.String(), p.Threads)
	}
}

func TestWeightBytes(t *testing.T) {
	manifest := `{"format":"vxpu-manifest/v1alpha1","files":{
		"a":{"size":100,"name":"a.safetensors","source":"http://x/a"},
		"b":{"size":250,"name":"b.safetensors","source":"http://x/b"}},
		"tensors":{},"config":{},"source":{}}`
	got, err := weightBytes(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if got != 350 {
		t.Errorf("expected 350, got %d", got)
	}
	if _, err := weightBytes("not json"); err == nil {
		t.Error("expected error for invalid manifest")
	}
}
