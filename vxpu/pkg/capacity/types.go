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

// Package capacity races flex-start VM requests across multiple zones.
//
// The first zone that grants capacity wins the race. The racer cancels
// all other pending requests immediately in parallel.
package capacity

import (
	"context"
	"errors"
	"time"
)

// InstanceStatus represents the lifecycle state of a VM instance.
type InstanceStatus string

const (
	// StatusPending indicates that the flex-start request is queued and waiting for capacity.
	StatusPending InstanceStatus = "PENDING"
	// StatusRunning indicates that the instance is running and has capacity.
	StatusRunning InstanceStatus = "RUNNING"
	// StatusTerminated indicates that the instance stopped or expired without capacity.
	StatusTerminated InstanceStatus = "TERMINATED"
)

// Instance describes a VM instance in a specific zone.
type Instance struct {
	Name              string
	Zone              string
	Status            InstanceStatus
	StatusMessage     string
	MachineType       string
	CreationTimestamp time.Time
}

// MachineSpec configures the VM instance to request in each zone.
type MachineSpec struct {
	Name                 string
	MachineType          string
	Image                string
	BootDiskType         string
	BootDiskSizeGB       int64
	Labels               map[string]string
	MaxRunDuration       time.Duration
	RequestValidDuration time.Duration
}

// Winner identifies the winning instance and how long it waited for capacity.
type Winner struct {
	Name string
	Zone string
	Wait time.Duration
}

// RaceOptions controls how the capacity racer operates.
type RaceOptions struct {
	Spec           MachineSpec
	Zones          []string
	Timeout        time.Duration
	PollInterval   time.Duration
	CleanupTimeout time.Duration
}

// Backend executes cloud operations for VM instances.
type Backend interface {
	// Create requests a new flex-start instance in the specified zone.
	Create(ctx context.Context, zone string, spec MachineSpec) error

	// Describe returns the current status of the instance in the specified zone.
	Describe(ctx context.Context, zone string, name string) (*Instance, error)

	// Delete removes the instance in the specified zone.
	Delete(ctx context.Context, zone string, name string) error
}

// ErrNotFound indicates that the requested instance does not exist.
var ErrNotFound = errors.New("instance not found")

// ErrUnsupportedZone indicates that the zone does not offer the requested machine type.
var ErrUnsupportedZone = errors.New("zone does not offer the requested machine type")
