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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	compute "cloud.google.com/go/compute/apiv1"
	computepb "cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/proto"
)

// GCEBackend communicates with the Google Compute Engine API.
type GCEBackend struct {
	client  *compute.InstancesClient
	project string
}

// NewGCEBackend constructs a GCEBackend for the given project.
func NewGCEBackend(ctx context.Context, project string, opts ...option.ClientOption) (*GCEBackend, error) {
	if project == "" {
		return nil, errors.New("GCE project must not be empty")
	}
	client, err := compute.NewInstancesRESTClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating Compute Engine instances client: %w", err)
	}
	return &GCEBackend{client: client, project: project}, nil
}

// Close closes the underlying Compute Engine client connection.
func (b *GCEBackend) Close() error {
	if b.client != nil {
		return b.client.Close()
	}
	return nil
}

// Create requests a new flex-start instance in the specified zone and waits for the operation to complete.
func (b *GCEBackend) Create(ctx context.Context, zone string, spec MachineSpec) error {
	machineType := spec.MachineType
	if !strings.Contains(machineType, "/") {
		machineType = fmt.Sprintf("zones/%s/machineTypes/%s", zone, machineType)
	}
	image := spec.Image
	if image == "" {
		image = "projects/debian-cloud/global/images/family/debian-12"
	}
	diskType := spec.BootDiskType
	if diskType == "" {
		diskType = "hyperdisk-balanced"
	}
	if !strings.Contains(diskType, "/") {
		diskType = fmt.Sprintf("zones/%s/diskTypes/%s", zone, diskType)
	}
	diskSizeGB := spec.BootDiskSizeGB
	if diskSizeGB <= 0 {
		diskSizeGB = 100
	}

	disks := []*computepb.AttachedDisk{{
		Boot: proto.Bool(true), AutoDelete: proto.Bool(true), Type: proto.String("PERSISTENT"),
		InitializeParams: &computepb.AttachedDiskInitializeParams{
			SourceImage: proto.String(image),
			DiskType:    proto.String(diskType),
			DiskSizeGb:  proto.Int64(diskSizeGB),
		},
	}}

	var guestAccelerators []*computepb.AcceleratorConfig
	for _, acc := range spec.Accelerators {
		accType := acc.Type
		if !strings.Contains(accType, "/") {
			accType = fmt.Sprintf("zones/%s/acceleratorTypes/%s", zone, accType)
		}
		guestAccelerators = append(guestAccelerators, &computepb.AcceleratorConfig{
			AcceleratorType:  proto.String(accType),
			AcceleratorCount: proto.Int32(int32(acc.Count)),
		})
	}

	networkInterfaces := []*computepb.NetworkInterface{{
		Network: proto.String("global/networks/default"),
		AccessConfigs: []*computepb.AccessConfig{{
			Type: proto.String("ONE_TO_ONE_NAT"),
			Name: proto.String("External NAT"),
		}},
	}}

	scheduling := &computepb.Scheduling{
		ProvisioningModel:         proto.String("FLEX_START"),
		OnHostMaintenance:         proto.String("TERMINATE"),
		InstanceTerminationAction: proto.String("DELETE"),
	}
	if spec.MaxRunDuration > 0 {
		scheduling.MaxRunDuration = &computepb.Duration{Seconds: proto.Int64(int64(spec.MaxRunDuration.Seconds()))}
	}

	var params *computepb.InstanceParams
	if spec.RequestValidDuration > 0 {
		params = &computepb.InstanceParams{
			RequestValidForDuration: &computepb.Duration{Seconds: proto.Int64(int64(spec.RequestValidDuration.Seconds()))},
		}
	}

	sa := spec.ServiceAccount
	if sa == "" {
		sa = "default"
	}
	serviceAccounts := []*computepb.ServiceAccount{{
		Email:  proto.String(sa),
		Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
	}}

	inst := &computepb.Instance{
		Name:              proto.String(spec.Name),
		MachineType:       proto.String(machineType),
		Disks:             disks,
		GuestAccelerators: guestAccelerators,
		NetworkInterfaces: networkInterfaces,
		Scheduling:        scheduling,
		Params:            params,
		Labels:            spec.Labels,
		ServiceAccounts:   serviceAccounts,
	}

	op, err := b.client.Insert(ctx, &computepb.InsertInstanceRequest{
		Project:          b.project,
		Zone:             zone,
		InstanceResource: inst,
	})
	if err != nil {
		if isUnsupportedZone(err) {
			return fmt.Errorf("%w: %w", ErrUnsupportedZone, err)
		}
		return err
	}

	if err := op.Wait(ctx); err != nil {
		if isUnsupportedZone(err) || isUnsupportedOp(op) {
			return fmt.Errorf("%w: %w", ErrUnsupportedZone, err)
		}
		return err
	}
	if isUnsupportedOp(op) {
		return ErrUnsupportedZone
	}
	return nil
}

// Describe returns the current status of the instance in the specified zone.
func (b *GCEBackend) Describe(ctx context.Context, zone, name string) (*Instance, error) {
	inst, err := b.client.Get(ctx, &computepb.GetInstanceRequest{
		Project:  b.project,
		Zone:     zone,
		Instance: name,
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	var creationTime time.Time
	if ts := inst.GetCreationTimestamp(); ts != "" {
		var err error
		creationTime, err = time.Parse(time.RFC3339, ts)
		if err != nil {
			return nil, fmt.Errorf("parsing creation timestamp: %w", err)
		}
	}

	return &Instance{
		Name:              name,
		Zone:              zone,
		Status:            InstanceStatus(inst.GetStatus()),
		StatusMessage:     inst.GetStatusMessage(),
		MachineType:       inst.GetMachineType(),
		CreationTimestamp: creationTime,
	}, nil
}

// Delete removes the instance in the specified zone.
func (b *GCEBackend) Delete(ctx context.Context, zone, name string) error {
	_, err := b.client.Delete(ctx, &computepb.DeleteInstanceRequest{
		Project:  b.project,
		Zone:     zone,
		Instance: name,
	})
	if err != nil && isNotFound(err) {
		return ErrNotFound
	}
	return err
}

func isNotFound(err error) bool {
	var gerr *googleapi.Error
	return errors.As(err, &gerr) && gerr.Code == http.StatusNotFound
}

type gceErrorResponse struct {
	Error struct {
		Errors []struct {
			Reason   string `json:"reason"`
			Location string `json:"location"`
		} `json:"errors"`
	} `json:"error"`
}

func isUnsupportedZone(err error) bool {
	var gerr *googleapi.Error
	if !errors.As(err, &gerr) || gerr.Code != http.StatusBadRequest {
		return false
	}
	for _, e := range gerr.Errors {
		if e.Reason == "invalidUsage" || e.Reason == "unsupported" {
			return true
		}
	}
	if gerr.Body != "" {
		var resp gceErrorResponse
		if json.Unmarshal([]byte(gerr.Body), &resp) == nil {
			for _, e := range resp.Error.Errors {
				if e.Reason == "invalidUsage" || e.Reason == "unsupported" {
					return true
				}
				if e.Reason == "invalid" && (strings.HasPrefix(e.Location, "resource.machineType") || strings.HasPrefix(e.Location, "resource.guestAccelerators")) {
					return true
				}
			}
		}
	}
	return false
}

func isUnsupportedOp(op *compute.Operation) bool {
	if op == nil || op.Proto() == nil || op.Proto().GetError() == nil {
		return false
	}
	for _, e := range op.Proto().GetError().GetErrors() {
		code, loc := e.GetCode(), e.GetLocation()
		if code == "INVALID_USAGE" || code == "UNSUPPORTED" {
			return true
		}
		if code == "INVALID" && (strings.HasPrefix(loc, "resource.machineType") || strings.HasPrefix(loc, "resource.guestAccelerators")) {
			return true
		}
	}
	return false
}
