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

package weightdisk

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	compute "cloud.google.com/go/compute/apiv1"
	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/api/googleapi"
	"k8s.io/klog/v2"
)

// Operation represents a long-running Compute Engine operation.
type Operation interface {
	Wait(ctx context.Context) error
}

// Client performs the Compute Engine operations weightdisk needs.
type Client interface {
	InsertInstance(ctx context.Context, req *computepb.InsertInstanceRequest) (Operation, error)
	GetGuestAttributes(ctx context.Context, req *computepb.GetGuestAttributesInstanceRequest) (*computepb.GuestAttributes, error)
	DeleteInstance(ctx context.Context, req *computepb.DeleteInstanceRequest) (Operation, error)
	GetInstance(ctx context.Context, req *computepb.GetInstanceRequest) (*computepb.Instance, error)
	AttachDisk(ctx context.Context, req *computepb.AttachDiskInstanceRequest) (Operation, error)
	DetachDisk(ctx context.Context, req *computepb.DetachDiskInstanceRequest) (Operation, error)
	InsertDisk(ctx context.Context, req *computepb.InsertDiskRequest) (Operation, error)
	GetDisk(ctx context.Context, req *computepb.GetDiskRequest) (*computepb.Disk, error)
	DeleteDisk(ctx context.Context, req *computepb.DeleteDiskRequest) (Operation, error)
	InsertSnapshot(ctx context.Context, req *computepb.InsertSnapshotRequest) (Operation, error)
	InsertImage(ctx context.Context, req *computepb.InsertImageRequest) (Operation, error)
	Close() error
}

// NewClient returns a GCE Compute Engine client using Application Default Credentials.
func NewClient(ctx context.Context) (Client, error) {
	log := klog.FromContext(ctx)
	c := &gceClient{}
	var err error
	defer func() {
		if err != nil {
			if cerr := c.Close(); cerr != nil {
				log.Error(cerr, "closing GCE client after initialization failure")
			}
		}
	}()

	if c.instances, err = compute.NewInstancesRESTClient(ctx); err != nil {
		return nil, fmt.Errorf("instances client: %w", err)
	}
	if c.disks, err = compute.NewDisksRESTClient(ctx); err != nil {
		return nil, fmt.Errorf("disks client: %w", err)
	}
	if c.snapshots, err = compute.NewSnapshotsRESTClient(ctx); err != nil {
		return nil, fmt.Errorf("snapshots client: %w", err)
	}
	if c.images, err = compute.NewImagesRESTClient(ctx); err != nil {
		return nil, fmt.Errorf("images client: %w", err)
	}
	return c, nil
}

type gceClient struct {
	instances *compute.InstancesClient
	disks     *compute.DisksClient
	snapshots *compute.SnapshotsClient
	images    *compute.ImagesClient
}

func (c *gceClient) InsertInstance(ctx context.Context, req *computepb.InsertInstanceRequest) (Operation, error) {
	op, err := c.instances.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) GetGuestAttributes(ctx context.Context, req *computepb.GetGuestAttributesInstanceRequest) (*computepb.GuestAttributes, error) {
	return c.instances.GetGuestAttributes(ctx, req)
}

func (c *gceClient) DeleteInstance(ctx context.Context, req *computepb.DeleteInstanceRequest) (Operation, error) {
	op, err := c.instances.Delete(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) GetInstance(ctx context.Context, req *computepb.GetInstanceRequest) (*computepb.Instance, error) {
	return c.instances.Get(ctx, req)
}

func (c *gceClient) AttachDisk(ctx context.Context, req *computepb.AttachDiskInstanceRequest) (Operation, error) {
	op, err := c.instances.AttachDisk(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) DetachDisk(ctx context.Context, req *computepb.DetachDiskInstanceRequest) (Operation, error) {
	op, err := c.instances.DetachDisk(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) InsertDisk(ctx context.Context, req *computepb.InsertDiskRequest) (Operation, error) {
	op, err := c.disks.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) GetDisk(ctx context.Context, req *computepb.GetDiskRequest) (*computepb.Disk, error) {
	return c.disks.Get(ctx, req)
}

func (c *gceClient) DeleteDisk(ctx context.Context, req *computepb.DeleteDiskRequest) (Operation, error) {
	op, err := c.disks.Delete(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) InsertSnapshot(ctx context.Context, req *computepb.InsertSnapshotRequest) (Operation, error) {
	op, err := c.snapshots.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) InsertImage(ctx context.Context, req *computepb.InsertImageRequest) (Operation, error) {
	op, err := c.images.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	return &gceOp{op: op}, nil
}

func (c *gceClient) Close() error {
	var errs []error
	if c.instances != nil {
		if err := c.instances.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if c.disks != nil {
		if err := c.disks.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if c.snapshots != nil {
		if err := c.snapshots.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if c.images != nil {
		if err := c.images.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

type gceOp struct {
	op *compute.Operation
}

func (o *gceOp) Wait(ctx context.Context) error {
	return o.op.Wait(ctx)
}

// isNotFound reports whether err represents an API 404 Not Found error.
func isNotFound(err error) bool {
	var gerr *googleapi.Error
	return errors.As(err, &gerr) && gerr.Code == http.StatusNotFound
}
