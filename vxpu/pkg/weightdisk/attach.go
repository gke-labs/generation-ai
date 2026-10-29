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
	"fmt"
	"strings"
	"time"

	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"
)

// AttachOptions describes a weights disk for a VM.
type AttachOptions struct {
	Project, Zone string
	// VM is the executor's VM. Image is the weights snapshot, or an image
	// when FromImage is set. Disk is the disk to make from it in the VM's
	// zone; Attach reuses the disk if it already exists.
	VM, Image, Disk string
	FromImage       bool
	// DiskType and ThroughputMBps describe a new disk. Attach makes a
	// hyperdisk-ml disk read-only-many, so other VMs can attach it too.
	DiskType       string
	ThroughputMBps int
}

// Attach makes the disk from the image if it doesn't exist, and attaches it
// to the VM read-only, under DeviceName. The VM's agent then mounts it
// (MountWeights).
func Attach(ctx context.Context, client Client, o AttachOptions) error {
	log := klog.FromContext(ctx)

	existingDisk, err := client.GetDisk(ctx, &computepb.GetDiskRequest{
		Project: o.Project,
		Zone:    o.Zone,
		Disk:    o.Disk,
	})
	switch {
	case isNotFound(err):
		diskResource := &computepb.Disk{
			Name:   proto.String(o.Disk),
			Type:   proto.String(fmt.Sprintf("zones/%s/diskTypes/%s", o.Zone, o.DiskType)),
			Labels: map[string]string{"vxpu-weights": "1"},
		}
		if o.FromImage {
			diskResource.SourceImage = proto.String(fmt.Sprintf("projects/%s/global/images/%s", o.Project, o.Image))
		} else {
			diskResource.SourceSnapshot = proto.String(fmt.Sprintf("projects/%s/global/snapshots/%s", o.Project, o.Image))
		}
		if o.ThroughputMBps > 0 {
			diskResource.ProvisionedThroughput = proto.Int64(int64(o.ThroughputMBps))
		}
		if o.DiskType == "hyperdisk-ml" {
			diskResource.AccessMode = proto.String("READ_ONLY_MANY")
		}
		t := time.Now()
		op, err := client.InsertDisk(ctx, &computepb.InsertDiskRequest{
			Project:      o.Project,
			Zone:         o.Zone,
			DiskResource: diskResource,
		})
		if err != nil {
			return err
		}
		if err := op.Wait(ctx); err != nil {
			return err
		}
		log.Info("Made a weights disk", "disk", o.Disk, "image", o.Image, "type", o.DiskType, "seconds", time.Since(t).Seconds())
	case err != nil:
		return err
	default:
		// A disk of that name exists. Attach it only if it was made from this image.
		src, wantSuffix := existingDisk.GetSourceSnapshot(), "/snapshots/"+o.Image
		if o.FromImage {
			src, wantSuffix = existingDisk.GetSourceImage(), "/images/"+o.Image
		}
		if !strings.HasSuffix(src, wantSuffix) {
			return fmt.Errorf("disk %s exists, made from %q, not %s", o.Disk, src, o.Image)
		}
	}

	inst, err := client.GetInstance(ctx, &computepb.GetInstanceRequest{
		Project:  o.Project,
		Zone:     o.Zone,
		Instance: o.VM,
	})
	if err != nil {
		return err
	}
	attached := false
	for _, d := range inst.GetDisks() {
		if strings.HasSuffix(d.GetSource(), "/disks/"+o.Disk) {
			attached = true
			break
		}
	}
	if attached {
		log.Info("The weights disk is attached already", "vm", o.VM, "disk", o.Disk)
		return nil
	}

	t := time.Now()
	op, err := client.AttachDisk(ctx, &computepb.AttachDiskInstanceRequest{
		Project:  o.Project,
		Zone:     o.Zone,
		Instance: o.VM,
		AttachedDiskResource: &computepb.AttachedDisk{
			Source:     proto.String(fmt.Sprintf("projects/%s/zones/%s/disks/%s", o.Project, o.Zone, o.Disk)),
			DeviceName: proto.String(DeviceName),
			Mode:       proto.String("READ_ONLY"),
		},
	})
	if err != nil {
		return err
	}
	if err := op.Wait(ctx); err != nil {
		return err
	}
	log.Info("Attached a weights disk", "vm", o.VM, "disk", o.Disk, "seconds", time.Since(t).Seconds())
	return nil
}

// Detach detaches the disk from the VM. Call it after the agent unmounts
// the disk.
func Detach(ctx context.Context, client Client, project, zone, vm, disk string) error {
	inst, err := client.GetInstance(ctx, &computepb.GetInstanceRequest{
		Project:  project,
		Zone:     zone,
		Instance: vm,
	})
	if err != nil {
		return err
	}
	deviceName := ""
	for _, d := range inst.GetDisks() {
		if strings.HasSuffix(d.GetSource(), "/disks/"+disk) {
			deviceName = d.GetDeviceName()
			break
		}
	}
	if deviceName == "" {
		return fmt.Errorf("disk %s is not attached to VM %s", disk, vm)
	}
	op, err := client.DetachDisk(ctx, &computepb.DetachDiskInstanceRequest{
		Project:    project,
		Zone:       zone,
		Instance:   vm,
		DeviceName: deviceName,
	})
	if err != nil {
		return err
	}
	return op.Wait(ctx)
}
