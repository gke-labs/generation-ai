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
	"errors"
	"strings"
	"testing"

	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/protobuf/proto"
)

func TestAttach(t *testing.T) {
	client := &fakeClient{}
	if err := Attach(t.Context(), client, AttachOptions{
		Project: "p", Zone: "z", VM: "vm", Image: "ds-flash",
		Disk: "ds-flash-z", DiskType: "hyperdisk-ml", ThroughputMBps: 2400,
	}); err != nil {
		t.Fatal(err)
	}

	getDisk, ok := findCall[*computepb.GetDiskRequest](client)
	if !ok || getDisk.GetProject() != "p" || getDisk.GetZone() != "z" || getDisk.GetDisk() != "ds-flash-z" {
		t.Errorf("get disk %+v", getDisk)
	}

	insDisk, ok := findCall[*computepb.InsertDiskRequest](client)
	if !ok {
		t.Fatal("InsertDiskRequest not found")
	}
	disk := insDisk.GetDiskResource()
	if disk.GetName() != "ds-flash-z" ||
		disk.GetSourceSnapshot() != "projects/p/global/snapshots/ds-flash" ||
		disk.GetType() != "zones/z/diskTypes/hyperdisk-ml" ||
		disk.GetAccessMode() != "READ_ONLY_MANY" ||
		disk.GetProvisionedThroughput() != 2400 ||
		disk.GetLabels()["vxpu-weights"] != "1" {
		t.Errorf("insert disk %+v", disk)
	}

	getInst, ok := findCall[*computepb.GetInstanceRequest](client)
	if !ok || getInst.GetInstance() != "vm" {
		t.Errorf("get instance %+v", getInst)
	}

	attachReq, ok := findCall[*computepb.AttachDiskInstanceRequest](client)
	if !ok {
		t.Fatal("AttachDiskInstanceRequest not found")
	}
	if attachReq.GetInstance() != "vm" ||
		attachReq.GetAttachedDiskResource().GetMode() != "READ_ONLY" ||
		attachReq.GetAttachedDiskResource().GetDeviceName() != DeviceName ||
		attachReq.GetAttachedDiskResource().GetSource() != "projects/p/zones/z/disks/ds-flash-z" {
		t.Errorf("attach req %+v", attachReq)
	}

	// Attach creates the disk only when GetDisk returns 404 (not found). It returns any other error directly.
	client = &fakeClient{diskErr: errors.New("network error")}
	if err := Attach(t.Context(), client, AttachOptions{Project: "p", Zone: "z", VM: "vm", Image: "ds-flash", Disk: "ds-flash-z"}); err == nil ||
		!strings.Contains(err.Error(), "network error") {
		t.Errorf("err %v, want network error", err)
	}
	if _, ok := findCall[*computepb.InsertDiskRequest](client); ok {
		t.Error("created disk when GetDisk failed")
	}

	// Attach reuses an existing disk made from the snapshot, and refuses one made from another snapshot.
	opts := AttachOptions{Project: "p", Zone: "z", VM: "vm2", Image: "ds-flash", Disk: "ds-flash-z", DiskType: "hyperdisk-ml"}
	client = &fakeClient{
		disk: &computepb.Disk{
			Name:           proto.String("ds-flash-z"),
			SourceSnapshot: proto.String("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash"),
		},
	}
	if err := Attach(t.Context(), client, opts); err != nil {
		t.Fatal(err)
	}
	if _, ok := findCall[*computepb.InsertDiskRequest](client); ok {
		t.Error("want existing disk attached, not made again")
	}
	if _, ok := findCall[*computepb.AttachDiskInstanceRequest](client); !ok {
		t.Error("want existing disk attached to vm2")
	}

	client = &fakeClient{
		disk: &computepb.Disk{
			Name:           proto.String("ds-flash-z"),
			SourceSnapshot: proto.String("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/other"),
		},
	}
	if err := Attach(t.Context(), client, opts); err == nil {
		t.Errorf("a disk of that name made from another snapshot must not be attached")
	}
	if _, ok := findCall[*computepb.AttachDiskInstanceRequest](client); ok {
		t.Error("attached disk made from other snapshot")
	}

	// A disk that is already attached needs nothing. A disk whose name only starts with ours is a different disk.
	client = &fakeClient{
		disk: &computepb.Disk{
			Name:           proto.String("ds-flash-z"),
			SourceSnapshot: proto.String("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash"),
		},
		instance: &computepb.Instance{
			Disks: []*computepb.AttachedDisk{
				{Source: proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/boot")},
				{Source: proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/ds-flash-z")},
			},
		},
	}
	if err := Attach(t.Context(), client, opts); err != nil {
		t.Fatal(err)
	}
	if _, ok := findCall[*computepb.AttachDiskInstanceRequest](client); ok {
		t.Error("attached a disk that was attached already")
	}

	client = &fakeClient{
		disk: &computepb.Disk{
			Name:           proto.String("ds-flash-z"),
			SourceSnapshot: proto.String("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash"),
		},
		instance: &computepb.Instance{
			Disks: []*computepb.AttachedDisk{
				{Source: proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/boot")},
				{Source: proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/ds-flash-z-old")},
			},
		},
	}
	if err := Attach(t.Context(), client, opts); err != nil {
		t.Fatal(err)
	}
	if _, ok := findCall[*computepb.AttachDiskInstanceRequest](client); !ok {
		t.Error("ds-flash-z-old taken for ds-flash-z")
	}

	// From an image, Attach passes source_image, not source_snapshot.
	client = &fakeClient{}
	if err := Attach(t.Context(), client, AttachOptions{
		Project: "p", Zone: "z", VM: "vm", Image: "ds-flash", FromImage: true,
		Disk: "ds-flash-z", DiskType: "hyperdisk-ml",
	}); err != nil {
		t.Fatal(err)
	}
	insDisk, ok = findCall[*computepb.InsertDiskRequest](client)
	if !ok {
		t.Fatal("InsertDiskRequest not found")
	}
	if insDisk.GetDiskResource().GetSourceImage() != "projects/p/global/images/ds-flash" ||
		insDisk.GetDiskResource().GetSourceSnapshot() != "" {
		t.Errorf("disk create from an image: %+v", insDisk.GetDiskResource())
	}
}

func TestDetach(t *testing.T) {
	client := &fakeClient{
		instance: &computepb.Instance{
			Disks: []*computepb.AttachedDisk{
				{
					Source:     proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/ds-flash-z"),
					DeviceName: proto.String(DeviceName),
				},
			},
		},
	}
	if err := Detach(t.Context(), client, "p", "z", "vm", "ds-flash-z"); err != nil {
		t.Fatal(err)
	}
	detachReq, ok := findCall[*computepb.DetachDiskInstanceRequest](client)
	if !ok {
		t.Fatal("DetachDiskInstanceRequest not found")
	}
	if detachReq.GetInstance() != "vm" || detachReq.GetDeviceName() != DeviceName {
		t.Errorf("detach request: %+v", detachReq)
	}

	// When GetInstance fails, Detach returns the error directly and does not detach.
	client = &fakeClient{instErr: errors.New("cannot get instance")}
	if err := Detach(t.Context(), client, "p", "z", "vm", "ds-flash-z"); err == nil ||
		!strings.Contains(err.Error(), "cannot get instance") {
		t.Errorf("err %v, want cannot get instance", err)
	}
	if _, ok := findCall[*computepb.DetachDiskInstanceRequest](client); ok {
		t.Error("detached disk when GetInstance failed")
	}

	// When the disk is not attached to the VM, Detach returns an error.
	client = &fakeClient{
		instance: &computepb.Instance{
			Disks: []*computepb.AttachedDisk{
				{
					Source:     proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/other-disk"),
					DeviceName: proto.String(DeviceName),
				},
			},
		},
	}
	if err := Detach(t.Context(), client, "p", "z", "vm", "ds-flash-z"); err == nil ||
		!strings.Contains(err.Error(), "not attached") {
		t.Errorf("err %v, want disk not attached", err)
	}
	if _, ok := findCall[*computepb.DetachDiskInstanceRequest](client); ok {
		t.Error("detached disk that was not attached")
	}

	// A disk whose device name equals the disk name but whose source is different is not detached.
	client = &fakeClient{
		instance: &computepb.Instance{
			Disks: []*computepb.AttachedDisk{
				{
					Source:     proto.String("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/other-disk"),
					DeviceName: proto.String("ds-flash-z"),
				},
			},
		},
	}
	if err := Detach(t.Context(), client, "p", "z", "vm", "ds-flash-z"); err == nil ||
		!strings.Contains(err.Error(), "not attached") {
		t.Errorf("err %v, want disk not attached", err)
	}
	if _, ok := findCall[*computepb.DetachDiskInstanceRequest](client); ok {
		t.Error("detached an unrelated disk that shared the device name")
	}
}
