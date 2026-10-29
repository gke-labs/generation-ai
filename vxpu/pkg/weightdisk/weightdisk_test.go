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
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/protobuf/proto"
)

func TestBuild(t *testing.T) {
	client := &fakeClient{}
	res, err := Build(t.Context(), client, BuildOptions{
		Project: "p", Zone: "us-central1-a", Image: "ds-flash", Bucket: "gs://weights",
		Files: map[string]int64{"aa": 300e9, "bb": 200e9}, Poll: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Files != 2 || res.Bytes != 500e9 || res.CopySeconds != 412 {
		t.Errorf("result %+v", res)
	}

	instReq, ok := findCall[*computepb.InsertInstanceRequest](client)
	if !ok {
		t.Fatal("InsertInstanceRequest not found")
	}
	if instReq.GetProject() != "p" || instReq.GetZone() != "us-central1-a" {
		t.Errorf("project %s, zone %s", instReq.GetProject(), instReq.GetZone())
	}
	inst := instReq.GetInstanceResource()
	if inst.GetName() != "ds-flash-factory" {
		t.Errorf("name %s, want ds-flash-factory", inst.GetName())
	}
	if inst.GetMachineType() != "zones/us-central1-a/machineTypes/c3d-standard-30" {
		t.Errorf("machine type %s", inst.GetMachineType())
	}
	if inst.GetLabels()["vxpu-role"] != "weights-factory" {
		t.Errorf("labels %v", inst.GetLabels())
	}
	if len(inst.GetDisks()) != 2 {
		t.Fatalf("len(disks) = %d, want 2", len(inst.GetDisks()))
	}
	boot := inst.GetDisks()[0]
	if !boot.GetBoot() || !boot.GetAutoDelete() ||
		boot.GetInitializeParams().GetSourceImage() != "projects/debian-cloud/global/images/family/debian-12" ||
		boot.GetInitializeParams().GetDiskSizeGb() != 20 {
		t.Errorf("boot disk %+v", boot)
	}
	data := inst.GetDisks()[1]
	if data.GetAutoDelete() || data.GetDeviceName() != DeviceName {
		t.Errorf("data disk flags autoDelete=%v, deviceName=%s", data.GetAutoDelete(), data.GetDeviceName())
	}
	params := data.GetInitializeParams()
	if params.GetDiskName() != "ds-flash-build" || params.GetDiskSizeGb() != 535 ||
		params.GetDiskType() != "zones/us-central1-a/diskTypes/hyperdisk-balanced" ||
		params.GetProvisionedThroughput() != 1200 || params.GetProvisionedIops() != 4800 {
		t.Errorf("data disk params %+v", params)
	}
	metaMap := make(map[string]string)
	for _, it := range inst.GetMetadata().GetItems() {
		metaMap[it.GetKey()] = it.GetValue()
	}
	if metaMap["enable-guest-attributes"] != "TRUE" || metaMap["vxpu-bucket"] != "gs://weights" {
		t.Errorf("metadata %v", metaMap)
	}
	if metaMap["startup-script"] == "" || !strings.Contains(metaMap["vxpu-files"], "aa 300000000000") {
		t.Errorf("script/files metadata %v", metaMap)
	}

	// the VM goes before the image is made from its disk; the disk after
	order := []string{
		"InsertInstance",
		"GetGuestAttributes",
		"DeleteInstance",
		"InsertSnapshot",
		"DeleteDisk",
	}
	at := 0
	for _, c := range client.calls {
		name := ""
		switch c.(type) {
		case *computepb.InsertInstanceRequest:
			name = "InsertInstance"
		case *computepb.GetGuestAttributesInstanceRequest:
			name = "GetGuestAttributes"
		case *computepb.DeleteInstanceRequest:
			name = "DeleteInstance"
		case *computepb.InsertSnapshotRequest:
			name = "InsertSnapshot"
		case *computepb.DeleteDiskRequest:
			name = "DeleteDisk"
		}
		if at < len(order) && name == order[at] {
			at++
		}
	}
	if at != len(order) {
		t.Errorf("calls out of order (matched %d/%d):\n%v", at, len(order), client.calls)
	}
	if client.polls != 2 {
		t.Errorf("polled %d times, want 2", client.polls)
	}

	snapReq, ok := findCall[*computepb.InsertSnapshotRequest](client)
	if !ok {
		t.Fatal("InsertSnapshotRequest not found")
	}
	snap := snapReq.GetSnapshotResource()
	if snap.GetName() != "ds-flash" ||
		snap.GetSourceDisk() != "projects/p/zones/us-central1-a/disks/ds-flash-build" ||
		snap.GetLabels()["vxpu-weights"] != "1" {
		t.Errorf("snapshot %+v", snap)
	}

	delDisk, ok := findCall[*computepb.DeleteDiskRequest](client)
	if !ok {
		t.Fatal("DeleteDiskRequest not found")
	}
	if delDisk.GetDisk() != "ds-flash-build" {
		t.Errorf("deleted disk %s, want ds-flash-build", delDisk.GetDisk())
	}
}

func TestBuildAsImage(t *testing.T) {
	client := &fakeClient{}
	res, err := Build(t.Context(), client, BuildOptions{
		Project: "p", Zone: "us-central1-a", Image: "ds-flash", Bucket: "gs://weights",
		Files: map[string]int64{"aa": 100e9}, AsImage: true, Poll: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Image != "ds-flash" {
		t.Errorf("res %+v", res)
	}
	if _, ok := findCall[*computepb.InsertSnapshotRequest](client); ok {
		t.Error("snapshot was created, want image")
	}
	imgReq, ok := findCall[*computepb.InsertImageRequest](client)
	if !ok {
		t.Fatal("InsertImageRequest not found")
	}
	img := imgReq.GetImageResource()
	if img.GetName() != "ds-flash" ||
		img.GetSourceDisk() != "projects/p/zones/us-central1-a/disks/ds-flash-build" ||
		img.GetLabels()["vxpu-weights"] != "1" {
		t.Errorf("image %+v", img)
	}
}

func TestBuildDeletesTheFactoryWhenTheCopyFails(t *testing.T) {
	client := &fakeClient{
		guestAttrs: []*computepb.GuestAttributes{
			{
				QueryValue: &computepb.GuestAttributesValue{
					Items: []*computepb.GuestAttributesEntry{
						{Key: proto.String("status"), Value: proto.String("failed: gcloud storage cp: 403")},
					},
				},
			},
		},
	}
	_, err := Build(t.Context(), client, BuildOptions{
		Project: "p", Zone: "z", Image: "x", Bucket: "gs://b",
		Files: map[string]int64{"aa": 1}, Poll: time.Millisecond,
	})
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("err %v, want the factory's failure", err)
	}
	delInst, ok := findCall[*computepb.DeleteInstanceRequest](client)
	if !ok || delInst.GetInstance() != "x-factory" {
		t.Errorf("want the factory deleted, got %v", delInst)
	}
	if _, ok := findCall[*computepb.InsertSnapshotRequest](client); ok {
		t.Error("snapshot was created despite copy failure")
	}
	delDisk, ok := findCall[*computepb.DeleteDiskRequest](client)
	if !ok || delDisk.GetDisk() != "x-build" {
		t.Errorf("want the factory's disk deleted, got %v", delDisk)
	}
}

func TestBuildKeepDisk(t *testing.T) {
	client := &fakeClient{}
	_, err := Build(t.Context(), client, BuildOptions{
		Project: "p", Zone: "us-central1-a", Image: "ds-flash", Bucket: "gs://weights",
		Files: map[string]int64{"aa": 100e9}, KeepDisk: true, Poll: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := findCall[*computepb.DeleteDiskRequest](client); ok {
		t.Error("build disk was deleted with KeepDisk=true")
	}
}
