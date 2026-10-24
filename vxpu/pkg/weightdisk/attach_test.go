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
	"strings"
	"testing"
)

func TestAttach(t *testing.T) {
	notFound := errors.New("ERROR: (gcloud.compute.disks.describe) NOT_FOUND: The resource was not found")
	g := &fakeGcloud{fail: map[string]error{"compute disks describe": notFound}}
	if err := Attach(t.Context(), g.run, AttachOptions{Project: "p", Zone: "z", VM: "vm", Image: "ds-flash",
		Disk: "ds-flash-z", DiskType: "hyperdisk-ml", ThroughputMBps: 2400}); err != nil {
		t.Fatal(err)
	}
	create := g.find("compute disks create ds-flash-z")
	for _, want := range []string{"--source-snapshot ds-flash", "--type hyperdisk-ml", "--access-mode READ_ONLY_MANY", "--provisioned-throughput 2400"} {
		if !strings.Contains(create, want) {
			t.Errorf("disk create %q lacks %q", create, want)
		}
	}
	if attach := g.find("compute instances attach-disk vm"); !strings.Contains(attach, "--mode ro") ||
		!strings.Contains(attach, "--device-name vxpu-weights") {
		t.Errorf("attach %q", attach)
	}
	// Attach creates the disk only when describe fails with NotFound. It returns any other error directly.
	g = &fakeGcloud{fail: map[string]error{"compute disks describe": errors.New("Required 'compute.disks.get' permission")}}
	if err := Attach(t.Context(), g.run, AttachOptions{Project: "p", Zone: "z", VM: "vm", Image: "ds-flash", Disk: "ds-flash-z"}); err == nil ||
		!strings.Contains(err.Error(), "compute.disks.get") {
		t.Errorf("err %v, want describe's permission error", err)
	}
	if g.find("compute disks create") != "" {
		t.Error("created the disk when describe failed with a permission error")
	}
	// Attach reuses an existing disk made from the image, and refuses one made from another snapshot.
	describe := func(source, attachedTo string) Gcloud {
		return func(ctx context.Context, args ...string) ([]byte, error) {
			call := strings.Join(args, " ")
			switch {
			case strings.HasPrefix(call, "compute disks describe"):
				return []byte(source + "\n"), nil
			case strings.HasPrefix(call, "compute instances describe") && attachedTo != "":
				return []byte("https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/boot;" +
					"https://www.googleapis.com/compute/v1/projects/p/zones/z/disks/" + attachedTo + "\n"), nil
			}
			return g.run(ctx, args...)
		}
	}
	opts := AttachOptions{Project: "p", Zone: "z", VM: "vm2", Image: "ds-flash", Disk: "ds-flash-z", DiskType: "hyperdisk-ml"}
	g = &fakeGcloud{}
	if err := Attach(t.Context(), describe("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash", ""), opts); err != nil {
		t.Fatal(err)
	}
	if g.find("compute disks create") != "" || g.find("compute instances attach-disk vm2") == "" {
		t.Errorf("want the existing disk attached, not made again:\n%s", strings.Join(g.calls, "\n"))
	}
	g = &fakeGcloud{}
	if err := Attach(t.Context(), describe("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/other", ""), opts); err == nil ||
		g.find("compute instances attach-disk") != "" {
		t.Errorf("err %v: a disk of that name made from another snapshot must not be attached", err)
	}
	// A disk that is already attached needs nothing. A disk whose name only starts with ours is a different disk.
	g = &fakeGcloud{}
	if err := Attach(t.Context(), describe("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash", "ds-flash-z"), opts); err != nil {
		t.Fatal(err)
	}
	if g.find("compute instances attach-disk") != "" {
		t.Error("attached a disk that was attached already")
	}
	g = &fakeGcloud{}
	if err := Attach(t.Context(), describe("https://www.googleapis.com/compute/v1/projects/p/global/snapshots/ds-flash", "ds-flash-z-old"), opts); err != nil {
		t.Fatal(err)
	}
	if g.find("compute instances attach-disk vm2") == "" {
		t.Error("ds-flash-z-old taken for ds-flash-z")
	}
	// From an image, Attach passes --image, not --source-snapshot.
	g = &fakeGcloud{fail: map[string]error{"compute disks describe": notFound}}
	if err := Attach(t.Context(), g.run, AttachOptions{Project: "p", Zone: "z", VM: "vm", Image: "ds-flash", FromImage: true,
		Disk: "ds-flash-z", DiskType: "hyperdisk-ml"}); err != nil {
		t.Fatal(err)
	}
	if create := g.find("compute disks create ds-flash-z"); !strings.Contains(create, "--image ds-flash") ||
		strings.Contains(create, "--source-snapshot") || !strings.Contains(create, "--image-project p") {
		t.Errorf("disk create from an image: %q", create)
	}
}

func TestIsNotFound(t *testing.T) {
	for _, err := range []error{
		errors.New("ERROR: (gcloud.compute.disks.describe) NOT_FOUND: The resource 'projects/p/zones/z/disks/d' was not found"),
		errors.New("The resource 'projects/p/zones/z/disks/d' was not found"),
		errors.New("NOT_FOUND"),
		errors.New("NotFound"),
		errors.New("disk not found"),
	} {
		if !isNotFound(err) {
			t.Errorf("isNotFound(%q) = false, want true", err)
		}
	}
	for _, err := range []error{
		nil,
		errors.New("Required 'compute.disks.get' permission"),
		errors.New("PERMISSION_DENIED"),
		errors.New("connection reset by peer"),
	} {
		if isNotFound(err) {
			t.Errorf("isNotFound(%v) = true, want false", err)
		}
	}
}
