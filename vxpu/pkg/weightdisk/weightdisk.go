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

// Package weightdisk puts a checkpoint's weights on GCE disks (#407), so an
// executor can attach them instead of copying the weights over the network.
//
// A weights disk uses the same layout as the executor's local weight tier
// (localblobs.py): one file, blobs/<sha256>, per checkpoint file, and an
// index, weights.json, that lists them.
//
// A disk factory makes the disk once per checkpoint. The factory is a
// short-lived VM. It copies the files from the weight bucket, which uses
// modelstore's layout (gs://BUCKET/_blobs/<sha256>), onto a fresh disk.
// Build then turns that disk into a snapshot, or into an image where the
// project allows images. Either one is global, so any zone can make disks
// from it.
//
// To use the weights, Attach makes a disk from the snapshot in the
// executor's zone and attaches it read-only. Some disk types allow many VMs
// to attach the same disk. The machine's agent mounts the disk and checks
// it against the index, and the executor reads the files in place
// (VXPU_BLOB_DIR).
//
// The package runs GCE operations through the Compute Engine API (cloud.google.com/go/compute/apiv1).
package weightdisk

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"
)

const (
	// Format is the index's format.
	Format = "vxpu-weights-disk/1"
	// IndexFile and BlobDir are the index and the files' directory at the
	// root of a weights disk.
	IndexFile = "weights.json"
	BlobDir   = "blobs"
	// DeviceName is the device name a weights disk is attached under, in
	// the factory and on executors (/dev/disk/by-id/google-vxpu-weights).
	DeviceName = "vxpu-weights"
)

// Index lists a weights disk's files.
type Index struct {
	Format string `json:"format"`
	// Source is the bucket the files were copied from.
	Source string `json:"source"`
	// Files maps each file's sha256 to its size in bytes.
	Files map[string]int64 `json:"files"`
}

// ReadIndex reads the index of a weights disk mounted at dir.
func ReadIndex(dir string) (*Index, error) {
	data, err := os.ReadFile(filepath.Join(dir, IndexFile))
	if err != nil {
		return nil, fmt.Errorf("not a weights disk: %w", err)
	}
	var index Index
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("%s: %w", IndexFile, err)
	}
	if index.Format != Format {
		return nil, fmt.Errorf("%s: format %q, want %q", IndexFile, index.Format, Format)
	}
	return &index, nil
}

//go:embed factory.sh
var factoryScript string

// BuildOptions describes a weights image to make.
type BuildOptions struct {
	Project, Zone string
	// Image is the name of the snapshot or image to make. Build also names
	// the factory VM and its disk after it.
	Image string
	// AsImage makes an image rather than a snapshot.
	AsImage bool
	// Bucket is the weight bucket, as gs://BUCKET. Files lists the
	// checkpoint's files in it, mapping each sha256 to the file's size.
	Bucket string
	Files  map[string]int64
	// MachineType and DiskType set the factory VM's machine type and its
	// disk's type. ThroughputMBps and IOPS set the disk's provisioned
	// performance (Hyperdisk), which limits how fast the copy can write.
	MachineType    string
	DiskType       string
	ThroughputMBps int
	IOPS           int
	// KeepDisk keeps the factory's disk after Build makes the snapshot. That
	// leaves a ready disk in the factory's zone.
	KeepDisk bool
	// Poll is how often Build checks the factory's progress.
	Poll time.Duration
}

// BuildResult is what a build made and how long it took.
type BuildResult struct {
	Image        string
	Files        int
	Bytes        int64
	CopySeconds  float64 // how long the factory took to copy the files from the bucket
	ImageSeconds float64 // how long GCE took to make the snapshot or image from the disk
	TotalSeconds float64
}

// Build makes a weights snapshot or image. It starts a factory VM, which
// copies the files onto a fresh disk and powers off. Build then makes the
// snapshot from that disk and deletes the VM. It also deletes the disk,
// unless KeepDisk is set.
func Build(ctx context.Context, client Client, o BuildOptions) (*BuildResult, error) {
	log := klog.FromContext(ctx)
	start := time.Now()
	if o.MachineType == "" {
		o.MachineType = "c3d-standard-30"
	}
	if o.DiskType == "" {
		o.DiskType = "hyperdisk-balanced"
	}
	if o.ThroughputMBps == 0 && strings.HasPrefix(o.DiskType, "hyperdisk") {
		o.ThroughputMBps = 1200
	}
	if o.IOPS == 0 && o.DiskType == "hyperdisk-balanced" {
		o.IOPS = 4 * o.ThroughputMBps // GCE wants both when either is provisioned
	}
	if o.Poll == 0 {
		o.Poll = 15 * time.Second
	}
	if len(o.Files) == 0 || !strings.HasPrefix(o.Bucket, "gs://") {
		return nil, fmt.Errorf("a build needs files and a gs:// bucket")
	}
	var total int64
	shas := make([]string, 0, len(o.Files))
	for sha, size := range o.Files {
		total += size
		shas = append(shas, sha)
	}
	sort.Strings(shas)
	var list strings.Builder
	for _, sha := range shas {
		fmt.Fprintf(&list, "%s %d\n", sha, o.Files[sha])
	}

	// ext4's own space, and some slack
	sizeGB := (total*21/20)/1e9 + 10
	factory, disk := o.Image+"-factory", o.Image+"-build"

	dataDiskParams := &computepb.AttachedDiskInitializeParams{
		DiskName:   proto.String(disk),
		DiskSizeGb: proto.Int64(sizeGB),
		DiskType:   proto.String(fmt.Sprintf("zones/%s/diskTypes/%s", o.Zone, o.DiskType)),
	}
	if o.ThroughputMBps > 0 {
		dataDiskParams.ProvisionedThroughput = proto.Int64(int64(o.ThroughputMBps))
	}
	if o.IOPS > 0 {
		dataDiskParams.ProvisionedIops = proto.Int64(int64(o.IOPS))
	}

	inst := &computepb.Instance{
		Name:        proto.String(factory),
		MachineType: proto.String(fmt.Sprintf("zones/%s/machineTypes/%s", o.Zone, o.MachineType)),
		Labels:      map[string]string{"vxpu-role": "weights-factory"},
		Disks: []*computepb.AttachedDisk{
			{
				Boot:       proto.Bool(true),
				AutoDelete: proto.Bool(true),
				InitializeParams: &computepb.AttachedDiskInitializeParams{
					SourceImage: proto.String("projects/debian-cloud/global/images/family/debian-12"),
					DiskSizeGb:  proto.Int64(20),
				},
			},
			{
				AutoDelete:       proto.Bool(false),
				DeviceName:       proto.String(DeviceName),
				InitializeParams: dataDiskParams,
			},
		},
		Metadata: &computepb.Metadata{
			Items: []*computepb.Items{
				{Key: proto.String("enable-guest-attributes"), Value: proto.String("TRUE")},
				{Key: proto.String("vxpu-bucket"), Value: proto.String(o.Bucket)},
				{Key: proto.String("startup-script"), Value: proto.String(factoryScript)},
				{Key: proto.String("vxpu-files"), Value: proto.String(list.String())},
			},
		},
		NetworkInterfaces: []*computepb.NetworkInterface{
			{
				AccessConfigs: []*computepb.AccessConfig{
					{
						Type: proto.String("ONE_TO_ONE_NAT"),
						Name: proto.String("External NAT"),
					},
				},
			},
		},
		ServiceAccounts: []*computepb.ServiceAccount{
			{
				Email:  proto.String("default"),
				Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
			},
		},
	}

	log.Info("Starting a weights disk factory", "vm", factory, "disk", disk, "files", len(shas), "bytes", total,
		"machineType", o.MachineType, "diskType", o.DiskType)
	op, err := client.InsertInstance(ctx, &computepb.InsertInstanceRequest{
		Project:          o.Project,
		Zone:             o.Zone,
		InstanceResource: inst,
	})
	if err != nil {
		return nil, err
	}
	if err := op.Wait(ctx); err != nil {
		return nil, err
	}

	deleteFactory := func() error {
		dop, err := client.DeleteInstance(context.WithoutCancel(ctx), &computepb.DeleteInstanceRequest{
			Project:  o.Project,
			Zone:     o.Zone,
			Instance: factory,
		})
		if err != nil {
			return err
		}
		return dop.Wait(context.WithoutCancel(ctx))
	}
	deleteDisk := func() error {
		dop, err := client.DeleteDisk(context.WithoutCancel(ctx), &computepb.DeleteDiskRequest{
			Project: o.Project,
			Zone:    o.Zone,
			Disk:    disk,
		})
		if err != nil {
			return err
		}
		return dop.Wait(context.WithoutCancel(ctx))
	}

	attrs, err := waitFactory(ctx, client, o.Project, o.Zone, factory, o.Poll)
	if err != nil {
		if derr := deleteFactory(); derr != nil {
			log.Error(derr, "Deleting the factory VM", "vm", factory)
		}
		if o.KeepDisk {
			return nil, fmt.Errorf("%w (the factory's disk %s is kept)", err, disk)
		}
		// its disk outlives it (auto-delete=no) and bills by the hour: delete it too, after the VM lets it go
		if derr := deleteDisk(); derr != nil {
			return nil, fmt.Errorf("%w (and deleting the factory's disk %s failed: %v)", err, disk, derr)
		}
		return nil, err
	}
	if err := deleteFactory(); err != nil {
		return nil, err
	}
	copySeconds, err := strconv.ParseFloat(attrs["copy_seconds"], 64)
	if err != nil {
		log.Error(err, "The factory's copy_seconds", "value", attrs["copy_seconds"])
	}
	kind := "snapshot"
	if o.AsImage {
		kind = "image"
	}
	log.Info("The factory copied the weights; making the "+kind, "disk", disk, "copySeconds", copySeconds)
	t := time.Now()
	desc := fmt.Sprintf("vxpu weights disk: %d files, %d bytes from %s (index: %s)", len(shas), total, o.Bucket, IndexFile)
	sourceDiskURL := fmt.Sprintf("projects/%s/zones/%s/disks/%s", o.Project, o.Zone, disk)

	if !o.AsImage {
		sop, err := client.InsertSnapshot(ctx, &computepb.InsertSnapshotRequest{
			Project: o.Project,
			SnapshotResource: &computepb.Snapshot{
				Name:        proto.String(o.Image),
				SourceDisk:  proto.String(sourceDiskURL),
				Description: proto.String(desc),
				Labels:      map[string]string{"vxpu-weights": "1"},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("the factory's disk %s is kept: %w", disk, err)
		}
		if err := sop.Wait(ctx); err != nil {
			return nil, fmt.Errorf("the factory's disk %s is kept: %w", disk, err)
		}
	} else {
		iop, err := client.InsertImage(ctx, &computepb.InsertImageRequest{
			Project: o.Project,
			ImageResource: &computepb.Image{
				Name:        proto.String(o.Image),
				SourceDisk:  proto.String(sourceDiskURL),
				Description: proto.String(desc),
				Labels:      map[string]string{"vxpu-weights": "1"},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("the factory's disk %s is kept: %w", disk, err)
		}
		if err := iop.Wait(ctx); err != nil {
			return nil, fmt.Errorf("the factory's disk %s is kept: %w", disk, err)
		}
	}

	res := &BuildResult{
		Image:        o.Image,
		Files:        len(shas),
		Bytes:        total,
		CopySeconds:  copySeconds,
		ImageSeconds: time.Since(t).Seconds(),
	}
	if !o.KeepDisk {
		if err := deleteDisk(); err != nil {
			return nil, err
		}
	}
	res.TotalSeconds = time.Since(start).Seconds()
	log.Info("Made a weights image", "image", o.Image, "files", res.Files, "bytes", res.Bytes,
		"copySeconds", res.CopySeconds, "imageSeconds", res.ImageSeconds, "totalSeconds", res.TotalSeconds)
	return res, nil
}

// waitFactory waits for the factory's status guest attribute to say done
// (returning its attributes) or failed.
func waitFactory(ctx context.Context, client Client, project, zone, factory string, poll time.Duration) (map[string]string, error) {
	log := klog.FromContext(ctx)
	last := ""
	for {
		attrs, err := guestAttributes(ctx, client, project, zone, factory)
		if err != nil { // before the VM boots there are no attributes yet: retried
			log.V(2).Info("Reading the factory's guest attributes", "vm", factory, "err", err)
		}
		state := attrs["status"]
		if state != last {
			log.Info("Weights disk factory", "vm", factory, "status", state)
			last = state
		}
		switch {
		case state == "done":
			return attrs, nil
		case strings.HasPrefix(state, "failed"):
			return nil, fmt.Errorf("weights disk factory %s: %s", factory, state)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(poll):
		}
	}
}

// guestAttributes reads the factory's guest attributes under vxpu/.
func guestAttributes(ctx context.Context, client Client, project, zone, factory string) (map[string]string, error) {
	ga, err := client.GetGuestAttributes(ctx, &computepb.GetGuestAttributesInstanceRequest{
		Project:   project,
		Zone:      zone,
		Instance:  factory,
		QueryPath: proto.String("vxpu/"),
	})
	if err != nil {
		return nil, err
	}
	attrs := make(map[string]string)
	if ga != nil && ga.QueryValue != nil {
		for _, it := range ga.QueryValue.Items {
			attrs[it.GetKey()] = it.GetValue()
		}
	}
	return attrs, nil
}
