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
	"strconv"
	"strings"
	"time"

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
func Attach(ctx context.Context, gcloud Gcloud, o AttachOptions) error {
	log := klog.FromContext(ctx)
	where := []string{"--project", o.Project, "--zone", o.Zone}
	source := "--source-snapshot"
	if o.FromImage {
		source = "--image"
	}
	existing, err := gcloud(ctx, append([]string{"compute", "disks", "describe", o.Disk, "--format", "value(sourceSnapshot,sourceImage)"}, where...)...)
	if err == nil {
		// A disk of that name exists. Attach it only if it was made from this image.
		made := false
		for _, src := range strings.Fields(string(existing)) {
			made = made || strings.HasSuffix(src, "/snapshots/"+o.Image) || strings.HasSuffix(src, "/images/"+o.Image)
		}
		if !made {
			return fmt.Errorf("disk %s exists, made from %q, not %s", o.Disk, strings.TrimSpace(string(existing)), o.Image)
		}
	} else if isNotFound(err) {
		args := []string{"compute", "disks", "create", o.Disk, source, o.Image,
			"--type", o.DiskType, "--labels", "vxpu-weights=1", "--quiet"}
		if o.FromImage {
			args = append(args, "--image-project", o.Project)
		}
		if o.ThroughputMBps > 0 {
			args = append(args, "--provisioned-throughput", strconv.Itoa(o.ThroughputMBps))
		}
		if o.DiskType == "hyperdisk-ml" {
			args = append(args, "--access-mode", "READ_ONLY_MANY")
		}
		t := time.Now()
		if _, err := gcloud(ctx, append(args, where...)...); err != nil {
			return err
		}
		log.Info("Made a weights disk", "disk", o.Disk, "image", o.Image, "type", o.DiskType, "seconds", time.Since(t).Seconds())
	} else {
		return err
	}
	out, err := gcloud(ctx, append([]string{"compute", "instances", "describe", o.VM, "--format", "value(disks[].source)"}, where...)...)
	if err != nil {
		return err
	}
	attached := false
	for _, src := range strings.FieldsFunc(string(out), func(r rune) bool { return r == ';' || r == '\n' || r == ' ' }) {
		attached = attached || strings.HasSuffix(src, "/disks/"+o.Disk) // match the whole name, so ds-flash doesn't match ds-flash-build
	}
	if attached {
		log.Info("The weights disk is attached already", "vm", o.VM, "disk", o.Disk)
		return nil
	}
	t := time.Now()
	if _, err := gcloud(ctx, append([]string{"compute", "instances", "attach-disk", o.VM, "--disk", o.Disk,
		"--device-name", DeviceName, "--mode", "ro", "--quiet"}, where...)...); err != nil {
		return err
	}
	log.Info("Attached a weights disk", "vm", o.VM, "disk", o.Disk, "seconds", time.Since(t).Seconds())
	return nil
}

// Detach detaches the disk from the VM. Call it after the agent unmounts
// the disk.
func Detach(ctx context.Context, gcloud Gcloud, project, zone, vm, disk string) error {
	_, err := gcloud(ctx, "compute", "instances", "detach-disk", vm, "--disk", disk, "--project", project, "--zone", zone, "--quiet")
	return err
}

// isNotFound reports whether err says the resource was not found.
func isNotFound(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "not found") || strings.Contains(s, "not_found") || strings.Contains(s, "notfound")
}
