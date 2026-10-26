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

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
	"k8s.io/klog/v2"
)

// cmdDisk makes and attaches weights disks (#407):
//
//	vxpu disk build --project P --zone Z --image NAME --bucket gs://B --manifest ARTIFACT/manifest.json
//	vxpu disk attach --project P --zone Z --vm VM --image NAME --disk DISK --agent HOST:PORT [--verify]
//	vxpu disk detach --project P --zone Z --vm VM --disk DISK --agent HOST:PORT
func cmdDisk(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu disk build|attach|detach [flags]")
	}
	switch args[0] {
	case "build":
		return cmdDiskBuild(ctx, args[1:])
	case "attach":
		return cmdDiskAttach(ctx, args[1:])
	case "detach":
		return cmdDiskDetach(ctx, args[1:])
	}
	return fmt.Errorf("unknown disk command %q (build, attach, detach)", args[0])
}

func cmdDiskBuild(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("disk build", flag.ContinueOnError)
	var o weightdisk.BuildOptions
	flags.StringVar(&o.Project, "project", "", "GCP project")
	flags.StringVar(&o.Zone, "zone", "", "zone for the factory VM (the image is regional or multi-regional)")
	flags.StringVar(&o.Image, "image", "", "name of the snapshot (or image, with --as-image) to make")
	flags.BoolVar(&o.AsImage, "as-image", false, "make an image rather than a snapshot")
	flags.StringVar(&o.Bucket, "bucket", "", "the weight bucket, as gs://BUCKET, in modelstore's layout (_blobs/<sha256>)")
	manifest := flags.String("manifest", "", "an artifact's manifest.json: the checkpoint files to put on the disk")
	flags.StringVar(&o.MachineType, "machine-type", "", "factory machine type (default c3d-standard-30)")
	flags.StringVar(&o.DiskType, "disk-type", "", "factory disk type (default hyperdisk-balanced)")
	flags.IntVar(&o.ThroughputMBps, "throughput", 0, "factory disk's provisioned throughput, MB/s (Hyperdisk; default 1200)")
	flags.BoolVar(&o.KeepDisk, "keep-disk", false, "keep the factory's disk after imaging")
	timeout := flags.Duration("timeout", 2*time.Hour, "give up after this long")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if o.Project == "" || o.Zone == "" || o.Image == "" || o.Bucket == "" || *manifest == "" {
		return fmt.Errorf("disk build needs --project, --zone, --image, --bucket and --manifest")
	}
	files, err := manifestFiles(*manifest)
	if err != nil {
		return err
	}
	o.Files = files
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	client, err := weightdisk.NewClient(ctx)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := client.Close(); cerr != nil {
			log.Error(cerr, "closing GCE client")
		}
	}()
	res, err := weightdisk.Build(ctx, client, o)
	if err != nil {
		return err
	}
	return printJSON(res)
}

// manifestFiles reads an artifact manifest's checkpoint files: sha256 -> size.
func manifestFiles(path string) (map[string]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m struct {
		Files map[string]struct {
			Size int64 `json:"size"`
		} `json:"files"`
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	files := make(map[string]int64, len(m.Files))
	for sha, f := range m.Files {
		files[sha] = f.Size
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("%s lists no files", path)
	}
	return files, nil
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
