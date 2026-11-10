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
	"flag"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
)

func cmdDiskAttach(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("disk attach", flag.ContinueOnError)
	var o weightdisk.AttachOptions
	flags.StringVar(&o.Project, "project", "", "GCP project")
	flags.StringVar(&o.Zone, "zone", "", "the VM's zone")
	flags.StringVar(&o.VM, "vm", "", "the executor's VM")
	flags.StringVar(&o.Image, "image", "", "the weights snapshot (or image, with --from-image)")
	flags.BoolVar(&o.FromImage, "from-image", false, "--image names an image, not a snapshot")
	flags.StringVar(&o.Disk, "disk", "", "the disk to make from the image in the zone (reused if it exists)")
	flags.StringVar(&o.DiskType, "type", "hyperdisk-ml", "type of a new disk (hyperdisk-ml: attachable read-only to many VMs)")
	flags.IntVar(&o.ThroughputMBps, "throughput", 0, "a new disk's provisioned throughput, MB/s (Hyperdisk)")
	agentAddr := flags.String("agent", "", "the VM's agent, HOST:PORT")
	verify := flags.Bool("verify", false, "read every file and check its sha256 (measures the disk's read throughput)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if o.Project == "" || o.Zone == "" || o.VM == "" || o.Image == "" || o.Disk == "" || *agentAddr == "" {
		return fmt.Errorf("disk attach needs --project, --zone, --vm, --image, --disk and --agent")
	}
	client, err := weightdisk.NewClient(ctx)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := client.Close(); cerr != nil {
			log.Error(cerr, "closing GCE client")
		}
	}()
	start := time.Now()
	if err := weightdisk.Attach(ctx, client, o); err != nil {
		return err
	}
	agent, done, err := dialAgent(ctx, *agentAddr)
	if err != nil {
		return err
	}
	defer done()
	resp, err := agent.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: weightdisk.DeviceName, Verify: *verify})
	if err != nil {
		return err
	}
	log.Info("Weights disk ready", "vm", o.VM, "blobDir", resp.GetBlobDir(), "files", len(resp.GetFiles()),
		"bytes", resp.GetTotalBytes(), "seconds", time.Since(start).Seconds())
	out := map[string]any{"blob_dir": resp.GetBlobDir(), "files": len(resp.GetFiles()), "bytes": resp.GetTotalBytes(),
		"mount_seconds": resp.GetMountSeconds(), "ready_seconds": time.Since(start).Seconds()}
	if *verify && resp.GetVerifySeconds() > 0 {
		out["verify_seconds"] = resp.GetVerifySeconds()
		out["read_bytes_per_sec"] = float64(resp.GetTotalBytes()) / resp.GetVerifySeconds()
	}
	return printJSON(out)
}

func cmdDiskDetach(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("disk detach", flag.ContinueOnError)
	project := flags.String("project", "", "GCP project")
	zone := flags.String("zone", "", "the VM's zone")
	vm := flags.String("vm", "", "the executor's VM")
	disk := flags.String("disk", "", "the weights disk")
	agentAddr := flags.String("agent", "", "the VM's agent, HOST:PORT")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *project == "" || *zone == "" || *vm == "" || *disk == "" || *agentAddr == "" {
		return fmt.Errorf("disk detach needs --project, --zone, --vm, --disk and --agent")
	}
	client, err := weightdisk.NewClient(ctx)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := client.Close(); cerr != nil {
			log.Error(cerr, "closing GCE client")
		}
	}()
	agent, done, err := dialAgent(ctx, *agentAddr)
	if err != nil {
		return err
	}
	defer done()
	if _, err := agent.UnmountWeights(ctx, &pb.UnmountWeightsRequest{DeviceName: weightdisk.DeviceName}); err != nil {
		return err
	}
	return weightdisk.Detach(ctx, client, *project, *zone, *vm, *disk)
}

// dialAgent connects to a machine's agent; done closes the connection.
func dialAgent(ctx context.Context, addr string) (pb.AgentClient, func(), error) {
	log := klog.FromContext(ctx)
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, fmt.Errorf("dialing the agent at %s: %w", addr, err)
	}
	return pb.NewAgentClient(conn), func() {
		if err := conn.Close(); err != nil {
			log.Error(err, "closing the agent connection", "address", addr)
		}
	}, nil
}
