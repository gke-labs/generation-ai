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

// vxpu-agent is the per-machine agent (#384). It runs as root on each
// machine the coordinator uses, and serves the Agent gRPC service
// (pkg/agent). It mounts weights disks (#407), supervises processes
// (#657), and reports what its machine is (#663).
//
//	vxpu-agent --port 50052
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func main() {
	ctx := context.Background()
	log := klog.FromContext(ctx)
	port := flag.Int("port", 50052, "port to serve the Agent service on")
	mountRoot := flag.String("mount-root", "/mnt/vxpu-weights", "where weights disks are mounted, one directory per device name")
	workRoot := flag.String("work-root", "/mnt/ssd", "root directory for process work directories and plan files")
	klog.InitFlags(nil)
	flag.Parse()
	if err := run(ctx, *port, *mountRoot, *workRoot); err != nil {
		log.Error(err, "vxpu-agent failed")
		os.Exit(1)
	}
}

func run(ctx context.Context, port int, mountRoot, workRoot string) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log := klog.FromContext(ctx)
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	s := grpc.NewServer()
	srv := &agent.Server{MountRoot: mountRoot, WorkRoot: workRoot}
	defer srv.Stop()
	pb.RegisterAgentServer(s, srv)

	go func() {
		<-ctx.Done()
		s.GracefulStop()
	}()

	log.Info("vxpu-agent serving", "address", lis.Addr().String(), "mountRoot", mountRoot, "workRoot", workRoot)
	return s.Serve(lis)
}
