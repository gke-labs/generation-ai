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
// (pkg/agent). It mounts weights disks (#407) and supervises processes
// (#657).
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
	"strings"
	"syscall"

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
	authToken := flag.String("auth-token", "", "authentication token required for incoming RPCs (or set VXPU_AGENT_TOKEN)")
	authTokenFile := flag.String("auth-token-file", "", "file containing authentication token required for incoming RPCs")
	insecureAllowAnonymous := flag.Bool("insecure-allow-anonymous", false, "allow calls from anyone without authentication; for tests and development machines")
	klog.InitFlags(nil)
	flag.Parse()

	token := *authToken
	if token == "" && *authTokenFile != "" {
		data, err := os.ReadFile(*authTokenFile)
		if err != nil {
			log.Error(err, "reading auth token file", "file", *authTokenFile)
			os.Exit(1)
		}
		token = strings.TrimSpace(string(data))
	}
	if token == "" {
		token = os.Getenv("VXPU_AGENT_TOKEN")
	}

	if err := run(ctx, *port, *mountRoot, *workRoot, token, *insecureAllowAnonymous); err != nil {
		log.Error(err, "vxpu-agent failed")
		os.Exit(1)
	}
}

func run(ctx context.Context, port int, mountRoot, workRoot, token string, insecureAllowAnonymous bool) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log := klog.FromContext(ctx)
	s, err := agent.NewGRPCServer(token, insecureAllowAnonymous)
	if err != nil {
		return err
	}
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
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
