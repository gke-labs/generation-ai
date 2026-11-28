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
	"strings"
	"syscall"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/mtls"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// softwareVersion is stamped at build time via -ldflags "-X main.softwareVersion=...".
var softwareVersion string

type agentOptions struct {
	// port is the TCP port where the Agent gRPC server listens.
	port int
	// host is the hostname or IP address published in serving process endpoints.
	host string
	// mountRoot is the directory where weights disks are mounted.
	mountRoot string
	// workRoot is the root directory for process work directories and staged blobs.
	workRoot string
	// storeURL is the URL of the content-addressed store (modelstore).
	storeURL string
	// tlsDir is the directory containing ca.crt, tls.crt, and tls.key for mTLS, or empty for plaintext.
	tlsDir string
}

func main() {
	ctx := context.Background()
	log := klog.FromContext(ctx)
	var opts agentOptions
	flag.IntVar(&opts.port, "port", 50052, "port to serve the Agent service on")
	flag.StringVar(&opts.host, "host", "", "hostname or IP address published in process endpoints")
	flag.StringVar(&opts.mountRoot, "mount-root", "/mnt/vxpu-weights", "where weights disks are mounted, one directory per device name")
	flag.StringVar(&opts.workRoot, "work-root", "/mnt/ssd", "root directory for process work directories and plan files")
	flag.StringVar(&opts.storeURL, "store", "", "URL of the content-addressed store (modelstore)")
	flag.StringVar(&opts.tlsDir, "tls-dir", "", "directory containing ca.crt, tls.crt, and tls.key for agent mTLS; empty serves plaintext")
	klog.InitFlags(nil)
	flag.Parse()
	if err := run(ctx, opts); err != nil {
		log.Error(err, "vxpu-agent failed")
		klog.Flush()
		os.Exit(1)
	}
	klog.Flush()
}

func run(ctx context.Context, opts agentOptions) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	log := klog.FromContext(ctx)
	var serverOpts []grpc.ServerOption
	if opts.tlsDir != "" {
		creds, err := mtls.LoadAgentServerCredentials(ctx, opts.tlsDir)
		if err != nil {
			return fmt.Errorf("loading agent mTLS credentials: %w", err)
		}
		serverOpts = append(serverOpts, grpc.Creds(creds))
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", opts.port))
	if err != nil {
		return err
	}
	s := grpc.NewServer(serverOpts...)
	var store *weightstore.Client
	if opts.storeURL != "" {
		store = weightstore.New(opts.storeURL)
	}
	version := strings.TrimSpace(softwareVersion)
	if version == "" {
		log.Info("vxpu-agent has no software version stamped; plans with packaged segments will be refused")
	}
	srv := agent.NewServer(opts.mountRoot, opts.workRoot, store)
	srv.Host = opts.host
	srv.SoftwareVersion = version
	srv.Start()
	defer srv.Stop()
	pb.RegisterAgentServer(s, srv)

	go func() {
		<-ctx.Done()
		srv.Stop()
		s.GracefulStop()
	}()

	srv.StartReaper()

	log.Info("vxpu-agent serving", "address", lis.Addr().String(), "mountRoot", opts.mountRoot, "workRoot", opts.workRoot, "softwareVersion", version, "mtls", opts.tlsDir != "")
	return s.Serve(lis)
}
