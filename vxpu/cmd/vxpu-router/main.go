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
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/router"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

func main() {
	klog.InitFlags(nil)
	defer klog.Flush()

	portFlag := flag.Int("port", 50051, "The server port")
	adminPortFlag := flag.Int("admin-port", 0, "The admin port for Executor service (0 to disable)")
	imageFlag := flag.String("image", "", "The executor image to run")
	nsFlag := flag.String("namespace", "", "The Kubernetes namespace to use")
	acceleratorFlag := flag.String("accelerator", "", "GKE accelerator label for the executor pod (e.g., nvidia-l4)")
	modelstoreFlag := flag.String("modelstore", "",
		"modelstore URL to keep weights in (default: VXPU_MODELSTORE_URL, else http://modelstore.modelstore if it resolves, else the router caches weights itself)")
	externalModelstoreFlag := flag.String("modelstore-external-url", "",
		"modelstore's address for external executors, which cannot resolve its in-cluster Service (default: VXPU_MODELSTORE_EXTERNAL_URL; unset, they get --modelstore's URL)")
	telemetryEvery := flag.Int("telemetry-every", 1,
		"for models served by name, record execution telemetry on every Nth request (0: none); every request's statistics are recorded")
	externalFlag := flag.String("external-executor", "",
		"comma-separated host:port of executors outside Kubernetes (GPU VMs, Macs) that models registered by name are placed on (default: VXPU_EXTERNAL_EXECUTORS)")
	flag.Parse()

	ctx := context.Background()
	log := klog.FromContext(ctx)

	// If image is not set via flag, fallback to environment variable
	imageName := *imageFlag
	if imageName == "" {
		imageName = os.Getenv("VXPU_EXECUTOR_IMAGE")
	}
	if imageName == "" {
		log.Error(errors.New("no executor image configured; set --image or VXPU_EXECUTOR_IMAGE"), "fatal configuration error")
		os.Exit(1)
	}

	accelerator := *acceleratorFlag
	if accelerator == "" {
		accelerator = os.Getenv("VXPU_ACCELERATOR")
	}
	modelstoreURL := *modelstoreFlag
	if modelstoreURL == "" {
		modelstoreURL = os.Getenv("VXPU_MODELSTORE_URL")
	}

	// Load Kubeconfig
	var config *rest.Config
	var err error
	if kubeconfig := os.Getenv("KUBECONFIG"); kubeconfig != "" {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	} else if config, err = rest.InClusterConfig(); err == nil {
		// successfully loaded in-cluster config
	} else {
		home := homedir.HomeDir()
		if home != "" {
			config, err = clientcmd.BuildConfigFromFlags("", filepath.Join(home, ".kube", "config"))
		} else {
			err = fmt.Errorf("neither in-cluster config nor local kubeconfig found")
		}
	}
	if err != nil {
		log.Error(err, "failed to load kubeconfig")
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Error(err, "failed to create kubernetes clientset")
		os.Exit(1)
	}

	namespace := *nsFlag
	if namespace == "" {
		if nsBytes, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
			namespace = string(nsBytes)
		} else {
			namespace = "default"
		}
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *portFlag))
	if err != nil {
		log.Error(err, "failed to listen")
		os.Exit(1)
	}

	maxMsgSize := 128 * 1024 * 1024
	s := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
		// LoadModel can run for minutes (caching tens of GB) and
		// clients often sit behind kubectl port-forward; let them ping
		// to keep the tunnel alive instead of closing with GOAWAY
		// too_many_pings (the executor accepts the same cadence).
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)

	srv := router.NewServer(clientset, namespace, imageName, accelerator)
	srv.UseWeightStore(weightstore.Discover(ctx, modelstoreURL))
	external := *externalFlag
	if external == "" {
		external = os.Getenv("VXPU_EXTERNAL_EXECUTORS")
	}
	var externalAddrs []string
	for _, addr := range strings.Split(external, ",") {
		if addr = strings.TrimSpace(addr); addr != "" {
			externalAddrs = append(externalAddrs, addr)
		}
	}
	srv.UseExternalExecutors(ctx, externalAddrs)
	externalMS := *externalModelstoreFlag
	if externalMS == "" {
		externalMS = os.Getenv("VXPU_MODELSTORE_EXTERNAL_URL")
	}
	srv.UseExternalModelstoreURL(externalMS)
	if !router.PinnedByDigest(imageName) {
		log.Info("The executor image is not pinned by digest: compiled artifacts are keyed by its tag, so a rebuilt "+
			"tag reuses artifacts the previous build compiled", "image", imageName)
	}
	srv.UseStats(*telemetryEvery)
	pb.RegisterEngineServer(s, srv)

	if *adminPortFlag != 0 {
		adminLis, err := net.Listen("tcp", fmt.Sprintf(":%d", *adminPortFlag))
		if err != nil {
			log.Error(err, "failed to listen on admin port", "port", *adminPortFlag)
			os.Exit(1)
		}
		adminServer := grpc.NewServer(
			grpc.MaxRecvMsgSize(maxMsgSize),
			grpc.MaxSendMsgSize(maxMsgSize),
			grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
				MinTime:             10 * time.Second,
				PermitWithoutStream: true,
			}),
		)
		pb.RegisterExecutorServer(adminServer, srv)
		go func() {
			log.Info("Starting vxpu-router admin gRPC server", "port", *adminPortFlag)
			if err := adminServer.Serve(adminLis); err != nil {
				log.Error(err, "admin server failed")
			}
		}()
	}

	// On SIGTERM (a pod being deleted or redeployed) write the buffered
	// statistics before exiting, within the pod's grace period.
	stopping, stopSignals := signal.NotifyContext(ctx, syscall.SIGTERM, os.Interrupt)
	defer stopSignals()
	go func() {
		<-stopping.Done()
		flushCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := srv.FlushStats(flushCtx); err != nil {
			log.Error(err, "flushing statistics on shutdown")
		}
		if err := srv.Close(); err != nil {
			log.Error(err, "closing router executor connections on shutdown")
		}
		s.Stop()
	}()

	log.Info("Starting vxpu-router gRPC server", "port", *portFlag, "namespace", namespace, "image", imageName,
		"accelerator", accelerator, "externalExecutors", externalAddrs)
	if err := s.Serve(lis); err != nil {
		log.Error(err, "failed to serve")
		os.Exit(1)
	}
}
