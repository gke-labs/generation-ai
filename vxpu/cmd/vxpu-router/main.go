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
	"net"
	"os"
	"path/filepath"
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
)

func main() {
	klog.InitFlags(nil)
	defer klog.Flush()

	portFlag := flag.Int("port", 50051, "The server port")
	imageFlag := flag.String("image", "", "The executor image to run")
	nsFlag := flag.String("namespace", "", "The Kubernetes namespace to use")
	acceleratorFlag := flag.String("accelerator", "", "GKE accelerator label for the executor pod (e.g., nvidia-l4)")
	flag.Parse()

	ctx := context.Background()
	log := klog.FromContext(ctx)

	// If image is not set via flag, fallback to environment variable
	imageName := *imageFlag
	if imageName == "" {
		imageName = os.Getenv("VXPU_EXECUTOR_IMAGE")
	}

	accelerator := *acceleratorFlag
	if accelerator == "" {
		accelerator = os.Getenv("VXPU_ACCELERATOR")
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
	pb.RegisterExecutorServer(s, srv)

	log.Info("Starting vxpu-router gRPC server", "port", *portFlag, "namespace", namespace, "image", imageName, "accelerator", accelerator)
	if err := s.Serve(lis); err != nil {
		log.Error(err, "failed to serve")
		os.Exit(1)
	}
}
