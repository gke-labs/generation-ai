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

// vxpu-pki-init mints the vXPU mutual TLS certificate authority and the
// coordinator and agent SPIFFE certificates. By default it runs as a
// Kubernetes Job that reconciles Secrets and ConfigMaps. With --out-dir, it
// writes the certificates to a local directory for a single machine with no
// cluster.
package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func main() {
	ctx := context.Background()
	if err := run(ctx, os.Args[1:]); err != nil {
		klog.FromContext(ctx).Error(err, "vxpu-pki-init failed")
		klog.Flush()
		os.Exit(1)
	}
	klog.Flush()
}

func run(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("vxpu-pki-init", flag.ContinueOnError)
	klog.InitFlags(fs)

	var opts pki.Options
	fs.StringVar(&opts.CANamespace, "ca-namespace", pki.DefaultCANamespace, "Kubernetes namespace holding the vxpu-ca Secret")
	fs.StringVar(&opts.WorkloadNamespace, "workload-namespace", pki.DefaultWorkloadNamespace, "Kubernetes namespace holding the CA bundle and workload Secrets")
	fs.StringVar(&opts.TrustDomain, "trust-domain", pki.DefaultTrustDomain, "SPIFFE trust domain for leaf certificates")
	fs.DurationVar(&opts.CATTL, "ca-ttl", pki.DefaultCATTL, "lifetime of a newly minted CA certificate")
	fs.DurationVar(&opts.LeafTTL, "leaf-ttl", pki.DefaultLeafTTL, "lifetime of newly minted leaf certificates")
	outDir := fs.String("out-dir", "", "write CA, coordinator, and agent files to this directory instead of Kubernetes")
	kubeconfig := fs.String("kubeconfig", "", "path to a kubeconfig file (uses in-cluster config when empty)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %v", fs.Args())
	}

	outDirSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "out-dir" {
			outDirSet = true
		}
	})
	if outDirSet {
		if *kubeconfig != "" {
			return fmt.Errorf("--out-dir and --kubeconfig cannot be used together")
		}
		return pki.EnsureDir(ctx, *outDir, opts)
	}

	cfg, err := loadRestConfig(*kubeconfig)
	if err != nil {
		return err
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return fmt.Errorf("creating Kubernetes client: %w", err)
	}
	return pki.EnsureK8s(ctx, client, opts)
}

func loadRestConfig(kubeconfig string) (*rest.Config, error) {
	if kc := cmp.Or(kubeconfig, os.Getenv("KUBECONFIG")); kc != "" {
		return clientcmd.BuildConfigFromFlags("", kc)
	}
	return rest.InClusterConfig()
}
