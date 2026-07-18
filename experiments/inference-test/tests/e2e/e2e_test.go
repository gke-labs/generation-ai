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

package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "inference-test-e2e")
	h.Setup()

	if h.ClusterName == "inference-test-e2e" || strings.Contains(h.ClusterName, "kind") {
		t.Skip("Skipping test in Kind cluster as inference tests require GPU")
	}

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/inference-test")
	modelstoreRoot := filepath.Join(gitRoot, "modelstore")

	// Build images
	h.DockerBuild("inference-test:e2e", filepath.Join(experimentRoot, "images/inference-test/Dockerfile"), experimentRoot)
	h.DockerBuild("modelstore:e2e", filepath.Join(modelstoreRoot, "images/modelstore/Dockerfile"), modelstoreRoot)

	// Load images into Kind
	h.KindLoad("inference-test:e2e")
	h.KindLoad("modelstore:e2e")

	// Read modelstore manifest
	msManifestPath := filepath.Join(modelstoreRoot, "k8s/manifest.yaml")
	msb, err := os.ReadFile(msManifestPath)
	if err != nil {
		t.Fatalf("Failed to read modelstore manifest: %v", err)
	}
	msManifest := string(msb)
	msManifest = strings.ReplaceAll(msManifest, "image: modelstore:latest", "image: modelstore:e2e\n          imagePullPolicy: Never")

	// Apply Modelstore CRD
	crdPath := filepath.Join(modelstoreRoot, "k8s/crds/generationai.labs.gke.io_models.yaml")
	h.RunCommand("kubectl", "apply", "-f", crdPath)

	// Deploy modelstore
	h.DeleteStatefulSet("modelstore", "modelstore")
	h.DeleteService("modelstore", "modelstore")

	h.KubectlApplyContent("modelstore", msManifest)
	if err := h.WaitForStatefulSet("modelstore", "modelstore", 2*time.Minute); err != nil {
		fmt.Fprintf(os.Stderr, "Modelstore failed to start: %v\n", err)
		fmt.Fprintf(os.Stderr, "Modelstore Pod YAML:\n%s\n", h.GetPodYaml("app=modelstore", "modelstore"))
		fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("modelstore"))
		t.Fatalf("Modelstore failed to start: %v", err)
	}

	// Upload the model to modelstore
	uploadJobPath := filepath.Join(modelstoreRoot, "examples/upload-job.yaml")
	ujb, err := os.ReadFile(uploadJobPath)
	if err != nil {
		t.Fatalf("Failed to read upload job: %v", err)
	}
	uploadJob := string(ujb)
	uploadJob = strings.ReplaceAll(uploadJob, "image: modelstore:latest", "image: modelstore:e2e\n          imagePullPolicy: Never")

	h.DeleteJob("opt-125m", "modelstore")
	h.KubectlApplyContent("model-upload", uploadJob, "-n", "modelstore")

	// Wait for upload job
	err = h.WaitForJobSuccess("opt-125m", "modelstore", 10*time.Minute)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Model upload logs:\n%s\n", h.GetPodLogs("batch.kubernetes.io/job-name=opt-125m", "modelstore"))
		t.Fatalf("Model upload job failed: %v", err)
	}

	// 1. Run Single-node CPU Test
	t.Run("SingleNodeCPU", func(t *testing.T) {
		h.DeleteJob("inference-test-single", "default")

		manifestPath := filepath.Join(experimentRoot, "examples/simple/manifest.yaml")
		b, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("Failed to read manifest: %v", err)
		}
		manifest := string(b)
		manifest = strings.ReplaceAll(manifest, "name: inference-test", "name: inference-test-single")
		manifest = strings.ReplaceAll(manifest, "image: inference-test:latest", "image: inference-test:e2e\n        imagePullPolicy: Never")
		manifest = strings.ReplaceAll(manifest, "facebook/opt-125m", "opt-125m")

		// Remove GPU requirement for CPU test
		manifest = strings.ReplaceAll(manifest, "nvidia.com/gpu: 1", "cpu: \"500m\"")
		manifest = strings.ReplaceAll(manifest, "cloud.google.com/gke-accelerator: nvidia-l4", "")

		// Add small memory limit
		manifest = strings.ReplaceAll(manifest, "resources:", "resources:\n          requests:\n            memory: \"4Gi\"\n          limits:\n            memory: \"8Gi\"")

		h.KubectlApplyContent("inference-test-single", manifest)
		err = h.WaitForJobSuccess("inference-test-single", "default", 10*time.Minute)
		logs := h.GetPodLogs("batch.kubernetes.io/job-name=inference-test-single", "default")
		if err != nil {
			msLogs := h.GetPodLogs("app=modelstore", "modelstore")
			fmt.Fprintf(os.Stderr, "Modelstore logs:\n%s\n", msLogs)
			fmt.Fprintf(os.Stderr, "Modelstore Pod YAML:\n%s\n", h.GetPodYaml("app=modelstore", "modelstore"))
			fmt.Fprintf(os.Stderr, "Single-node logs:\n%s\n", logs)
			fmt.Fprintf(os.Stderr, "Single-node Pod YAML:\n%s\n", h.GetPodYaml("batch.kubernetes.io/job-name=inference-test-single", "default"))
			fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("default"))
			t.Fatalf("Job failed: %v", err)
		}
		t.Logf("Single-node logs:\n%s", logs)
		if !strings.Contains(logs, "Tokens generated:") {
			t.Error("Logs do not contain expected metrics")
		}
	})

	// 2. Run Distributed CPU Test (FSDP requested but fallback)
	t.Run("DistributedCPU", func(t *testing.T) {
		h.DeleteJob("inference-test", "default")
		h.DeleteService("inference-test-headless", "default")

		manifestPath := filepath.Join(experimentRoot, "examples/distributed/manifest.yaml")
		b, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Log("manifest.yaml not found in distributed examples, skipping distributed test")
			return
		}
		manifest := string(b)
		manifest = strings.ReplaceAll(manifest, "image: inference-test:latest", "image: inference-test:e2e\n        imagePullPolicy: Never")

		// Tweak for CPU
		manifest = strings.ReplaceAll(manifest, "nvidia.com/gpu: 1", "cpu: \"500m\"")
		manifest = strings.ReplaceAll(manifest, "cloud.google.com/gke-accelerator: nvidia-l4", "")

		// Use smaller model for E2E
		manifest = strings.ReplaceAll(manifest, "facebook/opt-125m", "opt-125m")

		// Add small memory limit
		manifest = strings.ReplaceAll(manifest, "resources:", "resources:\n          requests:\n            memory: \"4Gi\"\n          limits:\n            memory: \"8Gi\"")

		h.KubectlApplyContent("inference-test-distributed", manifest)
		err = h.WaitForJobSuccess("inference-test", "default", 10*time.Minute)
		logs := h.GetPodLogs("app=inference-test", "default")
		if err != nil {
			msLogs := h.GetPodLogs("app=modelstore", "modelstore")
			fmt.Fprintf(os.Stderr, "Modelstore logs:\n%s\n", msLogs)
			fmt.Fprintf(os.Stderr, "Modelstore Pod YAML:\n%s\n", h.GetPodYaml("app=modelstore", "modelstore"))
			fmt.Fprintf(os.Stderr, "Distributed logs:\n%s\n", logs)
			fmt.Fprintf(os.Stderr, "Distributed Pod YAML:\n%s\n", h.GetPodYaml("app=inference-test", "default"))
			fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("default"))
			t.Fatalf("Job failed: %v", err)
		}
		t.Logf("Distributed logs:\n%s", logs)
		if !strings.Contains(logs, "Tokens generated:") {
			t.Error("Logs do not contain expected metrics")
		}
	})
}
