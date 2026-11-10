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

	h := NewHarness(t, "finetuning-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/finetuning")
	modelstoreRoot := filepath.Join(gitRoot, "modelstore")

	// Build images
	h.DockerBuild("finetuning-server:e2e", filepath.Join(experimentRoot, "images/finetuning-server/Dockerfile"), gitRoot)
	h.DockerBuild("finetuning-client:e2e", filepath.Join(experimentRoot, "images/finetuning-client/Dockerfile"), gitRoot)
	h.DockerBuild("modelstore:e2e", filepath.Join(modelstoreRoot, "images/modelstore/Dockerfile"), gitRoot)

	// Load images into Kind
	h.KindLoad("finetuning-server:e2e")
	h.KindLoad("finetuning-client:e2e")
	h.KindLoad("modelstore:e2e")

	// Read modelstore manifest
	msManifestPath := filepath.Join(modelstoreRoot, "k8s/manifest.yaml")
	msb, err := os.ReadFile(msManifestPath)
	if err != nil {
		t.Fatalf("Failed to read modelstore manifest: %v", err)
	}
	msManifest := string(msb)
	msManifest = strings.ReplaceAll(msManifest, "image: modelstore:latest", "image: modelstore:e2e\n          imagePullPolicy: Never")
	// No GCS in kind: blobs go to a directory in the container.
	msManifest = strings.ReplaceAll(msManifest, "--blob-store=gs://BUCKET", "--blob-store=/tmp/blobs")

	// Read manifest and replace placeholders
	manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read manifest: %v", err)
	}
	jobManifestPath := filepath.Join(experimentRoot, "examples/simple/manifest.yaml")
	jb, err := os.ReadFile(jobManifestPath)
	if err != nil {
		t.Fatalf("Failed to read job manifest: %v", err)
	}

	manifest := string(b) + "\n---\n" + string(jb)
	manifest = strings.ReplaceAll(manifest, "image: finetuning-server:latest", "image: finetuning-server:e2e\n        imagePullPolicy: Never")
	manifest = strings.ReplaceAll(manifest, "image: finetuning-client:latest", "image: finetuning-client:e2e\n        imagePullPolicy: Never")
	manifest = strings.ReplaceAll(manifest, "--max_steps\", \"5\"", "--max_steps\", \"5\", \"--model_id\", \"opt-125m\"")

	// Add MODELSTORE_URL to server
	envVar := `        env:
        - name: MODELSTORE_URL
          value: http://modelstore.modelstore`
	manifest = strings.ReplaceAll(manifest, "- name: server", "- name: server\n"+envVar)

	// Reduce resource requirements for E2E
	manifest = strings.ReplaceAll(manifest, "cpu: \"2\"", "cpu: \"500m\"")
	manifest = strings.ReplaceAll(manifest, "memory: \"8Gi\"", "memory: \"4Gi\"")
	manifest = strings.ReplaceAll(manifest, "cpu: \"4\"", "cpu: \"1\"")
	manifest = strings.ReplaceAll(manifest, "memory: \"16Gi\"", "memory: \"8Gi\"")

	// Apply manifests
	h.DeleteJob("finetuning-client", "default")
	h.DeleteDeployment("finetuning-server", "default")
	h.DeleteService("finetuning-server", "default")
	h.DeleteDeployment("modelstore", "modelstore")
	h.DeleteService("modelstore", "modelstore")

	h.KubectlApplyContent("modelstore", msManifest)

	// Wait for modelstore
	if err := h.WaitForDeployment("modelstore", "modelstore", 2*time.Minute); err != nil {
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
	err = h.WaitForJobSuccess("opt-125m", "modelstore", 10*time.Minute) // Might take time to download
	if err != nil {
		t.Logf("Model upload logs:\n%s", h.GetPodLogs("batch.kubernetes.io/job-name=opt-125m", "modelstore"))
		t.Fatalf("Model upload job failed: %v", err)
	}

	h.KubectlApplyContent("finetuning", manifest)

	// Wait for server
	if err := h.WaitForDeployment("finetuning-server", "default", 5*time.Minute); err != nil {
		fmt.Fprintf(os.Stderr, "Finetuning server failed to start: %v\n", err)
		fmt.Fprintf(os.Stderr, "Finetuning server Pod YAML:\n%s\n", h.GetPodYaml("app=finetuning-server", "default"))
		fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("default"))
		t.Fatalf("Finetuning server failed to start: %v", err)
	}

	// Wait for client job
	err = h.WaitForJobSuccess("finetuning-client", "default", 10*time.Minute)

	// Check logs (always, even on failure)
	msLogs := h.GetPodLogs("app=modelstore", "modelstore")
	fmt.Fprintf(os.Stderr, "Modelstore logs:\n%s\n", msLogs)
	t.Logf("Modelstore logs:\n%s", msLogs)

	logs := h.GetPodLogs("app=finetuning-server", "default")
	fmt.Fprintf(os.Stderr, "Server logs:\n%s\n", logs)
	t.Logf("Server logs:\n%s", logs)

	clientLogs := h.GetPodLogs("batch.kubernetes.io/job-name=finetuning-client", "default")
	fmt.Fprintf(os.Stderr, "Client logs:\n%s\n", clientLogs)
	t.Logf("Client logs:\n%s", clientLogs)

	if err != nil {
		t.Fatalf("Job failed: %v", err)
	}

	if !strings.Contains(clientLogs, "Fine-tuning completed successfully") {
		t.Error("Client logs do not indicate successful completion")
	}
}
