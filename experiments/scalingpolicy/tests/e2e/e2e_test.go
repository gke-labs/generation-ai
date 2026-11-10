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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeMetricsToCSV(t *testing.T, namespace, scenario string) {
	// Give it a moment to collect final stats
	time.Sleep(5 * time.Second)
	cmd := exec.Command("kubectl", "exec", "deployment/memcache-client", "-n", namespace, "--", "curl", "-s", "http://memcache-service:8080/metrics")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("Failed to get metrics for %s: %v\nOutput: %s", scenario, err, out)
		return
	}

	hits := "0"
	misses := "0"
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "memcached_get_hits ") {
			hits = strings.TrimSpace(strings.TrimPrefix(line, "memcached_get_hits "))
		}
		if strings.HasPrefix(line, "memcached_get_misses ") {
			misses = strings.TrimSpace(strings.TrimPrefix(line, "memcached_get_misses "))
		}
	}

	artifactsDir := os.Getenv("ARTIFACTS")
	if artifactsDir == "" {
		artifactsDir = "/tmp"
	}

	// Create test-specific artifacts directory
	testArtifactsDir := filepath.Join(artifactsDir, t.Name())
	if err := os.MkdirAll(testArtifactsDir, 0755); err != nil {
		t.Logf("Failed to create test artifacts directory: %v", err)
		return
	}

	csvPath := filepath.Join(testArtifactsDir, "metrics.csv")

	// Write header if file doesn't exist
	if _, err := os.Stat(csvPath); os.IsNotExist(err) {
		os.WriteFile(csvPath, []byte("timestamp,scenario,hits,misses\n"), 0644)
	}

	f, err := os.OpenFile(csvPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Logf("Failed to open csv file: %v", err)
		return
	}
	defer f.Close()

	timestamp := time.Now().Format(time.RFC3339)
	statLine := fmt.Sprintf("%s,%s,%s,%s\n", timestamp, scenario, hits, misses)
	if _, err := f.WriteString(statLine); err != nil {
		t.Logf("Failed to write to csv: %v", err)
	}
}

func deployMemcacheApp(t *testing.T, namespace string, manifestExtras string) {
	manifestPath := "testdata/memcached/manifest.yaml"
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read base manifest: %v", err)
	}

	baseManifest := string(b)
	// Replace the placeholder namespace with the target namespace
	baseManifest = strings.ReplaceAll(baseManifest, "memcached-test", namespace)

	finalManifest := baseManifest + "\n" + manifestExtras

	// Apply manifest using the child t to avoid subtest calling parent FailNow
	cmd := exec.Command("kubectl", "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(finalManifest)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to apply manifest: %v\nOutput: %s", err, out)
	}

	// Wait for deployment to be ready
	cmd = exec.Command("kubectl", "rollout", "status", "deployment/memcache-deployment", "-n", namespace, "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("memcache-deployment failed to start: %v\nOutput: %s", err, out)
	}

	cmd = exec.Command("kubectl", "rollout", "status", "deployment/memcache-client", "-n", namespace, "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("memcache-client failed to start: %v\nOutput: %s", err, out)
	}
}

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "scalingpolicy-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	experimentRoot := filepath.Join(gitRoot, "experiments/scalingpolicy")

	// Build image
	h.DockerBuild("scalingpolicy-controller:e2e", filepath.Join(experimentRoot, "images/scalingpolicy-controller/Dockerfile"), gitRoot)
	h.KindLoad("scalingpolicy-controller:e2e")

	h.DockerBuild("test-memcache-server:e2e", filepath.Join(experimentRoot, "images/test-memcache-server/Dockerfile"), gitRoot)
	h.KindLoad("test-memcache-server:e2e")

	h.DockerBuild("test-memcache-client:e2e", filepath.Join(experimentRoot, "images/test-memcache-client/Dockerfile"), gitRoot)
	h.KindLoad("test-memcache-client:e2e")

	// Apply CRD
	crdPath := filepath.Join(experimentRoot, "k8s/crds/scalingpolicy.yaml")
	b, err := os.ReadFile(crdPath)
	if err != nil {
		t.Fatalf("Failed to read CRD manifest: %v", err)
	}
	h.KubectlApplyContent("scalingpolicy-crd", string(b))

	// Apply manifest
	manifestPath := filepath.Join(experimentRoot, "k8s/manifest.yaml")
	b, err = os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("Failed to read manifest: %v", err)
	}
	manifest := string(b)
	manifest = strings.ReplaceAll(manifest, "image: scalingpolicy-controller:latest", "image: scalingpolicy-controller:e2e\n        imagePullPolicy: Never")
	h.KubectlApplyContent("scalingpolicy-manifest", manifest)

	// Install metrics-server
	h.InstallMetricsServer(t)

	// Install VPA
	h.InstallVPA(t)

	if err := h.WaitForStatefulSet("scalingpolicy-controller", "kube-scalingpolicy-system", 2*time.Minute); err != nil {
		t.Fatalf("ScalingPolicy Controller failed to start: %v", err)
	}
	t.Log("ScalingPolicy controller started successfully.")

	t.Run("ScalingPolicy", func(t *testing.T) {
		ns := "test-sp"
		policy := `---
apiVersion: generation-ai.gke.io/v1alpha1
kind: ScalingPolicy
metadata:
  name: memcache-policy
  namespace: test-sp
spec:
  target:
    kind: Deployment
    name: memcache-deployment
  inputs:
  - name: pod_memory
    metric: memory
  values:
  - path: spec.template.spec.containers[0].resources.limits.memory
    expression: "pod_memory + 128 * 1024 * 1024"
    min: 67108864 # 64MiB
    max: 536870912 # 512MiB
`
		deployMemcacheApp(t, ns, policy)

		sink := NewMetricsSink(t)
		sink.StartCollecting(t, ns)

		deadline := time.Now().Add(5 * time.Minute)
		success := false
		for time.Now().Before(deadline) {
			cmd := exec.Command("sh", "-c", "kubectl get pods -l app=memcache -n "+ns+" -o jsonpath='{.items[0].spec.containers[0].resources.limits.memory}'")
			out, err := cmd.Output()
			if err == nil {
				limitStr := string(out)
				if limitStr == "512Mi" || limitStr == "536870912" {
					success = true
					break
				}
			}
			time.Sleep(5 * time.Second)
		}
		if !success {
			t.Log("Warning: Deployment memory limit did not reach 512Mi")
		}
		writeMetricsToCSV(t, ns, "ScalingPolicy")
	})

	t.Run("VPA", func(t *testing.T) {
		ns := "test-vpa"
		vpa := `---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: memcache-vpa
  namespace: test-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind:       Deployment
    name:       memcache-deployment
  updatePolicy:
    updateMode: "Auto"
    minReplicas: 1
  resourcePolicy:
    containerPolicies:
    - containerName: "*"
      controlledResources: ["memory"]
      controlledValues: "RequestsAndLimits"
`
		deployMemcacheApp(t, ns, vpa)

		sink := NewMetricsSink(t)
		sink.StartCollecting(t, ns)

		// Let it run for 2 minutes to gather metrics and potentially scale
		time.Sleep(2 * time.Minute)
		writeMetricsToCSV(t, ns, "VPA")
	})

	t.Run("Fixed", func(t *testing.T) {
		ns := "test-fixed"
		// No extra policy or VPA
		deployMemcacheApp(t, ns, "")

		sink := NewMetricsSink(t)
		sink.StartCollecting(t, ns)

		// Let it run for 2 minutes to gather metrics
		time.Sleep(2 * time.Minute)
		writeMetricsToCSV(t, ns, "Fixed")
	})
}
