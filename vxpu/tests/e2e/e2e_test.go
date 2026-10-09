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
)

func TestE2E(t *testing.T) {
	if os.Getenv("RUN_E2E") == "" {
		t.Skip("Skipping E2E test; RUN_E2E not set")
	}

	h := NewHarness(t, "vxpu-e2e")
	h.Setup()

	gitRoot := h.GetGitRoot()
	vxpuRoot := filepath.Join(gitRoot, "vxpu")
	artifactDir := filepath.Join(vxpuRoot, "tests/e2e/test_artifact")
	vxpuBin := filepath.Join(vxpuRoot, "tests/e2e/vxpu_bin")

	// Build the executor image
	h.DockerBuild("vxpu-executor:e2e", filepath.Join(vxpuRoot, "images/executor/Dockerfile"), vxpuRoot)

	// Build the router image
	h.DockerBuild("vxpu-router:e2e", filepath.Join(vxpuRoot, "images/router/Dockerfile"), vxpuRoot)

	// Load the executor and router images into Kind
	h.KindLoad("vxpu-executor:e2e")
	h.KindLoad("vxpu-router:e2e")

	// Export the model using the built container image to guarantee environment consistency
	h.t.Logf("Exporting model to %s", artifactDir)
	h.RunCommand("docker", "run", "--name", "exporter-temp", "--entrypoint", "python", "vxpu-executor:e2e", "-m", "vxpu.export", "trl-internal-testing/tiny-random-LlamaForCausalLM", "-o", "/tmp/test_artifact")
	h.RunCommand("mkdir", "-p", artifactDir)
	h.RunCommand("docker", "cp", "exporter-temp:/tmp/test_artifact/.", artifactDir)
	h.RunCommand("docker", "rm", "exporter-temp")

	// Build the vxpu Go CLI binary
	h.t.Logf("Building vxpu CLI binary to %s", vxpuBin)
	h.RunCommand("go", "build", "-o", vxpuBin, filepath.Join(vxpuRoot, "cmd/vxpu"))

	// Create Pod-managing RBAC for the router's ServiceAccount (default)
	podManagerRbacYaml := `apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-manager
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: default
  name: pod-manager-binding
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
roleRef:
  kind: Role
  name: pod-manager
  apiGroup: rbac.authorization.k8s.io`

	// Apply RBAC
	h.KubectlApplyContent("pod-manager-rbac", podManagerRbacYaml)

	// Clean any pre-existing pods
	h.DeletePod("vxpu-router", "default")
	// Clean any previous executor pods that may start with 'vxpu-executor-'
	out, err := exec.Command("kubectl", "get", "pods", "-o", "jsonpath={.items[*].metadata.name}").Output()
	if err == nil {
		for _, podName := range strings.Fields(string(out)) {
			if strings.HasPrefix(podName, "vxpu-executor-") {
				h.DeletePod(podName, "default")
			}
		}
	}

	// Run vxpu CLI ask command to launch router, which in turn launches executor and chats
	h.t.Log("Running vxpu ask command via the router")
	cmd := exec.Command(vxpuBin, "ask", "--artifact", artifactDir, "--accelerator", "none", "Is the sky blue?")
	// Set the environment variables so vxpu CLI knows which images to orchestrate
	cmd.Env = append(os.Environ(),
		"VXPU_ROUTER_IMAGE=vxpu-router:e2e",
		"VXPU_EXECUTOR_IMAGE=vxpu-executor:e2e",
	)

	out, err = cmd.CombinedOutput()

	// Get and print router logs
	routerLogs := h.GetPodLogs("app=vxpu-router", "default")
	fmt.Fprintf(os.Stderr, "vxpu-router logs:\n%s\n", routerLogs)
	t.Logf("vxpu-router logs:\n%s", routerLogs)

	// Get and print executor logs (any executor pod created by router starts with 'vxpu-executor-')
	executorLogs := h.GetPodLogs("app=vxpu-executor", "default")
	fmt.Fprintf(os.Stderr, "vxpu-executor logs:\n%s\n", executorLogs)
	t.Logf("vxpu-executor logs:\n%s", executorLogs)

	if err != nil {
		t.Fatalf("vxpu ask failed: %v\nOutput: %s", err, string(out))
	}

	outputStr := string(out)
	fmt.Fprintf(os.Stderr, "vxpu ask output:\n%s\n", outputStr)
	t.Logf("vxpu ask output:\n%s", outputStr)

	if !strings.Contains(outputStr, "Is the sky blue?") {
		t.Error("CLI output does not contain the prompt")
	}

	// Bringing the router up again must be a no-op that keeps the
	// existing pod (and its cache) and still answers: `vxpu up` applies
	// the manifest every time, so Service/RBAC changes reach a cluster
	// that already has a router.
	h.t.Log("Running vxpu up and a second vxpu ask against the existing router")
	up := exec.Command(vxpuBin, "up")
	up.Env = cmd.Env
	if out, err := up.CombinedOutput(); err != nil {
		t.Fatalf("vxpu up failed: %v\nOutput: %s", err, string(out))
	}
	again := exec.Command(vxpuBin, "ask", "--artifact", artifactDir, "--accelerator", "none", "Is the sky blue?")
	again.Env = cmd.Env
	out, err = again.CombinedOutput()
	if err != nil {
		t.Fatalf("second vxpu ask failed: %v\nOutput: %s", err, string(out))
	}
	if !strings.Contains(string(out), "Is the sky blue?") {
		t.Error("second CLI output does not contain the prompt")
	}
	if !strings.Contains(string(out), "already loaded") && !strings.Contains(string(out), "model loaded in") {
		t.Logf("second ask output: %s", out)
	}

	// `vxpu down` removes the router and everything the manifest created.
	down := exec.Command(vxpuBin, "down")
	if out, err := down.CombinedOutput(); err != nil {
		t.Fatalf("vxpu down failed: %v\nOutput: %s", err, string(out))
	}
	for _, kind := range []string{"pod", "service", "serviceaccount", "role", "rolebinding"} {
		if exec.Command("kubectl", "get", kind, "vxpu-router").Run() == nil {
			t.Errorf("%s vxpu-router still exists after vxpu down", kind)
		}
	}
}
