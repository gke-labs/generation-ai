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
	"bytes"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

type Harness struct {
	ClusterName string
	t           *testing.T
}

func NewHarness(t *testing.T, clusterName string) *Harness {
	return &Harness{
		ClusterName: clusterName,
		t:           t,
	}
}

func (h *Harness) Setup() {
	h.t.Helper()
	// Check if cluster exists
	cmd := exec.Command("kind", "get", "clusters")
	out, err := cmd.Output()
	if err == nil && strings.Contains(string(out), h.ClusterName) {
		h.t.Logf("Cluster %s already exists", h.ClusterName)
		h.RunCommand("kind", "export", "kubeconfig", "--name", h.ClusterName)
	} else {
		h.t.Logf("Creating cluster %s", h.ClusterName)
		cmd = exec.Command("kind", "create", "cluster", "--name", h.ClusterName)
		if out, err := cmd.CombinedOutput(); err != nil {
			h.t.Fatalf("Failed to create cluster: %v\nOutput: %s", err, out)
		}
	}

	// Ensure default namespace is used, avoiding issues with environment-specific defaults
	h.RunCommand("kubectl", "config", "set-context", "--current", "--namespace=default")

	h.t.Cleanup(func() {
		h.Teardown()
	})
}

func (h *Harness) Teardown() {
	h.t.Helper()
	h.t.Logf("Deleting cluster %s", h.ClusterName)
	cmd := exec.Command("kind", "delete", "cluster", "--name", h.ClusterName)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Logf("Failed to delete cluster: %v\nOutput: %s", err, out)
	}
}

func (h *Harness) GetGitRoot() string {
	h.t.Helper()
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	out, err := cmd.Output()
	if err != nil {
		h.t.Fatalf("Failed to find git root: %v", err)
	}
	return strings.TrimSpace(string(out))
}

func (h *Harness) RunCommand(name string, args ...string) {
	h.t.Helper()
	cmd := exec.Command(name, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Command failed: %s %v\nOutput: %s", name, args, out)
	}
}

func (h *Harness) DockerBuild(tag, dockerfile, context string) {
	h.t.Helper()
	h.t.Logf("Building docker image %s", tag)
	h.RunCommand("docker", "build", "-t", tag, "-f", dockerfile, context)
}

func (h *Harness) KindLoad(tag string) {
	h.t.Helper()
	h.t.Logf("Loading image %s into kind", tag)
	h.RunCommand("kind", "load", "docker-image", tag, "--name", h.ClusterName)
}

func (h *Harness) KubectlApplyContent(name, content string, args ...string) {
	h.t.Helper()
	snippet := content
	if len(snippet) > 100 {
		snippet = snippet[:100] + "..."
	}
	h.t.Logf("Applying manifest content for %s:\n%s", name, snippet)
	cmdArgs := append([]string{"apply", "-f", "-"}, args...)
	cmd := exec.Command("kubectl", cmdArgs...)
	cmd.Stdin = bytes.NewBufferString(content)
	if out, err := cmd.CombinedOutput(); err != nil {
		h.t.Fatalf("Failed to apply content for %s: %v\nOutput: %s\nFull manifest:\n%s", name, err, out, content)
	}
}

func (h *Harness) WaitForPodReady(name, namespace string, timeout time.Duration) error {
	h.t.Helper()
	h.t.Logf("Waiting for pod %s in namespace %s to be Ready", name, namespace)
	cmd := exec.Command("kubectl", "wait", "--for=condition=Ready", "pod/"+name, "-n", namespace, "--timeout="+timeout.String())
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("pod %s failed to become ready: %v\nOutput: %s", name, err, out)
	}
	return nil
}

func (h *Harness) DeletePod(name, namespace string) {
	h.t.Helper()
	out, err := exec.Command("kubectl", "delete", "pod", name, "-n", namespace, "--ignore-not-found").CombinedOutput()
	if err != nil {
		h.t.Logf("kubectl delete pod %s/%s failed (ignored, best-effort cleanup): %v: %s", namespace, name, err, out)
	}
}

func (h *Harness) GetPodLogs(labelSelector, namespace string) string {
	h.t.Helper()
	out, err := exec.Command("kubectl", "logs", "-l", labelSelector, "-n", namespace).CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get logs for selector %s in namespace %s: %v", labelSelector, namespace, err)
		return string(out)
	}
	return string(out)
}

func (h *Harness) GetPodYaml(labelSelector, namespace string) string {
	h.t.Helper()
	out, err := exec.Command("kubectl", "get", "pod", "-l", labelSelector, "-n", namespace, "-o", "yaml").CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get pod yaml for selector %s in namespace %s: %v", labelSelector, namespace, err)
		return string(out)
	}
	return string(out)
}

func (h *Harness) GetEvents(namespace string) string {
	h.t.Helper()
	cmdArgs := []string{"get", "events", "--sort-by=.lastTimestamp"}
	if namespace != "" {
		cmdArgs = append(cmdArgs, "-n", namespace)
	}
	out, err := exec.Command("kubectl", cmdArgs...).CombinedOutput()
	if err != nil {
		h.t.Logf("Warning: failed to get events: %v", err)
		return string(out)
	}
	return string(out)
}
