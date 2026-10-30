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
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

	// The equivalence harness's CI tier: tiny models exported from a
	// local directory and from the Hub, run on the in-process engine,
	// token-exact against eager transformers (vxpu/python/vxpu/vectors_test.py).
	h.t.Log("Running the test-vectors selftest in the executor image")
	h.RunCommand("docker", "run", "--rm", "--entrypoint", "python", "vxpu-executor:e2e",
		"-m", "vxpu.vectors_test", "selftest", "--workdir", "/tmp/vectors")

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

	testByName(t, h, gitRoot, vxpuBin, cmd.Env)

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

	// Verify vxpu up, then vxpu up again with the same image preserves pod UID, then vxpu down.
	h.t.Log("Verifying vxpu up preserves pod UID across updates with the same image")
	upFirst := exec.Command(vxpuBin, "up")
	upFirst.Env = cmd.Env
	if out, err := upFirst.CombinedOutput(); err != nil {
		t.Fatalf("first vxpu up failed: %v\nOutput: %s", err, string(out))
	}
	uidFirst, err := exec.Command("kubectl", "get", "pod", "vxpu-router", "-o", "jsonpath={.metadata.uid}").CombinedOutput()
	if err != nil {
		t.Fatalf("getting pod UID after first vxpu up: %v (%s)", err, string(uidFirst))
	}
	upSecond := exec.Command(vxpuBin, "up")
	upSecond.Env = cmd.Env
	if out, err := upSecond.CombinedOutput(); err != nil {
		t.Fatalf("second vxpu up failed: %v\nOutput: %s", err, string(out))
	}
	uidSecond, err := exec.Command("kubectl", "get", "pod", "vxpu-router", "-o", "jsonpath={.metadata.uid}").CombinedOutput()
	if err != nil {
		t.Fatalf("getting pod UID after second vxpu up: %v (%s)", err, string(uidSecond))
	}
	if string(uidFirst) != string(uidSecond) {
		t.Errorf("pod UID changed across vxpu up with same image: before=%s after=%s", uidFirst, uidSecond)
	}
	downFinal := exec.Command(vxpuBin, "down")
	if out, err := downFinal.CombinedOutput(); err != nil {
		t.Fatalf("final vxpu down failed: %v\nOutput: %s", err, string(out))
	}
}

// externalExecutorManifest is an executor the router does not manage: in
// production a GPU VM or a Mac running `python -m vxpu.server`; here a
// pod and Service the test creates (#357).
const externalExecutorManifest = `apiVersion: v1
kind: Pod
metadata:
  name: vxpu-external-executor
  labels:
    app: vxpu-external-executor
spec:
  containers:
    - name: executor
      image: vxpu-executor:e2e
      imagePullPolicy: Never
      args: ["--keep-alive", "3600"]
      ports:
        - containerPort: 50051
---
apiVersion: v1
kind: Service
metadata:
  name: vxpu-external-executor
spec:
  selector:
    app: vxpu-external-executor
  ports:
    - port: 50051
      targetPort: 50051
`

// dumpByNameState prints what testByName's commands depend on: the
// router, the compile Jobs and their pods, the external executor and
// modelstore, with the events of both namespaces.
func dumpByNameState(h *Harness) {
	state, _ := exec.Command("kubectl", "get", "jobs,pods", "-A", "-o", "wide").CombinedOutput()
	fmt.Fprintf(os.Stderr, "jobs and pods:\n%s\n", state)
	fmt.Fprintf(os.Stderr, "router logs:\n%s\n", h.GetPodLogs("app=vxpu-router", "default"))
	fmt.Fprintf(os.Stderr, "compile pods:\n%s\n", h.GetPodYaml("app=vxpu-compile", "default"))
	fmt.Fprintf(os.Stderr, "compile logs:\n%s\n", h.GetPodLogs("app=vxpu-compile", "default"))
	fmt.Fprintf(os.Stderr, "external executor logs:\n%s\n", h.GetPodLogs("app=vxpu-external-executor", "default"))
	fmt.Fprintf(os.Stderr, "modelstore logs:\n%s\n", h.GetPodLogs("app=modelstore", "modelstore"))
	fmt.Fprintf(os.Stderr, "events (default):\n%s\n", h.GetEvents("default"))
	fmt.Fprintf(os.Stderr, "events (modelstore):\n%s\n", h.GetEvents("modelstore"))
}

// testByName registers a model by name (the router imports it into
// modelstore and compiles it in a Job with the executor image) and asks
// it a question by name, served by an external executor (#357).
func testByName(t *testing.T, h *Harness, gitRoot, vxpuBin string, env []string) {
	const model = "trl-internal-testing/tiny-random-LlamaForCausalLM"
	modelstoreRoot := filepath.Join(gitRoot, "modelstore")
	h.DockerBuild("modelstore:e2e", filepath.Join(modelstoreRoot, "images/modelstore/Dockerfile"), modelstoreRoot)
	h.KindLoad("modelstore:e2e")
	msManifest, err := os.ReadFile(filepath.Join(modelstoreRoot, "k8s/manifest.yaml"))
	if err != nil {
		t.Fatalf("reading the modelstore manifest: %v", err)
	}
	manifest := strings.ReplaceAll(string(msManifest), "image: modelstore:latest", "image: modelstore:e2e\n          imagePullPolicy: Never")
	manifest = strings.ReplaceAll(manifest, "--blob-store=gs://BUCKET", "--blob-store=/tmp/blobs") // no GCS in kind
	h.KubectlApplyContent("modelstore", manifest)
	// A reused kind cluster keeps the previous run's modelstore pod, and
	// with it that run's statistics and compiled artifact (its blob store
	// is the container's /tmp): restart it, so "first register compiles"
	// and "one request recorded" are about this run.
	h.RunCommand("kubectl", "rollout", "restart", "deployment/modelstore", "-n", "modelstore")
	h.RunCommand("kubectl", "rollout", "status", "deployment/modelstore", "-n", "modelstore", "--timeout=300s")

	// The first part's in-cluster executor still holds its CPU request: on
	// a single kind node the compile Job's pod then never schedules.
	if out, err := exec.Command("kubectl", "delete", "pod", "-l", "app=vxpu-executor", "-n", "default", "--wait").CombinedOutput(); err != nil {
		t.Logf("deleting the in-cluster executors (best effort): %v: %s", err, out)
	}
	h.DeletePod("vxpu-external-executor", "default")
	h.KubectlApplyContent("external-executor", externalExecutorManifest)
	if err := h.WaitForPodReady("vxpu-external-executor", "default", 5*time.Minute); err != nil {
		t.Fatalf("external executor: %v", err)
	}

	// Every command is bounded (a register imports the model and compiles
	// it in a Job: about two minutes in kind), so a hang fails here with the
	// cluster's state rather than at the test binary's timeout with none.
	run := func(args ...string) string {
		ctx, cancel := context.WithTimeout(t.Context(), 12*time.Minute)
		defer cancel()
		c := exec.CommandContext(ctx, vxpuBin, args...)
		c.Env = env
		// the CLI's kubectl port-forward inherits its output: without a
		// WaitDelay, CombinedOutput waits for that child after the deadline
		// has killed the CLI, and the bound never fires
		c.WaitDelay = 10 * time.Second
		out, err := c.CombinedOutput()
		t.Logf("vxpu %s:\n%s", strings.Join(args, " "), out)
		if err != nil {
			if ctx.Err() != nil {
				err = fmt.Errorf("no answer within 12 minutes: %w", ctx.Err())
			}
			dumpByNameState(h)
			t.Fatalf("vxpu %s: %v", strings.Join(args, " "), err)
		}
		return string(out)
	}
	run("up", "--modelstore", "http://modelstore.modelstore", "--external-executor", "vxpu-external-executor:50051")

	out := run("register", model)
	if !strings.Contains(out, "registered "+model) || !strings.Contains(out, ": compiled") {
		t.Errorf("first register should compile the model, got:\n%s", out)
	}
	out = run("ask", "--model", model, "Is the sky blue?")
	if !strings.Contains(out, "Is the sky blue?") {
		t.Errorf("ask by name: output lacks the prompt:\n%s", out)
	}
	if logs := h.GetPodLogs("app=vxpu-external-executor", "default"); !strings.Contains(logs, "loading") {
		t.Errorf("the model should have been served by the external executor; its logs:\n%s", logs)
	}
	// the coordinator asked the executor what it runs on (#394)
	if logs := h.GetPodLogs("app=vxpu-router", "default"); !strings.Contains(logs, `"Executor described"`) { // klog quotes the message: not "... itself unreadably"
		t.Errorf("the router should have described the external executor; its logs:\n%s", logs)
	}
	out = run("stats", model)
	if !strings.Contains(out, `"requests": 1`) || !strings.Contains(out, `"telemetry_sampled": 1`) {
		t.Errorf("the engine should have recorded the request and its telemetry, got:\n%s", out)
	}
	out = run("register", model)
	if !strings.Contains(out, "already compiled") {
		t.Errorf("a second register should find the compiled artifact, got:\n%s", out)
	}
	// The coordinator's plan (#394): a switch set at register goes with
	// the next load, which the executor counts as a new engine, and its
	// plan records the switch as the coordinator's.
	out = run("register", "--plan", "VXPU_FUSED_EXPERTS=0", model)
	if !strings.Contains(out, "plan: map[VXPU_FUSED_EXPERTS:0]") {
		t.Errorf("register --plan should report the plan, got:\n%s", out)
	}
	run("ask", "--model", model, "Is the sky blue?")
	if logs := h.GetPodLogs("app=vxpu-external-executor", "default"); !strings.Contains(logs, "fused_experts=False (coordinator)") {
		t.Errorf("the executor should have reloaded the model under the coordinator's plan; its logs:\n%s", logs)
	}
	// and the coordinator says what it chose and why
	out = run("explain", model)
	if !strings.Contains(out, `"VXPU_FUSED_EXPERTS": "0"`) || !strings.Contains(out, "no routed experts") {
		t.Errorf("explain should show the registered switch and the planner's reason, got:\n%s", out)
	}
	h.DeletePod("vxpu-external-executor", "default")
}
