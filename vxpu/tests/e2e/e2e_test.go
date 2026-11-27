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
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
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

	// Initialize submodules before building images so Docker COPY has all sources.
	h.RunCommand("git", "-C", gitRoot, "submodule", "update", "--init", "vxpu/third_party/transformers", "vxpu/third_party/deepseek_v41")

	// Fetch test models once into .cache/hf-mirror and start a host-side HTTP
	// mirror for client-side Python tests (facade_test.py, hero_test.py).
	cacheDir := ensureHFMirrorCache(t, gitRoot)
	hostMirrorURL := startHostHFMirror(t, gitRoot, cacheDir)

	// Concurrently build router, pki-init, modelstore images and vxpu CLI binary while executor image builds.
	var buildWg sync.WaitGroup
	var routerBuildErr, pkiBuildErr, modelstoreBuildErr, vxpuBuildErr error

	buildWg.Add(4)
	go func() {
		defer buildWg.Done()
		h.t.Log("Building docker image vxpu-router:e2e")
		routerBuildErr = h.DockerBuildErr("vxpu-router:e2e", filepath.Join(vxpuRoot, "images/router/Dockerfile"), gitRoot)
	}()
	go func() {
		defer buildWg.Done()
		h.t.Log("Building docker image vxpu-pki-init:e2e")
		pkiBuildErr = h.DockerBuildErr("vxpu-pki-init:e2e", filepath.Join(vxpuRoot, "images/pki-init/Dockerfile"), gitRoot)
	}()
	go func() {
		defer buildWg.Done()
		h.t.Log("Building docker image modelstore:e2e")
		modelstoreBuildErr = h.DockerBuildErr("modelstore:e2e", filepath.Join(gitRoot, "modelstore/images/modelstore/Dockerfile"), gitRoot)
	}()
	go func() {
		defer buildWg.Done()
		h.t.Logf("Building vxpu CLI binary to %s", vxpuBin)
		cmd := exec.Command("go", "build", "-o", vxpuBin, filepath.Join(vxpuRoot, "cmd/vxpu"))
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			vxpuBuildErr = fmt.Errorf("go build vxpu binary: %w\nOutput: %s", err, out)
		}
	}()

	// Read HEAD commit so the executor image is stamped with the tree commit it is built from.
	treeCommitOut, err := exec.Command("git", "-C", gitRoot, "rev-parse", "HEAD").Output()
	if err != nil {
		t.Fatalf("reading git HEAD commit: %v", err)
	}
	treeCommit := strings.TrimSpace(string(treeCommitOut))

	// Build the executor image in foreground
	h.DockerBuild("vxpu-executor:e2e", filepath.Join(vxpuRoot, "images/executor/Dockerfile"), gitRoot, "--build-arg", "TREE_COMMIT="+treeCommit)

	buildWg.Wait()
	if routerBuildErr != nil {
		h.t.Fatalf("failed building router image: %v", routerBuildErr)
	}
	if pkiBuildErr != nil {
		h.t.Fatalf("failed building pki-init image: %v", pkiBuildErr)
	}
	if modelstoreBuildErr != nil {
		h.t.Fatalf("failed building modelstore image: %v", modelstoreBuildErr)
	}
	if vxpuBuildErr != nil {
		h.t.Fatalf("failed building vxpu binary: %v", vxpuBuildErr)
	}

	// Build and deploy the in-cluster Hugging Face mirror and the test-only
	// executor image configured with HF_ENDPOINT pointing at that mirror.
	setupClusterHFMirror(t, h, gitRoot, cacheDir)

	// Load router and pki-init images into Kind (executor-mirror and hf-mirror
	// are loaded by setupClusterHFMirror).
	h.KindLoad("vxpu-router:e2e")
	h.KindLoad("vxpu-pki-init:e2e")

	// Export the model inside the Kind cluster against the in-cluster mirror so
	// manifest.json records the in-cluster mirror URL for the router's blob cache.
	exportModelInCluster(t, h, artifactDir)

	// Exercise binary plan exchange between Go planner and Python executor.
	h.t.Log("Exercising binary plan exchange between Go planner and Python executor")
	planPb := filepath.Join(artifactDir, "plan.pb")
	runRecordPb := filepath.Join(artifactDir, "run_record.pb")
	h.RunCommand(vxpuBin, "plan", "emit", "--artifact", artifactDir, "--out", planPb)

	// Verify that vxpu plan emit on tiny Llama produces a pass-through plan:
	// it contains the artifact's macroscopic family sites from binding.json,
	// all with strategy "default" and body unset, with zero knobs.
	emittedPlan, err := kernelplan.ReadPlan(planPb)
	if err != nil {
		h.t.Fatalf("reading emitted plan %s: %v", planPb, err)
	}
	if len(emittedPlan.Sites) == 0 {
		h.t.Fatalf("expected macroscopic family sites from binding.json, got 0")
	}
	for _, site := range emittedPlan.Sites {
		if site.Body != "" {
			h.t.Fatalf("expected body unset in pass-through plan, got %q on site %q", site.Body, site.Id)
		}
		if site.Strategy != "default" && site.Strategy != "" {
			h.t.Fatalf("expected default strategy, got %q on site %q", site.Strategy, site.Id)
		}
	}
	if emittedPlan.Knobs != nil {
		h.t.Fatalf("expected no knobs in pass-through plan, got: %v", emittedPlan.Knobs)
	}

	// Verify that the tree page for that artifact and plan renders, plainly states
	// that every call site runs the engine's own implementation, and omits the rendered unmatched-section.
	treeHTMLPath := filepath.Join(artifactDir, "tree.html")
	h.RunCommand(vxpuBin, "plan", "tree", "--artifact", artifactDir, "--plan", planPb, "--out", treeHTMLPath)
	treeHTMLBytes, err := os.ReadFile(treeHTMLPath)
	if err != nil {
		h.t.Fatalf("reading tree.html: %v", err)
	}
	if !strings.Contains(string(treeHTMLBytes), "Every call site runs the engine's own implementation") {
		h.t.Fatalf("tree.html does not plainly say every call site runs the engine's own implementation:\n%s", string(treeHTMLBytes))
	}
	if strings.Contains(string(treeHTMLBytes), `<details class="unmatched-section">`) {
		h.t.Fatalf("pass-through plan tree.html should not render the unmatched-section:\n%s", string(treeHTMLBytes))
	}

	planBytes, err := os.ReadFile(planPb)
	if err != nil {
		h.t.Fatalf("reading %s: %v", planPb, err)
	}
	fillCmd := exec.Command("docker", "run", "--rm", "-i", "--entrypoint", "python", "vxpu-executor:e2e",
		"-c", "import sys; from vxpu import physplan; open('/tmp/plan.pb', 'wb').write(sys.stdin.buffer.read()); pf = physplan.read_plan_file('/tmp/plan.pb'); rec = physplan.fill_run_record(pf, median_step_ms=1.23); physplan.write_plan_file('/tmp/run_record.pb', rec); sys.stdout.buffer.write(open('/tmp/run_record.pb', 'rb').read())")
	fillCmd.Stdin = bytes.NewReader(planBytes)
	var fillOut, fillErr bytes.Buffer
	fillCmd.Stdout = &fillOut
	fillCmd.Stderr = &fillErr
	if err := fillCmd.Run(); err != nil {
		h.t.Fatalf("fill_run_record in executor image: %v\nStderr: %s", err, fillErr.String())
	}
	if err := os.WriteFile(runRecordPb, fillOut.Bytes(), 0o644); err != nil {
		h.t.Fatalf("writing %s: %v", runRecordPb, err)
	}
	showCmd := exec.Command(vxpuBin, "plan", "show", "--type", "plan", "--json", runRecordPb)
	showOut, err := showCmd.CombinedOutput()
	if err != nil {
		h.t.Fatalf("vxpu plan show --json failed: %v\nOutput: %s", err, showOut)
	}
	if !strings.Contains(string(showOut), `"median_step_ms": 1.23`) {
		h.t.Fatalf("expected median_step_ms in vxpu plan show output, got:\n%s", showOut)
	}

	// Confirm that the Python engine contains zero golden-check logic.
	h.RunCommand("docker", "run", "--rm",
		"--entrypoint", "sh",
		"vxpu-executor:e2e",
		"-c", "! grep -i golden /app/vxpu/engine.py")

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
	askCtx, cancelAsk := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancelAsk()
	cmd := exec.CommandContext(askCtx, vxpuBin, "ask", "--artifact", artifactDir, "--accelerator", "none", "Is the sky blue?")
	// Set the environment variables so vxpu CLI knows which images to orchestrate
	cmd.Env = append(os.Environ(),
		"VXPU_ROUTER_IMAGE=vxpu-router:e2e",
		"VXPU_EXECUTOR_IMAGE="+executorMirrorImage,
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

	// Print executor pod process table on failure
	if err != nil {
		if podOut, pErr := exec.Command("kubectl", "get", "pod", "-l", "app=vxpu-executor", "-o", "jsonpath={.items[0].metadata.name}").CombinedOutput(); pErr == nil && len(podOut) > 0 {
			podName := strings.TrimSpace(string(podOut))
			if podName != "" {
				psOut, _ := exec.Command("kubectl", "exec", podName, "--", "ps", "-eo", "pid,ppid,args").CombinedOutput()
				fmt.Fprintf(os.Stderr, "executor pod process table:\n%s\n", psOut)
				t.Logf("executor pod process table:\n%s", psOut)
			}
		}
		t.Fatalf("first vxpu ask failed: %v\nOutput: %s", err, string(out))
	}

	outputStr := string(out)
	fmt.Fprintf(os.Stderr, "vxpu ask output:\n%s\n", outputStr)
	t.Logf("vxpu ask output:\n%s", outputStr)

	if !strings.Contains(outputStr, "Is the sky blue?") {
		t.Error("CLI output does not contain the prompt")
	}

	// Verify the router log confirms verification was skipped when no golden file is registered
	if !strings.Contains(routerLogs, "No golden file for model; skipping verification") {
		t.Errorf("expected router log to note verification skipped for unverified model, got:\n%s", routerLogs)
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

	// Verify executor pod process hierarchy: agent as PID 1, runtime as its child.
	podOut, err := exec.Command("kubectl", "get", "pod", "-l", "app=vxpu-executor", "-o", "jsonpath={.items[0].metadata.name}").CombinedOutput()
	if err != nil || strings.TrimSpace(string(podOut)) == "" {
		t.Fatalf("getting executor pod: %v (%s)", err, podOut)
	}
	podName := strings.TrimSpace(string(podOut))

	psOut, err := exec.Command("kubectl", "exec", podName, "--", "ps", "-eo", "pid,ppid,args").CombinedOutput()
	if err != nil {
		t.Fatalf("ps in executor pod: %v (%s)", err, psOut)
	}
	psStr := string(psOut)
	t.Logf("Executor pod processes:\n%s", psStr)

	lines := strings.Split(strings.TrimSpace(psStr), "\n")
	var foundAgentPID1, foundRuntimeChild bool
	for _, l := range lines {
		fields := strings.Fields(l)
		if len(fields) < 3 {
			continue
		}
		pid, ppid, comm := fields[0], fields[1], strings.Join(fields[2:], " ")
		if pid == "1" && strings.Contains(comm, "vxpu-agent") {
			foundAgentPID1 = true
		}
		if ppid == "1" && (strings.Contains(comm, "vxpu.server") || strings.Contains(comm, "python")) {
			foundRuntimeChild = true
		}
	}
	if !foundAgentPID1 {
		t.Errorf("executor pod PID 1 is not vxpu-agent:\n%s", psStr)
	}
	if !foundRuntimeChild {
		t.Errorf("executor pod runtime is not child of PID 1:\n%s", psStr)
	}

	// Verify the agent in the executor pod logged the tree commit the image was built from.
	podLogs, err := exec.Command("kubectl", "logs", podName).CombinedOutput()
	if err != nil {
		t.Fatalf("reading executor pod logs: %v (%s)", err, podLogs)
	}
	wantVersion := fmt.Sprintf("softwareVersion=%q", treeCommit)
	if !strings.Contains(string(podLogs), wantVersion) {
		t.Errorf("executor pod logs do not contain %s:\n%s", wantVersion, podLogs)
	}

	machineKind, softwareVersion := probeExecutorAgent(t, podName)
	if softwareVersion != treeCommit {
		t.Errorf("agent DescribeMachine software_version = %q, want %q", softwareVersion, treeCommit)
	}
	t.Logf("Cluster executor machineKind=%q softwareVersion=%q", machineKind, softwareVersion)

	t.Run("pki_init", func(t *testing.T) {
		testPKIInit(t, h, vxpuRoot)
	})

	t.Run("facade_equivalence", func(t *testing.T) {
		testFacadeEquivalence(t, h, gitRoot, artifactDir, hostMirrorURL)
	})

	// Start modelstore and wire its URL into the router before the hero journey so
	// the router and executor agent can stage compiled packages from modelstore.
	deployModelstore(t, h, gitRoot)
	upModelstore := exec.Command(vxpuBin, "up", "--modelstore", inClusterModelstoreURL)
	upModelstore.Env = cmd.Env
	if out, err := upModelstore.CombinedOutput(); err != nil {
		t.Fatalf("vxpu up --modelstore failed: %v\nOutput: %s", err, string(out))
	}

	t.Run("hero_journey", func(t *testing.T) {
		testHeroJourney(t, h, gitRoot, artifactDir, vxpuBin, hostMirrorURL, machineKind, softwareVersion)
	})

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
      image: vxpu-executor:e2e-mirror
      imagePullPolicy: Never
      command: ["python3", "-m", "vxpu.server", "--port", "50051", "--keep-alive", "3600"]
      ports:
        - name: executor
          containerPort: 50051
      readinessProbe:
        tcpSocket:
          port: 50051
        initialDelaySeconds: 1
        periodSeconds: 1
---
apiVersion: v1
kind: Service
metadata:
  name: vxpu-external-executor
spec:
  selector:
    app: vxpu-external-executor
  ports:
    - name: executor
      port: 50051
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

// deployModelstore loads modelstore:e2e into kind, applies its manifest with local
// disk storage and the in-cluster Hugging Face mirror, and waits for it to be ready.
func deployModelstore(t *testing.T, h *Harness, gitRoot string) {
	t.Helper()
	modelstoreRoot := filepath.Join(gitRoot, "modelstore")
	h.KindLoad("modelstore:e2e")
	msManifest, err := os.ReadFile(filepath.Join(modelstoreRoot, "k8s/manifest.yaml"))
	if err != nil {
		t.Fatalf("reading the modelstore manifest: %v", err)
	}
	manifest := strings.ReplaceAll(string(msManifest), "image: modelstore:latest", "image: modelstore:e2e\n          imagePullPolicy: Never")
	// Use local disk storage and the in-cluster Hugging Face mirror so modelstore
	// never reaches GCS or huggingface.co in kind.
	manifest = strings.ReplaceAll(manifest, "--blob-store=gs://BUCKET", "--blob-store=/tmp/blobs\n            - --hf-endpoint="+inClusterHFMirrorURL)
	h.KubectlApplyContent("modelstore", manifest)
	restartModelstore(h)
}

// restartModelstore rolls modelstore so its container-local /tmp/blobs starts empty.
func restartModelstore(h *Harness) {
	h.RunCommand("kubectl", "rollout", "restart", "deployment/modelstore", "-n", "modelstore")
	h.RunCommand("kubectl", "rollout", "status", "deployment/modelstore", "-n", "modelstore", "--timeout=300s")
}

// testByName registers a model by name (the router imports it into
// modelstore and compiles it in a Job with the executor image) and asks
// it a question by name, served by an external executor (#357).
func testByName(t *testing.T, h *Harness, gitRoot, vxpuBin string, env []string) {
	const model = "trl-internal-testing/tiny-random-LlamaForCausalLM"
	// Restart modelstore so the hero journey's packages/artifacts (and any reused
	// kind cluster state) do not affect "first register compiles" and "one request recorded".
	restartModelstore(h)

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

	// The first part's in-cluster executor still holds its CPU request: on
	// a single kind node the compile Job's pod then never schedules. Delete
	// it after `vxpu up` replaces the router so the old router's reconcile
	// loop does not recreate it.
	delCtx, cancelDel := context.WithTimeout(t.Context(), 1*time.Minute)
	defer cancelDel()
	delCmd := exec.CommandContext(delCtx, "kubectl", "delete", "pod", "-l", "app=vxpu-executor", "-n", "default", "--wait", "--timeout=1m")
	delCmd.WaitDelay = 5 * time.Second
	if out, err := delCmd.CombinedOutput(); err != nil {
		t.Logf("deleting the in-cluster executors (best effort): %v: %s", err, out)
	}

	goldenFile := filepath.Join(gitRoot, "vxpu/pkg/kernelplan/fixtures/golden/tiny-random-llama.json")
	goldenDoc, err := kernelplan.ReadGoldenFile(goldenFile)
	if err != nil {
		t.Fatalf("ReadGoldenFile failed: %v", err)
	}

	out := run("register", "--golden", goldenFile, model)
	if !strings.Contains(out, "registered "+model) || !strings.Contains(out, ": compiled") {
		t.Errorf("first register should compile the model, got:\n%s", out)
	}
	out = run("ask", "--model", model, "Is the sky blue?")
	if !strings.Contains(out, "Is the sky blue?") {
		t.Errorf("ask by name: output lacks the prompt:\n%s", out)
	}
	if logs := h.GetPodLogs("app=vxpu-router", "default"); !strings.Contains(logs, "Golden check passed") {
		t.Errorf("expected router log to confirm golden check passed, got:\n%s", logs)
	} else {
		for _, line := range strings.Split(logs, "\n") {
			if strings.Contains(line, "Golden check passed") {
				t.Logf("router log: %s", line)
				break
			}
		}
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

	// Verify TokenizeChat and GetVocab through the router to the executor.
	var pfOut bytes.Buffer
	pfCmd := exec.Command("kubectl", "port-forward", "pod/vxpu-router", "50051:50051")
	pfCmd.Stdout = &pfOut
	pfCmd.Stderr = &pfOut
	if err := pfCmd.Start(); err != nil {
		t.Fatalf("starting port-forward to pod/vxpu-router: %v", err)
	}

	pfDone := make(chan error, 1)
	go func() {
		defer close(pfDone)
		pfDone <- pfCmd.Wait()
	}()

	var stopOnce sync.Once
	stopPF := func() {
		stopOnce.Do(func() {
			if pfCmd.Process != nil {
				_ = pfCmd.Process.Kill()
			}
			select {
			case <-pfDone:
			case <-time.After(5 * time.Second):
				t.Logf("warning: kubectl port-forward did not exit within 5s after kill")
			}
		})
	}
	defer stopPF()

	portReady := false
	var lastDialErr error
	portDeadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(portDeadline) {
		select {
		case err := <-pfDone:
			t.Fatalf("kubectl port-forward exited unexpectedly: %v\nOutput: %s", err, pfOut.String())
		default:
		}

		conn, err := net.DialTimeout("tcp", "127.0.0.1:50051", 200*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			portReady = true
			break
		}
		lastDialErr = err
		time.Sleep(100 * time.Millisecond)
	}
	if !portReady {
		stopPF()
		t.Fatalf("port-forward to pod/vxpu-router on 50051 did not answer within 30s: %v\nOutput: %s", lastDialErr, pfOut.String())
	}

	conn, err := grpc.NewClient("127.0.0.1:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("connecting to forwarded router port: %v", err)
	}
	defer conn.Close()

	client := pb.NewEngineClient(conn)

	tokCtx, cancelTok := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancelTok()
	tokResp, err := client.TokenizeChat(tokCtx, &pb.TokenizeChatRequest{
		Model:    model,
		Messages: []*pb.ChatMessage{{Role: "user", Content: "Hello"}},
	}, grpc.WaitForReady(true))
	if err != nil {
		t.Errorf("TokenizeChat through router: %v", err)
	} else if len(tokResp.GetInputIds()) == 0 {
		t.Errorf("TokenizeChat through router returned no input ids")
	}

	vocabCtx, cancelVocab := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancelVocab()
	vocabResp, err := client.GetVocab(vocabCtx, &pb.GetVocabRequest{Model: model}, grpc.WaitForReady(true))
	if err != nil {
		t.Errorf("GetVocab through router: %v", err)
	} else if len(vocabResp.GetPieces()) == 0 {
		t.Errorf("GetVocab through router returned no pieces")
	}

	// Close the gRPC connection and stop port-forwarding before deleting the executor pod.
	_ = conn.Close()
	stopPF()

	// (b) with a changed expected token, the load is refused naming prompt and position
	mutatedGoldenFile := filepath.Join(t.TempDir(), "tiny_llama_mutated_golden.json")
	mutatedDoc := proto.Clone(goldenDoc).(*planpb.GoldenFile)
	mutatedDoc.Prompts[0].Tokens[0] = 99999
	if err := kernelplan.WriteGoldenFile(mutatedGoldenFile, mutatedDoc); err != nil {
		t.Fatalf("writing mutated golden file: %v", err)
	}
	_ = run("register", "--golden", mutatedGoldenFile, model)

	failCtx, cancelFail := context.WithTimeout(t.Context(), 1*time.Minute)
	defer cancelFail()
	failCmd := exec.CommandContext(failCtx, vxpuBin, "ask", "--model", model, "Is the sky blue?")
	failCmd.Env = env
	failOut, err := failCmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected ask with mutated golden to fail, got success:\n%s", failOut)
	}
	if !strings.Contains(string(failOut), "prompt 0") || !strings.Contains(string(failOut), "position 0") {
		t.Fatalf("expected failure to name prompt 0 and position 0, got:\n%s", failOut)
	}

	// (c) an ask after a refusal is refused
	failAgain := exec.CommandContext(failCtx, vxpuBin, "ask", "--model", model, "Is the sky blue?")
	failAgain.Env = env
	failAgainOut, err := failAgain.CombinedOutput()
	if err == nil {
		t.Fatalf("expected second ask after refusal to fail, got success:\n%s", failAgainOut)
	}
	if !strings.Contains(string(failAgainOut), "prompt 0") || !strings.Contains(string(failAgainOut), "position 0") {
		t.Fatalf("expected second ask after refusal to name prompt 0 and position 0, got:\n%s", failAgainOut)
	}

	// (d) a model that fails and later passes is served
	_ = run("register", "--golden", goldenFile, model)
	passCmd := exec.CommandContext(failCtx, vxpuBin, "ask", "--model", model, "Is the sky blue?")
	passCmd.Env = env
	passOut, err := passCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("expected ask after registering valid golden to succeed, got: %v\n%s", err, passOut)
	}
	if len(passOut) == 0 {
		t.Errorf("expected non-empty output from ask")
	}

	h.DeletePod("vxpu-external-executor", "default")
}

// checkPythonImports tests whether pythonBin can import torch, transformers, and vxpu.
func checkPythonImports(pythonBin, gitRoot string) error {
	cmd := exec.Command(pythonBin, "-c", "import torch; import transformers; from vxpu import AutoModelForCausalLM")
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("PYTHONPATH=%s:%s",
			filepath.Join(gitRoot, "vxpu/python"),
			filepath.Join(gitRoot, "vxpu/third_party/transformers"),
		),
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%s import check failed: %w (output: %s)", pythonBin, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// findClientPython returns the path to the client Python binary.
// In CI environments, it expects the task-provisioned virtualenv at .venv/bin/python.
// For local developer runs, it falls back to python3 on PATH.
func findClientPython(gitRoot string) (string, error) {
	if gitRoot != "" {
		venvPy := filepath.Join(gitRoot, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			return venvPy, nil
		}
	}
	if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
		return "", fmt.Errorf("in CI, client Python must be at %s", filepath.Join(gitRoot, ".venv/bin/python"))
	}
	if py, err := exec.LookPath("python3"); err == nil {
		return py, nil
	}
	return "", fmt.Errorf("no python3 found on PATH")
}

// testFacadeEquivalence verifies the AutoModelForCausalLM facade against the running
// router and executor in kind, asserting token-for-token equality with eager transformers.
func testFacadeEquivalence(t *testing.T, h *Harness, gitRoot, artifactDir, hostMirrorURL string) {
	t.Helper()
	h.t.Log("Testing AutoModelForCausalLM facade against eager transformers generate")

	pyBin, err := findClientPython(gitRoot)
	if err != nil {
		if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
			t.Fatalf("facade equivalence test requires client python: %v", err)
		}
		t.Logf("skipping facade equivalence test: %v", err)
		return
	}
	if err := checkPythonImports(pyBin, gitRoot); err != nil {
		if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
			t.Fatalf("facade equivalence test requires client python with torch and transformers fork: %v", err)
		}
		t.Logf("skipping facade equivalence test: %v", err)
		return
	}

	testCtx, cancelTest := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancelTest()

	testScript := filepath.Join(gitRoot, "vxpu/tests/e2e/facade_test.py")
	cmd := exec.CommandContext(testCtx, pyBin, testScript,
		"--artifact-dir", artifactDir,
	)
	cmd.WaitDelay = 5 * time.Second
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("PYTHONPATH=%s:%s",
			filepath.Join(gitRoot, "vxpu/python"),
			filepath.Join(gitRoot, "vxpu/third_party/transformers"),
		),
		"HF_ENDPOINT="+hostMirrorURL,
		"HF_HOME="+t.TempDir(),
	)

	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "router logs:\n%s\n", h.GetPodLogs("app=vxpu-router", "default"))
		fmt.Fprintf(os.Stderr, "executor logs:\n%s\n", h.GetPodLogs("app=vxpu-executor", "default"))
		t.Fatalf("facade equivalence test failed: %v\nOutput: %s", err, string(out))
	}
	h.t.Logf("Facade equivalence test passed:\n%s", string(out))
}

// startPortForward opens a kubectl port-forward to namespace/target:remotePort
// on a free local port and returns the local host:port and a stop function.
func startPortForward(t *testing.T, namespace, target string, remotePort int) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("picking local port for port-forward %s: %v", target, err)
	}
	localPort := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	args := []string{"port-forward"}
	if namespace != "" {
		args = append(args, "-n", namespace)
	}
	args = append(args, target, fmt.Sprintf("%d:%d", localPort, remotePort))

	pfCtx, cancel := context.WithCancel(t.Context())
	pf := exec.CommandContext(pfCtx, "kubectl", args...)
	pf.WaitDelay = 3 * time.Second
	var pfOut bytes.Buffer
	pf.Stdout = &pfOut
	pf.Stderr = &pfOut
	if err := pf.Start(); err != nil {
		cancel()
		t.Fatalf("starting kubectl %v: %v", args, err)
	}
	stop := func() {
		cancel()
		_ = pf.Wait()
	}

	addr := fmt.Sprintf("127.0.0.1:%d", localPort)
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		conn, dialErr := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if dialErr == nil {
			_ = conn.Close()
			return addr, stop
		}
		time.Sleep(200 * time.Millisecond)
	}
	stop()
	t.Fatalf("port-forward to %s did not become ready on %s: %s", target, addr, pfOut.String())
	return "", func() {}
}

// probeExecutorAgent queries vxpu-agent's DescribeMachine RPC on podName and
// returns the cluster's machine kind and reported executor software version.
func probeExecutorAgent(t *testing.T, podName string) (string, string) {
	t.Helper()
	addr, stop := startPortForward(t, "default", "pod/"+podName, 50052)
	defer stop()

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dialing executor agent on %s: %v", addr, err)
	}
	defer func() { _ = conn.Close() }()

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	probe, err := pb.NewAgentClient(conn).DescribeMachine(ctx, &pb.DescribeMachineRequest{})
	if err != nil {
		t.Fatalf("agent DescribeMachine on %s: %v", podName, err)
	}
	var machineKind string
	if m := machines.Match(probe); m != nil {
		machineKind, err = kernelplan.MachineKind(m)
	} else {
		machineKind, err = kernelplan.MachineKindFromProbe(probe)
	}
	if err != nil {
		t.Fatalf("deriving machine kind from agent probe: %v", err)
	}
	return machineKind, strings.TrimSpace(probe.GetSoftwareVersion())
}

// testHeroJourney runs the hero journey end-to-end test against the running
// router, executor, and modelstore in kind, failing the test if any of the
// four steps does not pass.
func testHeroJourney(t *testing.T, h *Harness, gitRoot, artifactDir, vxpuBin, hostMirrorURL, machineKind, softwareVersion string) {
	t.Helper()
	h.t.Log("Testing hero journey end-to-end on CPU")

	pyBin, err := findClientPython(gitRoot)
	if err != nil {
		if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
			t.Fatalf("hero journey test requires client python: %v", err)
		}
		t.Logf("skipping hero journey test: %v", err)
		return
	}
	if err := checkPythonImports(pyBin, gitRoot); err != nil {
		if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
			t.Fatalf("hero journey test requires client python with torch and transformers fork: %v", err)
		}
		t.Logf("skipping hero journey test: %v", err)
		return
	}

	msAddr, stopMS := startPortForward(t, "modelstore", "svc/modelstore", 80)
	defer stopMS()
	localModelstoreURL := "http://" + msAddr

	testCtx, cancelTest := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancelTest()

	testScript := filepath.Join(gitRoot, "vxpu/tests/e2e/hero_test.py")
	cmd := exec.CommandContext(testCtx, pyBin, testScript,
		"--artifact-dir", artifactDir,
		"--vxpu-bin", vxpuBin,
		"--modelstore", localModelstoreURL,
		"--in-cluster-modelstore", inClusterModelstoreURL,
		"--machine-kind", machineKind,
		"--software-version", softwareVersion,
	)
	cmd.WaitDelay = 5 * time.Second
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("PYTHONPATH=%s:%s",
			filepath.Join(gitRoot, "vxpu/python"),
			filepath.Join(gitRoot, "vxpu/third_party/transformers"),
		),
		"HF_ENDPOINT="+hostMirrorURL,
		"HF_HOME="+t.TempDir(),
	)

	out, err := cmd.CombinedOutput()
	h.t.Logf("Hero journey test output:\n%s", string(out))
	if err != nil {
		fmt.Fprintf(os.Stderr, "router logs:\n%s\n", h.GetPodLogs("app=vxpu-router", "default"))
		fmt.Fprintf(os.Stderr, "executor logs:\n%s\n", h.GetPodLogs("app=vxpu-executor", "default"))
		fmt.Fprintf(os.Stderr, "modelstore logs:\n%s\n", h.GetPodLogs("app=modelstore", "modelstore"))
		t.Fatalf("hero journey test failed: %v\nOutput: %s", err, string(out))
	}
}

const (
	inClusterHFMirrorURL   = "http://hf-mirror.default.svc.cluster.local:8080"
	inClusterModelstoreURL = "http://modelstore.modelstore"
	executorMirrorImage    = "vxpu-executor:e2e-mirror"
	hfMirrorImage          = "vxpu-hf-mirror:e2e"
)

const hfMirrorManifest = `apiVersion: v1
kind: Pod
metadata:
  name: hf-mirror
  labels:
    app: hf-mirror
spec:
  containers:
    - name: hf-mirror
      image: vxpu-hf-mirror:e2e
      imagePullPolicy: Never
      ports:
        - name: http
          containerPort: 8080
      readinessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 1
        periodSeconds: 1
---
apiVersion: v1
kind: Service
metadata:
  name: hf-mirror
spec:
  selector:
    app: hf-mirror
  ports:
    - name: http
      port: 8080
      targetPort: 8080
`

const exporterPodManifest = `apiVersion: v1
kind: Pod
metadata:
  name: vxpu-exporter
spec:
  restartPolicy: Never
  containers:
    - name: exporter
      image: vxpu-executor:e2e-mirror
      imagePullPolicy: Never
      command: ["sleep", "3600"]
`

func findMirrorPython(gitRoot string) string {
	venvPy := filepath.Join(gitRoot, ".venv/bin/python")
	if _, err := os.Stat(venvPy); err == nil {
		return venvPy
	}
	if py, err := exec.LookPath("python3"); err == nil {
		return py
	}
	return "python3"
}

// ensureHFMirrorCache populates .cache/hf-mirror once with retries and returns
// its path. Reuses existing verified files without contacting huggingface.co.
func ensureHFMirrorCache(t *testing.T, gitRoot string) string {
	t.Helper()
	cacheDir := filepath.Join(gitRoot, ".cache/hf-mirror")
	pyBin := findMirrorPython(gitRoot)
	mirrorScript := filepath.Join(gitRoot, "dev/ci/hf_mirror.py")
	cmd := exec.Command(pyBin, mirrorScript, "fetch", "--cache-dir", cacheDir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fetching Hugging Face mirror cache: %v\nOutput: %s", err, out)
	}
	t.Logf("Hugging Face mirror cache ready at %s:\n%s", cacheDir, strings.TrimSpace(string(out)))
	return cacheDir
}

// startHostHFMirror starts a local HTTP mirror on loopback for host-side test
// scripts and stops it when the test finishes.
func startHostHFMirror(t *testing.T, gitRoot, cacheDir string) string {
	t.Helper()
	pyBin := findMirrorPython(gitRoot)
	mirrorScript := filepath.Join(gitRoot, "dev/ci/hf_mirror.py")
	portFile := filepath.Join(t.TempDir(), "port")

	cmd := exec.Command(pyBin, mirrorScript, "serve",
		"--cache-dir", cacheDir,
		"--host", "127.0.0.1",
		"--port", "0",
		"--port-file", portFile,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting host Hugging Face mirror: %v", err)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	})

	deadline := time.Now().Add(15 * time.Second)
	var port string
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(portFile); err == nil {
			if p := strings.TrimSpace(string(data)); p != "" {
				port = p
				break
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	if port == "" {
		t.Fatalf("host Hugging Face mirror did not write port file; stderr:\n%s", stderr.String())
	}

	mirrorURL := "http://127.0.0.1:" + port
	resp, err := http.Get(mirrorURL + "/healthz")
	if err != nil {
		t.Fatalf("probing host Hugging Face mirror at %s: %v", mirrorURL, err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("host Hugging Face mirror /healthz returned %s", resp.Status)
	}
	t.Logf("Host Hugging Face mirror listening at %s", mirrorURL)
	return mirrorURL
}

// setupClusterHFMirror builds test-only images for the in-cluster Hugging Face
// mirror and mirror-configured executor, loads them into Kind, and starts the
// hf-mirror Pod and Service.
func setupClusterHFMirror(t *testing.T, h *Harness, gitRoot, cacheDir string) {
	t.Helper()

	mirrorCtx := t.TempDir()
	scriptBytes, err := os.ReadFile(filepath.Join(gitRoot, "dev/ci/hf_mirror.py"))
	if err != nil {
		t.Fatalf("reading hf_mirror.py: %v", err)
	}
	if err := os.WriteFile(filepath.Join(mirrorCtx, "hf_mirror.py"), scriptBytes, 0o755); err != nil {
		t.Fatalf("writing hf_mirror.py into build context: %v", err)
	}
	if err := os.CopyFS(filepath.Join(mirrorCtx, "cache"), os.DirFS(cacheDir)); err != nil {
		t.Fatalf("copying mirror cache into build context: %v", err)
	}
	mirrorDockerfile := filepath.Join(mirrorCtx, "Dockerfile")
	mirrorDF := "FROM vxpu-executor:e2e\n" +
		"COPY hf_mirror.py /mirror/hf_mirror.py\n" +
		"COPY cache /mirror/cache\n" +
		`ENTRYPOINT ["python3", "/mirror/hf_mirror.py", "serve", "--cache-dir", "/mirror/cache", "--host", "0.0.0.0", "--port", "8080"]` + "\n"
	if err := os.WriteFile(mirrorDockerfile, []byte(mirrorDF), 0o644); err != nil {
		t.Fatalf("writing hf-mirror Dockerfile: %v", err)
	}
	h.DockerBuild(hfMirrorImage, mirrorDockerfile, mirrorCtx)

	execCtx := t.TempDir()
	execDockerfile := filepath.Join(execCtx, "Dockerfile")
	execDF := "FROM vxpu-executor:e2e\nENV HF_ENDPOINT=" + inClusterHFMirrorURL + "\n"
	if err := os.WriteFile(execDockerfile, []byte(execDF), 0o644); err != nil {
		t.Fatalf("writing executor mirror Dockerfile: %v", err)
	}
	h.DockerBuild(executorMirrorImage, execDockerfile, execCtx)

	h.KindLoad(hfMirrorImage)
	h.KindLoad(executorMirrorImage)

	h.DeletePod("hf-mirror", "default")
	h.KubectlApplyContent("hf-mirror", hfMirrorManifest)
	if err := h.WaitForPodReady("hf-mirror", "default", 2*time.Minute); err != nil {
		t.Fatalf("hf-mirror pod not ready: %v\nlogs:\n%s", err, h.GetPodLogs("app=hf-mirror", "default"))
	}
	t.Cleanup(func() {
		h.DeletePod("hf-mirror", "default")
		_ = exec.Command("kubectl", "delete", "service", "hf-mirror", "-n", "default", "--ignore-not-found=true").Run()
	})
}

// exportModelInCluster exports the test model inside a Kind pod configured with
// the in-cluster mirror and extracts the artifact directory to artifactDir.
func exportModelInCluster(t *testing.T, h *Harness, artifactDir string) {
	t.Helper()
	h.t.Logf("Exporting model in Kind against %s to %s", inClusterHFMirrorURL, artifactDir)
	if err := os.RemoveAll(artifactDir); err != nil {
		t.Fatalf("cleaning artifactDir %s: %v", artifactDir, err)
	}
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("creating artifactDir %s: %v", artifactDir, err)
	}

	h.DeletePod("vxpu-exporter", "default")
	h.KubectlApplyContent("vxpu-exporter", exporterPodManifest)
	defer h.DeletePod("vxpu-exporter", "default")
	if err := h.WaitForPodReady("vxpu-exporter", "default", 2*time.Minute); err != nil {
		t.Fatalf("vxpu-exporter pod not ready: %v", err)
	}

	h.RunCommand("kubectl", "exec", "vxpu-exporter", "--",
		"python", "-m", "vxpu.export",
		"trl-internal-testing/tiny-random-LlamaForCausalLM",
		"-o", "/tmp/test_artifact",
	)

	tarCmd := exec.Command("kubectl", "exec", "vxpu-exporter", "--", "tar", "-C", "/tmp/test_artifact", "-cf", "-", ".")
	var tarOut, tarErr bytes.Buffer
	tarCmd.Stdout = &tarOut
	tarCmd.Stderr = &tarErr
	if err := tarCmd.Run(); err != nil {
		t.Fatalf("streaming exported artifact from vxpu-exporter: %v\nstderr: %s", err, tarErr.String())
	}
	if err := extractTar(bytes.NewReader(tarOut.Bytes()), artifactDir); err != nil {
		t.Fatalf("extracting exported artifact to %s: %v", artifactDir, err)
	}
	if _, err := os.Stat(filepath.Join(artifactDir, "manifest.json")); err != nil {
		t.Fatalf("exported artifact missing manifest.json: %v", err)
	}
}

func extractTar(r io.Reader, destDir string) error {
	cleanDest := filepath.Clean(destDir) + string(os.PathSeparator)
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, filepath.Clean(hdr.Name))
		if target != filepath.Clean(destDir) && !strings.HasPrefix(target, cleanDest) {
			return fmt.Errorf("tar entry escapes destination: %q", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(f, tr)
			closeErr := f.Close()
			if copyErr != nil {
				return copyErr
			}
			if closeErr != nil {
				return closeErr
			}
		}
	}
}
