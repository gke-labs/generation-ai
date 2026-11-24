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

package kernelplan

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

func fakeModelstoreServer(t *testing.T) *httptest.Server {
	t.Helper()
	blobs := map[string][]byte{}
	manifests := map[string][]byte{}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if sha, ok := strings.CutPrefix(r.URL.Path, "/blobs/"); ok {
			if data, ok := blobs["sha256:"+sha]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
				return
			}
			http.NotFound(w, r)
			return
		}
		path := strings.TrimPrefix(r.URL.Path, "/v2/")
		switch {
		case r.Method == http.MethodPost && strings.Contains(path, "/blobs/uploads/"):
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			blobs[digest] = data
			w.WriteHeader(http.StatusCreated)
		case (r.Method == http.MethodGet || r.Method == http.MethodHead) && strings.Contains(path, "/blobs/"):
			digest := path[strings.LastIndex(path, "/")+1:]
			if data, ok := blobs[digest]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
		case strings.Contains(path, "/manifests/"):
			i := strings.Index(path, "/manifests/")
			key := path[:i] + ":" + path[i+len("/manifests/"):]
			if r.Method == http.MethodPut {
				data, _ := io.ReadAll(r.Body)
				manifests[key] = data
				w.WriteHeader(http.StatusCreated)
			} else if data, ok := manifests[key]; ok {
				if r.Method == http.MethodHead {
					w.WriteHeader(http.StatusOK)
				} else {
					_, _ = w.Write(data)
				}
			} else {
				http.NotFound(w, r)
			}
		default:
			http.NotFound(w, r)
		}
	}))
}

func fakePutArtifact(t *testing.T, storeURL, repo, tag, artDir string) {
	t.Helper()
	var layers []weightstore.Layer
	for _, name := range []string{"manifest.json", "binding.json", "decode.pt2"} {
		data, err := os.ReadFile(filepath.Join(artDir, name))
		if err != nil {
			if os.IsNotExist(err) && name != "decode.pt2" {
				continue
			}
			t.Fatalf("read %s: %v", name, err)
		}
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, fmt.Sprintf("%s/v2/%s/blobs/uploads/?digest=%s", storeURL, repo, digest), bytes.NewReader(data))
		if err != nil {
			t.Fatalf("upload req %s: %v", name, err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusCreated {
			t.Fatalf("upload blob %s: %v", name, err)
		}
		_ = resp.Body.Close()
		layers = append(layers, weightstore.Layer{
			MediaType:   "application/octet-stream",
			Digest:      digest,
			Size:        int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": name},
		})
	}
	body, err := json.Marshal(weightstore.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.oci.image.manifest.v1+json",
		Layers:        layers,
	})
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", storeURL, repo, tag), bytes.NewReader(body))
	if err != nil {
		t.Fatalf("manifest req: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusCreated {
		t.Fatalf("put manifest: %v", err)
	}
	_ = resp.Body.Close()
}

type testStore struct {
	mu       sync.Mutex
	failures map[string]*planpb.CompileFailure
	packages map[string]bool
}

func newTestStore() *testStore {
	return &testStore{
		failures: make(map[string]*planpb.CompileFailure),
		packages: make(map[string]bool),
	}
}

func (s *testStore) HasPackage(ctx context.Context, model, key string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.packages[key], nil
}

func (s *testStore) RecordCompileFailure(ctx context.Context, model string, fail *planpb.CompileFailure) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if fail != nil && fail.GetPackage() != nil {
		s.failures[fail.GetPackage().GetKey()] = fail
	}
	return nil
}

func (s *testStore) ClearCompileFailure(ctx context.Context, model, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.failures, key)
	return nil
}

func (s *testStore) GetCompileFailure(ctx context.Context, model, key string) (*planpb.CompileFailure, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fail, ok := s.failures[key]
	if !ok {
		return nil, errors.New("not found")
	}
	return fail, nil
}

func (s *testStore) GetBodyCompileOutcome(ctx context.Context, op, body, machineKind, softwareVersion string) (*planpb.BodyCompileOutcome, error) {
	return nil, nil
}

func (s *testStore) RecordBodyCompileOutcome(ctx context.Context, outcome *planpb.BodyCompileOutcome) error {
	return nil
}

func (s *testStore) ListBodyCompileOutcomes(ctx context.Context, machineKind string) ([]*planpb.BodyCompileOutcome, error) {
	return nil, nil
}

func findPythonBin(root string) string {
	if py := os.Getenv("VXPU_PYTHON"); py != "" {
		return py
	}
	if venv := os.Getenv("VIRTUAL_ENV"); venv != "" {
		py := filepath.Join(venv, "bin/python")
		if _, err := os.Stat(py); err == nil {
			return py
		}
	}
	if root != "" {
		venvPy := filepath.Join(root, ".venv/bin/python")
		if _, err := os.Stat(venvPy); err == nil {
			return venvPy
		}
	}
	if py, err := exec.LookPath("python3"); err == nil {
		return py
	}
	return "python3"
}

func startRealAgentServer(t *testing.T, root, pyBin string, store *weightstore.Client, customProcs map[string]agent.ProcessConfig) (pb.AgentClient, *agent.Server, func()) {
	t.Helper()
	workDir := t.TempDir()
	procs := customProcs
	if procs == nil {
		procs = agent.DefaultProcesses()
		cfg := procs["compiler"]
		cfg.Binary = pyBin
		if root != "" {
			cfg.Env = append(cfg.Env, "PYTHONPATH="+filepath.Join(root, "vxpu/python"))
			venvBin := filepath.Join(root, ".venv/bin")
			if _, err := os.Stat(venvBin); err == nil {
				cfg.Env = append(cfg.Env, "PATH="+venvBin+":"+os.Getenv("PATH"))
			}
		}
		procs["compiler"] = cfg
	}

	srv := agent.NewServer("", workDir, store)
	srv.Host = "127.0.0.1"
	srv.Processes = procs

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}
	grpcSrv := grpc.NewServer()
	pb.RegisterAgentServer(grpcSrv, srv)
	go func() {
		_ = grpcSrv.Serve(lis)
	}()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial agent failed: %v", err)
	}
	client := pb.NewAgentClient(conn)

	cleanup := func() {
		_ = conn.Close()
		srv.Stop()
		grpcSrv.Stop()
		_ = lis.Close()
	}
	return client, srv, cleanup
}

func isCUDASupported(pyBin string) bool {
	cmd := exec.Command(pyBin, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)")
	return cmd.Run() == nil
}

// TestCompilerClient_ThroughRealAgentAndService verifies the compiler client against
// a real agent starting the real Python compiler service process on the tiny DeepSeek model.
// Where a skip is a failure: when VXPU_REQUIRE_MODEL_SOURCE=1, any missing dependency fails.
func TestCompilerClient_ThroughRealAgentAndService(t *testing.T) {
	root := findGitRoot()
	pyBin := findPythonBin(root)

	if root != "" {
		submoduleCheck := filepath.Join(root, "vxpu/third_party/deepseek_v41/inference/model.py")
		if _, err := os.Stat(submoduleCheck); err != nil {
			if os.Getenv("VXPU_REQUIRE_MODEL_SOURCE") == "1" {
				t.Fatalf("deepseek_v41 model source is absent (expected at %s)", submoduleCheck)
			}
			t.Skipf("deepseek_v41 model source is absent (expected at %s)", submoduleCheck)
		}
	}

	ts := fakeModelstoreServer(t)
	defer ts.Close()

	agentStore := weightstore.New(ts.URL)
	artDir := getTinyDeepSeekArtifactDir(t)
	artRepo := weightstore.ArtifactRepository("test-model")
	artTag := "v1"
	fakePutArtifact(t, ts.URL, artRepo, artTag, artDir)
	artRef := artRepo + ":" + artTag

	agentClient, srv, cleanup := startRealAgentServer(t, root, pyBin, agentStore, nil)
	defer cleanup()

	store := newTestStore()
	client := NewCompilerClient(agentClient, store)

	device := "cpu"
	machineKind := "cpu:test:avx512"
	if isCUDASupported(pyBin) {
		device = "cuda"
		machineKind = "gpu:l4"
	}

	pkgs := []*planpb.Package{
		{Name: "submod_2", Key: "key_submod_2_test"},
	}
	if device == "cpu" {
		// On CPU, also test toolchain failure reporting with submod_6 (Float8 C++ error).
		// On GPU (L4), building one small segment (submod_2) is sufficient to show the full service path.
		pkgs = append(pkgs, &planpb.Package{Name: "submod_6", Key: "key_submod_6_fail"})
	}

	req := &pb.CompileRequest{
		Artifact:        artRef,
		Plan:            &planpb.PlanFile{},
		MachineKind:     machineKind,
		SoftwareVersion: "test-software-v1",
		Device:          device,
		Packages:        pkgs,
	}

	report, err := client.Compile(t.Context(), "test-model", req)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	// Verify build outcome for submod_2.
	var foundSubmod2 bool
	for _, p := range report.GetPresent() {
		if p.GetName() == "submod_2" {
			foundSubmod2 = true
			break
		}
	}
	if !foundSubmod2 {
		t.Fatalf("expected package submod_2 present in report: %v", report.GetPresent())
	}
	if ok, err := agentStore.HasPackage(t.Context(), "test-model", "key_submod_2_test"); err != nil || !ok {
		t.Fatalf("expected compiled package key_submod_2_test in store: ok=%v err=%v", ok, err)
	}

	// On CPU, submod_6 fails toolchain compilation as a result (not an RPC error).
	if device == "cpu" {
		var foundSubmod6 bool
		for _, f := range report.GetFailed() {
			if f.GetPackage().GetName() == "submod_6" {
				foundSubmod6 = true
				if f.GetReason() == "" {
					t.Errorf("expected non-empty compile failure reason for submod_6")
				}
				break
			}
		}
		if !foundSubmod6 {
			t.Fatalf("expected package submod_6 in failed report on CPU: %v", report.GetFailed())
		}
		// Verify failure recorded in store.
		storedFail, err := store.GetCompileFailure(t.Context(), "test-model", "key_submod_6_fail")
		if err != nil || storedFail == nil {
			t.Errorf("expected compile failure recorded in store, got: %v", storedFail)
		}
	}

	// 3. Process stopped after success/failure completion.
	listResp, err := agentClient.ListProcesses(t.Context(), &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("ListProcesses failed: %v", err)
	}
	for _, p := range listResp.GetProcesses() {
		if p.GetProcess() == "compiler" && p.GetState() == "running" {
			t.Errorf("expected compiler process %s stopped after compile, got running", p.GetId())
		}
	}

	// 4. Process stopped after cancellation:
	cancelCtx, cancelFn := context.WithCancel(t.Context())
	cancelDone := make(chan error, 1)
	go func() {
		cancelDone <- func() error {
			_, err := client.Compile(cancelCtx, "test-model", req)
			return err
		}()
	}()

	time.Sleep(100 * time.Millisecond)
	cancelFn()

	cancelErr := <-cancelDone
	if cancelErr == nil || !errors.Is(cancelErr, context.Canceled) && !strings.Contains(cancelErr.Error(), "context canceled") {
		t.Errorf("expected context canceled error, got: %v", cancelErr)
	}

	// Verify process stopped on agent after cancellation.
	time.Sleep(200 * time.Millisecond)
	listRespCancel, err := agentClient.ListProcesses(t.Context(), &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("ListProcesses failed: %v", err)
	}
	for _, p := range listRespCancel.GetProcesses() {
		if p.GetProcess() == "compiler" && p.GetState() == "running" {
			t.Errorf("expected compiler process %s stopped after cancellation, got running", p.GetId())
		}
	}

	_ = srv
}

// TestCompilerClient_ProcessExitsBeforeServing verifies that a process that terminates
// before serving is reported immediately with the agent's exit status and output tail.
func TestCompilerClient_ProcessExitsBeforeServing(t *testing.T) {
	root := findGitRoot()
	pyBin := findPythonBin(root)

	// Custom process config that exits immediately with code 42 and prints an error message.
	badProcs := map[string]agent.ProcessConfig{
		"compiler": {
			Binary: pyBin,
			Args:   []string{"-c", "import sys; sys.stderr.write('fatal: missing required library libcuda.so.1\\n'); sys.exit(42)"},
			Serves: true,
		},
	}

	agentClient, _, cleanup := startRealAgentServer(t, root, pyBin, nil, badProcs)
	defer cleanup()

	store := newTestStore()
	client := NewCompilerClient(agentClient, store)
	req := &pb.CompileRequest{
		MachineKind: "gpu:l4",
		Packages: []*planpb.Package{
			{Name: "submod_0", Key: "key_0"},
		},
	}

	start := time.Now()
	_, err := client.Compile(t.Context(), "test-model", req)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error for process that exited before serving, got nil")
	}
	// Verify immediate reporting without waiting for a timeout.
	if elapsed > 3*time.Second {
		t.Errorf("expected immediate error reporting, but took %v", elapsed)
	}
	if !strings.Contains(err.Error(), "exit code 42") {
		t.Errorf("expected error to contain exit code 42, got: %v", err)
	}
	if !strings.Contains(err.Error(), "libcuda.so.1") {
		t.Errorf("expected error to contain output tail, got: %v", err)
	}
}

// TestCompilerClient_FailedStopSeparatedFromWaiters verifies that when a compilation succeeds
// but the agent fails to stop the compiler process during cleanup, waiting callers share the
// successful compilation verdict while only the leader caller receives the cleanup error.
func TestCompilerClient_FailedStopSeparatedFromWaiters(t *testing.T) {
	ctx := t.Context()
	fakeComp := &fakeTestCompiler{}
	endpoint, cleanupServer := startFakeCompilerServer(t, fakeComp)
	defer cleanupServer()

	compileStarted := make(chan struct{})
	unblockCompile := make(chan struct{})

	fakeComp.compileHook = func(req *pb.CompileRequest) (*pb.CompileResponse, error) {
		select {
		case <-compileStarted:
		default:
			close(compileStarted)
		}
		<-unblockCompile
		return &pb.CompileResponse{
			Report: &planpb.PackageReport{
				Present: []*planpb.Package{
					{Name: "submod_2", Key: "shared_key"},
				},
				MachineKind: req.GetMachineKind(),
			},
		}, nil
	}

	agentMock := newMockAgent()
	agentMock.startHook = func(id string) (*pb.ProcessStatus, error) {
		return &pb.ProcessStatus{
			Id:       id,
			Process:  "compiler",
			State:    "running",
			Endpoint: endpoint,
		}, nil
	}
	// StopProcess fails during cleanup for the leader process!
	agentMock.stopHook = func(id string) error {
		return errors.New("agent simulated StopProcess failure")
	}

	store := newTestStore()
	client := NewCompilerClient(agentMock, store)

	req := &pb.CompileRequest{
		Artifact:        "vxpu-artifacts/test-model:v1",
		Plan:            &planpb.PlanFile{},
		MachineKind:     "cpu:test",
		SoftwareVersion: "v1",
		Device:          "cpu",
		Packages: []*planpb.Package{
			{Name: "submod_2", Key: "shared_key"},
		},
	}

	var wg sync.WaitGroup
	var rep2 *planpb.PackageReport
	var err1, err2 error

	wg.Add(2)
	go func() {
		defer wg.Done()
		_, err1 = client.Compile(ctx, "test-model", req)
	}()

	<-compileStarted

	go func() {
		defer wg.Done()
		rep2, err2 = client.Compile(ctx, "test-model", req)
	}()

	time.Sleep(50 * time.Millisecond)
	close(unblockCompile)
	wg.Wait()

	// Leader received the failed stop error.
	if err1 == nil || !strings.Contains(err1.Error(), "simulated StopProcess failure") {
		t.Fatalf("expected leader to receive StopProcess cleanup failure, got: %v", err1)
	}

	// Waiter shared the successful build verdict without failing!
	if err2 != nil {
		t.Fatalf("expected waiter to receive successful verdict, got err: %v", err2)
	}
	if len(rep2.GetPresent()) != 1 || rep2.GetPresent()[0].GetKey() != "shared_key" {
		t.Errorf("waiter report mismatch: %v", rep2)
	}
}

// TestCompilerClient_DeduplicationSharedVerdict verifies that concurrent requests
// for identical compilation inputs share the in-flight attempt and receive the shared verdict.
func TestCompilerClient_DeduplicationSharedVerdict(t *testing.T) {
	ctx := t.Context()
	fakeComp := &fakeTestCompiler{}
	endpoint, cleanupServer := startFakeCompilerServer(t, fakeComp)
	defer cleanupServer()

	compileStarted := make(chan struct{})
	unblockCompile := make(chan struct{})

	fakeComp.compileHook = func(req *pb.CompileRequest) (*pb.CompileResponse, error) {
		select {
		case <-compileStarted:
		default:
			close(compileStarted)
		}
		<-unblockCompile
		return &pb.CompileResponse{
			Report: &planpb.PackageReport{
				Present: []*planpb.Package{
					{Name: "submod_2", Key: "shared_key"},
				},
				MachineKind: req.GetMachineKind(),
			},
		}, nil
	}

	agentMock := newMockAgent()
	agentMock.startHook = func(id string) (*pb.ProcessStatus, error) {
		return &pb.ProcessStatus{
			Id:       id,
			Process:  "compiler",
			State:    "running",
			Endpoint: endpoint,
		}, nil
	}

	store := newTestStore()
	client := NewCompilerClient(agentMock, store)

	req := &pb.CompileRequest{
		Artifact:        "vxpu-artifacts/test-model:v1",
		Plan:            &planpb.PlanFile{},
		MachineKind:     "cpu:test",
		SoftwareVersion: "v1",
		Device:          "cpu",
		Packages: []*planpb.Package{
			{Name: "submod_2", Key: "shared_key"},
		},
	}

	var wg sync.WaitGroup
	var rep1, rep2 *planpb.PackageReport
	var err1, err2 error

	wg.Add(2)
	go func() {
		defer wg.Done()
		rep1, err1 = client.Compile(ctx, "test-model", req)
	}()

	<-compileStarted

	go func() {
		defer wg.Done()
		rep2, err2 = client.Compile(ctx, "test-model", req)
	}()

	time.Sleep(50 * time.Millisecond)
	close(unblockCompile)
	wg.Wait()

	if err1 != nil {
		t.Fatalf("caller 1 failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("caller 2 failed: %v", err2)
	}

	if len(rep1.GetPresent()) != 1 || rep1.GetPresent()[0].GetKey() != "shared_key" {
		t.Errorf("rep1 mismatch: %v", rep1)
	}
	if len(rep2.GetPresent()) != 1 || rep2.GetPresent()[0].GetKey() != "shared_key" {
		t.Errorf("rep2 mismatch: %v", rep2)
	}

	if fakeComp.calls() != 1 {
		t.Errorf("expected 1 compile RPC call due to deduplication, got %d", fakeComp.calls())
	}
}

// TestCompilerClient_DeduplicationDistinctKeys verifies that requests with different
// models or artifacts do not share deduplication keys.
func TestCompilerClient_DeduplicationDistinctKeys(t *testing.T) {
	req1 := &pb.CompileRequest{
		Artifact:    "vxpu-artifacts/model:v1",
		Plan:        &planpb.PlanFile{},
		MachineKind: "mk1",
		Packages: []*planpb.Package{
			{Name: "s0", Key: "k0"},
		},
	}
	req2 := &pb.CompileRequest{
		Artifact:    "vxpu-artifacts/model:v2", // different artifact!
		Plan:        &planpb.PlanFile{},
		MachineKind: "mk1",
		Packages: []*planpb.Package{
			{Name: "s0", Key: "k0"},
		},
	}

	key1 := compileDedupKey("modelA", req1)
	key2 := compileDedupKey("modelA", req2)
	if key1 == key2 {
		t.Errorf("expected different dedup keys for different Artifact, got same: %s", key1)
	}

	keyModelA := compileDedupKey("modelA", req1)
	keyModelB := compileDedupKey("modelB", req1) // different model!
	if keyModelA == keyModelB {
		t.Errorf("expected different dedup keys for different model, got same: %s", keyModelA)
	}
}

type mockAgent struct {
	pb.UnimplementedAgentServer
	mu         sync.Mutex
	processes  map[string]*pb.ProcessStatus
	startedIDs []string
	stoppedIDs []string
	startHook  func(id string) (*pb.ProcessStatus, error)
	stopHook   func(id string) error
	getHook    func(id string) (*pb.ProcessStatus, error)
}

func newMockAgent() *mockAgent {
	return &mockAgent{
		processes: make(map[string]*pb.ProcessStatus),
	}
}

func (a *mockAgent) StartProcess(ctx context.Context, in *pb.StartProcessRequest, _ ...grpc.CallOption) (*pb.StartProcessResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.startedIDs = append(a.startedIDs, in.GetId())
	if a.startHook != nil {
		st, err := a.startHook(in.GetId())
		if err != nil {
			return nil, err
		}
		a.processes[in.GetId()] = st
		return &pb.StartProcessResponse{Status: st}, nil
	}
	st := &pb.ProcessStatus{
		Id:       in.GetId(),
		Process:  in.GetProcess(),
		State:    "running",
		Endpoint: "127.0.0.1:50051",
	}
	a.processes[in.GetId()] = st
	return &pb.StartProcessResponse{Status: st}, nil
}

func (a *mockAgent) StopProcess(ctx context.Context, in *pb.StopProcessRequest, _ ...grpc.CallOption) (*pb.StopProcessResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.stoppedIDs = append(a.stoppedIDs, in.GetId())
	if a.stopHook != nil {
		if err := a.stopHook(in.GetId()); err != nil {
			return nil, err
		}
	}
	delete(a.processes, in.GetId())
	return &pb.StopProcessResponse{}, nil
}

func (a *mockAgent) GetProcess(ctx context.Context, in *pb.GetProcessRequest, _ ...grpc.CallOption) (*pb.GetProcessResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.getHook != nil {
		st, err := a.getHook(in.GetId())
		if err != nil {
			return nil, err
		}
		return &pb.GetProcessResponse{Status: st}, nil
	}
	st, ok := a.processes[in.GetId()]
	if !ok {
		return nil, errors.New("not found")
	}
	return &pb.GetProcessResponse{Status: st}, nil
}

func (a *mockAgent) MountWeights(ctx context.Context, in *pb.MountWeightsRequest, opts ...grpc.CallOption) (*pb.MountWeightsResponse, error) {
	return nil, errors.New("unimplemented")
}
func (a *mockAgent) UnmountWeights(ctx context.Context, in *pb.UnmountWeightsRequest, opts ...grpc.CallOption) (*pb.UnmountWeightsResponse, error) {
	return nil, errors.New("unimplemented")
}
func (a *mockAgent) ListProcesses(ctx context.Context, in *pb.ListProcessesRequest, opts ...grpc.CallOption) (*pb.ListProcessesResponse, error) {
	return nil, errors.New("unimplemented")
}
func (a *mockAgent) StageBlobs(ctx context.Context, in *pb.StageBlobsRequest, opts ...grpc.CallOption) (*pb.StageBlobsResponse, error) {
	return nil, errors.New("unimplemented")
}
func (a *mockAgent) ListBlobs(ctx context.Context, in *pb.ListBlobsRequest, opts ...grpc.CallOption) (*pb.ListBlobsResponse, error) {
	return nil, errors.New("unimplemented")
}
func (a *mockAgent) DescribeMachine(ctx context.Context, in *pb.DescribeMachineRequest, opts ...grpc.CallOption) (*pb.Probe, error) {
	return nil, errors.New("unimplemented")
}

type fakeTestCompiler struct {
	pb.UnimplementedCompilerServer
	mu           sync.Mutex
	compileCalls int32
	compileHook  func(req *pb.CompileRequest) (*pb.CompileResponse, error)
}

func (s *fakeTestCompiler) calls() int32 {
	return atomic.LoadInt32(&s.compileCalls)
}

func startFakeCompilerServer(t *testing.T, srv *fakeTestCompiler) (string, func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterCompilerServer(s, srv)
	h := health.NewServer()
	h.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(s, h)

	go func() {
		_ = s.Serve(lis)
	}()

	cleanup := func() {
		s.Stop()
		_ = lis.Close()
	}
	return lis.Addr().String(), cleanup
}

func (s *fakeTestCompiler) Compile(ctx context.Context, req *pb.CompileRequest) (*pb.CompileResponse, error) {
	atomic.AddInt32(&s.compileCalls, 1)
	s.mu.Lock()
	hook := s.compileHook
	s.mu.Unlock()

	if hook != nil {
		return hook(req)
	}

	present := make([]*planpb.Package, len(req.GetPackages()))
	for i, p := range req.GetPackages() {
		present[i] = &planpb.Package{
			Name: p.GetName(),
			Key:  p.GetKey(),
		}
	}
	return &pb.CompileResponse{
		Report: &planpb.PackageReport{
			Present:     present,
			MachineKind: req.GetMachineKind(),
		},
	}, nil
}
