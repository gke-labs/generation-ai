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

package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
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
	"google.golang.org/grpc/health/grpc_health_v1"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

func fakeRegistry(t *testing.T) *httptest.Server {
	t.Helper()
	srv, _ := fakeRegistryWithCount(t)
	return srv
}

func fakeRegistryWithCount(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var mu sync.Mutex
	var resolveRequests atomic.Int32
	blobs := map[string][]byte{}
	manifests := map[string][]byte{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/resolve/") {
			resolveRequests.Add(1)
			before, rest, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/resolve/")
			tag, file, ok := strings.Cut(rest, "/")
			if !ok {
				http.NotFound(w, r)
				return
			}
			mu.Lock()
			raw, hasManifest := manifests[before+":"+tag]
			var data []byte
			var found bool
			if hasManifest {
				var m weightstore.Manifest
				if json.Unmarshal(raw, &m) == nil {
					for _, l := range m.Layers {
						if l.Annotations["org.opencontainers.image.title"] == file {
							data, found = blobs[l.Digest]
							break
						}
					}
				}
			}
			mu.Unlock()
			if !found {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(data)
			return
		}
		if sha, ok := strings.CutPrefix(r.URL.Path, "/blobs/"); ok {
			mu.Lock()
			data, ok := blobs["sha256:"+sha]
			mu.Unlock()
			if ok {
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
			mu.Lock()
			blobs[digest] = data
			mu.Unlock()
			w.WriteHeader(http.StatusCreated)
		case (r.Method == http.MethodGet || r.Method == http.MethodHead) && strings.Contains(path, "/blobs/"):
			digest := path[strings.LastIndex(path, "/")+1:]
			mu.Lock()
			data, ok := blobs[digest]
			mu.Unlock()
			if ok {
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
				mu.Lock()
				manifests[key] = data
				mu.Unlock()
				w.WriteHeader(http.StatusCreated)
			} else {
				mu.Lock()
				data, ok := manifests[key]
				mu.Unlock()
				if ok {
					if r.Method == http.MethodHead {
						w.WriteHeader(http.StatusOK)
					} else {
						_, _ = w.Write(data)
					}
				} else {
					http.NotFound(w, r)
				}
			}
		default:
			http.NotFound(w, r)
		}
	}))
	return srv, &resolveRequests
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

func findGitRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func findPythonBin(root string) string {
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

func getTinyDeepSeekArtifactDir(t *testing.T, root, pyBin string) string {
	t.Helper()

	cmd := exec.Command(pyBin, "-c", "import sys; from vxpu.fixtures import tiny_deepseek_v41; sys.stdout.write(tiny_deepseek_v41.get_artifact_dir(max_cache_len=64))")
	if root != "" {
		cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(root, "vxpu/python"))
		venvBin := filepath.Join(root, ".venv/bin")
		if _, err := os.Stat(venvBin); err == nil {
			cmd.Env = append(cmd.Env, "PATH="+venvBin+":"+os.Getenv("PATH"))
		}
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Skipf("tiny DeepSeek artifact unavailable: %v (stderr: %s)", err, stderr.String())
	}
	dir := strings.TrimSpace(stdout.String())
	if _, err := os.Stat(filepath.Join(dir, "decode.pt2")); err != nil {
		t.Skipf("decode.pt2 not found in generated artifact dir %s: %v", dir, err)
	}
	return dir
}

func TestCompilerServingOverGRPC(t *testing.T) {
	root := findGitRoot()
	pyBin := findPythonBin(root)

	// Check if deepseek submodule and python modules are available.
	if root != "" {
		submoduleCheck := filepath.Join(root, "vxpu/third_party/deepseek_v41/inference/model.py")
		if _, err := os.Stat(submoduleCheck); err != nil {
			t.Skipf("deepseek_v41 model source is absent (expected at %s)", submoduleCheck)
		}
	}

	artDir := getTinyDeepSeekArtifactDir(t, root, pyBin)

	// Configure agent server with compiler task.
	workDir := t.TempDir()

	procs := DefaultProcesses()
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

	ts := fakeRegistry(t)
	defer ts.Close()

	artRepo := weightstore.ArtifactRepository("test-model")
	artTag := "v1"
	fakePutArtifact(t, ts.URL, artRepo, artTag, artDir)
	artRef := artRepo + ":" + artTag

	// Use 127.0.0.2 to verify that the agent passes its host through to the process
	// and that the process binds to the assigned address, not a hardcoded default.
	s := NewServer("", workDir, weightstore.New(ts.URL))
	s.Host = "127.0.0.2"
	s.Processes = procs

	ctx := t.Context()

	// 1. Start the executor through the agent.
	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:       "compiler-test",
		Process:  "compiler",
		Artifact: artRef,
	})
	if err != nil {
		t.Fatalf("StartProcess compiler failed: %v", err)
	}
	defer func() {
		stopCtx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 5*time.Second)
		defer cancel()
		_, _ = s.StopProcess(stopCtx, &pb.StopProcessRequest{Id: "compiler-test"})
	}()

	// 2. Read its endpoint from the process table only (no fallback to StartProcessResponse).
	statusResp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "compiler-test"})
	if err != nil {
		t.Fatalf("GetProcess failed: %v", err)
	}
	endpoint := statusResp.GetStatus().GetEndpoint()
	if endpoint == "" {
		t.Fatalf("expected non-empty endpoint in process table, got empty (status=%v)", statusResp.GetStatus())
	}
	if !strings.HasPrefix(endpoint, "127.0.0.2:") {
		t.Fatalf("expected endpoint on host 127.0.0.2, got: %s", endpoint)
	}

	// 3. Connect to that endpoint directly (the agent does not forward).
	// Connections between coordinator and executor will be authenticated with
	// mTLS under #662. For now, use insecure credentials.
	conn, err := grpc.NewClient(endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dialing compiler endpoint %s: %v", endpoint, err)
	}
	defer conn.Close()

	// 4. Establish readiness over the connection via standard gRPC health service.
	healthClient := grpc_health_v1.NewHealthClient(conn)
	var ready bool
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		checkCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		resp, checkErr := healthClient.Check(checkCtx, &grpc_health_v1.HealthCheckRequest{})
		cancel()
		if checkErr == nil && resp.GetStatus() == grpc_health_v1.HealthCheckResponse_SERVING {
			ready = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !ready {
		status, _ := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "compiler-test"})
		t.Fatalf("compiler on %s did not become ready over gRPC health check; status output:\n%s",
			endpoint, status.GetStatus().GetOutput())
	}

	compilerClient := pb.NewCompilerClient(conn)
	planMsg := &planpb.PlanFile{}

	// 5. Build a package for a segment of the tiny model (submod_2 succeeds on CPU).
	req1 := &pb.CompileRequest{
		Artifact:        artRef,
		Plan:            planMsg,
		MachineKind:     "cpu:test:avx512",
		SoftwareVersion: "test-software-v1",
		Device:          "cpu",
		Packages: []*planpb.Package{
			{Name: "submod_2", Key: "key_submod_2_test"},
		},
		Model: "test-model",
	}
	resp1, err := compilerClient.Compile(ctx, req1)
	if err != nil {
		t.Fatalf("Compile request 1 failed: %v", err)
	}
	report1 := resp1.GetReport()
	if report1 == nil {
		t.Fatal("expected non-nil PackageReport in CompileResponse")
	}
	if len(report1.GetPresent()) != 1 {
		t.Fatalf("expected 1 present package, got %d: %v", len(report1.GetPresent()), report1.GetPresent())
	}
	if report1.GetPresent()[0].GetName() != "submod_2" {
		t.Errorf("expected package name submod_2, got %s", report1.GetPresent()[0].GetName())
	}
	if report1.GetPresent()[0].GetKey() != "key_submod_2_test" {
		t.Errorf("expected package key key_submod_2_test, got %s", report1.GetPresent()[0].GetKey())
	}
	if len(report1.GetFailed()) != 0 {
		t.Errorf("expected 0 failed packages, got: %v", report1.GetFailed())
	}
	if ok, err := weightstore.New(ts.URL).HasPackage(ctx, "test-model", "key_submod_2_test"); err != nil || !ok {
		t.Errorf("expected compiled package key_submod_2_test in store: ok=%v err=%v", ok, err)
	}

	// 6. A second request on the same connection works.
	req2 := &pb.CompileRequest{
		Artifact:        artRef,
		Plan:            planMsg,
		MachineKind:     "cpu:test:avx512",
		SoftwareVersion: "test-software-v1",
		Device:          "cpu",
		Packages: []*planpb.Package{
			{Name: "submod_2", Key: "key_submod_2_repeat"},
		},
		Model: "test-model",
	}
	resp2, err := compilerClient.Compile(ctx, req2)
	if err != nil {
		t.Fatalf("Compile request 2 (on same connection) failed: %v", err)
	}
	report2 := resp2.GetReport()
	if report2 == nil {
		t.Fatal("expected non-nil PackageReport in second CompileResponse")
	}
	if len(report2.GetPresent()) != 1 {
		t.Fatalf("expected 1 present package on second request, got %d: %v", len(report2.GetPresent()), report2.GetPresent())
	}
	if report2.GetPresent()[0].GetKey() != "key_submod_2_repeat" {
		t.Errorf("expected package key key_submod_2_repeat, got %s", report2.GetPresent()[0].GetKey())
	}
	if ok, err := weightstore.New(ts.URL).HasPackage(ctx, "test-model", "key_submod_2_repeat"); err != nil || !ok {
		t.Errorf("expected compiled package key_submod_2_repeat in store: ok=%v err=%v", ok, err)
	}

	// 7. A compile failure comes back as a result naming the package (submod_6 fails on CPU).
	req3 := &pb.CompileRequest{
		Artifact:        artRef,
		Plan:            planMsg,
		MachineKind:     "cpu:test:avx512",
		SoftwareVersion: "test-software-v1",
		Device:          "cpu",
		Packages: []*planpb.Package{
			{Name: "submod_6", Key: "key_submod_6_fail"},
		},
		Model: "test-model",
	}
	resp3, err := compilerClient.Compile(ctx, req3)
	if err != nil {
		t.Fatalf("expected compile failure as result, but got RPC error: %v", err)
	}
	report3 := resp3.GetReport()
	if report3 == nil {
		t.Fatal("expected non-nil PackageReport in third CompileResponse")
	}
	if len(report3.GetPresent()) != 0 {
		t.Errorf("expected 0 present packages for failing segment, got: %v", report3.GetPresent())
	}
	if len(report3.GetFailed()) != 1 {
		t.Fatalf("expected 1 failed package, got %d: %v", len(report3.GetFailed()), report3.GetFailed())
	}
	fail := report3.GetFailed()[0]
	if fail.GetPackage().GetName() != "submod_6" {
		t.Errorf("expected failed package name submod_6, got %s", fail.GetPackage().GetName())
	}
	if fail.GetPackage().GetKey() != "key_submod_6_fail" {
		t.Errorf("expected failed package key key_submod_6_fail, got %s", fail.GetPackage().GetKey())
	}
	if fail.GetMachineKind() != "cpu:test:avx512" {
		t.Errorf("expected failed machine kind cpu:test:avx512, got %s", fail.GetMachineKind())
	}
	if fail.GetSoftwareVersion() != "test-software-v1" {
		t.Errorf("expected failed software version test-software-v1, got %s", fail.GetSoftwareVersion())
	}
	if fail.GetReason() == "" {
		t.Errorf("expected non-empty compile failure reason")
	}
	fmt.Printf("[test] compile failure successfully returned as result: package=%s reason=%s\n",
		fail.GetPackage().GetName(), fail.GetReason())
}
