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

package router

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/blobserver"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

const testCommit = "0123456789abcdef0123"

var testManifest = `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"org/M","revision":"main","commit_sha":"` + testCommit + `"},` +
	`"config":{},"files":{"abc":{"size":42,"name":"model.safetensors","source":"https://huggingface.co/org/M/resolve/c/model.safetensors"}},` +
	`"tensors":{}}`

// fakeModelstore serves the routes the catalog uses for one model, org/M.
type fakeModelstore struct {
	mu        sync.Mutex
	compiled  bool              // the artifact for testCommit exists
	manifest  string            // the artifact's manifest.json (default testManifest)
	binding   string            // its binding.json (default {})
	blobs     map[string][]byte // digest -> bytes (registry writes: statistics)
	manifests map[string][]byte // name:tag -> manifest
}

// registry serves the generic registry routes statistics and plans use,
// and the file route plans are read by; false if r is not one of them.
func (f *fakeModelstore) registry(w http.ResponseWriter, r *http.Request) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.blobs == nil {
		f.blobs, f.manifests = map[string][]byte{}, map[string][]byte{}
	}
	if name, file, ok := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/resolve/"); ok && strings.HasPrefix(name, "vxpu-plans/") {
		tag, title, _ := strings.Cut(file, "/")
		var m weightstore.Manifest
		if err := json.Unmarshal(f.manifests[name+":"+tag], &m); err == nil {
			for _, l := range m.Layers {
				if l.Annotations["org.opencontainers.image.title"] == title {
					_, _ = w.Write(f.blobs[l.Digest])
					return true
				}
			}
		}
		http.NotFound(w, r)
		return true
	}
	path := strings.TrimPrefix(r.URL.Path, "/v2/")
	if !strings.HasPrefix(path, "vxpu-stats/") && !strings.HasPrefix(path, "vxpu-plans/") {
		return false
	}
	switch {
	case r.Method == http.MethodPost && strings.HasSuffix(path, "/blobs/uploads/"):
		body, _ := io.ReadAll(r.Body)
		f.blobs[r.URL.Query().Get("digest")] = body
		w.WriteHeader(http.StatusCreated)
	case r.Method == http.MethodGet && strings.Contains(path, "/blobs/"):
		_, _ = w.Write(f.blobs[path[strings.LastIndex(path, "/")+1:]])
	case strings.Contains(path, "/manifests/"):
		i := strings.Index(path, "/manifests/")
		key := path[:i] + ":" + path[i+len("/manifests/"):]
		if r.Method == http.MethodPut {
			body, _ := io.ReadAll(r.Body)
			f.manifests[key] = body
			w.WriteHeader(http.StatusCreated)
		} else if m, ok := f.manifests[key]; ok {
			_, _ = w.Write(m)
		} else {
			http.NotFound(w, r)
		}
	case strings.HasSuffix(path, "/tags/list"):
		name := strings.TrimSuffix(path, "/tags/list")
		var tags []string
		for key := range f.manifests {
			if n, tag, _ := strings.Cut(key, ":"); n == name {
				tags = append(tags, tag)
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"name": name, "tags": tags})
	default:
		http.NotFound(w, r)
	}
	return true
}

func (f *fakeModelstore) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.registry(w, r) {
			return
		}
		f.mu.Lock()
		compiled := f.compiled
		f.mu.Unlock()
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/models/import":
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"metadata":{"name":"org/m"}}`))
		case r.URL.Path == "/models/org/m":
			_, _ = w.Write([]byte(`{"status":{"phase":"Ready"}}`))
		case r.URL.Path == "/v2/org/m/manifests/main":
			_, _ = w.Write([]byte(`{"annotations":{"vnd.gke-labs.huggingface.commit":"` + testCommit + `"}}`))
		case strings.HasPrefix(r.URL.Path, "/v2/vxpu-artifacts/org/m/manifests/"):
			if !compiled {
				http.NotFound(w, r)
			}
		case strings.HasPrefix(r.URL.Path, "/vxpu-artifacts/org/m/resolve/") && compiled:
			manifest, binding := testManifest, `{}`
			if f.manifest != "" {
				manifest, binding = f.manifest, f.binding
			}
			switch r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:] {
			case "manifest.json":
				_, _ = w.Write([]byte(manifest))
			case "binding.json":
				_, _ = w.Write([]byte(binding))
			default:
				_, _ = w.Write([]byte("graph"))
			}
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/import"):
			w.WriteHeader(http.StatusCreated)
		default:
			t.Logf("fake modelstore: unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	})
}

// loadingExecutor is a fake executor that records loads and can be made
// to forget its model (as an idle keep-alive eviction does).
type loadingExecutor struct {
	pb.UnimplementedExecutorServer
	mu           sync.Mutex
	loads        int
	evicted      bool
	loading      bool
	loadFailed   bool              // its asynchronous load failed: GetLoad reports so until the next LoadModel
	lastManifest string            // the manifest of the last LoadModel
	lastPlan     map[string]string // the plan of the last LoadModel
	refused      int               // LoadModels refused for their plan
	probe        string            // what Describe says (default: a g4-standard-96)
	served       []string          // the nodes of the last ServeNodes
	servedFrom   string            // the manifest of the last ServeNodes
}

// DescribeNodes: two layers of experts, each its own node, as a split
// decode step names them.
func (e *loadingExecutor) DescribeNodes(context.Context, *pb.DescribeNodesRequest) (*pb.DescribeNodesResponse, error) {
	return &pb.DescribeNodesResponse{NodesJson: `[{"name":"submod_0","kind":"dense"},` +
		`{"name":"submod_1","kind":"experts","collection":"experts","index":0,"resident_bytes":4000000000},` +
		`{"name":"submod_2","kind":"dense"},` +
		`{"name":"submod_3","kind":"experts","collection":"experts","index":1,"resident_bytes":4000000000}]`}, nil
}

func (e *loadingExecutor) ServeNodes(_ context.Context, req *pb.ServeNodesRequest) (*pb.ServeNodesResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.served, e.servedFrom = req.Nodes, req.ManifestJson
	return &pb.ServeNodesResponse{Port: 7000}, nil
}

func (e *loadingExecutor) Describe(context.Context, *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	if e.probe != "" {
		return &pb.DescribeResponse{HardwareJson: e.probe}, nil
	}
	return &pb.DescribeResponse{HardwareJson: `{"device":"cuda","cpu":"AMD EPYC 9B45 48-Core Processor",` +
		`"cpu_threads":96,"simd":"avx512","gpu":"NVIDIA RTX PRO 6000","gpu_memory_bytes":102005473280,"gpu_count":2}`}, nil
}

func (e *loadingExecutor) LoadModel(_ context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, bad := req.Plan["VXPU_REFUSED"]; bad { // as the executor refuses a switch it may not take
		e.refused++
		return nil, status.Error(codes.InvalidArgument, "plan refused: VXPU_REFUSED")
	}
	if string(req.DecodeGraph) != "graph" || !strings.Contains(req.ManifestJson, "/blobs/abc") {
		return nil, status.Error(codes.InvalidArgument, "unexpected artifact")
	}
	e.loads++
	e.evicted = false
	e.loadFailed = false
	e.lastManifest = req.ManifestJson
	e.lastPlan = req.Plan
	return &pb.LoadModelResponse{Success: true}, nil
}

const testTelemetry = `{"format":"vxpu-telemetry/1","by_kind_ms_per_step":{"dense":20,"experts":18},` +
	`"host_ops":{"decode":{"moe_experts":{"impl_ms":15}}},"experts":{"decode":{"0":{"1":5,"2":1},"1":{"3":6}}}}`

func (e *loadingExecutor) ExecuteChat(_ context.Context, req *pb.ExecuteChatRequest) (*pb.ExecuteChatResponse, error) {
	resp := &pb.ExecuteChatResponse{
		Response: &pb.ChatResponse{Text: "yes", SessionTokens: 12, NewPromptTokens: 2, Generated: 10, PrefillMs: 5, MsPerToken: 40},
	}
	if req.GetTelemetry() != pb.Telemetry_TELEMETRY_NONE {
		resp.TelemetryJson = testTelemetry
	}
	return resp, nil
}

func (e *loadingExecutor) GetLoad(context.Context, *pb.GetLoadRequest) (*pb.GetLoadResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.evicted {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_NONE}, nil
	}
	if e.loadFailed {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_FAILED, Error: "weights unreachable"}, nil
	}
	if e.loading {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_LOADING}, nil
	}
	if e.loads == 0 {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_NONE}, nil
	}
	return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY}, nil
}

func (e *loadingExecutor) NewSession(context.Context, *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.evicted {
		return nil, status.Error(codes.FailedPrecondition, "call LoadModel first")
	}
	if e.loadFailed {
		return nil, status.Error(codes.Internal, "model load failed: weights unreachable")
	}
	if e.loading {
		return nil, status.Error(codes.Unavailable, "model still loading")
	}
	return &pb.NewSessionResponse{SessionId: "s1"}, nil
}

func newCatalogServer(t *testing.T, store *fakeModelstore) (*Server, *fake.Clientset) {
	t.Helper()
	ms := httptest.NewServer(store.handler(t))
	t.Cleanup(ms.Close)
	bs, err := blobserver.NewServer(t.TempDir(), 0)
	if err != nil {
		t.Fatal(err)
	}
	clientset := fake.NewClientset()
	s := &Server{
		clientset: clientset, namespace: "ns", imageName: "registry/vxpu-executor:v42", blobServer: bs,
		weightStore: weightstore.New(ms.URL), executorPort: 50051,
		peerToModel: map[string]string{}, externalModel: map[string]string{}, modelEndpoint: map[string]string{},
		loadedTags: map[string]string{}, keyModel: map[string]modelRef{}, compileResources: defaultCompileResources(),
		compilePoll: 10 * time.Millisecond,
	}
	return s, clientset
}

func TestArtifactTag(t *testing.T) {
	for image, want := range map[string]string{
		"gcr.io/p/vxpu-executor:v42":           "0123456789ab_vxpu-executor-v42",
		"localhost:5000/vxpu-executor":         "0123456789ab_vxpu-executor-latest",
		"vxpu-executor:e2e":                    "0123456789ab_vxpu-executor-e2e",
		"registry-b/other-executor:latest":     "0123456789ab_other-executor-latest", // not the same entry as vxpu-executor:latest
		"gcr.io/p/x@sha256:" + testCommit + "": "0123456789ab_0123456789ab",
	} {
		if got := artifactTag(testCommit, image); got != want {
			t.Errorf("artifactTag(%q) = %q, want %q", image, got, want)
		}
	}
}

func TestRegisterModel_AlreadyCompiled(t *testing.T) {
	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})
	resp, err := s.RegisterModel(t.Context(), &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if resp.Compiled || resp.Commit != testCommit || resp.Artifact != "vxpu-artifacts/org/m:0123456789ab_vxpu-executor-v42" || resp.WeightBytes != 42 {
		t.Errorf("unexpected response %+v", resp)
	}
	if jobs, _ := clientset.BatchV1().Jobs("ns").List(t.Context(), metav1.ListOptions{}); len(jobs.Items) != 0 {
		t.Errorf("no compile job expected, got %d", len(jobs.Items))
	}
}

// An empty switch would be refused by every load: registering refuses it.
func TestRegisterModel_EmptySwitchRefused(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	_, err := s.RegisterModel(t.Context(), &pb.RegisterModelRequest{Model: "org/M", Plan: map[string]string{"VXPU_EXPERT_MAILBOXES": " "}})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("err %v, want InvalidArgument", err)
	}
}

func TestRegisterModel_PlanGoesWithEveryLoad(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})

	plan := map[string]string{"VXPU_EXPERT_MAILBOXES": "1"}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Plan: plan}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if !reflect.DeepEqual(executor.lastPlan, plan) {
		t.Errorf("the executor got plan %v, want %v", executor.lastPlan, plan)
	}
	// the coordinator asked the executor what it runs on (#394)
	for endpoint, info := range s.executors {
		if info.Machine == nil || info.Machine.Name != "g4-standard-96" {
			t.Errorf("executor %s: matched %+v, want g4-standard-96", endpoint, info.Machine)
		}
	}
	if len(s.executors) != 1 {
		t.Errorf("want the one executor described, got %d", len(s.executors))
	}
	// a new plan reloads the model under it; registering without one keeps it
	plan = map[string]string{"VXPU_EXPERT_MAILBOXES": "0"}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Plan: plan}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if executor.loads != 2 || !reflect.DeepEqual(executor.lastPlan, plan) {
		t.Errorf("after a new plan: %d loads with plan %v, want 2 with %v", executor.loads, executor.lastPlan, plan)
	}
}

func TestLoadModel_RefusesAPlan(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{})
	_, err := s.LoadModel(t.Context(), &pb.LoadModelRequest{ManifestJson: testManifest, Plan: map[string]string{"VXPU_DATAFLOW": "1"}})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("a plan with an artifact: got %v, want INVALID_ARGUMENT (plans come with loads by name)", err)
	}
}

func TestRegisterModel_RefusedPlanFailsOnce(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Plan: map[string]string{"VXPU_REFUSED": "1"}}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	// final, not retried: the caller sees why at once
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("a refused plan: got %v, want INVALID_ARGUMENT", err)
	}
	if executor.refused != 1 {
		t.Errorf("the refused load should not be retried, it was tried %d times", executor.refused)
	}
}

// TestNewSessionByName_PlansFromTheModelsTraffic: the coordinator places
// a model's experts from its recorded selections (#394): none before it
// has traffic, the selected ones on the GPUs after.
func TestNewSessionByName_PlansFromTheModelsTraffic(t *testing.T) {
	ctx := t.Context()
	// two layers of four 1 GB experts (testTelemetry selected experts 1
	// and 2 of layer 0, and 3 of layer 1)
	manifest := strings.Replace(testManifest, `"tensors":{}`, `"tensors":{"e":{"length":1000000000}}`, 1)
	layer := `{"w1":{"parts":[["e"],["e"],["e"],["e"]]}}`
	binding := `{"bound":{"embed":{"length":1000}},"external":{"experts":{"0":` + layer + `,"1":` + layer + `}}}`
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true, manifest: manifest, binding: binding})
	s.UseStats(1)
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})

	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if _, placed := executor.lastPlan["VXPU_GPU_EXPERTS"]; placed {
		t.Errorf("without traffic, nothing should be placed: %v", executor.lastPlan)
	}
	s.stats.record(ctx, requestRecord{Format: requestFormat, Model: "org/M", RPC: "chat", Telemetry: json.RawMessage(testTelemetry)})

	executor.mu.Lock()
	executor.evicted = true // the next load plans again
	executor.mu.Unlock()
	_, _ = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}) // "being reloaded"
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession after eviction: %v", err)
	}
	var placement struct {
		Gpus []struct{ Layers map[string][]int }
	}
	if err := json.Unmarshal([]byte(executor.lastPlan["VXPU_GPU_EXPERTS"]), &placement); err != nil {
		t.Fatalf("the second load should place the selected experts: %v (plan %v)", err, executor.lastPlan)
	}
	if got := placement.Gpus[0].Layers; !reflect.DeepEqual(got, map[string][]int{"0": {1, 2}, "1": {3}}) {
		t.Errorf("GPU0 holds %v, want the three selected experts", got)
	}
	explained, err := s.ExplainModel(ctx, &pb.ExplainModelRequest{Model: "org/M"})
	if err != nil || !strings.Contains(explained.PlanJson, "experts: 3 of 8") || !strings.Contains(explained.PlanJson, "g4-standard-96") {
		t.Errorf("ExplainModel: %v\n%v", err, explained)
	}
}

// TestNewSessionByName_SpreadsExpertsOverExecutors: an engine whose RAM
// holds one of two layers of experts gets the other from a free executor
// the coordinator asks to serve it, wired to its mailboxes (#394).
func TestNewSessionByName_SpreadsExpertsOverExecutors(t *testing.T) {
	ctx := t.Context()
	manifest := strings.Replace(testManifest, `"tensors":{}`, `"tensors":{"e":{"length":1000000000}}`, 1)
	layer := `{"w1":{"parts":[["e"],["e"],["e"],["e"]]}}` // 4 GB a layer
	binding := `{"bound":{"embed":{"length":1000}},"external":{"experts":{"0":` + layer + `,"1":` + layer + `}}}`
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true, manifest: manifest, binding: binding})
	// 0.85 x 26 GB - 16 GiB: room for one layer; the other machine, for both
	engine := &loadingExecutor{probe: `{"device":"cpu","cpu":"AMD EPYC 9B14","cpu_threads":8,"ram_bytes":26000000000}`}
	helper := &loadingExecutor{probe: `{"device":"cpu","cpu":"AMD EPYC 9B14","cpu_threads":60,"ram_bytes":100000000000}`}
	helperAddr := startExecutor(t, helper)
	s.UseExternalExecutors([]string{startExecutor(t, engine), helperAddr})

	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if !reflect.DeepEqual(helper.served, []string{"submod_3"}) {
		t.Errorf("the other executor should serve layer 1's experts node, served %v", helper.served)
	}
	if !strings.Contains(helper.servedFrom, "/blobs/abc") {
		t.Errorf("the expert executor should read the weights from where the engine does (the rewritten manifest): %s", helper.servedFrom)
	}
	want := map[string]string{"VXPU_REMOTE_NODES": "submod_3=tcp://127.0.0.1:7000/0"}
	if !reflect.DeepEqual(engine.lastPlan, want) {
		t.Errorf("the engine's plan: got %v, want %v", engine.lastPlan, want)
	}
	if helper.loads != 0 {
		t.Errorf("the expert executor should not be loaded as an engine: %d loads", helper.loads)
	}
	explained, err := s.ExplainModel(ctx, &pb.ExplainModelRequest{Model: "org/M"})
	if err != nil || !strings.Contains(explained.PlanJson, "serves the nodes of layers 1-1") {
		t.Errorf("ExplainModel: %v\n%v", err, explained)
	}
	// a client cannot name the node servers an engine trusts
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Plan: want}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("a registered plan naming expert servers: got %v, want INVALID_ARGUMENT", err)
	}
}

func TestRegisterModel_RunsCompileJob(t *testing.T) {
	store := &fakeModelstore{}
	s, clientset := newCatalogServer(t, store)
	// The "cluster": when the compile job appears, it succeeds and the
	// artifact appears in modelstore.
	ctx := t.Context()
	go func() {
		for {
			jobs, err := clientset.BatchV1().Jobs("ns").List(ctx, metav1.ListOptions{})
			if err == nil && len(jobs.Items) == 1 {
				job := jobs.Items[0]
				store.mu.Lock()
				store.compiled = true
				store.mu.Unlock()
				job.Status.Succeeded = 1
				_, _ = clientset.BatchV1().Jobs("ns").UpdateStatus(ctx, &job, metav1.UpdateOptions{})
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()
	resp, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if !resp.Compiled || resp.WeightBytes != 42 {
		t.Errorf("unexpected response %+v", resp)
	}
	jobs, _ := clientset.BatchV1().Jobs("ns").List(t.Context(), metav1.ListOptions{})
	if len(jobs.Items) != 1 {
		t.Fatalf("one compile job expected, got %d", len(jobs.Items))
	}
	c := jobs.Items[0].Spec.Template.Spec.Containers[0]
	args := strings.Join(append(append([]string{}, c.Command...), c.Args...), " ")
	for _, want := range []string{"python -m vxpu.compile org/M", "--tag 0123456789ab_vxpu-executor-v42", "--revision " + testCommit, "--modelstore http"} {
		if !strings.Contains(args, want) {
			t.Errorf("compile job command %q lacks %q", args, want)
		}
	}
	if c.Image != "registry/vxpu-executor:v42" {
		t.Errorf("compile job image %q, want the executor image", c.Image)
	}
}

func TestNewSessionByName_ExternalExecutor(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	executor := &loadingExecutor{}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()
	s.UseExternalExecutors([]string{lis.Addr().String()})

	// unregistered models are refused
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/Other"}); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("unregistered model: got %v, want FAILED_PRECONDITION", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	modelKey, backend, err := splitSessionID(resp.SessionId)
	if err != nil || backend != "s1" {
		t.Fatalf("session id %q: %v", resp.SessionId, err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("second NewSession: %v", err)
	}
	if executor.loads != 1 {
		t.Errorf("the artifact should be loaded once, loaded %d times", executor.loads)
	}
	// sessions route to the external executor, not to a pod
	if endpoint, err := s.executorEndpoint(ctx, modelKey); err != nil || endpoint != lis.Addr().String() {
		t.Errorf("executorEndpoint = %q, %v; want the external executor", endpoint, err)
	}
	// the executor evicts the model: the next call says it is being
	// reloaded, and the one after that loads it again
	executor.mu.Lock()
	executor.evicted = true
	executor.mu.Unlock()
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("after eviction: got %v, want FAILED_PRECONDITION", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession after eviction: %v", err)
	}
	if executor.loads != 2 {
		t.Errorf("the evicted model should be loaded again, loads = %d", executor.loads)
	}
}

// startExecutor serves a fake executor on a local port.
func startExecutor(t *testing.T, executor pb.ExecutorServer) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return lis.Addr().String()
}

func TestNewSessionByName_FailedLoadIsForgotten(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	// the executor's asynchronous load fails (it answers so until the next
	// LoadModel): the router forgets the placement instead of passing the
	// same error on forever, and the next request loads the model again
	executor.mu.Lock()
	executor.loadFailed = true
	executor.mu.Unlock()
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.Unavailable {
		t.Errorf("after a failed load: got %v, want UNAVAILABLE", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession after a failed load: %v", err)
	}
	if executor.loads != 2 {
		t.Errorf("the model should be loaded again after its load failed, loads = %d", executor.loads)
	}
}

func TestNewSessionByName_LoadingStateReportsFailedPrecondition(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{loading: true}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("while loading: got %v, want FAILED_PRECONDITION", err)
	}
	executor.mu.Lock()
	executor.loading = false
	executor.mu.Unlock()
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession once ready: %v", err)
	}
}

func TestNewSessionByName_ExternalModelstoreURL(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	s.UseExternalModelstoreURL("http://10.0.0.7:8080/")
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	if !strings.Contains(executor.lastManifest, `"http://10.0.0.7:8080/blobs/abc"`) {
		t.Errorf("an external executor should fetch from the external modelstore URL; its manifest: %s", executor.lastManifest)
	}
}

func TestRegisterModel_ReplacesFinishedCompileJob(t *testing.T) {
	store := &fakeModelstore{}
	s, clientset := newCatalogServer(t, store)
	ctx := t.Context()
	// a compile that failed earlier, its Job still present (TTL 24 h)
	name := compileJobName(weightstore.ArtifactRepository("org/M"), artifactTag(testCommit, s.imageName))
	stale := &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns"},
		Status: batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}}}
	if _, err := clientset.BatchV1().Jobs("ns").Create(ctx, stale, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	// the "cluster": a fresh compile job (no conditions) succeeds and the
	// artifact appears in modelstore
	go func() {
		for {
			job, err := clientset.BatchV1().Jobs("ns").Get(ctx, name, metav1.GetOptions{})
			if err == nil && len(job.Status.Conditions) == 0 && job.Status.Succeeded == 0 {
				store.mu.Lock()
				store.compiled = true
				store.mu.Unlock()
				job.Status.Succeeded = 1
				_, _ = clientset.BatchV1().Jobs("ns").UpdateStatus(ctx, job, metav1.UpdateOptions{})
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()
	resp, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel with a stale failed compile job: %v", err)
	}
	if !resp.Compiled {
		t.Errorf("the model should have been compiled again, got %+v", resp)
	}
}

func TestRegisterModel_UnschedulableCompile(t *testing.T) {
	s, clientset := newCatalogServer(t, &fakeModelstore{})
	s.compileUnschedulable = 20 * time.Millisecond
	ctx := t.Context()
	name := compileJobName(weightstore.ArtifactRepository("org/M"), artifactTag(testCommit, s.imageName))
	// the "cluster": the compile job's pod finds no node with the CPU it asks for
	go func() {
		for {
			if _, err := clientset.BatchV1().Jobs("ns").Get(ctx, name, metav1.GetOptions{}); err == nil {
				pod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: name + "-x", Namespace: "ns", Labels: map[string]string{"vxpu.gke-labs.dev/compile": name}},
					Status: corev1.PodStatus{Phase: corev1.PodPending, Conditions: []corev1.PodCondition{{
						Type: corev1.PodScheduled, Status: corev1.ConditionFalse, Reason: corev1.PodReasonUnschedulable,
						Message: "0/1 nodes are available: 1 Insufficient cpu.", LastTransitionTime: metav1.Now()}}},
				}
				_, _ = clientset.CoreV1().Pods("ns").Create(ctx, pod, metav1.CreateOptions{})
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()
	_, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"})
	if status.Code(err) != codes.ResourceExhausted || !strings.Contains(status.Convert(err).Message(), "Insufficient cpu") {
		t.Fatalf("RegisterModel with an unschedulable compile pod: got %v, want RESOURCE_EXHAUSTED with the scheduler's reason", err)
	}
}
