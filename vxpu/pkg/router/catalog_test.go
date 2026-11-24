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
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/blobserver"
	"github.com/gke-labs/generation-ai/vxpu/pkg/collections"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

const testCommit = "0123456789abcdef0123"

var testManifest = `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"org/M","revision":"main","commit_sha":"` + testCommit + `"},` +
	`"config":{},"files":{"abc":{"size":42,"name":"model.safetensors","source":"https://huggingface.co/org/M/resolve/c/model.safetensors"}},` +
	`"tensors":{}}`

// fakeModelstore serves the routes the catalog uses for one model, org/M.
type fakeModelstore struct {
	mu sync.Mutex
	// compiled reports whether the artifact for testCommit exists.
	compiled bool
	// failGolden causes golden lookups to fail with 503 Service Unavailable.
	failGolden bool
	// manifest is the artifact's manifest.json content (defaults to testManifest).
	manifest string
	// binding is the artifact's binding.json content (defaults to "{}").
	binding string
	// blobs maps content digests to blob bytes, written by the registry when storing statistics.
	blobs map[string][]byte
	// manifests maps artifact tags to manifests.
	manifests map[string][]byte
	// importGate, when non-nil, blocks /import responses until closed.
	importGate chan struct{}
	// importStarted, when non-nil, is closed when an /import request arrives.
	importStarted chan struct{}
}

// registry serves the generic registry routes statistics and plans use,
// and the file route plans are read by; false if r is not one of them.
func (f *fakeModelstore) registry(w http.ResponseWriter, r *http.Request) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.blobs == nil {
		f.blobs, f.manifests = map[string][]byte{}, map[string][]byte{}
	}
	if name, file, ok := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/resolve/"); ok && (strings.HasPrefix(name, "vxpu-plans/") || strings.HasPrefix(name, "vxpu-golden/")) {
		if strings.HasPrefix(name, "vxpu-golden/") && f.failGolden {
			http.Error(w, "modelstore unavailable", http.StatusServiceUnavailable)
			return true
		}
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
	if !strings.HasPrefix(path, "vxpu-stats/") && !strings.HasPrefix(path, "vxpu-plans/") && !strings.HasPrefix(path, "vxpu-golden/") {
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
			if f.importStarted != nil {
				select {
				case <-f.importStarted:
				default:
					close(f.importStarted)
				}
			}
			if f.importGate != nil {
				<-f.importGate
			}
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
	mu      sync.Mutex
	loads   int
	evicted bool
	loading bool
	// loadFailed causes GetLoad to report an asynchronous load failure until the next LoadModel.
	loadFailed bool
	// lastManifest records the manifest provided to the most recent LoadModel call.
	lastManifest string
	// lastPlan records the plan provided to the most recent LoadModel call.
	lastPlan map[string]string
	// lastPlanFile records the plan file provided to the most recent LoadModel call.
	lastPlanFile *planpb.PlanFile
	// lastPlanFileRaw records the marshaled bytes of the plan file provided to the most recent LoadModel call.
	lastPlanFileRaw []byte
	// refused counts how many LoadModel calls were rejected because of their plan.
	refused int
	// probe is what Describe returns, defaulting to a g4-standard-96 probe.
	probe string
	// emptyProbe forces Describe to return an empty HardwareJson instead of the default probe.
	emptyProbe bool
	// describeErr, when non-nil, is returned by Describe.
	describeErr error
	// describes counts how many times Describe was called on this executor.
	describes int
	// acceptAnyGraph skips the dummy "graph" DecodeGraph check when true.
	acceptAnyGraph bool
	// served records the nodes provided to the most recent ServeNodes call.
	served []string
	// servedFrom records the manifest provided to the most recent ServeNodes call.
	servedFrom string
	// readyBetween causes NewSession to fail once as loading, after which GetLoad reports READY.
	readyBetween bool
	// loadGate blocks in LoadModel until closed, if not nil.
	loadGate chan struct{}
	// loadStarted is closed when LoadModel begins, if not nil.
	loadStarted chan struct{}
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
	e.mu.Lock()
	e.describes++
	e.mu.Unlock()
	if e.describeErr != nil {
		return nil, e.describeErr
	}
	if e.emptyProbe {
		return &pb.DescribeResponse{}, nil
	}
	if e.probe != "" {
		return &pb.DescribeResponse{HardwareJson: e.probe}, nil
	}
	return &pb.DescribeResponse{HardwareJson: `{"device":"cuda","cpu":"AMD EPYC 9B45 48-Core Processor",` +
		`"cpu_threads":96,"simd":"avx512","gpu":"NVIDIA RTX PRO 6000","gpu_memory_bytes":102005473280,"gpu_count":2}`}, nil
}

func (e *loadingExecutor) LoadModel(_ context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	started := e.loadStarted
	gate := e.loadGate
	e.mu.Unlock()
	if started != nil {
		select {
		case <-started:
		default:
			close(started)
		}
	}
	if gate != nil {
		<-gate
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, bad := req.Plan["VXPU_REFUSED"]; bad { // as the executor refuses a switch it may not take
		e.refused++
		return nil, status.Error(codes.InvalidArgument, "plan refused: VXPU_REFUSED")
	}
	if !e.acceptAnyGraph && (string(req.DecodeGraph) != "graph" || !strings.Contains(req.ManifestJson, "/blobs/abc")) {
		return nil, status.Error(codes.InvalidArgument, "unexpected artifact")
	}
	e.loads++
	e.evicted = false
	e.loadFailed = false
	e.lastManifest = req.ManifestJson
	e.lastPlan = req.Plan
	e.lastPlanFile = req.PlanFile
	if req.PlanFile != nil {
		raw, _ := proto.Marshal(req.PlanFile)
		e.lastPlanFileRaw = raw
	} else {
		e.lastPlanFileRaw = nil
	}
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
	if e.readyBetween {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY}, nil
	}
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
	if e.readyBetween {
		e.readyBetween = false
		return nil, status.Error(codes.FailedPrecondition, "model still loading")
	}
	if e.evicted {
		return nil, status.Error(codes.FailedPrecondition, "call LoadModel first")
	}
	if e.loadFailed {
		return nil, status.Error(codes.Internal, "model load failed: weights unreachable")
	}
	if e.loading {
		return nil, status.Error(codes.FailedPrecondition, "model still loading")
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
	agent := &fakeAgent{}
	s := &Server{
		clientset:    clientset,
		namespace:    "ns",
		imageName:    "registry/vxpu-executor:v42",
		blobServer:   bs,
		weightStore:  weightstore.New(ms.URL),
		executorPort: 50051,
		peerToModel:  map[string]string{}, externalModel: map[string]string{}, modelEndpoint: map[string]string{},
		loadedTags: map[string]string{}, keyModel: map[string]modelRef{}, compileResources: defaultCompileResources(),
		compilePoll: 10 * time.Millisecond,
		flights:     collections.NewDeduplicator[string, string](),
	}
	s.reconciler = NewModelReconciler(s)
	s.reconciler.Start(t.Context())
	s.agentDialer = func(addr string) (*grpc.ClientConn, error) {
		agent.endpoint = fmt.Sprintf("127.0.0.1:%d", s.executorPort)
		return fakeAgentDialer(t, agent)(addr)
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
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})

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

func TestLoadModel_RefusesRouterOwnedSwitches(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{})
	ctx := t.Context()

	// Every router-owned switch must be refused with InvalidArgument naming the switch.
	for _, sw := range routerOwnedSwitches {
		t.Run(sw, func(t *testing.T) {
			_, err := s.LoadModel(ctx, &pb.LoadModelRequest{
				ManifestJson: testManifest,
				Plan:         map[string]string{sw: "foo"},
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("LoadModel with router-owned switch %s: got %v, want INVALID_ARGUMENT", sw, err)
			}
			if !strings.Contains(err.Error(), sw) {
				t.Errorf("error message %q does not name switch %q", err.Error(), sw)
			}
		})
	}

	// Main's refusal of switch maps on load by artifact is preserved.
	t.Run("switch_map_refused", func(t *testing.T) {
		_, err := s.LoadModel(ctx, &pb.LoadModelRequest{
			ManifestJson: testManifest,
			Plan:         map[string]string{"VXPU_DATAFLOW": "1"},
		})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("LoadModel with switch map: got %v, want INVALID_ARGUMENT", err)
		}
		if !strings.Contains(err.Error(), "a plan comes with a model loaded by name") {
			t.Errorf("error message %q does not match main's switch map refusal", err.Error())
		}
	})
}

func TestLoadModel_RefusesBogusPlanHash(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{})

	req := &pb.LoadModelRequest{
		ManifestJson: testManifest,
		PlanFile: &planpb.PlanFile{
			ArtifactMd5: "0123456789abcdef0123456789abcdef",
			PlanHash:    "bogus",
		},
	}
	_, err := s.LoadModel(ctx, req)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("LoadModel with bogus plan_hash: got %v, want INVALID_ARGUMENT", err)
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error message %q does not name bogus hash", err.Error())
	}
}

func TestLoadModel_PlanFileReachesFakeExecutorByteForByte(t *testing.T) {
	ctx := t.Context()
	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})

	executor := &loadingExecutor{}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	modelKey := artifactKey(testManifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + modelKey, Namespace: "ns"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	if _, err := clientset.CoreV1().Pods("ns").Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Create pod: %v", err)
	}

	plan := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  testCommit,
		Sites: []*planpb.PlanSite{
			{
				Id:        "model.layers.0.attn.wq.weight",
				Kind:      planpb.SiteKind_SITE_KIND_DENSE_GEMV,
				Shape:     []int32{128, 256},
				RowCounts: []int32{1, 8},
				Strategy:  "gpu-fp8-gemv",
				Body:      "gemv_body",
			},
		},
		Knobs: &planpb.Knobs{
			GEMV_ROWS_MAX: proto.Int32(32),
		},
	}
	if err := kernelplan.EnsurePlanIdentity(plan); err != nil {
		t.Fatalf("EnsurePlanIdentity: %v", err)
	}
	expectedBytes, err := proto.Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal plan: %v", err)
	}

	req := &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
		PrefillGraph: []byte("graph"),
		DecodeGraph:  []byte("graph"),
		PlanFile:     plan,
	}
	if _, err := s.LoadModel(ctx, req); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if executor.loads != 1 {
		t.Fatalf("expected 1 load, got %d", executor.loads)
	}
	if !bytes.Equal(executor.lastPlanFileRaw, expectedBytes) {
		t.Errorf("plan bytes did not reach executor byte for byte:\ngot:  %x\nwant: %x", executor.lastPlanFileRaw, expectedBytes)
	}
	if !proto.Equal(executor.lastPlanFile, plan) {
		t.Errorf("plan proto did not match: got %v, want %v", executor.lastPlanFile, plan)
	}
}

func startCatalogPodExecutorWithStore(t *testing.T, manifest string, executor *loadingExecutor, ms *fakeModelstore) *Server {
	t.Helper()
	ctx := t.Context()
	s, clientset := newCatalogServer(t, ms)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + artifactKey(manifest), Namespace: "ns"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	if _, err := clientset.CoreV1().Pods("ns").Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Create pod: %v", err)
	}
	return s
}

func startCatalogPodExecutor(t *testing.T, manifest string, executor *loadingExecutor) *Server {
	return startCatalogPodExecutorWithStore(t, manifest, executor, &fakeModelstore{compiled: true})
}

func TestLoadModel_EmitsPlanAndReloads(t *testing.T) {
	ctx := t.Context()
	executor := &loadingExecutor{acceptAnyGraph: true}
	s := startCatalogPodExecutor(t, testManifest, executor)
	modelKey := artifactKey(testManifest)

	baseReq := &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
		PrefillGraph: []byte("graph"),
	}
	if _, err := s.LoadModel(ctx, baseReq); err != nil {
		t.Fatalf("initial LoadModel: %v", err)
	}
	if baseReq.PlanFile != nil {
		t.Error("LoadModel must not mutate caller's LoadModelRequest.PlanFile")
	}
	if desired, ok := s.reconciler.Desired(modelKey); !ok || desired.Request.PlanFile != nil {
		t.Fatalf("desired state must leave PlanFile unset on no-plan load, got ok=%v plan=%v", ok, desired.Request)
	}
	if executor.loads != 1 || executor.lastPlanFile == nil || executor.lastPlanFile.PlanHash == "" {
		t.Fatalf("expected 1 load with non-empty emitted PlanHash, got loads=%d plan=%v", executor.loads, executor.lastPlanFile)
	}
	emitted := proto.Clone(executor.lastPlanFile).(*planpb.PlanFile)
	if st := s.ModelStatus(modelKey); st.Phase != InstancePhasePassed || st.PlanHash != emitted.PlanHash {
		t.Fatalf("status = %+v, want Phase=Passed PlanHash=%q", st, emitted.PlanHash)
	}

	diffPlan := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		Sites: []*planpb.PlanSite{
			{Id: "model.layers.0.attn.wq.weight", Kind: planpb.SiteKind_SITE_KIND_DENSE_GEMV, Strategy: "reference"},
		},
	}

	for _, tc := range []struct {
		name               string
		plan               *planpb.PlanFile
		wantLoads          int
		wantDesiredHasPlan bool
		wantPlanHash       string
		wantGenUnchanged   bool
	}{
		{name: "no_plan", plan: nil, wantLoads: 1, wantDesiredHasPlan: false, wantPlanHash: emitted.PlanHash, wantGenUnchanged: true},
		{name: "different_explicit_plan", plan: diffPlan, wantLoads: 2, wantDesiredHasPlan: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prevGen := s.ModelStatus(modelKey).ObservedGeneration
			req := proto.Clone(baseReq).(*pb.LoadModelRequest)
			req.PlanFile = tc.plan
			if _, err := s.LoadModel(ctx, req); err != nil {
				t.Fatalf("LoadModel(%s): %v", tc.name, err)
			}
			if executor.loads != tc.wantLoads {
				t.Errorf("loads = %d, want %d", executor.loads, tc.wantLoads)
			}
			desired, ok := s.reconciler.Desired(modelKey)
			if !ok || (desired.Request.PlanFile != nil) != tc.wantDesiredHasPlan {
				t.Errorf("desired has PlanFile = %v, want %v", desired.Request.PlanFile != nil, tc.wantDesiredHasPlan)
			}
			st := s.ModelStatus(modelKey)
			wantHash := tc.wantPlanHash
			if wantHash == "" && tc.plan != nil {
				wantHash = tc.plan.PlanHash
			}
			if st.Phase != InstancePhasePassed || st.PlanHash != wantHash {
				t.Errorf("status = %+v, want Phase=Passed PlanHash=%q", st, wantHash)
			}
			if tc.wantGenUnchanged && st.ObservedGeneration != prevGen {
				t.Errorf("ObservedGeneration = %d, want unchanged %d", st.ObservedGeneration, prevGen)
			}
			if tc.plan == diffPlan && !proto.Equal(executor.lastPlanFile, diffPlan) {
				t.Errorf("executor did not receive new plan: got %v, want %v", executor.lastPlanFile, diffPlan)
			}
		})
	}
}

func TestLoadModel_PlanEmitWaitsForManifestRewrite(t *testing.T) {
	ctx := t.Context()
	importGate := make(chan struct{})
	importStarted := make(chan struct{})
	ms := &fakeModelstore{
		compiled:      true,
		importGate:    importGate,
		importStarted: importStarted,
	}
	executor := &loadingExecutor{acceptAnyGraph: true}
	s := startCatalogPodExecutorWithStore(t, testManifest, executor, ms)
	s.reconciler.pollInterval = 5 * time.Millisecond

	done := make(chan error, 1)
	go func() {
		_, err := s.LoadModel(ctx, &pb.LoadModelRequest{
			ManifestJson: testManifest,
			BindingJson:  "{}",
			PrefillGraph: []byte("graph"),
		})
		done <- err
	}()

	select {
	case <-importStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for background manifest import to start")
	}

	// Let several poll intervals pass while weights are still downloading.
	time.Sleep(40 * time.Millisecond)
	executor.mu.Lock()
	describesDuringDownload := executor.describes
	executor.mu.Unlock()
	if describesDuringDownload != 0 {
		t.Errorf("Describe called %d times during background manifest caching, want 0", describesDuringDownload)
	}

	close(importGate)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("LoadModel: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for LoadModel to complete")
	}

	executor.mu.Lock()
	describesAfter := executor.describes
	loads := executor.loads
	executor.mu.Unlock()
	if describesAfter != 1 {
		t.Errorf("Describe calls = %d, want 1", describesAfter)
	}
	if loads != 1 {
		t.Errorf("loads = %d, want 1", loads)
	}
	if st := s.ModelStatus(artifactKey(testManifest)); st.Phase != InstancePhasePassed || st.PlanHash == "" {
		t.Errorf("status = %+v, want Phase=Passed with non-empty PlanHash", st)
	}
}

func TestLoadModel_PlanEmitErrors(t *testing.T) {
	ctx := t.Context()

	for _, tc := range []struct {
		name       string
		executor   *loadingExecutor
		req        *pb.LoadModelRequest
		wantCode   codes.Code
		wantSubstr string
	}{
		{
			name:       "corrupt_decode_graph",
			executor:   &loadingExecutor{},
			req:        &pb.LoadModelRequest{ManifestJson: testManifest, BindingJson: "{}", DecodeGraph: []byte("not a zip")},
			wantCode:   codes.InvalidArgument,
			wantSubstr: "read segments",
		},
		{
			name:       "malformed_binding",
			executor:   &loadingExecutor{},
			req:        &pb.LoadModelRequest{ManifestJson: testManifest, BindingJson: "{not json"},
			wantCode:   codes.InvalidArgument,
			wantSubstr: "parse binding",
		},
		{
			name:       "describe_unimplemented",
			executor:   &loadingExecutor{describeErr: status.Error(codes.Unimplemented, "Describe not implemented")},
			req:        &pb.LoadModelRequest{ManifestJson: testManifest, BindingJson: "{}"},
			wantCode:   codes.FailedPrecondition,
			wantSubstr: "executor machine is not known",
		},
		{
			name:       "empty_hardware_json",
			executor:   &loadingExecutor{emptyProbe: true},
			req:        &pb.LoadModelRequest{ManifestJson: testManifest, BindingJson: "{}"},
			wantCode:   codes.FailedPrecondition,
			wantSubstr: "executor machine is not known",
		},
		{
			name:       "probe_without_gpu_or_cpu",
			executor:   &loadingExecutor{probe: `{"cpu_threads":4}`},
			req:        &pb.LoadModelRequest{ManifestJson: testManifest, BindingJson: "{}"},
			wantCode:   codes.FailedPrecondition,
			wantSubstr: "executor machine is not known",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := startCatalogPodExecutor(t, testManifest, tc.executor)
			_, err := s.LoadModel(ctx, tc.req)
			if status.Code(err) != tc.wantCode {
				t.Fatalf("code = %v, want %v (err: %v)", status.Code(err), tc.wantCode, err)
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantSubstr)
			}
			if tc.executor.loads != 0 {
				t.Errorf("failed emit must not load on executor: got %d loads, want 0", tc.executor.loads)
			}
			st := s.ModelStatus(artifactKey(tc.req.ManifestJson))
			if st.Phase != InstancePhaseRefused {
				t.Errorf("status phase = %v, want %v", st.Phase, InstancePhaseRefused)
			}
			if st.PlanHash != "" {
				t.Errorf("refused status PlanHash = %q, want empty", st.PlanHash)
			}
			if st.LastError == nil || !strings.Contains(st.LastError.Error(), tc.wantSubstr) {
				t.Errorf("status LastError = %v, want substring %q", st.LastError, tc.wantSubstr)
			}
		})
	}
}

func makePT2DecodeGraph(t *testing.T, nodesJSON, outputTensor string) []byte {
	t.Helper()
	var pt2Buf bytes.Buffer
	zw := zip.NewWriter(&pt2Buf)
	modelJSON := fmt.Sprintf(`{
		"schema_version": {"major": 8, "minor": 20},
		"graph_module": {
			"signature": {
				"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}],
				"output_specs": [{"user_output": {"arg": {"as_tensor": {"name": %q}}}}]
			},
			"graph": {"nodes": %s}
		}
	}`, outputTensor, nodesJSON)
	for name, content := range map[string]string{
		"decode/archive_format":    "pt2",
		"decode/archive_version":   "0",
		"decode/models/model.json": modelJSON,
	} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("zip Create %s: %v", name, err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatalf("zip Write %s: %v", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip Close: %v", err)
	}
	return pt2Buf.Bytes()
}

func probeJSONForMachine(t *testing.T, name string) string {
	t.Helper()
	m, err := machines.Lookup(name)
	if err != nil {
		t.Fatalf("machines.Lookup(%q): %v", name, err)
	}
	data, err := json.Marshal(machines.ProbeOf(m))
	if err != nil {
		t.Fatalf("marshal probe for %q: %v", name, err)
	}
	return string(data)
}

func TestLoadModel_EmittedIdentityMatchesCLIPlanEmit(t *testing.T) {
	ctx := t.Context()
	tmpDir := t.TempDir()

	vxpuBin := filepath.Join(tmpDir, "vxpu")
	if out, err := exec.CommandContext(ctx, "go", "build", "-o", vxpuBin, "../../cmd/vxpu").CombinedOutput(); err != nil {
		t.Fatalf("go build vxpu: %v\n%s", err, string(out))
	}

	tinyLlamaNodes := `[
		{"name": "linear_q", "target": "torch.ops.aten.linear.default", "outputs": [{"as_tensor": {"name": "linear_q"}}]},
		{"name": "linear_gate", "target": "torch.ops.aten.linear.default", "outputs": [{"as_tensor": {"name": "linear_gate"}}]}
	]`
	customOpsNodes := `[
		{
			"name": "fp8_linear_0",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"outputs": [{"as_tensor": {"name": "fp8_linear_0"}}],
			"metadata": {"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.attn.wq_a,model.layers.0.attn.wq_a,Linear"}
		},
		{
			"name": "moe_experts_0",
			"target": "torch.ops.vxpu.moe_experts.default",
			"outputs": [{"as_tensor": {"name": "moe_experts_0"}}],
			"metadata": {"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.ffn,model.layers.0.ffn,MoE"}
		},
		{
			"name": "fp8_linear_1",
			"target": "torch.ops.vxpu.fp8_linear.default",
			"outputs": [{"as_tensor": {"name": "fp8_linear_1"}}],
			"metadata": {"nn_module_stack": "L__self__,,Decode;L__self__model.layers.0.ffn.shared_experts.w1,model.layers.0.ffn.shared_experts.w1,Linear"}
		}
	]`
	customManifest := `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"vxpu-test/custom-ops-model","revision":"main","commit_sha":"def456"},"config":{"model_type":"deepseek_v3"},"files":{"abc":{"size":42,"name":"model.safetensors","source":"https://example.invalid/model.safetensors"}},"tensors":{}}`
	customBinding := `{"bound":{"model.layers.0.attn.wq_a.weight":{"shape":[16,16],"dtype":"float8_e4m3fn"}}}`

	hashes := map[string]string{}
	for _, tc := range []struct {
		name           string
		manifest       string
		binding        string
		nodes          string
		outputTensor   string
		machine        string
		wantSegmentRun planpb.SegmentRun
		wantOpBody     string
	}{
		{
			name:           "tiny_llama_cpu",
			manifest:       `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"trl-internal-testing/tiny-random-LlamaForCausalLM","revision":"main","commit_sha":"abc123"},"config":{"model_type":"llama"},"files":{"abc":{"size":42,"name":"model.safetensors","source":"https://huggingface.co/trl-internal-testing/tiny-random-LlamaForCausalLM/resolve/abc123/model.safetensors"}},"tensors":{}}`,
			binding:        `{"bound":{"model.layers.0.self_attn.q_proj.weight":{"shape":[16,16],"dtype":"float32"},"model.layers.0.mlp.gate_proj.weight":{"shape":[32,16],"dtype":"float32"}}}`,
			nodes:          tinyLlamaNodes,
			outputTensor:   "linear_gate",
			machine:        "c3d-standard-60",
			wantSegmentRun: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH,
		},
		{
			name:           "custom_ops_gpu_g2_standard_48",
			manifest:       customManifest,
			binding:        customBinding,
			nodes:          customOpsNodes,
			outputTensor:   "fp8_linear_1",
			machine:        "g2-standard-48",
			wantSegmentRun: planpb.SegmentRun_SEGMENT_RUN_CAPTURED,
			wantOpBody:     "gpu-bf16-copy",
		},
		{
			name:           "custom_ops_cpu_c3d_standard_60",
			manifest:       customManifest,
			binding:        customBinding,
			nodes:          customOpsNodes,
			outputTensor:   "fp8_linear_1",
			machine:        "c3d-standard-60",
			wantSegmentRun: planpb.SegmentRun_SEGMENT_RUN_PACKAGE,
			wantOpBody:     "reference",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			artifactDir := filepath.Join(tmpDir, tc.name)
			if err := os.MkdirAll(artifactDir, 0755); err != nil {
				t.Fatalf("MkdirAll: %v", err)
			}
			decodeBytes := makePT2DecodeGraph(t, tc.nodes, tc.outputTensor)
			for name, data := range map[string][]byte{
				"manifest.json": []byte(tc.manifest),
				"binding.json":  []byte(tc.binding),
				"decode.pt2":    decodeBytes,
			} {
				if err := os.WriteFile(filepath.Join(artifactDir, name), data, 0644); err != nil {
					t.Fatalf("write %s: %v", name, err)
				}
			}

			cliPlanPath := filepath.Join(tmpDir, "cli_plan_"+tc.name+".pb")
			if out, err := exec.CommandContext(ctx, vxpuBin, "plan", "emit",
				"--artifact", artifactDir,
				"--machine", tc.machine,
				"--tree-commit", "0123456789abcdef0123456789abcdef01234567",
				"--out", cliPlanPath,
			).CombinedOutput(); err != nil {
				t.Fatalf("vxpu plan emit: %v\n%s", err, string(out))
			}
			cliPlan, err := kernelplan.ReadPlan(cliPlanPath)
			if err != nil {
				t.Fatalf("ReadPlan: %v", err)
			}

			executor := &loadingExecutor{acceptAnyGraph: true, probe: probeJSONForMachine(t, tc.machine)}
			s := startCatalogPodExecutor(t, tc.manifest, executor)
			if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{
				ManifestJson: tc.manifest,
				BindingJson:  tc.binding,
				DecodeGraph:  decodeBytes,
			}); err != nil {
				t.Fatalf("router LoadModel: %v", err)
			}

			routerPlan := executor.lastPlanFile
			if routerPlan == nil || routerPlan.PlanHash == "" {
				t.Fatalf("router did not forward an emitted plan with PlanHash: %v", routerPlan)
			}
			if routerPlan.PlanHash != cliPlan.PlanHash {
				t.Fatalf("plan identity mismatch: router=%q, cli=%q", routerPlan.PlanHash, cliPlan.PlanHash)
			}
			if st := s.ModelStatus(artifactKey(tc.manifest)); st.PlanHash != cliPlan.PlanHash {
				t.Errorf("status PlanHash = %q, want %q", st.PlanHash, cliPlan.PlanHash)
			}
			if desired, ok := s.reconciler.Desired(artifactKey(tc.manifest)); !ok || desired.Request.PlanFile != nil {
				t.Errorf("desired state must not store emitted PlanFile, got ok=%v plan=%v", ok, desired.Request.PlanFile)
			}
			hashes[tc.name] = routerPlan.PlanHash

			for _, seg := range routerPlan.Segments {
				if !seg.Host && seg.Run != tc.wantSegmentRun {
					t.Errorf("segment %q run = %v, want %v", seg.Name, seg.Run, tc.wantSegmentRun)
				}
			}
			if tc.wantOpBody != "" {
				for _, site := range routerPlan.Sites {
					if strings.HasPrefix(site.Id, "fp8_linear_") && site.Body != tc.wantOpBody {
						t.Errorf("site %q body = %q, want %q", site.Id, site.Body, tc.wantOpBody)
					}
				}
			}

			cliNorm := proto.Clone(cliPlan).(*planpb.PlanFile)
			routerNorm := proto.Clone(routerPlan).(*planpb.PlanFile)
			cliNorm.TreeCommit = ""
			routerNorm.TreeCommit = ""
			if !proto.Equal(routerNorm, cliNorm) {
				t.Errorf("plans differ beyond TreeCommit:\nrouter: %v\ncli:    %v", routerNorm, cliNorm)
			}
		})
	}

	if hashes["custom_ops_gpu_g2_standard_48"] == hashes["custom_ops_cpu_c3d_standard_60"] {
		t.Errorf("custom-ops GPU and CPU plans should produce distinct identities, both got %q", hashes["custom_ops_gpu_g2_standard_48"])
	}

	// A probe whose GPU field is only whitespace with a valid CPU must plan as CPU, not CUDA.
	t.Run("whitespace_gpu_probe_plans_as_cpu", func(t *testing.T) {
		decodeBytes := makePT2DecodeGraph(t, customOpsNodes, "fp8_linear_1")
		executor := &loadingExecutor{acceptAnyGraph: true, probe: `{"gpu":"   ","cpu":"AMD EPYC","simd":"avx512"}`}
		s := startCatalogPodExecutor(t, customManifest, executor)
		if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{
			ManifestJson: customManifest,
			BindingJson:  customBinding,
			DecodeGraph:  decodeBytes,
		}); err != nil {
			t.Fatalf("router LoadModel: %v", err)
		}
		routerPlan := executor.lastPlanFile
		if routerPlan == nil || routerPlan.PlanHash == "" {
			t.Fatalf("router did not emit a plan: %v", routerPlan)
		}
		if routerPlan.PlanHash != hashes["custom_ops_cpu_c3d_standard_60"] {
			t.Errorf("whitespace GPU probe PlanHash = %q, want CPU plan hash %q", routerPlan.PlanHash, hashes["custom_ops_cpu_c3d_standard_60"])
		}
		for _, seg := range routerPlan.Segments {
			if !seg.Host && seg.Run != planpb.SegmentRun_SEGMENT_RUN_PACKAGE {
				t.Errorf("segment %q run = %v, want %v", seg.Name, seg.Run, planpb.SegmentRun_SEGMENT_RUN_PACKAGE)
			}
		}
		for _, site := range routerPlan.Sites {
			if strings.HasPrefix(site.Id, "fp8_linear_") && site.Body != "reference" {
				t.Errorf("site %q body = %q, want %q", site.Id, site.Body, "reference")
			}
		}
	})
}

func TestLoadModel_ServingModelDifferentPlanConverges(t *testing.T) {
	ctx := t.Context()
	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})

	executor := &loadingExecutor{}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	modelKey := artifactKey(testManifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + modelKey, Namespace: "ns"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	if _, err := clientset.CoreV1().Pods("ns").Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Create pod: %v", err)
	}

	plan1 := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  testCommit,
		Sites: []*planpb.PlanSite{
			{
				Id:       "model.layers.0.attn.wq.weight",
				Kind:     planpb.SiteKind_SITE_KIND_DENSE_GEMV,
				Strategy: "gpu-fp8-gemv",
			},
		},
	}
	req1 := &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
		PrefillGraph: []byte("graph"),
		DecodeGraph:  []byte("graph"),
		PlanFile:     plan1,
	}
	if _, err := s.LoadModel(ctx, req1); err != nil {
		t.Fatalf("First LoadModel: %v", err)
	}
	if executor.loads != 1 {
		t.Fatalf("expected 1 load, got %d", executor.loads)
	}

	// Loading again with the same plan succeeds and skips reload.
	if _, err := s.LoadModel(ctx, req1); err != nil {
		t.Fatalf("Second LoadModel with same plan: %v", err)
	}
	if executor.loads != 1 {
		t.Fatalf("expected reload to be skipped, got %d loads", executor.loads)
	}

	// Loading with a different plan for the serving model updates desired state and converges.
	plan2 := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  testCommit,
		Sites: []*planpb.PlanSite{
			{
				Id:       "model.layers.0.attn.wq.weight",
				Kind:     planpb.SiteKind_SITE_KIND_DENSE_GEMV,
				Strategy: "reference",
			},
		},
	}
	req2 := &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
		PrefillGraph: []byte("graph"),
		DecodeGraph:  []byte("graph"),
		PlanFile:     plan2,
	}
	if _, err := s.LoadModel(ctx, req2); err != nil {
		t.Fatalf("LoadModel with different plan failed: %v", err)
	}
	if executor.loads != 2 {
		t.Errorf("expected 2 loads after new plan applied, got %d", executor.loads)
	}
	if !proto.Equal(executor.lastPlanFile, plan2) {
		t.Errorf("executor did not receive new plan: got %v, want %v", executor.lastPlanFile, plan2)
	}
}

func TestRegisterModel_PlanFileRefusedForNow(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{})

	req := &pb.RegisterModelRequest{
		Model: "org/M",
		PlanFile: &planpb.PlanFile{
			ArtifactMd5: "0123456789abcdef0123456789abcdef",
		},
	}
	_, err := s.RegisterModel(ctx, req)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("RegisterModel with PlanFile: got %v, want INVALID_ARGUMENT", err)
	}
	if !strings.Contains(err.Error(), "registering a model with a plan file is not supported yet") {
		t.Errorf("error message %q does not explain plan file registration is not supported yet", err.Error())
	}
}

func TestRegisterModel_RefusedPlanFailsOnce(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})
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
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})

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
	s.UseExternalExecutors(ctx, []string{startExecutor(t, engine), helperAddr})

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
	s.UseExternalExecutors(ctx, []string{lis.Addr().String()})

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
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})
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
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})
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

// If the model finishes loading between NewSession and GetLoad, the router
// retries NewSession and succeeds without waiting for another polling cycle.
func TestNewSessionByName_ReadyBetweenNewSessionAndGetLoad(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{readyBetween: true}
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession should retry and succeed when GetLoad reports READY: %v", err)
	}
}

func TestNewSessionByName_ExternalModelstoreURL(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})
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

func TestLoadByName_DeduplicatesConcurrentLoads(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	gate := make(chan struct{})
	executor := &loadingExecutor{loadGate: gate}
	s.UseExternalExecutors(ctx, []string{startExecutor(t, executor)})

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	var wg sync.WaitGroup
	var key1, key2 string
	var err1, err2 error

	wg.Add(2)
	go func() {
		defer wg.Done()
		key1, err1 = s.loadByName(ctx, "org/M", "")
	}()
	go func() {
		defer wg.Done()
		key2, err2 = s.loadByName(ctx, "org/M", "")
	}()

	time.Sleep(20 * time.Millisecond)
	close(gate)
	wg.Wait()

	if err1 != nil || err2 != nil {
		t.Fatalf("loadByName failed: err1=%v, err2=%v", err1, err2)
	}
	if key1 == "" || key1 != key2 {
		t.Errorf("keys do not match: key1=%q, key2=%q", key1, key2)
	}
	executor.mu.Lock()
	loads := executor.loads
	executor.mu.Unlock()
	if loads != 1 {
		t.Errorf("expected 1 executor load, got %d", loads)
	}
}

func TestLoadByName_WaiterContextCancelled(t *testing.T) {
	ctxLeader := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	gate := make(chan struct{})
	started := make(chan struct{})
	executor := &loadingExecutor{loadGate: gate, loadStarted: started}
	s.UseExternalExecutors(ctxLeader, []string{startExecutor(t, executor)})

	if _, err := s.RegisterModel(ctxLeader, &pb.RegisterModelRequest{Model: "org/M"}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	ctxWaiter, cancelWaiter := context.WithCancel(t.Context())
	var wg sync.WaitGroup
	var leaderKey string
	var leaderErr, waiterErr error

	wg.Add(2)
	go func() {
		defer wg.Done()
		leaderKey, leaderErr = s.loadByName(ctxLeader, "org/M", "")
	}()

	// Wait until leader has entered LoadModel and is blocked on gate
	<-started

	go func() {
		defer wg.Done()
		_, waiterErr = s.loadByName(ctxWaiter, "org/M", "")
	}()

	time.Sleep(20 * time.Millisecond)
	cancelWaiter()

	time.Sleep(20 * time.Millisecond)
	close(gate)
	wg.Wait()

	if leaderErr != nil {
		t.Fatalf("leader failed: %v", leaderErr)
	}
	if leaderKey == "" {
		t.Errorf("expected non-empty leader key")
	}
	if status.Code(waiterErr) != codes.Canceled {
		t.Errorf("waiter got code %v (%v), want Canceled", status.Code(waiterErr), waiterErr)
	}
}

func TestLoadByName_UncompiledModelSharesVerdict(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: false})

	var wg sync.WaitGroup
	errs := make([]error, 4)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = s.loadByName(ctx, "org/M", "")
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if status.Code(err) != codes.FailedPrecondition {
			t.Errorf("caller %d got code %v (%v), want FailedPrecondition", i, status.Code(err), err)
		}
	}
}

func TestLoadModel_GoalStoredOnInstanceDesired(t *testing.T) {
	ctx := t.Context()
	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{acceptAnyGraph: true}
	addr := startExecutor(t, executor)
	_, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("SplitHostPort: %v", err)
	}
	var port int
	if _, err := fmt.Sscanf(portStr, "%d", &port); err != nil {
		t.Fatalf("parse port: %v", err)
	}
	s.executorPort = port

	modelKey := artifactKey(testManifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + modelKey, Namespace: "ns"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	if _, err := clientset.CoreV1().Pods("ns").Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Create pod: %v", err)
	}

	// Default goal (unset) records PLAN_GOAL_PASS_THROUGH on InstanceDesired.Request.
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
	}); err != nil {
		t.Fatalf("LoadModel default goal: %v", err)
	}
	d1, ok := s.reconciler.Desired(modelKey)
	if !ok {
		t.Fatal("expected desired state for loaded instance")
	}
	if d1.Request.GetGoal() != planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH {
		t.Errorf("d1.Request.Goal = %v, want PLAN_GOAL_PASS_THROUGH", d1.Request.GetGoal())
	}

	// Switching to PLAN_GOAL_SPEED updates InstanceDesired.Request.Goal and advances generation.
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: testManifest,
		BindingJson:  "{}",
		Goal:         planpb.PlanGoal_PLAN_GOAL_SPEED,
	}); err != nil {
		t.Fatalf("LoadModel speed goal: %v", err)
	}
	d2, ok := s.reconciler.Desired(modelKey)
	if !ok {
		t.Fatal("expected desired state after goal change")
	}
	if d2.Request.GetGoal() != planpb.PlanGoal_PLAN_GOAL_SPEED {
		t.Errorf("d2.Request.Goal = %v, want PLAN_GOAL_SPEED", d2.Request.GetGoal())
	}
	if d2.Generation <= d1.Generation {
		t.Errorf("generation did not advance on goal change: %d <= %d", d2.Generation, d1.Generation)
	}

	// Invalid goal is refused on LoadModel and RegisterModel.
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: testManifest,
		Goal:         planpb.PlanGoal(99),
	}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("LoadModel invalid goal got code %v (%v), want InvalidArgument", status.Code(err), err)
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model: "org/M",
		Goal:  planpb.PlanGoal(99),
	}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("RegisterModel invalid goal got code %v (%v), want InvalidArgument", status.Code(err), err)
	}

	// Non-default goal on RegisterModel is refused as not supported yet.
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model: "org/M",
		Goal:  planpb.PlanGoal_PLAN_GOAL_SPEED,
	}); status.Code(err) != codes.InvalidArgument || !strings.Contains(err.Error(), "not supported yet") {
		t.Errorf("RegisterModel speed goal got %v, want InvalidArgument with \"not supported yet\"", err)
	}
}
