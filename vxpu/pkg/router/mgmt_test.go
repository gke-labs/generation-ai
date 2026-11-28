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
	stderrors "errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/encoding/protojson"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

func newTestMgmtServer(t *testing.T) (*Server, *fake.Clientset) {
	t.Helper()
	cs := fake.NewSimpleClientset()
	s := &Server{
		clientset:        cs,
		namespace:        "default",
		imageName:        "vxpu-executor:test",
		peerToModel:      make(map[string]string),
		conns:            make(map[string]*grpc.ClientConn),
		externalModel:    make(map[string]string),
		modelEndpoint:    make(map[string]string),
		loadedTags:       make(map[string]string),
		keyModel:         make(map[string]modelRef),
		registeredModels: make(map[string]modelRef),
		executors:        make(map[string]executorInfo),
		models:           make(map[string]modelState),
		vocabs:           make(map[string]*pb.GetVocabResponse),
	}
	s.reconciler = NewModelReconciler(s)
	return s, cs
}

func testMgmtPlan(t *testing.T) (*planpb.PlanFile, []byte, string, string) {
	t.Helper()
	nodesJSON := `[
		{"target": "torch.ops.vxpu.fp8_linear.default", "name": "fp8_linear_0",
		 "inputs": [{"name": "x", "arg": {"as_tensor": {"name": "input_ids"}}}],
		 "outputs": [{"as_tensor": {"name": "out"}}],
		 "metadata": {"nn_module_stack": "L['self'],model.layers.0.mlp"}}
	]`
	decodeGraph := makePT2DecodeGraph(t, nodesJSON, "out")
	manifestJSON := `{"source":{"repo":"org/model"},"tensors":{}}`
	bindingJSON := `{"bound":{}}`
	plan := &planpb.PlanFile{
		ArtifactMd5: "deadbeef0123456789abcdef01234567",
		TreeCommit:  "abc1234",
		Goal:        planpb.PlanGoal_PLAN_GOAL_SPEED,
		Segments: []*planpb.Segment{
			{
				Name:       "submod_0",
				Run:        planpb.SegmentRun_SEGMENT_RUN_PACKAGE,
				PackageKey: "pkgkey0123456789",
			},
		},
		Sites: []*planpb.PlanSite{
			{
				Id:   "fp8_linear_0",
				Body: "triton_fp8",
			},
		},
	}
	hash, err := kernelplan.PlanHash(plan)
	if err != nil {
		t.Fatalf("PlanHash: %v", err)
	}
	plan.PlanHash = hash
	return plan, decodeGraph, manifestJSON, bindingJSON
}

func TestManagementHandler(t *testing.T) {
	plan, decodeGraph, manifestJSON, bindingJSON := testMgmtPlan(t)
	nextRetry := time.Date(2026, 11, 28, 12, 34, 56, 0, time.UTC)

	l4Machine, err := machines.Lookup("g2-standard-24")
	if err != nil {
		t.Fatalf("machines.Lookup: %v", err)
	}

	populateFleet := func(t *testing.T, s *Server) {
		t.Helper()
		s.registeredModels["org/registered-only"] = modelRef{
			Model:    "org/registered-only",
			Commit:   "1111222233334444",
			Artifact: "vxpu-org-registered-only:11112222-test",
		}
		s.registeredModels["org/model"] = modelRef{
			Model:    "org/model",
			Commit:   "aaaa1111bbbb2222",
			Artifact: "vxpu-org-model:aaaa1111-test",
		}

		// Two distinct physical instances of the same model on different machines and phases:
		// they must never be aggregated into one model-level status.
		s.keyModel["inst-gpu-key-111111"] = modelRef{
			Model:    "org/model",
			Commit:   "aaaa1111bbbb2222",
			Artifact: "vxpu-org-model:aaaa1111-test",
		}
		s.keyModel["inst-cpu-key-222222"] = modelRef{
			Model:    "org/model",
			Commit:   "aaaa1111bbbb2222",
			Artifact: "vxpu-org-model:aaaa1111-test",
		}
		s.executors["10.0.0.10:35001"] = executorInfo{
			Endpoint: "10.0.0.10:35001",
			Probe:    machines.ProbeOf(l4Machine),
			Machine:  l4Machine,
		}
		s.executors["10.0.0.11:35002"] = executorInfo{
			Endpoint: "10.0.0.11:35002",
			Probe:    &pb.Probe{Cpu: "Intel Xeon AVX512", CpuThreads: 16},
		}

		s.reconciler.instances.Store("inst-gpu-key-111111", &instanceState{
			desired: &InstanceDesired{
				Key:        "inst-gpu-key-111111",
				Model:      "org/model",
				Generation: 2,
				Request: &pb.LoadModelRequest{
					ManifestJson: manifestJSON,
					BindingJson:  bindingJSON,
					DecodeGraph:  decodeGraph,
					Plan:         map[string]string{"VXPU_SECRET_SWITCH": "must-not-leak"},
				},
			},
			status: InstanceStatus{
				Phase:              InstancePhasePassed,
				ObservedGeneration: 2,
				PlanHash:           plan.PlanHash,
			},
			endpoint: "10.0.0.10:35001",
			plan:     plan,
		})
		s.reconciler.instances.Store("inst-cpu-key-222222", &instanceState{
			desired: &InstanceDesired{
				Key:        "inst-cpu-key-222222",
				Model:      "org/model",
				Generation: 3,
				Request:    &pb.LoadModelRequest{ManifestJson: manifestJSON},
			},
			status: InstanceStatus{
				Phase:              InstancePhaseFailed,
				ObservedGeneration: 3,
				LastError:          stderrors.New("runtime exited with status 137"),
				NextAttempt:        nextRetry,
			},
			endpoint: "10.0.0.11:35002",
		})
		s.reconciler.instances.Store("inst-refused-333333", &instanceState{
			desired: &InstanceDesired{
				Key:        "inst-refused-333333",
				Model:      "org/bad-model",
				Generation: 1,
				Request: &pb.LoadModelRequest{
					PlanFile: &planpb.PlanFile{
						Goal: planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH,
						Segments: []*planpb.Segment{
							{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH},
						},
					},
				},
			},
			status: InstanceStatus{
				Phase:              InstancePhaseRefused,
				ObservedGeneration: 1,
				LastError:          stderrors.New("golden token mismatch at step 0"),
			},
			endpoint: "10.0.0.12:35003",
		})
	}

	tests := []struct {
		name            string
		populate        bool
		method          string
		path            string
		accept          string
		wantStatus      int
		wantContentType string
		wantContains    []string
		wantNotContains []string
		checkJSON       func(t *testing.T, body []byte)
	}{
		{
			name:            "empty router landing HTML",
			method:          http.MethodGet,
			path:            "/",
			wantStatus:      http.StatusOK,
			wantContentType: "text/html",
			wantContains: []string{
				"vXPU Router",
				"No registered or active models.",
				"No physical instances tracked by the reconciler.",
			},
		},
		{
			name:            "empty router landing JSON",
			method:          http.MethodGet,
			path:            "/",
			accept:          "application/json",
			wantStatus:      http.StatusOK,
			wantContentType: "application/json",
			checkJSON: func(t *testing.T, body []byte) {
				var st pb.RouterStatus
				if err := protojson.Unmarshal(body, &st); err != nil {
					t.Fatalf("protojson.Unmarshal RouterStatus: %v\n%s", err, body)
				}
				if len(st.Models) != 0 || len(st.Instances) != 0 {
					t.Errorf("got %d models and %d instances, want 0", len(st.Models), len(st.Instances))
				}
			},
		},
		{
			name:            "populated fleet landing HTML shows models and separate physical instances",
			populate:        true,
			method:          http.MethodGet,
			path:            "/",
			wantStatus:      http.StatusOK,
			wantContentType: "text/html",
			wantContains: []string{
				"org/registered-only",
				"org/model",
				"org/bad-model",
				"10.0.0.10:35001",
				"gpu:nvidia l4",
				"10.0.0.11:35002",
				"cpu:intel xeon avx512:",
				"runtime exited with status 137",
				"2026-11-28T12:34:56Z",
				"golden token mismatch at step 0",
				"2 / 2",
				"3 / 3",
				"/instances/inst-gpu-key-111111/plan",
				"/instances/inst-refused-333333/plan",
				plan.PlanHash[:12],
			},
			wantNotContains: []string{
				"VXPU_SECRET_SWITCH",
				"must-not-leak",
			},
		},
		{
			name:            "populated fleet landing JSON returns RouterStatus per physical instance",
			populate:        true,
			method:          http.MethodGet,
			path:            "/",
			accept:          "application/json",
			wantStatus:      http.StatusOK,
			wantContentType: "application/json",
			wantNotContains: []string{
				"VXPU_SECRET_SWITCH",
				"must-not-leak",
			},
			checkJSON: func(t *testing.T, body []byte) {
				var st pb.RouterStatus
				if err := protojson.Unmarshal(body, &st); err != nil {
					t.Fatalf("protojson.Unmarshal RouterStatus: %v\n%s", err, body)
				}
				if len(st.Models) != 3 {
					t.Fatalf("len(Models) = %d, want 3: %+v", len(st.Models), st.Models)
				}
				if len(st.Instances) != 3 {
					t.Fatalf("len(Instances) = %d, want 3: %+v", len(st.Instances), st.Instances)
				}
				byKey := make(map[string]*pb.FleetInstance, len(st.Instances))
				for _, inst := range st.Instances {
					byKey[inst.Key] = inst
				}
				gpuInst := byKey["inst-gpu-key-111111"]
				if gpuInst == nil {
					t.Fatal("missing inst-gpu-key-111111")
				}
				if gpuInst.Phase != pb.InstancePhase_INSTANCE_PHASE_PASSED {
					t.Errorf("gpuInst.Phase = %v, want PASSED", gpuInst.Phase)
				}
				if gpuInst.MachineKind != "gpu:nvidia l4" {
					t.Errorf("gpuInst.MachineKind = %q, want gpu:nvidia l4", gpuInst.MachineKind)
				}
				if gpuInst.PlanHash != plan.PlanHash {
					t.Errorf("gpuInst.PlanHash = %q, want %q", gpuInst.PlanHash, plan.PlanHash)
				}
				if !gpuInst.HasPlan {
					t.Errorf("gpuInst.HasPlan = false, want true")
				}
				if gpuInst.Generation != 2 || gpuInst.ObservedGeneration != 2 {
					t.Errorf("gpuInst gen = %d/%d, want 2/2", gpuInst.ObservedGeneration, gpuInst.Generation)
				}

				cpuInst := byKey["inst-cpu-key-222222"]
				if cpuInst == nil {
					t.Fatal("missing inst-cpu-key-222222")
				}
				if cpuInst.Phase != pb.InstancePhase_INSTANCE_PHASE_FAILED {
					t.Errorf("cpuInst.Phase = %v, want FAILED", cpuInst.Phase)
				}
				if cpuInst.MachineKind != "cpu:intel xeon avx512:" {
					t.Errorf("cpuInst.MachineKind = %q, want cpu:intel xeon avx512:", cpuInst.MachineKind)
				}
				if cpuInst.PlanHash != "" {
					t.Errorf("cpuInst.PlanHash = %q, want empty", cpuInst.PlanHash)
				}
				if cpuInst.HasPlan {
					t.Errorf("cpuInst.HasPlan = true, want false")
				}
				if cpuInst.LastError != "runtime exited with status 137" {
					t.Errorf("cpuInst.LastError = %q, want runtime error", cpuInst.LastError)
				}
				if cpuInst.NextAttemptRfc3339 != "2026-11-28T12:34:56Z" {
					t.Errorf("cpuInst.NextAttemptRfc3339 = %q", cpuInst.NextAttemptRfc3339)
				}
			},
		},
		{
			name:            "instance plan EXPLAIN HTML from decode graph and emitted plan",
			populate:        true,
			method:          http.MethodGet,
			path:            "/instances/inst-gpu-key-111111/plan",
			wantStatus:      http.StatusOK,
			wantContentType: "text/html",
			wantContains: []string{
				"org/model",
				"submod_0",
				"fp8_linear_0",
				"triton_fp8",
				"pkgkey0123456789",
			},
		},
		{
			name:            "instance plan EXPLAIN JSON returns ModelTree",
			populate:        true,
			method:          http.MethodGet,
			path:            "/instances/inst-gpu-key-111111/plan",
			accept:          "application/json",
			wantStatus:      http.StatusOK,
			wantContentType: "application/json",
			checkJSON: func(t *testing.T, body []byte) {
				var tree planpb.ModelTree
				if err := protojson.Unmarshal(body, &tree); err != nil {
					t.Fatalf("protojson.Unmarshal ModelTree: %v\n%s", err, body)
				}
				if tree.Model != "org/model" {
					t.Errorf("tree.Model = %q, want org/model", tree.Model)
				}
				if !tree.HasPlan {
					t.Errorf("tree.HasPlan = false, want true")
				}
				if len(tree.Segments) != 1 || tree.Segments[0].Name != "submod_0" {
					t.Fatalf("unexpected segments: %+v", tree.Segments)
				}
				if len(tree.Segments[0].CallSites) != 1 || tree.Segments[0].CallSites[0].Body != "triton_fp8" {
					t.Errorf("unexpected call sites: %+v", tree.Segments[0].CallSites)
				}
			},
		},
		{
			name:            "instance plan EXPLAIN HTML without decode graph falls back to plan segments",
			populate:        true,
			method:          http.MethodGet,
			path:            "/instances/inst-refused-333333/plan",
			wantStatus:      http.StatusOK,
			wantContentType: "text/html",
			wantContains: []string{
				"org/bad-model",
				"submod_0",
				"plain torch",
			},
		},
		{
			name:       "instance without plan returns 404",
			populate:   true,
			method:     http.MethodGet,
			path:       "/instances/inst-cpu-key-222222/plan",
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "unknown instance returns 404",
			populate:   true,
			method:     http.MethodGet,
			path:       "/instances/does-not-exist/plan",
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "unknown path returns 404",
			populate:   true,
			method:     http.MethodGet,
			path:       "/unknown",
			wantStatus: http.StatusNotFound,
		},
		{
			name:       "POST to landing page returns 405",
			populate:   true,
			method:     http.MethodPost,
			path:       "/",
			wantStatus: http.StatusMethodNotAllowed,
		},
		{
			name:       "DELETE to instance plan returns 405",
			populate:   true,
			method:     http.MethodDelete,
			path:       "/instances/inst-gpu-key-111111/plan",
			wantStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, cs := newTestMgmtServer(t)
			if tc.populate {
				populateFleet(t, s)
			}

			// Assert the management UI never makes Kubernetes API calls under locks or at all.
			cs.PrependReactor("*", "*", func(action ktesting.Action) (bool, runtime.Object, error) {
				t.Errorf("unexpected Kubernetes API call during management request: %s %s", action.GetVerb(), action.GetResource().Resource)
				return true, nil, stderrors.New("unexpected k8s call")
			})

			req := httptest.NewRequest(tc.method, tc.path, nil)
			if tc.accept != "" {
				req.Header.Set("Accept", tc.accept)
			}
			rec := httptest.NewRecorder()
			s.ManagementHandler().ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d; body:\n%s", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if tc.wantContentType != "" {
				gotCT := rec.Header().Get("Content-Type")
				if !strings.Contains(gotCT, tc.wantContentType) {
					t.Errorf("Content-Type = %q, want substring %q", gotCT, tc.wantContentType)
				}
			}
			body := rec.Body.String()
			for _, sub := range tc.wantContains {
				if !strings.Contains(body, sub) {
					t.Errorf("response missing %q;\nbody:\n%s", sub, body)
				}
			}
			for _, sub := range tc.wantNotContains {
				if strings.Contains(body, sub) {
					t.Errorf("response must not contain %q;\nbody:\n%s", sub, body)
				}
			}
			if tc.checkJSON != nil {
				tc.checkJSON(t, rec.Body.Bytes())
			}
		})
	}
}

func TestManagementHandler_LiveReconcilerPlan(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.agent.softwareVersion = "abc1234"
	f.server.reconciler.Stop()
	f.server.reconciler = f.reconciler
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	_, decodeGraph, manifestJSON, bindingJSON := testMgmtPlan(t)
	f.desired.Model = "org/live-model"
	f.desired.Request = &pb.LoadModelRequest{
		ManifestJson: manifestJSON,
		BindingJson:  bindingJSON,
		DecodeGraph:  decodeGraph,
		Goal:         planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH,
	}

	gen := f.reconciler.SetDesired(f.desired)
	st, err := f.reconciler.Wait(ctx, f.instanceKey, gen)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if st.Phase != InstancePhasePassed {
		t.Fatalf("phase = %v, want passed (err=%v)", st.Phase, st.LastError)
	}

	// Verify GET / JSON reports the live reconciled instance with emitted plan.
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Accept", "application/json")
	rec := httptest.NewRecorder()
	f.server.ManagementHandler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET / status = %d, want 200; body:\n%s", rec.Code, rec.Body.String())
	}
	var rs pb.RouterStatus
	if err := protojson.Unmarshal(rec.Body.Bytes(), &rs); err != nil {
		t.Fatalf("protojson.Unmarshal RouterStatus: %v", err)
	}
	if len(rs.Instances) != 1 {
		t.Fatalf("len(Instances) = %d, want 1", len(rs.Instances))
	}
	inst := rs.Instances[0]
	if inst.Phase != pb.InstancePhase_INSTANCE_PHASE_PASSED {
		t.Errorf("inst.Phase = %v, want PASSED", inst.Phase)
	}
	if inst.PlanHash == "" || !inst.HasPlan {
		t.Errorf("expected non-empty PlanHash and HasPlan=true, got PlanHash=%q HasPlan=%v", inst.PlanHash, inst.HasPlan)
	}
	if inst.Executor != f.execAddr {
		t.Errorf("inst.Executor = %q, want %q", inst.Executor, f.execAddr)
	}

	// Verify GET /instances/{key}/plan renders EXPLAIN HTML from the emitted plan.
	planReq := httptest.NewRequest(http.MethodGet, "/instances/"+f.instanceKey+"/plan", nil)
	planRec := httptest.NewRecorder()
	f.server.ManagementHandler().ServeHTTP(planRec, planReq)
	if planRec.Code != http.StatusOK {
		t.Fatalf("GET plan status = %d, want 200; body:\n%s", planRec.Code, planRec.Body.String())
	}
	if !strings.Contains(planRec.Body.String(), "org/live-model") || !strings.Contains(planRec.Body.String(), "fp8_linear_0") {
		t.Errorf("EXPLAIN HTML missing expected model/site content")
	}
}
