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
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protojson"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

func TestManagementHandler_FleetStatus(t *testing.T) {
	ctx := t.Context()
	s := NewServer(fake.NewSimpleClientset(), "default", "vxpu-executor:latest", "")

	// Set up environment with dummy secrets to verify they never leak
	t.Setenv("SECRET_TOKEN", "super-secret-auth-token-12345")
	t.Setenv("USER_PROMPT_SECRET", "confidential proprietary prompt text")

	s.mu.Lock()
	// 1. Model that passed golden check and is serving on external executor
	s.externalModel["10.0.0.1:50051"] = "model-passed-key"
	s.modelEndpoint["model-passed-key"] = "10.0.0.1:50051"
	s.models["model-passed-key"] = modelState{standing: standingPassed}
	s.keyModel["model-passed-key"] = modelRef{
		Model:    "org/passed-model",
		Commit:   "abc123456789",
		Artifact: "org/passed-model:tag1",
	}
	s.executors["10.0.0.1:50051"] = executorInfo{
		Endpoint: "10.0.0.1:50051",
		Probe:    &pb.Probe{Gpu: "NVIDIA-L4", RamBytes: 64 * 1024 * 1024 * 1024},
		Machine:  &machines.Machine{Name: "g2-standard-48"},
	}

	// Expert helper executor holding experts for 10.0.0.1:50051
	s.externalExecutors = append(s.externalExecutors, "10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051")
	s.expertHolders["10.0.0.3:50051"] = "10.0.0.1:50051"
	s.executors["10.0.0.3:50051"] = executorInfo{
		Endpoint: "10.0.0.3:50051",
		Probe:    &pb.Probe{Gpu: "NVIDIA-L4"},
		Machine:  &machines.Machine{Name: "g2-standard-48"},
	}

	// 2. Model with refused golden check
	refusedErr := errors.New("golden values: prompt 0 differs first at position 0: got 99999, expected 10")
	s.models["model-refused-key"] = modelState{standing: standingRefused, err: refusedErr}
	s.keyModel["model-refused-key"] = modelRef{
		Model:    "org/refused-model",
		Commit:   "def123456789",
		Artifact: "org/refused-model:tag2",
	}

	// 3. Model never loaded (unchecked)
	s.models["model-never-loaded-key"] = modelState{standing: standingUnchecked}
	s.keyModel["model-never-loaded-key"] = modelRef{
		Model:    "org/unloaded-model",
		Commit:   "ghi123456789",
		Artifact: "org/unloaded-model:tag3",
	}

	// 4. In-cluster pod executor holding a model
	s.podEndpoint["model-pod-key"] = "10.1.2.3:41234"
	s.podModel["10.1.2.3:41234"] = "model-pod-key"
	s.models["model-pod-key"] = modelState{standing: standingPassed}
	s.keyModel["model-pod-key"] = modelRef{
		Model:    "org/pod-model",
		Commit:   "jkl123456789",
		Artifact: "org/pod-model:tag4",
	}
	s.executors["10.1.2.3:41234"] = executorInfo{
		Endpoint: "10.1.2.3:41234",
	}
	s.mu.Unlock()

	handler := s.ManagementHandler()

	t.Run("HTML", func(t *testing.T) {
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
		contentType := rec.Header().Get("Content-Type")
		if !strings.Contains(contentType, "text/html") {
			t.Errorf("expected text/html, got %q", contentType)
		}

		body := rec.Body.String()

		// Verify model that passed
		if !strings.Contains(body, "org/passed-model") || !strings.Contains(body, "model-passed-key") || !strings.Contains(body, "10.0.0.1:50051") {
			t.Errorf("HTML should show org/passed-model with key and endpoint 10.0.0.1:50051:\n%s", body)
		}

		// Verify model with refused check
		if !strings.Contains(body, "org/refused-model") || !strings.Contains(body, "refused") || !strings.Contains(body, "got 99999, expected 10") {
			t.Errorf("HTML should show org/refused-model as refused with error:\n%s", body)
		}

		// Verify model never loaded
		if !strings.Contains(body, "org/unloaded-model") || !strings.Contains(body, "unchecked") {
			t.Errorf("HTML should show org/unloaded-model as unchecked:\n%s", body)
		}

		// Verify executors
		if !strings.Contains(body, "g2-standard-48") || !strings.Contains(body, "10.0.0.1:50051") {
			t.Errorf("HTML should show executor 10.0.0.1:50051 and machine kind g2-standard-48:\n%s", body)
		}
		// Verify helper executor lists the held model
		if !strings.Contains(body, "10.0.0.3:50051") {
			t.Errorf("HTML should show helper executor 10.0.0.3:50051:\n%s", body)
		}

		// Verify no secrets appear
		assertNoSecrets(t, body)
	})

	t.Run("JSON", func(t *testing.T) {
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/", nil)
		req.Header.Set("Accept", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
		contentType := rec.Header().Get("Content-Type")
		if !strings.Contains(contentType, "application/json") {
			t.Errorf("expected application/json, got %q", contentType)
		}

		body := rec.Body.String()
		var status pb.RouterStatus
		if err := protojson.Unmarshal([]byte(body), &status); err != nil {
			t.Fatalf("unmarshaling protojson: %v\nJSON:\n%s", err, body)
		}

		if len(status.Models) != 4 {
			t.Fatalf("expected 4 models, got %d", len(status.Models))
		}

		modelsByKey := make(map[string]*pb.ModelStatus)
		for _, m := range status.Models {
			modelsByKey[m.Key] = m
		}

		// Check passed model
		passed := modelsByKey["model-passed-key"]
		if passed == nil {
			t.Fatalf("missing model-passed-key in status")
		}
		if passed.Model != "org/passed-model" || passed.Standing != pb.ModelStanding_MODEL_STANDING_PASSED || passed.Endpoint != "10.0.0.1:50051" {
			t.Errorf("passed model mismatch: %+v", passed)
		}

		// Check refused model
		refused := modelsByKey["model-refused-key"]
		if refused == nil {
			t.Fatalf("missing model-refused-key in status")
		}
		if refused.Model != "org/refused-model" || refused.Standing != pb.ModelStanding_MODEL_STANDING_REFUSED || !strings.Contains(refused.Error, "got 99999, expected 10") {
			t.Errorf("refused model mismatch: %+v", refused)
		}

		// Check never loaded model
		unloaded := modelsByKey["model-never-loaded-key"]
		if unloaded == nil {
			t.Fatalf("missing model-never-loaded-key in status")
		}
		if unloaded.Model != "org/unloaded-model" || unloaded.Standing != pb.ModelStanding_MODEL_STANDING_UNCHECKED || unloaded.Endpoint != "" {
			t.Errorf("unloaded model mismatch: %+v", unloaded)
		}

		// Check in-cluster pod model
		podModel := modelsByKey["model-pod-key"]
		if podModel == nil {
			t.Fatalf("missing model-pod-key in status")
		}
		if podModel.Model != "org/pod-model" || podModel.Standing != pb.ModelStanding_MODEL_STANDING_PASSED || podModel.Endpoint != "10.1.2.3:41234" {
			t.Errorf("pod model mismatch: %+v", podModel)
		}

		// Check executors
		execByAddr := make(map[string]*pb.ExecutorStatus)
		for _, e := range status.Executors {
			execByAddr[e.Endpoint] = e
		}

		// Executor 1
		e1 := execByAddr["10.0.0.1:50051"]
		if e1 == nil || e1.MachineKind != "g2-standard-48" || len(e1.LoadedModels) != 1 || e1.LoadedModels[0] != "org/passed-model" {
			t.Errorf("executor 1 mismatch: %+v", e1)
		}

		// Helper executor 3: holds experts for 10.0.0.1:50051
		e3 := execByAddr["10.0.0.3:50051"]
		if e3 == nil || len(e3.LoadedModels) != 1 || e3.LoadedModels[0] != "org/passed-model" {
			t.Errorf("helper executor 3 mismatch: %+v", e3)
		}

		// Unprobed executor: MachineKind must be empty, not guessed
		e2 := execByAddr["10.0.0.2:50051"]
		if e2 == nil || e2.MachineKind != "" || !e2.External {
			t.Errorf("executor 2 mismatch: %+v", e2)
		}

		// In-cluster executor: not duplicated, not external
		ePod := execByAddr["10.1.2.3:41234"]
		if ePod == nil || ePod.External || len(ePod.LoadedModels) != 1 || ePod.LoadedModels[0] != "org/pod-model" {
			t.Errorf("pod executor mismatch: %+v", ePod)
		}

		// Verify no secrets appear
		assertNoSecrets(t, body)
	})

	t.Run("OneWayToAskForJSON", func(t *testing.T) {
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/?format=json", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
		if !strings.Contains(rec.Header().Get("Content-Type"), "text/html") {
			t.Errorf("expected text/html for ?format=json without Accept header, got %q", rec.Header().Get("Content-Type"))
		}
	})

	t.Run("ManagementPortNeverServesInference", func(t *testing.T) {
		for _, path := range []string{"/v1/chat/completions", "/v1/completions", "/generate", "/api/chat", "/status"} {
			req := httptest.NewRequestWithContext(ctx, http.MethodGet, path, nil)
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound {
				t.Errorf("path %s on management handler returned %d, want 404", path, rec.Code)
			}
		}

		postReq := httptest.NewRequestWithContext(ctx, http.MethodPost, "/", nil)
		postRec := httptest.NewRecorder()
		handler.ServeHTTP(postRec, postReq)
		if postRec.Code != http.StatusMethodNotAllowed {
			t.Errorf("POST / returned %d, want 405", postRec.Code)
		}
	})
}

func assertNoSecrets(t *testing.T, content string) {
	t.Helper()
	if strings.Contains(content, "super-secret-auth-token-12345") {
		t.Error("content leaked SECRET_TOKEN")
	}
	if strings.Contains(content, "confidential proprietary prompt text") {
		t.Error("content leaked USER_PROMPT_SECRET")
	}
	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) == 2 && len(parts[1]) > 5 {
			if strings.Contains(content, parts[1]) && !strings.Contains("0 1 true false default", parts[1]) {
				t.Errorf("content leaked environment value of %s: %q", parts[0], parts[1])
			}
		}
	}
}
