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

package weightstore

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

// fakeRegistry serves the /v2/ OCI routes for testing package and failure storage.
func fakeRegistry(t *testing.T) *httptest.Server {
	t.Helper()
	blobs := map[string][]byte{}
	manifests := map[string][]byte{}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/v2/")
		switch {
		case r.Method == http.MethodPost && strings.Contains(path, "/blobs/uploads/"):
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			blobs[digest] = data
			w.WriteHeader(http.StatusCreated)
		case r.Method == http.MethodGet && strings.Contains(path, "/blobs/"):
			digest := path[strings.LastIndex(path, "/")+1:]
			if data, ok := blobs[digest]; ok {
				_, _ = w.Write(data)
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
		case strings.HasSuffix(path, "/tags/list"):
			name := strings.TrimSuffix(path, "/tags/list")
			var tags []string
			for k := range manifests {
				if n, tag, _ := strings.Cut(k, ":"); n == name {
					tags = append(tags, tag)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"name": name, "tags": tags})
		default:
			http.NotFound(w, r)
		}
	}))
}

// fakeModelstore answers the calls the router makes.
func fakeModelstore(t *testing.T) *httptest.Server {
	t.Helper()
	present := map[string]bool{}
	mux := http.NewServeMux()
	mux.HandleFunc("/blobs/", func(w http.ResponseWriter, r *http.Request) {
		rest := strings.TrimPrefix(r.URL.Path, "/blobs/")
		sha, action, _ := strings.Cut(rest, "/")
		switch {
		case action == "import" && r.Method == http.MethodPost:
			var body map[string]string
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body["url"] == "" {
				http.Error(w, "bad body", http.StatusBadRequest)
				return
			}
			if sha == strings.Repeat("f", 64) {
				http.Error(w, "sha256 mismatch", http.StatusInternalServerError)
				return
			}
			present[sha] = true
			w.WriteHeader(http.StatusCreated)
		case action == "" && (r.Method == http.MethodHead || r.Method == http.MethodGet):
			if !present[sha] {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Length", "10")
			if r.Method == http.MethodGet {
				_, _ = w.Write([]byte("0123456789"))
			}
		default:
			http.NotFound(w, r)
		}
	})
	mux.HandleFunc("/models/import", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"metadata": {"name": "org-name"}, "status": {"phase": "Importing"}}`))
	})
	mux.HandleFunc("/models/org-name", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"status": {"phase": "Ready", "filesTotal": 2, "filesImported": 2, "bytesImported": 123}}`))
	})
	return httptest.NewServer(mux)
}

func TestEnsure(t *testing.T) {
	sha := strings.Repeat("a", 64)
	srv := fakeModelstore(t)
	defer srv.Close()
	c := New(srv.URL + "/")
	got, err := c.Ensure(t.Context(), sha, "https://hub/x.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if want := srv.URL + "/blobs/" + sha; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if _, err := c.Ensure(t.Context(), strings.Repeat("f", 64), "https://hub/y"); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Errorf("mismatch should surface modelstore's message, got %v", err)
	}
}

func TestOpenBlob(t *testing.T) {
	sha := strings.Repeat("a", 64)
	srv := fakeModelstore(t)
	defer srv.Close()
	c := New(srv.URL)

	if _, _, err := c.OpenBlob(t.Context(), sha); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}

	if _, err := c.Ensure(t.Context(), sha, "https://hub/x"); err != nil {
		t.Fatal(err)
	}

	rc, size, err := c.OpenBlob(t.Context(), sha)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()

	if size != 10 {
		t.Errorf("got size %d, want 10", size)
	}
	data, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "0123456789" {
		t.Errorf("got %q, want '0123456789'", string(data))
	}
}

func TestImportRepo(t *testing.T) {
	srv := fakeModelstore(t)
	defer srv.Close()
	c := New(srv.URL)
	name, err := c.ImportRepo(t.Context(), "org/name", "main")
	if err != nil || name != "org-name" {
		t.Fatalf("ImportRepo: %v %q", err, name)
	}
	st, err := c.GetModelStatus(t.Context(), name)
	if err != nil || st.Phase != "Ready" || st.BytesImported != 123 {
		t.Fatalf("GetModelStatus: %v %+v", err, st)
	}
}

func TestDiscoverExplicit(t *testing.T) {
	if c := Discover(t.Context(), "http://example.invalid:1"); c == nil || c.URL() != "http://example.invalid:1" {
		t.Fatalf("explicit URL should be used as given: %v", c)
	}
}

func TestCompileFailureStore(t *testing.T) {
	ctx := t.Context()
	srv := fakeRegistry(t)
	defer srv.Close()
	c := New(srv.URL)
	model := "test-model"
	key := "0123456789abcdef0123456789abcdef"

	// 1. Not found initially
	got, err := c.GetCompileFailure(ctx, model, key)
	if err != nil {
		t.Fatalf("GetCompileFailure: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil failure, got %+v", got)
	}

	// 2. Record failure
	fail := &planpb.CompileFailure{
		Package: &planpb.Package{
			Name: "submod_6",
			Key:  key,
		},
		MachineKind:     "cpu:intel xeon:avx-512",
		SoftwareVersion: "v1.0.0",
		Reason:          "type error in codegen",
	}
	if err := c.RecordCompileFailure(ctx, model, fail); err != nil {
		t.Fatalf("RecordCompileFailure: %v", err)
	}

	// 3. Get recorded failure
	got, err = c.GetCompileFailure(ctx, model, key)
	if err != nil {
		t.Fatalf("GetCompileFailure after record: %v", err)
	}
	if got == nil {
		t.Fatal("expected failure, got nil")
	}
	if got.Package.Name != "submod_6" || got.Package.Key != key {
		t.Errorf("Package mismatch: %+v", got.Package)
	}
	if got.MachineKind != fail.MachineKind || got.SoftwareVersion != fail.SoftwareVersion || got.Reason != fail.Reason {
		t.Errorf("got %+v, want %+v", got, fail)
	}

	// 4. Clear by key
	if err := c.ClearCompileFailure(ctx, model, key); err != nil {
		t.Fatalf("ClearCompileFailure: %v", err)
	}
	got, err = c.GetCompileFailure(ctx, model, key)
	if err != nil {
		t.Fatalf("GetCompileFailure after clear: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil after clear, got %+v", got)
	}

	// 5. Clearing an already-cleared failure is a no-op (does not rewrite)
	if err := c.ClearCompileFailure(ctx, model, key); err != nil {
		t.Fatalf("ClearCompileFailure repeat: %v", err)
	}

	// 6. Clearing a non-existent key is a no-op
	if err := c.ClearCompileFailure(ctx, model, "non-existent-key"); err != nil {
		t.Fatalf("ClearCompileFailure non-existent: %v", err)
	}
}

func TestCompileFailureStore_ErrorPropagated(t *testing.T) {
	ctx := t.Context()
	errSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer errSrv.Close()

	c := New(errSrv.URL)
	if err := c.ClearCompileFailure(ctx, "model", "key"); err == nil {
		t.Fatal("expected error on 500 internal server error, got nil")
	}
}

func TestHasPackage(t *testing.T) {
	ctx := t.Context()
	ts := fakeRegistry(t)
	defer ts.Close()

	c := New(ts.URL)
	model := "test-model"
	key := "key_submod_2_test"

	// 1. Initially absent
	has, err := c.HasPackage(ctx, model, key)
	if err != nil {
		t.Fatalf("HasPackage before put: %v", err)
	}
	if has {
		t.Fatalf("expected HasPackage to be false before put")
	}

	// 2. Put manifest
	manifest := &Manifest{
		SchemaVersion: 2,
		MediaType:     mediaTypeManifest,
		Config:        EmptyConfig,
	}
	if err := c.PutManifest(ctx, PackageRepository(model), key, manifest); err != nil {
		t.Fatalf("PutManifest: %v", err)
	}

	// 3. Now present
	has, err = c.HasPackage(ctx, model, key)
	if err != nil {
		t.Fatalf("HasPackage after put: %v", err)
	}
	if !has {
		t.Fatalf("expected HasPackage to be true after put")
	}
}

func TestBodyCompileOutcomeStore(t *testing.T) {
	ctx := t.Context()
	srv := fakeRegistry(t)
	defer srv.Close()

	c := New(srv.URL)
	op := "vxpu::fp8_linear"
	body := "ref_linear"
	machineKind := "cpu:intel xeon:avx-512"
	softwareVersion := "v1.0.0"

	// 1. Not found initially: returns nil, nil (not known)
	got, err := c.GetBodyCompileOutcome(ctx, op, body, machineKind, softwareVersion)
	if err != nil {
		t.Fatalf("GetBodyCompileOutcome: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil outcome (not known), got %+v", got)
	}

	// 2. Record failure outcome
	failOutcome := &planpb.BodyCompileOutcome{
		Op:              op,
		Body:            body,
		MachineKind:     machineKind,
		SoftwareVersion: softwareVersion,
		Compiled:        false,
		Reason:          "type error in codegen",
	}
	if err := c.RecordBodyCompileOutcome(ctx, failOutcome); err != nil {
		t.Fatalf("RecordBodyCompileOutcome(fail): %v", err)
	}

	// 3. Get recorded failure outcome (recorded as not compiling)
	got, err = c.GetBodyCompileOutcome(ctx, op, body, machineKind, softwareVersion)
	if err != nil {
		t.Fatalf("GetBodyCompileOutcome after failure record: %v", err)
	}
	if got == nil {
		t.Fatal("expected outcome, got nil")
	}
	if got.Compiled || got.Reason != failOutcome.Reason {
		t.Errorf("got %+v, want compiled=false with reason %q", got, failOutcome.Reason)
	}

	// A failure with no reason is refused
	failNoReason := &planpb.BodyCompileOutcome{
		Op:              op,
		Body:            body,
		MachineKind:     machineKind,
		SoftwareVersion: softwareVersion,
		Compiled:        false,
		Reason:          "",
	}
	if err := c.RecordBodyCompileOutcome(ctx, failNoReason); err == nil {
		t.Fatal("expected error recording failure without reason, got nil")
	}

	// 4. Record success outcome under the same identity: replaces the failure outcome
	// A success carrying an old reason drops the reason.
	successOutcome := &planpb.BodyCompileOutcome{
		Op:              op,
		Body:            body,
		MachineKind:     machineKind,
		SoftwareVersion: softwareVersion,
		Compiled:        true,
		Reason:          "stale failure reason that should be dropped",
	}
	if err := c.RecordBodyCompileOutcome(ctx, successOutcome); err != nil {
		t.Fatalf("RecordBodyCompileOutcome(success): %v", err)
	}

	got, err = c.GetBodyCompileOutcome(ctx, op, body, machineKind, softwareVersion)
	if err != nil {
		t.Fatalf("GetBodyCompileOutcome after success record: %v", err)
	}
	if got == nil || !got.Compiled || got.Reason != "" {
		t.Errorf("got %+v, want compiled=true with empty reason replacing failure", got)
	}

	// 5. Another software version is not known initially
	gotV2, err := c.GetBodyCompileOutcome(ctx, op, body, machineKind, "v2.0.0")
	if err != nil || gotV2 != nil {
		t.Fatalf("expected v2 not known (nil), got %+v", gotV2)
	}

	// 6. Record on machine kind with prefix relationship (e.g. gpu:nvidia l4 vs gpu:nvidia l40)
	outcomeL4 := &planpb.BodyCompileOutcome{
		Op:              op,
		Body:            body,
		MachineKind:     "gpu:nvidia l4",
		SoftwareVersion: softwareVersion,
		Compiled:        true,
	}
	outcomeL40 := &planpb.BodyCompileOutcome{
		Op:              op,
		Body:            body,
		MachineKind:     "gpu:nvidia l40",
		SoftwareVersion: softwareVersion,
		Compiled:        false,
		Reason:          "missing cuda architecture 89",
	}
	if err := c.RecordBodyCompileOutcome(ctx, outcomeL4); err != nil {
		t.Fatalf("RecordBodyCompileOutcome(l4): %v", err)
	}
	if err := c.RecordBodyCompileOutcome(ctx, outcomeL40); err != nil {
		t.Fatalf("RecordBodyCompileOutcome(l40): %v", err)
	}

	// 7. List recorded outcomes for gpu:nvidia l4 returns only l4
	listL4, err := c.ListBodyCompileOutcomes(ctx, "gpu:nvidia l4")
	if err != nil {
		t.Fatalf("ListBodyCompileOutcomes(gpu:nvidia l4): %v", err)
	}
	if len(listL4) != 1 || listL4[0].MachineKind != "gpu:nvidia l4" || !listL4[0].Compiled {
		t.Errorf("ListBodyCompileOutcomes(gpu:nvidia l4) = %+v, want 1 compiled item on l4", listL4)
	}

	// 8. Record validation: every field of the identity is required
	if err := c.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{Body: body, MachineKind: machineKind, SoftwareVersion: softwareVersion}); err == nil {
		t.Fatal("expected error recording without op, got nil")
	}
	if err := c.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{Op: op, MachineKind: machineKind, SoftwareVersion: softwareVersion}); err == nil {
		t.Fatal("expected error recording without body, got nil")
	}
	if err := c.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{Op: op, Body: body, SoftwareVersion: softwareVersion}); err == nil {
		t.Fatal("expected error recording without machine kind, got nil")
	}
	if err := c.RecordBodyCompileOutcome(ctx, &planpb.BodyCompileOutcome{Op: op, Body: body, MachineKind: machineKind}); err == nil {
		t.Fatal("expected error recording without software version, got nil")
	}
}

func TestBodyCompileOutcomeStore_NoLayersError(t *testing.T) {
	ctx := t.Context()
	srv := fakeRegistry(t)
	defer srv.Close()

	c := New(srv.URL)
	op := "vxpu::fp8_linear"
	body := "ref_linear"
	machineKind := "cpu:intel xeon:avx-512"
	softwareVersion := "v1.0.0"

	tag := BodyOutcomeTag(op, body, machineKind, softwareVersion)
	m := &Manifest{
		SchemaVersion: 2,
		MediaType:     mediaTypeManifest,
		ArtifactType:  outcomeArtifactType,
		Config:        EmptyConfig,
		Layers:        nil,
	}
	if err := c.PutManifest(ctx, BodyOutcomeRepository(), tag, m); err != nil {
		t.Fatalf("PutManifest: %v", err)
	}

	if _, err := c.GetBodyCompileOutcome(ctx, op, body, machineKind, softwareVersion); err == nil {
		t.Fatal("expected error from GetBodyCompileOutcome when manifest has no layers, got nil")
	}

	if _, err := c.ListBodyCompileOutcomes(ctx, machineKind); err == nil {
		t.Fatal("expected error from ListBodyCompileOutcomes when manifest has no layers, got nil")
	}
}

func TestParseArtifactRef(t *testing.T) {
	repo, tag, err := ParseArtifactRef("vxpu-artifacts/org/model:v1")
	if err != nil || repo != "vxpu-artifacts/org/model" || tag != "v1" {
		t.Fatalf("ParseArtifactRef valid = (%q, %q, %v)", repo, tag, err)
	}
	for _, bad := range []string{"", "model-without-tag", ":only-tag", "repo:", "a:b:c", "/local/path", "repo :tag"} {
		if _, _, err := ParseArtifactRef(bad); err == nil {
			t.Errorf("ParseArtifactRef(%q) succeeded, want error", bad)
		}
	}
}
