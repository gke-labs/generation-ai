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
