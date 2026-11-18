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
)

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
