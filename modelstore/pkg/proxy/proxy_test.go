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

package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	"github.com/gke-labs/generation-ai/modelstore/pkg/registry"
)

func newTestProxy(t *testing.T, hfEndpoint string) *Proxy {
	t.Helper()
	store, err := blobstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return NewProxy(registry.New(store), importer.New(store, hfEndpoint))
}

func shaOf(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func do(p *Proxy, method, path string, body io.Reader, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, body)
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	w := httptest.NewRecorder()
	p.ServeHTTP(w, req)
	return w
}

func TestRepositoryName(t *testing.T) {
	for in, want := range map[string]string{
		"Qwen/Qwen3-8B-FP8":     "qwen/qwen3-8b-fp8",
		"google/gemma-4-E4B-it": "google/gemma-4-e4b-it",
		"user/repo":             "user/repo",
		"org/my_model":          "org/my_model",
	} {
		if got, err := RepositoryName(in); err != nil || got != want {
			t.Errorf("RepositoryName(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"/", "org//name", "a/"} {
		if _, err := RepositoryName(bad); err == nil {
			t.Errorf("RepositoryName(%q) should fail", bad)
		}
	}
}

func TestPlainBlobRoutes(t *testing.T) {
	p := newTestProxy(t, "")
	content := []byte("0123456789abcdef")
	sha := shaOf(content)
	if w := do(p, http.MethodPut, "/blobs/"+sha, bytes.NewReader(content)); w.Code != http.StatusCreated {
		t.Fatalf("PUT: %d %s", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodPut, "/blobs/"+strings.Repeat("0", 64), bytes.NewReader(content)); w.Code != http.StatusBadRequest {
		t.Errorf("PUT with wrong sha: %d", w.Code)
	}
	w := do(p, http.MethodGet, "/blobs/"+sha, nil, "Range", "bytes=4-7")
	if w.Code != http.StatusPartialContent || w.Body.String() != "4567" {
		t.Errorf("range through /blobs: %d %q", w.Code, w.Body.String())
	}
	// The same blob through the registry route.
	if w := do(p, http.MethodHead, "/v2/any/name/blobs/sha256:"+sha, nil); w.Code != 200 {
		t.Errorf("registry HEAD: %d", w.Code)
	}
	if w := do(p, http.MethodGet, "/blobs/"+strings.Repeat("1", 64), nil); w.Code != http.StatusNotFound {
		t.Errorf("missing blob: %d", w.Code)
	}
}

// fakeHub serves a one-repo Hugging Face Hub: a listing with one LFS
// file (sha256 published) and one small file (no sha256).
func fakeHub(t *testing.T, weights, config []byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/name/revision/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"sha": "abc123",
			"siblings": []map[string]any{
				{"rfilename": "model.safetensors", "size": len(weights),
					"lfs": map[string]any{"sha256": shaOf(weights), "size": len(weights)}},
				{"rfilename": "config.json", "size": len(config)},
			},
		})
	})
	mux.HandleFunc("/org/name/resolve/abc123/model.safetensors", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(weights) })
	mux.HandleFunc("/org/name/resolve/abc123/config.json", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(config) })
	return httptest.NewServer(mux)
}

// waitImport polls GET /models/<name> until the import leaves Importing.
func waitImport(t *testing.T, p *Proxy, name string) ImportStatus {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		w := do(p, http.MethodGet, "/models/"+name, nil)
		var v modelView
		_ = json.Unmarshal(w.Body.Bytes(), &v)
		if v.Status.Phase != PhaseImporting || time.Now().After(deadline) {
			return v.Status
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestModelImportWritesAnArtifact(t *testing.T) {
	weights := bytes.Repeat([]byte("x"), 4096)
	config := []byte(`{"model_type": "test"}`)
	hub := fakeHub(t, weights, config)
	defer hub.Close()
	p := newTestProxy(t, hub.URL)

	w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name", "revision": "main"}`))
	if w.Code != http.StatusAccepted {
		t.Fatalf("import: %d %s", w.Code, w.Body.String())
	}
	var accepted modelView
	_ = json.Unmarshal(w.Body.Bytes(), &accepted)
	if accepted.Metadata.Name != "org/name" {
		t.Errorf("accepted name: %q", accepted.Metadata.Name)
	}
	st := waitImport(t, p, "org/name")
	if st.Phase != PhaseReady || st.FilesImported != 2 || st.BytesImported != int64(len(weights)+len(config)) {
		t.Fatalf("status: %+v", st)
	}
	// The result is an OCI artifact: tags for the revision and the commit, layers named by file.
	w = do(p, http.MethodGet, "/v2/org/name/tags/list", nil)
	if !strings.Contains(w.Body.String(), `"abc123"`) || !strings.Contains(w.Body.String(), `"main"`) {
		t.Errorf("tags: %s", w.Body.String())
	}
	w = do(p, http.MethodGet, "/v2/org/name/manifests/main", nil)
	var m registry.Manifest
	if err := json.Unmarshal(w.Body.Bytes(), &m); err != nil || len(m.Layers) != 2 || m.Annotations[registry.AnnotationHubRepo] != "org/name" {
		t.Fatalf("manifest: %v %s", err, w.Body.String())
	}
	// Hub-compatible reads come from the manifest (any casing of the id).
	w = do(p, http.MethodGet, "/api/models/Org/Name", nil)
	var meta map[string]any
	_ = json.NewDecoder(w.Body).Decode(&meta)
	if meta["sha"] != "abc123" || len(meta["siblings"].([]any)) != 2 {
		t.Errorf("metadata: %v", meta)
	}
	// The tree and paths-info routes huggingface_hub lists with.
	w = do(p, http.MethodGet, "/api/models/Org/Name/tree/abc123?recursive=true&expand=false", nil)
	var tree []map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &tree); err != nil || len(tree) != 2 || tree[0]["type"] != "file" || tree[0]["lfs"] == nil {
		t.Errorf("tree: %d %s", w.Code, w.Body.String())
	}
	w = do(p, http.MethodPost, "/api/models/org/name/paths-info/main", strings.NewReader("paths=config.json&paths=missing"), "Content-Type", "application/x-www-form-urlencoded")
	var infos []map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &infos); err != nil || len(infos) != 1 || infos[0]["path"] != "config.json" {
		t.Errorf("paths-info: %d %s", w.Code, w.Body.String())
	}
	w = do(p, http.MethodGet, "/Org/Name/resolve/main/config.json", nil)
	if w.Code != 200 || w.Body.String() != string(config) || w.Header().Get("X-Repo-Commit") != "abc123" {
		t.Errorf("config via resolve: %d %q commit=%s", w.Code, w.Body.String(), w.Header().Get("X-Repo-Commit"))
	}
	w = do(p, http.MethodGet, "/org/name/resolve/abc123/model.safetensors", nil, "Range", "bytes=0-3")
	if w.Code != http.StatusPartialContent || w.Body.String() != "xxxx" {
		t.Errorf("weights range via commit tag: %d %q", w.Code, w.Body.String())
	}
	// 404s carry the Hub's error codes: huggingface_hub turns them into
	// EntryNotFoundError / RepositoryNotFoundError / RevisionNotFoundError,
	// and transformers skips an optional file only on the first.
	for _, c := range []struct{ path, code string }{
		{"/org/name/resolve/main/missing.bin", "EntryNotFound"},
		{"/org/name/resolve/nosuchrev/config.json", "RevisionNotFound"},
		{"/nobody/nothing/resolve/main/config.json", "RepoNotFound"},
		{"/api/models/nobody/nothing", "RepoNotFound"},
		{"/api/models/org/name/tree/nosuchrev", "RevisionNotFound"},
	} {
		if w := do(p, http.MethodGet, c.path, nil); w.Code != http.StatusNotFound || w.Header().Get("X-Error-Code") != c.code {
			t.Errorf("%s: %d %q, want 404 %s", c.path, w.Code, w.Header().Get("X-Error-Code"), c.code)
		}
	}
	// Listing shows it; a second import of the same repo is allowed, a
	// different repo under the same name is not.
	w = do(p, http.MethodGet, "/models", nil)
	if !strings.Contains(w.Body.String(), `"org/name"`) {
		t.Errorf("list: %s", w.Body.String())
	}
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name"}`)); w.Code != http.StatusAccepted {
		t.Errorf("re-import: %d %s", w.Code, w.Body.String())
	}
	waitImport(t, p, "org/name")
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "Org/Name"}`)); w.Code != http.StatusConflict {
		t.Errorf("other repo under the same name: %d, want 409: %s", w.Code, w.Body.String())
	}
	for _, repo := range []string{"/", "a//b"} {
		if w := do(p, http.MethodPost, "/models/import", strings.NewReader(fmt.Sprintf(`{"repo": %q}`, repo))); w.Code != http.StatusBadRequest {
			t.Errorf("repo %q: %d, want 400", repo, w.Code)
		}
	}
}

func TestBlobImportRefusesForeignURLs(t *testing.T) {
	hub := fakeHub(t, []byte("w"), []byte("{}"))
	defer hub.Close()
	p := newTestProxy(t, hub.URL)
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("modelstore fetched %s from a host that is not the Hub", r.URL)
	}))
	defer other.Close()
	sha := strings.Repeat("a", 64)
	for _, u := range []string{other.URL + "/x", "file:///etc/passwd", "::bad"} {
		w := do(p, http.MethodPost, "/blobs/"+sha+"/import", strings.NewReader(fmt.Sprintf(`{"url": %q}`, u)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("url %q: got %d, want 400: %s", u, w.Code, w.Body.String())
		}
	}
	// And the Hub path works, with a wrong digest leaving nothing behind.
	weights := []byte("w")
	body := fmt.Sprintf(`{"url": %q}`, hub.URL+"/org/name/resolve/abc123/model.safetensors")
	if w := do(p, http.MethodPost, "/blobs/"+shaOf(weights)+"/import?wait=true", strings.NewReader(body)); w.Code != http.StatusCreated {
		t.Errorf("hub import: %d %s", w.Code, w.Body.String())
	}
	wrong := strings.Repeat("3", 64)
	// The client named a digest the url does not produce: its mistake (422), not a server error,
	// and asking again for the same pair is answered from memory rather than fetched again.
	if w := do(p, http.MethodPost, "/blobs/"+wrong+"/import?wait=true", strings.NewReader(body)); w.Code != http.StatusUnprocessableEntity {
		t.Errorf("mismatched import: %d", w.Code)
	}
	if w := do(p, http.MethodHead, "/blobs/"+wrong, nil); w.Code != http.StatusNotFound {
		t.Errorf("mismatched blob should not exist: %d", w.Code)
	}
	if w := do(p, http.MethodPost, "/blobs/"+wrong+"/import", strings.NewReader(body)); w.Code != http.StatusUnprocessableEntity || !strings.Contains(w.Body.String(), "sha256 mismatch") {
		t.Errorf("repeat of a failed pair: %d %s", w.Code, w.Body.String())
	}
}

// A revision that cannot be a tag is refused before anything is
// downloaded, and a second import under a busy name is refused rather
// than silently joined to a different revision.
func TestModelImportRefusesBadRevisionAndBusyName(t *testing.T) {
	hub := fakeHub(t, []byte("w"), []byte("{}"))
	defer hub.Close()
	p := newTestProxy(t, hub.URL)
	for _, rev := range []string{"refs/pr/1", "release/v1", "a b"} {
		w := do(p, http.MethodPost, "/models/import", strings.NewReader(fmt.Sprintf(`{"repo": "org/name", "revision": %q}`, rev)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("revision %q: %d, want 400", rev, w.Code)
		}
	}
	p.mu.Lock()
	p.imports["org/name"] = &ImportStatus{Phase: PhaseImporting, Repo: "org/name", Revision: "main"}
	p.mu.Unlock()
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name", "revision": "main"}`)); w.Code != http.StatusAccepted {
		t.Errorf("same revision joins: %d", w.Code)
	}
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name", "revision": "v2"}`)); w.Code != http.StatusConflict {
		t.Errorf("other revision while importing: %d, want 409", w.Code)
	}
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "Org/Name", "revision": "main"}`)); w.Code != http.StatusConflict {
		t.Errorf("other repo under the same name while importing: %d, want 409", w.Code)
	}
}

func TestModelImportEscapesFileNames(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/name/revision/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"sha": "c1",
			"siblings": []map[string]any{{"rfilename": "notes#1.md", "size": 5}, {"rfilename": "100%.json", "size": 2}}})
	})
	mux.HandleFunc("/org/name/resolve/c1/notes", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("WRONG")) })
	mux.HandleFunc("/org/name/resolve/c1/notes%231.md", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("right")) })
	mux.HandleFunc("/org/name/resolve/c1/100%25.json", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("{}")) })
	hub := httptest.NewServer(mux)
	defer hub.Close()
	p := newTestProxy(t, hub.URL)
	if w := do(p, http.MethodPost, "/models/import", strings.NewReader(`{"repo": "org/name"}`)); w.Code != http.StatusAccepted {
		t.Fatalf("import: %d %s", w.Code, w.Body.String())
	}
	if st := waitImport(t, p, "org/name"); st.Phase != PhaseReady {
		t.Fatalf("phase %q: %s", st.Phase, st.Message)
	}
	w := do(p, http.MethodGet, "/org/name/resolve/main/notes%231.md", nil)
	if w.Code != 200 || w.Body.String() != "right" {
		t.Errorf("escaped file: %d %q", w.Code, w.Body.String())
	}
}

// A model pushed the registry way (tag "latest", as modelstore upload
// does) is readable through the Hub facade at "main", which is what
// transformers asks for when no revision is given.
func TestMainReadsLatest(t *testing.T) {
	p := newTestProxy(t, "http://unused.invalid")
	cfg := `{"model_type": "opt"}`
	if w := do(p, http.MethodPut, "/blobs/c30559e07871356681c6f76200d795084b1b523a6d5ed4935c94297da8f5850e", strings.NewReader(cfg)); w.Code != http.StatusCreated {
		t.Fatalf("blob put: %d %s", w.Code, w.Body.String())
	}
	manifest := `{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",` +
		`"artifactType":"application/vnd.gke-labs.model.weights.v1+json",` +
		`"config":{"mediaType":"application/vnd.oci.empty.v1+json","digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","size":2},` +
		`"layers":[{"mediaType":"application/vnd.gke-labs.model.file.v1","digest":"sha256:c30559e07871356681c6f76200d795084b1b523a6d5ed4935c94297da8f5850e","size":21,` +
		`"annotations":{"org.opencontainers.image.title":"config.json"}}]}`
	w := do(p, http.MethodPut, "/v2/opt-125m/manifests/latest", strings.NewReader(manifest), "Content-Type", "application/vnd.oci.image.manifest.v1+json")
	if w.Code != http.StatusCreated {
		t.Fatalf("manifest put: %d %s", w.Code, w.Body.String())
	}
	w = do(p, http.MethodGet, "/api/models/opt-125m/tree/main", nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"config.json"`) {
		t.Errorf("tree at main: %d %s", w.Code, w.Body.String())
	}
	w = do(p, http.MethodGet, "/api/models/opt-125m/revision/main", nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"sha256":"c30559e07871356681c6f76200d795084b1b523a6d5ed4935c94297da8f5850e"`) {
		t.Errorf("model info carries each file's sha256: %d %s", w.Code, w.Body.String())
	}
	w = do(p, http.MethodGet, "/opt-125m/resolve/main/config.json", nil)
	if w.Code != 200 || w.Body.String() != cfg {
		t.Errorf("resolve at main: %d %q", w.Code, w.Body.String())
	}
	if w := do(p, http.MethodGet, "/opt-125m/resolve/v2/config.json", nil); w.Code != http.StatusNotFound || w.Header().Get("X-Error-Code") != "RevisionNotFound" {
		t.Errorf("other revisions do not fall back: %d %s", w.Code, w.Header().Get("X-Error-Code"))
	}
}

func TestDebugPprof(t *testing.T) {
	p := newTestProxy(t, "")
	for _, path := range []string{"/debug/pprof/", "/debug/pprof/cmdline", "/debug/pprof/goroutine"} {
		w := do(p, http.MethodGet, path, nil)
		if w.Code != http.StatusOK {
			t.Errorf("GET %s = %d, want 200", path, w.Code)
		}
	}
}
