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

package registry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
)

func newReg(t *testing.T) *Registry {
	t.Helper()
	store, err := blobstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return New(store)
}

func digestOf(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

func do(reg *Registry, method, path string, body io.Reader, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, body)
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	w := httptest.NewRecorder()
	handled, err := reg.Handle(w, req)
	if !handled {
		w.WriteHeader(http.StatusTeapot)
	}
	if err != nil && !w.Flushed && w.Code == 200 {
		w.Code = http.StatusInternalServerError
	}
	return w
}

func TestVersionCheckAndNames(t *testing.T) {
	reg := newReg(t)
	if w := do(reg, http.MethodGet, "/v2/", nil); w.Code != 200 || w.Body.String() != "{}" {
		t.Errorf("/v2/: %d %q", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodGet, "/other", nil); w.Code != http.StatusTeapot {
		t.Errorf("non-registry path should not be handled: %d", w.Code)
	}
	for name, ok := range map[string]bool{"qwen/qwen3-8b-fp8": true, "a": true, "a.b_c-d/e": true,
		"Qwen/Qwen3": false, "a//b": false, "-a": false, "a/": false} {
		if ValidName(name) != ok {
			t.Errorf("ValidName(%q) = %v, want %v", name, !ok, ok)
		}
	}
	w := do(reg, http.MethodGet, "/v2/Bad-Name/blobs/sha256:"+strings.Repeat("a", 64), nil)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "NAME_INVALID") {
		t.Errorf("invalid name: %d %s", w.Code, w.Body.String())
	}
}

func TestBlobUploadsAndRanges(t *testing.T) {
	reg := newReg(t)
	content := []byte("0123456789abcdef")
	digest := digestOf(content)

	// Two-step monolithic upload, the form oras and crane use.
	w := do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/", nil)
	if w.Code != http.StatusAccepted || w.Header().Get("Location") == "" {
		t.Fatalf("POST uploads: %d %v", w.Code, w.Header())
	}
	loc := w.Header().Get("Location")
	w = do(reg, http.MethodPut, loc+"?digest="+digest, bytes.NewReader(content))
	if w.Code != http.StatusCreated || w.Header().Get("Docker-Content-Digest") != digest {
		t.Fatalf("PUT upload: %d %s", w.Code, w.Body.String())
	}
	// A session cannot be reused, and a wrong digest leaves no blob.
	if w := do(reg, http.MethodPut, loc+"?digest="+digest, bytes.NewReader(content)); w.Code != http.StatusNotFound {
		t.Errorf("reused session: %d", w.Code)
	}
	wrong := "sha256:" + strings.Repeat("0", 64)
	w = do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+wrong, bytes.NewReader(content))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "DIGEST_INVALID") {
		t.Errorf("wrong digest: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodHead, "/v2/org/name/blobs/"+wrong, nil); w.Code != http.StatusNotFound {
		t.Errorf("wrong-digest blob should not exist: %d", w.Code)
	}
	// Single-POST monolithic upload and cross-repository mount.
	other := []byte("other blob")
	if w := do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+digestOf(other), bytes.NewReader(other)); w.Code != http.StatusCreated {
		t.Errorf("single POST: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodPost, "/v2/else/where/blobs/uploads/?mount="+digest+"&from=org/name", nil); w.Code != http.StatusCreated {
		t.Errorf("mount: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodPost, "/v2/else/where/blobs/uploads/?mount="+wrong+"&from=org/name", nil); w.Code != http.StatusAccepted {
		t.Errorf("mount of a missing blob should open a session: %d", w.Code)
	}
	if w := do(reg, http.MethodPatch, loc, bytes.NewReader(content)); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("chunked upload: %d, want 405", w.Code)
	}

	// Reads, with ranges, from any name (blobs are shared by digest).
	w = do(reg, http.MethodHead, "/v2/else/where/blobs/"+digest, nil)
	if w.Code != 200 || w.Header().Get("Content-Length") != "16" || w.Header().Get("Docker-Content-Digest") != digest {
		t.Errorf("HEAD: %d %v", w.Code, w.Header())
	}
	for _, tc := range []struct{ rng, want, cr string }{
		{"bytes=4-7", "4567", "bytes 4-7/16"}, {"bytes=12-", "cdef", "bytes 12-15/16"},
		{"bytes=-3", "def", "bytes 13-15/16"}, {"bytes=10-99", "abcdef", "bytes 10-15/16"},
	} {
		w := do(reg, http.MethodGet, "/v2/org/name/blobs/"+digest, nil, "Range", tc.rng)
		if w.Code != http.StatusPartialContent || w.Body.String() != tc.want || w.Header().Get("Content-Range") != tc.cr {
			t.Errorf("Range %s: %d %q %q", tc.rng, w.Code, w.Body.String(), w.Header().Get("Content-Range"))
		}
	}
	if w := do(reg, http.MethodGet, "/v2/org/name/blobs/"+digest, nil, "Range", "bytes=16-20"); w.Code != http.StatusRequestedRangeNotSatisfiable {
		t.Errorf("unsatisfiable range: %d", w.Code)
	}
	if w := do(reg, http.MethodDelete, "/v2/org/name/blobs/"+digest, nil); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("delete: %d", w.Code)
	}
}

func TestManifestsAndTags(t *testing.T) {
	reg := newReg(t)
	weights := []byte("weights")
	do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+digestOf(weights), bytes.NewReader(weights))
	manifest := Manifest{SchemaVersion: 2, MediaType: MediaTypeManifest, ArtifactType: ArtifactWeights, Config: EmptyConfig,
		Layers: []Descriptor{{MediaType: MediaTypeFile, Digest: digestOf(weights), Size: 7, Annotations: map[string]string{AnnotationTitle: "model.safetensors"}}}}
	body, _ := json.Marshal(manifest)
	mdigest := digestOf(body)

	// A manifest referencing a missing blob is refused.
	missing := manifest
	missing.Layers = []Descriptor{{MediaType: MediaTypeFile, Digest: "sha256:" + strings.Repeat("1", 64), Size: 1}}
	mb, _ := json.Marshal(missing)
	w := do(reg, http.MethodPut, "/v2/org/name/manifests/main", bytes.NewReader(mb), "Content-Type", MediaTypeManifest)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "MANIFEST_BLOB_UNKNOWN") {
		t.Fatalf("manifest with missing blob: %d %s", w.Code, w.Body.String())
	}
	// Tagged put, then read by tag and by digest; the empty config needs no push.
	w = do(reg, http.MethodPut, "/v2/org/name/manifests/main", bytes.NewReader(body), "Content-Type", MediaTypeManifest)
	if w.Code != http.StatusCreated || w.Header().Get("Docker-Content-Digest") != mdigest {
		t.Fatalf("PUT manifest: %d %s", w.Code, w.Body.String())
	}
	for _, ref := range []string{"main", mdigest} {
		w := do(reg, http.MethodGet, "/v2/org/name/manifests/"+ref, nil)
		if w.Code != 200 || w.Header().Get("Content-Type") != MediaTypeManifest || w.Header().Get("Docker-Content-Digest") != mdigest || !bytes.Equal(w.Body.Bytes(), body) {
			t.Errorf("GET manifest %s: %d %v", ref, w.Code, w.Header())
		}
	}
	if w := do(reg, http.MethodHead, "/v2/org/name/manifests/missing", nil); w.Code != http.StatusNotFound {
		t.Errorf("unknown tag: %d", w.Code)
	}
	if w := do(reg, http.MethodGet, "/v2/other/name/manifests/"+mdigest, nil); w.Code != http.StatusNotFound {
		t.Errorf("manifest digests are per name: %d", w.Code)
	}
	if w := do(reg, http.MethodGet, "/v2/org/name/blobs/"+EmptyConfig.Digest, nil); w.Code != 200 || w.Body.String() != "{}" {
		t.Errorf("empty config blob: %d %q", w.Code, w.Body.String())
	}
	// An index over the manifest, and the tag list.
	index := Index{SchemaVersion: 2, MediaType: MediaTypeIndex, Manifests: []Descriptor{{MediaType: MediaTypeManifest, Digest: mdigest, Size: int64(len(body)), ArtifactType: ArtifactWeights}}}
	ib, _ := json.Marshal(index)
	if w := do(reg, http.MethodPut, "/v2/org/name/manifests/v1", bytes.NewReader(ib), "Content-Type", MediaTypeIndex); w.Code != http.StatusCreated {
		t.Fatalf("PUT index: %d %s", w.Code, w.Body.String())
	}
	bad := Index{SchemaVersion: 2, MediaType: MediaTypeIndex, Manifests: []Descriptor{{Digest: "sha256:" + strings.Repeat("2", 64)}}}
	bb, _ := json.Marshal(bad)
	if w := do(reg, http.MethodPut, "/v2/org/name/manifests/v2", bytes.NewReader(bb), "Content-Type", MediaTypeIndex); w.Code != http.StatusBadRequest {
		t.Errorf("index over a missing manifest: %d", w.Code)
	}
	w = do(reg, http.MethodGet, "/v2/org/name/tags/list", nil)
	var tags struct {
		Name string   `json:"name"`
		Tags []string `json:"tags"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &tags)
	if tags.Name != "org/name" || strings.Join(tags.Tags, ",") != "main,v1" {
		t.Errorf("tags: %+v", tags)
	}
	names, _ := reg.Repositories(t.Context())
	if strings.Join(names, ",") != "org/name" {
		t.Errorf("repositories: %v", names)
	}
}

// A repository may be called "blobs", "manifests" or "tags": the route
// is the reserved component nearest the end, never the first found.
func TestRoutesWithReservedNameComponents(t *testing.T) {
	reg := newReg(t)
	w := do(reg, http.MethodGet, "/v2/someone/blobs/tags/list", nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"someone/blobs"`) {
		t.Errorf("tags of someone/blobs: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodGet, "/v2/someone/blobs/manifests/main", nil); w.Code != http.StatusNotFound || !strings.Contains(w.Body.String(), "MANIFEST_UNKNOWN") {
		t.Errorf("manifest of someone/blobs: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodPost, "/v2/someone/manifests/blobs/uploads/", nil); w.Code != http.StatusAccepted {
		t.Errorf("upload under someone/manifests: %d %s", w.Code, w.Body.String())
	}
}

// A refused tag leaves nothing behind, and a suffix range of an empty
// blob is unsatisfiable rather than a 206 with a nonsense Content-Range.
func TestTagValidatedBeforeWriteAndEmptyBlobRange(t *testing.T) {
	reg := newReg(t)
	manifest := []byte(`{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"mediaType":"application/vnd.oci.empty.v1+json","digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","size":2},"layers":[]}`)
	w := do(reg, http.MethodPut, "/v2/org/name/manifests/foo/bar", bytes.NewReader(manifest), "Content-Type", MediaTypeManifest)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "TAG_INVALID") {
		t.Fatalf("manifest with a bad tag: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodHead, "/v2/org/name/manifests/"+digestOf(manifest), nil); w.Code != http.StatusNotFound {
		t.Errorf("refused manifest is fetchable: %d", w.Code)
	}
	empty := digestOf(nil)
	if w := do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+empty, bytes.NewReader(nil)); w.Code != http.StatusCreated {
		t.Fatalf("empty blob: %d %s", w.Code, w.Body.String())
	}
	if w := do(reg, http.MethodGet, "/v2/org/name/blobs/"+empty, nil, "Range", "bytes=-5"); w.Code != http.StatusRequestedRangeNotSatisfiable {
		t.Errorf("suffix range of an empty blob: %d %q", w.Code, w.Header().Get("Content-Range"))
	}
}

// TestServeBlobLargeRange verifies that blobs larger than the 2 MiB copy buffer
// stream correctly across buffer boundaries.
func TestServeBlobLargeRange(t *testing.T) {
	reg := newReg(t)
	// 5 MiB test payload: exercises multiple 2 MiB buffer copy iterations
	content := make([]byte, 5<<20)
	for i := range content {
		content[i] = byte(i % 251)
	}
	digest := digestOf(content)
	rawSha := strings.TrimPrefix(digest, "sha256:")

	w := do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+digest, bytes.NewReader(content))
	if w.Code != http.StatusCreated {
		t.Fatalf("upload large blob: %d %s", w.Code, w.Body.String())
	}

	// 1. Read entire blob
	req := httptest.NewRequest(http.MethodGet, "/blobs/"+rawSha, nil)
	rec := httptest.NewRecorder()
	if err := reg.ServeBlob(rec, req, rawSha); err != nil {
		t.Fatalf("ServeBlob full: %v", err)
	}
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), content) {
		t.Fatalf("ServeBlob full mismatch: code=%d len=%d", rec.Code, rec.Body.Len())
	}

	// 2. Read range spanning across the 2 MiB boundary (1 MiB to 3 MiB)
	req = httptest.NewRequest(http.MethodGet, "/blobs/"+rawSha, nil)
	req.Header.Set("Range", "bytes=1048576-3145727")
	rec = httptest.NewRecorder()
	if err := reg.ServeBlob(rec, req, rawSha); err != nil {
		t.Fatalf("ServeBlob range: %v", err)
	}
	if rec.Code != http.StatusPartialContent || !bytes.Equal(rec.Body.Bytes(), content[1048576:3145728]) {
		t.Fatalf("ServeBlob range mismatch: code=%d len=%d", rec.Code, rec.Body.Len())
	}
}

func BenchmarkServeBlob(b *testing.B) {
	reg := newReg(&testing.T{})
	content := make([]byte, 64<<20) // 64 MB
	digest := digestOf(content)
	rawSha := strings.TrimPrefix(digest, "sha256:")

	w := do(reg, http.MethodPost, "/v2/org/name/blobs/uploads/?digest="+digest, bytes.NewReader(content))
	if w.Code != http.StatusCreated {
		b.Fatalf("upload: %d %s", w.Code, w.Body.String())
	}

	b.SetBytes(64 << 20)
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		req := httptest.NewRequest(http.MethodGet, "/blobs/"+rawSha, nil)
		req.Header.Set("Range", "bytes=0-67108863")
		for pb.Next() {
			rec := httptest.NewRecorder()
			if err := reg.ServeBlob(rec, req, rawSha); err != nil {
				b.Fatal(err)
			}
			if rec.Code != http.StatusPartialContent {
				b.Fatalf("code: %d", rec.Code)
			}
		}
	})
}
