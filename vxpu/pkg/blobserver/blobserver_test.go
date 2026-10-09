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

package blobserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestHandleBlob(t *testing.T) {
	tempDir := t.TempDir()
	s := &Server{
		cacheDir: tempDir,
	}

	// Create a fake blob file in the cache directory
	sha := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" // sha256 of "hello"
	content := []byte("hello world from cached blob!")
	err := os.WriteFile(filepath.Join(tempDir, sha), content, 0644)
	if err != nil {
		t.Fatalf("failed to write test blob file: %v", err)
	}

	// Test regular HTTP GET request without Range header
	req, err := http.NewRequest("GET", "/blobs/"+sha, nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	rr := httptest.NewRecorder()
	s.handleBlob(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code 200, got %d", rr.Code)
	}
	if !bytes.Equal(rr.Body.Bytes(), content) {
		t.Errorf("Expected body %q, got %q", string(content), rr.Body.String())
	}

	// Test HTTP GET request with Range header (bytes=6-10)
	reqRange, err := http.NewRequest("GET", "/blobs/"+sha, nil)
	if err != nil {
		t.Fatalf("failed to create range request: %v", err)
	}
	reqRange.Header.Set("Range", "bytes=6-10")
	rrRange := httptest.NewRecorder()
	s.handleBlob(rrRange, reqRange)

	if rrRange.Code != http.StatusPartialContent {
		t.Errorf("Expected status code 206, got %d", rrRange.Code)
	}
	expectedSubStr := "world"
	if rrRange.Body.String() != expectedSubStr {
		t.Errorf("Expected range body %q, got %q", expectedSubStr, rrRange.Body.String())
	}
}

func TestCacheAndRewriteManifest(t *testing.T) {
	// Spin up a dummy HTTP server to serve the mock Hugging Face model weight files
	fileContent := []byte("fake model weight content")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(fileContent)
	}))
	defer ts.Close()

	tempDir := t.TempDir()
	s, err := NewServer(tempDir, 0)
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	defer s.Close()

	sha := "1ab0ee7e6a298b845e6e6dc4b2dd383a06ca030d42a9d7f251eba1306cb63a4f" // sha256 of "fake model weight content"
	manifestData := v1alpha1.Manifest{
		Format: "vxpu-manifest/v1alpha1",
		Source: v1alpha1.ManifestSource{
			Repo:      "test-repo",
			Revision:  "main",
			CommitSHA: "abcdef123456",
		},
		Config: json.RawMessage(`{"model_type":"llama"}`),
		Files: map[string]v1alpha1.ManifestFile{
			sha: {
				Size:   int64(len(fileContent)),
				Name:   "model.safetensors",
				Source: ts.URL + "/model.safetensors",
			},
		},
		Tensors: json.RawMessage(`{"layernorm":"some-tensor-metadata"}`),
	}

	manifestBytes, err := json.Marshal(manifestData)
	if err != nil {
		t.Fatalf("failed to marshal original manifest: %v", err)
	}

	routerIP := "127.0.0.1"
	rewrittenJSON, err := s.CacheAndRewriteManifest(t.Context(), string(manifestBytes), routerIP)
	if err != nil {
		t.Fatalf("CacheAndRewriteManifest failed: %v", err)
	}

	// Unmarshal back to check if all keys are completely preserved
	var result v1alpha1.Manifest
	if err := json.Unmarshal([]byte(rewrittenJSON), &result); err != nil {
		t.Fatalf("failed to unmarshal rewritten manifest: %v", err)
	}

	if result.Format != "vxpu-manifest/v1alpha1" {
		t.Errorf("Expected Format 'vxpu-manifest/v1alpha1', got %q", result.Format)
	}
	if result.Source.Repo != "test-repo" {
		t.Errorf("Expected Source Repo 'test-repo', got %q", result.Source.Repo)
	}
	if string(result.Config) != `{"model_type":"llama"}` {
		t.Errorf("Expected Config to be preserved verbatim, got %s", result.Config)
	}
	if string(result.Tensors) != `{"layernorm":"some-tensor-metadata"}` {
		t.Errorf("Expected Tensors to be preserved verbatim, got %s", result.Tensors)
	}

	// Verify that the files URL was rewritten correctly to point to the local blob server
	rewrittenFile, exists := result.Files[sha]
	if !exists {
		t.Fatalf("Expected file sha to exist in Files map")
	}
	expectedSource := fmt.Sprintf("http://%s:%d/blobs/%s", routerIP, s.HTTPPort(), sha)
	if rewrittenFile.Source != expectedSource {
		t.Errorf("Expected rewritten file source %q, got %q", expectedSource, rewrittenFile.Source)
	}

	// Verify that the file was indeed cached on disk
	cachedData, err := os.ReadFile(filepath.Join(tempDir, sha))
	if err != nil {
		t.Fatalf("failed to read cached file: %v", err)
	}
	if !bytes.Equal(cachedData, fileContent) {
		t.Errorf("Expected cached file content %q, got %q", string(fileContent), string(cachedData))
	}
}

// The rewritten manifest must reproduce config and tensors byte for
// byte: transformers validates config field types, and a Go number
// round trip would turn 30.0 into 30.
func TestCacheAndRewriteManifest_PreservesConfigVerbatim(t *testing.T) {
	s := &Server{cacheDir: t.TempDir(), httpPort: 8080}
	in := `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"r","revision":"main","commit_sha":"c"},` +
		`"config":{"final_logit_softcapping":30.0,"rope_theta":1000000.0,"attention_dropout":0.0,"nested":{"x":1.0}},` +
		`"files":{},"tensors":{"w":{"dtype":"BF16","length":4,"offset":0,"shape":[2],"file_sha256":"f"}}}`
	out, err := s.CacheAndRewriteManifest(t.Context(), in, "10.0.0.1")
	if err != nil {
		t.Fatalf("CacheAndRewriteManifest: %v", err)
	}
	var got v1alpha1.Manifest
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("rewritten manifest is not JSON: %v", err)
	}
	wantConfig := `{"final_logit_softcapping":30.0,"rope_theta":1000000.0,"attention_dropout":0.0,"nested":{"x":1.0}}`
	if string(got.Config) != wantConfig {
		t.Errorf("config altered by round trip:\n got %s\nwant %s", got.Config, wantConfig)
	}
	wantTensors := `{"w":{"dtype":"BF16","length":4,"offset":0,"shape":[2],"file_sha256":"f"}}`
	if string(got.Tensors) != wantTensors {
		t.Errorf("tensors altered by round trip:\n got %s\nwant %s", got.Tensors, wantTensors)
	}
}

// A concurrent request for a blob that is already downloading must wait
// for that download rather than start a second copy.
func TestDownloadBlob_ConcurrentRequestsJoinOneDownload(t *testing.T) {
	content := []byte("weights weights weights")
	sum := sha256.Sum256(content)
	sha := hex.EncodeToString(sum[:])

	var requests atomic.Int32
	release := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		<-release // hold the first download open until both callers are in
		_, _ = w.Write(content)
	}))
	defer ts.Close()

	s := &Server{cacheDir: t.TempDir()}
	file := v1alpha1.ManifestFile{Size: int64(len(content)), Name: "m.safetensors", Source: ts.URL}

	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() { errs <- s.downloadBlobWithCache(t.Context(), sha, file) }()
	}
	// Let both goroutines reach claim(), then let the download finish.
	deadline := time.After(5 * time.Second)
	for requests.Load() == 0 {
		select {
		case <-deadline:
			t.Fatal("download never started")
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			t.Fatalf("download failed: %v", err)
		}
	}
	if got := requests.Load(); got != 1 {
		t.Errorf("expected one upstream request, got %d", got)
	}
	if _, err := os.Stat(filepath.Join(s.cacheDir, sha)); err != nil {
		t.Errorf("blob not cached: %v", err)
	}
}
