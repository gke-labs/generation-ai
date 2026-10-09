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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

type Server struct {
	cacheDir string
	httpPort int
	listener net.Listener

	// inflight dedups concurrent downloads of one blob: a retry (a
	// client whose RPC timed out and called LoadModel again) waits for
	// the running download instead of starting a second copy.
	mu       sync.Mutex
	inflight map[string]chan struct{}
}

func NewServer(cacheDir string, httpPort int) (*Server, error) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", httpPort))
	if err != nil {
		// Fallback to any available port
		lis, err = net.Listen("tcp", ":0")
		if err != nil {
			return nil, fmt.Errorf("failed to start HTTP server listener: %w", err)
		}
	}

	addr := lis.Addr().(*net.TCPAddr)
	s := &Server{
		cacheDir: cacheDir,
		httpPort: addr.Port,
		listener: lis,
	}

	return s, nil
}

func (s *Server) HTTPPort() int {
	return s.httpPort
}

func (s *Server) Start() {
	mux := http.NewServeMux()
	mux.HandleFunc("/blobs/", s.handleBlob)

	klog.Infof("Started blobserver HTTP server on port %d", s.httpPort)
	go func() {
		if err := http.Serve(s.listener, mux); err != nil && err != http.ErrServerClosed {
			klog.Errorf("HTTP server error: %v", err)
		}
	}()
}

func (s *Server) Close() error {
	if s.listener != nil {
		return s.listener.Close()
	}
	return nil
}

func (s *Server) handleBlob(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 || parts[2] == "" {
		http.Error(w, "missing blob SHA256", http.StatusBadRequest)
		return
	}
	sha256Hash := parts[2]

	filePath := filepath.Join(s.cacheDir, sha256Hash)
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		http.Error(w, "blob not found", http.StatusNotFound)
		return
	}

	http.ServeFile(w, r, filePath)
}

// CacheAndRewriteManifest downloads all files from the manifest into the cache,
// and rewrites the manifest files' source URLs to point to this local blobserver.
func (s *Server) CacheAndRewriteManifest(ctx context.Context, manifestJSON string, routerIP string) (string, error) {
	var manifest v1alpha1.Manifest
	if err := json.Unmarshal([]byte(manifestJSON), &manifest); err != nil {
		return "", fmt.Errorf("failed to parse manifest JSON: %w", err)
	}

	// Download blobs in parallel
	var wg sync.WaitGroup
	errChan := make(chan error, len(manifest.Files))
	for sha, file := range manifest.Files {
		wg.Add(1)
		go func(sha256Hash string, f v1alpha1.ManifestFile) {
			defer wg.Done()
			if err := s.downloadBlobWithCache(ctx, sha256Hash, f); err != nil {
				errChan <- err
			}
		}(sha, file)
	}
	wg.Wait()
	close(errChan)

	var downloadErrors []string
	for err := range errChan {
		downloadErrors = append(downloadErrors, err.Error())
	}
	if len(downloadErrors) > 0 {
		return "", fmt.Errorf("failed to download/cache model blobs: %s", strings.Join(downloadErrors, "; "))
	}

	// Rewrite files source URLs to point to this blobserver
	for sha, file := range manifest.Files {
		file.Source = fmt.Sprintf("http://%s:%d/blobs/%s", routerIP, s.httpPort, sha)
		manifest.Files[sha] = file
	}

	// Marshal rewritten manifest back to JSON
	rewrittenJSON, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("failed to serialize rewritten manifest: %w", err)
	}

	return string(rewrittenJSON), nil
}

func (s *Server) downloadBlobWithCache(ctx context.Context, sha string, file v1alpha1.ManifestFile) error {
	for {
		done, owner := s.claim(sha)
		if owner {
			err := s.downloadBlob(ctx, sha, file)
			s.release(sha)
			return err
		}
		// Another request is downloading this blob; wait for it, then
		// re-check the cache (the loop re-downloads if it failed).
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// claim registers sha as in flight. It returns (nil, true) when the
// caller owns the download, or (done, false) with a channel that
// closes when the current owner finishes.
func (s *Server) claim(sha string) (<-chan struct{}, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inflight == nil {
		s.inflight = make(map[string]chan struct{})
	}
	if done, ok := s.inflight[sha]; ok {
		return done, false
	}
	s.inflight[sha] = make(chan struct{})
	return nil, true
}

func (s *Server) release(sha string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	close(s.inflight[sha])
	delete(s.inflight, sha)
}

func (s *Server) downloadBlob(ctx context.Context, sha string, file v1alpha1.ManifestFile) error {
	destPath := filepath.Join(s.cacheDir, sha)

	if st, err := os.Stat(destPath); err == nil {
		if st.Size() == file.Size {
			klog.Infof("Blob %s already cached, size matches: %d bytes", sha, file.Size)
			return nil
		}
		klog.Warningf("Blob %s exists but size mismatch (expected %d, got %d). Re-downloading.", sha, file.Size, st.Size())
	}

	klog.Infof("Downloading blob from %s to %s", file.Source, destPath)

	if err := os.MkdirAll(s.cacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	tmpFile, err := os.CreateTemp(s.cacheDir, "download-")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if tmpPath != "" {
			_ = tmpFile.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	httpReq, err := http.NewRequestWithContext(ctx, "GET", file.Source, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request for %s: %w", file.Source, err)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to fetch %s: %w", file.Source, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP error fetching %s: status %d %s", file.Source, resp.StatusCode, resp.Status)
	}

	hasher := sha256.New()
	writer := io.MultiWriter(tmpFile, hasher)

	written, err := io.Copy(writer, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save blob %s: %w", file.Source, err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	if written != file.Size {
		return fmt.Errorf("size mismatch for %s: expected %d bytes, wrote %d bytes", file.Source, file.Size, written)
	}

	sum := hex.EncodeToString(hasher.Sum(nil))
	if sum != sha {
		return fmt.Errorf("SHA256 mismatch for %s: expected %s, got %s", file.Source, sha, sum)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("failed to rename temp file to %s: %w", destPath, err)
	}
	tmpPath = ""

	klog.Infof("Successfully downloaded and cached blob %s", sha)
	return nil
}
