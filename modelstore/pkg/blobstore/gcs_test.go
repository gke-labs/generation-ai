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

package blobstore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// TestGCS runs against a real bucket when MODELSTORE_TEST_BUCKET is set
// (gs://bucket[/prefix]); application default credentials must allow
// object admin there. It is skipped otherwise.
func TestGCS(t *testing.T) {
	location := os.Getenv("MODELSTORE_TEST_BUCKET")
	if location == "" {
		t.Skip("MODELSTORE_TEST_BUCKET not set")
	}
	ctx := t.Context()
	store, err := NewGCS(ctx, strings.TrimSuffix(location, "/")+fmt.Sprintf("/test-%d", time.Now().UnixNano()))
	if err != nil {
		t.Fatal(err)
	}
	content := bytes.Repeat([]byte("gcs blob "), 100000) // ~900 KB
	sum := sha256.Sum256(content)
	sha := hex.EncodeToString(sum[:])

	if _, err := store.Stat(ctx, sha); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Stat before Put: %v", err)
	}
	// A mismatched digest publishes nothing.
	if _, _, err := store.Put(ctx, strings.Repeat("0", 64), bytes.NewReader(content)); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("Put with wrong sha: %v", err)
	}
	if _, err := store.Stat(ctx, strings.Repeat("0", 64)); !errors.Is(err, ErrNotFound) {
		t.Fatalf("mismatched Put left a blob: %v", err)
	}
	got, n, err := store.Put(ctx, sha, bytes.NewReader(content))
	if err != nil || got != sha || n != int64(len(content)) {
		t.Fatalf("Put: %v got=%s n=%d", err, got, n)
	}
	// Named by content when no digest is given.
	got2, _, err := store.Put(ctx, "", bytes.NewReader(content))
	if err != nil || got2 != sha {
		t.Fatalf("Put by content: %v got=%s", err, got2)
	}
	size, err := store.Stat(ctx, sha)
	if err != nil || size != int64(len(content)) {
		t.Fatalf("Stat: %v size=%d", err, size)
	}
	rc, err := store.Open(ctx, sha, 9, 8)
	if err != nil {
		t.Fatal(err)
	}
	part, _ := io.ReadAll(rc)
	_ = rc.Close()
	if string(part) != "gcs blob" {
		t.Fatalf("range read: %q", part)
	}
	rc, err = store.Open(ctx, sha, int64(len(content))-4, -1)
	if err != nil {
		t.Fatal(err)
	}
	tail, _ := io.ReadAll(rc)
	_ = rc.Close()
	if string(tail) != "lob " {
		t.Fatalf("tail read: %q", tail)
	}
	if !strings.HasPrefix(store.DirectURL(sha), "gs://") || !strings.HasSuffix(store.DirectURL(sha), "/_blobs/"+sha) {
		t.Fatalf("DirectURL: %s", store.DirectURL(sha))
	}
	// Refs: tags and manifest records next to the blobs.
	if err := store.PutRef(ctx, "tags/org/name/main", []byte("sha256:"+sha)); err != nil {
		t.Fatalf("PutRef: %v", err)
	}
	if err := store.PutRef(ctx, "tags/org/name/v1", []byte("sha256:"+sha)); err != nil {
		t.Fatalf("PutRef: %v", err)
	}
	if v, err := store.GetRef(ctx, "tags/org/name/main"); err != nil || string(v) != "sha256:"+sha {
		t.Fatalf("GetRef: %v %q", err, v)
	}
	if _, err := store.GetRef(ctx, "tags/org/name/missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRef missing: %v", err)
	}
	keys, err := store.ListRefs(ctx, "tags/org/name/")
	if err != nil || strings.Join(keys, ",") != "tags/org/name/main,tags/org/name/v1" {
		t.Fatalf("ListRefs: %v %v", err, keys)
	}
	for _, k := range keys {
		if err := store.client.Bucket(store.bucket).Object(store.ref(k)).Delete(ctx); err != nil {
			t.Logf("cleanup ref %s: %v", k, err)
		}
	}
	// Clean up the test prefix.
	for _, name := range []string{store.object(sha)} {
		if err := store.client.Bucket(store.bucket).Object(name).Delete(ctx); err != nil {
			t.Logf("cleanup %s: %v", name, err)
		}
	}
}

func TestGCSSizeCache(t *testing.T) {
	g := &GCS{}
	sha := strings.Repeat("a", 64)
	g.sizeCache.Store(sha, int64(123456))
	size, err := g.Stat(t.Context(), sha)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if size != 123456 {
		t.Fatalf("Stat size = %d, want 123456", size)
	}
}

func TestGCSTransportConfiguration(t *testing.T) {
	tr := newGCSTransport()
	if tr.ForceAttemptHTTP2 {
		t.Errorf("ForceAttemptHTTP2 should be false")
	}
	if tr.TLSNextProto == nil || len(tr.TLSNextProto) != 0 {
		t.Errorf("TLSNextProto should be empty non-nil map to disable HTTP/2")
	}
	if tr.MaxIdleConnsPerHost < 256 {
		t.Errorf("MaxIdleConnsPerHost = %d, want >= 256", tr.MaxIdleConnsPerHost)
	}
}
