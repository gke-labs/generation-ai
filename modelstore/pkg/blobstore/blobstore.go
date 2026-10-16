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

// Package blobstore stores model files by their sha256.
//
// A blob is immutable and named by the sha256 of its bytes, so a store
// never needs to update in place: Put verifies the digest while the
// bytes stream through and only then publishes the blob under its
// name, and readers never observe a partial blob. Two backends: a
// local directory (kind, tests) and a Google Cloud Storage bucket (the
// cluster's weight store; readers with their own bucket access can
// range-read the object directly, see Store.DirectURL).
package blobstore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"strings"
	"time"
)

// ErrNotFound is returned by Stat and Open for a blob the store does not hold.
var ErrNotFound = errors.New("blob not found")

// Store is a content-addressed blob store.
type Store interface {
	// Stat reports the size of a blob, or ErrNotFound.
	Stat(ctx context.Context, sha string) (int64, error)
	// Put stores the bytes of r as the blob sha, verifying the digest;
	// with sha == "" the blob is named by its content. Returns the
	// blob's sha and the bytes written. A mismatch leaves no blob.
	Put(ctx context.Context, sha string, r io.Reader) (string, int64, error)
	// Open reads length bytes of the blob starting at offset; length < 0
	// reads to the end.
	Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error)
	// DirectURL returns a pre-signed URL to read the blob directly without
	// proxying through modelstore, or "" if not supported / no signer configured.
	DirectURL(ctx context.Context, sha string) (string, error)
	// String describes the backend for logs.
	String() string

	// Refs are small named values next to the blobs: registry tags and
	// manifest records. Keys are slash-separated paths such as
	// "tags/<name>/<tag>"; values are a few bytes to a few KB. PutRef
	// overwrites; GetRef returns ErrNotFound for a missing key; ListRefs
	// returns the keys under a prefix, sorted.
	PutRef(ctx context.Context, key string, value []byte) error
	GetRef(ctx context.Context, key string) ([]byte, error)
	ListRefs(ctx context.Context, prefix string) ([]string, error)
}

// Options configures a Store.
type Options struct {
	SignerServiceAccount string
	URLSigner            URLSigner
	Validity             time.Duration
	Now                  func() time.Time
}

// Option modifies Options.
type Option func(*Options)

// WithSignerServiceAccount sets the Google service account to impersonate
// for signing V4 URLs.
func WithSignerServiceAccount(sa string) Option {
	return func(o *Options) {
		o.SignerServiceAccount = sa
	}
}

// WithURLSigner sets a custom URLSigner (e.g. for testing).
func WithURLSigner(s URLSigner) Option {
	return func(o *Options) {
		o.URLSigner = s
	}
}

// WithValidity sets the validity duration for signed URLs (defaults to 7 days).
func WithValidity(d time.Duration) Option {
	return func(o *Options) {
		o.Validity = d
	}
}

// WithNow sets the time source for caching and expiry checks.
func WithNow(now func() time.Time) Option {
	return func(o *Options) {
		o.Now = now
	}
}

// ValidRefKey accepts the keys Refs are stored under: path segments of
// [A-Za-z0-9._-] plus ":" (digests), no empty or dot-only segments.
func ValidRefKey(key string) bool {
	if key == "" {
		return false
	}
	for _, seg := range strings.Split(key, "/") {
		if seg == "" || strings.Trim(seg, ".") == "" {
			return false
		}
		for _, c := range seg {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '.' || c == '_' || c == '-' || c == ':') {
				return false
			}
		}
	}
	return true
}

// Open opens a store from its location: a gs://bucket[/prefix] URL or
// an absolute local directory path. Anything else is an error at
// startup rather than a directory created on the pod's root filesystem
// (a misspelt scheme, or an unresolved $(VAR) from a manifest).
func Open(ctx context.Context, location string, opts ...Option) (Store, error) {
	switch {
	case strings.HasPrefix(location, "gs://"):
		return NewGCS(ctx, location, opts...)
	case strings.Contains(location, "://") || strings.Contains(location, "$("):
		return nil, fmt.Errorf("blob store location %q: want gs://bucket[/prefix] or an absolute directory", location)
	case !strings.HasPrefix(location, "/"):
		return nil, fmt.Errorf("blob store directory %q must be an absolute path", location)
	}
	return NewLocal(location)
}

// ValidSHA reports whether s is a lowercase hex sha256 digest.
func ValidSHA(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// digestWriter is w with a sha256 of everything written through it.
type digestWriter struct {
	io.Writer
	h hash.Hash
	n int64
}

func newDigestWriter(w io.Writer) *digestWriter {
	h := sha256.New()
	return &digestWriter{Writer: io.MultiWriter(w, h), h: h}
}

// copyFrom streams r through d, counting the bytes.
func (d *digestWriter) copyFrom(r io.Reader) error {
	n, err := io.Copy(d, r)
	d.n += n
	return err
}

// check verifies the digest against sha; an empty sha accepts any and
// returns the computed one.
func (d *digestWriter) check(sha string) (string, error) {
	got := hex.EncodeToString(d.h.Sum(nil))
	if sha != "" && got != sha {
		return got, fmt.Errorf("sha256 mismatch: expected %s, got %s", sha, got)
	}
	return got, nil
}
