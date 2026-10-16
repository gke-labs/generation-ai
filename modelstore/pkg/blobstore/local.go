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
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/klog/v2"
)

// Local keeps blobs as files under dir/blobs/<sha>.
type Local struct {
	dir string
}

// NewLocal opens (creating if needed) a directory-backed store.
func NewLocal(dir string) (*Local, error) {
	for _, sub := range []string{"blobs", filepath.Join("refs", ".tmp")} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			return nil, fmt.Errorf("creating %s directory: %w", sub, err)
		}
	}
	// A process killed mid-Put leaves its temp file; nothing else would
	// ever remove it.
	for _, pattern := range []string{filepath.Join(dir, "blobs", "upload-*"), filepath.Join(dir, "refs", ".tmp", "*")} {
		stale, err := filepath.Glob(pattern)
		if err != nil {
			return nil, fmt.Errorf("listing stale temp files: %w", err)
		}
		for _, p := range stale {
			if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
				return nil, fmt.Errorf("removing stale temp file %s: %w", p, err)
			}
		}
	}
	return &Local{dir: dir}, nil
}

func (l *Local) path(sha string) string { return filepath.Join(l.dir, "blobs", sha) }

func (l *Local) Stat(ctx context.Context, sha string) (int64, error) {
	info, err := os.Stat(l.path(sha))
	if errors.Is(err, os.ErrNotExist) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

func (l *Local) Put(ctx context.Context, sha string, r io.Reader) (string, int64, error) {
	tmp, err := os.CreateTemp(filepath.Join(l.dir, "blobs"), "upload-*")
	if err != nil {
		return "", 0, fmt.Errorf("creating temp file: %w", err)
	}
	tmpPath := tmp.Name()
	published := false
	log := klog.FromContext(ctx)
	defer func() {
		if err := tmp.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			log.Error(err, "closing temp file", "path", tmpPath)
		}
		if !published {
			if err := os.Remove(tmpPath); err != nil && !errors.Is(err, os.ErrNotExist) {
				log.Error(err, "removing temp file", "path", tmpPath)
			}
		}
	}()
	dw := newDigestWriter(tmp)
	if err := dw.copyFrom(r); err != nil {
		return "", dw.n, fmt.Errorf("writing blob: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", dw.n, fmt.Errorf("closing temp file: %w", err)
	}
	got, err := dw.check(sha)
	if err != nil {
		return got, dw.n, err
	}
	if err := os.Rename(tmpPath, l.path(got)); err != nil {
		return got, dw.n, fmt.Errorf("publishing blob: %w", err)
	}
	published = true
	return got, dw.n, nil
}

func (l *Local) Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error) {
	f, err := os.Open(l.path(sha))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if offset > 0 {
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			if cerr := f.Close(); cerr != nil {
				klog.FromContext(ctx).Error(cerr, "closing blob after failed seek", "sha", sha)
			}
			return nil, err
		}
	}
	if length < 0 {
		return f, nil
	}
	return &limitedFile{Reader: io.LimitReader(f, length), f: f}, nil
}

type limitedFile struct {
	io.Reader
	f *os.File
}

func (l *limitedFile) Close() error { return l.f.Close() }

func (l *Local) DirectURL(ctx context.Context, sha string) (string, error) { return "", nil }

// refPath is flat, like the GCS backend's object names: one file per
// key with its slashes encoded ('%' is outside the key grammar). Nested
// directories would make "tags/org/name" (a tag of org) and
// "tags/org/name/<tag>" (tags of org/name) a file and a directory of
// the same path.
func (l *Local) refPath(key string) string {
	return filepath.Join(l.dir, "refs", strings.ReplaceAll(key, "/", "%2F"))
}

func (l *Local) PutRef(ctx context.Context, key string, value []byte) error {
	if !ValidRefKey(key) {
		return fmt.Errorf("invalid ref key %q", key)
	}
	// Staged in a directory of its own, so no tag name can collide with
	// a temporary and ListRefs never sees one.
	tmp, err := os.CreateTemp(filepath.Join(l.dir, "refs", ".tmp"), "ref-*")
	if err != nil {
		return err
	}
	discard := func(err error) error {
		// the temp is worthless after a failed write: report the first error, log the rest
		log := klog.FromContext(ctx)
		if cerr := tmp.Close(); cerr != nil && !errors.Is(cerr, os.ErrClosed) {
			log.Error(cerr, "closing temp ref", "path", tmp.Name())
		}
		if rerr := os.Remove(tmp.Name()); rerr != nil && !errors.Is(rerr, os.ErrNotExist) {
			log.Error(rerr, "removing temp ref", "path", tmp.Name())
		}
		return err
	}
	if _, err := tmp.Write(value); err != nil {
		return discard(err)
	}
	if err := tmp.Close(); err != nil {
		return discard(err)
	}
	return os.Rename(tmp.Name(), l.refPath(key))
}

func (l *Local) GetRef(ctx context.Context, key string) ([]byte, error) {
	if !ValidRefKey(key) {
		return nil, ErrNotFound
	}
	b, err := os.ReadFile(l.refPath(key))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	return b, err
}

func (l *Local) ListRefs(ctx context.Context, prefix string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(l.dir, "refs"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var keys []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		key := strings.ReplaceAll(e.Name(), "%2F", "/")
		if strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	return keys, nil
}

func (l *Local) String() string { return "local:" + l.dir }
