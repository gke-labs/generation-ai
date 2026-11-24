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

// Package fileutil provides temporary-file helpers for atomic file placement.
package fileutil

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/klog/v2"
)

// TempFile manages an unfinished file being written before atomic placement.
type TempFile struct {
	// file is the open temporary file handle until Rename or Cleanup closes it.
	file *os.File
	// path is the filesystem path of the temporary file.
	path string
	// done reports whether the file has been renamed to its final destination.
	done bool
}

// CreateTemp creates a temporary file in dir with the given filename pattern.
func CreateTemp(dir, pattern string) (*TempFile, error) {
	f, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return nil, err
	}
	return &TempFile{file: f, path: f.Name()}, nil
}

// Write writes bytes to the underlying temporary file.
func (t *TempFile) Write(p []byte) (int, error) {
	return t.file.Write(p)
}

// Chmod sets the file mode of the underlying temporary file.
func (t *TempFile) Chmod(mode os.FileMode) error {
	return t.file.Chmod(mode)
}

// Rename closes the temporary file and atomically renames it to dest.
func (t *TempFile) Rename(dest string) error {
	if t.file != nil {
		if err := t.file.Close(); err != nil {
			return err
		}
		t.file = nil
	}
	if err := os.Rename(t.path, dest); err != nil {
		return err
	}
	t.done = true
	return nil
}

// Cleanup closes and removes the temporary file if it was not renamed.
func (t *TempFile) Cleanup() {
	if t == nil || t.done {
		return
	}
	if t.file != nil {
		if err := t.file.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			klog.V(2).Info("closing temp file during cleanup", "path", t.path, "error", err)
		}
		t.file = nil
	}
	if err := os.Remove(t.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		klog.V(2).Info("removing temp file during cleanup", "path", t.path, "error", err)
	}
}

// WriteAtomic writes data to path via a temporary file in the same directory,
// sets perm, and atomically renames the file into place.
func WriteAtomic(path string, data []byte, perm os.FileMode) error {
	tmp, err := CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("creating temp file for %s: %w", path, err)
	}
	defer tmp.Cleanup()

	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("writing temp file for %s: %w", path, err)
	}
	if err := tmp.Chmod(perm); err != nil {
		return fmt.Errorf("setting permissions on %s: %w", path, err)
	}
	if err := tmp.Rename(path); err != nil {
		return fmt.Errorf("renaming temp file to %s: %w", path, err)
	}
	return nil
}
