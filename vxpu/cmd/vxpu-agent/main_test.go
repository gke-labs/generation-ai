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

package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRun_RefusesWithoutAuth(t *testing.T) {
	ctx := t.Context()
	mountRoot := t.TempDir()
	workRoot := t.TempDir()

	err := run(ctx, 0, mountRoot, workRoot, "", false)
	if err == nil {
		t.Fatal("want error when running without auth token or allow-anonymous, got nil")
	}
	if !strings.Contains(err.Error(), "requires an authentication token") {
		t.Errorf("error %q does not mention authentication token requirement", err)
	}
}

func TestRun_RefusesConflictingAuth(t *testing.T) {
	ctx := t.Context()
	mountRoot := t.TempDir()
	workRoot := t.TempDir()

	err := run(ctx, 0, mountRoot, workRoot, "some-token", true)
	if err == nil {
		t.Fatal("want error when running with both auth token and allow-anonymous, got nil")
	}
	if !strings.Contains(err.Error(), "cannot specify both") {
		t.Errorf("error %q does not mention cannot specify both", err)
	}
}

func TestRun_StartsAndStopsGracefully(t *testing.T) {
	mountRoot := t.TempDir()
	workRoot := t.TempDir()

	ctx, cancel := context.WithCancel(t.Context())
	errCh := make(chan error, 1)
	go func() {
		errCh <- run(ctx, 0, mountRoot, workRoot, "test-token", false)
	}()

	// Allow server to start up listening on an ephemeral port.
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("run returned error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("run timed out shutting down")
	}
}
