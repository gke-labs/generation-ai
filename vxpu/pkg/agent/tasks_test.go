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

package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunTasks_Empty(t *testing.T) {
	if err := RunTasks(t.Context(), nil, 4); err != nil {
		t.Fatalf("expected nil error for nil tasks, got %v", err)
	}
	if err := RunTasks(t.Context(), []Task{}, 4); err != nil {
		t.Fatalf("expected nil error for empty tasks, got %v", err)
	}
}

func TestRunTasks_Success(t *testing.T) {
	const n = 10
	var completed atomic.Int32
	tasks := make([]Task, n)
	for i := 0; i < n; i++ {
		tasks[i] = func(ctx context.Context) error {
			completed.Add(1)
			return nil
		}
	}

	if err := RunTasks(t.Context(), tasks, 3); err != nil {
		t.Fatalf("RunTasks failed: %v", err)
	}
	if got := completed.Load(); got != n {
		t.Errorf("completed tasks = %d, want %d", got, n)
	}
}

func TestRunTasks_BoundedParallelism(t *testing.T) {
	const (
		n           = 12
		parallelism = 4
	)
	var inFlight atomic.Int32
	var maxInFlight atomic.Int32
	tasks := make([]Task, n)
	for i := 0; i < n; i++ {
		tasks[i] = func(ctx context.Context) error {
			cur := inFlight.Add(1)
			for {
				old := maxInFlight.Load()
				if cur <= old || maxInFlight.CompareAndSwap(old, cur) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			inFlight.Add(-1)
			return nil
		}
	}

	if err := RunTasks(t.Context(), tasks, parallelism); err != nil {
		t.Fatalf("RunTasks: %v", err)
	}
	got := maxInFlight.Load()
	if got <= 1 {
		t.Errorf("expected parallel execution (> 1), got max %d", got)
	}
	if got > parallelism {
		t.Errorf("concurrency exceeded parallelism limit %d, got %d", parallelism, got)
	}
}

func TestRunTasks_StopOnFirstErrorByDefault(t *testing.T) {
	errBoom := errors.New("boom")
	var started atomic.Int32
	var cancelledTasks atomic.Int32

	tasks := []Task{
		func(ctx context.Context) error {
			started.Add(1)
			return errBoom
		},
		func(ctx context.Context) error {
			started.Add(1)
			select {
			case <-ctx.Done():
				cancelledTasks.Add(1)
				return ctx.Err()
			case <-time.After(200 * time.Millisecond):
				return nil
			}
		},
		func(ctx context.Context) error {
			started.Add(1)
			select {
			case <-ctx.Done():
				cancelledTasks.Add(1)
				return ctx.Err()
			case <-time.After(200 * time.Millisecond):
				return nil
			}
		},
	}

	// Default behavior stops early on first error without passing any options.
	err := RunTasks(t.Context(), tasks, 3)
	if !errors.Is(err, errBoom) {
		t.Fatalf("expected errBoom, got %v", err)
	}
}

func TestRunTasks_ContinueOnError(t *testing.T) {
	err1 := errors.New("err1")
	err2 := errors.New("err2")
	var completed atomic.Int32

	tasks := []Task{
		func(ctx context.Context) error {
			completed.Add(1)
			return err1
		},
		func(ctx context.Context) error {
			completed.Add(1)
			return nil
		},
		func(ctx context.Context) error {
			completed.Add(1)
			return err2
		},
	}

	err := RunTasks(t.Context(), tasks, 2, ContinueOnError())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, err1) || !errors.Is(err, err2) {
		t.Errorf("expected joined errors containing err1 and err2, got %v", err)
	}
	if got := completed.Load(); got != 3 {
		t.Errorf("completed tasks = %d, want 3", got)
	}
}

func TestRunTasks_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	tasks := []Task{
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunTasks(ctx, tasks, 2)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestRunTasks_ParallelismClamping(t *testing.T) {
	var count atomic.Int32
	tasks := []Task{
		func(ctx context.Context) error {
			count.Add(1)
			return nil
		},
	}

	// parallelism <= 0 should default to 1 and succeed
	if err := RunTasks(t.Context(), tasks, 0); err != nil {
		t.Fatalf("RunTasks with 0 parallelism failed: %v", err)
	}
	if err := RunTasks(t.Context(), tasks, -5); err != nil {
		t.Fatalf("RunTasks with negative parallelism failed: %v", err)
	}
	if got := count.Load(); got != 2 {
		t.Errorf("executed tasks count = %d, want 2", got)
	}
}
