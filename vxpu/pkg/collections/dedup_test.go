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

package collections

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDeduplicator_SharedAttempt(t *testing.T) {
	d := NewDeduplicator[string, int]()
	var runs atomic.Int32
	gate := make(chan struct{})

	var wg sync.WaitGroup
	vals := make([]int, 4)
	errs := make([]error, 4)

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			vals[idx], errs[idx] = d.Do(t.Context(), "key1", func() (int, bool, error) {
				runs.Add(1)
				<-gate
				return 42, true, nil
			})
		}(i)
	}

	// Give goroutines time to join
	time.Sleep(20 * time.Millisecond)
	close(gate)
	wg.Wait()

	if runs.Load() != 1 {
		t.Errorf("expected 1 run across shared attempts, got %d", runs.Load())
	}
	for i := 0; i < 4; i++ {
		if errs[i] != nil {
			t.Errorf("caller %d got error: %v", i, errs[i])
		}
		if vals[i] != 42 {
			t.Errorf("caller %d got %d, want 42", i, vals[i])
		}
	}
}

func TestDeduplicator_WaiterCancelled(t *testing.T) {
	d := NewDeduplicator[string, string]()
	leaderRunning := make(chan struct{})
	gateLeader := make(chan struct{})

	var leaderVal string
	var leaderErr error
	var waiterErr error

	var wg sync.WaitGroup
	wg.Add(2)

	// Leader
	go func() {
		defer wg.Done()
		leaderVal, leaderErr = d.Do(t.Context(), "key1", func() (string, bool, error) {
			close(leaderRunning)
			<-gateLeader
			return "leader-ok", true, nil
		})
	}()

	<-leaderRunning

	waiterCtx, cancelWaiter := context.WithCancel(t.Context())
	go func() {
		defer wg.Done()
		_, waiterErr = d.Do(waiterCtx, "key1", func() (string, bool, error) {
			return "waiter-not-run", false, nil
		})
	}()

	time.Sleep(20 * time.Millisecond)
	cancelWaiter()

	time.Sleep(20 * time.Millisecond)
	close(gateLeader)
	wg.Wait()

	if leaderErr != nil {
		t.Errorf("leader failed: %v", leaderErr)
	}
	if leaderVal != "leader-ok" {
		t.Errorf("leader value = %q, want leader-ok", leaderVal)
	}
	if !errors.Is(waiterErr, context.Canceled) {
		t.Errorf("waiter error = %v, want context.Canceled", waiterErr)
	}
}

func TestDeduplicator_LeaderCancelled(t *testing.T) {
	d := NewDeduplicator[string, string]()
	leaderRunning := make(chan struct{})
	gateLeader := make(chan struct{})

	ctxLeader, cancelLeader := context.WithCancel(t.Context())
	ctxWaiter := t.Context()

	var leaderErr error
	var waiterVal string
	var waiterErr error

	var wg sync.WaitGroup
	wg.Add(2)

	// Leader
	go func() {
		defer wg.Done()
		_, leaderErr = d.Do(ctxLeader, "key1", func() (string, bool, error) {
			close(leaderRunning)
			<-gateLeader
			return "", false, ctxLeader.Err()
		})
	}()

	<-leaderRunning

	// Waiter
	go func() {
		defer wg.Done()
		waiterVal, waiterErr = d.Do(ctxWaiter, "key1", func() (string, bool, error) {
			return "waiter-success", true, nil
		})
	}()

	time.Sleep(20 * time.Millisecond)
	cancelLeader()
	close(gateLeader)

	wg.Wait()

	if !errors.Is(leaderErr, context.Canceled) {
		t.Errorf("leader error = %v, want context.Canceled", leaderErr)
	}
	if waiterErr != nil {
		t.Fatalf("waiter error = %v, want nil", waiterErr)
	}
	if waiterVal != "waiter-success" {
		t.Errorf("waiter val = %q, want waiter-success", waiterVal)
	}
}

func TestDeduplicator_LeaderFails(t *testing.T) {
	d := NewDeduplicator[string, string]()
	leaderRunning := make(chan struct{})
	gateLeader := make(chan struct{})
	var attempts atomic.Int32

	var leaderErr error
	var waiterVal string
	var waiterErr error

	var wg sync.WaitGroup
	wg.Add(2)

	// Leader
	go func() {
		defer wg.Done()
		_, leaderErr = d.Do(t.Context(), "key1", func() (string, bool, error) {
			attempts.Add(1)
			close(leaderRunning)
			<-gateLeader
			return "", false, errors.New("network glitch")
		})
	}()

	<-leaderRunning

	// Waiter
	go func() {
		defer wg.Done()
		waiterVal, waiterErr = d.Do(t.Context(), "key1", func() (string, bool, error) {
			attempts.Add(1)
			return "waiter-recovered", true, nil
		})
	}()

	time.Sleep(20 * time.Millisecond)
	close(gateLeader)
	wg.Wait()

	if leaderErr == nil || leaderErr.Error() != "network glitch" {
		t.Errorf("leader should get network glitch error, got: %v", leaderErr)
	}
	if waiterErr != nil {
		t.Fatalf("waiter should succeed on retry, got: %v", waiterErr)
	}
	if waiterVal != "waiter-recovered" {
		t.Errorf("waiter val = %q, want waiter-recovered", waiterVal)
	}
	if attempts.Load() != 2 {
		t.Errorf("expected 2 attempts, got %d", attempts.Load())
	}
}

func TestDeduplicator_LeaderPanics(t *testing.T) {
	d := NewDeduplicator[string, string]()
	leaderRunning := make(chan struct{})
	gateLeader := make(chan struct{})

	var leaderPanicked bool
	var waiterVal string
	var waiterErr error

	var wg sync.WaitGroup
	wg.Add(2)

	// Leader panics
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				leaderPanicked = true
			}
		}()
		_, _ = d.Do(t.Context(), "key1", func() (string, bool, error) {
			close(leaderRunning)
			<-gateLeader
			panic("leader exploded")
		})
	}()

	<-leaderRunning

	// Waiter should recover, retry, and succeed
	go func() {
		defer wg.Done()
		waiterVal, waiterErr = d.Do(t.Context(), "key1", func() (string, bool, error) {
			return "waiter-recovered", true, nil
		})
	}()

	time.Sleep(20 * time.Millisecond)
	close(gateLeader)
	wg.Wait()

	if !leaderPanicked {
		t.Errorf("expected leader to panic")
	}
	if waiterErr != nil {
		t.Fatalf("waiter should succeed after leader panic, got: %v", waiterErr)
	}
	if waiterVal != "waiter-recovered" {
		t.Errorf("waiter val = %q, want waiter-recovered", waiterVal)
	}

	// Verify subsequent caller is not wedged
	val, err := d.Do(t.Context(), "key1", func() (string, bool, error) {
		return "subsequent-success", true, nil
	})
	if err != nil {
		t.Fatalf("subsequent call failed: %v", err)
	}
	if val != "subsequent-success" {
		t.Errorf("subsequent call val = %q, want subsequent-success", val)
	}
}

func TestDeduplicator_VerdictCase(t *testing.T) {
	d := NewDeduplicator[string, string]()
	verdictErr := errors.New("definitive token mismatch")

	var executions atomic.Int32
	gate := make(chan struct{})

	var wg sync.WaitGroup
	errs := make([]error, 4)

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = d.Do(t.Context(), "key1", func() (string, bool, error) {
				executions.Add(1)
				<-gate
				return "", true, verdictErr
			})
		}(i)
	}

	time.Sleep(20 * time.Millisecond)
	close(gate)
	wg.Wait()

	if executions.Load() != 1 {
		t.Errorf("verdict should be shared without retries; executed %d times, want 1", executions.Load())
	}
	for i := 0; i < 4; i++ {
		if !errors.Is(errs[i], verdictErr) {
			t.Errorf("caller %d got error %v, want %v", i, errs[i], verdictErr)
		}
	}
}
