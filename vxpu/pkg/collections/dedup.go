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
)

// errLeaderPanicked is recorded on in-flight attempts when the leader function panics.
var errLeaderPanicked = errors.New("leader panicked")

// attempt tracks an active in-flight operation for a key.
type attempt[V any] struct {
	// done signals when the leader finishes.
	done chan struct{}

	// val is the value produced by the leader.
	val V

	// err is the error returned by the leader.
	err error

	// verdict indicates that val and err represent an answer that another
	// attempt would repeat, which waiters share rather than retry.
	verdict bool
}

// Deduplicator runs one call per key at a time, sharing results across concurrent callers.
//
// A verdict is an answer that another attempt would repeat, as against an attempt
// that did not complete.
//
// When multiple callers request the same key concurrently:
//   - All callers share the leader's attempt.
//   - A waiter stops waiting and returns immediately when its own context ends.
//   - When the leader fails or cancels without a verdict, a waiter makes its own
//     attempt rather than returning another caller's failure.
//   - When the leader finishes with a verdict, waiters share that verdict instead
//     of repeating the work.
//
// A plain Go map guarded by a mutex is used rather than ConcurrentHashMap because
// an in-flight attempt's entry exists in the map only for the duration of the leader's
// execution. Checking for an in-flight attempt and inserting a new one must be atomic,
// and removing the entry upon completion while closing the done channel must also happen
// atomically under the same lock so waiters waking from a failed attempt never observe
// a stale entry.
type Deduplicator[K comparable, V any] struct {
	// mu synchronizes access to inFlight.
	mu sync.Mutex

	// inFlight tracks active attempts by key.
	inFlight map[K]*attempt[V]
}

// NewDeduplicator returns a new, empty Deduplicator.
func NewDeduplicator[K comparable, V any]() *Deduplicator[K, V] {
	return &Deduplicator[K, V]{
		inFlight: make(map[K]*attempt[V]),
	}
}

// Do executes fn for key, deduplicating concurrent calls with the same key.
//
// The leader calls fn to perform the work. The function returns the produced value,
// a boolean indicating whether the outcome is a verdict (an answer that another attempt
// would repeat, as against an attempt that did not complete), and an error.
func (d *Deduplicator[K, V]) Do(ctx context.Context, key K, fn func() (V, bool, error)) (V, error) {
	for {
		d.mu.Lock()
		if att, ok := d.inFlight[key]; ok {
			d.mu.Unlock()
			select {
			case <-ctx.Done():
				var zero V
				return zero, ctx.Err()
			case <-att.done:
				if att.err == nil || att.verdict {
					return att.val, att.err
				}
				if ctx.Err() != nil {
					var zero V
					return zero, ctx.Err()
				}
				// The in-flight attempt failed or was cancelled. A waiter never
				// returns another caller's failure; loop around to make its own attempt.
				continue
			}
		}

		att := &attempt[V]{done: make(chan struct{})}
		d.inFlight[key] = att
		d.mu.Unlock()

		var (
			val      V
			verdict  bool
			err      error
			finished bool
		)

		func() {
			defer func() {
				d.mu.Lock()
				if !finished {
					att.err = errLeaderPanicked
					att.verdict = false
				} else {
					att.val = val
					att.err = err
					att.verdict = verdict
				}
				delete(d.inFlight, key)
				close(att.done)
				d.mu.Unlock()
			}()

			val, verdict, err = fn()
			finished = true
		}()

		return val, err
	}
}
