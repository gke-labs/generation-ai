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

// Package collections provides synchronized collection types for concurrent use.
package collections

import "sync"

// ConcurrentHashMap is a generic hash map that synchronizes its own operations.
// Callers can read and write keys concurrently without holding an external lock.
//
// A plain Go map guarded by a caller's own mutex remains the right choice when
// several fields must change together atomically.
type ConcurrentHashMap[K comparable, V any] struct {
	// mu guards entries against concurrent data races.
	mu sync.RWMutex

	// entries stores the key-value mappings.
	entries map[K]V
}

// NewConcurrentHashMap returns a new, empty ConcurrentHashMap.
func NewConcurrentHashMap[K comparable, V any]() *ConcurrentHashMap[K, V] {
	return &ConcurrentHashMap[K, V]{
		entries: make(map[K]V),
	}
}

// Get returns the value associated with key and reports whether the key was present.
func (m *ConcurrentHashMap[K, V]) Get(key K) (V, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.entries == nil {
		var zero V
		return zero, false
	}
	val, ok := m.entries[key]
	return val, ok
}

// Put associates key with val in the map.
func (m *ConcurrentHashMap[K, V]) Put(key K, val V) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.entries == nil {
		m.entries = make(map[K]V)
	}
	m.entries[key] = val
}

// Delete removes key and its associated value from the map.
func (m *ConcurrentHashMap[K, V]) Delete(key K) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.entries == nil {
		return
	}
	delete(m.entries, key)
}
