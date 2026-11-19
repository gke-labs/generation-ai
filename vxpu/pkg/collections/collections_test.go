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
	"fmt"
	"sync"
	"testing"
)

func TestConcurrentHashMap_PutAndGet(t *testing.T) {
	m := NewConcurrentHashMap[string, int]()

	if val, ok := m.Get("missing"); ok || val != 0 {
		t.Errorf("Get missing key: got (%v, %v), want (0, false)", val, ok)
	}

	m.Put("alpha", 42)
	if val, ok := m.Get("alpha"); !ok || val != 42 {
		t.Errorf("Get alpha: got (%v, %v), want (42, true)", val, ok)
	}

	m.Put("alpha", 99)
	if val, ok := m.Get("alpha"); !ok || val != 99 {
		t.Errorf("Get overwritten alpha: got (%v, %v), want (99, true)", val, ok)
	}
}

func TestConcurrentHashMap_Delete(t *testing.T) {
	m := NewConcurrentHashMap[string, string]()

	// Deleting a missing key must not fail.
	m.Delete("missing")

	m.Put("k1", "v1")
	m.Delete("k1")

	if val, ok := m.Get("k1"); ok || val != "" {
		t.Errorf("Get after Delete: got (%q, %v), want (\"\", false)", val, ok)
	}
}

func TestConcurrentHashMap_ZeroValue(t *testing.T) {
	var m ConcurrentHashMap[string, int]

	if val, ok := m.Get("k1"); ok || val != 0 {
		t.Errorf("Get on zero-value map: got (%v, %v), want (0, false)", val, ok)
	}

	m.Delete("k1")

	m.Put("k1", 100)
	if val, ok := m.Get("k1"); !ok || val != 100 {
		t.Errorf("Get after Put on zero-value map: got (%v, %v), want (100, true)", val, ok)
	}
}

func TestConcurrentHashMap_ConcurrentAccess_RaceDetector(t *testing.T) {
	m := NewConcurrentHashMap[string, int]()
	const (
		numGoroutines = 16
		numOps        = 200
	)

	var wg sync.WaitGroup
	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < numOps; i++ {
				keyShared := fmt.Sprintf("shared-%d", i%10)
				keyUnique := fmt.Sprintf("goroutine-%d-key-%d", gid, i)

				m.Put(keyShared, gid*1000+i)
				m.Put(keyUnique, i)

				val, ok := m.Get(keyShared)
				if ok && val < 0 {
					t.Errorf("unexpected negative value for shared key: %d", val)
				}

				if val, ok := m.Get(keyUnique); !ok || val != i {
					t.Errorf("Get %s: got (%d, %t), want (%d, true)", keyUnique, val, ok, i)
				}

				if i%3 == 0 {
					m.Delete(keyShared)
				}
			}
		}(g)
	}
	wg.Wait()
}
