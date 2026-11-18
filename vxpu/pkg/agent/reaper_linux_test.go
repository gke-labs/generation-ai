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

//go:build linux

package agent

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestProcess1ReapsOrphanChildren(t *testing.T) {
	if err := unix.Prctl(unix.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0); err != nil {
		t.Skipf("PR_SET_CHILD_SUBREAPER unavailable: %v", err)
	}

	ctx := t.Context()
	s := testServer(t)
	s.forceReaper = true
	s.StartReaper(ctx)
	defer s.Stop()

	cmd := exec.Command(os.Args[0], "spawn-orphan")
	cmd.Env = append(os.Environ(), "TEST_STANDIN=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("spawn-orphan failed: %v (%s)", err, out)
	}
	var orphanPid int
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "orphan pid: ") {
			orphanPid, _ = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "orphan pid: ")))
			break
		}
	}
	if orphanPid == 0 {
		t.Fatalf("could not parse orphan pid from: %s", out)
	}

	gone := false
	for range 50 {
		s.ReapOrphans()
		if _, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", orphanPid)); err != nil {
			gone = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !gone {
		t.Errorf("orphan pid %d still present in /proc after reaping", orphanPid)
	}
}

func TestReaperDoesNotStealHelperExitStatus(t *testing.T) {
	if err := unix.Prctl(unix.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0); err != nil {
		t.Skipf("PR_SET_CHILD_SUBREAPER unavailable: %v", err)
	}

	ctx := t.Context()
	s := testServer(t)
	s.forceReaper = true
	s.StartReaper(ctx)
	defer s.Stop()

	// Spawn multiple orphan grandchildren in background concurrently.
	for range 10 {
		cmd := exec.Command("sh", "-c", fmt.Sprintf("TEST_STANDIN=1 %s orphan-child &", os.Args[0]))
		if err := cmd.Run(); err != nil {
			t.Fatalf("spawn orphan failed: %v", err)
		}
	}

	// Concurrently run internal helper commands (s.run).
	// Exactly one place collects exit statuses, so none fails with "no child processes" / ECHILD.
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if err := s.run(ctx, "true"); err != nil {
				t.Errorf("run iteration %d failed: %v", idx, err)
			}
		}(i)
	}
	wg.Wait()
}
