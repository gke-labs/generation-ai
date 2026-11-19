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

	s := testServer(t)
	s.forceReaper = true
	s.StartReaper()
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
	for range 100 {
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
	s.StartReaper()
	defer s.Stop()

	// Spawn background orphans continuously while helper commands run.
	stopOrphans := make(chan struct{})
	var orphanWg sync.WaitGroup
	for range 3 {
		orphanWg.Add(1)
		go func() {
			defer orphanWg.Done()
			for {
				select {
				case <-stopOrphans:
					return
				default:
					cmd := exec.Command("sh", "-c", fmt.Sprintf("TEST_STANDIN=1 %s orphan-child &", os.Args[0]))
					_ = cmd.Run()
					time.Sleep(10 * time.Millisecond)
				}
			}
		}()
	}

	// Concurrently run internal helper commands (s.run) that both succeed and fail.
	// Exactly one place collects exit statuses, so none fails with "no child processes" / ECHILD,
	// and each keeps its exit status and output.
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if idx%2 == 0 {
				expected := fmt.Sprintf("helper-ok-%d", idx)
				out, err := s.run(ctx, "echo", expected)
				if err != nil {
					t.Errorf("helper %d failed: %v", idx, err)
					return
				}
				if !strings.Contains(string(out), expected) {
					t.Errorf("helper %d output = %q, want %q", idx, string(out), expected)
				}
			} else {
				expected := fmt.Sprintf("helper-fail-%d", idx)
				out, err := s.run(ctx, "sh", "-c", fmt.Sprintf("echo %s; exit 42", expected))
				if err == nil {
					t.Errorf("helper %d expected error, got nil", idx)
					return
				}
				if !strings.Contains(err.Error(), "42") {
					t.Errorf("helper %d error %q does not contain 42", idx, err)
				}
				if !strings.Contains(string(out), expected) && !strings.Contains(err.Error(), expected) {
					t.Errorf("helper %d output %q does not contain %q", idx, string(out), expected)
				}
			}
		}(i)
	}
	wg.Wait()

	close(stopOrphans)
	orphanWg.Wait()
}
