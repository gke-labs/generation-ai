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
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

func init() {
	if os.Getenv("TEST_STANDIN") == "1" {
		if len(os.Args) < 2 {
			os.Exit(1)
		}
		switch os.Args[1] {
		case "exit-42":
			fmt.Println("standin starting")
			fmt.Println("standin exiting with 42")
			os.Exit(42)
		case "loop":
			fmt.Println("standin started")
			fmt.Println("standin running")
			for {
				time.Sleep(100 * time.Millisecond)
			}
		case "parent-with-child":
			fmt.Println("parent starting")
			child := exec.Command("sleep", "300")
			if err := child.Start(); err != nil {
				fmt.Printf("child start failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("parent started child %d\n", child.Process.Pid)
			for {
				time.Sleep(100 * time.Millisecond)
			}
		case "large-output":
			chunk := strings.Repeat("A", 1024*1024)
			for range 64 {
				os.Stdout.WriteString(chunk)
			}
			os.Stdout.WriteString("\rprogress 100%\n")
			for {
				time.Sleep(100 * time.Millisecond)
			}
		case "print-card":
			fmt.Printf("card: %s\n", os.Getenv("CUDA_VISIBLE_DEVICES"))
			os.Exit(0)
		case "sigterm-trap":
			fmt.Println("sigterm-trap started")
			sigCh := make(chan os.Signal, 1)
			signal.Notify(sigCh, syscall.SIGTERM)
			<-sigCh
			fmt.Println("sigterm-trap caught sigterm")
			os.Exit(0)
		case "spawn-orphan":
			grandchild := exec.Command(os.Args[0], "orphan-child")
			grandchild.Env = append(os.Environ(), "TEST_STANDIN=1")
			if err := grandchild.Start(); err != nil {
				os.Exit(1)
			}
			fmt.Printf("orphan pid: %d\n", grandchild.Process.Pid)
			os.Exit(0)
		case "orphan-child":
			time.Sleep(50 * time.Millisecond)
			os.Exit(0)
		case "standin-agent":
			ctx, cancel := signal.NotifyContext(context.TODO(), os.Interrupt, syscall.SIGTERM)
			defer cancel()
			if err := runStandinServer(ctx, os.Args[2], os.Args[0]); err != nil {
				os.Exit(1)
			}
			os.Exit(0)
		}
		os.Exit(0)
	}
}

// fakeDisk writes a weights disk's contents into a directory, as the
// factory would. The fake mount in server links to that directory.
func fakeDisk(t *testing.T, contents map[string]string) (string, map[string]int64) {
	t.Helper()
	disk := t.TempDir()
	if err := os.MkdirAll(filepath.Join(disk, weightdisk.BlobDir), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]int64{}
	for _, c := range contents {
		sum := sha256.Sum256([]byte(c))
		sha := hex.EncodeToString(sum[:])
		if err := os.WriteFile(filepath.Join(disk, weightdisk.BlobDir, sha), []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
		files[sha] = int64(len(c))
	}
	data, err := json.Marshal(weightdisk.Index{Format: weightdisk.Format, Source: "gs://bucket", Files: files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(disk, weightdisk.IndexFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
	return disk, files
}

func server(t *testing.T, disk string, mounts *[]string) *Server {
	return &Server{
		MountRoot: t.TempDir(),
		Mount: func(_ context.Context, device, dir string) error {
			*mounts = append(*mounts, device)
			if err := os.Remove(dir); err != nil {
				return err
			}
			return os.Symlink(disk, dir)
		},
		Unmount: func(_ context.Context, dir string) error { return os.Remove(dir) },
		// the fake mount replaces the directory with a symlink to the fake disk
		MountPoint: func(dir string) bool { fi, err := os.Lstat(dir); return err == nil && fi.Mode()&os.ModeSymlink != 0 },
	}
}

func testServer(t *testing.T) *Server {
	workDir := t.TempDir()
	return &Server{
		WorkRoot: workDir,
		Processes: map[string]ProcessConfig{
			"loop": {
				Binary: os.Args[0],
				Args:   []string{"loop"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"exit-42": {
				Binary: os.Args[0],
				Args:   []string{"exit-42"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"parent-with-child": {
				Binary: os.Args[0],
				Args:   []string{"parent-with-child"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"large-output": {
				Binary: os.Args[0],
				Args:   []string{"large-output"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"print-card": {
				Binary: os.Args[0],
				Args:   []string{"print-card"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"sigterm-trap": {
				Binary: os.Args[0],
				Args:   []string{"sigterm-trap"},
				Env:    []string{"TEST_STANDIN=1"},
			},
			"with-plan": {
				Binary:          os.Args[0],
				Args:            []string{"loop"},
				Env:             []string{"TEST_STANDIN=1"},
				AcceptsPlanFile: true,
			},
		},
	}
}

func TestMountWeights(t *testing.T) {
	ctx := t.Context()
	disk, files := fakeDisk(t, map[string]string{"a": "the first checkpoint file", "b": "and the second"})
	var mounts []string
	s := server(t, disk, &mounts)
	resp, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights", Verify: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(mounts) != 1 || mounts[0] != "/dev/disk/by-id/google-vxpu-weights" {
		t.Errorf("mounted %v, want the vxpu-weights device once", mounts)
	}
	if resp.GetBlobDir() != filepath.Join(s.MountRoot, "vxpu-weights", weightdisk.BlobDir) {
		t.Errorf("blob dir %q", resp.GetBlobDir())
	}
	if len(resp.GetFiles()) != len(files) || resp.GetTotalBytes() != int64(len("the first checkpoint file")+len("and the second")) {
		t.Errorf("files %v, %d bytes", resp.GetFiles(), resp.GetTotalBytes())
	}
	// mounting again is a no-op: the server doesn't mount a mounted disk twice
	if _, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil || len(mounts) != 1 {
		t.Errorf("second mount: %v, mounts %v", err, mounts)
	}
	if _, err := s.UnmountWeights(ctx, &pb.UnmountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil {
		t.Fatal(err)
	}
	if s.mounted(filepath.Join(s.MountRoot, "vxpu-weights")) {
		t.Error("still mounted after UnmountWeights")
	}
}

func TestMountWeightsCatchesACorruptFile(t *testing.T) {
	ctx := t.Context()
	disk, files := fakeDisk(t, map[string]string{"a": "the first checkpoint file"})
	for sha := range files { // same size, different bytes
		if err := os.WriteFile(filepath.Join(disk, weightdisk.BlobDir, sha), []byte("THE FIRST CHECKPOINT FILE"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var mounts []string
	s := server(t, disk, &mounts)
	if _, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil {
		t.Fatalf("without verify, sizes are all that is checked: %v", err)
	}
	_, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights", Verify: true})
	if status.Code(err) != codes.DataLoss {
		t.Errorf("verify: %v, want DataLoss", err)
	}
}

func TestMountWeightsRefusesOddDeviceNames(t *testing.T) {
	s := &Server{MountRoot: t.TempDir()}
	for _, name := range []string{"", "../etc", "Weights", "a/b"} {
		_, err := s.MountWeights(t.Context(), &pb.MountWeightsRequest{DeviceName: name})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("%q: %v, want InvalidArgument", name, err)
		}
	}
}

// The server unmounts a disk that has no weights index, rather than leave
// it where a weights disk belongs. A plain directory isn't a mount point.
func TestMountWeightsUnmountsANonWeightsDisk(t *testing.T) {
	var mounts []string
	s := server(t, t.TempDir(), &mounts) // an empty disk: no index
	if _, err := s.MountWeights(t.Context(), &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("err %v, want FailedPrecondition", err)
	}
	if len(mounts) != 1 || s.mounted(filepath.Join(s.MountRoot, "vxpu-weights")) {
		t.Errorf("mounts %v; still mounted after the index check failed", mounts)
	}
	if (&Server{}).mounted(t.TempDir()) {
		t.Error("a plain directory counts as a mount point")
	}
}

func TestProcessLifecycle(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "worker",
		Process: "loop",
		Port:    50051,
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	if startResp.GetAlreadyRunning() {
		t.Errorf("got already_running true, want false")
	}
	if startResp.GetStatus().GetState() != "running" || startResp.GetStatus().GetPid() == 0 {
		t.Errorf("status not running: %v", startResp.GetStatus())
	}

	// Sees it running.
	getResp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "worker"})
	if err != nil {
		t.Fatalf("GetProcess: %v", err)
	}
	if getResp.GetStatus().GetState() != "running" {
		t.Errorf("GetProcess reported not running: %v", getResp.GetStatus())
	}

	// Stops it.
	if _, err := s.StopProcess(ctx, &pb.StopProcessRequest{Id: "worker"}); err != nil {
		t.Fatalf("StopProcess: %v", err)
	}

	// Sees it gone.
	_, err = s.GetProcess(ctx, &pb.GetProcessRequest{Id: "worker"})
	if status.Code(err) != codes.NotFound {
		t.Errorf("GetProcess after StopProcess got %v, want NotFound", err)
	}
	listResp, err := s.ListProcesses(ctx, &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("ListProcesses: %v", err)
	}
	if len(listResp.GetProcesses()) != 0 {
		t.Errorf("ListProcesses reported %v, want empty", listResp.GetProcesses())
	}
}

func TestProcessAlreadyRunning(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	req := &pb.StartProcessRequest{
		Id:      "r1",
		Process: "loop",
		Port:    50051,
	}
	resp1, err := s.StartProcess(ctx, req)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	if resp1.GetAlreadyRunning() {
		t.Error("first start reported already_running")
	}

	// Starting again with matching configuration reports already_running.
	resp2, err := s.StartProcess(ctx, req)
	if err != nil {
		t.Fatalf("second start: %v", err)
	}
	if !resp2.GetAlreadyRunning() {
		t.Error("second start did not report already_running")
	}
	if resp2.GetStatus().GetPid() != resp1.GetStatus().GetPid() {
		t.Errorf("pids differ: %d vs %d", resp2.GetStatus().GetPid(), resp1.GetStatus().GetPid())
	}

	// Starting with differing configuration is refused.
	diffReq := &pb.StartProcessRequest{
		Id:      "r1",
		Process: "loop",
		Port:    50052,
	}
	_, err = s.StartProcess(ctx, diffReq)
	if status.Code(err) != codes.AlreadyExists {
		t.Errorf("differing settings: got %v, want AlreadyExists", err)
	}
}

func TestMultipleProcesses(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	for i, id := range []string{"rank-0", "rank-1"} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:      id,
			Process: "loop",
			Port:    int32(50051 + i),
		})
		if err != nil {
			t.Fatalf("start %s: %v", id, err)
		}
	}

	list, err := s.ListProcesses(ctx, &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("ListProcesses: %v", err)
	}
	if len(list.GetProcesses()) != 2 {
		t.Fatalf("got %d processes, want 2", len(list.GetProcesses()))
	}
	if list.GetProcesses()[0].GetId() != "rank-0" || list.GetProcesses()[1].GetId() != "rank-1" {
		t.Errorf("processes: %v", list.GetProcesses())
	}
	if list.GetProcesses()[0].GetPid() == list.GetProcesses()[1].GetPid() {
		t.Error("different processes shared the same PID")
	}
}

func TestProcessExitsByItself(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "self-exit",
		Process: "exit-42",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}

	var lastStatus *pb.ProcessStatus
	for range 100 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "self-exit"})
		if err != nil {
			t.Fatalf("GetProcess: %v", err)
		}
		lastStatus = resp.GetStatus()
		if lastStatus.GetState() != "running" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if lastStatus.GetState() == "running" {
		t.Fatal("process still running after deadline")
	}
	if lastStatus.GetState() != "exited with code 42" {
		t.Errorf("state %q, want 'exited with code 42'", lastStatus.GetState())
	}
	if lastStatus.GetExitCode() != 42 {
		t.Errorf("exit_code %d, want 42", lastStatus.GetExitCode())
	}
	if !strings.Contains(lastStatus.GetOutput(), "standin exiting with 42") {
		t.Errorf("output %q, want standin exiting message", lastStatus.GetOutput())
	}
}

func TestProcessKilled(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "victim",
		Process: "loop",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	pid := int(startResp.GetStatus().GetPid())

	// Wait until standin has emitted initial output.
	for range 50 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "victim"})
		if err == nil && strings.Contains(resp.GetStatus().GetOutput(), "standin running") {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatalf("Kill: %v", err)
	}

	var lastStatus *pb.ProcessStatus
	for range 100 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "victim"})
		if err != nil {
			t.Fatalf("GetProcess: %v", err)
		}
		lastStatus = resp.GetStatus()
		if lastStatus.GetState() != "running" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if lastStatus.GetState() == "running" {
		t.Fatal("process still running after kill")
	}
	if lastStatus.GetState() != "killed by signal 9" {
		t.Errorf("state %q, want 'killed by signal 9'", lastStatus.GetState())
	}
	if lastStatus.GetSignal() != 9 {
		t.Errorf("signal %d, want 9", lastStatus.GetSignal())
	}
	if !strings.Contains(lastStatus.GetOutput(), "standin started") {
		t.Errorf("output %q, want output before termination", lastStatus.GetOutput())
	}
}

func TestStopProcessTerminatesProcessGroupAndDoesNotHang(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "parent",
		Process: "parent-with-child",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	parentPid := int(startResp.GetStatus().GetPid())

	// Wait for parent to output child PID.
	var childPid int
	for range 50 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "parent"})
		if err == nil {
			output := resp.GetStatus().GetOutput()
			if strings.Contains(output, "parent started child") {
				for _, line := range strings.Split(output, "\n") {
					if strings.HasPrefix(line, "parent started child ") {
						pidStr := strings.TrimSpace(strings.TrimPrefix(line, "parent started child "))
						childPid, _ = strconv.Atoi(pidStr)
						break
					}
				}
				if childPid > 0 {
					break
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if childPid == 0 {
		t.Fatal("parent did not start child in time")
	}

	// Stop parent: must terminate child process group and complete within 3 seconds.
	t0 := time.Now()
	if _, err := s.StopProcess(ctx, &pb.StopProcessRequest{Id: "parent"}); err != nil {
		t.Fatalf("StopProcess: %v", err)
	}
	if elapsed := time.Since(t0); elapsed > 3*time.Second {
		t.Errorf("StopProcess took %v, want < 3s", elapsed)
	}

	// Check that both parent and child are terminated (either reaped or zombie).
	if isAlive(parentPid) {
		t.Errorf("parent pid %d still alive after StopProcess", parentPid)
	}
	childGone := false
	for range 50 {
		if !isAlive(childPid) {
			childGone = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !childGone {
		t.Errorf("child pid %d still alive after StopProcess", childPid)
	}
}

func isAlive(pid int) bool {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return false
	}
	fields := strings.Fields(string(data))
	if len(fields) >= 3 && fields[2] == "Z" {
		return false
	}
	return true
}

func TestTailBufferBoundedInMemory(t *testing.T) {
	// 1. Direct tailBuffer test: 64 MiB with no newlines must not grow unboundedly.
	tb := newTailBuffer(100, 64*1024)
	chunk := make([]byte, 1024*1024)
	for i := range chunk {
		chunk[i] = 'A'
	}
	for range 64 {
		_, err := tb.Write(chunk)
		if err != nil {
			t.Fatal(err)
		}
	}
	out := tb.String()
	if len(out) > 64*1024 {
		t.Errorf("tailBuffer length %d exceeds max 64 KiB", len(out))
	}

	// Carriage returns are treated as line delimiters.
	_, err := tb.Write([]byte("\rprogress 50%\rprogress 100%\n"))
	if err != nil {
		t.Fatal(err)
	}
	out = tb.String()
	if !strings.Contains(out, "progress 100%") {
		t.Errorf("expected carriage return progress in output: %q", out)
	}

	// 2. Integration test via runtime: returned status output must not exceed MaxOutputBytes.
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "big-output",
		Process: "large-output",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}

	finishedWriting := false
	for range 500 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "big-output"})
		if err == nil {
			output := resp.GetStatus().GetOutput()
			if strings.Contains(output, "progress 100%") {
				finishedWriting = true
				if len(output) > MaxOutputBytes {
					t.Errorf("returned process status output length %d exceeds MaxOutputBytes %d", len(output), MaxOutputBytes)
				}
				if len(output) == 0 {
					t.Errorf("returned process status output is empty")
				}
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !finishedWriting {
		t.Fatal("large-output did not finish writing in time")
	}
}

func TestProcessStoppingRejectsConcurrentStart(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "stopping-check",
		Process: "loop",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}

	// Mark it stopping under lock.
	s.procMu.Lock()
	proc := s.processes["stopping-check"]
	proc.setStopping()
	s.procMu.Unlock()

	// Concurrent StartProcess for same ID must be rejected with "stopping".
	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "stopping-check",
		Process: "loop",
	})
	if status.Code(err) != codes.Unavailable {
		t.Errorf("concurrent start while stopping got %v, want Unavailable", err)
	}
}

func TestHammerStartAndStop(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	var mu sync.Mutex
	startedPIDs := make(map[int]bool)

	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Starter goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				resp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
					Id:      "hammer",
					Process: "loop",
				})
				if err == nil && resp.GetStatus().GetPid() > 0 {
					mu.Lock()
					startedPIDs[int(resp.GetStatus().GetPid())] = true
					mu.Unlock()
				}
				time.Sleep(1 * time.Millisecond)
			}
		}
	}()

	// Stopper goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_, _ = s.StopProcess(ctx, &pb.StopProcessRequest{Id: "hammer"})
				time.Sleep(1 * time.Millisecond)
			}
		}
	}()

	time.Sleep(2 * time.Second)
	close(stop)
	wg.Wait()

	// Final stop to ensure the process is stopped.
	_, err := s.StopProcess(ctx, &pb.StopProcessRequest{Id: "hammer"})
	if err != nil {
		t.Fatalf("final StopProcess: %v", err)
	}

	listResp, err := s.ListProcesses(ctx, &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("ListProcesses: %v", err)
	}
	trackedPIDs := make(map[int]bool)
	for _, r := range listResp.GetProcesses() {
		trackedPIDs[int(r.GetPid())] = true
	}

	mu.Lock()
	pids := make([]int, 0, len(startedPIDs))
	for pid := range startedPIDs {
		pids = append(pids, pid)
	}
	mu.Unlock()

	t.Logf("started %d processes during hammer test", len(pids))
	if len(pids) == 0 {
		t.Fatal("no processes were started during hammer test")
	}

	for _, pid := range pids {
		if !trackedPIDs[pid] {
			for range 25 {
				if !isAlive(pid) {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			if isAlive(pid) {
				t.Errorf("process %d is still alive outside ListProcesses", pid)
			}
		}
	}
}

func TestAgentStopTerminatesProcesses(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)

	resp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "orphan-check",
		Process: "loop",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	pid := int(resp.GetStatus().GetPid())

	// When the agent stops, the processes it started do not outlive it.
	s.Stop()

	gone := false
	for range 50 {
		if !isAlive(pid) {
			gone = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !gone {
		t.Errorf("process %d still alive after Server.Stop", pid)
	}
}

func TestProcessCardSetting(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	// 1. Omitted card: CUDA_VISIBLE_DEVICES is not set.
	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "card-omitted",
		Process: "print-card",
	})
	if err != nil {
		t.Fatalf("StartProcess omitted card: %v", err)
	}
	for range 50 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "card-omitted"})
		if err == nil && resp.GetStatus().GetState() == "exited with code 0" {
			output := strings.TrimSpace(resp.GetStatus().GetOutput())
			if output != "card:" {
				t.Errorf("omitted card: got output %q, want 'card:'", output)
			}
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	// 2. Card 0: CUDA_VISIBLE_DEVICES=0 is set.
	cardZero := int32(0)
	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "card-zero",
		Process: "print-card",
		Card:    &cardZero,
	})
	if err != nil {
		t.Fatalf("StartProcess card 0: %v", err)
	}
	for range 50 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "card-zero"})
		if err == nil && resp.GetStatus().GetState() == "exited with code 0" {
			output := strings.TrimSpace(resp.GetStatus().GetOutput())
			if output != "card: 0" {
				t.Errorf("card 0: got output %q, want 'card: 0'", output)
			}
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	// 3. Negative card is rejected.
	cardNeg := int32(-1)
	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "card-neg",
		Process: "print-card",
		Card:    &cardNeg,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("card -1: got %v, want InvalidArgument", err)
	}
}

func TestStartProcessRejectsArbitraryExecution(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	// 1. Script path or arbitrary unconfigured process name.
	for _, badProcess := range []string{
		"python3 /tmp/x/evil.py",
		"/tmp/x/python",
		"sh",
		"python3 -c print(1)",
		"unknown-process",
	} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:      "test-bad",
			Process: badProcess,
		})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("process %q: got %v, want InvalidArgument", badProcess, err)
		}
	}

	// 2. Default processes refuse plan_file when not supported.
	defaultServer := &Server{WorkRoot: t.TempDir()}
	_, err := defaultServer.StartProcess(ctx, &pb.StartProcessRequest{
		Id:       "exec-plan",
		Process:  "executor",
		PlanFile: "some.plan",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("executor with plan_file: got %v, want InvalidArgument", err)
	}

	// 3. Plan file path traversal or symlinks escaping work root.
	outsideDir := t.TempDir()
	outsideFile := filepath.Join(outsideDir, "secret.plan")
	if err := os.WriteFile(outsideFile, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	symlinkPath := filepath.Join(s.WorkRoot, "escapelink.plan")
	if err := os.Symlink(outsideFile, symlinkPath); err != nil {
		t.Fatal(err)
	}

	for _, badPlan := range []string{
		"/etc/passwd",
		"../../../etc/passwd",
		"escapelink.plan",
	} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:       "test-plan",
			Process:  "with-plan",
			PlanFile: badPlan,
		})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("plan_file %q: got %v, want InvalidArgument", badPlan, err)
		}
	}

	// Valid plan file inside work root is accepted.
	validPlan := filepath.Join(s.WorkRoot, "valid.plan")
	if err := os.WriteFile(validPlan, []byte("plan"), 0o644); err != nil {
		t.Fatal(err)
	}
	resp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:       "valid-plan-run",
		Process:  "with-plan",
		PlanFile: "valid.plan",
	})
	if err != nil {
		t.Fatalf("valid plan_file: %v", err)
	}
	_ = resp

	// 4. Invalid port numbers.
	for _, badPort := range []int32{-1, 70000} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:      "test-port",
			Process: "loop",
			Port:    badPort,
		})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("port %d: got %v, want InvalidArgument", badPort, err)
		}
	}

	// 5. Invalid card numbers.
	badCard := int32(-2)
	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "test-card",
		Process: "loop",
		Card:    &badCard,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("card -2: got %v, want InvalidArgument", err)
	}

	// 6. Invalid IDs.
	for _, id := range []string{"", "../etc", "Worker", "a/b"} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:      id,
			Process: "loop",
		})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("id %q: got %v, want InvalidArgument", id, err)
		}
	}
}

// TestDescribeMachine_VMAndPodShare verifies that when given synthetic system
// facts for a full VM and a pod with a smaller share (cgroup CPU/memory limits,
// visible devices in /dev, and downward API), the agent reports what each can actually
// use and the answers differ as expected (#663).
func TestDescribeMachine_VMAndPodShare(t *testing.T) {
	ctx := t.Context()

	// 1. Full VM setup.
	vmRoot := t.TempDir()
	cpuinfoContent := "model name\t: AMD EPYC 9B45 48-Core Processor\n" +
		"flags\t\t: fpu vme sse sse2 avx avx2 avx512f\n"
	for i := 0; i < 48; i++ {
		cpuinfoContent += fmt.Sprintf("processor\t: %d\ncore id\t: %d\n", i, i/2)
	}
	meminfoContent := "MemTotal:       188743680 kB\n"

	mustWrite := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	mustWrite(filepath.Join(vmRoot, "proc/cpuinfo"), cpuinfoContent)
	mustWrite(filepath.Join(vmRoot, "proc/meminfo"), meminfoContent)
	mustWrite(filepath.Join(vmRoot, "proc/driver/nvidia/gpus/0000:00:04.0/information"),
		"Model:\t\t NVIDIA RTX PRO 6000\nDevice Minor:\t 0\nIRQ:\t\t 10\n")
	mustWrite(filepath.Join(vmRoot, "dev/nvidia0"), "")

	for i := 0; i < 4; i++ {
		blkName := fmt.Sprintf("nvme%dn1", i)
		mustWrite(filepath.Join(vmRoot, "sys/class/block", blkName, "size"), "732421875\n") // 732421875 * 512 = 375,000,000,000 bytes
		link := filepath.Join(vmRoot, "dev/disk/by-id", fmt.Sprintf("google-local-nvme-ssd-%d", i))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join("/dev", blkName), link); err != nil {
			t.Fatal(err)
		}
	}

	vmServer := &Server{
		SysRoot: vmRoot,
		Getenv: func(string) string {
			return "" // do not read the test runner's real environment
		},
		MetadataFunc: func(ctx context.Context, subpath string) (string, error) {
			switch subpath {
			case "zone":
				return "projects/123/zones/us-central1-a", nil
			case "name":
				return "vm-test-executor-01", nil
			default:
				return "", fmt.Errorf("unknown metadata %s", subpath)
			}
		},
	}

	vmResp, err := vmServer.DescribeMachine(ctx, &pb.DescribeMachineRequest{})
	if err != nil {
		t.Fatalf("vm DescribeMachine: %v", err)
	}

	if vmResp.GetCpu() != "AMD EPYC 9B45 48-Core Processor" {
		t.Errorf("VM CPU model: got %q, want AMD EPYC 9B45 48-Core Processor", vmResp.GetCpu())
	}
	if vmResp.GetCpuThreads() != 48 {
		t.Errorf("VM CPU threads: got %d, want 48", vmResp.GetCpuThreads())
	}
	if vmResp.GetCpuCores() != 24 {
		t.Errorf("VM CPU cores: got %d, want 24", vmResp.GetCpuCores())
	}
	if vmResp.GetSimd() != "avx512" {
		t.Errorf("VM SIMD: got %q, want avx512", vmResp.GetSimd())
	}
	if wantRAM := int64(188743680) * 1024; vmResp.GetRamBytes() != wantRAM {
		t.Errorf("VM RAM: got %d, want %d", vmResp.GetRamBytes(), wantRAM)
	}
	if vmResp.GetGpu() != "NVIDIA RTX PRO 6000" {
		t.Errorf("VM GPU name: got %q, want NVIDIA RTX PRO 6000", vmResp.GetGpu())
	}
	if vmResp.GetGpuCount() != 1 {
		t.Errorf("VM GPU count: got %d, want 1", vmResp.GetGpuCount())
	}
	if vmResp.GetStorageDevices() != 4 {
		t.Errorf("VM storage devices: got %d, want 4", vmResp.GetStorageDevices())
	}
	if wantBytes := int64(4) * 375_000_000_000; vmResp.GetStorageBytes() != wantBytes {
		t.Errorf("VM storage bytes: got %d, want %d", vmResp.GetStorageBytes(), wantBytes)
	}
	if vmResp.GetZone() != "us-central1-a" {
		t.Errorf("VM zone: got %q, want us-central1-a", vmResp.GetZone())
	}
	if vmResp.GetInstanceName() != "vm-test-executor-01" {
		t.Errorf("VM instance: got %q, want vm-test-executor-01", vmResp.GetInstanceName())
	}

	// 2. Pod with smaller share setup.
	podRoot := t.TempDir()
	// Pod sees host /proc with 4 GPU cards
	mustWrite(filepath.Join(podRoot, "proc/cpuinfo"), cpuinfoContent)
	mustWrite(filepath.Join(podRoot, "proc/meminfo"), meminfoContent)
	models := []string{"NVIDIA Tesla T4", "NVIDIA Tesla V100", "NVIDIA A100-SXM4-40GB", "NVIDIA L4"}
	for i, m := range models {
		mustWrite(filepath.Join(podRoot, fmt.Sprintf("proc/driver/nvidia/gpus/0000:00:0%d.0/information", 4+i)),
			fmt.Sprintf("Model:\t\t %s\nDevice Minor:\t %d\nIRQ:\t\t 10\n", m, i))
	}
	// But in /dev the pod has only ONE device node (dev/nvidia3, testing nonzero minor mapping) and no environment variable.
	mustWrite(filepath.Join(podRoot, "dev/nvidia3"), "")

	// Pod has cgroups limiting CPU to 8 and RAM to 32 GiB
	mustWrite(filepath.Join(podRoot, "sys/fs/cgroup/cpu.max"), "800000 100000\n")
	mustWrite(filepath.Join(podRoot, "sys/fs/cgroup/memory.max"), "34359738368\n") // 32 GiB

	podEnv := map[string]string{
		"NODE_NAME": "gke-node-pool-turin-1",
	}

	podServer := &Server{
		SysRoot: podRoot,
		Getenv: func(key string) string {
			return podEnv[key]
		},
		MetadataFunc: func(ctx context.Context, subpath string) (string, error) {
			return "", fmt.Errorf("metadata blocked in pod")
		},
	}

	podResp, err := podServer.DescribeMachine(ctx, &pb.DescribeMachineRequest{})
	if err != nil {
		t.Fatalf("pod DescribeMachine: %v", err)
	}

	if podResp.GetCpuThreads() != 8 {
		t.Errorf("Pod CPU threads: got %d, want 8 (pod share)", podResp.GetCpuThreads())
	}
	if podResp.GetCpuCores() != 8 {
		t.Errorf("Pod CPU cores: got %d, want 8 (pod share)", podResp.GetCpuCores())
	}
	if podResp.GetRamBytes() != 34359738368 {
		t.Errorf("Pod RAM: got %d, want 34359738368 (pod share)", podResp.GetRamBytes())
	}
	// Four cards in /proc, but only one device node in /dev (nvidia3) -> count must be 1, model must be minor 3
	if podResp.GetGpuCount() != 1 {
		t.Errorf("Pod GPU count: got %d, want 1 (from /dev/nvidia3)", podResp.GetGpuCount())
	}
	if podResp.GetGpu() != "NVIDIA L4" {
		t.Errorf("Pod GPU name: got %q, want NVIDIA L4 (from minor 3)", podResp.GetGpu())
	}
	if podResp.GetStorageDevices() != 0 {
		t.Errorf("Pod storage devices: got %d, want 0", podResp.GetStorageDevices())
	}
	if podResp.GetStorageBytes() != 0 {
		t.Errorf("Pod storage bytes: got %d, want 0", podResp.GetStorageBytes())
	}
	// Zone is unread because metadata server is unreachable in a pod
	if podResp.GetZone() != "" {
		t.Errorf("Pod zone: got %q, want empty (unread)", podResp.GetZone())
	}
	var foundZoneUnread bool
	for _, u := range podResp.GetUnread() {
		if strings.Contains(u, "zone") {
			foundZoneUnread = true
			break
		}
	}
	if !foundZoneUnread {
		t.Errorf("expected zone unread notice in pod unread facts, got %v", podResp.GetUnread())
	}
	if podResp.GetInstanceName() != "gke-node-pool-turin-1" {
		t.Errorf("Pod instance: got %q, want gke-node-pool-turin-1", podResp.GetInstanceName())
	}

	// 3. Confirm that the VM and Pod answers differ as they should.
	if vmResp.GetCpuThreads() == podResp.GetCpuThreads() {
		t.Errorf("expected CPU threads to differ between VM (%d) and pod (%d)", vmResp.GetCpuThreads(), podResp.GetCpuThreads())
	}
	if vmResp.GetRamBytes() == podResp.GetRamBytes() {
		t.Errorf("expected RAM bytes to differ between VM (%d) and pod (%d)", vmResp.GetRamBytes(), podResp.GetRamBytes())
	}
	if vmResp.GetStorageDevices() == podResp.GetStorageDevices() {
		t.Errorf("expected storage devices to differ between VM (%d) and pod (%d)", vmResp.GetStorageDevices(), podResp.GetStorageDevices())
	}
	if vmResp.GetGpu() == podResp.GetGpu() {
		t.Errorf("expected GPU model to differ between VM (%q) and pod (%q)", vmResp.GetGpu(), podResp.GetGpu())
	}
}

// TestDescribeMachine_CardDriverUnreadable verifies that when a card's device
// node is present in /dev but its driver entry in /proc cannot be read, the
// agent reports an empty card name and records an unread entry (Item 5).
func TestDescribeMachine_CardDriverUnreadable(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()

	if err := os.MkdirAll(filepath.Join(root, "dev"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dev/nvidia0"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	server := &Server{
		SysRoot: root,
		Getenv:  func(string) string { return "" },
		MetadataFunc: func(context.Context, string) (string, error) {
			return "", fmt.Errorf("metadata blocked")
		},
	}

	resp, err := server.DescribeMachine(ctx, &pb.DescribeMachineRequest{})
	if err != nil {
		t.Fatalf("DescribeMachine: %v", err)
	}

	if resp.GetGpuCount() != 1 {
		t.Errorf("got GPU count %d, want 1", resp.GetGpuCount())
	}
	if resp.GetGpu() != "" {
		t.Errorf("got GPU name %q, want empty (unread)", resp.GetGpu())
	}
	var foundCardUnread bool
	for _, u := range resp.GetUnread() {
		if strings.Contains(u, "card 0: driver entry could not be read") {
			foundCardUnread = true
			break
		}
	}
	if !foundCardUnread {
		t.Errorf("expected card 0 unread notice, got %v", resp.GetUnread())
	}
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func blobsMap(blobs []*pb.BlobInfo) map[string]int64 {
	m := make(map[string]int64, len(blobs))
	for _, b := range blobs {
		m[b.GetName()] = b.GetSize()
	}
	return m
}

func TestStageBlobs_SuccessAndIdempotentTransfer(t *testing.T) {
	ctx := t.Context()
	blobA := []byte("first staged blob content")
	blobB := []byte("second staged blob content")
	shaA := sha256Hex(blobA)
	shaB := sha256Hex(blobB)

	var transferCount atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		transferCount.Add(1)
		switch r.URL.Path {
		case "/blobs/" + shaA:
			w.Header().Set("Content-Length", strconv.Itoa(len(blobA)))
			_, _ = w.Write(blobA)
		case "/blobs/" + shaB:
			w.Header().Set("Content-Length", strconv.Itoa(len(blobB)))
			_, _ = w.Write(blobB)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	// First call stages both blobs
	resp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{shaA, shaB},
	})
	if err != nil {
		t.Fatalf("StageBlobs: %v", err)
	}
	blobs1 := blobsMap(resp.GetBlobs())
	if blobs1[shaA] != int64(len(blobA)) {
		t.Errorf("got blobA size %d, want %d", blobs1[shaA], len(blobA))
	}
	if blobs1[shaB] != int64(len(blobB)) {
		t.Errorf("got blobB size %d, want %d", blobs1[shaB], len(blobB))
	}
	if got := transferCount.Load(); got != 2 {
		t.Errorf("expected 2 transfers on first call, got %d", got)
	}

	// Verify files exist on disk
	dataA, err := os.ReadFile(filepath.Join(blobDir, shaA))
	if err != nil || string(dataA) != string(blobA) {
		t.Errorf("blobA on disk mismatch: %v", err)
	}
	dataB, err := os.ReadFile(filepath.Join(blobDir, shaB))
	if err != nil || string(dataB) != string(blobB) {
		t.Errorf("blobB on disk mismatch: %v", err)
	}

	// Second identical call transfers nothing
	resp2, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{shaA, shaB},
	})
	if err != nil {
		t.Fatalf("second StageBlobs: %v", err)
	}
	blobs2 := blobsMap(resp2.GetBlobs())
	if blobs2[shaA] != int64(len(blobA)) || blobs2[shaB] != int64(len(blobB)) {
		t.Errorf("second call sizes mismatch: %+v", blobs2)
	}
	if got := transferCount.Load(); got != 2 {
		t.Errorf("expected no additional transfers on second call, got %d", got)
	}
}

func TestStageBlobs_CorruptBlobRefusedAndNoFileLeft(t *testing.T) {
	ctx := t.Context()
	validBytes := []byte("correct content")
	sha := sha256Hex(validBytes)
	corruptBytes := []byte("mismatched corrupted content")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/blobs/"+sha {
			_, _ = w.Write(corruptBytes)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{sha},
	})
	if err == nil {
		t.Fatal("expected error for corrupted blob, got nil")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.DataLoss {
		t.Errorf("expected DataLoss error, got %v", err)
	}
	if !strings.Contains(err.Error(), sha) {
		t.Errorf("expected error to name blob %s, got %v", sha, err)
	}

	// Assert no file left under that name
	if _, err := os.Stat(filepath.Join(blobDir, sha)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected no file under %s, got %v", sha, err)
	}

	// Assert no temporary files left in blobDir
	entries, _ := os.ReadDir(blobDir)
	if len(entries) != 0 {
		t.Errorf("expected empty blobDir after corrupt transfer, found %d entries", len(entries))
	}
}

func TestStageBlobs_InterruptedTransferLeavesNothing(t *testing.T) {
	blobContent := []byte("interrupted blob content that will be cut short")
	sha := sha256Hex(blobContent)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(blobContent)))
		_, _ = w.Write(blobContent[:10])
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, _ := hj.Hijack()
			_ = conn.Close()
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	_, err := server.StageBlobs(t.Context(), &pb.StageBlobsRequest{
		Names: []string{sha},
	})
	if err == nil {
		t.Fatal("expected error on interrupted transfer, got nil")
	}

	// No file left under sha
	if _, err := os.Stat(filepath.Join(blobDir, sha)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected no file under %s, got %v", sha, err)
	}

	// Now start a healthy server and verify a later call succeeds
	healthySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/blobs/"+sha {
			_, _ = w.Write(blobContent)
		} else {
			http.NotFound(w, r)
		}
	}))
	defer healthySrv.Close()
	server.Store = weightstore.New(healthySrv.URL)

	resp, err := server.StageBlobs(t.Context(), &pb.StageBlobsRequest{
		Names: []string{sha},
	})
	if err != nil {
		t.Fatalf("subsequent StageBlobs failed: %v", err)
	}
	blobs := blobsMap(resp.GetBlobs())
	if blobs[sha] != int64(len(blobContent)) {
		t.Errorf("got size %d, want %d", blobs[sha], len(blobContent))
	}
}

func TestStageBlobs_InvalidNameRefusedBeforeRequestLeavesMachine(t *testing.T) {
	ctx := t.Context()
	var serverHit atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serverHit.Store(true)
		http.NotFound(w, r)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	invalidNames := []string{
		"http://attacker.example.com/blob",
		"https://evil.com/payload",
		"../escape-path",
		"not-a-sha",
		strings.Repeat("A", 64),
		strings.Repeat("a", 63),
		strings.Repeat("a", 65),
		"7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d906g",
	}

	for _, name := range invalidNames {
		_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{name}})
		if err == nil {
			t.Errorf("expected StageBlobs to refuse invalid name %q", name)
		}
		if st, ok := status.FromError(err); !ok || st.Code() != codes.InvalidArgument {
			t.Errorf("expected InvalidArgument for name %q, got %v", name, err)
		}
	}

	if serverHit.Load() {
		t.Error("a request left the machine for an invalid blob name")
	}
}

func TestStageBlobs_AskWhatIsStagedWithoutStaging(t *testing.T) {
	ctx := t.Context()
	blobDir := t.TempDir()
	content := []byte("already present on local disk")
	sha := sha256Hex(content)

	if err := os.WriteFile(filepath.Join(blobDir, sha), content, 0o644); err != nil {
		t.Fatal(err)
	}

	// Server has NO store configured (Store is nil)
	server := &Server{
		BlobDir: blobDir,
		Store:   nil,
	}

	// ListBlobs returns all staged blobs
	listResp, err := server.ListBlobs(ctx, &pb.ListBlobsRequest{})
	if err != nil {
		t.Fatalf("ListBlobs: %v", err)
	}
	blobs := blobsMap(listResp.GetBlobs())
	if len(blobs) != 1 || blobs[sha] != int64(len(content)) {
		t.Errorf("ListBlobs got %+v, want 1 entry for %s", blobs, sha)
	}

	// StageBlobs with empty names stages nothing and returns empty list
	stageResp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{})
	if err != nil {
		t.Fatalf("StageBlobs empty: %v", err)
	}
	if len(stageResp.GetBlobs()) != 0 {
		t.Errorf("StageBlobs empty should return empty list, got %+v", stageResp.GetBlobs())
	}
}

func TestStageBlobs_ConcurrentSameBlobTransfersOnce(t *testing.T) {
	ctx := t.Context()
	content := []byte("shared content for concurrent transfer")
	sha := sha256Hex(content)

	var transfers atomic.Int32
	started := make(chan struct{})
	gate := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		transfers.Add(1)
		close(started)
		<-gate
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	var wg sync.WaitGroup
	var resp1, resp2 *pb.StageBlobsResponse
	var err1, err2 error

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp1, err1 = server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	<-started

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp2, err2 = server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	time.Sleep(20 * time.Millisecond)
	close(gate)

	wg.Wait()

	if err1 != nil {
		t.Fatalf("call 1 failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("call 2 failed: %v", err2)
	}
	b1 := blobsMap(resp1.GetBlobs())
	b2 := blobsMap(resp2.GetBlobs())
	if b1[sha] != int64(len(content)) || b2[sha] != int64(len(content)) {
		t.Errorf("sizes mismatch: resp1=%+v, resp2=%+v", b1, b2)
	}
	if got := transfers.Load(); got != 1 {
		t.Errorf("expected exactly 1 transfer from store, got %d", got)
	}
}

func TestStageBlobs_ConcurrentDifferentBlobsProceedTogether(t *testing.T) {
	ctx := t.Context()
	contentA := []byte("blob A content")
	contentB := []byte("blob B content")
	shaA := sha256Hex(contentA)
	shaB := sha256Hex(contentB)

	startedA := make(chan struct{})
	startedB := make(chan struct{})
	releaseBoth := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/blobs/" + shaA:
			close(startedA)
			<-releaseBoth
			w.Header().Set("Content-Length", strconv.Itoa(len(contentA)))
			_, _ = w.Write(contentA)
		case "/blobs/" + shaB:
			close(startedB)
			<-releaseBoth
			w.Header().Set("Content-Length", strconv.Itoa(len(contentB)))
			_, _ = w.Write(contentB)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	var wg sync.WaitGroup
	var errA, errB error

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, errA = server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{shaA}})
	}()

	<-startedA

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, errB = server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{shaB}})
	}()

	select {
	case <-startedB:
	case <-time.After(2 * time.Second):
		t.Fatal("call for B was blocked behind A's transfer")
	}

	close(releaseBoth)
	wg.Wait()

	if errA != nil {
		t.Fatalf("stage A failed: %v", errA)
	}
	if errB != nil {
		t.Fatalf("stage B failed: %v", errB)
	}
}

func TestStageBlobs_ContextCancellationWhileWaitingInFlight(t *testing.T) {
	content := []byte("blob for cancellation test")
	sha := sha256Hex(content)

	started := make(chan struct{})
	holdTransfer := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-holdTransfer
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = server.StageBlobs(t.Context(), &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	<-started

	ctx2, cancel2 := context.WithCancel(t.Context())
	errChan := make(chan error, 1)
	go func() {
		_, err := server.StageBlobs(ctx2, &pb.StageBlobsRequest{Names: []string{sha}})
		errChan <- err
	}()

	time.Sleep(20 * time.Millisecond)
	cancel2()

	select {
	case err := <-errChan:
		if st, ok := status.FromError(err); !ok || st.Code() != codes.Canceled {
			t.Errorf("expected Canceled status, got %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("caller 2 hung instead of noticing its context cancellation")
	}

	close(holdTransfer)
	wg.Wait()
}

func TestStageBlobs_WaiterFetchesWhenLeaderCancelled(t *testing.T) {
	content := []byte("blob for leader cancellation takeover test")
	sha := sha256Hex(content)

	var transferAttempts atomic.Int32
	startedAttempt1 := make(chan struct{})
	gateAttempt1 := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := transferAttempts.Add(1)
		if attempt == 1 {
			close(startedAttempt1)
			<-gateAttempt1
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	ctx1, cancel1 := context.WithCancel(t.Context())
	ctx2 := t.Context()

	var wg sync.WaitGroup
	var resp2 *pb.StageBlobsResponse
	var err1, err2 error

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err1 = server.StageBlobs(ctx1, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	<-startedAttempt1

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp2, err2 = server.StageBlobs(ctx2, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	time.Sleep(20 * time.Millisecond)
	cancel1()
	close(gateAttempt1)

	wg.Wait()

	if st, ok := status.FromError(err1); !ok || st.Code() != codes.Canceled {
		t.Errorf("caller 1 expected Canceled error, got %v", err1)
	}
	if err2 != nil {
		t.Fatalf("caller 2 should have fetched the blob itself after leader cancelled, but failed: %v", err2)
	}
	b2 := blobsMap(resp2.GetBlobs())
	if b2[sha] != int64(len(content)) {
		t.Errorf("caller 2 got size %d, want %d", b2[sha], len(content))
	}

	data, err := os.ReadFile(filepath.Join(blobDir, sha))
	if err != nil || string(data) != string(content) {
		t.Errorf("file on disk mismatch: %v", err)
	}
}

func TestStageBlobs_WaiterFetchesWhenLeaderFails(t *testing.T) {
	content := []byte("blob for leader failure takeover test")
	sha := sha256Hex(content)

	var transferAttempts atomic.Int32
	startedAttempt1 := make(chan struct{})
	gateAttempt1 := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := transferAttempts.Add(1)
		if attempt == 1 {
			close(startedAttempt1)
			<-gateAttempt1
			corrupt := []byte("corrupt body returned on first attempt")
			w.Header().Set("Content-Length", strconv.Itoa(len(corrupt)))
			_, _ = w.Write(corrupt)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	ctx1 := t.Context()
	ctx2 := t.Context()

	var wg sync.WaitGroup
	var resp2 *pb.StageBlobsResponse
	var err1, err2 error

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err1 = server.StageBlobs(ctx1, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	<-startedAttempt1

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp2, err2 = server.StageBlobs(ctx2, &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	time.Sleep(20 * time.Millisecond)
	close(gateAttempt1)

	wg.Wait()

	if st, ok := status.FromError(err1); !ok || st.Code() != codes.DataLoss {
		t.Errorf("caller 1 expected DataLoss error from corrupt store response, got %v", err1)
	}
	if err2 != nil {
		t.Fatalf("caller 2 should have fetched the blob itself after leader failed, but failed: %v", err2)
	}
	b2 := blobsMap(resp2.GetBlobs())
	if b2[sha] != int64(len(content)) {
		t.Errorf("caller 2 got size %d, want %d", b2[sha], len(content))
	}

	data, err := os.ReadFile(filepath.Join(blobDir, sha))
	if err != nil || string(data) != string(content) {
		t.Errorf("file on disk mismatch: %v", err)
	}
}

func TestStageBlobs_StoreAlwaysFailsGivesEachCallerItsOwnError(t *testing.T) {
	content := []byte("blob that always fails in store")
	sha := sha256Hex(content)

	var transferAttempts atomic.Int32
	startedAttempt1 := make(chan struct{})
	gateAttempt1 := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := transferAttempts.Add(1)
		if attempt == 1 {
			close(startedAttempt1)
			<-gateAttempt1
		}
		corrupt := []byte("always corrupt body")
		w.Header().Set("Content-Length", strconv.Itoa(len(corrupt)))
		_, _ = w.Write(corrupt)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	const callers = 3
	var wg sync.WaitGroup
	errs := make([]error, callers)

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, errs[0] = server.StageBlobs(t.Context(), &pb.StageBlobsRequest{Names: []string{sha}})
	}()

	<-startedAttempt1

	for i := 1; i < callers; i++ {
		idx := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[idx] = server.StageBlobs(t.Context(), &pb.StageBlobsRequest{Names: []string{sha}})
		}()
	}

	time.Sleep(20 * time.Millisecond)
	close(gateAttempt1)

	wg.Wait()

	for i, err := range errs {
		if st, ok := status.FromError(err); !ok || st.Code() != codes.DataLoss {
			t.Errorf("caller %d expected DataLoss error, got %v", i, err)
		}
	}

	if got := transferAttempts.Load(); got != callers {
		t.Errorf("expected %d transfer attempts, got %d", callers, got)
	}

	if _, err := os.Stat(filepath.Join(blobDir, sha)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected no file on disk, got %v", err)
	}
}

func TestStageBlobs_LeaderCancelledBeforeBodyArrivesReportsCanceled(t *testing.T) {
	content := []byte("blob for pre-headers cancellation test")
	sha := sha256Hex(content)

	started := make(chan struct{})
	gate := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-gate
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	ctx, cancel := context.WithCancel(t.Context())
	errChan := make(chan error, 1)

	go func() {
		_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{sha}})
		errChan <- err
	}()

	<-started
	cancel()

	select {
	case err := <-errChan:
		if st, ok := status.FromError(err); !ok || st.Code() != codes.Canceled {
			t.Errorf("expected Canceled when cancelled before body arrives, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("hung waiting for cancellation error")
	}
	close(gate)
}

func TestStageBlobs_CancelledTransferReportsCanceledNotDataLoss(t *testing.T) {
	content := []byte("blob for cancelled transfer reports canceled test")
	sha := sha256Hex(content)

	ctx, cancel := context.WithCancel(t.Context())

	bodyGate := make(chan struct{})
	started := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content[:5])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(started)
		<-bodyGate
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	errChan := make(chan error, 1)
	go func() {
		_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{sha}})
		errChan <- err
	}()

	<-started
	cancel()
	close(bodyGate)

	err := <-errChan
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.Canceled {
		t.Errorf("expected Canceled when transfer cut short by cancellation, got code %v (%v)", st.Code(), err)
	}
}

func TestStageBlobs_EmptyWorkRootRefused(t *testing.T) {
	ctx := t.Context()
	server := &Server{
		WorkRoot: "",
		BlobDir:  "",
	}

	_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{Names: []string{strings.Repeat("a", 64)}})
	if err == nil {
		t.Fatal("expected error with empty work root, got nil")
	}
	if st, ok := status.FromError(err); !ok || st.Code() != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", err)
	}

	_, err = server.ListBlobs(ctx, &pb.ListBlobsRequest{})
	if err == nil {
		t.Fatal("expected error with empty work root in ListBlobs, got nil")
	}
	if st, ok := status.FromError(err); !ok || st.Code() != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", err)
	}
}

func TestListBlobs_ExcludesNonRegularFiles(t *testing.T) {
	ctx := t.Context()
	blobDir := t.TempDir()

	validContent := []byte("regular file content")
	validSha := sha256Hex(validContent)
	if err := os.WriteFile(filepath.Join(blobDir, validSha), validContent, 0o644); err != nil {
		t.Fatal(err)
	}

	dirSha := strings.Repeat("d", 64)
	if err := os.Mkdir(filepath.Join(blobDir, dirSha), 0o755); err != nil {
		t.Fatal(err)
	}

	symlinkSha := strings.Repeat("e", 64)
	targetPath := filepath.Join(t.TempDir(), "target")
	if err := os.WriteFile(targetPath, []byte("target"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(targetPath, filepath.Join(blobDir, symlinkSha)); err != nil {
		t.Fatal(err)
	}

	server := &Server{
		BlobDir: blobDir,
	}

	respAll, err := server.ListBlobs(ctx, &pb.ListBlobsRequest{})
	if err != nil {
		t.Fatalf("ListBlobs: %v", err)
	}
	blobs := blobsMap(respAll.GetBlobs())
	if _, ok := blobs[validSha]; !ok {
		t.Errorf("expected validSha in listing")
	}
	if _, ok := blobs[dirSha]; ok {
		t.Errorf("dirSha should not be listed as a blob")
	}
	if _, ok := blobs[symlinkSha]; ok {
		t.Errorf("symlinkSha should not be listed as a blob")
	}
}

func TestStageBlobs_CleansStaleTempFiles(t *testing.T) {
	blobDir := t.TempDir()

	staleTemp := filepath.Join(blobDir, ".0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.tmp-12345")
	if err := os.WriteFile(staleTemp, []byte("abandoned bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	server := &Server{
		BlobDir: blobDir,
	}
	server.Start()

	if _, err := os.Stat(staleTemp); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected stale temp file to be removed on server start, stat err: %v", err)
	}
}

func TestStageBlobs_LargeBlobBoundedInMemory(t *testing.T) {
	ctx := t.Context()
	const blobSize = 16 * 1024 * 1024 // 16 MiB
	h := sha256.New()
	chunk := make([]byte, 64*1024)
	for i := range chunk {
		chunk[i] = byte(i % 251)
	}
	for written := 0; written < blobSize; written += len(chunk) {
		h.Write(chunk)
	}
	sha := hex.EncodeToString(h.Sum(nil))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/blobs/"+sha {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(blobSize))
		for written := 0; written < blobSize; written += len(chunk) {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	runtime.GC()
	var mBefore runtime.MemStats
	runtime.ReadMemStats(&mBefore)

	resp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{sha},
	})
	if err != nil {
		t.Fatalf("StageBlobs large blob: %v", err)
	}
	blobs := blobsMap(resp.GetBlobs())
	if blobs[sha] != blobSize {
		t.Errorf("got size %d, want %d", blobs[sha], blobSize)
	}

	var mAfter runtime.MemStats
	runtime.ReadMemStats(&mAfter)

	allocDelta := mAfter.TotalAlloc - mBefore.TotalAlloc
	if allocDelta >= blobSize {
		t.Errorf("memory not bounded: heap alloc delta %d bytes >= blob size %d bytes", allocDelta, blobSize)
	}

	fi, err := os.Stat(filepath.Join(blobDir, sha))
	if err != nil {
		t.Fatalf("stat staged blob: %v", err)
	}
	if fi.Size() != blobSize {
		t.Errorf("file size on disk %d, want %d", fi.Size(), blobSize)
	}
}

func runStandinServer(ctx context.Context, workRoot, binary string) error {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	s := grpc.NewServer()
	srv := &Server{
		WorkRoot: workRoot,
		Processes: map[string]ProcessConfig{
			"sigterm-trap": {
				Binary: binary,
				Args:   []string{"sigterm-trap"},
				Env:    []string{"TEST_STANDIN=1"},
			},
		},
	}
	defer srv.Stop()
	pb.RegisterAgentServer(s, srv)
	go func() {
		<-ctx.Done()
		srv.Stop()
		s.GracefulStop()
	}()
	srv.forceReaper = true
	srv.StartReaper()
	fmt.Printf("ready: %s\n", lis.Addr().String())
	return s.Serve(lis)
}

func TestTerminationSignalForwardedToRuntime(t *testing.T) {
	ctx := t.Context()
	workRoot := t.TempDir()

	cmd := exec.Command(os.Args[0], "standin-agent", workRoot)
	cmd.Env = append(os.Environ(), "TEST_STANDIN=1")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting agent process: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}()

	scanner := bufio.NewScanner(stdout)
	var agentAddr string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "ready: ") {
			agentAddr = strings.TrimPrefix(line, "ready: ")
			break
		}
	}
	if agentAddr == "" {
		t.Fatal("agent failed to report ready address")
	}

	conn, err := grpc.NewClient(agentAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dialing agent: %v", err)
	}
	defer conn.Close()

	client := pb.NewAgentClient(conn)
	resp, err := client.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "runtime",
		Process: "sigterm-trap",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	runtimePid := int(resp.GetStatus().GetPid())

	// Wait until runtime process is up and running.
	for range 100 {
		pResp, err := client.GetProcess(ctx, &pb.GetProcessRequest{Id: "runtime"})
		if err == nil && strings.Contains(pResp.GetStatus().GetOutput(), "sigterm-trap started") {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	// Deliver actual SIGTERM signal to the agent process.
	if err := syscall.Kill(cmd.Process.Pid, syscall.SIGTERM); err != nil {
		t.Fatalf("sending SIGTERM to agent: %v", err)
	}

	waitDone := make(chan error, 1)
	go func() {
		waitDone <- cmd.Wait()
	}()

	select {
	case err := <-waitDone:
		if err != nil {
			t.Logf("agent exited: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("agent process failed to stop after SIGTERM")
	}

	// Verify runtime process also received SIGTERM and is no longer alive.
	dead := false
	for range 100 {
		if !isAlive(runtimePid) {
			dead = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !dead {
		t.Errorf("runtime process %d still alive after SIGTERM was sent to agent", runtimePid)
	}
}

func TestProcessExecution_BothSituations(t *testing.T) {
	for _, asProcess1 := range []bool{false, true} {
		name := "ordinary"
		if asProcess1 {
			name = "process1"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			s := testServer(t)
			if asProcess1 {
				s.forceReaper = true
				s.StartReaper()
			}
			defer s.Stop()

			// 1. Helper commands (s.run):
			// Successful helper command:
			out, err := s.run(ctx, "echo", "helper-success")
			if err != nil {
				t.Fatalf("helper command error: %v", err)
			}
			if !strings.Contains(string(out), "helper-success") {
				t.Errorf("helper output = %q, want helper-success", string(out))
			}

			// Failing helper command with exit status:
			outFail, errFail := s.run(ctx, "sh", "-c", "echo helper-failure; exit 42")
			if errFail == nil {
				t.Fatal("expected failing helper command to return error, got nil")
			}
			if !strings.Contains(errFail.Error(), "42") {
				t.Errorf("error %q does not contain exit status 42", errFail)
			}
			if !strings.Contains(string(outFail), "helper-failure") && !strings.Contains(errFail.Error(), "helper-failure") {
				t.Errorf("output %q does not contain helper-failure", string(outFail))
			}

			// 2. Started processes (s.StartProcess):
			// Process that runs and exits with status 42:
			startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
				Id:      "exiter",
				Process: "exit-42",
			})
			if err != nil {
				t.Fatalf("StartProcess exit-42: %v", err)
			}
			if startResp.GetStatus().GetPid() == 0 {
				t.Error("StartProcess returned pid 0")
			}

			// Poll until process exits:
			var lastStatus *pb.ProcessStatus
			for range 100 {
				getResp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "exiter"})
				if err == nil {
					lastStatus = getResp.GetStatus()
					if lastStatus.GetState() != "running" {
						break
					}
				}
				time.Sleep(20 * time.Millisecond)
			}
			if lastStatus == nil || lastStatus.GetState() == "running" {
				t.Fatalf("process did not exit: %v", lastStatus)
			}
			if lastStatus.GetExitCode() != 42 {
				t.Errorf("exit code = %d, want 42", lastStatus.GetExitCode())
			}
			if !strings.Contains(lastStatus.GetState(), "42") {
				t.Errorf("state = %q, want exited with code 42", lastStatus.GetState())
			}
			if !strings.Contains(lastStatus.GetOutput(), "standin exiting with 42") {
				t.Errorf("output = %q, want standin exiting with 42", lastStatus.GetOutput())
			}

			// Process that runs and is stopped:
			loopResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
				Id:      "looper",
				Process: "loop",
			})
			if err != nil {
				t.Fatalf("StartProcess loop: %v", err)
			}
			loopPid := int(loopResp.GetStatus().GetPid())

			// Wait until loop is confirmed running:
			for range 100 {
				getResp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "looper"})
				if err == nil && strings.Contains(getResp.GetStatus().GetOutput(), "standin started") {
					break
				}
				time.Sleep(20 * time.Millisecond)
			}

			stopResp, err := s.StopProcess(ctx, &pb.StopProcessRequest{Id: "looper"})
			if err != nil {
				t.Fatalf("StopProcess: %v", err)
			}
			if stopResp == nil {
				t.Error("StopProcess returned nil response")
			}
			if isAlive(loopPid) {
				t.Errorf("process %d still alive after StopProcess", loopPid)
			}
		})
	}
}

func TestStopWithLiveChildren_BothSituations(t *testing.T) {
	for _, asProcess1 := range []bool{false, true} {
		name := "ordinary"
		if asProcess1 {
			name = "process1"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			s := testServer(t)
			if asProcess1 {
				s.forceReaper = true
				s.StartReaper()
			}

			resp1, err := s.StartProcess(ctx, &pb.StartProcessRequest{
				Id:      "child1",
				Process: "loop",
			})
			if err != nil {
				t.Fatalf("StartProcess child1: %v", err)
			}
			pid1 := int(resp1.GetStatus().GetPid())

			resp2, err := s.StartProcess(ctx, &pb.StartProcessRequest{
				Id:      "child2",
				Process: "loop",
			})
			if err != nil {
				t.Fatalf("StartProcess child2: %v", err)
			}
			pid2 := int(resp2.GetStatus().GetPid())

			// Verify both children are running.
			for range 100 {
				r1, err1 := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "child1"})
				r2, err2 := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "child2"})
				if err1 == nil && err2 == nil &&
					strings.Contains(r1.GetStatus().GetOutput(), "standin started") &&
					strings.Contains(r2.GetStatus().GetOutput(), "standin started") {
					break
				}
				time.Sleep(20 * time.Millisecond)
			}

			// Call s.Stop() and ensure it returns promptly without hanging.
			stopDone := make(chan struct{})
			go func() {
				s.Stop()
				close(stopDone)
			}()

			select {
			case <-stopDone:
			case <-time.After(15 * time.Second):
				t.Fatal("s.Stop() hung with live children")
			}

			// Verify both children are dead.
			dead1, dead2 := false, false
			for range 100 {
				if !dead1 && !isAlive(pid1) {
					dead1 = true
				}
				if !dead2 && !isAlive(pid2) {
					dead2 = true
				}
				if dead1 && dead2 {
					break
				}
				time.Sleep(20 * time.Millisecond)
			}
			if !dead1 {
				t.Errorf("child process 1 (pid %d) still alive after s.Stop()", pid1)
			}
			if !dead2 {
				t.Errorf("child process 2 (pid %d) still alive after s.Stop()", pid2)
			}
		})
	}
}

func TestStopWithHelperInFlight_BothSituations(t *testing.T) {
	for _, asProcess1 := range []bool{false, true} {
		name := "ordinary"
		if asProcess1 {
			name = "process1"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			s := testServer(t)
			if asProcess1 {
				s.forceReaper = true
				s.StartReaper()
			}

			helperDone := make(chan struct{})
			var helperOut []byte
			var helperErr error

			go func() {
				defer close(helperDone)
				helperOut, helperErr = s.run(ctx, "sh", "-c", "sleep 0.1; echo helper-done")
			}()

			// Give the helper command time to start and register in the server.
			if asProcess1 {
				for range 100 {
					s.procMu.Lock()
					n := len(s.helperPids)
					s.procMu.Unlock()
					if n > 0 {
						break
					}
					time.Sleep(5 * time.Millisecond)
				}
			} else {
				time.Sleep(20 * time.Millisecond)
			}

			// Stop the server while the helper is in flight.
			stopDone := make(chan struct{})
			go func() {
				s.Stop()
				close(stopDone)
			}()

			select {
			case <-stopDone:
			case <-time.After(15 * time.Second):
				t.Fatal("s.Stop() hung with helper in flight")
			}

			// Verify helper completes cleanly and preserves its exit status and output.
			select {
			case <-helperDone:
				if helperErr != nil {
					t.Fatalf("helper failed: %v (output: %s)", helperErr, helperOut)
				}
				if !strings.Contains(string(helperOut), "helper-done") {
					t.Errorf("helper output = %q, want helper-done", string(helperOut))
				}
			case <-time.After(15 * time.Second):
				t.Fatal("helper command hung after s.Stop()")
			}
		})
	}
}

func TestStopWithHangingHelper_Process1(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	s.forceReaper = true
	s.StartReaper()

	helperDone := make(chan struct{})
	var helperOut []byte
	var helperErr error

	go func() {
		defer close(helperDone)
		helperOut, helperErr = s.run(ctx, "sh", "-c", "sleep 300")
	}()

	// Wait until helper is registered.
	for range 100 {
		s.procMu.Lock()
		n := len(s.helperPids)
		s.procMu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Stop the server. It should kill lingering helper after deadline and return promptly.
	stopDone := make(chan struct{})
	go func() {
		s.Stop()
		close(stopDone)
	}()

	select {
	case <-stopDone:
	case <-time.After(10 * time.Second):
		t.Fatal("s.Stop() hung with hanging helper")
	}

	select {
	case <-helperDone:
		if helperErr == nil {
			t.Errorf("expected hanging helper to fail on stop, got output: %s", helperOut)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("hanging helper command hung after s.Stop()")
	}
}

func TestStageBlobs_ParallelTransfersBoundedAndFasterThanSequential(t *testing.T) {
	ctx := t.Context()
	const (
		numBlobs   = 12
		blobDelay  = 50 * time.Millisecond
		blobLength = 1024
	)

	blobMap := make(map[string][]byte, numBlobs)
	names := make([]string, numBlobs)
	for i := 0; i < numBlobs; i++ {
		content := []byte(fmt.Sprintf("blob-content-payload-%d-%s", i, strings.Repeat("x", blobLength)))
		sha := sha256Hex(content)
		blobMap[sha] = content
		names[i] = sha
	}

	var inFlight atomic.Int32
	var maxInFlight atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := inFlight.Add(1)
		for {
			old := maxInFlight.Load()
			if cur <= old || maxInFlight.CompareAndSwap(old, cur) {
				break
			}
		}
		time.Sleep(blobDelay)
		inFlight.Add(-1)

		sha := strings.TrimPrefix(r.URL.Path, "/blobs/")
		content, ok := blobMap[sha]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	start := time.Now()
	resp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: names,
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("StageBlobs: %v", err)
	}

	// 12 sequential transfers at 50ms each would take at least 600ms.
	// With 8 parallel transfers, 12 blobs take 2 batches (~100-150ms).
	sumOfDelays := time.Duration(numBlobs) * blobDelay
	if elapsed >= sumOfDelays {
		t.Errorf("parallel staging took %v, not faster than sum of delays %v", elapsed, sumOfDelays)
	}

	got := maxInFlight.Load()
	if got <= 1 {
		t.Errorf("expected parallel transfers in flight (> 1), got max %d", got)
	}
	if got > int32(defaultMaxConcurrentTransfers) {
		t.Errorf("exceeded max concurrency bound %d: got %d in flight at once", defaultMaxConcurrentTransfers, got)
	}

	blobs := blobsMap(resp.GetBlobs())
	if len(blobs) != numBlobs {
		t.Fatalf("got %d staged blobs in response, want %d", len(blobs), numBlobs)
	}
	for _, name := range names {
		if blobs[name] != int64(len(blobMap[name])) {
			t.Errorf("blob %s size mismatch: got %d, want %d", name, blobs[name], len(blobMap[name]))
		}
		fi, err := os.Stat(filepath.Join(blobDir, name))
		if err != nil || fi.Size() != int64(len(blobMap[name])) {
			t.Errorf("blob %s file on disk missing or size mismatch: %v", name, err)
		}
	}
}

func TestStageBlobs_CustomConcurrencyBounded(t *testing.T) {
	ctx := t.Context()
	const (
		numBlobs  = 6
		blobDelay = 40 * time.Millisecond
		customMax = 3
	)

	blobMap := make(map[string][]byte, numBlobs)
	names := make([]string, numBlobs)
	for i := 0; i < numBlobs; i++ {
		content := []byte(fmt.Sprintf("custom-concurrency-blob-%d", i))
		sha := sha256Hex(content)
		blobMap[sha] = content
		names[i] = sha
	}

	var inFlight atomic.Int32
	var maxInFlight atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := inFlight.Add(1)
		for {
			old := maxInFlight.Load()
			if cur <= old || maxInFlight.CompareAndSwap(old, cur) {
				break
			}
		}
		time.Sleep(blobDelay)
		inFlight.Add(-1)

		sha := strings.TrimPrefix(r.URL.Path, "/blobs/")
		content, ok := blobMap[sha]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir:                blobDir,
		Store:                  weightstore.New(srv.URL),
		MaxConcurrentTransfers: customMax,
	}

	resp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: names,
	})
	if err != nil {
		t.Fatalf("StageBlobs: %v", err)
	}

	got := maxInFlight.Load()
	if got <= 1 {
		t.Errorf("expected parallel transfers in flight (> 1), got max %d", got)
	}
	if got > int32(customMax) {
		t.Errorf("exceeded custom concurrency bound %d: got %d in flight at once", customMax, got)
	}

	blobs := blobsMap(resp.GetBlobs())
	if len(blobs) != numBlobs {
		t.Fatalf("got %d staged blobs in response, want %d", len(blobs), numBlobs)
	}
}

func TestStageBlobs_FailureReportsFailedBlobAndStopsOthers(t *testing.T) {
	ctx := t.Context()
	contentA := []byte("first good blob")
	shaA := sha256Hex(contentA)
	contentB := []byte("second failing blob")
	shaB := sha256Hex(contentB)
	contentC := []byte("third delayed blob")
	shaC := sha256Hex(contentC)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/blobs/" + shaA:
			w.Header().Set("Content-Length", strconv.Itoa(len(contentA)))
			_, _ = w.Write(contentA)
		case "/blobs/" + shaB:
			// Fail immediately with 404
			http.NotFound(w, r)
		case "/blobs/" + shaC:
			// Slow transfer that should be stopped
			time.Sleep(200 * time.Millisecond)
			w.Header().Set("Content-Length", strconv.Itoa(len(contentC)))
			_, _ = w.Write(contentC)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	_, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{shaA, shaB, shaC},
	})
	if err == nil {
		t.Fatal("expected error from failing blob, got nil")
	}

	// The reply must say which blob failed
	if !strings.Contains(err.Error(), shaB) {
		t.Errorf("expected error to name failing blob %s, got: %v", shaB, err)
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.NotFound {
		t.Errorf("expected NotFound status, got %v", err)
	}

	// The other transfer (shaC) was stopped and must not be placed on disk
	if _, err := os.Stat(filepath.Join(blobDir, shaC)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected stopped blob %s to not exist on disk, got %v", shaC, err)
	}

	// No temporary files left behind
	entries, _ := os.ReadDir(blobDir)
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") {
			t.Errorf("found leftover temp file: %s", entry.Name())
		}
	}
}

func TestStageBlobs_DuplicateNamesInSameRequest(t *testing.T) {
	ctx := t.Context()
	contentA := []byte("blob A duplicate test")
	contentB := []byte("blob B duplicate test")
	shaA := sha256Hex(contentA)
	shaB := sha256Hex(contentB)

	var transfers atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		transfers.Add(1)
		switch r.URL.Path {
		case "/blobs/" + shaA:
			w.Header().Set("Content-Length", strconv.Itoa(len(contentA)))
			_, _ = w.Write(contentA)
		case "/blobs/" + shaB:
			w.Header().Set("Content-Length", strconv.Itoa(len(contentB)))
			_, _ = w.Write(contentB)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	blobDir := t.TempDir()
	server := &Server{
		BlobDir: blobDir,
		Store:   weightstore.New(srv.URL),
	}

	resp, err := server.StageBlobs(ctx, &pb.StageBlobsRequest{
		Names: []string{shaA, shaB, shaA},
	})
	if err != nil {
		t.Fatalf("StageBlobs: %v", err)
	}

	// Each unique blob should be transferred only once
	if got := transfers.Load(); got != 2 {
		t.Errorf("expected 2 store transfers for duplicate names, got %d", got)
	}

	blobs := resp.GetBlobs()
	if len(blobs) != 3 {
		t.Fatalf("got %d blobs in response, want 3", len(blobs))
	}
	if blobs[0].GetName() != shaA || blobs[1].GetName() != shaB || blobs[2].GetName() != shaA {
		t.Errorf("unexpected blob order in response: %+v", blobs)
	}
}
