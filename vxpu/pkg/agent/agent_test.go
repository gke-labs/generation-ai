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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
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
