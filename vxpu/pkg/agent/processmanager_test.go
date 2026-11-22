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
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
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
		case "serve":
			port := 0
			host := "127.0.0.1"
			for i := 2; i < len(os.Args)-1; i++ {
				if os.Args[i] == "--port" {
					p, err := strconv.Atoi(os.Args[i+1])
					if err == nil {
						port = p
					}
				}
				if os.Args[i] == "--host" {
					host = os.Args[i+1]
				}
			}
			if port == 0 {
				fmt.Fprintln(os.Stderr, "serve: missing or invalid --port")
				os.Exit(2)
			}
			lis, err := net.Listen("tcp", fmt.Sprintf("%s:%d", host, port))
			if err != nil {
				fmt.Fprintf(os.Stderr, "serve: bind error on port %d: %v\n", port, err)
				os.Exit(1)
			}
			defer lis.Close()
			fmt.Printf("serve: listening on %s:%d\n", host, port)
			for {
				conn, err := lis.Accept()
				if err != nil {
					return
				}
				conn.Write([]byte("ok\n"))
				conn.Close()
			}
		}
		os.Exit(0)
	}
}

func testServer(t *testing.T) *Server {
	workDir := t.TempDir()
	return &Server{
		WorkRoot: workDir,
		Host:     "127.0.0.1",
		Processes: map[string]ProcessConfig{
			"serve": {
				Binary: os.Args[0],
				Args:   []string{"serve"},
				Env:    []string{"TEST_STANDIN=1"},
				Serves: true,
			},
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

func TestProcessLifecycle(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "worker",
		Process: "loop",
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
	card := int32(1)
	diffReq := &pb.StartProcessRequest{
		Id:      "r1",
		Process: "loop",
		Card:    &card,
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

	for _, id := range []string{"rank-0", "rank-1"} {
		_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
			Id:      id,
			Process: "loop",
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
	pm := s.procMgr()
	pm.mu.Lock()
	proc := pm.processes["stopping-check"]
	proc.setStopping()
	pm.mu.Unlock()

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

	// 4. Invalid card numbers.
	badCard := int32(-2)
	_, err = s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "test-card",
		Process: "loop",
		Card:    &badCard,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("card -2: got %v, want InvalidArgument", err)
	}

	// 5. Invalid IDs.
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
					pm := s.procMgr()
					pm.mu.Lock()
					n := len(pm.helperPids)
					pm.mu.Unlock()
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
		pm := s.procMgr()
		pm.mu.Lock()
		n := len(pm.helperPids)
		pm.mu.Unlock()
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

func TestServingProcessesEndpoints(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	// Start two serving processes on one agent.
	resp1, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "server-1",
		Process: "serve",
	})
	if err != nil {
		t.Fatalf("start server-1: %v", err)
	}

	resp2, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "server-2",
		Process: "serve",
	})
	if err != nil {
		t.Fatalf("start server-2: %v", err)
	}

	endpoint1 := resp1.GetStatus().GetEndpoint()
	endpoint2 := resp2.GetStatus().GetEndpoint()
	if endpoint1 == "" {
		t.Fatal("server-1 returned empty endpoint")
	}
	if endpoint2 == "" {
		t.Fatal("server-2 returned empty endpoint")
	}
	if endpoint1 == endpoint2 {
		t.Fatalf("both processes assigned identical endpoint %q", endpoint1)
	}

	// Both endpoints must be reachable.
	dialWithRetry := func(endpoint string) {
		deadline := time.Now().Add(5 * time.Second)
		for {
			conn, err := net.Dial("tcp", endpoint)
			if err == nil {
				buf := make([]byte, 16)
				n, _ := conn.Read(buf)
				_ = conn.Close()
				if string(buf[:n]) == "ok\n" {
					return
				}
			}
			if time.Now().After(deadline) {
				t.Fatalf("endpoint %s was not reachable before deadline: %v", endpoint, err)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	dialWithRetry(endpoint1)
	dialWithRetry(endpoint2)

	// Status contains the endpoint while the processes are running.
	st1, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "server-1"})
	if err != nil {
		t.Fatalf("GetProcess server-1: %v", err)
	}
	if st1.GetStatus().GetEndpoint() != endpoint1 {
		t.Errorf("server-1 status endpoint = %q, want %q", st1.GetStatus().GetEndpoint(), endpoint1)
	}
	st2, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "server-2"})
	if err != nil {
		t.Fatalf("GetProcess server-2: %v", err)
	}
	if st2.GetStatus().GetEndpoint() != endpoint2 {
		t.Errorf("server-2 status endpoint = %q, want %q", st2.GetStatus().GetEndpoint(), endpoint2)
	}

	// Kill server-1 directly so it exits without being removed by StopProcess.
	pid1 := int(resp1.GetStatus().GetPid())
	if err := syscall.Kill(pid1, syscall.SIGTERM); err != nil {
		t.Fatalf("killing server-1: %v", err)
	}

	// Wait until server-1 has exited.
	var lastStatus *pb.ProcessStatus
	for range 100 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "server-1"})
		if err == nil {
			lastStatus = resp.GetStatus()
			if lastStatus.GetState() != "running" {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if lastStatus == nil || lastStatus.GetState() == "running" {
		t.Fatal("server-1 did not exit")
	}

	// Once exited, server-1 must no longer publish its endpoint.
	if lastStatus.GetEndpoint() != "" {
		t.Errorf("exited server-1 status has endpoint %q, want empty", lastStatus.GetEndpoint())
	}

	// server-2 must still be running with its endpoint published.
	st2Running, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "server-2"})
	if err != nil {
		t.Fatalf("GetProcess server-2: %v", err)
	}
	if st2Running.GetStatus().GetState() != "running" {
		t.Errorf("server-2 state = %q, want running", st2Running.GetStatus().GetState())
	}
	if st2Running.GetStatus().GetEndpoint() != endpoint2 {
		t.Errorf("server-2 endpoint = %q, want %q", st2Running.GetStatus().GetEndpoint(), endpoint2)
	}
}

func TestProcessCannotBindPortFailsToStart(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	// Occupy a port with a test listener so any process trying to bind it will fail.
	blocker, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("opening blocker listener: %v", err)
	}
	defer func() {
		if cerr := blocker.Close(); cerr != nil {
			t.Logf("closing blocker: %v", cerr)
		}
	}()
	busyPort := blocker.Addr().(*net.TCPAddr).Port

	// Force the agent to allocate the occupied port for the process.
	s.allocatePort = func() (int, error) {
		return busyPort, nil
	}

	startResp, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "cant-bind",
		Process: "serve",
	})
	if err != nil {
		t.Fatalf("StartProcess: %v", err)
	}
	origPid := startResp.GetStatus().GetPid()

	// Wait until the process exits due to failure to bind.
	var lastStatus *pb.ProcessStatus
	for range 100 {
		resp, err := s.GetProcess(ctx, &pb.GetProcessRequest{Id: "cant-bind"})
		if err == nil {
			lastStatus = resp.GetStatus()
			if lastStatus.GetState() != "running" {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}

	if lastStatus == nil || lastStatus.GetState() == "running" {
		t.Fatal("process did not exit on bind failure")
	}

	// Verify the process table records that the process exited with its error and output.
	if lastStatus.GetExitCode() != 1 {
		t.Errorf("exit code = %d, want 1", lastStatus.GetExitCode())
	}
	if !strings.Contains(lastStatus.GetOutput(), "bind error on port") {
		t.Errorf("output %q does not contain bind error message", lastStatus.GetOutput())
	}
	if lastStatus.GetEndpoint() != "" {
		t.Errorf("endpoint = %q, want empty (cleared on exit)", lastStatus.GetEndpoint())
	}
	if lastStatus.GetPid() != origPid {
		t.Errorf("pid changed from %d to %d (unexpected silent retry)", origPid, lastStatus.GetPid())
	}
}

func TestServingProcessFailsWhenHostUnknown(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	s.Host = ""
	defer s.Stop()

	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "no-host",
		Process: "serve",
	})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("got %v, want FailedPrecondition when host is unknown", err)
	}
}

func TestServingProcessRejectsConfiguredPortArg(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	s.Processes["has-port-arg"] = ProcessConfig{
		Binary: os.Args[0],
		Args:   []string{"serve", "--port", "50051"},
		Env:    []string{"TEST_STANDIN=1"},
		Serves: true,
	}

	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "bad-cfg",
		Process: "has-port-arg",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("got %v, want InvalidArgument when configured process has --port", err)
	}
}

func TestServingProcessRejectsConfiguredHostArg(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	s.Processes["has-host-arg"] = ProcessConfig{
		Binary: os.Args[0],
		Args:   []string{"serve", "--host", "127.0.0.1"},
		Env:    []string{"TEST_STANDIN=1"},
		Serves: true,
	}

	_, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "bad-host-cfg",
		Process: "has-host-arg",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("got %v, want InvalidArgument when configured process has --host", err)
	}
}

func TestServingProcessAlreadyRunning(t *testing.T) {
	ctx := t.Context()
	s := testServer(t)
	defer s.Stop()

	resp1, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "srv-match",
		Process: "serve",
	})
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	resp2, err := s.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      "srv-match",
		Process: "serve",
	})
	if err != nil {
		t.Fatalf("second start: %v", err)
	}
	if !resp2.GetAlreadyRunning() {
		t.Error("second start did not report already_running")
	}
	if resp2.GetStatus().GetEndpoint() != resp1.GetStatus().GetEndpoint() {
		t.Errorf("endpoint mismatch: %q vs %q", resp2.GetStatus().GetEndpoint(), resp1.GetStatus().GetEndpoint())
	}
}
