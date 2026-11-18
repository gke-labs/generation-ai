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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

const (
	// MaxOutputBytes is the maximum byte size of recent output retained and
	// reported in ProcessStatus.
	MaxOutputBytes = 64 * 1024 // 64 KiB
	maxTailLines   = 100
)

var processID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)

// ProcessConfig defines a process configuration that the agent is permitted
// to start.
type ProcessConfig struct {
	// Binary is the executable to run, e.g. "python3".
	Binary string

	// Args are the base arguments for the process, e.g. ["-m", "vxpu.server"].
	Args []string

	// Env contains fixed environment variables (KEY=VAL) for the process.
	Env []string

	// AcceptsPlanFile indicates whether this process takes a plan file (--plan-file).
	AcceptsPlanFile bool

	// Serves indicates whether this process serves network calls and is assigned an endpoint by the agent.
	Serves bool
}

// DefaultProcesses returns the default process configurations.
func DefaultProcesses() map[string]ProcessConfig {
	return map[string]ProcessConfig{
		"executor": {
			Binary: "python3",
			Args:   []string{"-m", "vxpu.server"},
			Serves: true,
		},
		"nodeserver": {
			Binary: "python3",
			Args:   []string{"-m", "vxpu.nodeserver", "serve-echo"},
		},
		"compiler": {
			Binary: "python3",
			Args:   []string{"-m", "vxpu.compile_server"},
			Serves: true,
		},
	}
}

// processManager manages started processes, helper commands, and orphan reaping.
type processManager struct {
	// workRoot is the base directory under which process working directories
	// and plan files are resolved.
	workRoot string

	// host is the hostname or IP address published in process endpoints.
	host string

	// configs defines the permitted processes by name.
	configs map[string]ProcessConfig

	// allocatePort finds an unused port. Defaults to defaultAllocatePortLocked.
	allocatePort func() (int, error)

	// mu synchronizes access to the processes table and reaper lifecycle.
	mu sync.Mutex

	// processes tracks child processes started directly by this agent, keyed by process ID.
	processes map[string]*trackedProcess

	// pids maps running process PIDs to their trackedProcess for exit status dispatch.
	pids map[int]*trackedProcess

	// helperPids maps running helper command PIDs to channels receiving their exit status.
	helperPids map[int]chan syscall.WaitStatus

	// reaperCancel cancels the background orphan reaper.
	reaperCancel context.CancelFunc

	// reaperDone is closed when the background orphan reaper exits.
	reaperDone chan struct{}

	// reaperActive indicates whether the unified PID 1 reaper is running.
	reaperActive bool

	// forceReaper enables the orphan reaper in tests even when PID is not 1.
	forceReaper bool
}

func newProcessManager(workRoot, host string, configs map[string]ProcessConfig) *processManager {
	return &processManager{
		workRoot:   workRoot,
		host:       host,
		configs:    configs,
		processes:  make(map[string]*trackedProcess),
		pids:       make(map[int]*trackedProcess),
		helperPids: make(map[int]chan syscall.WaitStatus),
	}
}

// StartProcess starts a process with typed settings.
func (m *processManager) StartProcess(ctx context.Context, req *pb.StartProcessRequest) (*pb.StartProcessResponse, error) {
	id := req.GetId()
	if !processID.MatchString(id) {
		return nil, status.Errorf(codes.InvalidArgument, "process id %q may contain only lowercase letters, digits and dashes", id)
	}

	cfg, err := m.lookupProcess(req.GetProcess())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	if req.Card != nil && req.GetCard() < 0 {
		return nil, status.Errorf(codes.InvalidArgument, "card %d must be >= 0", req.GetCard())
	}
	if req.GetPlanFile() != "" && !cfg.AcceptsPlanFile {
		return nil, status.Errorf(codes.InvalidArgument, "process %q does not accept plan_file", req.GetProcess())
	}
	if cfg.Serves {
		for _, a := range cfg.Args {
			if a == "--port" || strings.HasPrefix(a, "--port=") {
				return nil, status.Errorf(codes.InvalidArgument, "configured process %q carries --port in its arguments; the agent adds --port itself", req.GetProcess())
			}
			if a == "--host" || strings.HasPrefix(a, "--host=") {
				return nil, status.Errorf(codes.InvalidArgument, "configured process %q carries --host in its arguments; the agent adds --host itself", req.GetProcess())
			}
		}
		if m.host == "" {
			return nil, status.Errorf(codes.FailedPrecondition, "cannot start serving process %q: agent host is unknown", id)
		}
	}

	workRoot := m.workRoot
	if workRoot == "" {
		workRoot = "."
	}
	absWorkRoot, err := filepath.Abs(workRoot)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "resolving work root: %v", err)
	}

	var planPath string
	if req.GetPlanFile() != "" {
		if filepath.IsAbs(req.GetPlanFile()) {
			return nil, status.Errorf(codes.InvalidArgument, "plan_file %q must be a relative path", req.GetPlanFile())
		}
		target := filepath.Join(absWorkRoot, req.GetPlanFile())
		resolvedTarget, err := filepath.EvalSymlinks(target)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "plan_file %q: %v", req.GetPlanFile(), err)
		}
		resolvedRoot, err := filepath.EvalSymlinks(absWorkRoot)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "resolving work root: %v", err)
		}
		rel, err := filepath.Rel(resolvedRoot, resolvedTarget)
		if err != nil || strings.HasPrefix(rel, "..") || rel == ".." {
			return nil, status.Errorf(codes.InvalidArgument, "plan_file %q escapes work root", req.GetPlanFile())
		}
		fi, err := os.Stat(resolvedTarget)
		if err != nil || fi.IsDir() {
			return nil, status.Errorf(codes.InvalidArgument, "plan_file %q does not exist or is a directory", req.GetPlanFile())
		}
		planPath = resolvedTarget
	}

	runDir := filepath.Join(absWorkRoot, "processes", id)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, status.Errorf(codes.Internal, "creating process directory: %v", err)
	}

	m.mu.Lock()
	if m.processes == nil {
		m.processes = make(map[string]*trackedProcess)
	}
	if existing, ok := m.processes[id]; ok {
		if existing.isStopping() {
			m.mu.Unlock()
			return nil, status.Errorf(codes.Unavailable, "process %q is stopping", id)
		}
		if existing.isRunning() {
			m.mu.Unlock()
			if !sameConfig(existing.req, req) {
				return nil, status.Errorf(codes.AlreadyExists, "process %q is already running with different settings", id)
			}
			return &pb.StartProcessResponse{
				AlreadyRunning: true,
				Status:         existing.status(),
			}, nil
		}
	}

	var assignedPort int
	var endpoint string
	args := append([]string(nil), cfg.Args...)
	if cfg.Serves {
		p, err := m.allocatePortLocked()
		if err != nil {
			m.mu.Unlock()
			return nil, status.Errorf(codes.ResourceExhausted, "assigning port for process %q: %v", id, err)
		}
		assignedPort = p
		endpoint = net.JoinHostPort(m.host, strconv.Itoa(assignedPort))
		args = append(args, "--host", m.host, "--port", strconv.Itoa(assignedPort))
	}
	if planPath != "" {
		args = append(args, "--plan-file", planPath)
	}
	if req.GetKeepAliveSeconds() > 0 {
		args = append(args, "--keep-alive", strconv.Itoa(int(req.GetKeepAliveSeconds())))
	}

	cmd := exec.Command(cfg.Binary, args...)
	cmd.Dir = runDir
	env := os.Environ()
	env = append(env, cfg.Env...)
	if req.Card != nil {
		env = append(env, fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", req.GetCard()))
	}
	cmd.Env = env
	// Run each process in its own process group so StopProcess signals all child
	// processes spawned by the command. We avoid Linux Pdeathsig because Go's
	// runtime can retire OS threads, which would prematurely kill healthy processes.
	// A child process that explicitly calls setsid leaves this process group.
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
	// Bound waiting for stdout/stderr pipes after process exit so surviving
	// children holding open pipes cannot block StopProcess or Server.Stop.
	cmd.WaitDelay = 2 * time.Second

	tb := newTailBuffer(maxTailLines, MaxOutputBytes)
	cmd.Stdout = io.MultiWriter(tb, os.Stdout)
	cmd.Stderr = io.MultiWriter(tb, os.Stderr)

	if err := cmd.Start(); err != nil {
		m.mu.Unlock()
		return nil, status.Errorf(codes.Internal, "starting process %q: %v", id, err)
	}

	proc := &trackedProcess{
		id:       id,
		process:  req.GetProcess(),
		req:      req,
		cmd:      cmd,
		output:   tb,
		done:     make(chan struct{}),
		state:    "running",
		pid:      cmd.Process.Pid,
		port:     assignedPort,
		endpoint: endpoint,
	}
	m.processes[id] = proc
	if m.pids == nil {
		m.pids = make(map[int]*trackedProcess)
	}
	m.pids[proc.pid] = proc
	reaperActive := m.reaperActive
	m.mu.Unlock()

	if !reaperActive {
		go proc.wait()
	}

	klog.FromContext(ctx).Info("Started process", "id", id, "pid", proc.pid, "process", req.GetProcess(), "endpoint", endpoint)
	return &pb.StartProcessResponse{
		AlreadyRunning: false,
		Status:         proc.status(),
	}, nil
}

// StopProcess stops a running process and removes it from tracking.
func (m *processManager) StopProcess(ctx context.Context, req *pb.StopProcessRequest) (*pb.StopProcessResponse, error) {
	id := req.GetId()
	m.mu.Lock()
	proc, ok := m.processes[id]
	if !ok {
		m.mu.Unlock()
		return &pb.StopProcessResponse{}, nil
	}
	if proc.isStopping() {
		m.mu.Unlock()
		select {
		case <-proc.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return &pb.StopProcessResponse{}, nil
	}
	if !proc.isRunning() {
		delete(m.processes, id)
		m.mu.Unlock()
		return &pb.StopProcessResponse{}, nil
	}
	proc.setStopping()
	m.mu.Unlock()

	proc.stop(ctx)

	m.mu.Lock()
	if current, ok := m.processes[id]; ok && current == proc {
		delete(m.processes, id)
	}
	m.mu.Unlock()

	klog.FromContext(ctx).Info("Stopped process", "id", id)
	return &pb.StopProcessResponse{}, nil
}

// GetProcess reports the status of a tracked process.
func (m *processManager) GetProcess(ctx context.Context, req *pb.GetProcessRequest) (*pb.GetProcessResponse, error) {
	id := req.GetId()
	m.mu.Lock()
	proc, ok := m.processes[id]
	m.mu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "process %q not found", id)
	}
	return &pb.GetProcessResponse{Status: proc.status()}, nil
}

// ListProcesses lists all tracked processes in sorted order by id.
func (m *processManager) ListProcesses(ctx context.Context, req *pb.ListProcessesRequest) (*pb.ListProcessesResponse, error) {
	m.mu.Lock()
	list := make([]*pb.ProcessStatus, 0, len(m.processes))
	for _, p := range m.processes {
		list = append(list, p.status())
	}
	m.mu.Unlock()

	sort.Slice(list, func(i, j int) bool {
		return list[i].Id < list[j].Id
	})
	return &pb.ListProcessesResponse{Processes: list}, nil
}

// Stop terminates all running processes and waits for in-flight helper commands.
func (m *processManager) Stop() {
	ctx := context.Background()
	m.mu.Lock()
	procs := make([]*trackedProcess, 0, len(m.processes))
	for _, p := range m.processes {
		procs = append(procs, p)
	}
	m.mu.Unlock()

	for _, p := range procs {
		p.stop(ctx)
	}

	// Wait for any in-flight helper commands to complete while the reaper is still running.
	deadline := time.Now().Add(2 * time.Second)
	for {
		m.mu.Lock()
		n := len(m.helperPids)
		m.mu.Unlock()
		if n == 0 {
			break
		}
		if time.Now().After(deadline) {
			m.mu.Lock()
			for pid := range m.helperPids {
				if err := syscall.Kill(pid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
					klog.FromContext(ctx).V(2).Info("killing lingering helper on stop", "pid", pid, "error", err)
				}
			}
			m.mu.Unlock()
		}
		time.Sleep(10 * time.Millisecond)
	}

	m.mu.Lock()
	if m.reaperCancel != nil {
		m.reaperCancel()
		m.reaperCancel = nil
	}
	m.reaperActive = false
	m.processes = make(map[string]*trackedProcess)
	m.pids = make(map[int]*trackedProcess)
	m.mu.Unlock()
}

// StartReaper starts background orphan child reaping when the agent is PID 1.
// In tests, forceReaper can enable it.
func (m *processManager) StartReaper() {
	if os.Getpid() != 1 && !m.forceReaper {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.processes == nil {
		m.processes = make(map[string]*trackedProcess)
	}
	if m.reaperCancel != nil {
		return
	}
	m.reaperActive = true
	reaperCtx, cancel := context.WithCancel(context.Background())
	m.reaperCancel = cancel
	m.reaperDone = make(chan struct{})
	go m.runReaper(reaperCtx)
}

func (m *processManager) runReaper(ctx context.Context) {
	defer close(m.reaperDone)
	sigCh := make(chan os.Signal, 10)
	signal.Notify(sigCh, syscall.SIGCHLD)
	defer signal.Stop(sigCh)

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			m.reapAll()
			return
		case <-sigCh:
			m.reapAll()
		case <-ticker.C:
			m.reapAll()
		}
	}
}

// ReapOrphans collects exit statuses of all exited child processes.
func (m *processManager) ReapOrphans() {
	m.reapAll()
}

func (m *processManager) reapAll() {
	for {
		var ws syscall.WaitStatus
		pid, err := syscall.Wait4(-1, &ws, syscall.WNOHANG, nil)
		if pid <= 0 || err != nil {
			break
		}
		m.mu.Lock()
		if ch, ok := m.helperPids[pid]; ok {
			delete(m.helperPids, pid)
			ch <- ws
		} else if proc, ok := m.pids[pid]; ok {
			delete(m.pids, pid)
			proc.recordExit(ws)
		}
		m.mu.Unlock()
	}
}

// allocatePortLocked finds an unused TCP port on loopback that is not assigned
// to any currently running process on this manager. The caller must hold m.mu.
func (m *processManager) allocatePortLocked() (int, error) {
	if m.allocatePort != nil {
		return m.allocatePort()
	}
	for range 50 {
		lis, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return 0, fmt.Errorf("finding free port: %w", err)
		}
		port := lis.Addr().(*net.TCPAddr).Port
		if cerr := lis.Close(); cerr != nil {
			return 0, fmt.Errorf("closing port listener: %w", cerr)
		}

		inUse := false
		for _, p := range m.processes {
			if p.isRunning() && p.port == port {
				inUse = true
				break
			}
		}
		if !inUse {
			return port, nil
		}
	}
	return 0, fmt.Errorf("exhausted attempts to allocate an unused port")
}

func (m *processManager) lookupProcess(name string) (ProcessConfig, error) {
	if name == "" {
		return ProcessConfig{}, fmt.Errorf("process name cannot be empty")
	}
	procs := m.configs
	if len(procs) == 0 {
		procs = DefaultProcesses()
	}
	cfg, ok := procs[name]
	if !ok {
		return ProcessConfig{}, fmt.Errorf("process %q is not configured on this agent", name)
	}
	return cfg, nil
}

func (m *processManager) run(ctx context.Context, name string, args ...string) ([]byte, error) {
	m.mu.Lock()
	if !m.reaperActive {
		m.mu.Unlock()
		out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
		if err != nil {
			return out, fmt.Errorf("%s: %w: %s", name, err, out)
		}
		return out, nil
	}

	reaperDone := m.reaperDone
	pr, pw, err := os.Pipe()
	if err != nil {
		m.mu.Unlock()
		return nil, fmt.Errorf("%s: %w", name, err)
	}

	cmd := exec.Command(name, args...)
	cmd.Stdout = pw
	cmd.Stderr = pw

	done := make(chan syscall.WaitStatus, 1)

	if err := cmd.Start(); err != nil {
		m.mu.Unlock()
		if cerr := pw.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing pipe writer on start error", "error", cerr)
		}
		if cerr := pr.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing pipe reader on start error", "error", cerr)
		}
		return nil, fmt.Errorf("%s: %w", name, err)
	}

	pid := cmd.Process.Pid
	if m.helperPids == nil {
		m.helperPids = make(map[int]chan syscall.WaitStatus)
	}
	m.helperPids[pid] = done
	m.mu.Unlock()

	// Close write end in parent process so read end sees EOF when child exits.
	if cerr := pw.Close(); cerr != nil {
		klog.FromContext(ctx).V(2).Info("closing pipe writer in parent", "error", cerr)
	}

	var output []byte
	var readErr error
	readDone := make(chan struct{})
	go func() {
		output, readErr = io.ReadAll(pr)
		if cerr := pr.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing pipe reader", "error", cerr)
		}
		close(readDone)
	}()

	defer func() {
		m.mu.Lock()
		delete(m.helperPids, pid)
		m.mu.Unlock()
	}()

	var ws syscall.WaitStatus
	select {
	case <-ctx.Done():
		if cmd.Process != nil {
			if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				klog.FromContext(ctx).V(2).Info("killing helper process on cancel", "error", err)
			}
		}
		if cerr := pr.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing pipe reader on cancel", "error", cerr)
		}
		<-readDone
		return nil, ctx.Err()
	case ws = <-done:
	case <-reaperDone:
		select {
		case ws = <-done:
		default:
			m.mu.Lock()
			delete(m.helperPids, pid)
			m.mu.Unlock()

			for {
				var status syscall.WaitStatus
				wpid, err := syscall.Wait4(pid, &status, 0, nil)
				if wpid == pid {
					ws = status
					break
				}
				if errors.Is(err, syscall.ECHILD) {
					break
				}
				if err == syscall.EINTR {
					continue
				}
				if err != nil {
					klog.FromContext(ctx).V(2).Info("fallback wait4 error", "pid", pid, "error", err)
					break
				}
			}
		}
	}

	// Wait for pipe reading to finish.
	// Since child has exited and parent closed pw, EOF should arrive promptly.
	// We give a generous bounded wait in case any descendant inherited the pipe.
	select {
	case <-readDone:
	case <-time.After(2 * time.Second):
		if cerr := pr.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing pipe reader on timeout", "error", cerr)
		}
		<-readDone
	}

	if readErr != nil && !errors.Is(readErr, os.ErrClosed) && !errors.Is(readErr, io.EOF) {
		klog.FromContext(ctx).V(2).Info("reading helper output", "error", readErr)
	}

	if !ws.Exited() || ws.ExitStatus() != 0 {
		if ws.Signaled() {
			return output, fmt.Errorf("%s: signal %d: %s", name, ws.Signal(), output)
		}
		return output, fmt.Errorf("%s: exit status %d: %s", name, ws.ExitStatus(), output)
	}
	return output, nil
}

func sameConfig(a, b *pb.StartProcessRequest) bool {
	sameCard := (a.Card == nil && b.Card == nil) || (a.Card != nil && b.Card != nil && *a.Card == *b.Card)
	sameKeepAlive := (a.KeepAliveSeconds == nil && b.KeepAliveSeconds == nil) ||
		(a.KeepAliveSeconds != nil && b.KeepAliveSeconds != nil && *a.KeepAliveSeconds == *b.KeepAliveSeconds)
	return a.GetProcess() == b.GetProcess() &&
		sameCard &&
		a.GetPlanFile() == b.GetPlanFile() &&
		sameKeepAlive
}

type trackedProcess struct {
	// id is the unique process ID assigned to this task.
	id string

	// process is the configured process name (e.g. "loop").
	process string

	// req is the StartProcessRequest used to launch this process.
	req *pb.StartProcessRequest

	// cmd is the underlying exec.Cmd for the running process.
	cmd *exec.Cmd

	// output captures recent standard output and error.
	output *tailBuffer

	// done is closed when the process exits.
	done chan struct{}

	// mu synchronizes access to mutable state fields.
	mu sync.Mutex

	// state is the human-readable process state.
	state string

	// stopping indicates whether a stop has been requested.
	stopping bool

	// pid is the OS process ID.
	pid int

	// port is the port the process serves on, or 0 if not serving.
	port int

	// endpoint is the host and port published while the process is running.
	endpoint string

	// exitCode is the process exit code once terminated.
	exitCode int

	// signal is the terminating signal if killed.
	signal int
}

func (p *trackedProcess) isRunning() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.state == "running"
}

func (p *trackedProcess) isStopping() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stopping
}

func (p *trackedProcess) setStopping() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.stopping = true
	p.state = "stopping"
}

func (p *trackedProcess) status() *pb.ProcessStatus {
	p.mu.Lock()
	defer p.mu.Unlock()
	endpoint := ""
	if p.state == "running" {
		endpoint = p.endpoint
	}
	return &pb.ProcessStatus{
		Id:       p.id,
		Pid:      int32(p.pid),
		Process:  p.process,
		State:    p.state,
		Output:   p.output.String(),
		ExitCode: int32(p.exitCode),
		Signal:   int32(p.signal),
		Endpoint: endpoint,
	}
}

func (p *trackedProcess) recordExit(ws syscall.WaitStatus) {
	p.mu.Lock()
	if ws.Signaled() {
		p.signal = int(ws.Signal())
		if !p.stopping {
			p.state = fmt.Sprintf("killed by signal %d", ws.Signal())
		}
	} else {
		p.exitCode = ws.ExitStatus()
		if !p.stopping {
			p.state = fmt.Sprintf("exited with code %d", ws.ExitStatus())
		}
	}
	p.mu.Unlock()
	close(p.done)
	go func() {
		if err := p.cmd.Wait(); err != nil && !errors.Is(err, syscall.ECHILD) {
			klog.V(2).Info("reaped process wait cleanup", "id", p.id, "error", err)
		}
	}()
}

func (p *trackedProcess) wait() {
	err := p.cmd.Wait()
	p.mu.Lock()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			if ws, ok := exitErr.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
				p.signal = int(ws.Signal())
				if !p.stopping {
					p.state = fmt.Sprintf("killed by signal %d", ws.Signal())
				}
			} else {
				p.exitCode = exitErr.ExitCode()
				if !p.stopping {
					p.state = fmt.Sprintf("exited with code %d", exitErr.ExitCode())
				}
			}
		} else {
			p.exitCode = -1
			if !p.stopping {
				p.state = err.Error()
			}
		}
	} else {
		p.exitCode = 0
		if !p.stopping {
			p.state = "exited with code 0"
		}
	}
	p.mu.Unlock()
	close(p.done)
}

func (p *trackedProcess) stop(ctx context.Context) {
	p.mu.Lock()
	p.stopping = true
	p.state = "stopping"
	p.mu.Unlock()

	if p.cmd.Process == nil {
		return
	}

	pgid := p.cmd.Process.Pid
	if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
		klog.FromContext(ctx).V(2).Info("Signaling process group", "id", p.id, "pgid", pgid, "error", err)
	}

	select {
	case <-p.done:
		return
	case <-time.After(2 * time.Second):
		if err := syscall.Kill(-pgid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
			klog.FromContext(ctx).V(2).Info("Killing process group", "id", p.id, "pgid", pgid, "error", err)
		}
		<-p.done
	}
}

type tailBuffer struct {
	// mu synchronizes access to buffer contents.
	mu sync.Mutex

	// lines holds recently accumulated complete output lines.
	lines []string

	// partial holds incomplete bytes for the current line.
	partial []byte

	// maxLines is the maximum number of lines to retain.
	maxLines int

	// maxBytes is the maximum total bytes to retain.
	maxBytes int

	// curBytes tracks the current byte count of retained lines and partial data.
	curBytes int
}

func newTailBuffer(maxLines, maxBytes int) *tailBuffer {
	return &tailBuffer{
		maxLines: maxLines,
		maxBytes: maxBytes,
	}
}

func (b *tailBuffer) Write(p []byte) (n int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n = len(p)

	for len(p) > 0 {
		idx := bytes.IndexAny(p, "\n\r")
		if idx >= 0 {
			chunk := p[:idx]
			p = p[idx+1:]
			b.partial = append(b.partial, chunk...)
			if len(b.partial) > b.maxBytes {
				b.partial = b.partial[len(b.partial)-b.maxBytes:]
			}
			line := string(b.partial)
			b.lines = append(b.lines, line)
			b.curBytes += len(line) + 1
			b.partial = b.partial[:0]
			for len(b.lines) > b.maxLines || b.curBytes > b.maxBytes {
				dropped := b.lines[0]
				b.lines = b.lines[1:]
				b.curBytes -= len(dropped) + 1
			}
		} else {
			if len(p) > b.maxBytes {
				p = p[len(p)-b.maxBytes:]
			}
			b.partial = append(b.partial, p...)
			for b.curBytes+len(b.partial) > b.maxBytes && len(b.lines) > 0 {
				dropped := b.lines[0]
				b.lines = b.lines[1:]
				b.curBytes -= len(dropped) + 1
			}
			if len(b.partial) > b.maxBytes {
				b.partial = b.partial[len(b.partial)-b.maxBytes:]
			}
			break
		}
	}
	return n, nil
}

func (b *tailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.lines) == 0 && len(b.partial) == 0 {
		return ""
	}
	var out []string
	if len(b.lines) > 0 {
		out = append(out, b.lines...)
	}
	if len(b.partial) > 0 {
		out = append(out, string(b.partial))
	}
	s := strings.Join(out, "\n")
	if len(s) > b.maxBytes {
		s = s[len(s)-b.maxBytes:]
	}
	return s
}
