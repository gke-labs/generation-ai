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

// Package agent is the per-machine agent (#384). The agent is a gRPC server
// that runs as root on every machine the coordinator uses, and does the work
// that needs the machine itself.
//
// It mounts weights disks (#407), supervises processes (#657), and reports
// what its machine is (#663). A weights disk holds a checkpoint's files, in
// the same layout as the executor's local weight tier. The coordinator
// attaches the disk read-only, the agent mounts it, and the executor reads
// the files in place instead of copying them over the network.
//
// The agent also launches configured processes, tracks their lifecycle,
// captures recent output, and reports their exit status to the coordinator.
package agent

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
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
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
)

var (
	deviceName = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}$`)
	processID  = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)
)

const (
	// MaxOutputBytes is the maximum byte size of recent output retained and
	// reported in ProcessStatus.
	MaxOutputBytes = 64 * 1024 // 64 KiB
	maxTailLines   = 100
)

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
}

// Server implements the Agent service.
type Server struct {
	pb.UnimplementedAgentServer
	// MountRoot holds one directory per mounted weights disk, named by its
	// device name.
	MountRoot string
	// WorkRoot is the base directory under which process working directories
	// and plan files must reside.
	WorkRoot string

	// SysRoot is the root directory for reading system facts (/proc, /sys, /dev).
	// When empty, "/" is used. Tests set it to point to synthetic system facts.
	SysRoot string

	// Getenv reads an environment variable. When nil, os.Getenv is used.
	Getenv func(key string) string

	// MetadataFunc queries GCE instance metadata paths. When nil, an HTTP
	// request to the GCE metadata server is performed.
	MetadataFunc func(ctx context.Context, path string) (string, error)

	// Processes defines the permitted processes by name.
	// When nil or empty, DefaultProcesses() is used.
	Processes map[string]ProcessConfig

	// Mount mounts device read-only at dir, and Unmount unmounts dir. When
	// they're nil, the server uses mount(8) and umount(8). Tests replace
	// them.
	Mount   func(ctx context.Context, device, dir string) error
	Unmount func(ctx context.Context, dir string) error
	// MountPoint reports whether dir has a filesystem mounted on it. When
	// it's nil, the server compares dir's device with its parent's. Tests
	// replace it along with Mount.
	MountPoint func(dir string) bool

	// one mount or unmount at a time
	mu sync.Mutex

	procMu    sync.Mutex
	processes map[string]*trackedProcess
}

// DefaultProcesses returns the default process configurations.
func DefaultProcesses() map[string]ProcessConfig {
	return map[string]ProcessConfig{
		"executor": {
			Binary: "python3",
			Args:   []string{"-m", "vxpu.server"},
		},
		"nodeserver": {
			Binary: "python3",
			Args:   []string{"-m", "vxpu.nodeserver", "serve-echo"},
		},
	}
}

func (s *Server) MountWeights(ctx context.Context, req *pb.MountWeightsRequest) (*pb.MountWeightsResponse, error) {
	log := klog.FromContext(ctx)
	if !deviceName.MatchString(req.GetDeviceName()) {
		return nil, status.Errorf(codes.InvalidArgument, "device name %q may contain only lowercase letters, digits and dashes", req.GetDeviceName())
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	start := time.Now()
	device := filepath.Join("/dev/disk/by-id", "google-"+req.GetDeviceName())
	dir := filepath.Join(s.MountRoot, req.GetDeviceName())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, status.Errorf(codes.Internal, "making the mount directory: %v", err)
	}
	mountedHere := false
	if !s.mounted(dir) {
		mount := s.Mount
		if mount == nil {
			mount = mountReadOnly
		}
		if err := mount(ctx, device, dir); err != nil {
			return nil, status.Errorf(codes.FailedPrecondition, "mounting %s: %v", device, err)
		}
		mountedHere = true
	}
	index, err := weightdisk.ReadIndex(dir)
	if err != nil {
		if mountedHere { // this disk isn't a weights disk, so don't leave it mounted where one is expected
			if uerr := s.unmount(ctx, dir); uerr != nil {
				log.Error(uerr, "Unmounting a disk that is not a weights disk", "dir", dir)
			}
		}
		return nil, status.Errorf(codes.FailedPrecondition, "%s: %v", device, err)
	}
	blobs := filepath.Join(dir, weightdisk.BlobDir)
	var total int64
	for sha, size := range index.Files {
		fi, err := os.Stat(filepath.Join(blobs, sha))
		if err != nil || fi.Size() != size {
			return nil, status.Errorf(codes.DataLoss, "%s: %s is missing or not %d bytes", device, sha, size)
		}
		total += size
	}
	resp := &pb.MountWeightsResponse{MountDir: dir, BlobDir: blobs, Files: index.Files, TotalBytes: total,
		MountSeconds: time.Since(start).Seconds()}
	if req.GetVerify() {
		t := time.Now()
		if err := verify(ctx, blobs, index.Files); err != nil {
			return nil, status.Errorf(codes.DataLoss, "%s: %v", device, err)
		}
		resp.VerifySeconds = time.Since(t).Seconds()
	}
	log.Info("Mounted a weights disk", "device", device, "dir", dir, "files", len(index.Files), "bytes", total,
		"mountSeconds", resp.MountSeconds, "verifySeconds", resp.VerifySeconds)
	return resp, nil
}

func (s *Server) UnmountWeights(ctx context.Context, req *pb.UnmountWeightsRequest) (*pb.UnmountWeightsResponse, error) {
	if !deviceName.MatchString(req.GetDeviceName()) {
		return nil, status.Errorf(codes.InvalidArgument, "device name %q may contain only lowercase letters, digits and dashes", req.GetDeviceName())
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	dir := filepath.Join(s.MountRoot, req.GetDeviceName())
	if !s.mounted(dir) {
		return &pb.UnmountWeightsResponse{}, nil
	}
	if err := s.unmount(ctx, dir); err != nil {
		return nil, status.Errorf(codes.FailedPrecondition, "unmounting %s: %v", dir, err)
	}
	klog.FromContext(ctx).Info("Unmounted a weights disk", "dir", dir)
	return &pb.UnmountWeightsResponse{}, nil
}

// StartProcess starts a process with typed settings.
func (s *Server) StartProcess(ctx context.Context, req *pb.StartProcessRequest) (*pb.StartProcessResponse, error) {
	id := req.GetId()
	if !processID.MatchString(id) {
		return nil, status.Errorf(codes.InvalidArgument, "process id %q may contain only lowercase letters, digits and dashes", id)
	}

	cfg, err := s.lookupProcess(req.GetProcess())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	if req.GetPort() < 0 || req.GetPort() > 65535 {
		return nil, status.Errorf(codes.InvalidArgument, "port %d must be between 1 and 65535", req.GetPort())
	}
	if req.Card != nil && req.GetCard() < 0 {
		return nil, status.Errorf(codes.InvalidArgument, "card %d must be >= 0", req.GetCard())
	}

	workRoot := s.WorkRoot
	if workRoot == "" {
		workRoot = "."
	}
	absWorkRoot, err := filepath.Abs(workRoot)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "resolving work root: %v", err)
	}

	var planPath string
	if req.GetPlanFile() != "" {
		if !cfg.AcceptsPlanFile {
			return nil, status.Errorf(codes.InvalidArgument, "process %q does not accept plan_file", req.GetProcess())
		}
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

	s.procMu.Lock()
	if s.processes == nil {
		s.processes = make(map[string]*trackedProcess)
	}
	if existing, ok := s.processes[id]; ok {
		if existing.isStopping() {
			s.procMu.Unlock()
			return nil, status.Errorf(codes.Unavailable, "process %q is stopping", id)
		}
		if existing.isRunning() {
			s.procMu.Unlock()
			if !sameConfig(existing.req, req) {
				return nil, status.Errorf(codes.AlreadyExists, "process %q is already running with different settings", id)
			}
			return &pb.StartProcessResponse{
				AlreadyRunning: true,
				Status:         existing.status(),
			}, nil
		}
	}

	args := append([]string(nil), cfg.Args...)
	if req.GetPort() > 0 {
		args = append(args, "--port", strconv.Itoa(int(req.GetPort())))
	}
	if planPath != "" {
		args = append(args, "--plan-file", planPath)
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
	cmd.Stdout = tb
	cmd.Stderr = tb

	if err := cmd.Start(); err != nil {
		s.procMu.Unlock()
		return nil, status.Errorf(codes.Internal, "starting process %q: %v", id, err)
	}

	proc := &trackedProcess{
		id:      id,
		process: req.GetProcess(),
		req:     req,
		cmd:     cmd,
		output:  tb,
		done:    make(chan struct{}),
		state:   "running",
		pid:     cmd.Process.Pid,
	}
	s.processes[id] = proc
	s.procMu.Unlock()

	go proc.wait()

	klog.FromContext(ctx).Info("Started process", "id", id, "pid", proc.pid, "process", req.GetProcess())
	return &pb.StartProcessResponse{
		AlreadyRunning: false,
		Status:         proc.status(),
	}, nil
}

// StopProcess stops a running process and removes it from tracking.
func (s *Server) StopProcess(ctx context.Context, req *pb.StopProcessRequest) (*pb.StopProcessResponse, error) {
	id := req.GetId()
	s.procMu.Lock()
	proc, ok := s.processes[id]
	if !ok {
		s.procMu.Unlock()
		return &pb.StopProcessResponse{}, nil
	}
	if proc.isStopping() {
		s.procMu.Unlock()
		select {
		case <-proc.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return &pb.StopProcessResponse{}, nil
	}
	if !proc.isRunning() {
		delete(s.processes, id)
		s.procMu.Unlock()
		return &pb.StopProcessResponse{}, nil
	}
	proc.setStopping()
	s.procMu.Unlock()

	proc.stop(ctx)

	s.procMu.Lock()
	if current, ok := s.processes[id]; ok && current == proc {
		delete(s.processes, id)
	}
	s.procMu.Unlock()

	klog.FromContext(ctx).Info("Stopped process", "id", id)
	return &pb.StopProcessResponse{}, nil
}

// GetProcess reports the status of a tracked process.
func (s *Server) GetProcess(ctx context.Context, req *pb.GetProcessRequest) (*pb.GetProcessResponse, error) {
	id := req.GetId()
	s.procMu.Lock()
	proc, ok := s.processes[id]
	s.procMu.Unlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "process %q not found", id)
	}
	return &pb.GetProcessResponse{Status: proc.status()}, nil
}

// ListProcesses lists all tracked processes in sorted order by id.
func (s *Server) ListProcesses(ctx context.Context, req *pb.ListProcessesRequest) (*pb.ListProcessesResponse, error) {
	s.procMu.Lock()
	list := make([]*pb.ProcessStatus, 0, len(s.processes))
	for _, p := range s.processes {
		list = append(list, p.status())
	}
	s.procMu.Unlock()

	sort.Slice(list, func(i, j int) bool {
		return list[i].Id < list[j].Id
	})
	return &pb.ListProcessesResponse{Processes: list}, nil
}

// Stop terminates all running processes.
func (s *Server) Stop() {
	ctx := context.Background()
	s.procMu.Lock()
	procs := make([]*trackedProcess, 0, len(s.processes))
	for _, p := range s.processes {
		procs = append(procs, p)
	}
	s.processes = make(map[string]*trackedProcess)
	s.procMu.Unlock()

	for _, p := range procs {
		p.stop(ctx)
	}
}

func (s *Server) lookupProcess(name string) (ProcessConfig, error) {
	if name == "" {
		return ProcessConfig{}, fmt.Errorf("process name cannot be empty")
	}
	procs := s.Processes
	if len(procs) == 0 {
		procs = DefaultProcesses()
	}
	cfg, ok := procs[name]
	if !ok {
		return ProcessConfig{}, fmt.Errorf("process %q is not configured on this agent", name)
	}
	return cfg, nil
}

func sameConfig(a, b *pb.StartProcessRequest) bool {
	sameCard := (a.Card == nil && b.Card == nil) || (a.Card != nil && b.Card != nil && *a.Card == *b.Card)
	return a.GetProcess() == b.GetProcess() &&
		a.GetPort() == b.GetPort() &&
		sameCard &&
		a.GetPlanFile() == b.GetPlanFile()
}

type trackedProcess struct {
	id      string
	process string
	req     *pb.StartProcessRequest
	cmd     *exec.Cmd
	output  *tailBuffer
	done    chan struct{}

	mu       sync.Mutex
	state    string
	stopping bool
	pid      int
	exitCode int
	signal   int
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
	return &pb.ProcessStatus{
		Id:       p.id,
		Pid:      int32(p.pid),
		Process:  p.process,
		State:    p.state,
		Output:   p.output.String(),
		ExitCode: int32(p.exitCode),
		Signal:   int32(p.signal),
	}
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
	mu       sync.Mutex
	lines    []string
	partial  []byte
	maxLines int
	maxBytes int
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

// mounted reports whether dir is a mount point. It doesn't check that the
// mounted filesystem is a weights disk; the index check does that. A stale
// mount, left after the disk was detached, still counts as mounted, so
// UnmountWeights unmounts it rather than reporting nothing to do.
func (s *Server) mounted(dir string) bool {
	if s.MountPoint != nil {
		return s.MountPoint(dir)
	}
	fi, err := os.Stat(dir)
	if err != nil {
		return false
	}
	parent, err := os.Stat(filepath.Dir(dir))
	if err != nil {
		return false
	}
	a, ok1 := fi.Sys().(*syscall.Stat_t)
	b, ok2 := parent.Sys().(*syscall.Stat_t)
	return ok1 && ok2 && a.Dev != b.Dev
}

func (s *Server) unmount(ctx context.Context, dir string) error {
	if s.Unmount != nil {
		return s.Unmount(ctx, dir)
	}
	return run(ctx, "umount", dir)
}

// mountReadOnly mounts an ext4 weights disk read-only, without replaying
// its journal. The disk is attached read-only, and the factory unmounted it
// cleanly, so the journal has nothing to replay.
func mountReadOnly(ctx context.Context, device, dir string) error {
	return run(ctx, "mount", "-t", "ext4", "-o", "ro,noload,noatime", device, dir)
}

func run(ctx context.Context, name string, args ...string) error {
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %w: %s", name, err, out)
	}
	return nil
}

// verify reads every file and checks its sha256. It reads several files at
// once, because a disk only reaches its throughput with many reads in
// flight.
func verify(ctx context.Context, blobs string, files map[string]int64) error {
	work := make(chan string)
	errs := make(chan error, len(files))
	var wg sync.WaitGroup
	for range min(len(files), max(4, runtime.NumCPU()/2)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sha := range work {
				errs <- verifyFile(filepath.Join(blobs, sha), sha)
			}
		}()
	}
	for sha := range files {
		select {
		case work <- sha:
		case <-ctx.Done():
		}
	}
	close(work)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return err
		}
	}
	return ctx.Err()
}

func verifyFile(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	h := sha256.New()
	_, err = io.CopyBuffer(h, f, make([]byte, 4<<20))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("%s: sha256 %s", path, got)
	}
	return nil
}

func (s *Server) sysPath(elem ...string) string {
	if s.SysRoot == "" {
		return filepath.Join(append([]string{"/"}, elem...)...)
	}
	return filepath.Join(append([]string{s.SysRoot}, elem...)...)
}

func (s *Server) getenv(key string) string {
	if s.Getenv != nil {
		return s.Getenv(key)
	}
	return os.Getenv(key)
}

func (s *Server) queryMetadata(ctx context.Context, subpath string) (string, error) {
	if s.MetadataFunc != nil {
		return s.MetadataFunc(ctx, subpath)
	}
	url := "http://metadata.google.internal/computeMetadata/v1/instance/" + subpath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing metadata response body", "error", cerr)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("metadata HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(body)), nil
}

// DescribeMachine reports what this machine is: its processors and memory,
// its cards, its local disks, and where it sits. The answer reflects what
// the agent's process can actually use (in a pod, the pod's share).
func (s *Server) DescribeMachine(ctx context.Context, req *pb.DescribeMachineRequest) (*pb.Probe, error) {
	resp := &pb.Probe{
		Device: "cpu",
	}
	s.probeCPU(ctx, resp)
	s.probeMemory(ctx, resp)
	s.probeCards(ctx, resp)
	s.probeDisks(ctx, resp)
	s.probePlacement(ctx, resp)
	if resp.GetGpuCount() > 0 {
		resp.Device = "cuda"
	}
	return resp, nil
}

func (s *Server) probeCPU(ctx context.Context, resp *pb.Probe) {
	var hostThreads, hostCores int
	uniqueCores := map[string]bool{}

	f, err := os.Open(s.sysPath("proc/cpuinfo"))
	if err != nil {
		resp.Unread = append(resp.Unread, fmt.Sprintf("cpu: %v", err))
		klog.FromContext(ctx).V(2).Info("reading cpuinfo", "error", err)
	} else {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			colon := strings.Index(line, ":")
			if colon < 0 {
				continue
			}
			key := strings.TrimSpace(line[:colon])
			val := strings.TrimSpace(line[colon+1:])
			switch key {
			case "model name":
				if resp.Cpu == "" {
					resp.Cpu = val
				}
			case "flags", "Features":
				if resp.Simd == "" {
					resp.Simd = parseSIMD(val)
				}
			case "processor":
				hostThreads++
			case "core id":
				uniqueCores[val] = true
			case "cpu cores":
				n, err := strconv.Atoi(val)
				if err != nil {
					klog.FromContext(ctx).V(2).Info("parsing cpu cores", "error", err)
				} else if n > hostCores {
					hostCores = n
				}
			}
		}
		if cerr := f.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing cpuinfo", "error", cerr)
		}
	}

	if len(uniqueCores) > 0 {
		hostCores = len(uniqueCores)
	}

	// Check cgroup CPU limits for pod share
	var cgroupCPUs int32
	// cgroup v2: cpu.max ("quota period\n")
	data, err := os.ReadFile(s.sysPath("sys/fs/cgroup/cpu.max"))
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			klog.FromContext(ctx).V(2).Info("reading cgroup cpu.max", "error", err)
		}
	} else {
		fields := strings.Fields(string(data))
		if len(fields) >= 2 && fields[0] != "max" {
			quota, qErr := strconv.ParseInt(fields[0], 10, 64)
			if qErr != nil {
				klog.FromContext(ctx).V(2).Info("parsing cgroup cpu quota", "error", qErr)
			}
			period, pErr := strconv.ParseInt(fields[1], 10, 64)
			if pErr != nil {
				klog.FromContext(ctx).V(2).Info("parsing cgroup cpu period", "error", pErr)
			}
			if qErr == nil && pErr == nil && period > 0 && quota > 0 {
				cgroupCPUs = int32((quota + period - 1) / period)
			}
		}
	}
	// cgroup v1: cpu/cpu.cfs_quota_us and cpu/cpu.cfs_period_us
	if cgroupCPUs == 0 {
		qData, qErr := os.ReadFile(s.sysPath("sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
		if qErr != nil && !errors.Is(qErr, os.ErrNotExist) {
			klog.FromContext(ctx).V(2).Info("reading cpu.cfs_quota_us", "error", qErr)
		}
		pData, pErr := os.ReadFile(s.sysPath("sys/fs/cgroup/cpu/cpu.cfs_period_us"))
		if pErr != nil && !errors.Is(pErr, os.ErrNotExist) {
			klog.FromContext(ctx).V(2).Info("reading cpu.cfs_period_us", "error", pErr)
		}
		if qErr == nil && pErr == nil {
			quota, qErr := strconv.ParseInt(strings.TrimSpace(string(qData)), 10, 64)
			if qErr != nil {
				klog.FromContext(ctx).V(2).Info("parsing cfs_quota_us", "error", qErr)
			}
			period, pErr := strconv.ParseInt(strings.TrimSpace(string(pData)), 10, 64)
			if pErr != nil {
				klog.FromContext(ctx).V(2).Info("parsing cfs_period_us", "error", pErr)
			}
			if qErr == nil && pErr == nil && period > 0 && quota > 0 {
				cgroupCPUs = int32((quota + period - 1) / period)
			}
		}
	}

	if cgroupCPUs > 0 {
		resp.CpuThreads = cgroupCPUs
		resp.CpuCores = cgroupCPUs
	} else if hostThreads > 0 {
		t := int32(hostThreads)
		c := int32(hostCores)
		if c == 0 {
			c = t
		}
		resp.CpuThreads = t
		resp.CpuCores = c
	} else {
		resp.Unread = append(resp.Unread, "cpu: threads could not be determined")
	}
}

func parseSIMD(flags string) string {
	f := " " + strings.ToLower(flags) + " "
	switch {
	case strings.Contains(f, " avx512f "):
		return "avx512"
	case strings.Contains(f, " avx2 "):
		return "avx2"
	case strings.Contains(f, " asimd ") || strings.Contains(f, " neon "):
		return "neon"
	case strings.Contains(f, " sse "):
		return "sse"
	default:
		return "unknown"
	}
}

func (s *Server) probeMemory(ctx context.Context, resp *pb.Probe) {
	var hostRAM int64
	f, err := os.Open(s.sysPath("proc/meminfo"))
	if err != nil {
		resp.Unread = append(resp.Unread, fmt.Sprintf("ram: %v", err))
		klog.FromContext(ctx).V(2).Info("reading meminfo", "error", err)
	} else {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					kb, err := strconv.ParseInt(fields[1], 10, 64)
					if err != nil {
						klog.FromContext(ctx).V(2).Info("parsing MemTotal", "error", err)
					} else {
						hostRAM = kb * 1024
					}
				}
				break
			}
		}
		if cerr := f.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing meminfo", "error", cerr)
		}
	}

	// Check cgroup memory limits (pod share)
	var cgroupMem int64
	// cgroup v2: memory.max
	data, err := os.ReadFile(s.sysPath("sys/fs/cgroup/memory.max"))
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			klog.FromContext(ctx).V(2).Info("reading cgroup memory.max", "error", err)
		}
	} else {
		trimmed := strings.TrimSpace(string(data))
		if trimmed != "" && trimmed != "max" {
			v, err := strconv.ParseInt(trimmed, 10, 64)
			if err != nil {
				klog.FromContext(ctx).V(2).Info("parsing cgroup memory.max", "error", err)
			} else if v > 0 {
				cgroupMem = v
			}
		}
	}
	// cgroup v1: memory/memory.limit_in_bytes
	if cgroupMem == 0 {
		data, err := os.ReadFile(s.sysPath("sys/fs/cgroup/memory/memory.limit_in_bytes"))
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				klog.FromContext(ctx).V(2).Info("reading cgroup memory.limit_in_bytes", "error", err)
			}
		} else {
			trimmed := strings.TrimSpace(string(data))
			v, err := strconv.ParseInt(trimmed, 10, 64)
			if err != nil {
				klog.FromContext(ctx).V(2).Info("parsing cgroup memory.limit_in_bytes", "error", err)
			} else if v > 0 && v < (1<<60) {
				cgroupMem = v
			}
		}
	}

	if cgroupMem > 0 && (hostRAM == 0 || cgroupMem < hostRAM) {
		resp.RamBytes = cgroupMem
	} else if hostRAM > 0 {
		resp.RamBytes = hostRAM
	} else {
		resp.Unread = append(resp.Unread, "ram: memory could not be determined")
	}
}

func (s *Server) probeCards(ctx context.Context, resp *pb.Probe) {
	gpuModelsByMinor := map[int]string{}
	entries, err := os.ReadDir(s.sysPath("proc/driver/nvidia/gpus"))
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			klog.FromContext(ctx).V(2).Info("reading nvidia gpus dir", "error", err)
		}
	} else {
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			infoPath := filepath.Join(s.sysPath("proc/driver/nvidia/gpus"), e.Name(), "information")
			infoData, err := os.ReadFile(infoPath)
			if err != nil {
				if !errors.Is(err, os.ErrNotExist) {
					klog.FromContext(ctx).V(2).Info("reading gpu info", "path", infoPath, "error", err)
				}
				continue
			}
			scanner := bufio.NewScanner(strings.NewReader(string(infoData)))
			var model string
			minor := -1
			for scanner.Scan() {
				line := scanner.Text()
				colon := strings.Index(line, ":")
				if colon < 0 {
					continue
				}
				key := strings.TrimSpace(line[:colon])
				val := strings.TrimSpace(line[colon+1:])
				switch key {
				case "Model":
					model = val
				case "Device Minor":
					m, err := strconv.Atoi(val)
					if err != nil {
						klog.FromContext(ctx).V(2).Info("parsing device minor", "error", err)
					} else {
						minor = m
					}
				}
			}
			if minor >= 0 && model != "" {
				gpuModelsByMinor[minor] = model
			}
		}
	}

	// Count card device nodes this process can access in its /dev.
	// On a GKE node with eight RTX PRO 6000 cards (g4-standard-384), a pod
	// requesting one GPU sees one card node in /dev (nvidia3) while /proc
	// lists all eight cards of the node, with no NVIDIA_* variable set.
	// The agent counts the accessible card nodes in /dev and matches them
	// to /proc by device minor.
	var visibleMinors []int
	devEntries, err := os.ReadDir(s.sysPath("dev"))
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			resp.Unread = append(resp.Unread, fmt.Sprintf("cards: %v", err))
			klog.FromContext(ctx).V(2).Info("reading dev dir", "error", err)
			return
		}
	} else {
		for _, e := range devEntries {
			name := e.Name()
			if strings.HasPrefix(name, "nvidia") {
				suffix := strings.TrimPrefix(name, "nvidia")
				m, err := strconv.Atoi(suffix)
				if err != nil || m < 0 {
					continue
				}
				if _, err := os.Stat(filepath.Join(s.sysPath("dev"), name)); err != nil {
					continue
				}
				visibleMinors = append(visibleMinors, m)
			}
		}
	}

	sort.Ints(visibleMinors)
	count := int32(len(visibleMinors))
	resp.GpuCount = count

	if count > 0 {
		firstMinor := visibleMinors[0]
		if m, ok := gpuModelsByMinor[firstMinor]; ok {
			resp.Gpu = m
		} else {
			resp.Gpu = ""
			resp.Unread = append(resp.Unread, fmt.Sprintf("card %d: driver entry could not be read", firstMinor))
		}
	}
}

func (s *Server) probeDisks(ctx context.Context, resp *pb.Probe) {
	entries, err := os.ReadDir(s.sysPath("dev/disk/by-id"))
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			resp.Unread = append(resp.Unread, fmt.Sprintf("disks: %v", err))
			klog.FromContext(ctx).V(2).Info("reading disk by-id dir", "error", err)
			return
		}
		resp.StorageDevices = 0
		return
	}

	var devNames []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, "google-local-nvme-ssd-") && !strings.Contains(name, "-part") {
			devNames = append(devNames, name)
		}
	}

	count := int32(len(devNames))
	resp.StorageDevices = count

	if count > 0 {
		var totalBytes int64
		readAll := true
		for _, name := range devNames {
			linkPath := filepath.Join(s.sysPath("dev/disk/by-id"), name)
			target, err := os.Readlink(linkPath)
			if err != nil {
				readAll = false
				break
			}
			blkName := filepath.Base(target)
			sizeData, err := os.ReadFile(s.sysPath("sys/class/block", blkName, "size"))
			if err != nil {
				readAll = false
				break
			}
			sectors, err := strconv.ParseInt(strings.TrimSpace(string(sizeData)), 10, 64)
			if err != nil {
				readAll = false
				break
			}
			totalBytes += sectors * 512
		}
		if readAll {
			resp.StorageBytes = totalBytes
		}
	}
}

func (s *Server) probePlacement(ctx context.Context, resp *pb.Probe) {
	var zone string
	z, err := s.queryMetadata(ctx, "zone")
	if err != nil {
		resp.Unread = append(resp.Unread, "zone: metadata server unreachable")
	} else if z != "" {
		zone = path.Base(z)
	}

	instance := s.getenv("NODE_NAME")
	if instance == "" {
		n, err := s.queryMetadata(ctx, "name")
		if err != nil {
			klog.FromContext(ctx).V(2).Info("querying metadata name", "error", err)
		} else if n != "" {
			instance = n
		}
	}
	if instance == "" {
		instance = s.getenv("HOSTNAME")
	}
	if instance == "" {
		resp.Unread = append(resp.Unread, "instance: could not determine instance name")
	}

	resp.Zone = zone
	resp.InstanceName = instance
}
