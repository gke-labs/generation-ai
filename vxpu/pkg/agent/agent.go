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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
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
	"github.com/gke-labs/generation-ai/vxpu/pkg/collections"
	"github.com/gke-labs/generation-ai/vxpu/pkg/fileutil"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

var (
	deviceName        = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}$`)
	sha256Pattern     = regexp.MustCompile(`^[0-9a-f]{64}$`)
	packageKeyPattern = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$`)
)

const defaultMaxConcurrentTransfers = 8

// Server implements the Agent service.
type Server struct {
	pb.UnimplementedAgentServer
	// MountRoot holds one directory per mounted weights disk, named by its
	// device name.
	MountRoot string
	// WorkRoot is the base directory under which process working directories
	// and plan files must reside.
	WorkRoot string

	// BlobDir is the directory where staged content-addressed blobs are stored,
	// each named by its sha256. When empty, filepath.Join(WorkRoot, "blobs")
	// is used.
	BlobDir string

	// PackageDir is the directory where staged compiled packages are stored,
	// each named <key>.pt2. When empty, filepath.Join(WorkRoot, "packages")
	// is used.
	PackageDir string

	// Store is the client used to fetch content-addressed blobs from modelstore.
	// When nil, the agent cannot fetch missing blobs.
	Store *weightstore.Client

	// MaxConcurrentTransfers is the maximum number of blob downloads in flight
	// at once during a StageBlobs call. When zero or negative, a default of 8
	// is used.
	MaxConcurrentTransfers int

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

	// mu serializes mount and unmount operations.
	mu sync.Mutex

	// stageDedup deduplicates concurrent blob downloads by sha256.
	stageDedup *collections.Deduplicator[string, int64]

	// procMu synchronizes access to procs.
	procMu sync.Mutex

	// procs manages child processes, helper commands, and exit collection.
	procs *processManager

	// forceReaper enables the orphan reaper in tests even when PID is not 1.
	forceReaper bool

	// Host is the hostname or IP address published in process endpoints, set from the --host flag.
	Host string

	// SoftwareVersion is the git tree commit stamped into the executor image at build time, or empty when unstamped.
	SoftwareVersion string

	// allocatePort overrides the default port allocation function for testing.
	allocatePort func() (int, error)
}

// NewServer returns a new agent Server configured to serve from mountRoot and workRoot.
func NewServer(mountRoot, workRoot string, store *weightstore.Client) *Server {
	return &Server{
		MountRoot:  mountRoot,
		WorkRoot:   workRoot,
		Store:      store,
		stageDedup: collections.NewDeduplicator[string, int64](),
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
			mount = s.mountReadOnly
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

func (s *Server) maxConcurrentTransfers() int {
	if s.MaxConcurrentTransfers > 0 {
		return s.MaxConcurrentTransfers
	}
	return defaultMaxConcurrentTransfers
}

// StageBlobs makes the blobs named by their sha256 present on local disk,
// fetching any missing blobs from the agent's configured store. It reports
// each blob as present with its size.
func (s *Server) StageBlobs(ctx context.Context, req *pb.StageBlobsRequest) (*pb.StageBlobsResponse, error) {
	for _, name := range req.GetNames() {
		if !sha256Pattern.MatchString(name) {
			return nil, status.Errorf(codes.InvalidArgument, "invalid blob sha256 %q: must be 64 lowercase hex digits", name)
		}
	}

	blobDir, err := s.blobDir()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		return nil, status.Errorf(codes.Internal, "creating blob directory: %v", err)
	}

	names := req.GetNames()
	if len(names) == 0 {
		return &pb.StageBlobsResponse{}, nil
	}

	unique := make([]string, 0, len(names))
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if !seen[name] {
			seen[name] = true
			unique = append(unique, name)
		}
	}

	sizes := collections.NewConcurrentHashMap[string, int64]()

	tasks := make([]Task, len(unique))
	for i, sha := range unique {
		tasks[i] = func(ctx context.Context) error {
			size, err := s.stageOneBlob(ctx, blobDir, sha)
			if err != nil {
				return err
			}
			sizes.Put(sha, size)
			return nil
		}
	}

	limit := s.maxConcurrentTransfers()
	if err := RunTasks(ctx, tasks, limit); err != nil {
		if ctx.Err() != nil {
			return nil, status.FromContextError(ctx.Err()).Err()
		}
		return nil, err
	}

	result := make([]*pb.BlobInfo, len(names))
	for i, name := range names {
		size, _ := sizes.Get(name)
		result[i] = &pb.BlobInfo{
			Name: name,
			Size: size,
		}
	}
	return &pb.StageBlobsResponse{Blobs: result}, nil
}

// ListBlobs reports which blobs are currently staged on local disk with their sizes.
func (s *Server) ListBlobs(ctx context.Context, _ *pb.ListBlobsRequest) (*pb.ListBlobsResponse, error) {
	blobDir, err := s.blobDir()
	if err != nil {
		return nil, err
	}

	blobs, err := stagedBlobs(blobDir)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "listing staged blobs: %v", err)
	}
	return &pb.ListBlobsResponse{Blobs: blobs}, nil
}

func (s *Server) stageOneBlob(ctx context.Context, blobDir, sha string) (int64, error) {
	target := filepath.Join(blobDir, sha)
	if fi, err := os.Lstat(target); err == nil && fi.Mode().IsRegular() {
		return fi.Size(), nil
	}

	size, err := s.stageDedup.Do(ctx, sha, func() (int64, bool, error) {
		if fi, err := os.Lstat(target); err == nil && fi.Mode().IsRegular() {
			return fi.Size(), true, nil
		}
		n, err := s.downloadBlob(ctx, blobDir, sha)
		return n, false, err
	})
	if err != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) {
		return 0, status.FromContextError(err).Err()
	}
	return size, err
}

// stageArtifact fetches an artifact's manifest from the store, stages its layer
// blobs via StageBlobs, and links each named file into destDir.
func (s *Server) stageArtifact(ctx context.Context, ref, destDir string) error {
	repo, tag, err := weightstore.ParseArtifactRef(ref)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "staging artifact %q: %v", ref, err)
	}
	if s.Store == nil {
		return status.Errorf(codes.FailedPrecondition, "staging artifact %q: no store configured on agent", ref)
	}
	blobDir, err := s.blobDir()
	if err != nil {
		return status.Errorf(status.Code(err), "staging artifact %q: %v", ref, status.Convert(err).Message())
	}
	manifest, err := s.Store.GetManifest(ctx, repo, tag)
	if err != nil {
		return status.Errorf(codes.Unavailable, "staging artifact %q: %v", ref, err)
	}
	type layerLink struct {
		title string
		sha   string
	}
	var links []layerLink
	var shas []string
	for _, layer := range manifest.Layers {
		title := layer.Annotations["org.opencontainers.image.title"]
		if title == "" {
			continue
		}
		cleanTitle := filepath.Clean(title)
		if strings.HasPrefix(cleanTitle, "..") || filepath.IsAbs(cleanTitle) {
			return status.Errorf(codes.InvalidArgument, "staging artifact %q: layer title %q escapes destination", ref, title)
		}
		sha, ok := strings.CutPrefix(layer.Digest, "sha256:")
		if !ok {
			return status.Errorf(codes.InvalidArgument, "staging artifact %q: unsupported digest %q for %s", ref, layer.Digest, cleanTitle)
		}
		links = append(links, layerLink{title: cleanTitle, sha: sha})
		shas = append(shas, sha)
	}
	if len(links) == 0 {
		return status.Errorf(codes.Unavailable, "staging artifact %q: manifest has no named layers", ref)
	}
	if _, err := s.StageBlobs(ctx, &pb.StageBlobsRequest{Names: shas}); err != nil {
		return status.Errorf(status.Code(err), "staging artifact %q: %v", ref, status.Convert(err).Message())
	}
	if err := os.RemoveAll(destDir); err != nil {
		return status.Errorf(codes.Internal, "staging artifact %q: resetting directory: %v", ref, err)
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return status.Errorf(codes.Internal, "staging artifact %q: creating directory: %v", ref, err)
	}
	for _, l := range links {
		linkPath := filepath.Join(destDir, l.title)
		if err := os.MkdirAll(filepath.Dir(linkPath), 0o755); err != nil {
			return status.Errorf(codes.Internal, "staging artifact %q: creating parent directory for %s: %v", ref, l.title, err)
		}
		if err := os.Symlink(filepath.Join(blobDir, l.sha), linkPath); err != nil {
			return status.Errorf(codes.Internal, "staging artifact %q: linking %s: %v", ref, l.title, err)
		}
	}
	return nil
}

func (s *Server) downloadBlob(ctx context.Context, blobDir, sha string) (int64, error) {
	target := filepath.Join(blobDir, sha)
	if s.Store == nil {
		return 0, status.Errorf(codes.FailedPrecondition, "no store configured on this agent to fetch blob %s", sha)
	}

	rc, _, err := s.Store.OpenBlob(ctx, sha)
	if err != nil {
		if ctx.Err() != nil {
			return 0, status.FromContextError(ctx.Err()).Err()
		}
		if errors.Is(err, weightstore.ErrNotFound) {
			return 0, status.Errorf(codes.NotFound, "blob %s not found in store", sha)
		}
		return 0, status.Errorf(codes.Internal, "fetching blob %s: %v", sha, err)
	}
	defer func() {
		if rc != nil {
			if cerr := rc.Close(); cerr != nil {
				klog.FromContext(ctx).V(2).Info("closing blob store reader", "blob", sha, "error", cerr)
			}
		}
	}()

	tmp, err := fileutil.CreateTemp(blobDir, "."+sha+".tmp-*")
	if err != nil {
		return 0, status.Errorf(codes.Internal, "creating temp file for blob %s: %v", sha, err)
	}
	defer tmp.Cleanup()

	h := sha256.New()
	w := io.MultiWriter(tmp, h)
	buf := make([]byte, 1<<20)
	n, copyErr := io.CopyBuffer(w, rc, buf)
	if cerr := rc.Close(); cerr != nil {
		klog.FromContext(ctx).V(2).Info("closing blob store reader", "blob", sha, "error", cerr)
	}
	rc = nil

	if copyErr != nil {
		if ctx.Err() != nil {
			return 0, status.FromContextError(ctx.Err()).Err()
		}
		return 0, status.Errorf(codes.Internal, "transferring blob %s: %v", sha, copyErr)
	}

	if ctx.Err() != nil {
		return 0, status.FromContextError(ctx.Err()).Err()
	}

	got := hex.EncodeToString(h.Sum(nil))
	if got != sha {
		return 0, status.Errorf(codes.DataLoss, "blob %s: sha256 mismatch: got %s, want %s", sha, got, sha)
	}

	if err := tmp.Rename(target); err != nil {
		if ctx.Err() != nil {
			return 0, status.FromContextError(ctx.Err()).Err()
		}
		return 0, status.Errorf(codes.Internal, "placing blob %s: %v", sha, err)
	}
	return n, nil
}

// Start initializes the server and cleans stale temporary files from previous runs.
func (s *Server) Start() {
	s.procMgr()
	for _, dirFn := range []func() (string, error){s.blobDir, s.packageDir} {
		dir, err := dirFn()
		if err != nil {
			klog.V(2).Info("skipping stale temp cleanup on start", "error", err)
			continue
		}
		cleanStaleTempBlobs(dir)
	}
}

func (s *Server) blobDir() (string, error) {
	if s.BlobDir != "" {
		return s.BlobDir, nil
	}
	if s.WorkRoot == "" {
		return "", status.Errorf(codes.FailedPrecondition, "agent has no work root configured for blob staging")
	}
	return filepath.Join(s.WorkRoot, "blobs"), nil
}

func (s *Server) packageDir() (string, error) {
	if s.PackageDir != "" {
		return s.PackageDir, nil
	}
	if s.WorkRoot == "" {
		return "", status.Errorf(codes.FailedPrecondition, "agent has no work root configured for package staging")
	}
	return filepath.Join(s.WorkRoot, "packages"), nil
}

// StagePackages makes the compiled packages named by key for model present in
// the executor's local package cache directory, fetching any missing packages
// from the agent's configured store. It skips keys already on disk, writes
// fetched files atomically, never builds packages, and never retries store
// requests.
func (s *Server) StagePackages(ctx context.Context, req *pb.StagePackagesRequest) (*pb.StagePackagesResponse, error) {
	keys := req.GetKeys()
	unique := make([]string, 0, len(keys))
	seen := make(map[string]bool, len(keys))
	for _, key := range keys {
		if !packageKeyPattern.MatchString(key) {
			return nil, status.Errorf(codes.InvalidArgument, "invalid package key %q", key)
		}
		if !seen[key] {
			seen[key] = true
			unique = append(unique, key)
		}
	}

	pkgDir, err := s.packageDir()
	if err != nil {
		return nil, err
	}
	if len(unique) == 0 {
		return &pb.StagePackagesResponse{PackageDir: pkgDir}, nil
	}

	var need []string
	for _, key := range unique {
		target := filepath.Join(pkgDir, key+".pt2")
		if fi, err := os.Lstat(target); err == nil && fi.Mode().IsRegular() {
			continue
		}
		need = append(need, key)
	}
	if len(need) == 0 {
		return &pb.StagePackagesResponse{PackageDir: pkgDir}, nil
	}

	if s.Store == nil {
		return nil, status.Errorf(
			codes.FailedPrecondition,
			"missing package(s) [%s]: not in local cache %s and no modelstore is configured; build the plan's packages",
			strings.Join(need, ", "), pkgDir,
		)
	}
	model := req.GetModel()
	if model == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model is required to fetch packages from modelstore")
	}
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		return nil, status.Errorf(codes.Internal, "creating package directory: %v", err)
	}

	repo := weightstore.PackageRepository(model)
	var missing []string
	for _, key := range need {
		data, err := s.Store.FetchFile(ctx, repo, key, key+".pt2")
		if err != nil {
			if ctx.Err() != nil {
				return nil, status.FromContextError(ctx.Err()).Err()
			}
			if errors.Is(err, weightstore.ErrNotFound) {
				missing = append(missing, key)
				continue
			}
			return nil, status.Errorf(codes.Unavailable, "fetching package %s from %s: %v", key, s.Store.BaseURL(), err)
		}
		target := filepath.Join(pkgDir, key+".pt2")
		if err := fileutil.WriteAtomic(target, data, 0o644); err != nil {
			return nil, status.Errorf(codes.Internal, "placing package %s: %v", key, err)
		}
	}
	if len(missing) > 0 {
		return nil, status.Errorf(
			codes.FailedPrecondition,
			"missing package(s) [%s]: not in local cache %s or modelstore %s (%s); build the plan's packages",
			strings.Join(missing, ", "), pkgDir, s.Store.BaseURL(), repo,
		)
	}
	return &pb.StagePackagesResponse{PackageDir: pkgDir}, nil
}

func cleanStaleTempBlobs(blobDir string) {
	entries, err := os.ReadDir(blobDir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, ".") && strings.Contains(name, ".tmp-") {
			if err := os.Remove(filepath.Join(blobDir, name)); err != nil && !os.IsNotExist(err) {
				klog.V(2).Info("removing stale temp blob file", "name", name, "error", err)
			}
		}
	}
}

func stagedBlobs(blobDir string) ([]*pb.BlobInfo, error) {
	entries, err := os.ReadDir(blobDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var blobs []*pb.BlobInfo
	for _, entry := range entries {
		if !entry.Type().IsRegular() {
			continue
		}
		name := entry.Name()
		if !sha256Pattern.MatchString(name) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		blobs = append(blobs, &pb.BlobInfo{
			Name: name,
			Size: info.Size(),
		})
	}
	sort.Slice(blobs, func(i, j int) bool {
		return blobs[i].Name < blobs[j].Name
	})
	return blobs, nil
}

func (s *Server) procMgr() *processManager {
	s.procMu.Lock()
	defer s.procMu.Unlock()
	if s.procs == nil {
		s.procs = newProcessManager(s.WorkRoot, s.Host, s.Processes)
		s.procs.stageArtifact = s.stageArtifact
		if pkgDir, err := s.packageDir(); err != nil {
			klog.V(2).Info("agent has no package directory configured for executor processes", "error", err)
		} else {
			s.procs.packageDir = pkgDir
		}
		s.procs.softwareVersion = s.SoftwareVersion
		if s.Store != nil {
			s.procs.storeURL = s.Store.BaseURL()
		}
		if s.forceReaper {
			s.procs.forceReaper = true
		}
		if s.allocatePort != nil {
			s.procs.allocatePort = s.allocatePort
		}
	}
	return s.procs
}

// StartProcess starts a process with typed settings.
func (s *Server) StartProcess(ctx context.Context, req *pb.StartProcessRequest) (*pb.StartProcessResponse, error) {
	return s.procMgr().StartProcess(ctx, req)
}

// StopProcess stops a running process and removes it from tracking.
func (s *Server) StopProcess(ctx context.Context, req *pb.StopProcessRequest) (*pb.StopProcessResponse, error) {
	return s.procMgr().StopProcess(ctx, req)
}

// GetProcess reports the status of a tracked process.
func (s *Server) GetProcess(ctx context.Context, req *pb.GetProcessRequest) (*pb.GetProcessResponse, error) {
	return s.procMgr().GetProcess(ctx, req)
}

// ListProcesses lists all tracked processes in sorted order by id.
func (s *Server) ListProcesses(ctx context.Context, req *pb.ListProcessesRequest) (*pb.ListProcessesResponse, error) {
	return s.procMgr().ListProcesses(ctx, req)
}

// Stop terminates all running processes.
func (s *Server) Stop() {
	s.procMu.Lock()
	pm := s.procs
	s.procMu.Unlock()
	if pm != nil {
		pm.Stop()
	}
}

// StartReaper starts background orphan child reaping when the agent is PID 1.
// In tests, forceReaper can enable it.
func (s *Server) StartReaper() {
	pm := s.procMgr()
	if s.forceReaper {
		pm.forceReaper = true
	}
	pm.StartReaper()
}

// ReapOrphans collects exit statuses of all exited child processes.
func (s *Server) ReapOrphans() {
	s.procMgr().ReapOrphans()
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
	_, err := s.run(ctx, "umount", dir)
	return err
}

// mountReadOnly mounts an ext4 weights disk read-only, without replaying
// its journal. The disk is attached read-only, and the factory unmounted it
// cleanly, so the journal has nothing to replay.
func (s *Server) mountReadOnly(ctx context.Context, device, dir string) error {
	_, err := s.run(ctx, "mount", "-t", "ext4", "-o", "ro,noload,noatime", device, dir)
	return err
}

func (s *Server) run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return s.procMgr().run(ctx, name, args...)
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
// its cards, its local disks, where it sits, and its software version. The
// hardware answer reflects what the agent's process can actually use (in a pod,
// the pod's share).
func (s *Server) DescribeMachine(ctx context.Context, req *pb.DescribeMachineRequest) (*pb.Probe, error) {
	resp := &pb.Probe{
		Device:          "cpu",
		SoftwareVersion: s.SoftwareVersion,
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
