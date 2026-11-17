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
// Today it mounts weights disks (#407). A weights disk holds a checkpoint's
// files, in the same layout as the executor's local weight tier. The
// coordinator attaches the disk read-only, the agent mounts it, and the
// executor reads the files in place instead of copying them over the
// network.
package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
)

var deviceName = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}$`)

// Server implements the Agent service.
type Server struct {
	pb.UnimplementedAgentServer
	// MountRoot holds one directory per mounted weights disk, named by its
	// device name.
	MountRoot string
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
