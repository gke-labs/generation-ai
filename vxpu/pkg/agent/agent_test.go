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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

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
