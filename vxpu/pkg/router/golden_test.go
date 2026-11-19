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

package router

import (
	"context"
	"encoding/json"
	"net"
	"strings"
	"sync"
	"testing"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

type goldenFakeExecutor struct {
	pb.UnimplementedExecutorServer
	mu           sync.Mutex
	loads        int
	sessions     int
	tokens       [][]int32
	transportErr error
}

func (e *goldenFakeExecutor) Describe(ctx context.Context, req *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	return &pb.DescribeResponse{}, nil
}

func (e *goldenFakeExecutor) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.loads++
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *goldenFakeExecutor) GetLoad(ctx context.Context, req *pb.GetLoadRequest) (*pb.GetLoadResponse, error) {
	return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY}, nil
}

func (e *goldenFakeExecutor) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.transportErr != nil {
		return nil, e.transportErr
	}
	e.sessions++
	return &pb.NewSessionResponse{SessionId: "sess"}, nil
}

func (e *goldenFakeExecutor) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	toks := []int32{1, 2, 3}
	if len(e.tokens) > 0 {
		toks = e.tokens[0]
		e.tokens = e.tokens[1:]
	}
	if err := stream.Send(&pb.ExecuteResponse{
		Response: &pb.GenerateResponse{TokenIds: toks},
	}); err != nil {
		return err
	}
	return stream.Send(&pb.ExecuteResponse{
		Response: &pb.GenerateResponse{Done: true},
	})
}

func TestVerifyGolden_Matches(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	client := pb.NewExecutorClient(conn)
	router := &Server{}

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}

	if err := router.verifyGolden(ctx, client, "m1", golden); err != nil {
		t.Fatalf("verifyGolden failed: %v", err)
	}
}

func TestVerifyGolden_TokenMismatchFails(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 999, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	client := pb.NewExecutorClient(conn)
	router := &Server{}

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}

	err = router.verifyGolden(ctx, client, "m1", golden)
	if err == nil {
		t.Fatal("expected verifyGolden to fail on mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}
}

func TestLoadModel_WithGoldenJSON_RefusesOnMismatch(t *testing.T) {
	fakePeer := &peer.Peer{Addr: &net.TCPAddr{IP: net.ParseIP("127.0.0.1"), Port: 12345}}
	ctx := peer.NewContext(t.Context(), fakePeer)
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 999, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)

	manifest := `{"model": "test-model"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	if err := s.putGolden(ctx, "test-model", golden, goldenBytes); err != nil {
		t.Fatalf("putGolden: %v", err)
	}

	_, err = s.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: manifest,
	})
	if err == nil {
		t.Fatal("expected LoadModel with mismatched golden to fail, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}

	// An ask after a refusal is refused.
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{})
	if err == nil {
		t.Fatal("expected NewSession after refused LoadModel to fail, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}
}

func TestRegisterModel_WithGoldenJSON_RefusesOnMismatch(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 999, 30},
		},
	}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:      "org/M",
		GoldenJson: string(goldenBytes),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	_, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err == nil {
		t.Fatal("expected NewSession to fail when golden check fails, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}

	// An ask after a refusal is refused on named model too.
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err == nil {
		t.Fatal("expected second NewSession to fail when golden check failed, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}
}

func TestRegisterModel_WithGoldenJSON_Succeeds(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
	}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:      "org/M",
		GoldenJson: string(goldenBytes),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId, got %q", resp.SessionId)
	}
}

func TestVerifyGolden_CancellationReportedAccurately(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	exec := &goldenFakeExecutor{}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-model-cancel"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(ctx, "test-model-cancel", golden, goldenBytes)

	_, err = s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest})
	if err == nil {
		t.Fatal("expected LoadModel to fail on cancelled context")
	}
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected codes.Canceled, got %v", status.Code(err))
	}
}

func TestModelStanding_ModelFailsThenLaterPassesIsServed(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 999, 30},
			{10, 20, 30},
		},
	}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	mismatchedGolden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 888, 30}},
		},
	}
	mismatchedBytes, _ := json.Marshal(mismatchedGolden)

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:      "org/M",
		GoldenJson: string(mismatchedBytes),
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	// 1. Initial ask fails because golden check fails
	_, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err == nil {
		t.Fatal("expected NewSession to fail when golden check fails, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}

	// 2. An ask after the refusal is also refused
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err == nil {
		t.Fatal("expected second NewSession to fail after refusal, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}

	// 3. Registering matching golden document clears refusal on next load and serves
	validGolden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	validBytes, _ := json.Marshal(validGolden)
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:      "org/M",
		GoldenJson: string(validBytes),
	}); err != nil {
		t.Fatalf("RegisterModel with valid golden: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession should succeed after golden passed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId, got %q", resp.SessionId)
	}
}

func TestModelStanding_CancelledCallerDoesNotRefuseModel(t *testing.T) {
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
			{10, 20, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-cancel-standing"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(t.Context(), "test-cancel-standing", golden, goldenBytes)

	// Caller 1 cancels context
	cancelCtx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = s.LoadModel(cancelCtx, &pb.LoadModelRequest{ManifestJson: manifest})
	if err == nil {
		t.Fatal("expected LoadModel to fail on cancelled context")
	}
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected codes.Canceled, got %v", status.Code(err))
	}

	// Model standing should NOT be refused
	if st := s.modelStanding(modelKey); st.standing == standingRefused {
		t.Errorf("cancelled caller should not move standing to refused, got %v", st.standing)
	}

	// Caller 2 with healthy context loads and passes
	healthyCtx := t.Context()
	_, err = s.LoadModel(healthyCtx, &pb.LoadModelRequest{ManifestJson: manifest})
	if err != nil {
		t.Fatalf("LoadModel with healthy context should succeed: %v", err)
	}
	if st := s.modelStanding(modelKey); st.standing != standingPassed {
		t.Errorf("standing = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_TransportErrorDoesNotRefuseModel(t *testing.T) {
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
		transportErr: status.Error(codes.Unavailable, "network connection dropped"),
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-transport-standing"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(t.Context(), "test-transport-standing", golden, goldenBytes)

	// Caller 1 experiences transport error
	_, err = s.LoadModel(t.Context(), &pb.LoadModelRequest{ManifestJson: manifest})
	if err == nil {
		t.Fatal("expected LoadModel to fail on transport error")
	}
	if status.Code(err) != codes.Unavailable {
		t.Errorf("expected codes.Unavailable, got %v", status.Code(err))
	}

	// Model standing should NOT be refused
	if st := s.modelStanding(modelKey); st.standing == standingRefused {
		t.Errorf("transport error should not move standing to refused, got %v", st.standing)
	}

	// Transport recovers
	exec.mu.Lock()
	exec.transportErr = nil
	exec.mu.Unlock()

	// Caller 2 loads and passes
	_, err = s.LoadModel(t.Context(), &pb.LoadModelRequest{ManifestJson: manifest})
	if err != nil {
		t.Fatalf("LoadModel after transport recovery should succeed: %v", err)
	}
	if st := s.modelStanding(modelKey); st.standing != standingPassed {
		t.Errorf("standing = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_LoadsOfOneModelDoNotOverlap(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-overlap"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(ctx, "test-overlap", golden, goldenBytes)

	// Launch two concurrent LoadModel calls for the same model
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest})
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("caller %d failed: %v", i, err)
		}
	}
	if exec.loads != 1 {
		t.Errorf("expected exactly 1 load across concurrent calls, got %d", exec.loads)
	}
}

func TestModelStanding_ReloadServingModelIsNoOp(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-reload-noop"}`
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(ctx, "test-reload-noop", golden, goldenBytes)

	// First load succeeds
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest}); err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}
	if exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", exec.loads)
	}

	// Second load of the serving model is a no-op
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest}); err != nil {
		t.Fatalf("second LoadModel: %v", err)
	}
	if exec.loads != 1 {
		t.Errorf("reload of serving model should be no-op, got loads = %d", exec.loads)
	}
}

func TestModelStanding_ExecutorGoneAllowsReload(t *testing.T) {
	fakePeer := &peer.Peer{Addr: &net.TCPAddr{IP: net.ParseIP("127.0.0.1"), Port: 12345}}
	ctx := peer.NewContext(t.Context(), fakePeer)
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
			{10, 20, 30},
		},
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterExecutorServer(grpcServer, exec)
	go func() { _ = grpcServer.Serve(lis) }()
	t.Cleanup(func() { grpcServer.Stop() })

	golden := &kernelplan.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []kernelplan.GoldenPrompt{
			{IDs: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	goldenBytes, _ := json.Marshal(golden)
	manifest := `{"model": "test-executor-gone"}`
	modelKey := artifactKey(manifest)
	podName := "vxpu-executor-" + modelKey
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	clientset := fake.NewSimpleClientset(pod)
	s := NewServer(clientset, "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(ctx, "test-executor-gone", golden, goldenBytes)

	// First load succeeds
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest}); err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}
	if exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", exec.loads)
	}

	// Executor pod goes away (deleted in Kubernetes)
	_ = clientset.CoreV1().Pods("default").Delete(ctx, podName, metav1.DeleteOptions{})

	// Calling NewSession detects executor is gone
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("NewSession after executor gone: got %v, want FailedPrecondition", err)
	}

	// Now re-create pod and load again
	_, _ = clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if _, err := s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest}); err != nil {
		t.Fatalf("reload after executor was gone failed: %v", err)
	}
	if exec.loads != 2 {
		t.Errorf("expected second load after executor was gone, got %d", exec.loads)
	}
}
