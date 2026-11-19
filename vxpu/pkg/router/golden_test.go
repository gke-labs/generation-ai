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
	"net"
	"strings"
	"sync"
	"testing"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

type goldenFakeExecutor struct {
	pb.UnimplementedExecutorServer
	mu       sync.Mutex
	sessions int
	tokens   [][]int32
}

func (e *goldenFakeExecutor) Describe(ctx context.Context, req *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	return &pb.DescribeResponse{}, nil
}

func (e *goldenFakeExecutor) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *goldenFakeExecutor) GetLoad(ctx context.Context, req *pb.GetLoadRequest) (*pb.GetLoadResponse, error) {
	return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY}, nil
}

func (e *goldenFakeExecutor) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
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

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
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

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
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

func TestLoadModel_WithGolden_RefusesOnMismatch(t *testing.T) {
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

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}

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

	if err := s.putGolden(ctx, "test-model", golden); err != nil {
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
}

func TestRegisterModel_WithGolden_RefusesOnMismatch(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 999, 30},
		},
	}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:  "org/M",
		Golden: golden,
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
}

func TestRegisterModel_WithGolden_Succeeds(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},
		},
	}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:  "org/M",
		Golden: golden,
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

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
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
	_ = s.putGolden(ctx, "test-model-cancel", golden)

	_, err = s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifest})
	if err == nil {
		t.Fatal("expected LoadModel to fail on cancelled context")
	}
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected codes.Canceled, got %v", status.Code(err))
	}
}
