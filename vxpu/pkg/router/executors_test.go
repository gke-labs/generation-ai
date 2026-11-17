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
	"fmt"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// An executor holding one engine's experts is that engine's until the
// engine is re-planned, forgotten or replaced: another engine planning
// at the same time cannot take it, and it is not stranded afterwards.
func TestHelpers_ClaimedAndReleased(t *testing.T) {
	s := &Server{
		externalExecutors: []string{"helper:1"},
		executors:         map[string]executorInfo{"helper:1": {}},
		externalModel:     map[string]string{},
		modelEndpoint:     map[string]string{},
	}
	free := func(engine string) int { return len(s.freeExecutors(t.Context(), engine)) }
	if !s.claimHelper("helper:1", "a:1") {
		t.Fatal("a free executor could not be claimed")
	}
	if s.claimHelper("helper:1", "b:1") || free("b:1") != 0 {
		t.Error("an executor holding a's experts was offered to b")
	}
	if free("a:1") != 1 {
		t.Error("a's own helper is not offered when a is planned again")
	}
	s.forgetModel("model-a", "a:1")
	if free("b:1") != 1 || !s.claimHelper("helper:1", "b:1") {
		t.Error("forgetting a did not free its helper")
	}
	s.externalModel["b:1"] = "model-b"
	s.placedExternally("model-c", "b:1")
	if !s.claimHelper("helper:1", "c:1") {
		t.Error("replacing b's model did not free b's helper")
	}
}

type countedListener struct {
	net.Listener
	accepted atomic.Int32
}

func (l *countedListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err == nil {
		l.accepted.Add(1)
	}
	return c, err
}

type mockExecutorServer struct {
	pb.UnimplementedExecutorServer
}

func (m *mockExecutorServer) TokenizeChat(ctx context.Context, req *pb.TokenizeChatRequest) (*pb.TokenizeChatResponse, error) {
	return &pb.TokenizeChatResponse{InputIds: []int32{1, 2, 3}}, nil
}

// Many calls to one executor use one connection, and a call after the executor restarted succeeds.
func TestExecutorConnectionReuseAndRestart(t *testing.T) {
	ctx := t.Context()

	rawLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := rawLis.Addr().String()
	lis := &countedListener{Listener: rawLis}

	exec := &mockExecutorServer{}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	router := NewServer(fake.NewSimpleClientset(), "default", "img", "")
	modelKey := "test-model"
	router.placedExternally(modelKey, addr)

	// Many calls to one executor use one connection.
	for i := 0; i < 10; i++ {
		resp, err := router.TokenizeChat(ctx, &pb.TokenizeChatRequest{Model: modelKey})
		if err != nil {
			t.Fatalf("call %d failed: %v", i, err)
		}
		if len(resp.GetInputIds()) != 3 {
			t.Fatalf("unexpected response: %v", resp)
		}
	}

	if got := lis.accepted.Load(); got != 1 {
		t.Errorf("expected 1 connection accepted across 10 calls, got %d", got)
	}

	// Stop executor and restart on the same address; make the call race the
	// restart on purpose before the client could have noticed the old transport is gone.
	for i := 0; i < 5; i++ {
		srv.Stop()
		lis.Close()

		type callResult struct {
			resp *pb.TokenizeChatResponse
			err  error
		}
		ch := make(chan callResult, 1)
		go func() {
			resp, err := router.TokenizeChat(ctx, &pb.TokenizeChatRequest{Model: modelKey})
			ch <- callResult{resp: resp, err: err}
		}()

		// Give the call time to hit the closed connection before the new server listens.
		time.Sleep(5 * time.Millisecond)

		rawLis, err = net.Listen("tcp", addr)
		if err != nil {
			t.Fatal(err)
		}
		lis = &countedListener{Listener: rawLis}
		srv = grpc.NewServer()
		pb.RegisterExecutorServer(srv, exec)
		go func(s *grpc.Server, l net.Listener) { _ = s.Serve(l) }(srv, lis)

		res := <-ch
		if res.err != nil {
			t.Fatalf("call racing restart %d failed: %v", i, res.err)
		}
		if len(res.resp.GetInputIds()) != 3 {
			t.Fatalf("unexpected response after restart %d: %v", i, res.resp)
		}
	}
	defer srv.Stop()
}

// Connections are closed when their executor is removed and when the router shuts down; none is left.
func TestExecutorConnectionsClosedOnRemovalAndShutdown(t *testing.T) {
	ctx := t.Context()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := lis.Addr().String()

	exec := &mockExecutorServer{}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	router := NewServer(fake.NewSimpleClientset(), "default", "img", "")
	modelKey := "test-model"
	router.placedExternally(modelKey, addr)

	// Establish connection.
	if _, err := router.TokenizeChat(ctx, &pb.TokenizeChatRequest{Model: modelKey}); err != nil {
		t.Fatal(err)
	}

	router.mu.RLock()
	connsCount := len(router.conns)
	router.mu.RUnlock()
	if connsCount != 1 {
		t.Fatalf("expected 1 pooled connection, got %d", connsCount)
	}

	// Connections are closed when their executor is removed.
	router.RemoveExecutor(ctx, addr)

	router.mu.RLock()
	connsCount = len(router.conns)
	router.mu.RUnlock()
	if connsCount != 0 {
		t.Errorf("expected 0 connections after RemoveExecutor, got %d", connsCount)
	}

	// Re-establish connection.
	if _, err := router.TokenizeChat(ctx, &pb.TokenizeChatRequest{Model: modelKey}); err != nil {
		t.Fatal(err)
	}

	router.mu.RLock()
	connsCount = len(router.conns)
	router.mu.RUnlock()
	if connsCount != 1 {
		t.Fatalf("expected 1 pooled connection, got %d", connsCount)
	}

	// Connections are closed when the router shuts down; a test shows none is left.
	if err := router.Close(); err != nil {
		t.Fatalf("router.Close failed: %v", err)
	}

	router.mu.RLock()
	connsCount = len(router.conns)
	router.mu.RUnlock()
	if connsCount != 0 {
		t.Errorf("expected 0 connections after router.Close, got %d", connsCount)
	}
}

type streamBlockerExecutor struct {
	pb.UnimplementedExecutorServer
	streamStarted chan struct{}
	releaseStream chan struct{}
}

func (e *streamBlockerExecutor) Describe(ctx context.Context, req *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	return &pb.DescribeResponse{}, nil
}

func (e *streamBlockerExecutor) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *streamBlockerExecutor) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	if err := stream.Send(&pb.ExecuteResponse{Response: &pb.GenerateResponse{TokenIds: []int32{1}}}); err != nil {
		return err
	}
	close(e.streamStarted)
	select {
	case <-e.releaseStream:
	case <-stream.Context().Done():
		return stream.Context().Err()
	}
	return stream.Send(&pb.ExecuteResponse{Response: &pb.GenerateResponse{TokenIds: []int32{2}, Done: true}})
}

// Loading an artifact that is already served leaves the pooled connection open
// so requests in flight on it are not canceled.
func TestLoadArtifact_PreservesInFlightStream(t *testing.T) {
	ctx := t.Context()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	exec := &streamBlockerExecutor{
		streamStarted: make(chan struct{}),
		releaseStream: make(chan struct{}),
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	manifestJSON := `{}`
	modelKey := artifactKey(manifestJSON)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-executor-" + modelKey,
			Namespace: "default",
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	router := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	router.executorPort = lis.Addr().(*net.TCPAddr).Port

	// Initial load places and connects to the executor.
	if _, err := router.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifestJSON}); err != nil {
		t.Fatalf("initial LoadModel failed: %v", err)
	}

	endpoint := fmt.Sprintf("127.0.0.1:%d", router.executorPort)
	conn, err := router.executorConn(endpoint)
	if err != nil {
		t.Fatalf("executorConn failed: %v", err)
	}

	stream, err := pb.NewExecutorClient(conn).Execute(ctx, &pb.ExecuteRequest{
		Request: &pb.GenerateRequest{SessionId: modelKey + ":s1", InputIds: []int32{10}},
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Read initial token from stream.
	resp, err := stream.Recv()
	if err != nil {
		t.Fatalf("stream.Recv token 1 failed: %v", err)
	}
	if len(resp.GetResponse().GetTokenIds()) == 0 || resp.GetResponse().GetTokenIds()[0] != 1 {
		t.Fatalf("expected token 1, got %v", resp)
	}

	// Stream is now in flight. Concurrently, a second load runs for the already-served model.
	<-exec.streamStarted

	if _, err := router.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifestJSON}); err != nil {
		t.Fatalf("concurrent LoadModel failed: %v", err)
	}

	// Release stream on executor and verify it completes without being canceled.
	close(exec.releaseStream)

	resp, err = stream.Recv()
	if err != nil {
		t.Fatalf("stream.Recv token 2 failed (connection closed under in-flight stream): %v", err)
	}
	if !resp.GetResponse().GetDone() {
		t.Errorf("expected final token marked done")
	}

	_, err = stream.Recv()
	if err != io.EOF {
		t.Errorf("expected io.EOF at end of stream, got %v", err)
	}
}
