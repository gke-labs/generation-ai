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
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
)

type goldenFakeExecutor struct {
	pb.UnimplementedExecutorServer
	mu           sync.Mutex
	loads        int
	sessions     int
	tokens       [][]int32
	transportErr error
	loadErr      error
	getLoadErr   error
	unloaded     bool
	onExecute    func()
	onLoad       func()
}

func (e *goldenFakeExecutor) Describe(ctx context.Context, req *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	return &pb.DescribeResponse{}, nil
}

func (e *goldenFakeExecutor) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	onLoad := e.onLoad
	e.loads++
	e.unloaded = false
	err := e.loadErr
	e.mu.Unlock()
	if onLoad != nil {
		onLoad()
	}
	if err != nil {
		return nil, err
	}
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *goldenFakeExecutor) GetLoad(ctx context.Context, req *pb.GetLoadRequest) (*pb.GetLoadResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.getLoadErr != nil {
		return nil, e.getLoadErr
	}
	if e.unloaded {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_NONE}, nil
	}
	return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY}, nil
}

func (e *goldenFakeExecutor) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.transportErr != nil {
		return nil, e.transportErr
	}
	if e.unloaded {
		return nil, status.Error(codes.FailedPrecondition, "model not loaded")
	}
	e.sessions++
	return &pb.NewSessionResponse{SessionId: "sess"}, nil
}

func (e *goldenFakeExecutor) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	e.mu.Lock()
	onExec := e.onExecute
	toks := []int32{1, 2, 3}
	if len(e.tokens) > 0 {
		toks = e.tokens[0]
		e.tokens = e.tokens[1:]
	}
	e.mu.Unlock()
	if onExec != nil {
		onExec()
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

// goldenFixture encapsulates a common test environment for model verification tests.
type goldenFixture struct {
	ctx       context.Context
	server    *Server
	exec      *goldenFakeExecutor
	manifest  string
	modelKey  string
	golden    *planpb.GoldenFile
	clientset *fake.Clientset
}

func newGoldenFixture(t *testing.T, modelName string, tokens ...[]int32) *goldenFixture {
	t.Helper()
	fakePeer := &peer.Peer{Addr: &net.TCPAddr{IP: net.ParseIP("127.0.0.1"), Port: 12345}}
	ctx := peer.NewContext(t.Context(), fakePeer)

	exec := &goldenFakeExecutor{tokens: tokens}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts: []*planpb.GoldenPrompt{
			{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}},
		},
	}
	manifest := fmt.Sprintf(`{"model": %q}`, modelName)
	modelKey := artifactKey(manifest)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + modelKey, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	clientset := fake.NewSimpleClientset(pod)
	s := NewServer(clientset, "default", "img", "")
	s.executorPort = lis.Addr().(*net.TCPAddr).Port
	_ = s.putGolden(ctx, modelName, golden)

	return &goldenFixture{
		ctx:       ctx,
		server:    s,
		exec:      exec,
		manifest:  manifest,
		modelKey:  modelKey,
		golden:    golden,
		clientset: clientset,
	}
}

func TestVerifyGolden_Matches(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 20, 30}}}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

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
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	if err := router.verifyGolden(ctx, client, "m1", golden); err != nil {
		t.Fatalf("verifyGolden failed: %v", err)
	}
}

func TestVerifyGolden_TokenMismatchFails(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 999, 30}}}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

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
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	err = router.verifyGolden(ctx, client, "m1", golden)
	if err == nil {
		t.Fatal("expected verifyGolden to fail on token mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}
}

func TestLoadModel_WithGolden_RefusesOnMismatch(t *testing.T) {
	f := newGoldenFixture(t, "test-model", []int32{10, 999, 30})
	_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if err == nil {
		t.Fatal("expected LoadModel to fail on mismatch, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if !strings.Contains(err.Error(), "prompt 0") || !strings.Contains(err.Error(), "position 1") {
		t.Errorf("expected error to name prompt 0 and position 1, got: %v", err)
	}

	_, err = f.server.NewSession(f.ctx, &pb.NewSessionRequest{})
	if err == nil {
		t.Fatal("expected NewSession to fail after refusal, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
}

func TestRegisterModel_WithGolden_RefusesOnMismatch(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 999, 30}}}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: golden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	_, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err == nil {
		t.Fatal("expected NewSession to fail on mismatch, got nil")
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
}

func TestRegisterModel_WithGolden_Succeeds(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 20, 30}}}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: golden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
}

func TestVerifyGolden_CancellationReportedAccurately(t *testing.T) {
	f := newGoldenFixture(t, "test-model-cancel")
	cancelCtx, cancel := context.WithCancel(f.ctx)
	cancel()
	_, err := f.server.LoadModel(cancelCtx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected codes.Canceled, got %v", status.Code(err))
	}
}

func TestModelStanding_ModelFailsThenLaterPassesIsServed(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 999, 30}, {10, 20, 30}}}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	mismatchedGolden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 888, 30}}},
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: mismatchedGolden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	// 1. Initial ask fails because golden check fails
	_, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	// 2. An ask after the refusal is also refused
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected second NewSession to fail with FailedPrecondition, got %v", status.Code(err))
	}

	// 3. Registering matching golden document clears refusal on next load and serves
	validGolden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: validGolden}); err != nil {
		t.Fatalf("RegisterModel with valid golden: %v", err)
	}
	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession should succeed after golden passed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
}

func TestModelStanding_CancelledCallerDoesNotRefuseModel(t *testing.T) {
	f := newGoldenFixture(t, "test-cancel-standing", []int32{10, 20, 30})
	cancelCtx, cancel := context.WithCancel(f.ctx)
	cancel()
	_, err := f.server.LoadModel(cancelCtx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected codes.Canceled, got %v", status.Code(err))
	}
	if st := f.server.modelStanding(f.modelKey); st.standing == standingRefused {
		t.Errorf("cancelled caller should not refuse model, got %v", st.standing)
	}
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel after cancel failed: %v", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Errorf("standing = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_CancelledLeaderAllowsLiveWaiterToSucceed(t *testing.T) {
	var leaderStarted atomic.Bool
	leaderExecuting := make(chan struct{})
	gateLeader := make(chan struct{})

	f := newGoldenFixture(t, "test-leader-cancel", []int32{10, 20, 30}, []int32{10, 20, 30})
	f.exec.onExecute = func() {
		if !leaderStarted.Swap(true) {
			close(leaderExecuting)
			<-gateLeader
		}
	}

	ctxLeader, cancelLeader := context.WithCancel(f.ctx)
	ctxWaiter := f.ctx
	var wg sync.WaitGroup
	var errLeader, errWaiter error

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, errLeader = f.server.LoadModel(ctxLeader, &pb.LoadModelRequest{ManifestJson: f.manifest})
	}()

	<-leaderExecuting

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, errWaiter = f.server.LoadModel(ctxWaiter, &pb.LoadModelRequest{ManifestJson: f.manifest})
	}()

	time.Sleep(20 * time.Millisecond)
	cancelLeader()
	close(gateLeader)
	wg.Wait()

	if status.Code(errLeader) != codes.Canceled {
		t.Errorf("leader should receive Canceled, got %v", errLeader)
	}
	if errWaiter != nil {
		t.Fatalf("waiter should succeed after leader cancelled, got error: %v", errWaiter)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Errorf("standing = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_SessionDuringFirstCheckIsRefused(t *testing.T) {
	executing := make(chan struct{})
	gateExecute := make(chan struct{})
	f := newGoldenFixture(t, "test-session-during-check", []int32{10, 20, 30})
	f.exec.onExecute = func() {
		close(executing)
		<-gateExecute
	}

	var wg sync.WaitGroup
	var loadErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, loadErr = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	}()

	<-executing
	if st := f.server.modelStanding(f.modelKey); st.standing != standingUnchecked {
		t.Errorf("standing during check should be unchecked, got %v", st.standing)
	}

	_, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition during check, got %v", status.Code(err))
	}

	close(gateExecute)
	wg.Wait()

	if loadErr != nil {
		t.Fatalf("LoadModel failed: %v", loadErr)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Fatalf("standing after check should be standingPassed, got %v", st.standing)
	}
	resp, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{})
	if err != nil {
		t.Fatalf("NewSession after check passed failed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
}

func TestModelStanding_TransportErrorDoesNotRefuseModel(t *testing.T) {
	f := newGoldenFixture(t, "test-transport-standing", []int32{10, 20, 30})
	f.exec.transportErr = status.Error(codes.Unavailable, "network connection dropped")

	_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.Unavailable {
		t.Errorf("expected Unavailable, got %v", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing == standingRefused {
		t.Errorf("transport error should not refuse model, got %v", st.standing)
	}

	f.exec.mu.Lock()
	f.exec.transportErr = nil
	f.exec.mu.Unlock()

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel after recovery failed: %v", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Errorf("standing = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_LoadsOfOneModelDoNotOverlap(t *testing.T) {
	f := newGoldenFixture(t, "test-overlap", []int32{10, 20, 30})
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("caller %d failed: %v", i, err)
		}
	}
	if f.exec.loads != 1 {
		t.Errorf("expected 1 load across concurrent calls, got %d", f.exec.loads)
	}
}

func TestModelStanding_ReloadServingModelIsNoOp(t *testing.T) {
	f := newGoldenFixture(t, "test-reload-noop", []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}
	if f.exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", f.exec.loads)
	}
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("second LoadModel: %v", err)
	}
	if f.exec.loads != 1 {
		t.Errorf("reload of serving model should be no-op, got loads = %d", f.exec.loads)
	}
}

func TestModelStanding_ExecutorUnloadRequiresReload(t *testing.T) {
	f := newGoldenFixture(t, "test-executor-unload", []int32{10, 20, 30}, []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("initial LoadModel: %v", err)
	}
	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); err != nil {
		t.Fatalf("first NewSession: %v", err)
	}

	f.exec.mu.Lock()
	f.exec.unloaded = true
	f.exec.mu.Unlock()

	_, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("NewSession after unload: got %v, want FailedPrecondition", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingUnchecked {
		t.Errorf("standing after unload should be unchecked, got %v", st.standing)
	}

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel after unload failed: %v", err)
	}
	if f.exec.loads != 2 {
		t.Errorf("expected 2 loads after unload, got %d", f.exec.loads)
	}
	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); err != nil {
		t.Fatalf("NewSession after reload failed: %v", err)
	}
}

func TestModelStanding_ExecutorGoneAllowsReload(t *testing.T) {
	f := newGoldenFixture(t, "test-executor-gone", []int32{10, 20, 30}, []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}

	podName := "vxpu-executor-" + f.modelKey
	_ = f.clientset.CoreV1().Pods("default").Delete(f.ctx, podName, metav1.DeleteOptions{})

	_, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("NewSession after executor gone: got %v, want FailedPrecondition", err)
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	_, _ = f.clientset.CoreV1().Pods("default").Create(f.ctx, pod, metav1.CreateOptions{})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("reload after executor gone failed: %v", err)
	}
	if f.exec.loads != 2 {
		t.Errorf("expected 2 loads after executor gone, got %d", f.exec.loads)
	}
}

func TestModelStanding_ExternalExecutorEvictedModelNotPassed(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{
		tokens: [][]int32{
			{10, 20, 30},  // Model A passes
			{10, 999, 30}, // Model B fails
		},
	}
	s := NewServer(fake.NewSimpleClientset(), "default", "img", "")
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	goldenA := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	goldenB := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}

	manifestA := `{"model": "test-external-model-a"}`
	manifestB := `{"model": "test-external-model-b"}`
	keyA := artifactKey(manifestA)
	keyB := artifactKey(manifestB)

	_ = s.putGolden(ctx, "test-external-model-a", goldenA)
	_ = s.putGolden(ctx, "test-external-model-b", goldenB)

	// 1. Model A loads and passes on external executor
	if _, err := s.loadArtifact(ctx, &pb.LoadModelRequest{ManifestJson: manifestA}, true, nil); err != nil {
		t.Fatalf("loadArtifact A: %v", err)
	}
	if st := s.modelStanding(keyA); st.standing != standingPassed {
		t.Fatalf("Model A standing = %v, want standingPassed", st.standing)
	}

	// 2. Model B loads on the same external executor, evicting Model A, and fails verification
	_, err := s.loadArtifact(ctx, &pb.LoadModelRequest{ManifestJson: manifestB}, true, nil)
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition for Model B, got %v", status.Code(err))
	}
	if st := s.modelStanding(keyB); st.standing != standingRefused {
		t.Errorf("Model B standing = %v, want standingRefused", st.standing)
	}

	// 3. Model A must NOT remain in standingPassed, and must NOT be routed to the external executor
	if st := s.modelStanding(keyA); st.standing == standingPassed {
		t.Errorf("Model A standing should no longer be standingPassed after eviction, got %v", st.standing)
	}
	if _, err := s.executorEndpoint(ctx, keyA); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("executorEndpoint for Model A after eviction: got %v, want FailedPrecondition", err)
	}

	// 4. Subsequent load of Model A on the external executor reloads Model A and passes
	exec.mu.Lock()
	exec.tokens = append(exec.tokens, []int32{10, 20, 30})
	exec.mu.Unlock()

	if _, err := s.loadArtifact(ctx, &pb.LoadModelRequest{ManifestJson: manifestA}, true, nil); err != nil {
		t.Fatalf("loadArtifact A after eviction failed: %v", err)
	}
	if st := s.modelStanding(keyA); st.standing != standingPassed {
		t.Errorf("Model A standing after reload = %v, want standingPassed", st.standing)
	}
}

func TestModelStanding_PodLossOnByNamePathReloads(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 20, 30}, {10, 20, 30}}}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}

	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				pods, err := clientset.CoreV1().Pods("ns").List(ctx, metav1.ListOptions{})
				if err == nil {
					for _, p := range pods.Items {
						if p.Status.Phase != corev1.PodRunning {
							p.Status.Phase = corev1.PodRunning
							p.Status.PodIP = "127.0.0.1"
							_, _ = clientset.CoreV1().Pods("ns").UpdateStatus(ctx, &p, metav1.UpdateOptions{})
						}
					}
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
	}()

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: golden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("first NewSession by name: %v", err)
	}
	key, _, err := splitSessionID(resp.SessionId)
	if err != nil {
		t.Fatalf("splitSessionID: %v", err)
	}
	podName := "vxpu-executor-" + key

	_ = clientset.CoreV1().Pods("ns").Delete(ctx, podName, metav1.DeleteOptions{})

	resp2, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession after pod loss should reload and succeed, got: %v", err)
	}
	if resp2.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
}

func TestModelStanding_RefusedModelWithPlanDoesNotReloadEveryCall(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 999, 30}, {10, 20, 30}}}
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	mismatchedGolden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 888, 30}}},
	}
	plan := map[string]string{"VXPU_EXPERT_MAILBOXES": "1"}

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:  "org/M",
		Plan:   plan,
		Golden: mismatchedGolden,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	_, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", status.Code(err))
	}
	if exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", exec.loads)
	}

	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected second NewSession to fail with FailedPrecondition, got %v", status.Code(err))
	}
	if exec.loads != 1 {
		t.Errorf("second ask on refused model should not reload; loads = %d, want 1", exec.loads)
	}

	validGolden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{
		Model:  "org/M",
		Plan:   plan,
		Golden: validGolden,
	}); err != nil {
		t.Fatalf("RegisterModel with valid golden: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession should succeed after golden passed: %v", err)
	}
	if resp.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
	if exec.loads != 2 {
		t.Errorf("expected 2 loads after golden update, got %d", exec.loads)
	}
}

func TestModelStanding_ServingEndpointTransientErrorDoesNotForgetModel(t *testing.T) {
	f := newGoldenFixture(t, "test-transient-error", []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("initial LoadModel: %v", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Fatalf("standing = %v, want standingPassed", st.standing)
	}

	cancelCtx, cancel := context.WithCancel(f.ctx)
	cancel()
	_, err := f.server.LoadModel(cancelCtx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.Canceled {
		t.Errorf("expected Canceled, got %v", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Errorf("transient error should not reset standingPassed, got %v", st.standing)
	}
}

func TestModelStanding_ReloadTransportErrorDoesNotForgetModel(t *testing.T) {
	f := newGoldenFixture(t, "test-reload-transport", []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("initial LoadModel: %v", err)
	}

	f.exec.mu.Lock()
	f.exec.getLoadErr = status.Error(codes.Unavailable, "executor connection dropped")
	f.exec.mu.Unlock()

	_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.Unavailable {
		t.Errorf("LoadModel during transport error: got %v, want Unavailable", err)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingPassed {
		t.Errorf("standing after transport error = %v, want standingPassed", st.standing)
	}
	if f.exec.loads != 1 {
		t.Errorf("loads after transport error = %d, want 1", f.exec.loads)
	}

	f.exec.mu.Lock()
	f.exec.getLoadErr = nil
	f.exec.mu.Unlock()

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel after transport recovery failed: %v", err)
	}
	if f.exec.loads != 1 {
		t.Errorf("loads after recovery = %d, want 1", f.exec.loads)
	}
}

func TestModelStanding_GoldenLookupFailureDoesNotForgetModel(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 20, 30}}}
	ms := &fakeModelstore{compiled: true}
	s, _ := newCatalogServer(t, ms)
	s.UseExternalExecutors(ctx, []string{startExecutor(t, exec)})

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: golden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("initial NewSession: %v", err)
	}
	key, _, err := splitSessionID(resp.SessionId)
	if err != nil {
		t.Fatalf("splitSessionID: %v", err)
	}
	if st := s.modelStanding(key); st.standing != standingPassed {
		t.Fatalf("model standing = %v, want standingPassed", st.standing)
	}
	if exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", exec.loads)
	}

	// Clear in-memory golden cache and make store fail golden queries
	s.mu.Lock()
	delete(s.goldens, statsKey("org/M"))
	s.mu.Unlock()
	ms.mu.Lock()
	ms.failGolden = true
	ms.mu.Unlock()

	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.Unavailable {
		t.Errorf("NewSession during store failure: got %v, want Unavailable", err)
	}
	if st := s.modelStanding(key); st.standing != standingPassed {
		t.Errorf("standing after golden store failure = %v, want standingPassed", st.standing)
	}

	ms.mu.Lock()
	ms.failGolden = false
	ms.mu.Unlock()

	resp2, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession after store recovery failed: %v", err)
	}
	if resp2.SessionId == "" {
		t.Errorf("expected valid session ID")
	}
	if exec.loads != 1 {
		t.Errorf("expected loads to remain 1, got %d", exec.loads)
	}
}

func TestModelStanding_ConcurrentLoadsShareGoldenVerdict(t *testing.T) {
	f := newGoldenFixture(t, "test-verdict-share", []int32{10, 999, 30})
	const callers = 4
	var wg sync.WaitGroup
	errs := make([]error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}
	wg.Wait()
	for i := 0; i < callers; i++ {
		if status.Code(errs[i]) != codes.FailedPrecondition {
			t.Errorf("caller %d got code %v, want FailedPrecondition", i, status.Code(errs[i]))
		}
	}
	if f.exec.loads != 1 {
		t.Errorf("expected exactly 1 load across concurrent callers sharing verdict, got %d", f.exec.loads)
	}
	if st := f.server.modelStanding(f.modelKey); st.standing != standingRefused {
		t.Errorf("standing = %v, want standingRefused", st.standing)
	}
}

func TestModelStanding_FailedPreconditionWithoutMismatchIsNotSharedAsVerdict(t *testing.T) {
	f := newGoldenFixture(t, "test-non-verdict-fp", []int32{10, 20, 30})
	var attemptCount atomic.Int32
	f.exec.onLoad = func() {
		if attemptCount.Add(1) == 1 {
			f.exec.loadErr = status.Error(codes.FailedPrecondition, "device temporarily busy")
		} else {
			f.exec.loadErr = nil
		}
	}

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}
	wg.Wait()

	// Waiter must NOT share leader's non-verdict FailedPrecondition; it retries and succeeds
	successes := 0
	for _, err := range errs {
		if err == nil {
			successes++
		}
	}
	if successes == 0 {
		t.Errorf("expected at least one caller to succeed on retry after non-verdict FailedPrecondition, got errs: %v", errs)
	}
}

func TestModelStanding_ByNamePathAPIServerErrorDoesNotForgetModel(t *testing.T) {
	ctx := t.Context()
	exec := &goldenFakeExecutor{tokens: [][]int32{{10, 20, 30}}}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, exec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}

	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.executorPort = lis.Addr().(*net.TCPAddr).Port

	var failPodGet atomic.Bool
	clientset.PrependReactor("get", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		if failPodGet.Load() {
			return true, nil, apierrors.NewInternalError(errors.New("api server connection lost"))
		}
		return false, nil, nil
	})

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				pods, err := clientset.CoreV1().Pods("ns").List(ctx, metav1.ListOptions{})
				if err == nil {
					for _, p := range pods.Items {
						if p.Status.Phase != corev1.PodRunning {
							p.Status.Phase = corev1.PodRunning
							p.Status.PodIP = "127.0.0.1"
							_, _ = clientset.CoreV1().Pods("ns").UpdateStatus(ctx, &p, metav1.UpdateOptions{})
						}
					}
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
	}()

	if _, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M", Golden: golden}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("initial NewSession: %v", err)
	}
	key, _, err := splitSessionID(resp.SessionId)
	if err != nil {
		t.Fatalf("splitSessionID: %v", err)
	}
	if st := s.modelStanding(key); st.standing != standingPassed {
		t.Fatalf("standing = %v, want standingPassed", st.standing)
	}
	if exec.loads != 1 {
		t.Fatalf("expected 1 load, got %d", exec.loads)
	}

	// API server fails pod lookups
	failPodGet.Store(true)

	// NewSession returns Internal and must NOT forget the serving model
	_, err = s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if status.Code(err) != codes.Internal {
		t.Errorf("NewSession during API server error: got %v, want Internal", err)
	}
	if st := s.modelStanding(key); st.standing != standingPassed {
		t.Errorf("standing after API server error = %v, want standingPassed", st.standing)
	}

	// API server recovers: next NewSession succeeds without reloading
	failPodGet.Store(false)

	resp2, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession after recovery failed: %v", err)
	}
	if resp2.SessionId == "" {
		t.Errorf("expected non-empty SessionId")
	}
	if exec.loads != 1 {
		t.Errorf("expected loads to remain 1, got %d", exec.loads)
	}
}
