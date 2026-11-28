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
	"sync/atomic"
	"testing"
	"time"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/workqueue"
)

type reconcilerFakeExecutor struct {
	pb.UnimplementedExecutorServer
	mu           sync.Mutex
	loads        int
	sessions     int
	tokens       [][]int32
	loadErr      error
	getLoadErr   error
	execErr      error
	state        pb.LoadState
	lastPlan     map[string]string
	lastPlanHash string
	unloaded     bool
	lastModel    string
	modelStates  map[string]pb.LoadState
}

func (e *reconcilerFakeExecutor) Describe(ctx context.Context, req *pb.DescribeRequest) (*pb.DescribeResponse, error) {
	return &pb.DescribeResponse{
		HardwareJson: `{"cpu": "fake-cpu", "cpu_threads": 4, "ram_bytes": 1000000000}`,
	}, nil
}

func (e *reconcilerFakeExecutor) GetLoad(ctx context.Context, req *pb.GetLoadRequest) (*pb.GetLoadResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.getLoadErr != nil {
		return nil, e.getLoadErr
	}
	if e.unloaded {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_NONE}, nil
	}
	if e.modelStates != nil {
		if st, ok := e.modelStates[e.lastModel]; ok {
			return &pb.GetLoadResponse{State: st, PlanHash: e.lastPlanHash}, nil
		}
	}
	if e.state != pb.LoadState_LOAD_STATE_NONE {
		return &pb.GetLoadResponse{State: e.state, PlanHash: e.lastPlanHash}, nil
	}
	if e.loads == 0 {
		return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_NONE}, nil
	}
	return &pb.GetLoadResponse{State: pb.LoadState_LOAD_STATE_READY, PlanHash: e.lastPlanHash}, nil
}

func (e *reconcilerFakeExecutor) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.loads++
	e.unloaded = false
	e.lastPlan = req.Plan
	e.lastPlanHash = req.GetPlanFile().GetPlanHash()
	m := modelFromManifest(req.ManifestJson)
	e.lastModel = m
	if e.loadErr != nil {
		return nil, e.loadErr
	}
	if e.modelStates == nil {
		e.modelStates = make(map[string]pb.LoadState)
	}
	if _, ok := e.modelStates[m]; !ok {
		e.modelStates[m] = pb.LoadState_LOAD_STATE_READY
	}
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *reconcilerFakeExecutor) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.sessions++
	return &pb.NewSessionResponse{SessionId: "test-session"}, nil
}

func (e *reconcilerFakeExecutor) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.execErr != nil {
		return e.execErr
	}
	toks := []int32{10, 20, 30}
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

type reconcilerFakeAgent struct {
	pb.UnimplementedAgentServer
	mu                  sync.Mutex
	endpoint            string
	processStatus       *pb.ProcessStatus
	startCalls          int
	stopCalls           int
	stopErr             error
	getProcessErr       error
	getProcessCalls     int
	failGetProcessAfter int
	stagePkgCalls       []*pb.StagePackagesRequest
	stagePkgErrors      []error
	softwareVersion     string
}

func (f *reconcilerFakeAgent) StartProcess(ctx context.Context, req *pb.StartProcessRequest) (*pb.StartProcessResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.startCalls++
	ep := f.endpoint
	if ep == "" {
		ep = "127.0.0.1:50051"
	}
	pst := &pb.ProcessStatus{
		Id:       req.Id,
		Process:  req.Process,
		State:    "running",
		Pid:      1234,
		Endpoint: ep,
	}
	f.processStatus = pst
	return &pb.StartProcessResponse{
		AlreadyRunning: false,
		Status:         pst,
	}, nil
}

func (f *reconcilerFakeAgent) StopProcess(ctx context.Context, req *pb.StopProcessRequest) (*pb.StopProcessResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopCalls++
	if f.stopErr != nil {
		return nil, f.stopErr
	}
	f.processStatus = nil
	return &pb.StopProcessResponse{}, nil
}

func (f *reconcilerFakeAgent) GetProcess(ctx context.Context, req *pb.GetProcessRequest) (*pb.GetProcessResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.getProcessCalls++
	if f.getProcessErr != nil && (f.failGetProcessAfter == 0 || f.getProcessCalls > f.failGetProcessAfter) {
		return nil, f.getProcessErr
	}
	if f.processStatus == nil {
		return nil, status.Errorf(codes.NotFound, "process %q not found", req.Id)
	}
	return &pb.GetProcessResponse{Status: f.processStatus}, nil
}

func (f *reconcilerFakeAgent) StagePackages(ctx context.Context, req *pb.StagePackagesRequest) (*pb.StagePackagesResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stagePkgCalls = append(f.stagePkgCalls, req)
	if len(f.stagePkgErrors) > 0 {
		err := f.stagePkgErrors[0]
		f.stagePkgErrors = f.stagePkgErrors[1:]
		if err != nil {
			return nil, err
		}
	}
	return &pb.StagePackagesResponse{}, nil
}

func (f *reconcilerFakeAgent) DescribeMachine(_ context.Context, _ *pb.DescribeMachineRequest) (*pb.Probe, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return &pb.Probe{SoftwareVersion: f.softwareVersion}, nil
}

type reconcilerFixture struct {
	server      *Server
	reconciler  *ModelReconciler
	exec        *reconcilerFakeExecutor
	agent       *reconcilerFakeAgent
	instanceKey string
	desired     InstanceDesired
	execAddr    string
	limiter     workqueue.TypedRateLimiter[string]
}

func newReconcilerFixture(t *testing.T, golden *planpb.GoldenFile) *reconcilerFixture {
	t.Helper()

	exec := &reconcilerFakeExecutor{}
	execLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("exec net.Listen: %v", err)
	}
	execSrv := grpc.NewServer()
	pb.RegisterExecutorServer(execSrv, exec)
	go func() { _ = execSrv.Serve(execLis) }()
	t.Cleanup(func() {
		execSrv.Stop()
		_ = execLis.Close()
	})

	agent := &reconcilerFakeAgent{endpoint: execLis.Addr().String()}
	agentLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("agent net.Listen: %v", err)
	}
	agentSrv := grpc.NewServer()
	pb.RegisterAgentServer(agentSrv, agent)
	go func() { _ = agentSrv.Serve(agentLis) }()
	t.Cleanup(func() {
		agentSrv.Stop()
		_ = agentLis.Close()
	})

	instanceKey := "test-instance-key"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + instanceKey, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	clientset := fake.NewSimpleClientset(pod)
	s := NewServer(clientset, "default", "img", "")
	s.executorPort = execLis.Addr().(*net.TCPAddr).Port
	s.agentDialer = func(string) (*grpc.ClientConn, error) {
		return grpc.NewClient(agentLis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	fastLimiter := workqueue.NewTypedItemExponentialFailureRateLimiter[string](2*time.Millisecond, 20*time.Millisecond)
	recon := NewModelReconciler(s, WithRateLimiter(fastLimiter), WithPollInterval(2*time.Millisecond))

	desired := InstanceDesired{
		Key:     instanceKey,
		Model:   "m",
		Request: &pb.LoadModelRequest{ManifestJson: `{"model":"m"}`},
		Golden:  golden,
	}

	return &reconcilerFixture{
		server:      s,
		reconciler:  recon,
		exec:        exec,
		agent:       agent,
		instanceKey: instanceKey,
		desired:     desired,
		execAddr:    execLis.Addr().String(),
		limiter:     fastLimiter,
	}
}

// Drive through: not loaded → loading → checking → passed.
func TestModelReconciler_SuccessPath_NotLoadedToPassed(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	f := newReconcilerFixture(t, golden)
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	if st := f.reconciler.Status(f.instanceKey); st.Phase != InstancePhaseNotLoaded {
		t.Fatalf("initial phase = %v, want %v", st.Phase, InstancePhaseNotLoaded)
	}

	f.reconciler.SetDesired(f.desired)

	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhasePassed {
			if st.LastError != nil {
				t.Errorf("expected LastError nil, got %v", st.LastError)
			}
			if !st.NextAttempt.IsZero() {
				t.Errorf("expected NextAttempt zero on passed, got %v", st.NextAttempt)
			}
			if st.ObservedGeneration != 1 {
				t.Errorf("ObservedGeneration = %d, want 1", st.ObservedGeneration)
			}
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for passed; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	loads := f.exec.loads
	f.exec.mu.Unlock()
	if loads != 1 {
		t.Errorf("loads = %d, want 1", loads)
	}
}

// A changed desired state is detected and applied on the executor.
func TestModelReconciler_ChangedDesiredState_AppliesNewLoad(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	f.reconciler.SetDesired(f.desired)

	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for initial passed; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	loadsBefore := f.exec.loads
	f.exec.mu.Unlock()
	if loadsBefore != 1 {
		t.Fatalf("expected 1 load, got %d", loadsBefore)
	}

	// Update desired state with a different execution plan.
	newDesired := f.desired
	newDesired.Request = &pb.LoadModelRequest{
		ManifestJson: `{"model":"m"}`,
		Plan:         map[string]string{"VXPU_EXPERT_MAILBOXES": "1"},
	}
	f.reconciler.SetDesired(newDesired)

	// Wait for passed on generation 2.
	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhasePassed && st.ObservedGeneration == 2 {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for generation 2 passed; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	loadsAfter := f.exec.loads
	plan := f.exec.lastPlan
	f.exec.mu.Unlock()

	if loadsAfter != 2 {
		t.Errorf("expected 2 loads after desired state change, got %d", loadsAfter)
	}
	if plan["VXPU_EXPERT_MAILBOXES"] != "1" {
		t.Errorf("executor did not receive new plan: %v", plan)
	}
}

// A stale run cannot overwrite the status of a newer desired state generation.
func TestModelReconciler_StaleRunResultDiscarded(t *testing.T) {
	f := newReconcilerFixture(t, nil)
	f.reconciler.SetDesired(f.desired) // generation 1
	d2 := f.desired
	d2.Request = &pb.LoadModelRequest{
		ManifestJson: `{"model":"m2"}`,
	}
	f.reconciler.SetDesired(d2) // generation 2

	f.reconciler.setStatus(t.Context(), f.instanceKey, InstanceStatus{
		Phase:              InstancePhasePassed,
		ObservedGeneration: 1,
	})

	st := f.reconciler.Status(f.instanceKey)
	if st.ObservedGeneration != 2 || st.Phase != InstancePhaseNotLoaded {
		t.Errorf("status = %v (gen %d), want phase %v (gen 2)", st.Phase, st.ObservedGeneration, InstancePhaseNotLoaded)
	}
}

// Reconciling an already-passed instance does not repeat golden verification.
func TestModelReconciler_ReconcilingPassedInstanceDoesNotRepeatGoldenCheck(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	f := newReconcilerFixture(t, golden)
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	f.reconciler.SetDesired(f.desired)

	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for passed; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	sessionsBefore := f.exec.sessions
	f.exec.mu.Unlock()
	if sessionsBefore != 1 {
		t.Fatalf("expected 1 session for golden check, got %d", sessionsBefore)
	}

	for i := 0; i < 5; i++ {
		f.reconciler.Enqueue(f.instanceKey)
		time.Sleep(10 * time.Millisecond)
	}

	f.exec.mu.Lock()
	sessionsAfter := f.exec.sessions
	f.exec.mu.Unlock()

	if sessionsAfter != sessionsBefore {
		t.Errorf("golden verification was repeated on passed instance: %d -> %d", sessionsBefore, sessionsAfter)
	}
}

// Asynchronous loading reports loading without blocking other instances from reconciling.
func TestModelReconciler_AsynchronousLoadingDoesNotBlockOtherInstances(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.exec.mu.Lock()
	f.exec.state = pb.LoadState_LOAD_STATE_LOADING
	f.exec.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	// Step 1: observes loading state.
	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	st := f.reconciler.Status(f.instanceKey)
	if st.Phase != InstancePhaseLoading {
		t.Errorf("status = %v, want %v", st.Phase, InstancePhaseLoading)
	}

	// Instance B runs on its own pod (127.0.0.2) with its own executor and agent.
	execB := &reconcilerFakeExecutor{state: pb.LoadState_LOAD_STATE_READY}
	execLisB, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("execLisB: %v", err)
	}
	execSrvB := grpc.NewServer()
	pb.RegisterExecutorServer(execSrvB, execB)
	go func() { _ = execSrvB.Serve(execLisB) }()
	t.Cleanup(func() {
		execSrvB.Stop()
		_ = execLisB.Close()
	})

	agentB := &reconcilerFakeAgent{endpoint: execLisB.Addr().String()}
	agentLisB, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("agentLisB: %v", err)
	}
	agentSrvB := grpc.NewServer()
	pb.RegisterAgentServer(agentSrvB, agentB)
	go func() { _ = agentSrvB.Serve(agentLisB) }()
	t.Cleanup(func() {
		agentSrvB.Stop()
		_ = agentLisB.Close()
	})

	origDialer := f.server.agentDialer
	f.server.agentDialer = func(host string) (*grpc.ClientConn, error) {
		if strings.HasPrefix(host, "127.0.0.2") {
			return grpc.NewClient(agentLisB.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		}
		return origDialer(host)
	}

	instanceBKey := "instance-b-key"
	podB := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + instanceBKey, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.2"},
	}
	_, _ = f.server.clientset.CoreV1().Pods("default").Create(ctx, podB, metav1.CreateOptions{})

	desiredB := InstanceDesired{
		Key:     instanceBKey,
		Model:   "mB",
		Request: &pb.LoadModelRequest{ManifestJson: `{"model":"mB"}`},
	}

	f.reconciler.SetDesired(desiredB)
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	for {
		if f.reconciler.Status(instanceBKey).Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for instance B to pass; last status: %v", f.reconciler.Status(instanceBKey))
		case <-time.After(5 * time.Millisecond):
		}
	}

	stA := f.reconciler.Status(f.instanceKey)
	if stA.Phase != InstancePhaseLoading {
		t.Errorf("instance A phase = %v, want %v", stA.Phase, InstancePhaseLoading)
	}
}

// Runtime crash during golden checks marks the instance failed with error and next attempt.
func TestModelReconciler_RuntimeCrashDuringGoldenCheckMarksFailed(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	f := newReconcilerFixture(t, golden)

	f.exec.mu.Lock()
	f.exec.execErr = status.Error(codes.Unavailable, "runtime crashed")
	f.exec.mu.Unlock()

	f.agent.mu.Lock()
	f.agent.processStatus = &pb.ProcessStatus{
		Id:     "executor",
		State:  "killed by signal 9",
		Output: "OOMKilled during golden inference",
	}
	f.agent.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	st := f.reconciler.Status(f.instanceKey)
	if st.Phase != InstancePhaseFailed {
		t.Fatalf("phase = %v, want %v", st.Phase, InstancePhaseFailed)
	}
	if !strings.Contains(st.LastError.Error(), "killed by signal 9") || !strings.Contains(st.LastError.Error(), "OOMKilled") {
		t.Errorf("LastError = %v, want signal 9 and OOMKilled", st.LastError)
	}
	if st.NextAttempt.IsZero() {
		t.Error("expected non-zero NextAttempt for failed instance")
	}
}

// Refused plan stops retrying.
func TestModelReconciler_RefusedPlan_NoFurtherAttempts(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.exec.mu.Lock()
	f.exec.loadErr = status.Error(codes.InvalidArgument, "invalid plan switches")
	f.exec.mu.Unlock()

	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	f.reconciler.SetDesired(f.desired)

	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhaseRefused {
			if !strings.Contains(st.LastError.Error(), "invalid plan switches") {
				t.Errorf("LastError %v does not contain refusal reason", st.LastError)
			}
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for refused; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	initialLoads := f.exec.loads
	f.exec.mu.Unlock()

	for i := 0; i < 5; i++ {
		f.reconciler.Enqueue(f.instanceKey)
		time.Sleep(10 * time.Millisecond)
	}

	f.exec.mu.Lock()
	afterLoads := f.exec.loads
	f.exec.mu.Unlock()

	if afterLoads != initialLoads {
		t.Errorf("refused plan had further attempts: %d -> %d", initialLoads, afterLoads)
	}
}

// Golden mismatch stops retrying.
func TestModelReconciler_GoldenMismatch_RefusedWithNoFurtherAttempts(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	golden := &planpb.GoldenFile{
		Format:          "vxpu-golden/1",
		TokensPerPrompt: 3,
		Prompts:         []*planpb.GoldenPrompt{{Ids: []int32{1, 2}, Tokens: []int32{10, 20, 30}}},
	}
	f := newReconcilerFixture(t, golden)
	f.exec.mu.Lock()
	f.exec.tokens = [][]int32{{10, 999, 30}}
	f.exec.mu.Unlock()

	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	f.reconciler.SetDesired(f.desired)

	for {
		st := f.reconciler.Status(f.instanceKey)
		if st.Phase == InstancePhaseRefused {
			if !strings.Contains(st.LastError.Error(), "position 1") {
				t.Errorf("LastError %v does not contain mismatch info", st.LastError)
			}
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for refused; last status: %v", st)
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	initialLoads := f.exec.loads
	f.exec.mu.Unlock()

	for i := 0; i < 5; i++ {
		f.reconciler.Enqueue(f.instanceKey)
		time.Sleep(10 * time.Millisecond)
	}

	f.exec.mu.Lock()
	afterLoads := f.exec.loads
	f.exec.mu.Unlock()

	if afterLoads != initialLoads {
		t.Errorf("golden mismatch had further attempts: %d -> %d", initialLoads, afterLoads)
	}
}

// Runtime dies with exit status, then recovers to passed on later attempt.
func TestModelReconciler_RuntimeDiesThenRecovers(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)

	f.agent.mu.Lock()
	f.agent.processStatus = &pb.ProcessStatus{
		Id:     "executor",
		State:  "killed by signal 9",
		Output: "fatal: OOMKilled",
	}
	f.agent.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	st := f.reconciler.Status(f.instanceKey)
	if st.Phase != InstancePhaseFailed {
		t.Fatalf("phase = %v, want %v", st.Phase, InstancePhaseFailed)
	}
	if !strings.Contains(st.LastError.Error(), "killed by signal 9") || !strings.Contains(st.LastError.Error(), "OOMKilled") {
		t.Errorf("LastError = %v, want signal 9 and OOMKilled", st.LastError)
	}
	if st.NextAttempt.IsZero() {
		t.Error("expected non-zero NextAttempt for failed instance")
	}

	f.agent.mu.Lock()
	stopCalls := f.agent.stopCalls
	f.agent.mu.Unlock()
	if stopCalls == 0 {
		t.Errorf("expected StopProcess call on agent for dead runtime")
	}

	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	for {
		current := f.reconciler.Status(f.instanceKey)
		if current.Phase == InstancePhasePassed {
			if current.LastError != nil {
				t.Errorf("recovered instance has non-nil LastError: %v", current.LastError)
			}
			if !current.NextAttempt.IsZero() {
				t.Errorf("recovered instance has non-zero NextAttempt: %v", current.NextAttempt)
			}
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for recovery to passed; last status: %v", current)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// Dynamic port connections are cleaned up on runtime crash rather than leaked.
func TestModelReconciler_DynamicPortConnectionCleanedUpOnCrash(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.reconciler.SetDesired(f.desired)

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	f.server.mu.RLock()
	connsBefore := len(f.server.conns)
	f.server.mu.RUnlock()
	if connsBefore == 0 {
		t.Fatal("expected at least one pooled connection")
	}

	f.agent.mu.Lock()
	f.agent.processStatus = &pb.ProcessStatus{
		Id:     "executor",
		State:  "exited with code 1",
		Output: "segmentation fault",
	}
	f.agent.mu.Unlock()

	f.reconciler.Enqueue(f.instanceKey)

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	f.server.mu.RLock()
	_, connExists := f.server.conns[f.execAddr]
	f.server.mu.RUnlock()
	if connExists {
		t.Errorf("connection at dynamic port %s was leaked after runtime crash", f.execAddr)
	}
}

// Transport error leaves status unchanged and requeues.
func TestModelReconciler_TransportErrorLeavesStatusUnchangedAndRequeues(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)

	f.agent.mu.Lock()
	f.agent.processStatus = &pb.ProcessStatus{
		Id:       "executor",
		Process:  "executor",
		State:    "running",
		Endpoint: f.agent.endpoint,
	}
	f.agent.mu.Unlock()

	f.exec.mu.Lock()
	f.exec.getLoadErr = status.Error(codes.Unavailable, "executor network unreachable")
	f.exec.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	initialStatus := f.reconciler.Status(f.instanceKey)
	if initialStatus.Phase != InstancePhaseNotLoaded {
		t.Fatalf("initial phase = %v, want %v", initialStatus.Phase, InstancePhaseNotLoaded)
	}

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	afterErrStatus := f.reconciler.Status(f.instanceKey)
	if afterErrStatus.Phase != initialStatus.Phase {
		t.Errorf("transport error changed phase: was %v, now %v", initialStatus.Phase, afterErrStatus.Phase)
	}

	f.exec.mu.Lock()
	f.exec.getLoadErr = nil
	f.exec.mu.Unlock()

	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	for {
		current := f.reconciler.Status(f.instanceKey)
		if current.Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for passed after transport recovery; last status: %v", current)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// Concurrent reconciles of the same instance do not overlap.
func TestModelReconciler_ConcurrentReconcilesDoNotOverlap(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.reconciler.SetDesired(f.desired)

	var inFlight atomic.Int32
	var overlapped atomic.Bool

	// Spawn 10 concurrent workers processing items from the queue.
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}
				key, shutdown := f.reconciler.queue.Get()
				if shutdown {
					return
				}
				// Assert at most one worker processes this key at any time.
				if key == f.instanceKey {
					if c := inFlight.Add(1); c > 1 {
						overlapped.Store(true)
					}
					time.Sleep(1 * time.Millisecond)
					inFlight.Add(-1)
				}
				_ = f.reconciler.Reconcile(ctx, key)
				f.reconciler.queue.Done(key)

				if f.reconciler.Status(f.instanceKey).Phase == InstancePhasePassed {
					return
				}
			}
		}()
	}

	// Concurrently enqueue the key multiple times.
	for i := 0; i < 20; i++ {
		f.reconciler.Enqueue(f.instanceKey)
		time.Sleep(1 * time.Millisecond)
	}

	// Wait for instance to reach passed.
	for {
		if f.reconciler.Status(f.instanceKey).Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("timed out waiting for passed under concurrent reconciliation")
		case <-time.After(2 * time.Millisecond):
		}
	}

	f.reconciler.Stop()
	wg.Wait()

	if overlapped.Load() {
		t.Error("concurrent workers overlapped on the same instance key")
	}
}

// In-progress loading polls without advancing exponential failure backoff.
func TestModelReconciler_InProgressLoadingDoesNotAdvanceBackoff(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.exec.mu.Lock()
	f.exec.state = pb.LoadState_LOAD_STATE_LOADING
	f.exec.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}

	if failures := f.limiter.NumRequeues(f.instanceKey); failures != 0 {
		t.Errorf("NumRequeues = %d, want 0", failures)
	}

	if !f.reconciler.ProcessNext(ctx) {
		t.Fatal("ProcessNext returned false")
	}
	if failures := f.limiter.NumRequeues(f.instanceKey); failures != 0 {
		t.Errorf("NumRequeues after second loading check = %d, want 0", failures)
	}
}

// Periodic resync loop detects an unloaded instance and re-reconciles it.
func TestModelReconciler_PeriodicResyncDetectsUnloadedInstance(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.reconciler.resyncPeriod = 20 * time.Millisecond
	f.reconciler.Start(ctx)
	defer f.reconciler.Stop()

	f.reconciler.SetDesired(f.desired)

	for {
		if f.reconciler.Status(f.instanceKey).Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("timed out waiting for initial passed")
		case <-time.After(5 * time.Millisecond):
		}
	}

	f.exec.mu.Lock()
	f.exec.unloaded = true
	loadsBefore := f.exec.loads
	f.exec.mu.Unlock()

	for {
		f.exec.mu.Lock()
		loads := f.exec.loads
		f.exec.mu.Unlock()
		if loads > loadsBefore && f.reconciler.Status(f.instanceKey).Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("timed out waiting for periodic resync to reload unloaded instance")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// Agent process query failure is appended to executor error.
func TestModelReconciler_AgentQueryErrorAppended(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	f := newReconcilerFixture(t, nil)
	f.agent.mu.Lock()
	f.agent.processStatus = &pb.ProcessStatus{
		Id:       "executor",
		Process:  "executor",
		State:    "running",
		Endpoint: f.agent.endpoint,
	}
	f.agent.getProcessErr = status.Error(codes.Unavailable, "agent process table unavailable")
	f.agent.failGetProcessAfter = 1
	f.agent.mu.Unlock()

	f.exec.mu.Lock()
	f.exec.getLoadErr = status.Error(codes.Unavailable, "executor connection dropped")
	f.exec.mu.Unlock()

	f.reconciler.SetDesired(f.desired)

	err := f.reconciler.Reconcile(ctx, f.instanceKey)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "executor connection dropped") {
		t.Errorf("error %v does not contain executor error", err)
	}
	if !strings.Contains(err.Error(), "agent process table unavailable") {
		t.Errorf("error %v does not append agent query failure", err)
	}
}

func TestModelReconciler_StagePackagesBeforeLoad(t *testing.T) {
	planWithPkgs := &planpb.PlanFile{
		Segments: []*planpb.Segment{
			{Name: "submod_0", Run: planpb.SegmentRun_SEGMENT_RUN_PACKAGE, PackageKey: strings.Repeat("a", 32)},
			{Name: "submod_1", Run: planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH},
			{Name: "submod_2", Run: planpb.SegmentRun_SEGMENT_RUN_CAPTURED, PackageKey: strings.Repeat("b", 32)},
		},
	}

	t.Run("missing_package_refuses_without_loading", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()

		f := newReconcilerFixture(t, nil)
		f.agent.stagePkgErrors = []error{status.Error(codes.FailedPrecondition, "missing package(s) [aaa]: build the plan's packages")}
		f.reconciler.Start(ctx)
		defer f.reconciler.Stop()

		d := f.desired
		d.Request = &pb.LoadModelRequest{ManifestJson: `{"model":"m"}`, PlanFile: planWithPkgs}
		gen := f.reconciler.SetDesired(d)
		st, err := f.reconciler.Wait(ctx, f.instanceKey, gen)
		if err != nil {
			t.Fatalf("Wait: %v", err)
		}
		if st.Phase != InstancePhaseRefused {
			t.Fatalf("phase = %v, want %v", st.Phase, InstancePhaseRefused)
		}
		if st.LastError == nil || !strings.Contains(st.LastError.Error(), "missing package(s)") {
			t.Errorf("LastError = %v, want missing package message", st.LastError)
		}
		if f.exec.loads != 0 {
			t.Errorf("LoadModel calls = %d, want 0", f.exec.loads)
		}
	})

	t.Run("store_failure_retries_then_loads", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()

		f := newReconcilerFixture(t, nil)
		f.agent.stagePkgErrors = []error{status.Error(codes.Unavailable, "store unreachable"), nil}
		f.reconciler.Start(ctx)
		defer f.reconciler.Stop()

		d := f.desired
		d.Request = &pb.LoadModelRequest{ManifestJson: `{"model":"m"}`, PlanFile: planWithPkgs}
		gen := f.reconciler.SetDesired(d)
		st, err := f.reconciler.Wait(ctx, f.instanceKey, gen)
		if err != nil {
			t.Fatalf("Wait: %v", err)
		}
		if st.Phase != InstancePhasePassed {
			t.Fatalf("phase = %v, want %v (lastErr=%v)", st.Phase, InstancePhasePassed, st.LastError)
		}
		f.agent.mu.Lock()
		calls := len(f.agent.stagePkgCalls)
		var gotKeys []string
		if calls > 0 {
			gotKeys = f.agent.stagePkgCalls[0].GetKeys()
		}
		f.agent.mu.Unlock()
		if calls != 2 {
			t.Errorf("StagePackages calls = %d, want 2 (1 transient error + 1 retry)", calls)
		}
		wantKeys := []string{strings.Repeat("a", 32), strings.Repeat("b", 32)}
		if len(gotKeys) != 2 || gotKeys[0] != wantKeys[0] || gotKeys[1] != wantKeys[1] {
			t.Errorf("StagePackages keys = %v, want %v", gotKeys, wantKeys)
		}
		if f.exec.loads != 1 {
			t.Errorf("LoadModel calls = %d, want 1", f.exec.loads)
		}
	})
}
