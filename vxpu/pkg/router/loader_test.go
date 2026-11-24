/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package router

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// A caller whose context ends stops waiting and the load continues.
func TestLoadModel_CancelledCallerStopsWaitingAndLoadContinues(t *testing.T) {
	loadStarted := make(chan struct{})
	gateLoad := make(chan struct{})

	f := newGoldenFixture(t, "test-cancel-load-continues", []int32{10, 20, 30})
	f.exec.onLoad = func() {
		close(loadStarted)
		<-gateLoad
	}

	cancelCtx, cancel := context.WithCancel(f.ctx)
	callerDone := make(chan error, 1)
	go func() {
		_, err := f.server.LoadModel(cancelCtx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		callerDone <- err
	}()

	<-loadStarted
	cancel()

	select {
	case err := <-callerDone:
		if status.Code(err) != codes.Canceled {
			t.Fatalf("expected caller to stop waiting with Canceled, got: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("caller did not stop waiting after context cancellation")
	}

	// Release executor gate: the load should continue in the background until passed.
	close(gateLoad)

	ctxWait, cancelWait := context.WithTimeout(f.ctx, 5*time.Second)
	defer cancelWait()
	for {
		if st := f.server.ModelStatus(f.modelKey); st.Phase == InstancePhasePassed {
			break
		}
		select {
		case <-ctxWait.Done():
			t.Fatalf("load did not complete in background after caller cancellation; last status: %v", f.server.ModelStatus(f.modelKey))
		case <-time.After(10 * time.Millisecond):
		}
	}

	if f.exec.loads != 1 {
		t.Errorf("expected exactly 1 load, got %d", f.exec.loads)
	}
}

// Two concurrent LoadModel calls for one instance produce one load.
func TestLoadModel_ConcurrentCallsProduceOneLoad(t *testing.T) {
	loadStarted := make(chan struct{})
	gateLoad := make(chan struct{})
	var loadOnce sync.Once

	f := newGoldenFixture(t, "test-concurrent-one-load", []int32{10, 20, 30})
	f.exec.onLoad = func() {
		loadOnce.Do(func() {
			close(loadStarted)
			<-gateLoad
		})
	}

	const callers = 2
	var wg sync.WaitGroup
	errs := make([]error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}

	<-loadStarted
	close(gateLoad)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("caller %d failed: %v", i, err)
		}
	}
	if f.exec.loads != 1 {
		t.Errorf("expected exactly 1 load across concurrent calls, got %d", f.exec.loads)
	}
	if st := f.server.ModelStatus(f.modelKey); st.Phase != InstancePhasePassed {
		t.Errorf("status = %v, want InstancePhasePassed", st.Phase)
	}
}

// A refused plan or a golden mismatch returns the recorded refusal at once on a second call.
func TestLoadModel_RefusedPlanOrGoldenMismatchReturnsAtOnceOnSecondCall(t *testing.T) {
	// Case 1: Golden mismatch
	{
		f := newGoldenFixture(t, "test-golden-mismatch-second-call", []int32{10, 999, 30})
		_, err1 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		if status.Code(err1) != codes.FailedPrecondition {
			t.Fatalf("first call: expected FailedPrecondition, got %v", err1)
		}
		if f.exec.loads != 1 {
			t.Fatalf("first call: expected 1 load, got %d", f.exec.loads)
		}

		_, err2 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		if status.Code(err2) != codes.FailedPrecondition {
			t.Fatalf("second call: expected FailedPrecondition, got %v", err2)
		}
		if f.exec.loads != 1 {
			t.Errorf("second call: expected no new load on recorded refusal, got %d loads", f.exec.loads)
		}
	}

	// Case 2: Refused plan
	{
		f := newGoldenFixture(t, "test-refused-plan-second-call", []int32{10, 20, 30})
		f.exec.mu.Lock()
		f.exec.loadErr = status.Error(codes.InvalidArgument, "invalid execution plan")
		f.exec.mu.Unlock()

		_, err1 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		if status.Code(err1) != codes.InvalidArgument {
			t.Fatalf("first call: expected InvalidArgument, got %v", err1)
		}
		if f.exec.loads != 1 {
			t.Fatalf("first call: expected 1 load, got %d", f.exec.loads)
		}

		_, err2 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		if status.Code(err2) != codes.InvalidArgument {
			t.Fatalf("second call: expected InvalidArgument, got %v", err2)
		}
		if f.exec.loads != 1 {
			t.Errorf("second call: expected no new load on recorded refusal, got %d loads", f.exec.loads)
		}
	}
}

// An executor that refuses for a reason retrying cannot change is asked once per desired generation.
func TestLoadModel_TerminalRefusalAskedOncePerGeneration(t *testing.T) {
	f := newGoldenFixture(t, "test-asked-once-per-gen", []int32{10, 20, 30})
	f.exec.mu.Lock()
	f.exec.loadErr = status.Error(codes.InvalidArgument, "executor refused plan")
	f.exec.mu.Unlock()

	// Generation 1: first load attempt fails with InvalidArgument
	_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("call 1: expected InvalidArgument, got %v", err)
	}
	if f.exec.loads != 1 {
		t.Fatalf("loads after call 1 = %d, want 1", f.exec.loads)
	}

	// Repeated calls and resync passes for generation 1 do not re-ask executor.
	for i := 0; i < 5; i++ {
		_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("repeated call %d: expected InvalidArgument, got %v", i, err)
		}
	}
	f.server.Reconciler().Enqueue(f.modelKey)
	time.Sleep(50 * time.Millisecond)

	if f.exec.loads != 1 {
		t.Fatalf("loads for generation 1 = %d, want exactly 1", f.exec.loads)
	}

	// Generation 2: changing desired state (new plan file) triggers exactly one new load.
	plan2 := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites: []*planpb.PlanSite{
			{
				Id:       "site-gen-2",
				Strategy: "ref",
			},
		},
	}
	if err := kernelplan.EnsurePlanIdentity(plan2); err != nil {
		t.Fatalf("EnsurePlanIdentity: %v", err)
	}
	reqGen2 := &pb.LoadModelRequest{
		ManifestJson: f.manifest,
		PlanFile:     plan2,
	}
	_, err = f.server.LoadModel(f.ctx, reqGen2)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("generation 2: expected InvalidArgument, got %v", err)
	}
	if f.exec.loads != 2 {
		t.Errorf("total loads across 2 generations = %d, want 2 (1 per generation)", f.exec.loads)
	}
}

// An executor answering FailedPrecondition is treated as a refusal:
// asked once per generation, and all callers get the refusal promptly.
func TestLoadModel_ExecutorFailedPreconditionRefusalReturnsPromptly(t *testing.T) {
	f := newGoldenFixture(t, "test-failed-precondition-refusal", []int32{10, 20, 30})
	f.exec.mu.Lock()
	f.exec.loadErr = status.Error(codes.FailedPrecondition, "unsupported model format for accelerator")
	f.exec.mu.Unlock()

	const callers = 4
	var wg sync.WaitGroup
	errs := make([]error, callers)
	start := time.Now()
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}
	wg.Wait()
	duration := time.Since(start)

	for i, err := range errs {
		if status.Code(err) != codes.FailedPrecondition {
			t.Errorf("caller %d got code %v, want FailedPrecondition (err: %v)", i, status.Code(err), err)
		}
	}
	if f.exec.loads != 1 {
		t.Errorf("expected exactly 1 load per generation on refusal, got %d", f.exec.loads)
	}
	if duration > 2*time.Second {
		t.Errorf("expected refusal to return promptly, took %v", duration)
	}
}

// An executor whose background load failed can be retried fresh when desired state
// is re-submitted.
func TestLoadModel_ExecutorLoadStateFailedCanRetry(t *testing.T) {
	f := newGoldenFixture(t, "test-load-failed-retry", []int32{10, 20, 30})
	var failNext bool = true
	f.exec.mu.Lock()
	f.exec.onLoad = func() {
		f.exec.mu.Lock()
		f.exec.loadFailed = failNext
		f.exec.mu.Unlock()
	}
	f.exec.mu.Unlock()

	_, err1 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if status.Code(err1) != codes.Internal {
		t.Fatalf("first call: expected Internal, got %v", err1)
	}

	// Clear executor error: a retry of LoadModel should attempt again fresh and succeed.
	f.exec.mu.Lock()
	failNext = false
	f.exec.mu.Unlock()

	_, err2 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
	if err2 != nil {
		t.Fatalf("second call after error recovery failed: %v", err2)
	}
	if f.exec.loads < 2 {
		t.Errorf("expected at least 2 loads after retry, got %d", f.exec.loads)
	}
}

// A caller waiting on a generation unblocks when a newer generation supersedes it.
func TestLoadModel_WaitUnblocksWhenNewerGenerationAdvances(t *testing.T) {
	f := newGoldenFixture(t, "test-wait-advances", []int32{10, 20, 30})
	loadGate := make(chan struct{})
	var loadOnce sync.Once
	f.exec.onLoad = func() {
		loadOnce.Do(func() {
			<-loadGate
		})
	}

	gen1Done := make(chan error, 1)
	go func() {
		_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		gen1Done <- err
	}()

	time.Sleep(20 * time.Millisecond)

	// Advance desired generation with a different plan
	plan2 := &planpb.PlanFile{
		ArtifactMd5: "0123456789abcdef0123456789abcdef",
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites: []*planpb.PlanSite{
			{Id: "site-adv", Strategy: "ref"},
		},
	}
	_ = kernelplan.EnsurePlanIdentity(plan2)
	close(loadGate)

	_, err2 := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest, PlanFile: plan2})
	if err2 != nil {
		t.Fatalf("generation 2 LoadModel failed: %v", err2)
	}

	select {
	case err1 := <-gen1Done:
		if err1 != nil {
			t.Errorf("generation 1 waiter failed: %v", err1)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("generation 1 waiter did not unblock when generation 2 completed")
	}
}

// Malformed ManifestJson returns InvalidArgument rather than a golden check refusal.
func TestLoadModel_MalformedManifestReturnsInvalidArgument(t *testing.T) {
	f := newGoldenFixture(t, "test-malformed-manifest", []int32{10, 20, 30})
	_, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: "{not-json"})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument, got %v (err: %v)", status.Code(err), err)
	}
	if strings.Contains(err.Error(), "golden check") {
		t.Errorf("malformed manifest error should not mention golden check: %v", err)
	}
}

// Reloading an unloaded executor on resync clears verifiedGen and re-runs golden verification.
func TestLoadModel_ReloadUnloadedExecutorRerunsGoldenCheck(t *testing.T) {
	f := newGoldenFixture(t, "test-resync-golden", []int32{10, 20, 30})
	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("initial LoadModel: %v", err)
	}
	if st := f.server.ModelStatus(f.modelKey); st.Phase != InstancePhasePassed {
		t.Fatalf("initial phase = %v, want Passed", st.Phase)
	}

	// Simulate executor losing model and producing bad tokens after reload.
	f.exec.mu.Lock()
	f.exec.unloaded = true
	f.exec.tokens = [][]int32{{10, 999, 30}}
	f.exec.sessions = 0
	f.exec.mu.Unlock()

	f.server.Reconciler().Enqueue(f.modelKey)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if st := f.server.ModelStatus(f.modelKey); st.Phase == InstancePhaseRefused {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("expected resync reload to re-run golden check and refuse, got status: %+v", f.server.ModelStatus(f.modelKey))
}

// A slow manifest weight rewrite runs in the background so the single reconciler
// worker continues reconciling other model instances without blocking.
func TestLoadModel_SlowManifestRewriteDoesNotBlockOtherInstances(t *testing.T) {
	ctx := t.Context()
	manifestA := `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"model-a","revision":"main","commit_sha":"c1"},"files":{"sha_a":{"size":4,"name":"a.safetensors","source":"https://example.com/a"}},"tensors":{}}`
	manifestB := `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"model-b","revision":"main","commit_sha":"c2"},"files":{"sha_b":{"size":4,"name":"b.safetensors","source":"https://example.com/b"}},"tensors":{}}`
	keyA := artifactKey(manifestA)
	keyB := artifactKey(manifestB)

	startedA := make(chan struct{})
	gateA := make(chan struct{})
	var onceA sync.Once
	store := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/import") {
			if strings.Contains(r.URL.Path, "sha_a") {
				onceA.Do(func() { close(startedA) })
				<-gateA
			}
			w.WriteHeader(http.StatusCreated)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(store.Close)

	startExec := func(e *goldenFakeExecutor) string {
		lis, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("net.Listen: %v", err)
		}
		srv := grpc.NewServer()
		pb.RegisterExecutorServer(srv, e)
		go func() { _ = srv.Serve(lis) }()
		t.Cleanup(srv.Stop)
		return lis.Addr().String()
	}
	addrA := startExec(&goldenFakeExecutor{})
	addrB := startExec(&goldenFakeExecutor{})

	podA := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + keyA, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "10.0.0.1"},
	}
	podB := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + keyB, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "10.0.0.2"},
	}
	s := NewServer(fake.NewSimpleClientset(podA, podB), "default", "img", "")
	s.UseWeightStore(weightstore.New(store.URL))
	t.Cleanup(func() { _ = s.Close() })

	dialerA := fakeAgentDialer(t, &fakeAgent{endpoint: addrA})
	dialerB := fakeAgentDialer(t, &fakeAgent{endpoint: addrB})
	s.agentDialer = func(addr string) (*grpc.ClientConn, error) {
		if strings.HasPrefix(addr, "10.0.0.1") {
			return dialerA(addr)
		}
		return dialerB(addr)
	}

	doneA := make(chan error, 1)
	go func() {
		_, err := s.LoadModel(ctx, &pb.LoadModelRequest{ManifestJson: manifestA})
		doneA <- err
	}()

	<-startedA

	// While Model A's weight rewrite is blocked, Model B must complete promptly.
	ctxB, cancelB := context.WithTimeout(ctx, 2*time.Second)
	defer cancelB()
	if _, err := s.LoadModel(ctxB, &pb.LoadModelRequest{ManifestJson: manifestB}); err != nil {
		close(gateA)
		t.Fatalf("LoadModel for model B blocked behind model A's manifest rewrite: %v", err)
	}

	close(gateA)
	if err := <-doneA; err != nil {
		t.Fatalf("LoadModel for model A failed: %v", err)
	}
}

// When the executor fails with Internal, status transitions to Failed with error
// and NextAttempt so waiting callers get the error immediately while the loop
// keeps retrying behind them until it recovers.
func TestLoadModel_ExecutorInternalErrorNotifiesCallersAndRetries(t *testing.T) {
	f := newGoldenFixture(t, "test-internal-error", []int32{10, 20, 30})
	f.exec.mu.Lock()
	f.exec.loadErr = status.Error(codes.Internal, "device fault")
	f.exec.mu.Unlock()

	const numCallers = 4
	var wg sync.WaitGroup
	errs := make([]error, numCallers)
	start := time.Now()
	for i := 0; i < numCallers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			callCtx, cancel := context.WithTimeout(f.ctx, 3*time.Second)
			defer cancel()
			_, errs[idx] = f.server.LoadModel(callCtx, &pb.LoadModelRequest{ManifestJson: f.manifest})
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)

	for i, err := range errs {
		if status.Code(err) != codes.Internal {
			t.Errorf("caller %d: got %v (code %v), want Internal", i, err, status.Code(err))
		}
	}
	if elapsed > 1500*time.Millisecond {
		t.Errorf("callers waited %v instead of failing promptly", elapsed)
	}

	st := f.server.ModelStatus(f.modelKey)
	if st.Phase != InstancePhaseFailed {
		t.Errorf("phase = %v, want Failed", st.Phase)
	}
	if st.LastError == nil || status.Code(st.LastError) != codes.Internal {
		t.Errorf("LastError = %v, want Internal", st.LastError)
	}
	if st.NextAttempt.IsZero() {
		t.Error("expected non-zero NextAttempt on Failed status")
	}

	// Clearing the fault allows the background backoff retry to converge to Passed.
	f.exec.mu.Lock()
	f.exec.loadErr = nil
	f.exec.mu.Unlock()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if f.server.ModelStatus(f.modelKey).Phase == InstancePhasePassed {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("background retries did not recover to Passed; last status: %+v", f.server.ModelStatus(f.modelKey))
}

// Removing desired state deletes the executor pod and keeps it deleted through several resyncs.
func TestModelReconciler_DeleteDesiredDeletesPodAndStaysDeleted(t *testing.T) {
	f := newGoldenFixture(t, "test-delete-desired", []int32{10, 20, 30})
	t.Cleanup(func() { _ = f.server.Close() })

	// Speed up resync to verify several resync cycles do not recreate the pod.
	f.server.Reconciler().Stop()
	reconCtx, reconCancel := context.WithCancel(t.Context())
	t.Cleanup(reconCancel)
	f.server.reconciler = NewModelReconciler(f.server, WithPollInterval(10*time.Millisecond), WithResyncPeriod(20*time.Millisecond))
	f.server.reconciler.Start(reconCtx)

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	podName := "vxpu-executor-" + f.modelKey
	if _, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{}); err != nil {
		t.Fatalf("pod should exist before DeleteDesired: %v", err)
	}

	f.server.Reconciler().DeleteDesired(f.modelKey)

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		_, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected pod %s to be deleted after DeleteDesired, got err=%v", podName, err)
	}

	// Wait through several resync periods (5+ cycles at 20ms) and confirm the pod stays deleted.
	time.Sleep(150 * time.Millisecond)
	if _, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("pod %s was recreated across resyncs after DeleteDesired, got err=%v", podName, err)
	}
}

// Deleting a passed instance's pod externally is an observation; while desired
// state remains, resync recreates the pod rather than removing desired state.
func TestModelReconciler_ExternalPodDeletionRecreatesPod(t *testing.T) {
	f := newGoldenFixture(t, "test-external-pod-recreate", []int32{10, 20, 30})
	t.Cleanup(func() { _ = f.server.Close() })

	f.server.Reconciler().Stop()
	reconCtx, reconCancel := context.WithCancel(t.Context())
	t.Cleanup(reconCancel)
	f.server.reconciler = NewModelReconciler(f.server, WithPollInterval(10*time.Millisecond), WithResyncPeriod(20*time.Millisecond))
	f.server.reconciler.Start(reconCtx)

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	podName := "vxpu-executor-" + f.modelKey
	if err := f.clientset.CoreV1().Pods("default").Delete(f.ctx, podName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("external pod Delete: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{}); err == nil {
			if f.server.ModelStatus(f.modelKey).Phase == InstancePhasePassed {
				return
			}
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("expected resync to recreate externally deleted pod %s while desired state remains", podName)
}

// Forgetting a model key the reconciler never owned (such as a model loaded by name)
// must not delete its executor pod.
func TestModelReconciler_ForgetUnownedModelDoesNotDeletePod(t *testing.T) {
	unownedKey := "by-name-unowned-key"
	podName := "vxpu-executor-" + unownedKey
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.1"},
	}
	clientset := fake.NewSimpleClientset(pod)
	s := NewServer(clientset, "default", "img", "")
	t.Cleanup(func() { _ = s.Close() })

	s.forgetModel(unownedKey, "")
	if err := s.reconciler.Reconcile(t.Context(), unownedKey); err != nil {
		t.Fatalf("Reconcile unowned key: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	if _, err := clientset.CoreV1().Pods("default").Get(t.Context(), podName, metav1.GetOptions{}); err != nil {
		t.Fatalf("forgetModel deleted unowned pod %s: %v", podName, err)
	}
}

// When a request hits executorEndpoint for a reconciler-owned instance whose pod was
// deleted externally, the request returns FailedPrecondition and enqueues the loop
// without removing desired state, so the loop recreates the pod and recovers.
func TestRequestObservation_MissingPodDoesNotRemoveDesiredState(t *testing.T) {
	f := newGoldenFixture(t, "test-req-missing-pod", []int32{10, 20, 30}, []int32{10, 20, 30})
	t.Cleanup(func() { _ = f.server.Close() })

	// Long resync period ensures recovery comes from the request enqueueing, not resync.
	f.server.Reconciler().Stop()
	f.server.reconciler = NewModelReconciler(f.server, WithPollInterval(10*time.Millisecond), WithResyncPeriod(time.Hour))
	f.server.reconciler.Start(t.Context())

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	podName := "vxpu-executor-" + f.modelKey
	if err := f.clientset.CoreV1().Pods("default").Delete(f.ctx, podName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Delete pod: %v", err)
	}

	// Request observes missing pod before any resync; must return FailedPrecondition
	// without removing desired state.
	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("NewSession with missing pod: got %v, want FailedPrecondition", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := f.clientset.CoreV1().Pods("default").Get(f.ctx, podName, metav1.GetOptions{}); err == nil {
			if _, sessErr := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); sessErr == nil {
				return
			}
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("expected reconciler to recreate pod %s and serve again after request observation; status=%+v",
		podName, f.server.ModelStatus(f.modelKey))
}

// When NewSession observes LOAD_STATE_NONE or LOAD_STATE_FAILED on a reconciler-owned
// instance, it returns an error and enqueues the loop without removing desired state,
// so the loop reloads the model automatically.
func TestRequestObservation_SessionUnloadAndFailedDoNotRemoveDesiredState(t *testing.T) {
	f := newGoldenFixture(t, "test-req-session-obs",
		[]int32{10, 20, 30}, []int32{10, 20, 30}, []int32{10, 20, 30})
	t.Cleanup(func() { _ = f.server.Close() })

	// Long resync period ensures recovery comes from NewSession enqueueing, not resync.
	f.server.Reconciler().Stop()
	f.server.reconciler = NewModelReconciler(f.server, WithPollInterval(10*time.Millisecond), WithResyncPeriod(time.Hour))
	f.server.reconciler.Start(t.Context())

	if _, err := f.server.LoadModel(f.ctx, &pb.LoadModelRequest{ManifestJson: f.manifest}); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Subcase 1: executor reports LOAD_STATE_NONE.
	f.exec.mu.Lock()
	f.exec.unloaded = true
	f.exec.mu.Unlock()

	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("NewSession on LOAD_STATE_NONE: got %v, want FailedPrecondition", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		f.exec.mu.Lock()
		loads := f.exec.loads
		f.exec.mu.Unlock()
		if loads >= 2 && f.server.ModelStatus(f.modelKey).Phase == InstancePhasePassed {
			break
		}
		time.Sleep(15 * time.Millisecond)
	}
	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); err != nil {
		t.Fatalf("NewSession after LOAD_STATE_NONE recovery: %v", err)
	}

	// Subcase 2: executor reports LOAD_STATE_FAILED.
	f.exec.mu.Lock()
	f.exec.loadFailed = true
	f.exec.transportErr = status.Error(codes.FailedPrecondition, "not loaded")
	f.exec.mu.Unlock()

	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); status.Code(err) != codes.Internal {
		t.Fatalf("NewSession on LOAD_STATE_FAILED: got %v, want Internal", err)
	}

	// Clear the synthetic NewSession error; the reconciler must reload without another LoadModel call.
	f.exec.mu.Lock()
	f.exec.transportErr = nil
	f.exec.mu.Unlock()

	deadline = time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		f.exec.mu.Lock()
		loads := f.exec.loads
		f.exec.mu.Unlock()
		if loads >= 3 && f.server.ModelStatus(f.modelKey).Phase == InstancePhasePassed {
			break
		}
		time.Sleep(15 * time.Millisecond)
	}
	if _, err := f.server.NewSession(f.ctx, &pb.NewSessionRequest{}); err != nil {
		t.Fatalf("NewSession after LOAD_STATE_FAILED recovery: %v", err)
	}
}

// Stopping the reconciler cancels any in-flight background manifest rewrite.
func TestModelReconciler_StopCancelsInFlightRewrite(t *testing.T) {
	manifest := `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"model-stop","revision":"main","commit_sha":"c1"},"files":{"sha_stop":{"size":4,"name":"a.safetensors","source":"https://example.com/a"}},"tensors":{}}`
	key := artifactKey(manifest)

	started := make(chan struct{})
	canceled := make(chan struct{})
	var once sync.Once
	store := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/import") {
			_, _ = io.Copy(io.Discard, r.Body)
			once.Do(func() { close(started) })
			<-r.Context().Done()
			close(canceled)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(func() {
		store.CloseClientConnections()
		store.Close()
	})

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, &goldenFakeExecutor{})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "vxpu-executor-" + key, Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "10.0.0.1"},
	}
	s := NewServer(fake.NewSimpleClientset(pod), "default", "img", "")
	s.UseWeightStore(weightstore.New(store.URL))
	s.agentDialer = fakeAgentDialer(t, &fakeAgent{endpoint: lis.Addr().String()})

	s.Reconciler().SetDesired(InstanceDesired{
		Key:     key,
		Model:   "model-stop",
		Request: &pb.LoadModelRequest{ManifestJson: manifest},
	})

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for background rewrite to start")
	}

	_ = s.Close()

	select {
	case <-canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("stopping reconciler did not cancel in-flight manifest rewrite")
	}
}
