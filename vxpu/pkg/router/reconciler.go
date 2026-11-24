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
	stderrors "errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// errInstanceLoading indicates that asynchronous progress is underway (manifest
// weight caching or executor weight rehydration).
var errInstanceLoading = stderrors.New("instance is loading on executor")

// runtimeDeadError wraps an exit error reported by the agent process table.
type runtimeDeadError struct {
	err error
}

func (e *runtimeDeadError) Error() string { return e.err.Error() }
func (e *runtimeDeadError) Unwrap() error { return e.err }

// manifestRewrite tracks an asynchronous manifest rewrite (weight caching) for an instance.
type manifestRewrite struct {
	// generation is the desired generation this rewrite was started for.
	generation int64
	// cancel aborts the background rewrite if the instance changes generation or is deleted.
	cancel context.CancelFunc
	// done is true once the background rewrite has finished.
	done bool
	// manifest is the rewritten manifest JSON when err is nil.
	manifest string
	// err is the rewrite error, if any.
	err error
}

// InstancePhase is the lifecycle phase of an instance in the reconciler.
type InstancePhase string

const (
	InstancePhaseNotLoaded InstancePhase = "not loaded"
	InstancePhaseLoading   InstancePhase = "loading"
	InstancePhaseChecking  InstancePhase = "checking"
	InstancePhasePassed    InstancePhase = "passed"
	InstancePhaseRefused   InstancePhase = "refused"
	InstancePhaseFailed    InstancePhase = "failed"
)

// InstanceStatus records the lifecycle standing of an instance.
type InstanceStatus struct {
	// Phase is the current reconciliation phase of the instance.
	Phase InstancePhase

	// LastError is the last error encountered during reconciliation, if any.
	LastError error

	// NextAttempt is when the next reconciliation attempt is due if the instance failed.
	NextAttempt time.Time

	// ObservedGeneration is the generation of desired state this status reflects.
	ObservedGeneration int64
}

func (s InstanceStatus) String() string {
	if s.LastError != nil {
		return fmt.Sprintf("%s (%v)", s.Phase, s.LastError)
	}
	return string(s.Phase)
}

// InstanceDesired records the desired specification for one model instance on an executor.
type InstanceDesired struct {
	// Key is the unique identifier for this instance.
	Key string

	// Model is the model name or artifact key.
	Model string

	// Request is the load request containing manifest, binding, graphs, and plan.
	Request *pb.LoadModelRequest

	// Golden is the optional golden verification reference file.
	Golden *planpb.GoldenFile

	// ByName reports whether the instance was loaded by catalog name and uses planLoad.
	ByName bool

	// Generation is the sequence number of this desired state.
	Generation int64
}

// ModelReconciler reconciles desired model instances with actual executor state.
type ModelReconciler struct {
	// server is the router server providing executor and agent connections.
	server *Server

	// mu protects desired, statuses, endpoints, loadedGen, verifiedGen, passedGen, rewrites, teardown, notify, stopCancel, and stopped.
	mu sync.RWMutex

	// desired stores the desired state for each instance key.
	desired map[string]InstanceDesired

	// statuses stores the current observed status for each instance key.
	statuses map[string]InstanceStatus

	// endpoints records the last resolved executor endpoint for each instance key.
	endpoints map[string]string

	// loadedGen records the generation successfully loaded on an executor for each instance key.
	loadedGen map[string]int64

	// verifiedGen records the generation that passed verification checks for each instance key.
	verifiedGen map[string]int64

	// passedGen records the latest generation that reached Passed for each instance key.
	passedGen map[string]int64

	// lastUsed maps each passed instance key to an *atomic.Int64 storing
	// monotonic nanoseconds since epoch. Storing per-instance atomics in a
	// sync.Map lets Touch record activity without acquiring mu.
	lastUsed sync.Map

	// epoch is the monotonic time base for lastUsed timestamps.
	epoch time.Time

	// rewrites tracks in-flight or completed background manifest rewrites per instance key.
	rewrites map[string]*manifestRewrite

	// teardown tracks instances previously owned by the reconciler whose desired state was removed.
	teardown map[string]bool

	// notify sends a signal to waiters whenever an instance's status or generation changes.
	notify map[string]chan struct{}

	// queue is the work queue of instance keys to reconcile.
	queue workqueue.TypedDelayingInterface[string]

	// limiter is the rate limiter used for backoff.
	limiter workqueue.TypedRateLimiter[string]

	// pollInterval is the fixed interval to poll when an instance is actively loading.
	pollInterval time.Duration

	// resyncPeriod is the interval between periodic resyncs of all desired instances.
	resyncPeriod time.Duration

	// keepAlive is how long an idle instance remains desired before removal.
	keepAlive time.Duration

	// stopCancel cancels the loop context and any background rewrites when Stop is called.
	stopCancel context.CancelFunc

	// stopped is true after Stop has been called.
	stopped bool

	// wg tracks background goroutines started by the reconciler.
	wg sync.WaitGroup
}

// ReconcilerOption configures a ModelReconciler.
type ReconcilerOption func(*ModelReconciler)

// WithRateLimiter sets the rate limiter used for backoff.
func WithRateLimiter(limiter workqueue.TypedRateLimiter[string]) ReconcilerOption {
	return func(r *ModelReconciler) {
		r.limiter = limiter
	}
}

// WithPollInterval sets the polling interval used while an instance is loading.
func WithPollInterval(d time.Duration) ReconcilerOption {
	return func(r *ModelReconciler) {
		r.pollInterval = d
	}
}

// WithResyncPeriod sets the periodic resync interval for desired instances.
func WithResyncPeriod(d time.Duration) ReconcilerOption {
	return func(r *ModelReconciler) {
		r.resyncPeriod = d
	}
}

// WithKeepAlive sets how long an idle instance remains desired before removal.
// A non-positive duration disables idle expiry.
func WithKeepAlive(d time.Duration) ReconcilerOption {
	return func(r *ModelReconciler) {
		r.keepAlive = d
	}
}

// NewModelReconciler creates a reconciler for the given server.
func NewModelReconciler(s *Server, opts ...ReconcilerOption) *ModelReconciler {
	r := &ModelReconciler{
		server:       s,
		desired:      make(map[string]InstanceDesired),
		statuses:     make(map[string]InstanceStatus),
		endpoints:    make(map[string]string),
		loadedGen:    make(map[string]int64),
		verifiedGen:  make(map[string]int64),
		passedGen:    make(map[string]int64),
		epoch:        time.Now(),
		rewrites:     make(map[string]*manifestRewrite),
		teardown:     make(map[string]bool),
		notify:       make(map[string]chan struct{}),
		queue:        workqueue.NewTypedDelayingQueue[string](),
		limiter:      workqueue.NewTypedItemExponentialFailureRateLimiter[string](100*time.Millisecond, 5*time.Second),
		pollInterval: 100 * time.Millisecond,
		resyncPeriod: 10 * time.Second,
		keepAlive:    time.Duration(executorKeepAliveSeconds()) * time.Second,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Status reports the current status of an instance.
func (r *ModelReconciler) Status(key string) InstanceStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()
	st, ok := r.statuses[key]
	if !ok {
		return InstanceStatus{Phase: InstancePhaseNotLoaded}
	}
	return st
}

// Desired reports the current desired state recorded for an instance.
func (r *ModelReconciler) Desired(key string) (InstanceDesired, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.desired[key]
	return d, ok
}

func (r *ModelReconciler) notifyLocked(key string) {
	if ch, ok := r.notify[key]; ok {
		close(ch)
		r.notify[key] = make(chan struct{})
	}
}

func (r *ModelReconciler) cancelRewriteLocked(key string) {
	if rw, ok := r.rewrites[key]; ok {
		if rw.cancel != nil {
			rw.cancel()
		}
		delete(r.rewrites, key)
	}
}

// Touch records activity for a desired instance that has passed verification,
// extending its keep-alive window. A serving request pays one lock-free
// sync.Map lookup and one atomic store; it never acquires r.mu or contends
// with readers or the reconcile loop.
func (r *ModelReconciler) Touch(key string) {
	if v, ok := r.lastUsed.Load(key); ok {
		v.(*atomic.Int64).Store(int64(time.Since(r.epoch)))
	}
}

func (r *ModelReconciler) deleteDesiredLocked(key string) {
	delete(r.desired, key)
	delete(r.loadedGen, key)
	delete(r.verifiedGen, key)
	delete(r.passedGen, key)
	r.lastUsed.Delete(key)
	delete(r.statuses, key)
	r.cancelRewriteLocked(key)
	r.teardown[key] = true
	r.notifyLocked(key)
}

func (r *ModelReconciler) isExpiredLocked(key string, nowNanos int64) bool {
	if r.keepAlive <= 0 {
		return false
	}
	st := r.statuses[key]
	if st.Phase == InstancePhaseNotLoaded || st.Phase == InstancePhaseLoading || st.Phase == InstancePhaseChecking {
		return false
	}
	v, ok := r.lastUsed.Load(key)
	if !ok {
		return false
	}
	return time.Duration(nowNanos-v.(*atomic.Int64).Load()) >= r.keepAlive
}

// DeleteDesired removes the desired state for an instance owned by the reconciler
// and enqueues it for teardown. Calling DeleteDesired for a key the reconciler
// never held does nothing.
func (r *ModelReconciler) DeleteDesired(key string) {
	r.mu.Lock()
	if _, owned := r.desired[key]; !owned {
		r.mu.Unlock()
		return
	}
	r.deleteDesiredLocked(key)
	r.mu.Unlock()
	r.limiter.Forget(key)
	r.queue.Add(key)
}

func (r *ModelReconciler) setStatus(ctx context.Context, key string, st InstanceStatus) {
	r.mu.Lock()
	defer r.mu.Unlock()
	desired, ok := r.desired[key]
	if !ok || desired.Generation != st.ObservedGeneration {
		return
	}
	prev := r.statuses[key]
	r.statuses[key] = st
	if st.Phase == InstancePhasePassed && r.passedGen[key] != st.ObservedGeneration {
		r.passedGen[key] = st.ObservedGeneration
		ts := &atomic.Int64{}
		ts.Store(int64(time.Since(r.epoch)))
		r.lastUsed.Store(key, ts)
	}
	r.notifyLocked(key)
	if prev.Phase != st.Phase || prev.ObservedGeneration != st.ObservedGeneration || prev.LastError != st.LastError {
		klog.FromContext(ctx).Info("Instance status changed",
			"key", key, "phase", st.Phase, "generation", st.ObservedGeneration, "error", st.LastError)
	}
}

func desiredEqual(a, b InstanceDesired) bool {
	return a.Key == b.Key && a.Model == b.Model && a.ByName == b.ByName && proto.Equal(a.Golden, b.Golden) && proto.Equal(a.Request, b.Request)
}

// SetDesired registers or updates desired state for an instance and enqueues it.
// It returns the generation of the desired state.
func (r *ModelReconciler) SetDesired(desired InstanceDesired) int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped {
		return 0
	}
	delete(r.teardown, desired.Key)
	if existing, ok := r.desired[desired.Key]; ok {
		if desiredEqual(existing, desired) {
			r.Touch(desired.Key)
			st := r.statuses[desired.Key]
			if st.Phase != InstancePhasePassed && st.Phase != InstancePhaseRefused {
				if st.Phase == InstancePhaseFailed {
					r.statuses[desired.Key] = InstanceStatus{Phase: InstancePhaseNotLoaded, ObservedGeneration: existing.Generation}
					delete(r.loadedGen, desired.Key)
					r.notifyLocked(desired.Key)
				}
				r.queue.Add(desired.Key)
			}
			return existing.Generation
		}
		desired.Generation = existing.Generation + 1
	} else {
		desired.Generation = 1
	}
	r.desired[desired.Key] = desired
	r.lastUsed.Delete(desired.Key)
	r.cancelRewriteLocked(desired.Key)
	if current, has := r.statuses[desired.Key]; !has || current.Phase != InstancePhaseLoading {
		r.statuses[desired.Key] = InstanceStatus{Phase: InstancePhaseNotLoaded, ObservedGeneration: desired.Generation}
	}
	r.notifyLocked(desired.Key)
	r.queue.Add(desired.Key)
	return desired.Generation
}

// Wait waits until the instance reaches a terminal phase (Passed, Refused, or Failed)
// for gen (or a newer generation), until desired state is removed, until the reconciler
// stops, or until ctx ends.
func (r *ModelReconciler) Wait(ctx context.Context, key string, gen int64) (InstanceStatus, error) {
	for {
		if ctx.Err() != nil {
			return InstanceStatus{}, ctx.Err()
		}
		r.mu.Lock()
		stopped := r.stopped
		_, hasDesired := r.desired[key]
		st, ch := r.statuses[key], r.notify[key]
		if ch == nil {
			ch = make(chan struct{})
			r.notify[key] = ch
		}
		r.mu.Unlock()

		if stopped {
			return InstanceStatus{}, status.Error(codes.Unavailable, "router is shutting down")
		}
		if !hasDesired {
			return InstanceStatus{Phase: InstancePhaseNotLoaded}, status.Error(codes.Aborted, "model instance was removed")
		}
		if st.ObservedGeneration >= gen && (st.Phase == InstancePhasePassed || st.Phase == InstancePhaseRefused || st.Phase == InstancePhaseFailed) {
			return st, nil
		}

		select {
		case <-ctx.Done():
			return InstanceStatus{}, ctx.Err()
		case <-ch:
		}
	}
}

// Reconcile step outcomes fall into three categories for the loop:
//
//  1. Not ready yet (errInstanceLoading): work is in progress (the router is
//     caching manifest weights in the background or the executor is loading
//     weights). The loop requeues at the fixed pollInterval without advancing
//     exponential failure backoff, and callers keep waiting.
//  2. Refused (isRefusal): the request itself is wrong or cannot be satisfied as
//     it stands (InvalidArgument, FailedPrecondition, or TokenMismatchError). The
//     loop marks the instance Refused with LastError and stops retrying until
//     desired state changes; waiting callers receive the error immediately.
//  3. Failed (any other error): the step failed (runtime crash, executor
//     LOAD_STATE_FAILED, Internal or other RPC error, or infrastructure error).
//     The loop requeues with exponential backoff. Unless the error is a transient
//     Unavailable transport error on a live runtime (which leaves status
//     unchanged while retrying), the loop records InstancePhaseFailed with
//     LastError and NextAttempt so waiting callers receive the error immediately
//     while the loop retries behind them.
func isRefusal(err error) bool {
	if err == nil {
		return false
	}
	var mismatch *kernelplan.TokenMismatchError
	if stderrors.As(err, &mismatch) {
		return true
	}
	code := status.Code(err)
	return code == codes.InvalidArgument || code == codes.FailedPrecondition
}

// Enqueue adds an instance key to the reconciliation queue.
func (r *ModelReconciler) Enqueue(key string) {
	r.queue.Add(key)
}

// Start runs the worker loop in the background until ctx is canceled or Stop is called.
func (r *ModelReconciler) Start(ctx context.Context) {
	r.mu.Lock()
	if r.stopped || r.stopCancel != nil {
		r.mu.Unlock()
		return
	}
	loopCtx, cancel := context.WithCancel(ctx)
	r.stopCancel = cancel
	r.wg.Add(2)
	r.mu.Unlock()

	go func() {
		defer r.wg.Done()
		for r.processNext(loopCtx) {
		}
	}()
	go func() {
		defer r.wg.Done()
		if r.resyncPeriod <= 0 {
			<-loopCtx.Done()
			r.markStopped()
			r.queue.ShutDown()
			return
		}
		ticker := time.NewTicker(r.resyncPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-loopCtx.Done():
				r.markStopped()
				r.queue.ShutDown()
				return
			case <-ticker.C:
				r.resync(loopCtx)
			}
		}
	}()
}

func (r *ModelReconciler) markStopped() (alreadyStopped bool, remaining int, cancel context.CancelFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	alreadyStopped = r.stopped
	r.stopped = true
	cancel = r.stopCancel
	remaining = len(r.desired)
	for k := range r.rewrites {
		r.cancelRewriteLocked(k)
	}
	for k := range r.notify {
		r.notifyLocked(k)
	}
	return alreadyStopped, remaining, cancel
}

func (r *ModelReconciler) resync(ctx context.Context) {
	r.mu.Lock()
	nowNanos := int64(time.Since(r.epoch))
	var expired, active []string
	for k := range r.desired {
		if r.isExpiredLocked(k, nowNanos) {
			r.deleteDesiredLocked(k)
			expired = append(expired, k)
		} else {
			active = append(active, k)
		}
	}
	r.mu.Unlock()
	for _, k := range expired {
		klog.FromContext(ctx).Info("Instance keep-alive expired; removing desired state", "key", k, "keep_alive", r.keepAlive)
		r.limiter.Forget(k)
		r.queue.Add(k)
	}
	for _, k := range active {
		r.queue.Add(k)
	}
}

// Stop shuts down the reconcile loop, cancels in-flight work, and waits for all
// loop goroutines to exit. Stop leaves executor pods running so a restarted
// router can adopt them without reloading models from scratch.
func (r *ModelReconciler) Stop() {
	alreadyStopped, remaining, cancel := r.markStopped()
	if cancel != nil {
		cancel()
	}
	r.queue.ShutDown()
	r.wg.Wait()
	if !alreadyStopped && remaining > 0 {
		klog.Background().Info("Reconcile loop stopped; leaving executor pods running for adoption after restart",
			"instances", remaining)
	}
}

// ProcessNext processes a single item from the work queue.
func (r *ModelReconciler) ProcessNext(ctx context.Context) bool {
	return r.processNext(ctx)
}

func (r *ModelReconciler) processNext(ctx context.Context) bool {
	key, shutdown := r.queue.Get()
	if shutdown {
		return false
	}
	defer r.queue.Done(key)
	if ctx.Err() != nil {
		return false
	}
	err := r.Reconcile(ctx, key)
	if ctx.Err() != nil {
		return false
	}
	r.handleResult(ctx, key, err)
	return true
}

func (r *ModelReconciler) handleResult(ctx context.Context, key string, err error) {
	st := r.Status(key)
	if st.Phase == InstancePhaseRefused {
		r.limiter.Forget(key)
		return
	}
	if err == nil {
		if st.Phase == InstancePhasePassed {
			r.limiter.Forget(key)
		}
		return
	}
	if stderrors.Is(err, errInstanceLoading) {
		r.queue.AddAfter(key, r.pollInterval)
		return
	}
	delay := r.limiter.When(key)
	r.queue.AddAfter(key, delay)
	if status.Code(err) != codes.Unavailable {
		r.mu.RLock()
		desired, ok := r.desired[key]
		r.mu.RUnlock()
		if ok {
			r.setStatus(ctx, key, InstanceStatus{
				Phase:              InstancePhaseFailed,
				LastError:          err,
				NextAttempt:        time.Now().Add(delay),
				ObservedGeneration: desired.Generation,
			})
		}
	}
}

// Reconcile observes the actual state of an instance fresh and takes the next idempotent step.
func (r *ModelReconciler) Reconcile(ctx context.Context, key string) error {
	r.mu.RLock()
	desired, ok := r.desired[key]
	shouldTeardown := r.teardown[key]
	st := r.statuses[key]
	loadedGen := r.loadedGen[key]
	verifiedGen := r.verifiedGen[key]
	r.mu.RUnlock()
	if !ok {
		if !shouldTeardown {
			return nil
		}
		if err := r.teardownInstance(ctx, key); err != nil {
			return err
		}
		r.mu.Lock()
		delete(r.teardown, key)
		r.mu.Unlock()
		return nil
	}
	if st.Phase == InstancePhaseRefused && st.ObservedGeneration == desired.Generation {
		return nil
	}

	endpoint, agentClient, err := r.resolveEndpoint(ctx, desired)
	if err != nil {
		return err
	}
	conn, err := r.server.executorConn(endpoint)
	if err != nil {
		return err
	}
	execClient := pb.NewExecutorClient(conn)

	loadResp, err := execClient.GetLoad(ctx, &pb.GetLoadRequest{})
	if err != nil {
		return r.handleRPCErr(ctx, agentClient, key, err)
	}

	if loadResp.State == pb.LoadState_LOAD_STATE_FAILED && loadedGen == desired.Generation {
		r.mu.Lock()
		delete(r.loadedGen, key)
		r.mu.Unlock()
		return fmt.Errorf("model load failed on executor: %s", loadResp.Error)
	}
	if loadResp.State == pb.LoadState_LOAD_STATE_LOADING {
		r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
		return r.handleRPCErr(ctx, agentClient, key, errInstanceLoading)
	}

	if loadResp.State != pb.LoadState_LOAD_STATE_READY || loadedGen != desired.Generation {
		r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
		reqCopy := &pb.LoadModelRequest{}
		if desired.Request != nil {
			reqCopy = proto.Clone(desired.Request).(*pb.LoadModelRequest)
			if reqCopy.ManifestJson != "" {
				rewritten, err := r.resolveRewrittenManifest(ctx, desired)
				if err != nil {
					return err
				}
				reqCopy.ManifestJson = rewritten
			}
		}
		if desired.ByName {
			info, err := r.server.describeExecutor(ctx, execClient, endpoint)
			if err != nil {
				return r.handleRPCErr(ctx, agentClient, key, err)
			}
			reqCopy.Plan = r.server.planLoad(ctx, desired.Model, []byte(reqCopy.ManifestJson),
				[]byte(reqCopy.BindingJson), reqCopy.PrefillGraph, reqCopy.DecodeGraph, reqCopy.Plan, info)
		}
		klog.FromContext(ctx).Info("Loading model on executor", "key", key, "generation", desired.Generation, "model", desired.Model)
		if _, loadErr := execClient.LoadModel(ctx, reqCopy); loadErr != nil {
			if isRefusal(loadErr) {
				r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: loadErr, ObservedGeneration: desired.Generation})
				return nil
			}
			return r.handleRPCErr(ctx, agentClient, key, loadErr)
		}
		r.mu.Lock()
		r.loadedGen[key] = desired.Generation
		delete(r.verifiedGen, key)
		r.mu.Unlock()
		return errInstanceLoading
	}

	if verifiedGen != desired.Generation {
		if desired.Golden != nil {
			r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseChecking, ObservedGeneration: desired.Generation})
			klog.FromContext(ctx).Info("Running golden check", "key", key, "generation", desired.Generation, "model", desired.Model)
			if err := r.server.verifyGolden(ctx, execClient, key, desired.Golden); err != nil {
				if isRefusal(err) {
					klog.FromContext(ctx).Error(err, "Golden check failed", "key", key, "generation", desired.Generation, "model", desired.Model)
					r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation})
					return nil
				}
				return r.handleRPCErr(ctx, agentClient, key, err)
			}
			klog.FromContext(ctx).Info("Golden check passed", "key", key, "generation", desired.Generation, "model", desired.Model)
		} else {
			klog.FromContext(ctx).Info("No golden file for model; skipping verification", "key", key, "generation", desired.Generation, "model", desired.Model)
		}
		r.mu.Lock()
		r.verifiedGen[key] = desired.Generation
		r.mu.Unlock()
	}

	r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhasePassed, ObservedGeneration: desired.Generation})
	return nil
}

// resolveRewrittenManifest caches weights asynchronously so a multi-GB download
// never holds the reconcile worker. When a rewrite starts, it returns
// errInstanceLoading and enqueues the key as soon as the download completes.
func (r *ModelReconciler) resolveRewrittenManifest(ctx context.Context, desired InstanceDesired) (string, error) {
	raw := desired.Request.ManifestJson
	if !r.manifestNeedsAsyncRewrite(raw) {
		return r.server.rewriteManifest(ctx, raw, false)
	}
	r.mu.Lock()
	if rw := r.rewrites[desired.Key]; rw != nil && rw.generation == desired.Generation {
		if !rw.done {
			r.mu.Unlock()
			return "", errInstanceLoading
		}
		manifest, err := rw.manifest, rw.err
		if err != nil {
			delete(r.rewrites, desired.Key)
		}
		r.mu.Unlock()
		return manifest, err
	}
	if r.stopped || ctx.Err() != nil {
		r.mu.Unlock()
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", context.Canceled
	}
	r.cancelRewriteLocked(desired.Key)
	rwCtx, cancel := context.WithCancel(ctx)
	r.rewrites[desired.Key] = &manifestRewrite{generation: desired.Generation, cancel: cancel}
	r.wg.Add(1)
	r.mu.Unlock()

	klog.FromContext(ctx).Info("Starting background manifest weight rewrite", "key", desired.Key, "generation", desired.Generation, "model", desired.Model)
	go func(key string, gen int64, manifestJSON string) {
		defer r.wg.Done()
		defer cancel()
		rewritten, err := r.server.rewriteManifest(rwCtx, manifestJSON, false)
		r.mu.Lock()
		if cur := r.rewrites[key]; cur != nil && cur.generation == gen {
			cur.done, cur.manifest, cur.err = true, rewritten, err
		}
		r.mu.Unlock()
		r.queue.Add(key)
	}(desired.Key, desired.Generation, raw)
	return "", errInstanceLoading
}

func (r *ModelReconciler) manifestNeedsAsyncRewrite(manifestJSON string) bool {
	if r.server.weightStore == nil && r.server.blobServer == nil {
		return false
	}
	var m struct {
		Files map[string]json.RawMessage `json:"files"`
	}
	return json.Unmarshal([]byte(manifestJSON), &m) == nil && len(m.Files) > 0
}

func (r *ModelReconciler) teardownInstance(ctx context.Context, key string) error {
	r.removeExecutorForInstance(ctx, key)
	if r.server.clientset == nil {
		return nil
	}
	podName := "vxpu-executor-" + key
	err := r.server.clientset.CoreV1().Pods(r.server.namespace).Delete(ctx, podName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting executor pod %s: %w", podName, err)
	}
	if err == nil {
		klog.FromContext(ctx).Info("Deleted executor pod; desired state removed", "key", key, "pod", podName)
	}
	return nil
}

func (r *ModelReconciler) handleRPCErr(ctx context.Context, agentClient pb.AgentClient, key string, origErr error) error {
	agentErr := r.checkRuntimeDied(ctx, agentClient, key)
	if agentErr == nil {
		return origErr
	}
	var dead *runtimeDeadError
	if stderrors.As(agentErr, &dead) {
		return dead.err
	}
	return stderrors.Join(origErr, agentErr)
}

func (r *ModelReconciler) resolveEndpoint(ctx context.Context, desired InstanceDesired) (string, pb.AgentClient, error) {
	manifest := ""
	if desired.Request != nil {
		manifest = desired.Request.ManifestJson
	}
	bytes, err := weightBytes(manifest)
	if err != nil && manifest != "" {
		invErr := status.Errorf(codes.InvalidArgument, "invalid manifest: %v", err)
		r.setStatus(ctx, desired.Key, InstanceStatus{Phase: InstancePhaseRefused, LastError: invErr, ObservedGeneration: desired.Generation})
		return "", nil, invErr
	}
	placement := PlanPlacement(r.server.accelerator, bytes)
	podIP, err := r.server.getOrStartPod(ctx, desired.Key, &placement)
	if err != nil {
		return "", nil, err
	}
	if podIP == "" {
		return "", nil, stderrors.New("pod has no IP assigned yet")
	}

	aConn, err := r.server.agentConn(podIP)
	if err != nil {
		return "", nil, err
	}
	agentClient := pb.NewAgentClient(aConn)

	pResp, pErr := agentClient.GetProcess(ctx, &pb.GetProcessRequest{Id: "executor"})
	if pErr == nil && pResp.GetStatus() != nil {
		pst := pResp.GetStatus()
		if pst.State != "" && pst.State != "running" {
			runErr := formatExitError(pst)
			_, stopErr := agentClient.StopProcess(ctx, &pb.StopProcessRequest{Id: "executor"})
			r.removeExecutorForInstance(ctx, desired.Key)
			if stopErr != nil {
				return "", nil, stderrors.Join(runErr, fmt.Errorf("stopping runtime on agent: %w", stopErr))
			}
			return "", nil, runErr
		}
		if pst.State == "running" {
			if pst.Endpoint == "" {
				return "", nil, fmt.Errorf("agent reported running process %q with empty endpoint", "executor")
			}
			r.mu.Lock()
			r.endpoints[desired.Key] = pst.Endpoint
			r.mu.Unlock()
			return pst.Endpoint, agentClient, nil
		}
	}
	if pErr != nil && status.Code(pErr) != codes.NotFound {
		return "", nil, pErr
	}

	r.setStatus(ctx, desired.Key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
	ep, err := r.server.startRuntime(ctx, podIP)
	if err != nil {
		return "", nil, err
	}
	r.mu.Lock()
	prev := r.endpoints[desired.Key]
	r.endpoints[desired.Key] = ep
	r.mu.Unlock()
	if prev != "" && prev != ep {
		r.server.RemoveExecutor(ctx, prev)
	}
	r.server.mu.Lock()
	delete(r.server.executors, ep)
	r.server.mu.Unlock()
	return ep, agentClient, nil
}

func (r *ModelReconciler) removeExecutorForInstance(ctx context.Context, key string) {
	r.mu.Lock()
	ep := r.endpoints[key]
	delete(r.endpoints, key)
	delete(r.loadedGen, key)
	delete(r.verifiedGen, key)
	r.mu.Unlock()
	if ep != "" {
		r.server.RemoveExecutor(ctx, ep)
	}
}

func (r *ModelReconciler) checkRuntimeDied(ctx context.Context, agentClient pb.AgentClient, key string) error {
	if agentClient == nil {
		return nil
	}
	pResp, err := agentClient.GetProcess(ctx, &pb.GetProcessRequest{Id: "executor"})
	if err != nil {
		return fmt.Errorf("querying agent process table: %w", err)
	}
	if st := pResp.GetStatus(); st != nil && st.State != "" && st.State != "running" {
		runErr := formatExitError(st)
		_, stopErr := agentClient.StopProcess(ctx, &pb.StopProcessRequest{Id: "executor"})
		r.removeExecutorForInstance(ctx, key)
		if stopErr != nil {
			return &runtimeDeadError{err: stderrors.Join(runErr, fmt.Errorf("stopping runtime on agent: %w", stopErr))}
		}
		return &runtimeDeadError{err: runErr}
	}
	return nil
}

func formatExitError(st *pb.ProcessStatus) error {
	if out := strings.TrimSpace(st.Output); out != "" {
		return fmt.Errorf("runtime %s: %s", st.State, out)
	}
	return fmt.Errorf("runtime %s", st.State)
}
