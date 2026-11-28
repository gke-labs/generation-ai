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

	// PlanHash is the identity of the plan the executor reported for the loaded instance.
	PlanHash string
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

// instanceState holds everything the reconciler knows about one model instance.
// ModelReconciler.mu guards all fields except lastUsed, which uses atomic
// operations so Touch records activity without acquiring mu.
type instanceState struct {
	// desired holds the desired specification for this instance, or nil when none is set.
	desired *InstanceDesired
	// status holds the current observed lifecycle standing of this instance.
	status InstanceStatus
	// endpoint holds the last resolved executor endpoint for this instance.
	endpoint string
	// loadedGen records the generation that finished loading on an executor.
	loadedGen int64
	// verifiedGen records the generation that passed verification checks.
	verifiedGen int64
	// passedGen records the latest generation that reached Passed.
	passedGen int64
	// lastUsed stores monotonic nanoseconds since the reconciler epoch for keep-alive tracking.
	lastUsed atomic.Int64
	// rewrite tracks an in-flight or completed background manifest rewrite.
	rewrite *manifestRewrite
	// teardown reports whether the reconciler removed desired state and still needs to tear down this instance.
	teardown bool
	// notify wakes blocked Wait callers when status or desired state changes.
	notify chan struct{}
}

func (inst *instanceState) notifyLocked() {
	if inst.notify != nil {
		close(inst.notify)
		inst.notify = nil
	}
}

func (inst *instanceState) cancelRewriteLocked() {
	if inst.rewrite != nil {
		if inst.rewrite.cancel != nil {
			inst.rewrite.cancel()
		}
		inst.rewrite = nil
	}
}

func (inst *instanceState) deleteDesiredLocked() {
	inst.desired = nil
	inst.loadedGen, inst.verifiedGen, inst.passedGen = 0, 0, 0
	inst.status = InstanceStatus{}
	inst.cancelRewriteLocked()
	inst.teardown = true
	inst.notifyLocked()
}

func (inst *instanceState) isExpiredLocked(keepAlive time.Duration, nowNanos int64) bool {
	if keepAlive <= 0 || inst.desired == nil || inst.passedGen != inst.desired.Generation {
		return false
	}
	if inst.status.Phase == InstancePhaseNotLoaded || inst.status.Phase == InstancePhaseLoading || inst.status.Phase == InstancePhaseChecking {
		return false
	}
	return time.Duration(nowNanos-inst.lastUsed.Load()) >= keepAlive
}

// ModelReconciler reconciles desired model instances with actual executor state.
type ModelReconciler struct {
	// server is the router server providing executor and agent connections.
	server *Server

	// mu guards writes to instances, all non-atomic fields of each instanceState, stopCancel, and stopped.
	mu sync.RWMutex

	// instances maps each instance key to its *instanceState.
	// Storing entries in a sync.Map lets Touch update lastUsed without acquiring mu.
	instances sync.Map

	// epoch is the monotonic time base for lastUsed timestamps.
	epoch time.Time

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

	// stopCancel cancels the loop context and any background rewrites when Stop runs.
	stopCancel context.CancelFunc

	// stopped reports whether Stop has run.
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
		epoch:        time.Now(),
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

func (r *ModelReconciler) get(key string) *instanceState {
	if v, ok := r.instances.Load(key); ok {
		return v.(*instanceState)
	}
	return nil
}

func (r *ModelReconciler) getOrCreateLocked(key string) *instanceState {
	if inst := r.get(key); inst != nil {
		return inst
	}
	inst := &instanceState{}
	r.instances.Store(key, inst)
	return inst
}

// Status reports the current status of an instance.
func (r *ModelReconciler) Status(key string) InstanceStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if inst := r.get(key); inst != nil && inst.status.Phase != "" {
		return inst.status
	}
	return InstanceStatus{Phase: InstancePhaseNotLoaded}
}

// Desired reports the current desired state recorded for an instance.
func (r *ModelReconciler) Desired(key string) (InstanceDesired, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if inst := r.get(key); inst != nil && inst.desired != nil {
		return *inst.desired, true
	}
	return InstanceDesired{}, false
}

// Touch records activity for an instance, extending its keep-alive window.
// Serving requests pay one lock-free sync.Map lookup and one atomic store without acquiring r.mu.
func (r *ModelReconciler) Touch(key string) {
	if inst := r.get(key); inst != nil {
		inst.lastUsed.Store(int64(time.Since(r.epoch)))
	}
}

// DeleteDesired removes the desired state for an instance owned by the reconciler
// and enqueues it for teardown. Calling DeleteDesired for a key the reconciler
// never held does nothing.
func (r *ModelReconciler) DeleteDesired(key string) {
	r.mu.Lock()
	inst := r.get(key)
	if inst == nil || inst.desired == nil {
		r.mu.Unlock()
		return
	}
	inst.deleteDesiredLocked()
	r.mu.Unlock()
	r.limiter.Forget(key)
	r.queue.Add(key)
}

func (r *ModelReconciler) setStatus(ctx context.Context, key string, st InstanceStatus) {
	r.mu.Lock()
	defer r.mu.Unlock()
	inst := r.get(key)
	if inst == nil || inst.desired == nil || inst.desired.Generation != st.ObservedGeneration {
		return
	}
	prev := inst.status
	inst.status = st
	if st.Phase == InstancePhasePassed && inst.passedGen != st.ObservedGeneration {
		inst.passedGen = st.ObservedGeneration
		inst.lastUsed.Store(int64(time.Since(r.epoch)))
	}
	inst.notifyLocked()
	if prev.Phase != st.Phase || prev.ObservedGeneration != st.ObservedGeneration || prev.LastError != st.LastError || prev.PlanHash != st.PlanHash {
		klog.FromContext(ctx).Info("Instance status changed",
			"key", key, "phase", st.Phase, "generation", st.ObservedGeneration, "plan_hash", st.PlanHash, "error", st.LastError)
	}
}

func desiredEqual(a, b InstanceDesired) bool {
	return a.Key == b.Key && a.Model == b.Model && a.ByName == b.ByName && proto.Equal(a.Golden, b.Golden) && proto.Equal(a.Request, b.Request)
}

// SetDesired registers or updates desired state for an instance and enqueues it.
// It returns the generation of the desired state.
func (r *ModelReconciler) SetDesired(desired InstanceDesired) int64 {
	localAgent := r.server.hasLocalAgent()
	var evicted []string

	r.mu.Lock()
	if r.stopped {
		r.mu.Unlock()
		return 0
	}
	if localAgent {
		// Local-agent mode runs one shared executor process, so switching to a
		// different model evicts any other active instance immediately.
		r.instances.Range(func(k, v any) bool {
			otherKey, other := k.(string), v.(*instanceState)
			if otherKey != desired.Key && other.desired != nil {
				other.deleteDesiredLocked()
				r.limiter.Forget(otherKey)
				r.queue.Add(otherKey)
				evicted = append(evicted, otherKey)
			}
			return true
		})
	}
	inst := r.getOrCreateLocked(desired.Key)
	inst.teardown = false
	desired.Generation = 1
	if inst.desired != nil {
		if desiredEqual(*inst.desired, desired) {
			inst.lastUsed.Store(int64(time.Since(r.epoch)))
			if inst.status.Phase != InstancePhasePassed && inst.status.Phase != InstancePhaseRefused {
				if inst.status.Phase == InstancePhaseFailed {
					inst.status = InstanceStatus{Phase: InstancePhaseNotLoaded, ObservedGeneration: inst.desired.Generation}
					inst.loadedGen = 0
					inst.notifyLocked()
				}
				r.queue.Add(desired.Key)
			}
			gen := inst.desired.Generation
			r.mu.Unlock()
			for _, k := range evicted {
				r.server.forgetLocalModel(k)
			}
			return gen
		}
		desired.Generation = inst.desired.Generation + 1
	}
	inst.desired = &desired
	inst.cancelRewriteLocked()
	if inst.status.Phase != InstancePhaseLoading {
		inst.status = InstanceStatus{Phase: InstancePhaseNotLoaded, ObservedGeneration: desired.Generation}
	}
	inst.notifyLocked()
	r.queue.Add(desired.Key)
	r.mu.Unlock()
	for _, k := range evicted {
		r.server.forgetLocalModel(k)
	}
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
		if r.stopped {
			r.mu.Unlock()
			return InstanceStatus{}, status.Error(codes.Unavailable, "router is shutting down")
		}
		inst := r.get(key)
		if inst == nil || inst.desired == nil {
			r.mu.Unlock()
			return InstanceStatus{Phase: InstancePhaseNotLoaded}, status.Error(codes.Aborted, "model instance was removed")
		}
		st := inst.status
		if st.ObservedGeneration >= gen && (st.Phase == InstancePhasePassed || st.Phase == InstancePhaseRefused || st.Phase == InstancePhaseFailed) {
			r.mu.Unlock()
			return st, nil
		}
		if inst.notify == nil {
			inst.notify = make(chan struct{})
		}
		ch := inst.notify
		r.mu.Unlock()

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
	r.instances.Range(func(_, v any) bool {
		inst := v.(*instanceState)
		if inst.desired != nil {
			remaining++
		}
		inst.cancelRewriteLocked()
		inst.notifyLocked()
		return true
	})
	return alreadyStopped, remaining, cancel
}

func (r *ModelReconciler) resync(ctx context.Context) {
	r.mu.Lock()
	nowNanos := int64(time.Since(r.epoch))
	var expired, active []string
	r.instances.Range(func(k, v any) bool {
		key, inst := k.(string), v.(*instanceState)
		if inst.desired != nil {
			if inst.isExpiredLocked(r.keepAlive, nowNanos) {
				inst.deleteDesiredLocked()
				expired = append(expired, key)
			} else {
				active = append(active, key)
			}
		}
		return true
	})
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
		if desired, ok := r.Desired(key); ok {
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
	inst := r.get(key)
	if inst == nil {
		return nil
	}
	r.mu.RLock()
	desired, shouldTeardown, st, loadedGen, verifiedGen := inst.desired, inst.teardown, inst.status, inst.loadedGen, inst.verifiedGen
	r.mu.RUnlock()
	if desired == nil {
		if !shouldTeardown {
			return nil
		}
		if err := r.teardownInstance(ctx, key, inst); err != nil {
			return err
		}
		r.mu.Lock()
		inst.teardown = false
		if inst.desired == nil && inst.endpoint == "" && inst.rewrite == nil && inst.notify == nil && inst.status.Phase == "" {
			r.instances.Delete(key)
		}
		r.mu.Unlock()
		return nil
	}
	if st.Phase == InstancePhaseRefused && st.ObservedGeneration == desired.Generation {
		return nil
	}

	endpoint, agentClient, err := r.resolveEndpoint(ctx, inst, desired)
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
		return r.handleRPCErr(ctx, agentClient, inst, err)
	}

	if loadResp.State == pb.LoadState_LOAD_STATE_FAILED && loadedGen == desired.Generation {
		r.mu.Lock()
		inst.loadedGen = 0
		r.mu.Unlock()
		return fmt.Errorf("model load failed on executor: %s", loadResp.Error)
	}
	if loadResp.State == pb.LoadState_LOAD_STATE_LOADING {
		r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
		return r.handleRPCErr(ctx, agentClient, inst, errInstanceLoading)
	}

	if loadResp.State != pb.LoadState_LOAD_STATE_READY || loadedGen != desired.Generation {
		r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
		reqCopy := &pb.LoadModelRequest{}
		if desired.Request != nil {
			reqCopy = proto.Clone(desired.Request).(*pb.LoadModelRequest)
			if reqCopy.ManifestJson != "" {
				rewritten, err := r.resolveRewrittenManifest(ctx, inst, desired)
				if err != nil {
					return err
				}
				reqCopy.ManifestJson = rewritten
			}
		}
		var info *executorInfo
		if desired.ByName || (desired.Request != nil && desired.Request.ManifestJson != "" && reqCopy.PlanFile == nil) {
			var err error
			info, err = r.server.describeExecutor(ctx, execClient, endpoint)
			if err != nil {
				return r.handleRPCErr(ctx, agentClient, inst, err)
			}
		}
		if desired.Request != nil && desired.Request.ManifestJson != "" && reqCopy.PlanFile == nil {
			planFile, err := emitPlanForExecutorInfo(ctx, info, agentClient, desired.Request)
			if err != nil {
				if isRefusal(err) {
					r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation})
					return nil
				}
				return err
			}
			reqCopy.PlanFile = planFile
		}
		if desired.ByName {
			reqCopy.Plan = stripPlanReplacedSwitches(
				r.server.planLoad(ctx, desired.Model, []byte(reqCopy.ManifestJson),
					[]byte(reqCopy.BindingJson), reqCopy.PrefillGraph, reqCopy.DecodeGraph, reqCopy.Plan, info, reqCopy.PlanFile),
				reqCopy.PlanFile,
			)
		}
		if err := r.stagePlanPackages(ctx, agentClient, desired, reqCopy); err != nil {
			if isRefusal(err) {
				r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation})
				return nil
			}
			return err
		}
		klog.FromContext(ctx).Info("Loading model on executor", "key", key, "generation", desired.Generation, "model", desired.Model, "plan_hash", reqCopy.GetPlanFile().GetPlanHash())
		if _, loadErr := execClient.LoadModel(ctx, reqCopy); loadErr != nil {
			if isRefusal(loadErr) {
				r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: loadErr, ObservedGeneration: desired.Generation})
				return nil
			}
			return r.handleRPCErr(ctx, agentClient, inst, loadErr)
		}
		r.mu.Lock()
		inst.loadedGen, inst.verifiedGen = desired.Generation, 0
		r.mu.Unlock()
		return errInstanceLoading
	}

	reportedPlanHash := loadResp.GetPlanHash()
	if verifiedGen != desired.Generation {
		if desired.Golden != nil {
			r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseChecking, ObservedGeneration: desired.Generation, PlanHash: reportedPlanHash})
			klog.FromContext(ctx).Info("Running golden check", "key", key, "generation", desired.Generation, "model", desired.Model)
			if err := r.server.verifyGolden(ctx, execClient, key, desired.Golden); err != nil {
				if isRefusal(err) {
					klog.FromContext(ctx).Error(err, "Golden check failed", "key", key, "generation", desired.Generation, "model", desired.Model)
					r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation, PlanHash: reportedPlanHash})
					return nil
				}
				return r.handleRPCErr(ctx, agentClient, inst, err)
			}
			klog.FromContext(ctx).Info("Golden check passed", "key", key, "generation", desired.Generation, "model", desired.Model)
		} else {
			klog.FromContext(ctx).Info("No golden file for model; skipping verification", "key", key, "generation", desired.Generation, "model", desired.Model)
		}
		r.mu.Lock()
		inst.verifiedGen = desired.Generation
		r.mu.Unlock()
	}

	r.setStatus(ctx, key, InstanceStatus{Phase: InstancePhasePassed, ObservedGeneration: desired.Generation, PlanHash: reportedPlanHash})
	return nil
}

func (r *ModelReconciler) stagePlanPackages(ctx context.Context, agentClient pb.AgentClient, desired *InstanceDesired, req *pb.LoadModelRequest) error {
	keys := planPackageKeys(req.GetPlanFile())
	if len(keys) == 0 {
		return nil
	}
	if agentClient == nil {
		return status.Error(codes.FailedPrecondition, "plan names compiled packages but executor has no agent to stage them")
	}
	model := desired.Model
	if model == "" {
		model = modelFromManifest(req.GetManifestJson())
	}
	_, err := agentClient.StagePackages(ctx, &pb.StagePackagesRequest{
		Model: model,
		Keys:  keys,
	})
	return err
}

func planPackageKeys(pf *planpb.PlanFile) []string {
	if pf == nil {
		return nil
	}
	var keys []string
	seen := make(map[string]bool)
	for _, seg := range pf.GetSegments() {
		k := seg.GetPackageKey()
		if k != "" && !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	return keys
}

// emitPlanForExecutorInfo derives a PlanFile for req targeting the probed
// executor described by info. It reads the software version from agentClient,
// or falls back to info.Probe when no agent is attached.
func emitPlanForExecutorInfo(ctx context.Context, info *executorInfo, agentClient pb.AgentClient, req *pb.LoadModelRequest) (*planpb.PlanFile, error) {
	machineKind, err := executorMachineKind(info)
	if err != nil {
		return nil, status.Errorf(codes.FailedPrecondition, "plan emit: %v", err)
	}
	softwareVersion, err := executorSoftwareVersion(ctx, agentClient, info)
	if err != nil {
		return nil, err
	}
	planFile, err := kernelplan.PlanArtifact(ctx, kernelplan.EmitOptions{
		ManifestBytes: []byte(req.ManifestJson),
		BindingBytes:  []byte(req.BindingJson),
		DecodeGraph:   req.DecodeGraph,
		TreeCommit:    softwareVersion,
		MachineKind:   machineKind,
		Goal:          req.Goal,
	})
	if err != nil {
		if stderrors.Is(err, kernelplan.ErrSoftwareVersionRequired) {
			return nil, status.Error(codes.FailedPrecondition, "plan emit: the router does not yet know the executor's software version and so cannot name packages")
		}
		return nil, status.Errorf(codes.InvalidArgument, "plan emit: %v", err)
	}
	return planFile, nil
}

// executorSoftwareVersion reads the software version the agent reports in its
// machine probe, falling back to the executor's hardware probe when no agent
// is attached. When neither reports a version, it returns an empty string so
// plans without packaged segments still succeed.
func executorSoftwareVersion(ctx context.Context, agentClient pb.AgentClient, info *executorInfo) (string, error) {
	if agentClient != nil {
		probe, err := agentClient.DescribeMachine(ctx, &pb.DescribeMachineRequest{})
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(probe.GetSoftwareVersion()), nil
	}
	if info != nil && info.Probe != nil {
		return strings.TrimSpace(info.Probe.GetSoftwareVersion()), nil
	}
	return "", nil
}

// executorMachineKind derives the target hardware identifier for planning from
// an executor's matched catalog machine or hardware probe. It calls the same
// kernelplan.MachineKind function as `vxpu plan emit` whenever a catalog
// machine is known, and returns an error when the machine is not known.
func executorMachineKind(info *executorInfo) (string, error) {
	if info == nil {
		return "", stderrors.New("executor machine is not known: executor did not describe its hardware")
	}
	if info.Machine != nil {
		return kernelplan.MachineKind(info.Machine)
	}
	if info.Probe != nil {
		if strings.TrimSpace(info.Probe.Gpu) == "" && strings.TrimSpace(info.Probe.Cpu) == "" {
			return "", stderrors.New("executor machine is not known: hardware probe reported neither GPU nor CPU")
		}
		return kernelplan.MachineKindFromProbe(info.Probe)
	}
	return "", stderrors.New("executor machine is not known: executor reported neither catalog machine nor hardware probe")
}

// resolveRewrittenManifest caches weights asynchronously so a multi-GB download
// never holds the reconcile worker. When a rewrite starts, it returns
// errInstanceLoading and enqueues the key as soon as the download completes.
func (r *ModelReconciler) resolveRewrittenManifest(ctx context.Context, inst *instanceState, desired *InstanceDesired) (string, error) {
	raw := desired.Request.ManifestJson
	if !r.manifestNeedsAsyncRewrite(raw) {
		return r.server.rewriteManifest(ctx, raw, false)
	}
	r.mu.Lock()
	if rw := inst.rewrite; rw != nil && rw.generation == desired.Generation {
		if !rw.done {
			r.mu.Unlock()
			return "", errInstanceLoading
		}
		manifest, err := rw.manifest, rw.err
		if err != nil {
			inst.rewrite = nil
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
	inst.cancelRewriteLocked()
	rwCtx, cancel := context.WithCancel(ctx)
	rw := &manifestRewrite{generation: desired.Generation, cancel: cancel}
	inst.rewrite = rw
	r.wg.Add(1)
	r.mu.Unlock()

	klog.FromContext(ctx).Info("Starting background manifest weight rewrite", "key", desired.Key, "generation", desired.Generation, "model", desired.Model)
	go func(key, manifestJSON string) {
		defer r.wg.Done()
		defer cancel()
		rewritten, err := r.server.rewriteManifest(rwCtx, manifestJSON, false)
		r.mu.Lock()
		if inst.rewrite == rw {
			rw.done, rw.manifest, rw.err = true, rewritten, err
		}
		r.mu.Unlock()
		r.queue.Add(key)
	}(desired.Key, raw)
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

func (r *ModelReconciler) teardownInstance(ctx context.Context, key string, inst *instanceState) error {
	if r.server.hasLocalAgent() {
		r.mu.Lock()
		inst.endpoint, inst.loadedGen, inst.verifiedGen = "", 0, 0
		r.mu.Unlock()
		return nil
	}
	r.removeExecutorForInstance(ctx, inst)
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

func (r *ModelReconciler) handleRPCErr(ctx context.Context, agentClient pb.AgentClient, inst *instanceState, origErr error) error {
	agentErr := r.checkRuntimeDied(ctx, agentClient, inst)
	if agentErr == nil {
		return origErr
	}
	var dead *runtimeDeadError
	if stderrors.As(agentErr, &dead) {
		return dead.err
	}
	return stderrors.Join(origErr, agentErr)
}

func (r *ModelReconciler) resolveEndpoint(ctx context.Context, inst *instanceState, desired *InstanceDesired) (string, pb.AgentClient, error) {
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
			r.removeExecutorForInstance(ctx, inst)
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
			inst.endpoint = pst.Endpoint
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
	prev := inst.endpoint
	inst.endpoint = ep
	if prev != "" && prev != ep {
		inst.loadedGen = 0
	}
	r.mu.Unlock()
	if prev != "" && prev != ep {
		r.server.RemoveExecutor(ctx, prev)
	}
	r.server.mu.Lock()
	delete(r.server.executors, ep)
	r.server.mu.Unlock()
	return ep, agentClient, nil
}

func (r *ModelReconciler) removeExecutorForInstance(ctx context.Context, inst *instanceState) {
	r.mu.Lock()
	ep := inst.endpoint
	inst.endpoint, inst.loadedGen, inst.verifiedGen = "", 0, 0
	r.mu.Unlock()
	if ep != "" {
		r.server.RemoveExecutor(ctx, ep)
	}
}

func (r *ModelReconciler) checkRuntimeDied(ctx context.Context, agentClient pb.AgentClient, inst *instanceState) error {
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
		r.removeExecutorForInstance(ctx, inst)
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
