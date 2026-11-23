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
	stderrors "errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/util/workqueue"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

// errInstanceLoading indicates that an executor is actively rehydrating weights.
var errInstanceLoading = stderrors.New("instance is loading on executor")

// runtimeDeadError wraps an exit error reported by the agent process table.
type runtimeDeadError struct {
	err error
}

func (e *runtimeDeadError) Error() string { return e.err.Error() }
func (e *runtimeDeadError) Unwrap() error { return e.err }

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

	// Generation is the sequence number of this desired state.
	Generation int64
}

// ModelReconciler reconciles desired model instances with actual executor state.
type ModelReconciler struct {
	// server is the router server providing executor and agent connections.
	server *Server

	// mu protects desired, statuses, endpoints, loadedGen, and verifiedGen.
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

	// queue is the work queue of instance keys to reconcile.
	queue workqueue.TypedDelayingInterface[string]

	// limiter is the rate limiter used for backoff.
	limiter workqueue.TypedRateLimiter[string]

	// pollInterval is the fixed interval to poll when an instance is actively loading.
	pollInterval time.Duration

	// resyncPeriod is the interval between periodic resyncs of all desired instances.
	resyncPeriod time.Duration
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

// NewModelReconciler creates a reconciler for the given server.
func NewModelReconciler(s *Server, opts ...ReconcilerOption) *ModelReconciler {
	r := &ModelReconciler{
		server:       s,
		desired:      make(map[string]InstanceDesired),
		statuses:     make(map[string]InstanceStatus),
		endpoints:    make(map[string]string),
		loadedGen:    make(map[string]int64),
		verifiedGen:  make(map[string]int64),
		queue:        workqueue.NewTypedDelayingQueue[string](),
		limiter:      workqueue.NewTypedItemExponentialFailureRateLimiter[string](100*time.Millisecond, 5*time.Second),
		pollInterval: 100 * time.Millisecond,
		resyncPeriod: 10 * time.Second,
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

func (r *ModelReconciler) setStatus(key string, st InstanceStatus) {
	r.mu.Lock()
	defer r.mu.Unlock()
	desired, ok := r.desired[key]
	if !ok || desired.Generation != st.ObservedGeneration {
		return
	}
	r.statuses[key] = st
}

// SetDesired registers or updates desired state for an instance and enqueues it.
func (r *ModelReconciler) SetDesired(desired InstanceDesired) {
	r.mu.Lock()
	gen := int64(1)
	if existing, ok := r.desired[desired.Key]; ok {
		gen = existing.Generation + 1
	}
	desired.Generation = gen
	r.desired[desired.Key] = desired
	current, hasStatus := r.statuses[desired.Key]
	if !hasStatus || current.Phase != InstancePhaseLoading {
		r.statuses[desired.Key] = InstanceStatus{
			Phase:              InstancePhaseNotLoaded,
			ObservedGeneration: gen,
		}
	}
	r.mu.Unlock()
	r.queue.Add(desired.Key)
}

// Enqueue adds an instance key to the reconciliation queue.
func (r *ModelReconciler) Enqueue(key string) {
	r.queue.Add(key)
}

// Start runs the worker loop in the background until ctx is canceled.
func (r *ModelReconciler) Start(ctx context.Context) {
	go func() {
		for r.processNext(ctx) {
		}
	}()
	if r.resyncPeriod > 0 {
		go func() {
			ticker := time.NewTicker(r.resyncPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					r.enqueueAllDesired()
				}
			}
		}()
	}
}

func (r *ModelReconciler) enqueueAllDesired() {
	r.mu.RLock()
	keys := make([]string, 0, len(r.desired))
	for k := range r.desired {
		keys = append(keys, k)
	}
	r.mu.RUnlock()
	for _, k := range keys {
		r.queue.Add(k)
	}
}

// Stop shuts down the work queue.
func (r *ModelReconciler) Stop() {
	r.queue.ShutDown()
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
	r.handleResult(key, err)
	return true
}

func (r *ModelReconciler) handleResult(key string, err error) {
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
	if st.Phase == InstancePhaseFailed {
		r.mu.Lock()
		cur := r.statuses[key]
		if cur.ObservedGeneration == st.ObservedGeneration {
			cur.NextAttempt = time.Now().Add(delay)
			r.statuses[key] = cur
		}
		r.mu.Unlock()
	}
}

// Reconcile observes the actual state of an instance fresh and takes the next idempotent step.
func (r *ModelReconciler) Reconcile(ctx context.Context, key string) error {
	r.mu.RLock()
	desired, ok := r.desired[key]
	st := r.statuses[key]
	loadedGen := r.loadedGen[key]
	verifiedGen := r.verifiedGen[key]
	r.mu.RUnlock()
	if !ok || (st.Phase == InstancePhaseRefused && st.ObservedGeneration == desired.Generation) {
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
		return r.handleRPCErr(ctx, agentClient, key, desired.Generation, err)
	}

	if loadResp.State == pb.LoadState_LOAD_STATE_FAILED {
		loadErr := fmt.Errorf("model load failed on executor: %s", loadResp.Error)
		r.setStatus(key, InstanceStatus{Phase: InstancePhaseFailed, LastError: loadErr, ObservedGeneration: desired.Generation})
		return loadErr
	}
	if loadResp.State == pb.LoadState_LOAD_STATE_LOADING {
		return r.handleLoading(ctx, agentClient, key, desired.Generation)
	}

	if loadResp.State != pb.LoadState_LOAD_STATE_READY || loadedGen != desired.Generation {
		r.setStatus(key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
		reqCopy := desired.Request
		if reqCopy == nil {
			reqCopy = &pb.LoadModelRequest{}
		}
		if _, loadErr := execClient.LoadModel(ctx, reqCopy); loadErr != nil {
			if status.Code(loadErr) == codes.InvalidArgument {
				r.setStatus(key, InstanceStatus{Phase: InstancePhaseRefused, LastError: loadErr, ObservedGeneration: desired.Generation})
				return nil
			}
			return r.handleRPCErr(ctx, agentClient, key, desired.Generation, loadErr)
		}
		r.mu.Lock()
		r.loadedGen[key] = desired.Generation
		r.mu.Unlock()
		return errInstanceLoading
	}

	if desired.Golden != nil && verifiedGen != desired.Generation {
		r.setStatus(key, InstanceStatus{Phase: InstancePhaseChecking, ObservedGeneration: desired.Generation})
		if err := r.server.verifyGolden(ctx, execClient, key, desired.Golden); err != nil {
			var mismatch *kernelplan.TokenMismatchError
			if stderrors.As(err, &mismatch) {
				r.setStatus(key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation})
				return nil
			}
			return r.handleRPCErr(ctx, agentClient, key, desired.Generation, err)
		}
		r.mu.Lock()
		r.verifiedGen[key] = desired.Generation
		r.mu.Unlock()
	}

	r.setStatus(key, InstanceStatus{Phase: InstancePhasePassed, ObservedGeneration: desired.Generation})
	return nil
}

func (r *ModelReconciler) handleLoading(ctx context.Context, agentClient pb.AgentClient, key string, gen int64) error {
	r.setStatus(key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: gen})
	if agentErr := r.checkRuntimeDied(ctx, agentClient, key); agentErr != nil {
		var dead *runtimeDeadError
		if stderrors.As(agentErr, &dead) {
			r.setStatus(key, InstanceStatus{Phase: InstancePhaseFailed, LastError: dead.err, ObservedGeneration: gen})
			return dead.err
		}
		return stderrors.Join(errInstanceLoading, agentErr)
	}
	return errInstanceLoading
}

func (r *ModelReconciler) handleRPCErr(ctx context.Context, agentClient pb.AgentClient, key string, desiredGen int64, origErr error) error {
	agentErr := r.checkRuntimeDied(ctx, agentClient, key)
	if agentErr == nil {
		return origErr
	}
	var dead *runtimeDeadError
	if stderrors.As(agentErr, &dead) {
		r.setStatus(key, InstanceStatus{Phase: InstancePhaseFailed, LastError: dead.err, ObservedGeneration: desiredGen})
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
		r.setStatus(desired.Key, InstanceStatus{Phase: InstancePhaseRefused, LastError: err, ObservedGeneration: desired.Generation})
		return "", nil, err
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
			r.setStatus(desired.Key, InstanceStatus{Phase: InstancePhaseFailed, LastError: runErr, ObservedGeneration: desired.Generation})
			_, stopErr := agentClient.StopProcess(ctx, &pb.StopProcessRequest{Id: "executor"})
			r.removeExecutorForInstance(desired.Key)
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

	r.setStatus(desired.Key, InstanceStatus{Phase: InstancePhaseLoading, ObservedGeneration: desired.Generation})
	ep, err := r.server.startRuntime(ctx, podIP)
	if err != nil {
		return "", nil, err
	}
	r.mu.Lock()
	r.endpoints[desired.Key] = ep
	r.mu.Unlock()
	return ep, agentClient, nil
}

func (r *ModelReconciler) removeExecutorForInstance(key string) {
	r.mu.Lock()
	ep := r.endpoints[key]
	delete(r.endpoints, key)
	delete(r.loadedGen, key)
	delete(r.verifiedGen, key)
	r.mu.Unlock()
	if ep != "" {
		r.server.RemoveExecutor(context.Background(), ep)
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
		r.removeExecutorForInstance(key)
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
