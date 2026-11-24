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

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"strings"
)

// External executors (#357): executor servers on machines the router does
// not create (GPU VMs launched with gcloud, Macs), each running
// `python -m vxpu.server`. GPUs are easier to obtain as VMs than as
// Kubernetes nodes, an executor machine is not shared with other
// workloads, and its local disk is its weight cache. The router only
// dials them: models loaded by name are placed on one, and sessions are
// routed to the executor holding their model. Executor pods remain the
// path for artifacts shipped with LoadModel.

// UseExternalExecutors sets the external executors' addresses (host:port).
func (s *Server) UseExternalExecutors(ctx context.Context, addrs []string) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	newAddrs := make(map[string]bool, len(addrs))
	for _, a := range addrs {
		newAddrs[a] = true
	}
	for _, old := range s.externalExecutors {
		if !newAddrs[old] {
			s.removeExecutorLocked(ctx, old)
		}
	}
	s.externalExecutors = append([]string(nil), addrs...)
}

// UseExternalModelstoreURL sets the address external executors fetch
// weights from: modelstore as reachable from outside the cluster (an
// internal load balancer, say). Unset, they are given the router's own
// modelstore URL, which suits executors that can resolve it (in the
// cluster) or that copy weights from the bucket themselves
// (VXPU_BLOB_BUCKET, localblobs.py).
func (s *Server) UseExternalModelstoreURL(url string) {
	s.externalModelstoreURL = strings.TrimRight(url, "/")
}

// pickExternalExecutor chooses where model modelKey goes: the executor
// that already holds it, else one holding nothing, else the first (an
// executor holds one model; loading another replaces it).
func (s *Server) pickExternalExecutor(modelKey string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if addr, ok := s.modelEndpoint[modelKey]; ok {
		return addr
	}
	for _, addr := range s.externalExecutors {
		_, busy := s.externalModel[addr]
		_, serving := s.expertHolders[addr] // its RAM holds another model's experts
		if !busy && !serving {
			return addr
		}
	}
	return s.externalExecutors[0]
}

// placedExternally records that the executor at addr now holds modelKey
// (and no longer whatever it held before).
func (s *Server) placedExternally(modelKey, addr string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if previous, ok := s.externalModel[addr]; ok && previous != modelKey {
		s.releaseHelpersLocked(addr) // the experts previous's engine had elsewhere
		delete(s.modelEndpoint, previous)
		delete(s.models, previous)
		for tag, key := range s.loadedTags {
			if key == previous {
				delete(s.loadedTags, tag)
			}
		}
	}
	s.externalModel[addr] = modelKey
	s.modelEndpoint[modelKey] = addr
}

// isExternal reports whether endpoint is an unmanaged external executor.
func (s *Server) isExternal(endpoint string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, addr := range s.externalExecutors {
		if addr == endpoint {
			return true
		}
	}
	return false
}

// executorEndpoint returns the address of the executor serving modelKey: the
// external executor holding it, or its reconciler-managed executor pod. A model
// is served only after its golden check has passed.
func (s *Server) executorEndpoint(ctx context.Context, modelKey string) (string, error) {
	s.mu.RLock()
	addr, isExt := s.modelEndpoint[modelKey]
	s.mu.RUnlock()
	if isExt {
		if st := s.modelStanding(modelKey); st.standing == standingRefused {
			return "", status.Errorf(codes.FailedPrecondition, "golden check failed: %v", st.err)
		} else if st.standing == standingUnchecked {
			return "", status.Error(codes.FailedPrecondition, "model has not passed verification check; call LoadModel")
		}
		return addr, nil
	}
	switch st := s.reconciler.Status(modelKey); st.Phase {
	case InstancePhasePassed:
	case InstancePhaseRefused:
		return "", status.Errorf(codes.FailedPrecondition, "golden check failed: %v", st.LastError)
	default:
		return "", status.Error(codes.FailedPrecondition, "model has not passed verification check; call LoadModel")
	}
	podIP, err := s.getOrStartPod(ctx, modelKey, nil)
	if stderrors.Is(err, errNoExecutor) {
		// Request observations never edit desired state; nudge the loop to recreate the pod.
		s.reconciler.Enqueue(modelKey)
		return "", status.Error(codes.FailedPrecondition, err.Error())
	}
	if err != nil {
		return "", status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}
	ep, err := s.startRuntime(ctx, podIP)
	if err != nil {
		return "", status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}
	if s.reconciler != nil {
		s.reconciler.Touch(modelKey)
	}
	return ep, nil
}

// forgetModel drops what the router knows about where an external executor
// holds modelKey, so the next request loads it again, and frees any helper
// executors that held its experts.
func (s *Server) forgetModel(modelKey, endpoint string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseHelpersLocked(endpoint) // the executors holding its engine's experts
	if addr, ok := s.modelEndpoint[modelKey]; ok && s.externalModel[addr] == modelKey {
		delete(s.externalModel, addr)
	}
	delete(s.modelEndpoint, modelKey)
	delete(s.vocabs, modelKey)
	delete(s.models, modelKey)
	for tag, key := range s.loadedTags {
		if key == modelKey {
			delete(s.loadedTags, tag)
		}
	}
}
