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
	"net"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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

// servingEndpoint returns the address of the executor currently serving modelKey,
// or an error if no executor holds the model. It never creates a pod, but may
// start a runtime on an existing pod.
func (s *Server) servingEndpoint(ctx context.Context, modelKey string) (string, error) {
	s.mu.RLock()
	addr, ok := s.modelEndpoint[modelKey]
	isExt := ok && s.externalModel[addr] == modelKey
	s.mu.RUnlock()
	if isExt {
		return addr, nil
	}
	podIP, err := s.getOrStartPod(ctx, modelKey, nil)
	if err != nil {
		return "", err
	}
	if ok && addr != "" {
		host, _, err := net.SplitHostPort(addr)
		if err == nil && host == podIP {
			return addr, nil
		}
	}
	ep, err := s.startRuntime(ctx, podIP)
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	s.modelEndpoint[modelKey] = ep
	s.mu.Unlock()
	return ep, nil
}

// executorEndpoint is the address of the executor serving modelKey: the
// external executor holding it, else its executor pod. A model is served only
// after its golden check has passed. It never creates a pod (only LoadModel
// and loading by name do), but may start a runtime on an existing pod.
func (s *Server) executorEndpoint(ctx context.Context, modelKey string) (string, error) {
	st := s.modelStanding(modelKey)
	if st.standing == standingUnchecked {
		return "", status.Error(codes.FailedPrecondition, "model has not passed verification check; call LoadModel")
	}
	if st.standing == standingRefused {
		return "", status.Errorf(codes.FailedPrecondition, "golden check failed: %v", st.err)
	}
	if st.standing == standingFailed {
		return "", status.Errorf(codes.Unavailable, "%s", st.err)
	}

	endpoint, err := s.servingEndpoint(ctx, modelKey)
	if stderrors.Is(err, errNoExecutor) {
		s.forgetModel(modelKey, "")
		return "", status.Error(codes.FailedPrecondition, err.Error())
	}
	if err != nil {
		return "", status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}
	return endpoint, nil
}

// forgetModel drops what the router knows about where modelKey is loaded
// (its executor, at endpoint, no longer holds it), so the next request
// loads it again, and frees the executors that held its experts.
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
