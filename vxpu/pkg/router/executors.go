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
	"fmt"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

// executorInfo is what the coordinator knows about one executor (#394):
// what it runs on, as it described itself, and the catalog machine that
// matches (nil: none does). The planner reasons over it.
type executorInfo struct {
	Endpoint string
	Probe    machines.Probe
	Machine  *machines.Machine
}

// waitForExecutor waits until the executor at endpoint answers, and
// returns what it runs on: it asks it to describe itself (once per
// endpoint) until it does. An executor that cannot describe itself (an
// older image) is ready but unknown: nil.
func (s *Server) waitForExecutor(ctx context.Context, client pb.ExecutorClient, endpoint string, attempts int) *executorInfo {
	s.mu.RLock()
	info, known := s.executors[endpoint]
	s.mu.RUnlock()
	if known {
		return &info
	}
	log := klog.FromContext(ctx)
	var resp *pb.DescribeResponse
	var err error
	for i := 0; i < attempts; i++ {
		callCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		resp, err = client.Describe(callCtx, &pb.DescribeRequest{})
		cancel()
		if err == nil || status.Code(err) == codes.Unimplemented || ctx.Err() != nil {
			break
		}
		if i+1 < attempts {
			log.Info("Waiting for the executor to answer", "endpoint", endpoint, "attempt", i, "error", err.Error())
			time.Sleep(2 * time.Second)
		}
	}
	if err != nil {
		log.Info("Executor did not describe itself", "endpoint", endpoint, "error", err.Error())
		return nil
	}
	var probe machines.Probe
	if err := json.Unmarshal([]byte(resp.HardwareJson), &probe); err != nil {
		log.Info("Executor described itself unreadably", "endpoint", endpoint, "error", err.Error())
		return nil
	}
	info = executorInfo{Endpoint: endpoint, Probe: probe, Machine: machines.Match(probe)}
	s.mu.Lock()
	if s.executors == nil {
		s.executors = map[string]executorInfo{}
	}
	s.executors[endpoint] = info
	s.mu.Unlock()
	machine := "not in the catalog"
	if info.Machine != nil {
		machine = info.Machine.Name
	}
	log.Info("Executor described", "endpoint", endpoint, "machine", machine, "cpu", probe.CPU,
		"cpu_threads", probe.CPUThreads, "ram_bytes", probe.RAMBytes, "gpu", probe.GPU, "gpu_count", probe.GPUCount)
	return &info
}

// freeExecutors describes the external executors the planner may give
// the experts of the engine at engine to: not engine itself, holding no
// model, and serving no other engine's experts. One that does not answer
// within describeWait is left out. With no engine address, none.
func (s *Server) freeExecutors(ctx context.Context, engine string) []planner.Executor {
	if engine == "" {
		return nil
	}
	var free []planner.Executor
	for _, addr := range s.externalExecutors {
		s.mu.RLock()
		_, busy := s.externalModel[addr]
		holder, serving := s.expertHolders[addr]
		info, known := s.executors[addr]
		s.mu.RUnlock()
		if addr == engine || busy || (serving && holder != engine) {
			continue
		}
		if !known {
			conn, err := s.executorConn(addr)
			if err != nil {
				continue
			}
			describeCtx, cancel := context.WithTimeout(ctx, describeWait)
			described := s.waitForExecutor(describeCtx, pb.NewExecutorClient(conn), addr, 1)
			cancel()
			if described == nil {
				continue
			}
			info = *described
		}
		free = append(free, planner.Executor{Address: addr, Probe: info.Probe})
	}
	return free
}

// describeWait is how long planning waits for an executor it has not
// heard from: the load it plans is waiting too.
const describeWait = 5 * time.Second

// claimHelper books the executor at addr for the experts of the engine at
// engine, before it is asked to load them, so two loads planning at once
// cannot both take it; false if another engine holds it.
func (s *Server) claimHelper(addr, engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if holder, ok := s.expertHolders[addr]; ok && holder != engine {
		return false
	}
	if s.expertHolders == nil {
		s.expertHolders = map[string]string{}
	}
	s.expertHolders[addr] = engine
	return true
}

// releaseHelpersLocked frees the executors holding experts for the engine
// at engine (re-planned, evicted or replaced). The caller holds s.mu.
func (s *Server) releaseHelpersLocked(engine string) {
	for addr, holder := range s.expertHolders {
		if holder == engine {
			delete(s.expertHolders, addr)
		}
	}
}

// remoteNodes is one executor's share of a model's nodes, as placed.
type remoteNodes struct {
	address string   // where they answer: host:port
	layers  []int    // the experts layers whose nodes they are
	entries []string // the engine's mailboxes for them: name=tcp://host:port/index
	reason  string
	err     error
}

// serveNodesOf has each executor (by address) serve the experts nodes of
// its layers: the decode step's nodes, as the engine's executor describes
// them (DescribeNodes), named by layer, then ServeNodes on each.
func (s *Server) serveNodesOf(ctx context.Context, model string, engine *executorInfo, manifest, binding, decode []byte,
	layersByExecutor map[string][]int) ([]remoteNodes, error) {
	if engine == nil {
		return nil, fmt.Errorf("the engine's executor did not describe itself")
	}
	conn, err := s.executorConn(engine.Endpoint)
	if err != nil {
		return nil, err
	}
	describeCtx, cancel := context.WithTimeout(ctx, 5*time.Minute) // it loads the decode graph, without weights
	described, err := pb.NewExecutorClient(conn).DescribeNodes(describeCtx, &pb.DescribeNodesRequest{
		ManifestJson: string(manifest), BindingJson: string(binding), DecodeGraph: decode})
	cancel()
	if err != nil {
		return nil, fmt.Errorf("DescribeNodes: %w", err)
	}
	var nodes []struct {
		Name       string `json:"name"`
		Collection string `json:"collection"`
		Index      int    `json:"index"`
	}
	if err := json.Unmarshal([]byte(described.NodesJson), &nodes); err != nil {
		return nil, fmt.Errorf("DescribeNodes: %w", err)
	}
	experts := map[int]string{} // layer -> its experts node
	for _, n := range nodes {
		if n.Collection == "experts" {
			experts[n.Index] = n.Name
		}
	}
	var out []remoteNodes
	for _, addr := range sortedKeys(layersByExecutor) {
		layers := layersByExecutor[addr]
		r := remoteNodes{layers: layers}
		var names []string
		for _, layer := range layers {
			name, ok := experts[layer]
			if !ok {
				r.err = fmt.Errorf("layer %d has no experts node", layer)
				break
			}
			names = append(names, name)
		}
		if r.err == nil {
			r.address, r.err = s.serveNodes(ctx, engine.Endpoint, addr, manifest, binding, decode, names)
		}
		if r.err != nil {
			r.reason = fmt.Sprintf("experts: %s could not serve layers %v (%v): the engine holds them", addr, layers, r.err)
		} else {
			for i, name := range names {
				r.entries = append(r.entries, fmt.Sprintf("%s=tcp://%s/%d", name, r.address, i))
			}
			r.reason = fmt.Sprintf("experts: %s serves the nodes of layers %d-%d at tcp://%s", addr, layers[0], layers[len(layers)-1], r.address)
		}
		out = append(out, r)
	}
	return out, nil
}

// serveNodes asks the executor at addr to serve nodes of the artifact to
// the engine at engine, and returns where (host:port). The executor is
// claimed for that engine first, and freed if it fails.
func (s *Server) serveNodes(ctx context.Context, engine, addr string, manifest, binding, decode []byte, names []string) (string, error) {
	if !s.claimHelper(addr, engine) {
		return "", fmt.Errorf("%s serves another engine's experts", addr)
	}
	served := false
	defer func() {
		if !served {
			s.mu.Lock()
			if s.expertHolders[addr] == engine {
				delete(s.expertHolders, addr)
			}
			s.mu.Unlock()
		}
	}()
	conn, err := s.executorConn(addr)
	if err != nil {
		return "", err
	}
	callCtx, cancel := context.WithTimeout(ctx, 30*time.Minute) // it loads the residents and builds the packages first
	defer cancel()
	resp, err := pb.NewExecutorClient(conn).ServeNodes(callCtx, &pb.ServeNodesRequest{
		ManifestJson: string(manifest), BindingJson: string(binding), DecodeGraph: decode, Nodes: names})
	if err != nil {
		return "", err
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return "", err
	}
	served = true
	return net.JoinHostPort(host, strconv.Itoa(int(resp.Port))), nil
}

const executorServiceConfig = `{
	"methodConfig": [{
		"name": [
			{"service": "vxpu.Executor", "method": "TokenizeChat"},
			{"service": "vxpu.Executor", "method": "GetVocab"},
			{"service": "vxpu.Executor", "method": "NewSession"},
			{"service": "vxpu.Executor", "method": "Describe"},
			{"service": "vxpu.Executor", "method": "DescribeNodes"},
			{"service": "vxpu.Executor", "method": "GetLoad"},
			{"service": "vxpu.Executor", "method": "LoadModel"}
		],
		"waitForReady": true,
		"retryPolicy": {
			"maxAttempts": 5,
			"initialBackoff": "0.01s",
			"maxBackoff": "0.1s",
			"backoffMultiplier": 2.0,
			"retryableStatusCodes": ["UNAVAILABLE"]
		}
	}]
}`

// dialExecutor connects to the executor at addr, with room for an
// artifact's graphs in one message. It configures retries for methods that are
// safe to repeat if an executor restarts.
func dialExecutor(addr string) (*grpc.ClientConn, error) {
	maxMsgSize := 128 * 1024 * 1024
	return grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
		grpc.WithDefaultServiceConfig(executorServiceConfig),
	)
}
