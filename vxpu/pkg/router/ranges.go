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
	"maps"
	"net"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

// This file runs a model as ranges of whole layers on several executors
// (#387). The planner may find that the cheapest layout splits the model's
// layers into ranges on different executors. When it does, the router
// loads the model on each of those executors with its range set
// (VXPU_LAYER_RANGE) and asks each to serve its range (ServeRange). The
// engine runs the first range, and its plan tells it where the others are
// (VXPU_REMOTE_RANGES), so it can send each step on to them.

// candidateSummary describes one layout the planner priced. ExplainModel
// reports these.
type candidateSummary struct {
	Kind       string  `json:"kind"`
	Summary    string  `json:"summary"`
	MsPerToken float64 `json:"ms_per_token,omitempty"`
	Infeasible string  `json:"infeasible,omitempty"`
}

// readCuts asks the engine's executor where the artifact's step can be cut
// into ranges of whole layers. It sends the prefill graph with
// DescribeNodes, and the executor analyzes both graphs.
func (s *Server) readCuts(ctx context.Context, info *executorInfo, a *planner.Artifact, manifest, binding, prefill, decode []byte) error {
	conn, err := dialExecutor(info.Endpoint)
	if err != nil {
		return err
	}
	defer closeConn(ctx, conn)
	callCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	resp, err := pb.NewExecutorClient(conn).DescribeNodes(callCtx, &pb.DescribeNodesRequest{
		ManifestJson: string(manifest), BindingJson: string(binding), DecodeGraph: decode, PrefillGraph: prefill})
	if err != nil {
		return fmt.Errorf("DescribeNodes: %w", err)
	}
	if resp.CutsJson == "" {
		return fmt.Errorf("the executor did not say (it predates cuts)")
	}
	return planner.ReadCuts(a, []byte(resp.CutsJson))
}

// chooseRanges prices the layouts that start on the engine. If the cheapest
// layout runs ranges of whole layers, chooseRanges loads each later range on
// its executor, starts serving it, and returns the switches for the engine.
//
// Otherwise it returns nil switches, and the caller plans as before by
// spreading the experts. It records the reason in mp. If a range failed to
// load, it also returns the executors that already hold a range, so the
// caller doesn't offer them for anything else.
func (s *Server) chooseRanges(ctx context.Context, model string, a planner.Artifact, info *executorInfo, free []planner.Executor,
	counts map[int]map[int]int64, hints map[string]string, manifest, binding, prefill, decode []byte,
	mp *modelPlan) (map[string]string, []string) {
	if info == nil || len(free) == 0 || a.Cuts == nil || len(a.Experts) == 0 {
		return nil, nil
	}
	var others []planner.Node
	for _, e := range free {
		others = append(others, planner.Node{Name: e.Address, Probe: e.Probe})
	}
	cands := planner.CandidatesFor(a, planner.Node{Name: info.Endpoint, Probe: info.Probe}, others, counts, hints)
	for _, c := range cands {
		mp.Candidates = append(mp.Candidates, candidateSummary{c.Kind, c.Summary, c.Prediction.MsPerToken, c.Infeasible})
	}
	if len(cands) == 0 || cands[0].Kind != "layer-ranges" || cands[0].Infeasible != "" {
		return nil, nil
	}
	best := cands[0]
	entries, loaded, err := s.serveRanges(ctx, model, info.Endpoint, best, hints, manifest, binding, prefill, decode)
	if err != nil {
		mp.Reasons = append(mp.Reasons, fmt.Sprintf("layer ranges (%s, %.1f ms predicted) could not start: %v; spreading the experts instead",
			best.Summary, best.Prediction.MsPerToken, err))
		return nil, loaded
	}
	head := best.Placements[0]
	sw := maps.Clone(head.Switches)
	if sw == nil {
		sw = map[string]string{}
	}
	sw["VXPU_DATAFLOW"] = "1"
	sw["VXPU_LAYER_RANGE"] = fmt.Sprintf("0:%d", best.Placements[1].Layers[0])
	sw["VXPU_REMOTE_RANGES"] = strings.Join(entries, ",")
	mp.Switches, mp.RemoteRanges, mp.Prediction = sw, entries, best.Prediction
	mp.Reasons = append(mp.Reasons, fmt.Sprintf("layout: %s, %.1f ms predicted, the cheapest of %d priced", best.Summary,
		best.Prediction.MsPerToken, len(cands)))
	mp.Reasons = append(mp.Reasons, best.Reasons...)
	return sw, nil
}

// serveRanges loads each range after the first on its executor, and asks
// that executor to serve the range to the engine at engine. Before loading,
// it claims the executor for the engine. It returns one START:END=tcp://host:port
// entry per range, in order, and the executors it loaded.
func (s *Server) serveRanges(ctx context.Context, model, engine string, c planner.Candidate, hints map[string]string,
	manifest, binding, prefill, decode []byte) ([]string, []string, error) {
	log := klog.FromContext(ctx)
	var entries, loaded []string
	for i := 1; i < len(c.Placements); i++ {
		p := c.Placements[i]
		end := ""
		if i+1 < len(c.Placements) {
			end = strconv.Itoa(c.Placements[i+1].Layers[0])
		}
		span := fmt.Sprintf("%d:%s", p.Layers[0], end)
		plan := maps.Clone(p.Switches)
		if plan == nil {
			plan = map[string]string{}
		}
		for k, v := range hints {
			if k != "VXPU_GPU_EXPERTS" { // this placement describes the engine's experts, not this range's
				plan[k] = v
			}
		}
		plan["VXPU_DATAFLOW"] = "1"
		plan["VXPU_LAYER_RANGE"] = span
		if !s.claimHelper(p.Node, engine) {
			return nil, loaded, fmt.Errorf("layers %s on %s: it serves another engine", span, p.Node)
		}
		hostport, err := s.loadRange(ctx, p.Node, plan, manifest, binding, prefill, decode)
		if err != nil {
			s.mu.Lock()
			if s.expertHolders[p.Node] == engine {
				delete(s.expertHolders, p.Node)
			}
			s.mu.Unlock()
			return nil, loaded, fmt.Errorf("layers %s on %s: %w", span, p.Node, err)
		}
		loaded = append(loaded, p.Node)
		entries = append(entries, fmt.Sprintf("%s=tcp://%s", span, hostport))
		log.Info("Serving a range of layers", "model", model, "layers", span, "executor", p.Node, "at", hostport)
	}
	return entries, loaded, nil
}

// loadRange loads the model on the executor at addr with plan, which sets
// the executor's range. It then waits for the executor to serve the range,
// and returns the address it serves on, as host:port.
func (s *Server) loadRange(ctx context.Context, addr string, plan map[string]string, manifest, binding, prefill, decode []byte) (string, error) {
	conn, err := dialExecutor(addr)
	if err != nil {
		return "", err
	}
	defer closeConn(ctx, conn)
	client := pb.NewExecutorClient(conn)
	req := &pb.LoadModelRequest{ManifestJson: string(manifest), BindingJson: string(binding), PrefillGraph: prefill,
		DecodeGraph: decode, Plan: plan}
	var loadErr error
	for range 30 {
		loadCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, loadErr = client.LoadModel(loadCtx, req)
		cancel()
		if loadErr == nil || status.Code(loadErr) == codes.InvalidArgument {
			break
		}
		time.Sleep(2 * time.Second)
	}
	if loadErr != nil {
		return "", fmt.Errorf("LoadModel: %w", loadErr)
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return "", err
	}
	deadline := time.Now().Add(60 * time.Minute) // the executor loads the range's weights before it can serve
	for {
		callCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		resp, err := client.ServeRange(callCtx, &pb.ServeRangeRequest{})
		cancel()
		switch {
		case time.Now().After(deadline):
			return "", fmt.Errorf("ServeRange: still loading after an hour")
		case err == nil && resp.Port > 0 && resp.LayerRange != plan["VXPU_LAYER_RANGE"]:
			return "", fmt.Errorf("ServeRange: it serves layers %s, not %s", resp.LayerRange, plan["VXPU_LAYER_RANGE"])
		case err == nil && resp.Port > 0:
			return net.JoinHostPort(host, strconv.Itoa(int(resp.Port))), nil
		case err == nil && resp.State == pb.LoadState_LOAD_STATE_LOADING:
			// The executor is still loading its engine; keep polling.
		case err != nil:
			return "", fmt.Errorf("ServeRange: %w", err)
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
}
