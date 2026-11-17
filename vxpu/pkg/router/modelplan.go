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
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// A model's plan is the switches the coordinator sends with every load
// of the model (LoadModelRequest.plan, #394): RegisterModel sets it, and
// it is kept in modelstore so it outlives the router.
const planArtifactType = "application/vnd.gke-labs.vxpu.plan.v1+json"

func planRepository(model string) string { return "vxpu-plans/" + statsKey(model) }

// putPlan replaces model's plan.
func (s *Server) putPlan(ctx context.Context, model string, plan map[string]string) error {
	data, err := json.Marshal(plan)
	if err != nil {
		return err
	}
	name := planRepository(model)
	digest, err := s.weightStore.PutBlob(ctx, name, data)
	if err != nil {
		return err
	}
	return s.weightStore.PutManifest(ctx, name, "latest", &weightstore.Manifest{
		SchemaVersion: 2, MediaType: "application/vnd.oci.image.manifest.v1+json",
		ArtifactType: planArtifactType, Config: weightstore.EmptyConfig,
		Layers: []weightstore.Layer{{MediaType: weightstore.MediaTypeFile, Digest: digest, Size: int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": "plan.json"}}},
		Annotations: map[string]string{"vnd.gke-labs.vxpu.model": model},
	})
}

// getPlan returns model's plan; none was set is an empty plan.
func (s *Server) getPlan(ctx context.Context, model string) (map[string]string, error) {
	data, err := s.weightStore.FetchFile(ctx, planRepository(model), "latest", "plan.json")
	if errors.Is(err, weightstore.ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var plan map[string]string
	if err := json.Unmarshal(data, &plan); err != nil {
		return nil, fmt.Errorf("plan of %s: %w", model, err)
	}
	return plan, nil
}

// hintsKey is a plan as one comparable string (encoding/json sorts the
// keys); no plan is "".
func hintsKey(plan map[string]string) string {
	if len(plan) == 0 {
		return ""
	}
	data, err := json.Marshal(plan)
	if err != nil { // a map of strings always encodes
		return fmt.Sprint(plan)
	}
	return string(data)
}

// setHints records model's registered plan: always when it was just
// stored, otherwise (read for a load) only if none is known, so a load
// that read the plan before a newer one was stored cannot overwrite it.
func (s *Server) setHints(model string, plan map[string]string, stored bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hints == nil {
		s.hints = map[string]string{}
	}
	if _, known := s.hints[statsKey(model)]; stored || !known {
		s.hints[statsKey(model)] = hintsKey(plan)
	}
}

// planSwitches names a plan's switches, for logs: a value can be a whole
// inline placement.
func planSwitches(plan map[string]string) []string {
	names := make([]string, 0, len(plan))
	for name := range plan {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// modelPlan is the plan the coordinator chose for a model's last load,
// and why (ExplainModel).
type modelPlan struct {
	Model string `json:"model"`
	// the executor's, as it described itself
	Machine string `json:"machine,omitempty"`
	Time    string `json:"time"`
	// what was sent
	Switches map[string]string `json:"switches"`
	// the registered plan's switches, which win
	Hints map[string]string `json:"hints,omitempty"`
	// the layers other executors serve to the engine's mailboxes, by
	// where they answer
	ExpertExecutors map[string][]int `json:"expert_executors,omitempty"`
	// the engine's mailboxes for nodes other executors serve:
	// name=tcp://host:port/index
	RemoteNodes []string `json:"remote_nodes,omitempty"`
	// RemoteRanges lists the ranges of whole layers that other executors run
	// after the engine's own range (#387), each as START:END=tcp://host:port.
	RemoteRanges []string `json:"remote_ranges,omitempty"`
	// Candidates lists the layouts the planner priced, cheapest first. vxpu
	// explain shows them.
	Candidates []candidateSummary `json:"candidates,omitempty"`
	Reasons    []string           `json:"reasons"`
	// the decode time the cost model predicts for the plan (before the
	// registered plan's switches; telemetry measures it)
	Prediction planner.Prediction `json:"prediction"`
}

// planLoad chooses the switches a load of model sends: the planner's,
// from the artifact, the executor's machine and the model's recorded
// selections, with the registered plan's switches (hints) over them.
func (s *Server) planLoad(ctx context.Context, model string, manifest, binding, prefill, decode []byte, hints map[string]string,
	info *executorInfo) map[string]string {
	log := klog.FromContext(ctx)
	var probe *machines.Probe
	mp := modelPlan{Model: model, Time: time.Now().UTC().Format(time.RFC3339), Hints: hints, Switches: map[string]string{}}
	if info != nil {
		probe = &info.Probe
		if info.Machine != nil {
			mp.Machine = info.Machine.Name
		}
	}
	if s.stats != nil { // the requests still buffered count too
		if err := s.stats.flush(ctx, model); err != nil {
			log.Error(err, "flushing statistics before planning", "model", model)
		}
	}
	artifact, err := planner.ReadArtifact(manifest, binding)
	if err != nil {
		mp.Reasons = append(mp.Reasons, fmt.Sprintf("planner: could not read the artifact (%v); the registered plan only", err))
	} else {
		counts, err := s.expertCounts(ctx, model)
		if err != nil {
			mp.Reasons = append(mp.Reasons, fmt.Sprintf("planner: could not read the statistics (%v)", err))
		}
		engine := ""
		if info != nil {
			engine = info.Endpoint
		}
		s.mu.Lock()
		s.releaseHelpersLocked(engine) // its experts are placed again
		s.mu.Unlock()
		free := s.freeExecutors(ctx, engine)
		if info != nil && len(free) > 0 && len(prefill) > 0 { // ask where the step can be cut into ranges of whole layers
			if err := s.readCuts(ctx, info, &artifact, manifest, binding, prefill, decode); err != nil {
				mp.Reasons = append(mp.Reasons, fmt.Sprintf("planner: where the step can be cut is not known (%v): no layer ranges", err))
				artifact.Cuts = []int{} // no cuts: the planner starts a range only where it knows the cut is safe
			}
		}
		ranges, loaded := s.chooseRanges(ctx, model, artifact, info, free, counts, hints, manifest, binding, prefill, decode, &mp)
		if ranges != nil {
			for k, v := range hints { // the registered plan's switches override the planner's, as below
				ranges[k] = v
			}
			s.mu.Lock()
			if s.plans == nil {
				s.plans = map[string]modelPlan{}
			}
			s.plans[statsKey(model)] = mp
			s.mu.Unlock()
			log.Info("Planned the load", "model", model, "machine", mp.Machine, "layout", "layer-ranges",
				"predictedMsPerToken", mp.Prediction.MsPerToken, "ranges", mp.RemoteRanges, "reasons", mp.Reasons)
			return ranges
		}
		free = slices.DeleteFunc(free, func(e planner.Executor) bool { return slices.Contains(loaded, e.Address) }) // these executors now hold ranges
		p := planner.Choose(artifact, probe, free, counts, hints)
		mp.Switches, mp.Reasons, mp.Prediction = p.Switches, append(mp.Reasons, p.Reasons...), p.Prediction
		// the other executors serve the experts nodes of their layers, on a
		// port each; the engine's mailboxes for those nodes are wired to them
		if len(p.ExpertExecutors) > 0 {
			planned := sortedKeys(p.ExpertExecutors) // serveNodesOf answers in this order
			remote, err := s.serveNodesOf(ctx, model, info, manifest, binding, decode, p.ExpertExecutors)
			if err != nil {
				mp.Reasons = append(mp.Reasons, fmt.Sprintf("experts: their nodes could not be placed (%v): the engine holds them", err))
			}
			served := map[string][]int{}
			for i, r := range remote {
				mp.Reasons = append(mp.Reasons, r.reason)
				if r.err != nil {
					continue
				}
				served[planned[i]] = r.layers
				if mp.ExpertExecutors == nil {
					mp.ExpertExecutors = map[string][]int{}
				}
				mp.ExpertExecutors[r.address] = r.layers
				mp.RemoteNodes = append(mp.RemoteNodes, r.entries...)
			}
			if len(mp.RemoteNodes) > 0 {
				mp.Switches["VXPU_REMOTE_NODES"] = strings.Join(mp.RemoteNodes, ",")
			}
			if len(served) < len(planned) { // the engine holds what an executor could not serve: price that
				p.ExpertExecutors = served
				others := map[string]machines.Probe{}
				for _, e := range free {
					others[e.Address] = e.Probe
				}
				mp.Prediction = planner.Predict(artifact, p, probe, others, counts)
			}
		}
	}
	for k, v := range hints {
		if _, chosen := mp.Switches[k]; chosen {
			mp.Reasons = append(mp.Reasons, fmt.Sprintf("%s: the registered plan's value, over the planner's", k))
		}
		mp.Switches[k] = v
	}
	s.mu.Lock()
	if s.plans == nil {
		s.plans = map[string]modelPlan{}
	}
	s.plans[statsKey(model)] = mp
	s.mu.Unlock()
	log.Info("Planned the load", "model", model, "machine", mp.Machine, "switches", len(mp.Switches),
		"predictedMsPerToken", mp.Prediction.MsPerToken, "reasons", mp.Reasons)
	return mp.Switches
}

// ExplainModel returns the plan the coordinator chose for a model's last
// load, and why.
func (s *Server) ExplainModel(ctx context.Context, req *pb.ExplainModelRequest) (*pb.ExplainModelResponse, error) {
	s.mu.RLock()
	mp, ok := s.plans[statsKey(req.Model)]
	s.mu.RUnlock()
	if !ok {
		return nil, status.Errorf(codes.NotFound, "%s has not been loaded by name since the router started", req.Model)
	}
	data, err := json.MarshalIndent(mp, "", "  ")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}
	return &pb.ExplainModelResponse{PlanJson: string(data)}, nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// routerOwnedSwitches are the plan switches only the router sets: which
// node servers an engine trusts is the router's to wire (the executors it
// asked to serve them), never a client's to name.
var routerOwnedSwitches = []string{"VXPU_REMOTE_NODES", "VXPU_REMOTE_RANGES", "VXPU_LAYER_RANGE"}
