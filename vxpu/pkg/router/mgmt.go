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
	"bytes"
	_ "embed"
	stderrors "errors"
	"fmt"
	"html/template"
	"maps"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

//go:embed mgmtui/index.html
var mgmtIndexHTML string

var mgmtIndexTemplate = template.Must(template.New("mgmt-index").Funcs(template.FuncMap{
	"formatPhase": func(p pb.InstancePhase) string {
		switch p {
		case pb.InstancePhase_INSTANCE_PHASE_NOT_LOADED:
			return "not loaded"
		case pb.InstancePhase_INSTANCE_PHASE_LOADING:
			return "loading"
		case pb.InstancePhase_INSTANCE_PHASE_CHECKING:
			return "checking"
		case pb.InstancePhase_INSTANCE_PHASE_PASSED:
			return "passed"
		case pb.InstancePhase_INSTANCE_PHASE_REFUSED:
			return "refused"
		case pb.InstancePhase_INSTANCE_PHASE_FAILED:
			return "failed"
		default:
			return "unknown"
		}
	},
	"phaseClass": func(p pb.InstancePhase) string {
		switch p {
		case pb.InstancePhase_INSTANCE_PHASE_PASSED:
			return "phase-passed"
		case pb.InstancePhase_INSTANCE_PHASE_LOADING, pb.InstancePhase_INSTANCE_PHASE_CHECKING:
			return "phase-active"
		case pb.InstancePhase_INSTANCE_PHASE_REFUSED:
			return "phase-refused"
		case pb.InstancePhase_INSTANCE_PHASE_FAILED:
			return "phase-failed"
		default:
			return "phase-idle"
		}
	},
	"shortHash": func(h string) string {
		if len(h) > 12 {
			return h[:12]
		}
		return h
	},
	"pathEscape": url.PathEscape,
}).Parse(mgmtIndexHTML))

var (
	errInstanceNotFound = stderrors.New("instance not found")
	errPlanUnavailable  = stderrors.New("instance has no execution plan")
)

// ManagementHandler returns a read-only HTTP handler for the router management UI.
// It serves the fleet status landing page at GET / and each instance's EXPLAIN
// plan tree at GET /instances/{key}/plan. Both URLs return HTML by default and
// protobuf JSON when requested with Accept: application/json.
func (s *Server) ManagementHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleMgmtIndex)
	mux.HandleFunc("GET /instances/{key}/plan", s.handleMgmtInstancePlan)
	return mux
}

func (s *Server) handleMgmtIndex(w http.ResponseWriter, r *http.Request) {
	st := s.RouterStatus()
	if acceptsJSON(r) {
		writeProtoJSON(w, r, st)
		return
	}
	var buf bytes.Buffer
	if err := mgmtIndexTemplate.Execute(&buf, st); err != nil {
		klog.FromContext(r.Context()).Error(err, "Failed to render management index page")
		http.Error(w, "failed to render management page", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if _, err := w.Write(buf.Bytes()); err != nil {
		klog.FromContext(r.Context()).V(2).Info("Failed to write management index response", "error", err)
	}
}

func (s *Server) handleMgmtInstancePlan(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")
	tree, err := s.InstanceModelTree(key)
	if err != nil {
		if stderrors.Is(err, errInstanceNotFound) || stderrors.Is(err, errPlanUnavailable) {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		klog.FromContext(r.Context()).Error(err, "Failed to build instance plan tree", "key", key)
		http.Error(w, fmt.Sprintf("failed to build instance plan tree: %v", err), http.StatusInternalServerError)
		return
	}
	if acceptsJSON(r) {
		writeProtoJSON(w, r, tree)
		return
	}
	htmlBytes, err := kernelplan.RenderTreeHTML(tree)
	if err != nil {
		klog.FromContext(r.Context()).Error(err, "Failed to render instance plan tree HTML", "key", key)
		http.Error(w, "failed to render plan tree", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if _, err := w.Write(htmlBytes); err != nil {
		klog.FromContext(r.Context()).V(2).Info("Failed to write plan tree response", "key", key, "error", err)
	}
}

// serverMgmtSnapshot holds point-in-time copies of the Server maps needed to
// render the management landing page without holding Server.mu.
type serverMgmtSnapshot struct {
	// registered holds the latest model reference recorded by RegisterModel per model name.
	registered map[string]modelRef
	// keyModel maps each model key to its named model reference.
	keyModel map[string]modelRef
	// executors maps each executor runtime endpoint to its probed hardware and catalog machine.
	executors map[string]executorInfo
	// extModels records verification standing for external executor models.
	extModels map[string]modelState
	// extEndpoints maps each external model key to its executor endpoint.
	extEndpoints map[string]string
}

func (s *Server) snapshotMgmtState() serverMgmtSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return serverMgmtSnapshot{
		registered:   maps.Clone(s.registeredModels),
		keyModel:     maps.Clone(s.keyModel),
		executors:    maps.Clone(s.executors),
		extModels:    maps.Clone(s.models),
		extEndpoints: maps.Clone(s.modelEndpoint),
	}
}

func (snap *serverMgmtSnapshot) machineKind(endpoint string) string {
	if endpoint == "" {
		return ""
	}
	info, ok := snap.executors[endpoint]
	if !ok {
		return ""
	}
	mk, err := executorMachineKind(&info)
	if err != nil {
		return ""
	}
	return mk
}

// RouterStatus returns a point-in-time read-only snapshot of registered and active
// models alongside each physical instance's status. It reads strictly from
// in-memory state and never holds locks across network or Kubernetes API calls.
func (s *Server) RouterStatus() *pb.RouterStatus {
	var reconcilerSnaps []InstanceSnapshot
	if s.reconciler != nil {
		reconcilerSnaps = s.reconciler.Snapshot()
	}
	srvSnap := s.snapshotMgmtState()
	instances := buildFleetInstances(reconcilerSnaps, &srvSnap)
	models := buildFleetModels(&srvSnap, instances)
	return &pb.RouterStatus{
		Models:    models,
		Instances: instances,
	}
}

func reconcilerFleetInstance(snap InstanceSnapshot, srv *serverMgmtSnapshot) *pb.FleetInstance {
	model := snap.Model
	artifact := ""
	if ref, ok := srv.keyModel[snap.Key]; ok {
		if model == "" {
			model = ref.Model
		}
		artifact = ref.Artifact
	}
	var lastErr string
	if snap.Status.LastError != nil {
		lastErr = snap.Status.LastError.Error()
	}
	var nextAttempt string
	if !snap.Status.NextAttempt.IsZero() {
		nextAttempt = snap.Status.NextAttempt.UTC().Format(time.RFC3339)
	}
	return &pb.FleetInstance{
		Key:                snap.Key,
		Model:              model,
		Artifact:           artifact,
		Phase:              instancePhaseProto(snap.Status.Phase),
		Executor:           snap.Endpoint,
		MachineKind:        srv.machineKind(snap.Endpoint),
		Generation:         snap.Generation,
		ObservedGeneration: snap.Status.ObservedGeneration,
		PlanHash:           snap.Status.PlanHash,
		HasPlan:            snap.HasPlan,
		LastError:          lastErr,
		NextAttemptRfc3339: nextAttempt,
	}
}

func buildFleetInstances(reconcilerSnaps []InstanceSnapshot, srv *serverMgmtSnapshot) []*pb.FleetInstance {
	instances := make([]*pb.FleetInstance, 0, len(reconcilerSnaps))
	seen := make(map[string]bool, len(reconcilerSnaps))
	for _, snap := range reconcilerSnaps {
		seen[snap.Key] = true
		instances = append(instances, reconcilerFleetInstance(snap, srv))
	}

	// Include external executor instances that do not yet run through the reconciler.
	extKeys := make(map[string]bool, len(srv.extModels)+len(srv.extEndpoints))
	for k := range srv.extModels {
		if !seen[k] {
			extKeys[k] = true
		}
	}
	for k := range srv.extEndpoints {
		if !seen[k] {
			extKeys[k] = true
		}
	}
	for key := range extKeys {
		ms := srv.extModels[key]
		ep := srv.extEndpoints[key]
		ref := srv.keyModel[key]
		var lastErr string
		if ms.err != nil {
			lastErr = ms.err.Error()
		}
		instances = append(instances, &pb.FleetInstance{
			Key:         key,
			Model:       ref.Model,
			Artifact:    ref.Artifact,
			Phase:       standingPhaseProto(ms.standing),
			Executor:    ep,
			MachineKind: srv.machineKind(ep),
			LastError:   lastErr,
		})
	}

	sort.Slice(instances, func(i, j int) bool {
		if instances[i].Model != instances[j].Model {
			return instances[i].Model < instances[j].Model
		}
		return instances[i].Key < instances[j].Key
	})
	return instances
}

func buildFleetModels(srv *serverMgmtSnapshot, instances []*pb.FleetInstance) []*pb.FleetModel {
	type modelIdentity struct {
		model    string
		artifact string
	}
	modelSet := make(map[modelIdentity]*pb.FleetModel)
	addModel := func(model, commit, artifact string) {
		if model == "" {
			return
		}
		id := modelIdentity{model: model, artifact: artifact}
		existing, ok := modelSet[id]
		if !ok {
			modelSet[id] = &pb.FleetModel{
				Model:    model,
				Commit:   commit,
				Artifact: artifact,
			}
			return
		}
		if existing.Commit == "" && commit != "" {
			existing.Commit = commit
		}
	}
	for _, ref := range srv.registered {
		addModel(ref.Model, ref.Commit, ref.Artifact)
	}
	for _, ref := range srv.keyModel {
		addModel(ref.Model, ref.Commit, ref.Artifact)
	}
	// Also list active models loaded by artifact that have no catalog registration.
	activeModelNames := make(map[string]bool, len(modelSet))
	for id := range modelSet {
		activeModelNames[id.model] = true
	}
	for _, inst := range instances {
		if inst.Model != "" && !activeModelNames[inst.Model] {
			addModel(inst.Model, "", inst.Artifact)
			activeModelNames[inst.Model] = true
		}
	}
	models := make([]*pb.FleetModel, 0, len(modelSet))
	for _, m := range modelSet {
		models = append(models, m)
	}
	sort.Slice(models, func(i, j int) bool {
		if models[i].Model != models[j].Model {
			return models[i].Model < models[j].Model
		}
		if models[i].Artifact != models[j].Artifact {
			return models[i].Artifact < models[j].Artifact
		}
		return models[i].Commit < models[j].Commit
	})
	return models
}

// InstanceModelTree builds the EXPLAIN ModelTree for one physical instance
// tracked by the reconciler without performing network or Kubernetes API calls.
func (s *Server) InstanceModelTree(key string) (*planpb.ModelTree, error) {
	if s.reconciler == nil {
		return nil, errInstanceNotFound
	}
	data, ok := s.reconciler.InstancePlan(key)
	if !ok {
		return nil, errInstanceNotFound
	}
	if data.Plan == nil {
		return nil, errPlanUnavailable
	}

	modelName := data.Model
	if modelName == "" {
		s.mu.RLock()
		modelName = s.keyModel[key].Model
		s.mu.RUnlock()
	}
	if modelName == "" {
		modelName = key
	}

	var segments []*planpb.Segment
	if len(data.DecodeGraph) > 0 {
		var err error
		segments, err = kernelplan.ReadSegmentsFromBytes(data.DecodeGraph, []byte(data.ManifestJSON), []byte(data.BindingJSON))
		if err != nil {
			return nil, fmt.Errorf("read decode graph segments: %w", err)
		}
	} else {
		for _, seg := range data.Plan.GetSegments() {
			if seg != nil {
				segments = append(segments, proto.Clone(seg).(*planpb.Segment))
			}
		}
	}

	tree := kernelplan.BuildModelTree(modelName, segments)
	if err := kernelplan.AttachPlan(tree, "", data.Plan, nil); err != nil {
		return nil, fmt.Errorf("attach plan: %w", err)
	}
	return tree, nil
}

func instancePhaseProto(p InstancePhase) pb.InstancePhase {
	switch p {
	case InstancePhaseNotLoaded:
		return pb.InstancePhase_INSTANCE_PHASE_NOT_LOADED
	case InstancePhaseLoading:
		return pb.InstancePhase_INSTANCE_PHASE_LOADING
	case InstancePhaseChecking:
		return pb.InstancePhase_INSTANCE_PHASE_CHECKING
	case InstancePhasePassed:
		return pb.InstancePhase_INSTANCE_PHASE_PASSED
	case InstancePhaseRefused:
		return pb.InstancePhase_INSTANCE_PHASE_REFUSED
	case InstancePhaseFailed:
		return pb.InstancePhase_INSTANCE_PHASE_FAILED
	default:
		return pb.InstancePhase_INSTANCE_PHASE_UNSPECIFIED
	}
}

func standingPhaseProto(st modelStanding) pb.InstancePhase {
	switch st {
	case standingPassed:
		return pb.InstancePhase_INSTANCE_PHASE_PASSED
	case standingRefused:
		return pb.InstancePhase_INSTANCE_PHASE_REFUSED
	default:
		return pb.InstancePhase_INSTANCE_PHASE_NOT_LOADED
	}
}

func acceptsJSON(r *http.Request) bool {
	for _, part := range strings.Split(r.Header.Get("Accept"), ",") {
		mediaType, _, _ := strings.Cut(part, ";")
		switch strings.ToLower(strings.TrimSpace(mediaType)) {
		case "text/html":
			return false
		case "application/json":
			return true
		}
	}
	return false
}

func writeProtoJSON(w http.ResponseWriter, r *http.Request, msg proto.Message) {
	b, err := protojson.MarshalOptions{
		Multiline:     true,
		Indent:        "  ",
		UseProtoNames: true,
	}.Marshal(msg)
	if err != nil {
		klog.FromContext(r.Context()).Error(err, "Failed to marshal management JSON response")
		http.Error(w, "failed to marshal JSON", http.StatusInternalServerError)
		return
	}
	b = append(b, '\n')
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if _, err := w.Write(b); err != nil {
		klog.FromContext(r.Context()).V(2).Info("Failed to write management JSON response", "error", err)
	}
}
