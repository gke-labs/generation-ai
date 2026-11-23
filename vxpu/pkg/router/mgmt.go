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
	_ "embed"
	"html/template"
	"mime"
	"net/http"
	"sort"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

//go:embed statusui/status.html
var statusHTML string

var statusTemplate = template.Must(template.New("status").Parse(statusHTML))

// ManagementHandler returns an http.Handler that serves the router's read-only
// management interface. The root path ("/") serves the fleet status page as
// HTML, or as JSON when the client sets Accept: application/json.
func (s *Server) ManagementHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleFleetStatus)
	return mux
}

func (s *Server) handleFleetStatus(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status := s.FleetStatus(r.Context())

	if isJSONRequested(r) {
		data, err := protojson.MarshalOptions{Multiline: true, Indent: "  "}.Marshal(status)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			klog.FromContext(r.Context()).Error(err, "writing JSON fleet status response")
		}
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := statusTemplate.Execute(w, status); err != nil {
		klog.FromContext(r.Context()).Error(err, "rendering fleet status template")
	}
}

// isJSONRequested checks whether the client asks for JSON via HTTP Accept header negotiation.
func isJSONRequested(r *http.Request) bool {
	accept := r.Header.Get("Accept")
	for _, part := range strings.Split(accept, ",") {
		mediaType, _, err := mime.ParseMediaType(strings.TrimSpace(part))
		if err == nil && mediaType == "application/json" {
			return true
		}
	}
	return false
}

// FleetStatus builds a snapshot of the fleet's current status from where the
// router records it at the moment of the request. It makes no external network
// calls and acquires only a read lock on the router's recorded state.
func (s *Server) FleetStatus(ctx context.Context) *pb.RouterStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()

	status := &pb.RouterStatus{}

	// 1. Models
	knownModels := make(map[string]bool)
	for k := range s.models {
		knownModels[k] = true
	}
	for k := range s.keyModel {
		knownModels[k] = true
	}
	for k := range s.modelEndpoint {
		knownModels[k] = true
	}
	for k := range s.podEndpoint {
		knownModels[k] = true
	}

	var modelKeys []string
	for k := range knownModels {
		modelKeys = append(modelKeys, k)
	}
	sort.Strings(modelKeys)

	for _, key := range modelKeys {
		ms := s.models[key]
		ref := s.keyModel[key]

		endpoint := s.modelEndpoint[key]
		if endpoint == "" {
			endpoint = s.podEndpoint[key]
		}

		var errStr string
		if ms.err != nil {
			errStr = ms.err.Error()
		}

		status.Models = append(status.Models, &pb.ModelStatus{
			Key:      key,
			Model:    ref.Model,
			Standing: toProtoStanding(ms.standing),
			Endpoint: endpoint,
			Error:    errStr,
			Artifact: ref.Artifact,
		})
	}

	// 2. Executors
	knownExecutors := make(map[string]bool)
	for _, addr := range s.externalExecutors {
		knownExecutors[addr] = true
	}
	for ep := range s.executors {
		knownExecutors[ep] = true
	}
	for _, ep := range s.modelEndpoint {
		if ep != "" {
			knownExecutors[ep] = true
		}
	}
	for ep := range s.podModel {
		if ep != "" {
			knownExecutors[ep] = true
		}
	}
	for ep := range s.expertHolders {
		if ep != "" {
			knownExecutors[ep] = true
		}
	}

	var execAddrs []string
	for ep := range knownExecutors {
		execAddrs = append(execAddrs, ep)
	}
	sort.Strings(execAddrs)

	for _, ep := range execAddrs {
		info, hasInfo := s.executors[ep]
		machineKind := ""
		if hasInfo && info.Machine != nil {
			machineKind = info.Machine.Name
		}

		var loaded []string
		if mKey, ok := s.externalModel[ep]; ok && mKey != "" {
			loaded = s.appendModelName(loaded, mKey)
		}
		if mKey, ok := s.podModel[ep]; ok && mKey != "" {
			loaded = s.appendModelName(loaded, mKey)
		}
		if engine, ok := s.expertHolders[ep]; ok && engine != "" {
			if mKey, ok := s.externalModel[engine]; ok && mKey != "" {
				loaded = s.appendModelName(loaded, mKey)
			}
		}
		sort.Strings(loaded)

		status.Executors = append(status.Executors, &pb.ExecutorStatus{
			Endpoint:     ep,
			MachineKind:  machineKind,
			LoadedModels: loaded,
			External:     s.isExternalLocked(ep),
		})
	}

	return status
}

func (s *Server) appendModelName(loaded []string, modelKey string) []string {
	name := modelKey
	if ref, ok := s.keyModel[modelKey]; ok && ref.Model != "" {
		name = ref.Model
	}
	for _, existing := range loaded {
		if existing == name {
			return loaded
		}
	}
	return append(loaded, name)
}

func toProtoStanding(s modelStanding) pb.ModelStanding {
	switch s {
	case standingPassed:
		return pb.ModelStanding_MODEL_STANDING_PASSED
	case standingRefused:
		return pb.ModelStanding_MODEL_STANDING_REFUSED
	default:
		return pb.ModelStanding_MODEL_STANDING_UNCHECKED
	}
}
