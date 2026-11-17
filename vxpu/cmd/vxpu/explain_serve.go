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

package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"strconv"
	"time"

	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

//go:embed searchui
var searchUI embed.FS

// serveSearch shows the planner searching layouts live (vxpu explain
// --serve): a page that draws the layout being scanned and the best found
// so far, and a stream of the search's progress (server-sent events).
// Every page load starts its own search; nothing runs on any machine.
func serveSearch(ctx context.Context, addr, model string, a planner.Artifact, fleet []planner.Node, counts map[int]map[int]int64,
	cands []planner.Candidate) error {
	log := klog.FromContext(ctx)
	ui, err := fs.Sub(searchUI, "searchui")
	if err != nil {
		return err
	}
	if len(a.Experts) == 0 {
		return fmt.Errorf("the model has no layers of experts to lay out")
	}
	scorer := planner.NewScorer(a, fleet, counts)
	type machineInfo struct {
		Name    string  `json:"name"`
		Machine string  `json:"machine"`
		GPU     string  `json:"gpu,omitempty"`
		RAMGB   float64 `json:"ram_gb"`
	}
	type candidateInfo struct {
		Kind    string  `json:"kind"`
		Summary string  `json:"summary"`
		Ms      float64 `json:"ms_per_token"`
	}
	meta := struct {
		Model      string          `json:"model"`
		Machines   []machineInfo   `json:"machines"`
		Layers     int             `json:"layers"`
		Engram     []int           `json:"engram"`
		Candidates []candidateInfo `json:"candidates"`
	}{Model: model, Layers: scorer.Layers(), Engram: a.Engram}
	for _, n := range fleet {
		mi := machineInfo{Name: n.Name, Machine: n.Name, RAMGB: float64(n.Probe.RamBytes) / 1e9}
		if m := machines.Match(n.Probe); m != nil {
			mi.Machine = m.Name
		}
		if n.Probe.Gpu != "" {
			mi.GPU = fmt.Sprintf("%dx %s", n.Probe.GpuCount, n.Probe.Gpu)
		}
		meta.Machines = append(meta.Machines, mi)
	}
	for _, c := range cands {
		if c.Infeasible == "" {
			meta.Candidates = append(meta.Candidates, candidateInfo{c.Kind, c.Summary, c.Prediction.MsPerToken})
		}
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServerFS(ui))
	mux.HandleFunc("/meta", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(meta); err != nil {
			log.Error(err, "writing /meta")
		}
	})
	mux.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		// ?steps=N layouts scored per frame (the one shown is the last), ?ms=N between frames,
		// ?start=best to perturb the best candidate rather than start from random layouts
		steps, frame := min(queryInt(r, "steps", 40), 10000), time.Duration(queryInt(r, "ms", 80))*time.Millisecond
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		sr := planner.NewSearch(scorer, 10, time.Now().UnixNano())
		if r.URL.Query().Get("start") == "best" && len(cands) > 0 && cands[0].Infeasible == "" {
			sr.StartFrom(scorer.LayoutOf(cands[0]))
		}
		start := time.Now()
		best := 0.0
		ticker := time.NewTicker(frame)
		defer ticker.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-ticker.C:
			}
			var probe planner.Scored
			for range steps {
				probe = sr.Step()
			}
			ev := map[string]any{"probe": probe, "probe_text": scorer.Describe(probe.Layout), "scanned": sr.Scanned,
				"feasible": sr.Valid, "elapsed_s": time.Since(start).Seconds(), "top": sr.Top}
			if len(sr.Top) > 0 && (best == 0 || sr.Top[0].Ms < best) {
				best = sr.Top[0].Ms
				ev["new_best"] = true
				ev["best_text"] = scorer.Describe(sr.Top[0].Layout)
			}
			data, err := json.Marshal(ev)
			if err != nil {
				log.Error(err, "encoding a search event")
				return
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return // the page went away
			}
			flusher.Flush()
		}
	})
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Info("Showing the plan search", "url", "http://"+displayAddr(addr)+"/", "model", model, "machines", len(fleet))
	return srv.ListenAndServe()
}

func queryInt(r *http.Request, name string, def int) int {
	if v, err := strconv.Atoi(r.URL.Query().Get(name)); err == nil && v > 0 {
		return v
	}
	return def
}

func displayAddr(addr string) string {
	if len(addr) > 0 && addr[0] == ':' {
		return "localhost" + addr
	}
	return addr
}
