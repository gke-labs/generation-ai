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
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

// cmdTrace routes `vxpu trace ...` subcommands.
func cmdTrace(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu trace summarize [flags] FILE")
	}
	switch args[0] {
	case "summarize":
		return runTraceSummarize(ctx, args[1:])
	default:
		return fmt.Errorf("vxpu trace: unknown subcommand %q (summarize)", args[0])
	}
}

func runTraceSummarize(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("trace summarize", flag.ContinueOnError)
	top := flags.Int("top", 5, "number of slowest steps to show")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu trace summarize [flags] FILE")
	}
	path := flags.Arg(0)
	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return err
	}
	return summarizeTrace(data, os.Stdout, *top)
}

// stepTrace represents one JSON line written by the native decode loop in node_server.cpp.
type stepTrace struct {
	Step            int64      `json:"step"`
	Ms              float64    `json:"ms"`
	Launch0Ms       float64    `json:"launch0_ms"`
	LaunchedMs      float64    `json:"launched_ms"`
	SyncedMs        []float64  `json:"synced_ms"`
	DraftLaunch0Ms  *float64   `json:"draft_launch0_ms,omitempty"`
	DraftLaunchedMs *float64   `json:"draft_launched_ms,omitempty"`
	DraftCheckedMs  *float64   `json:"draft_checked_ms,omitempty"`
	DraftHost       []hostNode `json:"draft_host,omitempty"`
	Host            []hostNode `json:"host,omitempty"`
}

type hostNode struct {
	StartMs float64 `json:"start_ms"`
	EndMs   float64 `json:"end_ms"`
}

func (h hostNode) duration() float64 {
	return math.Max(0, h.EndMs-h.StartMs)
}

type phaseValue struct {
	name     string
	duration float64
}

func appendHostPhases(phases []phaseValue, format string, nodes []hostNode) []phaseValue {
	for i, h := range nodes {
		phases = append(phases, phaseValue{
			name:     fmt.Sprintf(format, i),
			duration: h.duration(),
		})
	}
	return phases
}

func stepPhases(s *stepTrace) []phaseValue {
	phases := []phaseValue{{name: "wall", duration: s.Ms}}
	if s.DraftLaunch0Ms != nil && s.DraftLaunchedMs != nil {
		phases = append(phases, phaseValue{name: "draft launch", duration: math.Max(0, *s.DraftLaunchedMs-*s.DraftLaunch0Ms)})
	}
	phases = appendHostPhases(phases, "draft host node %d", s.DraftHost)
	phases = append(phases, phaseValue{name: "launch", duration: math.Max(0, s.LaunchedMs-s.Launch0Ms)})
	for g, syncMs := range s.SyncedMs {
		prev := s.LaunchedMs
		if g > 0 {
			prev = s.SyncedMs[g-1]
		}
		phases = append(phases, phaseValue{name: fmt.Sprintf("gpu %d", g), duration: math.Max(0, syncMs-prev)})
	}
	if s.DraftCheckedMs != nil && len(s.SyncedMs) > 0 {
		phases = append(phases, phaseValue{name: "draft check", duration: math.Max(0, *s.DraftCheckedMs-s.SyncedMs[len(s.SyncedMs)-1])})
	}
	return appendHostPhases(phases, "host node %d", s.Host)
}

type phaseStats struct {
	name   string
	p10    float64
	median float64
	p90    float64
	max    float64
}

func readTrace(data []byte) ([]stepTrace, error) {
	lines := bytes.Split(data, []byte("\n"))
	var steps []stepTrace
	for i, raw := range lines {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}
		var step stepTrace
		if err := json.Unmarshal(line, &step); err != nil {
			return nil, fmt.Errorf("line %d: %w", i+1, err)
		}
		steps = append(steps, step)
	}
	if len(steps) == 0 {
		return nil, fmt.Errorf("trace contains no steps")
	}
	return steps, nil
}

func computePhaseStats(steps []stepTrace) []phaseStats {
	var phaseOrder []string
	samples := make(map[string][]float64)
	for i := range steps {
		for _, p := range stepPhases(&steps[i]) {
			if _, exists := samples[p.name]; !exists {
				phaseOrder = append(phaseOrder, p.name)
			}
			samples[p.name] = append(samples[p.name], p.duration)
		}
	}
	stats := make([]phaseStats, 0, len(phaseOrder))
	for _, name := range phaseOrder {
		s := append([]float64(nil), samples[name]...)
		sort.Float64s(s)
		stats = append(stats, phaseStats{
			name:   name,
			p10:    tracePercentile(s, 0.10),
			median: tracePercentile(s, 0.50),
			p90:    tracePercentile(s, 0.90),
			max:    s[len(s)-1],
		})
	}
	return stats
}

func printPhaseStats(out io.Writer, numSteps int, stats []phaseStats) error {
	if _, err := fmt.Fprintf(out, "%d steps:\n\n", numSteps); err != nil {
		return err
	}
	w := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	if _, err := fmt.Fprintln(w, "PHASE\tP10 (ms)\tMEDIAN (ms)\tP90 (ms)\tMAX (ms)"); err != nil {
		return err
	}
	for _, s := range stats {
		if _, err := fmt.Fprintf(w, "%s\t%.2f\t%.2f\t%.2f\t%.2f\n", s.name, s.p10, s.median, s.p90, s.max); err != nil {
			return err
		}
	}
	return w.Flush()
}

func printSlowestSteps(out io.Writer, steps []stepTrace, top int) error {
	if top <= 0 {
		return nil
	}
	sorted := append([]stepTrace(nil), steps...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Ms != sorted[j].Ms {
			return sorted[i].Ms > sorted[j].Ms
		}
		return sorted[i].Step < sorted[j].Step
	})
	if top > len(sorted) {
		top = len(sorted)
	}

	if _, err := fmt.Fprintf(out, "\nslowest steps:\n"); err != nil {
		return err
	}
	for i := 0; i < top; i++ {
		st := sorted[i]
		var subPhases []phaseValue
		for _, p := range stepPhases(&st) {
			if p.name != "wall" {
				subPhases = append(subPhases, p)
			}
		}
		sort.SliceStable(subPhases, func(a, b int) bool {
			return subPhases[a].duration > subPhases[b].duration
		})
		var parts []string
		for _, p := range subPhases {
			parts = append(parts, fmt.Sprintf("%s: %.2f ms", p.name, p.duration))
		}
		detail := ""
		if len(parts) > 0 {
			detail = fmt.Sprintf(" (%s)", strings.Join(parts, ", "))
		}
		if _, err := fmt.Fprintf(out, "  step %d: wall %.2f ms%s\n", st.Step, st.Ms, detail); err != nil {
			return err
		}
	}
	return nil
}

// summarizeTrace parses per-step host trace data and prints statistics to out.
func summarizeTrace(data []byte, out io.Writer, top int) error {
	steps, err := readTrace(data)
	if err != nil {
		return err
	}
	stats := computePhaseStats(steps)
	if err := printPhaseStats(out, len(steps), stats); err != nil {
		return err
	}
	return printSlowestSteps(out, steps, top)
}

func tracePercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Min(float64(len(sorted)-1), math.Floor(p*float64(len(sorted)))))
	return sorted[idx]
}
