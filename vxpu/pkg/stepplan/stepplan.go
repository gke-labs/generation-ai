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

// Package stepplan computes wait dependencies and stream assignments for
// segment nodes in a decode step graph.
package stepplan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"text/tabwriter"

	planv1alpha1 "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/encoding/protojson"
)

type (
	StepGraphDescription = planv1alpha1.StepGraphDescription
	StepNode             = planv1alpha1.StepNode
	HostMapping          = planv1alpha1.HostMapping
	StepPlan             = planv1alpha1.StepPlan
	NodePlan             = planv1alpha1.NodePlan
	Stream               = planv1alpha1.Stream
	Execution            = planv1alpha1.Execution
	Answer               = planv1alpha1.Answer
	HostCall             = planv1alpha1.HostCall
)

const (
	StreamUnspecified = planv1alpha1.Stream_STREAM_UNSPECIFIED
	StreamMain        = planv1alpha1.Stream_STREAM_MAIN
	StreamParallel    = planv1alpha1.Stream_STREAM_PARALLEL

	ExecutionUnspecified = planv1alpha1.Execution_EXECUTION_UNSPECIFIED
	ExecutionGraph       = planv1alpha1.Execution_EXECUTION_GRAPH
	ExecutionExchange    = planv1alpha1.Execution_EXECUTION_EXCHANGE

	AnswerUnspecified = planv1alpha1.Answer_ANSWER_UNSPECIFIED
	AnswerCPU         = planv1alpha1.Answer_ANSWER_CPU
	AnswerGPUPool     = planv1alpha1.Answer_ANSWER_GPU_POOL
	AnswerPeerGPU     = planv1alpha1.Answer_ANSWER_PEER_GPU
	AnswerRemote      = planv1alpha1.Answer_ANSWER_REMOTE
)

// Options configures step planning.
type Options struct {
	// Serialize puts every node on the main stream in call order.
	Serialize bool
	// NoParallel keeps the computed wait dependencies and uses only the main stream.
	NoParallel bool
}

var compactOptions = protojson.MarshalOptions{
	UseProtoNames: true,
}

// Marshal serializes a StepPlan to indented JSON bytes with stable spacing.
func Marshal(p *StepPlan) ([]byte, error) {
	raw, err := compactOptions.Marshal(p)
	if err != nil {
		return nil, fmt.Errorf("marshal step plan: %w", err)
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "", "  "); err != nil {
		return nil, fmt.Errorf("indent step plan JSON: %w", err)
	}
	buf.WriteByte('\n')
	return buf.Bytes(), nil
}

// ParseDescription unmarshals JSON bytes into a StepGraphDescription using protojson.
func ParseDescription(data []byte) (*StepGraphDescription, error) {
	var desc StepGraphDescription
	if err := protojson.Unmarshal(data, &desc); err != nil {
		return nil, fmt.Errorf("unmarshal step graph description: %w", err)
	}
	return &desc, nil
}

// ReadDescription reads and unmarshals a StepGraphDescription from a JSON file.
func ReadDescription(path string) (*StepGraphDescription, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read step graph description: %w", err)
	}
	return ParseDescription(data)
}

// ParsePlan unmarshals JSON bytes into a StepPlan using protojson.
func ParsePlan(data []byte) (*StepPlan, error) {
	var plan StepPlan
	if err := protojson.Unmarshal(data, &plan); err != nil {
		return nil, fmt.Errorf("unmarshal step plan: %w", err)
	}
	return &plan, nil
}

// ReadPlan reads and unmarshals a StepPlan from a JSON file.
func ReadPlan(path string) (*StepPlan, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read step plan: %w", err)
	}
	return ParsePlan(data)
}

// FormatStream returns the lowercase name of an execution stream ("main" or "parallel").
// Returns an error if the stream is unspecified or invalid.
func FormatStream(s Stream) (string, error) {
	switch s {
	case StreamMain:
		return "main", nil
	case StreamParallel:
		return "parallel", nil
	default:
		return "", fmt.Errorf("invalid stream %v", s)
	}
}

// FormatExecution returns the lowercase name of an execution mode ("graph" or "exchange").
func FormatExecution(e Execution) (string, error) {
	switch e {
	case ExecutionGraph:
		return "graph", nil
	case ExecutionExchange:
		return "exchange", nil
	default:
		return "", fmt.Errorf("invalid execution %v", e)
	}
}

// PrintWithDescription writes one line per node of a StepPlan: the node name, where it ran
// (its stream), its execution mode from the description if provided, and its measured device time.
func PrintWithDescription(w io.Writer, p *StepPlan, desc *StepGraphDescription) error {
	if p == nil {
		return fmt.Errorf("step plan is nil")
	}
	execByName := make(map[string]string)
	if desc != nil {
		for _, n := range desc.Nodes {
			execStr, err := FormatExecution(n.Execution)
			if err != nil {
				return fmt.Errorf("node %q: %w", n.Name, err)
			}
			execByName[n.Name] = execStr
		}
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	for _, n := range p.Nodes {
		streamStr, err := FormatStream(n.Stream)
		if err != nil {
			return fmt.Errorf("node %q: %w", n.Name, err)
		}
		timeStr := "not measured"
		if n.Stats != nil && n.Stats.DeviceTimeMs != nil {
			timeStr = fmt.Sprintf("%.3f ms", *n.Stats.DeviceTimeMs)
		}
		if desc != nil {
			execStr, ok := execByName[n.Name]
			if !ok {
				return fmt.Errorf("node %q not found in step graph description", n.Name)
			}
			if _, err := fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", n.Name, streamStr, execStr, timeStr); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintf(tw, "%s\t%s\t%s\n", n.Name, streamStr, timeStr); err != nil {
				return err
			}
		}
	}
	return tw.Flush()
}

// Print writes one line per node of a StepPlan: the node name, where it ran (its stream),
// and its measured device time (or "not measured" when unset).
func Print(w io.Writer, p *StepPlan) error {
	return PrintWithDescription(w, p, nil)
}

// validate ensures every node is valid, every node name is non-empty and unique,
// every read specifies a valid variant, resident names are non-empty, every host
// mapping is unique and non-empty, and every edge names an earlier node.
func validate(desc *StepGraphDescription) error {
	if desc == nil || len(desc.Nodes) == 0 {
		return fmt.Errorf("step graph description has no nodes")
	}
	seenMappings := make(map[string]struct{}, len(desc.HostMappings))
	for i, m := range desc.HostMappings {
		if m == nil {
			return fmt.Errorf("host mapping %d is nil", i)
		}
		if m.Name == "" {
			return fmt.Errorf("host mapping %d has empty name", i)
		}
		if _, exists := seenMappings[m.Name]; exists {
			return fmt.Errorf("duplicate host mapping name %q at index %d", m.Name, i)
		}
		seenMappings[m.Name] = struct{}{}
	}

	seen := make(map[string]struct{}, len(desc.Nodes))
	for i, node := range desc.Nodes {
		if node == nil {
			return fmt.Errorf("node %d is nil", i)
		}
		if node.Name == "" {
			return fmt.Errorf("node %d has empty name", i)
		}
		if _, exists := seen[node.Name]; exists {
			return fmt.Errorf("duplicate node name %q at index %d", node.Name, i)
		}
		if node.Execution != ExecutionGraph && node.Execution != ExecutionExchange {
			return fmt.Errorf("node %d (%q) has invalid or unspecified execution: %v", i, node.Name, node.Execution)
		}
		if node.Execution == ExecutionGraph && node.Host != nil {
			return fmt.Errorf("node %d (%q) has execution graph but defines host call", i, node.Name)
		}
		if node.Execution == ExecutionGraph && node.Answer != AnswerUnspecified {
			return fmt.Errorf("node %d (%q) has execution graph but defines exchange answer: %v", i, node.Name, node.Answer)
		}
		if node.Execution == ExecutionExchange && node.Host == nil {
			return fmt.Errorf("node %d (%q) has execution exchange but defines no host call", i, node.Name)
		}
		if node.Execution == ExecutionExchange && (node.Answer < AnswerUnspecified || node.Answer > AnswerRemote) {
			return fmt.Errorf("node %d (%q) has invalid exchange answer: %v", i, node.Name, node.Answer)
		}
		for j, read := range node.Reads {
			if read == nil || read.GetRef() == nil {
				return fmt.Errorf("node %q read %d has no variant set", node.Name, j)
			}
			switch r := read.GetRef().(type) {
			case *planv1alpha1.ValueRef_Edge:
				if r.Edge == nil || r.Edge.Node == "" {
					return fmt.Errorf("node %q read %d edge node is empty", node.Name, j)
				}
				if _, ok := seen[r.Edge.Node]; !ok {
					return fmt.Errorf("node %q read %d edge %q does not name an earlier node", node.Name, j, r.Edge.Node)
				}
			case *planv1alpha1.ValueRef_Resident:
				if r.Resident == "" {
					return fmt.Errorf("node %q read %d resident name is empty", node.Name, j)
				}
			case *planv1alpha1.ValueRef_Input:
				if r.Input == "" {
					return fmt.Errorf("node %q read %d input name is empty", node.Name, j)
				}
			case *planv1alpha1.ValueRef_HostMapping:
				if r.HostMapping == "" {
					return fmt.Errorf("node %q read %d host mapping name is empty", node.Name, j)
				}
				if _, ok := seenMappings[r.HostMapping]; !ok {
					return fmt.Errorf("node %q read %d references undeclared host mapping %q", node.Name, j, r.HostMapping)
				}
			}
		}
		for k, res := range node.Writes {
			if res == "" {
				return fmt.Errorf("node %q write %d resident name is empty", node.Name, k)
			}
		}
		seen[node.Name] = struct{}{}
	}
	return nil
}

// Waits computes the wait dependencies for each node in call order.
// Two nodes may not run at the same time when they touch the same memory
// and at least one of them writes it.
//
// For each node in call order:
// - it waits on every earlier node whose output it reads (an edge);
// - if it reads a resident, it waits on the last earlier node that wrote that resident;
// - if it writes a resident, it waits on the last earlier writer of it and on every node that read it since that write.
func Waits(desc *StepGraphDescription) (map[string][]string, error) {
	if err := validate(desc); err != nil {
		return nil, err
	}

	waits := make(map[string][]string, len(desc.Nodes))
	lastWriter := make(map[string]string)
	readersSinceLastWrite := make(map[string][]string)

	for _, node := range desc.Nodes {
		deps := make(map[string]struct{})

		var readsResidents []string
		for _, read := range node.Reads {
			switch r := read.GetRef().(type) {
			case *planv1alpha1.ValueRef_Edge:
				if edge := r.Edge; edge != nil {
					deps[edge.Node] = struct{}{}
				}
			case *planv1alpha1.ValueRef_Resident:
				if r.Resident != "" {
					readsResidents = append(readsResidents, r.Resident)
				}
			}
		}

		// A node that reads a resident waits on the last earlier node that wrote that resident.
		for _, res := range readsResidents {
			if writer, ok := lastWriter[res]; ok {
				deps[writer] = struct{}{}
			}
		}

		// A node that writes a resident waits on the last earlier writer of it and on every node that read it since that write.
		for _, res := range node.Writes {
			if writer, ok := lastWriter[res]; ok {
				deps[writer] = struct{}{}
			}
			for _, reader := range readersSinceLastWrite[res] {
				deps[reader] = struct{}{}
			}
		}

		// Record readers and writers for subsequent nodes.
		for _, res := range readsResidents {
			readers := readersSinceLastWrite[res]
			if len(readers) == 0 || readers[len(readers)-1] != node.Name {
				readersSinceLastWrite[res] = append(readers, node.Name)
			}
		}
		for _, res := range node.Writes {
			lastWriter[res] = node.Name
			readersSinceLastWrite[res] = nil
		}

		waitList := make([]string, 0, len(deps))
		for dep := range deps {
			waitList = append(waitList, dep)
		}
		slices.Sort(waitList)
		waits[node.Name] = waitList
	}

	return waits, nil
}

// releaseHostMappings identifies file-backed host mappings to release.
// A mapping is released if and only if:
//  1. It is file-backed (m.FileBacked == true).
//  2. At least one node reads it.
//  3. No node answered by the CPU (nor any node whose answerer is unstated) reads it.
//
// Mappings with no known reader are kept. Mappings read by any CPU-executing node or
// any node whose answerer is unstated are kept.
func releaseHostMappings(desc *StepGraphDescription) []string {
	if len(desc.HostMappings) == 0 {
		return nil
	}
	anyReads := make(map[string]bool)
	cpuReads := make(map[string]bool)
	for _, node := range desc.Nodes {
		isCPU := node.Answer == AnswerCPU || node.Answer == AnswerUnspecified
		for _, read := range node.Reads {
			if hm := read.GetHostMapping(); hm != "" {
				anyReads[hm] = true
				if isCPU {
					cpuReads[hm] = true
				}
			}
		}
	}
	var release []string
	for _, m := range desc.HostMappings {
		if !m.FileBacked {
			continue
		}
		if anyReads[m.Name] && !cpuReads[m.Name] {
			release = append(release, m.Name)
		}
	}
	return release
}

// Plan computes the StepPlan from a StepGraphDescription.
//
// A step runs on a main stream. A graph node may run on a second, parallel stream
// when nothing it waits on is the node placed last on the main stream. An exchange node
// stays on the main stream.
//
// opts.Serialize puts every node on the main stream in call order, with each node
// waiting on the immediately preceding node.
// opts.NoParallel keeps the computed wait dependencies and uses only the main stream.
func Plan(desc *StepGraphDescription, opts Options) (*StepPlan, error) {
	if err := validate(desc); err != nil {
		return nil, err
	}

	releases := releaseHostMappings(desc)

	if opts.Serialize {
		nodes := make([]*NodePlan, len(desc.Nodes))
		for i, node := range desc.Nodes {
			var nodeWaits []string
			if i > 0 {
				nodeWaits = []string{desc.Nodes[i-1].Name}
			}
			nodes[i] = &NodePlan{
				Name:   node.Name,
				Stream: StreamMain,
				Waits:  nodeWaits,
			}
		}
		return &StepPlan{Nodes: nodes, ReleaseHostMappings: releases}, nil
	}

	waitsMap, err := Waits(desc)
	if err != nil {
		return nil, err
	}

	if opts.NoParallel {
		nodes := make([]*NodePlan, len(desc.Nodes))
		for i, node := range desc.Nodes {
			nodes[i] = &NodePlan{
				Name:   node.Name,
				Stream: StreamMain,
				Waits:  waitsMap[node.Name],
			}
		}
		return &StepPlan{Nodes: nodes, ReleaseHostMappings: releases}, nil
	}

	nodes := make([]*NodePlan, len(desc.Nodes))
	var (
		mainLast    string
		hasMainLast bool
	)

	for i, node := range desc.Nodes {
		nodeWaits := waitsMap[node.Name]

		isGraph := node.Execution == ExecutionGraph
		onParallel := false
		if isGraph && hasMainLast {
			waitsOnMainLast := false
			for _, w := range nodeWaits {
				if w == mainLast {
					waitsOnMainLast = true
					break
				}
			}
			if !waitsOnMainLast {
				onParallel = true
			}
		}

		stream := StreamMain
		if onParallel {
			stream = StreamParallel
		} else {
			mainLast = node.Name
			hasMainLast = true
		}

		nodes[i] = &NodePlan{
			Name:   node.Name,
			Stream: stream,
			Waits:  nodeWaits,
		}
	}

	return &StepPlan{Nodes: nodes, ReleaseHostMappings: releases}, nil
}
