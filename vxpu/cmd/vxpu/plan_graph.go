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
	_ "embed"
	"flag"
	"fmt"
	"html/template"
	"os"

	"google.golang.org/protobuf/encoding/protojson"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
)

//go:embed graphui/graph.html
var graphHTML string

var graphTemplate = template.Must(template.New("graph").Funcs(template.FuncMap{
	"formatFloat": func(f *float64, decimals int) string {
		if f == nil {
			return ""
		}
		return fmt.Sprintf(fmt.Sprintf("%%.%df", decimals), *f)
	},
	"formatStream": func(s planpb.Stream) string {
		switch s {
		case planpb.Stream_STREAM_MAIN:
			return "main"
		case planpb.Stream_STREAM_PARALLEL:
			return "parallel"
		default:
			return "unspecified"
		}
	},
	"nodeMap": func(nodes []*planpb.StepGraphNode) map[string]*planpb.StepGraphNode {
		m := make(map[string]*planpb.StepGraphNode, len(nodes))
		for _, n := range nodes {
			m[n.Name] = n
		}
		return m
	},
	"nodesInStream": func(nodeNames []string, nodeMap map[string]*planpb.StepGraphNode, stream planpb.Stream) []*planpb.StepGraphNode {
		var res []*planpb.StepGraphNode
		for _, name := range nodeNames {
			if n, ok := nodeMap[name]; ok && n.Stream == stream {
				res = append(res, n)
			}
		}
		return res
	},
	"hasParallel": func(nodes []*planpb.StepGraphNode) bool {
		for _, n := range nodes {
			if n.Stream == planpb.Stream_STREAM_PARALLEL {
				return true
			}
		}
		return false
	},
}).Parse(graphHTML))

func renderGraphHTML(graph *planpb.StepPlanGraph) ([]byte, error) {
	var buf bytes.Buffer
	if err := graphTemplate.Execute(&buf, graph); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// cmdPlanGraph reads a step plan file and renders it as an interactive graph in HTML or JSON format,
// showing nodes, stream assignments, wait dependencies, card crossings, and execution actuals.
func cmdPlanGraph(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan graph", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan graph --plan FILE [--graph FILE] [--out FILE] [--json]\n\nDraw a decode step plan as a graph of nodes, streams, waits, and card crossings.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	planPath := flags.String("plan", "", "path to step plan file (required)")
	graphPath := flags.String("graph", "", "path to step graph description file (optional)")
	outPath := flags.String("out", "", "output file path (default: stdout)")
	asJSON := flags.Bool("json", false, "output report as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}

	target := *planPath
	if target == "" {
		if flags.NArg() == 1 {
			target = flags.Arg(0)
		} else if flags.NArg() > 1 {
			return fmt.Errorf("unexpected arguments: %v", flags.Args()[1:])
		}
	} else if flags.NArg() > 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if target == "" {
		return fmt.Errorf("--plan is required")
	}

	writeJSON, err := shouldWriteJSON(*outPath, *asJSON)
	if err != nil {
		return err
	}

	plan, err := stepplan.ReadPlan(target)
	if err != nil {
		return fmt.Errorf("read step plan: %w", err)
	}

	var desc *planpb.StepGraphDescription
	if *graphPath != "" {
		desc, err = stepplan.ReadDescription(*graphPath)
		if err != nil {
			return fmt.Errorf("read step graph description: %w", err)
		}
	}

	graph := stepplan.BuildStepPlanGraph(plan, desc)

	var out []byte
	if writeJSON {
		opts := protojson.MarshalOptions{
			Multiline:     true,
			Indent:        "  ",
			UseProtoNames: true,
		}
		out, err = opts.Marshal(graph)
		if err != nil {
			return fmt.Errorf("marshaling step plan graph: %w", err)
		}
		out = append(out, '\n')
	} else {
		out, err = renderGraphHTML(graph)
		if err != nil {
			return fmt.Errorf("render graph html: %w", err)
		}
	}

	if *outPath == "" || *outPath == "-" {
		_, err = os.Stdout.Write(out)
		return err
	}
	return os.WriteFile(*outPath, out, 0644)
}
