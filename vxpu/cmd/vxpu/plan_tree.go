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
	"path/filepath"

	"google.golang.org/protobuf/encoding/protojson"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

//go:embed treeui/tree.html
var treeHTML string

var treeTemplate = template.Must(template.New("tree").Funcs(template.FuncMap{
	"formatFloat": func(f float64, decimals int) string {
		return fmt.Sprintf(fmt.Sprintf("%%.%df", decimals), f)
	},
	"formatBytes": func(b int64) string {
		if b >= 1024*1024*1024 {
			return fmt.Sprintf("%.2f GB (%d bytes)", float64(b)/(1024*1024*1024), b)
		} else if b >= 1024*1024 {
			return fmt.Sprintf("%.1f MB (%d bytes)", float64(b)/(1024*1024), b)
		}
		return fmt.Sprintf("%d bytes", b)
	},
	"formatSegmentRun": func(r planpb.SegmentRun) string {
		switch r {
		case planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH:
			return "plain torch"
		case planpb.SegmentRun_SEGMENT_RUN_JIT:
			return "jit"
		case planpb.SegmentRun_SEGMENT_RUN_PACKAGE:
			return "package"
		case planpb.SegmentRun_SEGMENT_RUN_CAPTURED:
			return "captured"
		default:
			return ""
		}
	},
	"isPassThrough": func(segments []*planpb.Segment) bool {
		return kernelplan.IsPassThrough(segments)
	},
}).Parse(treeHTML))

func renderTreeHTML(tree *planpb.ModelTree) ([]byte, error) {
	var buf bytes.Buffer
	if err := treeTemplate.Execute(&buf, tree); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// cmdPlanTree inspects an exported graph artifact and renders its decode program
// as a tree of segments and custom operator call sites in HTML or JSON format,
// optionally displaying execution plans (EXPLAIN) and run measurements (EXPLAIN ANALYZE).
func cmdPlanTree(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan tree", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan tree --artifact DIR [--plan FILE] [--record FILE] [--out FILE] [--json] [--model NAME]\n\nDraw the model's segments and call sites as a tree, optionally showing physical execution plan and run measurements.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	artifact := flags.String("artifact", "", "artifact directory containing decode.pt2")
	planPath := flags.String("plan", "", "path to physical execution plan file (optional)")
	recordPath := flags.String("record", "", "path to run record file containing execution measurements (optional)")
	outPath := flags.String("out", "", "output file path (default: stdout)")
	asJSON := flags.Bool("json", false, "output report as JSON")
	model := flags.String("model", "", "model name at the root of the tree (default: artifact directory name)")
	if err := flags.Parse(args); err != nil {
		return err
	}

	if *artifact == "" {
		return fmt.Errorf("--artifact is required")
	}

	writeJSON, err := shouldWriteJSON(*outPath, *asJSON)
	if err != nil {
		return err
	}

	segments, err := kernelplan.ReadSegments(*artifact)
	if err != nil {
		return fmt.Errorf("read segments: %w", err)
	}

	modelName := *model
	if modelName == "" {
		modelName = filepath.Base(*artifact)
		if modelName == "." || modelName == "" || modelName == "/" {
			modelName = "model"
		}
	}

	tree := kernelplan.BuildModelTree(modelName, segments)

	var plan *planpb.PlanFile
	if *planPath != "" {
		p, err := kernelplan.ReadPlan(*planPath)
		if err != nil {
			return fmt.Errorf("read plan: %w", err)
		}
		plan = p
	}

	var record *planpb.PlanFile
	if *recordPath != "" {
		r, err := kernelplan.ReadPlan(*recordPath)
		if err != nil {
			return fmt.Errorf("read record: %w", err)
		}
		record = r
	}

	if err := kernelplan.AttachPlan(tree, *artifact, plan, record); err != nil {
		return err
	}

	var out []byte
	if writeJSON {
		opts := protojson.MarshalOptions{
			Multiline:     true,
			Indent:        "  ",
			UseProtoNames: true,
		}
		out, err = opts.Marshal(tree)
		if err != nil {
			return fmt.Errorf("marshaling model tree: %w", err)
		}
		out = append(out, '\n')
	} else {
		out, err = renderTreeHTML(tree)
		if err != nil {
			return fmt.Errorf("render tree html: %w", err)
		}
	}

	if *outPath == "" || *outPath == "-" {
		_, err = os.Stdout.Write(out)
		return err
	}
	return os.WriteFile(*outPath, out, 0644)
}
