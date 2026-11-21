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
	"os"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

var buildTreeCommit string

func init() {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				buildTreeCommit = s.Value
				break
			}
		}
	}
}

func shouldWriteJSON(outPath string, asJSON bool) (bool, error) {
	if outPath == "" || outPath == "-" {
		return asJSON, nil
	}
	isJSON := strings.HasSuffix(strings.ToLower(outPath), ".json")
	if asJSON && !isJSON {
		return false, fmt.Errorf("--json conflicts with output filename %s: filename must end in .json", outPath)
	}
	return isJSON || asJSON, nil
}

func cmdPlan(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: vxpu plan [emit|step|show|packages|segments|tree] [flags]")
	}
	switch args[0] {
	case "emit":
		return cmdPlanEmit(ctx, args[1:])
	case "step":
		return cmdPlanStep(ctx, args[1:])
	case "show":
		return cmdPlanShow(ctx, args[1:])
	case "packages":
		return cmdPlanPackages(ctx, args[1:])
	case "segments":
		return cmdPlanSegments(ctx, args[1:])
	case "tree":
		return cmdPlanTree(ctx, args[1:])
	default:
		return fmt.Errorf("unknown plan command %q", args[0])
	}
}

// cmdPlanPackages computes the required compiled packages for a model plan
// and reports which exist or are missing from the package store.
func cmdPlanPackages(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan packages", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan packages --plan FILE --machine NAME --segments NAMES --model NAME [--modelstore URL] [--out FILE] [--json]\n\nCompute the required compiled packages for a model plan.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	planPath := flags.String("plan", "", "path to plan file (required)")
	machine := flags.String("machine", "", "machine kind name (catalog machine, required)")
	segmentsList := flags.String("segments", "", "comma-separated segment names (required)")
	model := flags.String("model", "", "model name in package store (required)")
	storeURL := flags.String("modelstore", os.Getenv("VXPU_MODELSTORE_URL"), "modelstore URL (required, default: $VXPU_MODELSTORE_URL)")
	outPath := flags.String("out", "", "output file path (default: stdout)")
	asJSON := flags.Bool("json", false, "output report as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *planPath == "" {
		return fmt.Errorf("--plan is required")
	}
	if *machine == "" {
		return fmt.Errorf("--machine is required")
	}
	if *model == "" {
		return fmt.Errorf("--model is required")
	}
	if *storeURL == "" {
		return fmt.Errorf("--modelstore or $VXPU_MODELSTORE_URL is required")
	}
	var segments []string
	if *segmentsList != "" {
		for _, s := range strings.Split(*segmentsList, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				segments = append(segments, s)
			}
		}
	}
	for _, arg := range flags.Args() {
		arg = strings.TrimSpace(arg)
		if arg != "" {
			segments = append(segments, arg)
		}
	}
	if len(segments) == 0 {
		return fmt.Errorf("--segments is required")
	}
	writeJSON, err := shouldWriteJSON(*outPath, *asJSON)
	if err != nil {
		return err
	}
	pf, err := kernelplan.ReadPlan(*planPath)
	if err != nil {
		return fmt.Errorf("reading plan: %w", err)
	}
	mk, err := kernelplan.MachineKindFromName(*machine)
	if err != nil {
		return fmt.Errorf("resolving machine kind %q: %w", *machine, err)
	}
	store := weightstore.New(*storeURL)
	report, err := kernelplan.CheckPackages(ctx, store, *model, pf, nil, mk, segments...)
	if err != nil {
		return fmt.Errorf("checking packages: %w", err)
	}
	var out []byte
	if writeJSON {
		opts := protojson.MarshalOptions{
			Multiline: true,
			Indent:    "  ",
		}
		out, err = opts.Marshal(report)
		if err != nil {
			return fmt.Errorf("marshaling package report: %w", err)
		}
		out = append(out, '\n')
	} else {
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "Required: %d, Present: %d, Missing: %d\n", len(report.Required), len(report.Present), len(report.Missing))
		for _, p := range report.Required {
			fmt.Fprintf(&buf, "  %s: %s\n", p.Name, p.Key)
		}
		out = buf.Bytes()
	}
	if *outPath == "" || *outPath == "-" {
		_, err = os.Stdout.Write(out)
		return err
	}
	return os.WriteFile(*outPath, out, 0644)
}

// cmdPlanShow shows a step plan: stream assignments (EXPLAIN),
// with measured actuals beside each node when present (EXPLAIN ANALYZE),
// or formats any plan document as stable JSON.
func cmdPlanShow(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan show", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan show --type TYPE [--graph FILE] [--json] PLAN_FILE\n\nShow a plan document: stream assignments (EXPLAIN) with measured actuals beside each node when present (EXPLAIN ANALYZE), or stable JSON.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	docType := flags.String("type", "", "document type: stepplan, stepgraph, plan, record, catalog, table")
	graphPath := flags.String("graph", "", "optional step graph description file")
	asJSON := flags.Bool("json", false, "show plan document as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *docType == "" {
		return fmt.Errorf("--type is required: must be one of stepplan, stepgraph, plan, record, catalog, table")
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu plan show --type TYPE [--graph FILE] [--json] PLAN_FILE")
	}

	target := flags.Arg(0)
	switch *docType {
	case "stepplan", "step":
		plan, err := stepplan.ReadPlan(target)
		if err != nil {
			return fmt.Errorf("read step plan: %w", err)
		}
		if *asJSON {
			out, err := kernelplan.MarshalAsJSON(plan)
			if err != nil {
				return fmt.Errorf("format JSON: %w", err)
			}
			fmt.Print(string(out))
			return nil
		}
		var desc *planpb.StepGraphDescription
		if *graphPath != "" {
			desc, err = stepplan.ReadDescription(*graphPath)
			if err != nil {
				return fmt.Errorf("read step graph: %w", err)
			}
		}
		return stepplan.PrintWithDescription(os.Stdout, plan, desc)

	case "stepgraph", "graph":
		desc, err := stepplan.ReadDescription(target)
		if err != nil {
			return fmt.Errorf("read step graph: %w", err)
		}
		out, err := kernelplan.MarshalAsJSON(desc)
		if err != nil {
			return fmt.Errorf("format JSON: %w", err)
		}
		fmt.Print(string(out))
		return nil

	case "plan", "planfile":
		pf, err := kernelplan.ReadPlan(target)
		if err != nil {
			return fmt.Errorf("read plan: %w", err)
		}
		out, err := kernelplan.MarshalAsJSON(pf)
		if err != nil {
			return fmt.Errorf("format JSON: %w", err)
		}
		fmt.Print(string(out))
		return nil

	case "record", "kernelrecord":
		rec, err := kernelplan.ReadKernelRecord(target)
		if err != nil {
			return fmt.Errorf("read kernel record: %w", err)
		}
		out, err := kernelplan.MarshalAsJSON(rec)
		if err != nil {
			return fmt.Errorf("format JSON: %w", err)
		}
		fmt.Print(string(out))
		return nil

	case "catalog", "kernelcatalog":
		cat, err := kernelplan.ReadKernelCatalog(target)
		if err != nil {
			return fmt.Errorf("read kernel catalog: %w", err)
		}
		out, err := kernelplan.MarshalAsJSON(cat)
		if err != nil {
			return fmt.Errorf("format JSON: %w", err)
		}
		fmt.Print(string(out))
		return nil

	case "table", "kerneltable":
		var tbl planpb.KernelTable
		if err := kernelplan.ReadJSON(target, &tbl); err != nil {
			return fmt.Errorf("read kernel table: %w", err)
		}
		out, err := kernelplan.MarshalAsJSON(&tbl)
		if err != nil {
			return fmt.Errorf("format JSON: %w", err)
		}
		fmt.Print(string(out))
		return nil

	default:
		return fmt.Errorf("unknown document type %q: must be one of stepplan, stepgraph, plan, record, catalog, table", *docType)
	}
}

func cmdPlanStep(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan step", flag.ContinueOnError)
	graphPath := flags.String("graph", "", "input step graph description file")
	outPath := flags.String("out", "stepplan.pb", "output step plan file")
	serialize := flags.Bool("serialize", false, "serialize all nodes on the main stream in call order")
	asJSON := flags.Bool("json", false, "output step plan as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *graphPath == "" {
		return fmt.Errorf("--graph is required")
	}

	writeJSON, err := shouldWriteJSON(*outPath, *asJSON)
	if err != nil {
		return err
	}

	desc, err := stepplan.ReadDescription(*graphPath)
	if err != nil {
		return fmt.Errorf("read step graph: %w", err)
	}

	plan, err := stepplan.Plan(desc, stepplan.Options{
		Serialize: *serialize,
	})
	if err != nil {
		return fmt.Errorf("plan step graph: %w", err)
	}

	var outBytes []byte
	if writeJSON {
		outBytes, err = stepplan.MarshalAsJSON(plan)
	} else {
		outBytes, err = stepplan.Marshal(plan)
	}
	if err != nil {
		return fmt.Errorf("marshal step plan: %w", err)
	}

	if *outPath == "" || *outPath == "-" {
		_, err := os.Stdout.Write(outBytes)
		return err
	}
	return os.WriteFile(*outPath, outBytes, 0644)
}

// cmdPlanEmit emits a physical execution plan (EXPLAIN) for an artifact before execution.
func cmdPlanEmit(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan emit", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan emit --artifact DIR [--out FILE] [--rows N,...] [--tree-commit COMMIT] [--catalog FILE] [--table FILE] [--device DEVICE] [--json]\n\nEmit a physical execution plan (EXPLAIN) for an artifact before execution.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	artifact := flags.String("artifact", "", "artifact directory containing decode.pt2, manifest.json, binding.json")
	out := flags.String("out", "plan.pb", "output plan file")
	rowsStr := flags.String("rows", "", "comma-separated row counts to plan for")
	treeCommitFlag := flags.String("tree-commit", buildTreeCommit, "git tree commit")
	catalogPath := flags.String("catalog", "", "path to kernel catalog file")
	tablePath := flags.String("table", "", "path to kernel table JSON file")
	device := flags.String("device", "cpu", "target device (default: cpu)")
	asJSON := flags.Bool("json", false, "output plan as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *artifact == "" {
		return fmt.Errorf("--artifact is required")
	}

	writeJSON, err := shouldWriteJSON(*out, *asJSON)
	if err != nil {
		return err
	}

	treeCommit := *treeCommitFlag
	if treeCommit == "" || treeCommit == "unknown" {
		return fmt.Errorf("tree commit is unknown; build with VCS info or set via -ldflags")
	}

	var rowCounts []int
	if *rowsStr != "" {
		for _, part := range strings.Split(*rowsStr, ",") {
			r, err := strconv.Atoi(part)
			if err != nil || r <= 0 {
				return fmt.Errorf("invalid row count %q: must be positive integer", part)
			}
			rowCounts = append(rowCounts, r)
		}
		if len(rowCounts) == 0 {
			return fmt.Errorf("invalid --rows %q: must specify at least one positive integer", *rowsStr)
		}
	}

	plan, err := kernelplan.Emit(*artifact, treeCommit, rowCounts)
	if err != nil {
		return fmt.Errorf("emit plan: %w", err)
	}

	pt2Path := filepath.Join(*artifact, "decode.pt2")
	switch _, err := os.Stat(pt2Path); {
	case err == nil:
		callSites, err := kernelplan.ReadCallSites(*artifact)
		if err != nil {
			return fmt.Errorf("read call sites: %w", err)
		}
		if len(rowCounts) > 0 {
			rowCounts32 := make([]int32, len(rowCounts))
			for i, r := range rowCounts {
				rowCounts32[i] = int32(r)
			}
			for _, cs := range callSites {
				cs.Rows = rowCounts32[0]
				cs.RowCounts = rowCounts32
			}
		}
		var cat *planpb.KernelCatalog
		if *catalogPath != "" {
			cat, err = kernelplan.ReadKernelCatalog(*catalogPath)
			if err != nil {
				return fmt.Errorf("read kernel catalog: %w", err)
			}
		} else {
			cat, err = kernelplan.DefaultKernelCatalog()
			if err != nil {
				return fmt.Errorf("default kernel catalog: %w", err)
			}
		}

		var table *planpb.KernelTable
		if *tablePath != "" {
			var tbl planpb.KernelTable
			if err := kernelplan.ReadJSON(*tablePath, &tbl); err != nil {
				return fmt.Errorf("read kernel table: %w", err)
			}
			table = &tbl
		}

		sites, err := kernelplan.PlanSites(callSites, cat, table, *device)
		if err != nil {
			return fmt.Errorf("plan call sites: %w", err)
		}
		plan.Sites = append(plan.Sites, sites...)
		if table != nil && table.Knobs != nil {
			plan.Knobs = table.Knobs
		}
	case !os.IsNotExist(err):
		return fmt.Errorf("stat %s: %w", pt2Path, err)
	}

	var data []byte
	if writeJSON {
		data, err = kernelplan.MarshalAsJSON(plan)
	} else {
		data, err = kernelplan.Marshal(plan)
	}
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}

	if *out == "" || *out == "-" {
		_, err := os.Stdout.Write(data)
		return err
	}
	return os.WriteFile(*out, data, 0644)
}

// cmdPlanSegments inspects the exported graph in an artifact and lists its
// segments and custom operator call sites.
func cmdPlanSegments(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan segments", flag.ContinueOnError)
	flags.Usage = func() {
		if _, err := fmt.Fprintf(flags.Output(), "Usage: vxpu plan segments --artifact DIR [--out FILE] [--json] [--host-ops]\n\nList the segments and custom operator call sites in an artifact.\n\nFlags:\n"); err != nil {
			return
		}
		flags.PrintDefaults()
	}
	artifact := flags.String("artifact", "", "artifact directory containing decode.pt2")
	outPath := flags.String("out", "", "output file path (default: stdout)")
	asJSON := flags.Bool("json", false, "output report as JSON")
	hostOps := flags.Bool("host-ops", false, "print the set of operators that cut segments")
	if err := flags.Parse(args); err != nil {
		return err
	}

	if *hostOps {
		writeJSON, err := shouldWriteJSON(*outPath, *asJSON)
		if err != nil {
			return err
		}
		ops := kernelplan.HostOpNames()
		var out []byte
		if writeJSON {
			data, err := json.MarshalIndent(ops, "", "  ")
			if err != nil {
				return fmt.Errorf("marshaling host ops: %w", err)
			}
			out = append(data, '\n')
		} else {
			out = []byte(strings.Join(ops, "\n") + "\n")
		}
		if *outPath == "" || *outPath == "-" {
			_, err = os.Stdout.Write(out)
			return err
		}
		return os.WriteFile(*outPath, out, 0644)
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

	var out []byte
	if writeJSON {
		data, err := json.MarshalIndent(segments, "", "  ")
		if err != nil {
			return fmt.Errorf("marshaling segments: %w", err)
		}
		out = append(data, '\n')
	} else {
		var buf bytes.Buffer
		for _, seg := range segments {
			kind := "dense"
			if seg.Host {
				kind = "host"
			}
			fmt.Fprintf(&buf, "%s (%s):\n", seg.Name, kind)
			for _, site := range seg.CallSites {
				if site.Site != "" {
					fmt.Fprintf(&buf, "  %s: %s at %s\n", site.Node, site.Op, site.Site)
				} else {
					fmt.Fprintf(&buf, "  %s: %s\n", site.Node, site.Op)
				}
			}
		}
		out = buf.Bytes()
	}

	if *outPath == "" || *outPath == "-" {
		_, err = os.Stdout.Write(out)
		return err
	}
	return os.WriteFile(*outPath, out, 0644)
}
