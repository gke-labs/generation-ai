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
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/stepplan"
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

func cmdPlan(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: vxpu plan [emit|step] [flags]")
	}
	switch args[0] {
	case "emit":
		return cmdPlanEmit(ctx, args[1:])
	case "step":
		return cmdPlanStep(ctx, args[1:])
	default:
		return fmt.Errorf("unknown plan command %q", args[0])
	}
}

func cmdPlanStep(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan step", flag.ContinueOnError)
	graphPath := flags.String("graph", "", "input step graph description JSON file")
	outPath := flags.String("out", "stepplan.json", "output step plan JSON file")
	serialize := flags.Bool("serialize", false, "serialize all nodes on the main stream in call order")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *graphPath == "" {
		return fmt.Errorf("--graph is required")
	}

	data, err := os.ReadFile(*graphPath)
	if err != nil {
		return fmt.Errorf("read step graph: %w", err)
	}

	desc, err := stepplan.ParseDescription(data)
	if err != nil {
		return fmt.Errorf("parse step graph: %w", err)
	}

	plan, err := stepplan.Plan(desc, stepplan.Options{
		Serialize: *serialize,
	})
	if err != nil {
		return fmt.Errorf("plan step graph: %w", err)
	}

	outBytes, err := stepplan.Marshal(plan)
	if err != nil {
		return fmt.Errorf("marshal step plan: %w", err)
	}

	if *outPath == "" || *outPath == "-" {
		fmt.Print(string(outBytes))
		return nil
	}
	return os.WriteFile(*outPath, outBytes, 0644)
}

func cmdPlanEmit(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan emit", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "artifact directory (manifest.json, binding.json)")
	out := flags.String("out", "plan.json", "output plan JSON file")
	rowsStr := flags.String("rows", "", "comma-separated row counts to plan for")
	treeCommitFlag := flags.String("tree-commit", buildTreeCommit, "git tree commit")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *artifact == "" {
		return fmt.Errorf("--artifact is required")
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

	data, err := kernelplan.Marshal(plan)
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}

	if *out == "" || *out == "-" {
		fmt.Print(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0644)
}
