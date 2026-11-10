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
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
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
		return fmt.Errorf("usage: vxpu plan emit [flags]")
	}
	if args[0] == "emit" {
		return cmdPlanEmit(ctx, args[1:])
	}
	return fmt.Errorf("unknown plan command %q", args[0])
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

	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}

	if *out == "" || *out == "-" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, append(data, '\n'), 0644)
}
