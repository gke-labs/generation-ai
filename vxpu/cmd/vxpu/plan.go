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
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func cmdPlan(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: vxpu plan emit [flags] | vxpu plan record [flags]")
	}
	switch args[0] {
	case "emit":
		return cmdPlanEmit(ctx, args[1:])
	case "record":
		return cmdPlanRecord(ctx, args[1:])
	default:
		return fmt.Errorf("unknown plan command %q", args[0])
	}
}

func cmdPlanEmit(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan emit", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "artifact directory (manifest.json, binding.json)")
	out := flags.String("out", "plan.json", "output plan JSON file")
	rowsStr := flags.String("rows", "1,5,8,16,32,40,64", "comma-separated row counts to plan for")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *artifact == "" {
		return fmt.Errorf("--artifact is required")
	}

	var rowCounts []int
	if *rowsStr != "" {
		for _, part := range strings.Split(*rowsStr, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			r, err := strconv.Atoi(part)
			if err != nil || r <= 0 {
				return fmt.Errorf("invalid row count %q: must be positive integer", part)
			}
			rowCounts = append(rowCounts, r)
		}
	}

	plan, err := kernelplan.Emit(*artifact, rowCounts)
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

	if err := os.WriteFile(*out, append(data, '\n'), 0644); err != nil {
		return fmt.Errorf("write plan to %s: %w", *out, err)
	}
	return nil
}

func cmdPlanRecord(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan record", flag.ContinueOnError)
	in := flags.String("in", "", "path to run record JSON file")
	ledger := flags.String("ledger", "ledger.jsonl", "path to local ledger file")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *in == "" {
		return fmt.Errorf("--in is required")
	}

	data, err := os.ReadFile(*in)
	if err != nil {
		return fmt.Errorf("read run record %s: %w", *in, err)
	}

	rec, err := kernelplan.ValidateRecord(data)
	if err != nil {
		return fmt.Errorf("validate run record: %w", err)
	}

	ledgerPath := *ledger
	if envLedger := os.Getenv("VXPU_LEDGER"); envLedger != "" && ledgerPath == "ledger.jsonl" {
		ledgerPath = envLedger
	}

	if err := kernelplan.AppendLedger(ledgerPath, rec); err != nil {
		return fmt.Errorf("append to ledger: %w", err)
	}
	return nil
}
