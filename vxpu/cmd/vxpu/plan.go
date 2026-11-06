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
	"math"
	"os"

	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
	"k8s.io/klog/v2"
)

// cmdPlan dispatches subcommands of `vxpu plan`.
func cmdPlan(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu plan cost|verify [flags]")
	}
	switch args[0] {
	case "cost":
		return runPlanCost(ctx, args[1:])
	case "verify":
		return runPlanCost(ctx, append([]string{"--verify"}, args[1:]...))
	default:
		return fmt.Errorf("unknown plan command %q (usage: vxpu plan cost|verify [flags])", args[0])
	}
}

// cmdCost provides a top-level alias for `vxpu plan cost`.
func cmdCost(ctx context.Context, args []string) error {
	return runPlanCost(ctx, args)
}

func runPlanCost(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("plan cost", flag.ContinueOnError)
	ledgerPath := flags.String("ledger", "", "path to ledger file containing run records (JSON lines or JSON array)")
	recordPath := flags.String("record", "", "path to a single run record or phase trace JSON file")
	inPath := flags.String("in", "", "alias for --record")
	tree := flags.String("tree", "", "filter records by tree commit hash")
	shapeStr := flags.String("shape", "", "target shape name (e.g. five-row, block-7, 8-stream, or rows=N,streams=M)")
	band := flags.Float64("band", 0, "acceptable median error percentage band (default: 1.0 for 1-stream, stated per shape)")
	verify := flags.Bool("verify", false, "verify predictions against actual runs in ledger and report error distribution")
	asJSON := flags.Bool("json", false, "output report as JSON")

	if err := flags.Parse(args); err != nil {
		return err
	}

	targetRecord := *recordPath
	if targetRecord == "" {
		targetRecord = *inPath
	}

	if *ledgerPath == "" && targetRecord == "" {
		return fmt.Errorf("usage: vxpu plan cost --ledger LEDGER [flags] or vxpu plan cost --record RECORD [flags]")
	}

	// 1. Ledger verification mode.
	if *ledgerPath != "" || *verify {
		if *ledgerPath == "" {
			return fmt.Errorf("--ledger is required when --verify is requested")
		}
		records, err := planner.ReadLedger(*ledgerPath)
		if err != nil {
			return fmt.Errorf("read ledger: %w", err)
		}
		if len(records) == 0 {
			return fmt.Errorf("ledger %s is empty", *ledgerPath)
		}

		results, err := planner.VerifyLedger(records, *tree, *shapeStr, *band)
		if err != nil {
			return fmt.Errorf("verify ledger: %w", err)
		}

		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(results); err != nil {
				return err
			}
		} else {
			if err := planner.FormatVerificationReport(os.Stdout, results); err != nil {
				return err
			}
		}

		var rejected []string
		for _, r := range results {
			if !r.Accepted {
				rejected = append(rejected, fmt.Sprintf("%s/%s (median error %.2f%% > band %.2f%%)",
					r.Tree, r.Shape, r.MedianErrorPct, r.BandPct))
			}
		}
		if len(rejected) > 0 {
			log.Info("cost model verification rejected", "count", len(rejected))
			return fmt.Errorf("cost model rejected: median error exceeds band for: %s", rejected[0])
		}
		return nil
	}

	// 2. Single run record or trace evaluation mode.
	data, err := os.ReadFile(targetRecord)
	if err != nil {
		return fmt.Errorf("read record %s: %w", targetRecord, err)
	}

	var rec planner.RunRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return fmt.Errorf("unmarshal record: %w", err)
	}

	sh := planner.ParseShape(*shapeStr)
	if *shapeStr == "" {
		sh = planner.ParseShape(rec.GetShape())
	}
	if rec.Rows > 0 && sh.Rows == 1 && *shapeStr == "" {
		sh.Rows = rec.Rows
	}
	if rec.Streams > 0 && sh.Streams == 1 && *shapeStr == "" {
		sh.Streams = rec.Streams
	}

	nodes := rec.GetNodes()
	if len(nodes) == 0 {
		return fmt.Errorf("record %s contains no in-graph stamps (nodes)", targetRecord)
	}

	model := planner.NewCriticalPathCostModel()
	pred, err := model.Predict(nodes, sh)
	if err != nil {
		return fmt.Errorf("predict: %w", err)
	}

	actualStepMs := rec.GetStepMs()
	type singleReport struct {
		Shape          string                            `json:"shape"`
		PredictedMs    float64                           `json:"predicted_step_ms"`
		ActualStepMs   float64                           `json:"actual_step_ms,omitempty"`
		ErrorMs        float64                           `json:"error_ms,omitempty"`
		ErrorPct       float64                           `json:"error_pct,omitempty"`
		Accepted       bool                              `json:"accepted,omitempty"`
		ByClassMs      map[string]float64                `json:"by_class_ms"`
		SideBranchCost float64                           `json:"side_branch_cost_ms"`
		HiddenWorkMs   float64                           `json:"hidden_work_ms"`
		FusedSavingsMs float64                           `json:"fused_savings_ms"`
		CollectiveWait float64                           `json:"collective_wait_ms"`
		CriticalNodes  []planner.CriticalPathNodeSummary `json:"critical_nodes,omitempty"`
	}

	rep := singleReport{
		Shape:          pred.Shape,
		PredictedMs:    pred.PredictedStepMs,
		ByClassMs:      pred.ByClassMs,
		SideBranchCost: pred.SideBranchCost,
		HiddenWorkMs:   pred.HiddenWorkMs,
		FusedSavingsMs: pred.FusedSavingsMs,
		CollectiveWait: pred.CollectiveWait,
		CriticalNodes:  pred.CriticalNodes,
	}

	if actualStepMs > 0 {
		rep.ActualStepMs = actualStepMs
		rep.ErrorMs = pred.PredictedStepMs - actualStepMs
		rep.ErrorPct = (rep.ErrorMs / actualStepMs) * 100.0
		effectiveBand := *band
		if effectiveBand <= 0 {
			if sh.Streams <= 1 {
				effectiveBand = 1.0
			} else {
				effectiveBand = 2.0
			}
		}
		rep.Accepted = math.Abs(rep.ErrorPct) <= effectiveBand
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}

	fmt.Printf("Critical Path Cost Model v0 (shape: %s):\n", pred.Shape)
	fmt.Printf("  predicted step: %.2f ms\n", pred.PredictedStepMs)
	if actualStepMs > 0 {
		status := "ACCEPTED"
		if !rep.Accepted {
			status = "REJECTED"
		}
		fmt.Printf("  actual step:    %.2f ms (error: %+.2f ms, %+.2f%%) [%s]\n",
			rep.ActualStepMs, rep.ErrorMs, rep.ErrorPct, status)
	}
	fmt.Printf("  side branches:  %.2f ms wall cost (hidden work: %.2f ms overlapped)\n",
		pred.SideBranchCost, pred.HiddenWorkMs)
	fmt.Printf("  fused regions:  in place (saved: %.2f ms vs isolated parts)\n", pred.FusedSavingsMs)
	if pred.CollectiveWait > 0 {
		fmt.Printf("  collective wait: %.2f ms measured at row count %d\n", pred.CollectiveWait, sh.Rows)
	}
	fmt.Println("  per class breakdown:")
	for c, ms := range pred.ByClassMs {
		fmt.Printf("    %-14s %.2f ms\n", c, ms)
	}

	return nil
}
