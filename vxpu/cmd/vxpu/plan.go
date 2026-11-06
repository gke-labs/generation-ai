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
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
	"k8s.io/klog/v2"
)

// vxpu plan pipeline: the physical plan of a step across GPUs (vxpu-plan/1)
// and what the cost model predicts for it, from an artifact's inventory
// (hero run 9). vxpu plan serve: the serving shape the planner picks for a
// load (batches in flight, drafts and the accept policy) with the
// alternatives it rejected (hero run 10). vxpu plan matrix: the
// predictions for hero 10's table, as dev/hero/predictions.json.
// vxpu plan emit: reads the strategy table and requested shape to emit a plan.
func cmdPlan(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu plan pipeline|serve|matrix|emit [flags]")
	}
	switch args[0] {
	case "pipeline":
		return runPlanPipeline(ctx, args[1:])
	case "serve":
		return runPlanServe(ctx, args[1:])
	case "matrix":
		return runPlanMatrix(ctx, args[1:])
	case "emit":
		return runPlanEmit(ctx, args[1:])
	}
	return fmt.Errorf("usage: vxpu plan pipeline|serve|matrix|emit [flags]")
}

// readInventory reads the model's per-layer bytes from --layers or --artifact.
func readInventory(layersPath, artifact string) (planner.LayerBytes, error) {
	switch {
	case layersPath != "":
		data, err := os.ReadFile(layersPath)
		if err != nil {
			return planner.LayerBytes{}, err
		}
		return planner.ReadLayerBytesJSON(data)
	case artifact != "":
		manifest, err := os.ReadFile(filepath.Join(artifact, "manifest.json"))
		if err != nil {
			return planner.LayerBytes{}, err
		}
		binding, err := os.ReadFile(filepath.Join(artifact, "binding.json"))
		if err != nil {
			return planner.LayerBytes{}, err
		}
		return planner.ReadLayerBytes(manifest, binding)
	}
	return planner.LayerBytes{}, fmt.Errorf("--artifact DIR or --layers FILE")
}

func runPlanServe(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("plan serve", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "an artifact directory (manifest.json and binding.json): the model's per-layer bytes")
	layersPath := flags.String("layers", "", "the per-layer inventory as JSON (vxpu plan pipeline --layers-out writes one)")
	rangesSpec := flags.String("ranges", "0:8,8:20,20:32,32:43", "the layer ranges, one per GPU in step order")
	rows := flags.Int("rows", 1, "rows to serve, in all")
	batches := flags.Int("batches", 0, "batches in flight (0: the planner chooses between one batch and one per range)")
	mix := flags.String("mix", planner.MixDifferent, "the prompts: 'different' (a prompt per row) or 'same' (one prompt in every row)")
	latency := flags.Float64("latency-ms", 0, "the most a row may take per token (0: no bound)")
	drafter := flags.Bool("drafter", false, "the artifact has the drafter: consider drafting")
	maxDraft := flags.Int("max-draft", 5, "the longest draft to consider")
	newTokens := flags.Int("new-tokens", 0, "also predict what a run of this many new tokens a row measures (the hero harness's 64 or 256): below the steady state when rows draft, since the batch steps until its slowest row is done")
	bandwidth := flags.Float64("bandwidth-gbs", planner.RTXPro6000.BandwidthGBs, "the GPU's copy bandwidth")
	gpu := flags.String("gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "the card the plan is for")
	out := flags.String("out", "", "where to write the chosen plan (default: stdout, after the options)")
	asJSON := flags.Bool("json", false, "print the options as JSON")
	statsPath := flags.String("speculation-stats", "", "a kernel record whose stats.speculation (a drafter run's per-position acceptance and confidence thresholds) replaces the model's fitted acceptance and estimated confidence policy")
	parallel := flags.String("parallel", "auto", "the shapes to choose among: 'layers' (each range whole on its GPU, batches in flight), 'tp4' (the dense layers split across the cards and the experts spread over them, one batch), or 'auto' (both; hero run 14)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	lb, err := readInventory(*layersPath, *artifact)
	if err != nil {
		return fmt.Errorf("vxpu plan serve: %w", err)
	}
	acc, spec, err := speculationInputs(*statsPath)
	if err != nil {
		return fmt.Errorf("vxpu plan serve: %w", err)
	}
	ranges, err := kernelplan.ParseRanges(*rangesSpec)
	if err != nil {
		return err
	}
	plan := &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, GPU: *gpu, Ranges: ranges, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: 1, Schedule: kernelplan.ScheduleRangeOrder}}
	cost := planner.RTXPro6000
	cost.BandwidthGBs = *bandwidth
	load := planner.Load{Rows: *rows, Mix: *mix, Batches: *batches, LatencyMs: *latency, Drafter: *drafter, MaxDraft: *maxDraft, NewTokens: *newTokens}
	var options []planner.ServingOption
	switch *parallel {
	case "layers":
		shaped, err := planner.ChooseShape(lb, plan, load, cost, spec, acc)
		if err != nil {
			for _, o := range shaped {
				log.Info("option", "shape", o.Describe(), "ms_per_token", o.Prediction.MsPerToken)
			}
			return err
		}
		for _, o := range shaped {
			options = append(options, planner.ServingOption{Option: o})
		}
	case "tp4", "auto":
		all, err := planner.ChooseServing(lb, plan, load, cost, spec, acc, planner.RTXPro6000Parallel, planner.RTXPro6000Memory)
		if err != nil {
			for _, o := range all {
				log.Info("option", "shape", o.Describe(), "ms_per_token", o.Prediction.MsPerToken, "rejected", o.Rejected)
			}
			return err
		}
		for _, o := range all {
			if *parallel == "tp4" && o.Parallel == nil {
				continue
			}
			options = append(options, o)
		}
		if *parallel == "tp4" && (len(options) == 0 || options[0].Parallel == nil) {
			return fmt.Errorf("vxpu plan serve: no tensor-parallel shape serves %d rows", load.Rows)
		}
	default:
		return fmt.Errorf("vxpu plan serve: --parallel %q: want layers, tp4 or auto", *parallel)
	}
	if *asJSON {
		data, err := json.MarshalIndent(options, "", " ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
	} else {
		printServingOptions(load, options)
	}
	best := options[0]
	if best.Parallel != nil {
		plan = planner.ParallelPlan(plan, best.Parallel.Reduce)
		plan.GPU = *gpu
	}
	plan.Pipeline.Batches = best.Batches
	plan.Speculation = best.Speculation
	plan.Explain = append(plan.Explain, fmt.Sprintf("chosen for %d rows (%s prompts): %s", load.Rows, load.Mix, best.Describe()))
	plan.Explain = append(plan.Explain, best.Prediction.Basis...)
	plan.Explain = append(plan.Explain, best.Prediction.Pipeline.Basis...)
	data, err := plan.Write()
	if err != nil {
		return err
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}

func printServingOptions(load planner.Load, options []planner.ServingOption) {
	best := options[0]
	fmt.Printf("%d rows, %s prompts: %s\n  %.0f tok/s aggregate, a batch step every %.1f ms, %.1f ms a step, %.2f tokens a step, %.1f ms a token for a row\n",
		load.Rows, load.Mix, best.Describe(), best.Prediction.AggregateTokS, best.Prediction.PeriodMs, best.Prediction.StepMs, best.Prediction.TokensPerStep, best.Prediction.MsPerToken)
	for _, b := range best.Prediction.Basis {
		fmt.Println("  " + b)
	}
	if best.Parallel != nil {
		for _, b := range best.Prediction.Pipeline.Basis {
			fmt.Println("  " + b)
		}
		fmt.Println("  memory: " + best.Memory.Why)
	}
	fmt.Println("rejected:")
	for _, o := range options[1:] {
		fmt.Printf("  %-100s %s\n", o.Describe(), o.Rejected)
	}
}

// matrixCase is one line of hero 10's table as predictions.json carries it
// (dev/hero/table.py reads the file).
type matrixCase struct {
	Rows      int    `json:"rows"`
	Mode      string `json:"mode"`
	Batches   int    `json:"batches"`
	Draft     int    `json:"draft"`
	Policy    string `json:"policy,omitempty"`
	Parallel  string `json:"parallel,omitempty"` // "" or "layers": each range whole on its GPU; "tp4-ep4": the dense layers split across the four cards, the experts spread over them (hero run 14)
	Command   string `json:"command"`
	Chosen    bool   `json:"planner_chooses"`
	ChosenWhy string `json:"planner_choice,omitempty"`

	AggregateTokS float64 `json:"predicted_aggregate_tokens_per_s"`
	SlotMs        float64 `json:"predicted_slot_ms"`
	LatencyMs     float64 `json:"predicted_latency_ms"`
	SingleMs      float64 `json:"predicted_single_ms"`
	TokensPerStep float64 `json:"predicted_tokens_per_step"`
	Distinct      float64 `json:"predicted_distinct_experts"`
	CeilingTokS   float64 `json:"ceiling_tokens_per_s"`
}

func runPlanMatrix(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan matrix", flag.ContinueOnError)
	layersPath := flags.String("layers", "", "the per-layer inventory of the plain artifact (rows12-b1)")
	drafterLayers := flags.String("drafter-layers", "", "the per-layer inventory of the drafter artifact (specm5-b1)")
	rangesSpec := flags.String("ranges", "0:8,8:20,20:32,32:43", "the layer ranges, one per GPU in step order")
	out := flags.String("out", "", "where to write predictions.json (default: stdout)")
	statsPath := flags.String("speculation-stats", "", "a kernel record whose stats.speculation replaces the fitted acceptance and the estimated confidence policy")
	if err := flags.Parse(args); err != nil {
		return err
	}
	acc, spec, err := speculationInputs(*statsPath)
	if err != nil {
		return fmt.Errorf("vxpu plan matrix: %w", err)
	}
	plain, err := readInventory(*layersPath, "")
	if err != nil {
		return fmt.Errorf("vxpu plan matrix: --layers: %w", err)
	}
	withDrafter, err := readInventory(*drafterLayers, "")
	if err != nil {
		return fmt.Errorf("vxpu plan matrix: --drafter-layers: %w", err)
	}
	ranges, err := kernelplan.ParseRanges(*rangesSpec)
	if err != nil {
		return err
	}
	plan := &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, Ranges: ranges, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: 1, Schedule: kernelplan.ScheduleRangeOrder}}
	// the loads of hero 10's table: rows in all, the mix, the batches in
	// flight (0: the planner chooses between one batch and one per range)
	// and whether a drafter artifact exists for them (specm5-b1 and -b8: one
	// and eight rows a batch; specm5x-b33 for 33, which the batched-drafter
	// workstream exports; the plain artifacts at 9 and the wide batches)
	type load struct {
		rows    int
		mix     string
		batches int
		drafter bool
	}
	loads := []load{
		{1, planner.MixSame, 1, true}, {4, planner.MixSame, 4, true},
		{8, planner.MixSame, 1, true}, {8, planner.MixDifferent, 1, true},
		{32, planner.MixSame, 4, true}, {32, planner.MixDifferent, 4, true},
		{9, planner.MixSame, 1, false}, {9, planner.MixDifferent, 1, false},
		{33, planner.MixSame, 1, true}, {33, planner.MixDifferent, 1, true},
		{66, planner.MixSame, 1, false}, {66, planner.MixDifferent, 1, false},
		{99, planner.MixSame, 1, false}, {99, planner.MixDifferent, 1, false},
		{132, planner.MixSame, 0, true}, {132, planner.MixDifferent, 0, true},
	}
	var cases []matrixCase
	for _, l := range loads {
		lb := plain
		if l.drafter {
			lb = withDrafter
		}
		// every shape the planner knows: the layer-sharded ones (one batch or one per range) and,
		// for hero run 14, the dense layers split across the four cards with the experts spread
		// over them; the planner's choice among all of them is marked on one case
		options, err := planner.ChooseServing(lb, plan, planner.Load{Rows: l.rows, Mix: l.mix, Batches: l.batches, Drafter: l.drafter, MaxDraft: 5},
			planner.RTXPro6000, spec, acc, planner.RTXPro6000Parallel, planner.RTXPro6000Memory)
		if err != nil {
			return err
		}
		best := options[0]
		for i, o := range options {
			// the table keeps the shapes the engine can run today: the fixed policy and no drafter
			if o.Speculation != nil && o.Speculation.Policy != kernelplan.PolicyFixed && i > 0 {
				continue
			}
			mode := l.mix
			if o.Rows == 1 {
				mode = "one"
			}
			draft := 0
			policy := ""
			if o.Speculation != nil {
				draft, policy = o.Speculation.Draft, o.Speculation.Policy
			}
			parallel := "layers"
			if o.Parallel != nil {
				parallel = fmt.Sprintf("tp%d-ep%d", o.Parallel.DenseTP, o.Parallel.ExpertEP)
			}
			c := matrixCase{Rows: o.Rows, Mode: mode, Batches: o.Batches, Draft: draft, Policy: policy, Parallel: parallel,
				Command: fmt.Sprintf("plan serve --rows %d --mix %s --batches %d%s --parallel auto", l.rows, l.mix, l.batches, map[bool]string{true: " --drafter", false: ""}[l.drafter]),
				Chosen:  i == 0, AggregateTokS: math.Round(o.Prediction.AggregateTokS), SlotMs: round1(o.Prediction.PeriodMs), LatencyMs: round1(o.Prediction.StepMs),
				SingleMs: round1(o.Prediction.Pipeline.SingleBatchMs), Distinct: round1(o.Prediction.Distinct),
				CeilingTokS: math.Round(o.Prediction.Pipeline.FloorTokS * o.Prediction.TokensPerStep)}
			c.TokensPerStep = math.Round(o.Prediction.TokensPerStep*100) / 100
			if i == 0 {
				c.ChosenWhy = fmt.Sprintf("the planner's choice for %d rows (%s): %s", l.rows, l.mix, best.Describe())
			}
			cases = append(cases, c)
		}
	}
	doc := map[string]any{
		"source": "vxpu plan matrix (pkg/planner: the pipeline model of hero 9, the speculation model of hero 10 and 13, the tensor-parallel model of hero 14), run against the rows12-b1 and specm5-b1 inventories; distinct experts a layer from the measured rows curve (6 at one row or one prompt, 37.3 at 9 different prompts, 90.3 at 33, 91 at 66, 111 at 99, 127 at 132); the tensor-parallel cases rest on the crossing costs measured on pod A (dev/hero/crossings.py) and on an estimated dense efficiency at a quarter of the columns (0.75) and ring reduction (half the one-shot past 33 rows), marked in the plan's basis lines",
		"cases":  cases,
		"note":   "A case is keyed by rows a batch, mode (one/same/different), batches in flight, draft (drafts a step; 0 is no drafter) and parallel ('layers': each range whole on its GPU; 'tp4-ep4': the dense layers split across the four cards and the experts spread over them, one batch). planner_chooses marks the shape the planner picks for the load among every shape; the other shapes of the load are listed so the measurements say how far the choice is from the best measured. The adaptive policies (confidence, experts) are predicted by vxpu plan serve and left out of the table until the engine runs them. The ceiling is a batch's rows times the tokens a step over the slowest range's bytes at one GPU's bandwidth (over the cards' bandwidth together under tensor parallelism).",
	}
	data, err := json.MarshalIndent(doc, "", " ")
	if err != nil {
		return err
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, append(data, '\n'), 0o644)
}

func round1(x float64) float64 { return math.Round(x*10) / 10 }

func runPlanPipeline(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("plan pipeline", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "an artifact directory (manifest.json and binding.json): the model's per-layer bytes")
	layersPath := flags.String("layers", "", "the per-layer inventory as JSON instead of an artifact (vxpu plan pipeline --layers-out writes one)")
	layersOut := flags.String("layers-out", "", "write the artifact's per-layer inventory here")
	rangesSpec := flags.String("ranges", "0:8,8:20,20:32,32:43", "the layer ranges, one per GPU in step order")
	batches := flags.Int("batches", 4, "batches in flight; 0: the planner lays --rows out as one to as many batches as ranges and picks the shape (hero run 10)")
	rows := flags.Int("rows", 1, "rows a batch; with --batches 0, the rows to serve at once")
	distinct := flags.Float64("distinct", 0, "distinct experts a layer the batch selects (default: top-k, the same prompt in every row)")
	distinctAt := flags.String("distinct-at", "", "measured distinct experts a layer by batch width, ROWS=COUNT,...: the curve --batches 0 estimates each candidate's count from (default: --distinct at every width)")
	latency := flags.Float64("latency-ms", 0, "with --batches 0, the per-batch step the chosen shape must stay under (0: no budget)")
	maxRows := flags.Int("max-batch-rows", planner.MaxBatchRows, "with --batches 0, the widest batch the kernels run as planned (the one-pass kernels' row limit)")
	drafter := flags.Bool("drafter", false, "the step verifies the drafter's tokens (its stages and dense weights run on the last range)")
	cutsSpec := flags.String("cuts", "", "layers a range may start at, comma-separated (default: any), for --balance")
	balance := flags.Bool("balance", false, "look for ranges with a shorter slowest range within --budget-gb a GPU")
	budget := flags.Float64("budget-gb", 92, "GPU memory a range's weights may take, for --balance")
	placement := flags.String("placement", "", "the expert placement the engine runs with (vxpu-placement/2): the experts resident per GPU and layer for the memory ledger (default: every expert of a range's layers)")
	memoryGB := flags.Float64("memory-gb", planner.RTXPro6000Memory.TotalGB, "the card's memory for the ledger")
	coldShare := flags.Float64("cold-share", 0, "with a GPU over its memory: the share of a layer's selections the experts evicted to make room would receive (0: uniform popularity, their share of the experts)")
	bandwidth := flags.Float64("bandwidth-gbs", planner.RTXPro6000.BandwidthGBs, "the GPU's copy bandwidth")
	gpu := flags.String("gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "the card the plan is for")
	kernels := flags.String("kernels", "", "the kernel table to name in the plan (a path)")
	out := flags.String("out", "", "where to write the plan (default: stdout, after the prediction)")
	asJSON := flags.Bool("json", false, "print the prediction as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	lb, err := readInventory(*layersPath, *artifact)
	if err != nil {
		return fmt.Errorf("vxpu plan pipeline: %w", err)
	}
	log.Info("read the model's inventory", "layers", len(lb.Layers), "head_gb", float64(lb.HeadBytes)/1e9, "top_k", lb.TopK)
	if *layersOut != "" {
		data, err := json.MarshalIndent(lb, "", " ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*layersOut, data, 0o644); err != nil {
			return err
		}
	}
	ranges, err := kernelplan.ParseRanges(*rangesSpec)
	if err != nil {
		return err
	}
	plan := &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, GPU: *gpu, Ranges: ranges, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: *batches, Schedule: kernelplan.ScheduleRangeOrder}}
	if *kernels != "" {
		plan.Kernels = json.RawMessage(strconv.Quote(*kernels))
	}
	cost := planner.RTXPro6000
	cost.BandwidthGBs = *bandwidth
	mem := planner.LayoutMemory{Cost: planner.RTXPro6000Memory}
	mem.Cost.TotalGB = *memoryGB
	if *placement != "" {
		data, err := os.ReadFile(*placement)
		if err != nil {
			return err
		}
		var doc map[string]any
		if err := json.Unmarshal(data, &doc); err != nil {
			return fmt.Errorf("--placement: %w", err)
		}
		if mem.Residency, err = planner.ReadResidency(doc); err != nil {
			return err
		}
	}
	var choice *planner.ShapeChoice
	if *batches == 0 {
		curve, err := distinctCurve(lb, *distinct, *distinctAt)
		if err != nil {
			return err
		}
		c, err := planner.ChooseLayoutWithMemory(lb, plan, *rows, len(plan.Ranges), *latency, curve, *drafter, cost, *maxRows, mem)
		if err != nil {
			return err
		}
		choice = &c
		*batches, *rows, *distinct = c.Chosen.Batches, c.Chosen.RowsPerBatch, c.Chosen.Distinct
		plan.Pipeline.Batches = *batches
		plan.Explain = append(plan.Explain, "shape: "+c.Why)
	}
	w := planner.Workload{Rows: *rows, DistinctExperts: *distinct, Batches: *batches, Drafter: *drafter}
	pred, err := planner.PredictPipeline(lb, plan, w, cost)
	if err != nil {
		return err
	}
	if *balance {
		var cuts []int
		if *cutsSpec != "" {
			for _, s := range strings.Split(*cutsSpec, ",") {
				l, err := strconv.Atoi(strings.TrimSpace(s))
				if err != nil {
					return fmt.Errorf("--cuts: %w", err)
				}
				cuts = append(cuts, l)
			}
		}
		if plan, pred, err = planner.Balance(lb, plan, w, cost, cuts, *budget); err != nil {
			return err
		}
	}
	ledger, err := planner.PredictMemory(lb, plan, w, mem.Residency, *maxRows, mem.Cost)
	if err != nil {
		return err
	}
	plan.Explain = append(plan.Explain, pred.Basis...)
	plan.Explain = append(plan.Explain, "memory: "+ledger.Why)
	if *asJSON {
		var v any = struct {
			Prediction planner.PipelinePrediction `json:"prediction"`
			Memory     planner.MemoryLedger       `json:"memory"`
		}{pred, ledger}
		if choice != nil {
			v = choice
		}
		data, err := json.MarshalIndent(v, "", " ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
	} else {
		if choice != nil {
			fmt.Println("shape: " + choice.Why)
			for _, cand := range choice.Candidates {
				fits := "fits"
				if !cand.Runnable {
					fits = "wider than the kernels' row limit"
				} else if !cand.Fits {
					fits = "over the budget"
				}
				fmt.Printf("  %d batches of %3d rows: %5.0f tok/s, a batch step every %5.1f ms, latency %5.1f ms, %3.0f distinct experts a layer (%s)\n",
					cand.Batches, cand.RowsPerBatch, cand.AggregateTS, cand.Prediction.PeriodMs, cand.LatencyMs, cand.Distinct, fits)
			}
		}
		fmt.Printf("%d batches of %d rows over %d ranges: %.0f tok/s aggregate, a batch step every %.1f ms, latency %.1f ms (one batch alone %.1f); floor %.0f tok/s at the slowest range's bytes\n",
			*batches, *rows, len(plan.Ranges), pred.AggregateTokS, pred.PeriodMs, pred.LatencyMs, pred.SingleBatchMs, pred.FloorTokS)
		for _, rc := range pred.Ranges {
			fmt.Printf("  gpu %d layers %-6s %2d layers  dense %5.2f GB  experts %6.2f GB  holds %4.0f GB  bytes %5.1f ms + fixed %4.1f = %5.1f ms\n",
				rc.Range.GPU, rc.Range.Layers, rc.Layers, rc.DenseGB, rc.ExpertGB, rc.HeldGB, rc.BytesMs, rc.FixedMs, rc.Ms)
		}
		if pred.BalanceWarning != "" {
			fmt.Println("  " + pred.BalanceWarning)
		}
		for _, b := range pred.Basis {
			fmt.Println("  " + b)
		}
		fmt.Print(ledger.String())
		for _, g := range ledger.GPUs {
			if !g.Fits {
				ec := planner.Evictions(lb, g, w, *coldShare, mem.Cost)
				fmt.Printf("  residency lever on gpu %d: %s\n", g.GPU, ec.Basis)
			}
		}
	}
	data, err := plan.Write()
	if err != nil {
		return err
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}

// distinctCurve is what --batches 0 estimates each candidate width's
// distinct experts from: the measured points of --distinct-at, else a
// fixed --distinct at every width (0: top-k, the same prompt in every row).
func distinctCurve(lb planner.LayerBytes, fixed float64, at string) (planner.DistinctCurve, error) {
	experts := 0
	for _, l := range lb.Layers {
		if l.Experts > experts {
			experts = l.Experts
		}
	}
	curve := planner.DistinctCurve{TopK: lb.TopK, Experts: experts}
	if at != "" {
		pts, err := planner.ParseDistinctPoints(at)
		if err != nil {
			return curve, err
		}
		curve.Points = pts
		return curve, nil
	}
	if fixed > 0 {
		// the same count at every width: two points on a flat line
		curve.Points = []planner.DistinctPoint{{Rows: 2, Distinct: fixed}, {Rows: 1 << 20, Distinct: fixed}}
	}
	return curve, nil
}

// speculationInputs is the speculation model's acceptance and policy
// statistics: the fitted and estimated defaults, or the measured ones from
// a kernel record's stats.speculation when a path is given.
func speculationInputs(statsPath string) (planner.Acceptance, planner.SpecCost, error) {
	acc, spec := planner.DeepSeekV41FlashAcceptance, planner.RTXPro6000Spec
	if statsPath == "" {
		return acc, spec, nil
	}
	data, err := os.ReadFile(statsPath)
	if err != nil {
		return acc, spec, err
	}
	measured, thresholds, replays, err := planner.MeasuredSpeculation(data)
	if err != nil {
		return acc, spec, err
	}
	spec.Thresholds = thresholds
	spec.Replayed = replays
	return measured, spec, nil
}

func runPlanEmit(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("plan emit", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "an artifact directory (manifest.json and binding.json)")
	tablePath := flags.String("table", "", "the strategy table file (JSON)")
	streams := flags.Int("streams", 1, "stream count for requested shape")
	rows := flags.Int("rows", 0, "token rows for requested shape (0: defaults to streams)")
	allowClass := flags.String("allow-class", kernelplan.ClassExact, "allowed numerics class (exact or approximation)")
	budget := flags.Float64("acceptance-budget", 0.0, "maximum acceptance cost for approximations")
	out := flags.String("out", "", "where to write the emitted plan JSON (default: stdout)")
	rangesSpec := flags.String("ranges", "0:43", "the layer ranges, one per GPU in step order")
	gpu := flags.String("gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "the card the plan is for")
	batches := flags.Int("batches", 1, "batches in flight")
	denseTP := flags.Int("dense-tp", 4, "dense tensor parallel degree (1, 2, or 4)")
	expertEP := flags.Int("expert-ep", 4, "expert parallel degree")
	if err := flags.Parse(args); err != nil {
		return err
	}

	var artifactMD5 string
	if *artifact != "" {
		manifestData, err := os.ReadFile(filepath.Join(*artifact, "manifest.json"))
		if err != nil {
			return fmt.Errorf("read artifact manifest: %w", err)
		}
		bindingData, err := os.ReadFile(filepath.Join(*artifact, "binding.json"))
		if err != nil {
			return fmt.Errorf("read artifact binding: %w", err)
		}
		h := md5.New()
		h.Write(manifestData)
		h.Write(bindingData)
		artifactMD5 = hex.EncodeToString(h.Sum(nil))
	}

	commit := os.Getenv("GIT_COMMIT")
	if commit == "" {
		cmdOut, err := exec.CommandContext(ctx, "git", "rev-parse", "HEAD").Output()
		if err != nil {
			klog.FromContext(ctx).V(2).Info("could not resolve git commit", "err", err)
		} else {
			commit = strings.TrimSpace(string(cmdOut))
		}
	}

	var table *kernelplan.StrategyTable
	if *tablePath != "" {
		var err error
		table, err = kernelplan.LoadStrategyTable(*tablePath)
		if err != nil {
			return err
		}
	} else {
		for _, p := range []string{
			filepath.Join("pkg", "kernelplan", "testdata", "strategy-hero15.json"),
			filepath.Join("vxpu", "pkg", "kernelplan", "testdata", "strategy-hero15.json"),
			"strategy-hero15.json",
		} {
			if _, err := os.Stat(p); err == nil {
				t, err := kernelplan.LoadStrategyTable(p)
				if err != nil {
					return fmt.Errorf("load default strategy table %q: %w", p, err)
				}
				table = t
				break
			} else if !os.IsNotExist(err) {
				return fmt.Errorf("stat default strategy table %q: %w", p, err)
			}
		}
	}

	shapeRows := *rows
	if shapeRows == 0 {
		shapeRows = *streams
	}
	shape := kernelplan.Shape{Rows: shapeRows, Streams: *streams}
	allowApproximations := *allowClass == kernelplan.ClassApproximation || *budget > 0
	req := kernelplan.StrategyRequest{
		Shape:               shape,
		AllowApproximations: allowApproximations,
		AcceptanceBudget:    *budget,
	}

	ranges, err := kernelplan.ParseRanges(*rangesSpec)
	if err != nil {
		return err
	}

	plan := &kernelplan.PipelinePlan{
		Format:      kernelplan.PlanFormat,
		GPU:         *gpu,
		Ranges:      ranges,
		Experts:     "with-layers",
		Pipeline:    kernelplan.Pipeline{Batches: *batches, Schedule: kernelplan.ScheduleRangeOrder},
		ArtifactMD5: artifactMD5,
		Commit:      commit,
	}

	if *denseTP > 1 {
		gpus := make([]int, *denseTP)
		for i := range gpus {
			gpus[i] = i
		}
		plan.Parallel = &kernelplan.Parallel{
			DenseTP:  *denseTP,
			ExpertEP: *expertEP,
			GPUs:     gpus,
			Experts:  "by-id",
			Reduce:   kernelplan.ReduceOneShot,
		}
	}

	kernelplan.ApplyStrategies(plan, table, req, kernelplan.SiteClassSumWire)

	data, err := json.MarshalIndent(plan, "", " ")
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}

	if *out != "" {
		if err := os.WriteFile(*out, data, 0644); err != nil {
			return fmt.Errorf("write plan to %q: %w", *out, err)
		}
		return nil
	}
	fmt.Println(string(data))
	return nil
}
