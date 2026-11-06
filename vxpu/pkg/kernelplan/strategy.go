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

package kernelplan

import (
	"encoding/json"
	"fmt"
	"os"
)

// StrategyTableFormat names the strategy table format.
const StrategyTableFormat = "vxpu-strategy-table/1"

// NumericsClass names whether a strategy is bit-exact or an approximation.
const (
	ClassExact         = "exact"
	ClassApproximation = "approximation"
)

// Known site classes and their strategies.
const (
	SiteClassSumWire = "sum_wire"

	StrategySumWireFP32 = "fp32"
	StrategySumWireBF16 = "bf16"

	SiteClassExpert = "expert"

	StrategyExpertGrouped = "grouped"
	StrategyExpertDirect  = "direct"
)

// Shape describes the workload shape: row count and stream count.
type Shape struct {
	Rows    int `json:"rows"`
	Streams int `json:"streams"`
}

// StrategyRow is one measured row in the strategy table.
type StrategyRow struct {
	SiteClass      string   `json:"site_class"`
	Strategy       string   `json:"strategy"`
	Shape          Shape    `json:"shape"`
	StepMs         float64  `json:"step_ms"`
	TokensPerStep  float64  `json:"tokens_per_step"`
	Class          string   `json:"class"`
	AcceptanceCost float64  `json:"acceptance_cost"`
	RunRecordIDs   []string `json:"run_record_ids"`
}

// StrategyTable holds measured pairs across shapes and site classes.
type StrategyTable struct {
	Format string        `json:"format"`
	GPU    string        `json:"gpu,omitempty"`
	Rows   []StrategyRow `json:"rows"`
}

// StrategyRequest defines the workload shape and constraints for strategy selection.
type StrategyRequest struct {
	Shape               Shape
	AllowApproximations bool
	AcceptanceBudget    float64
}

// StrategyChoice records the planner's choice for one site class.
type StrategyChoice struct {
	SiteClass string        `json:"site_class"`
	Strategy  string        `json:"strategy"`
	Default   bool          `json:"default,omitempty"`
	Reason    string        `json:"reason"`
	Rows      []StrategyRow `json:"rows,omitempty"`
}

type unmarshalStrategyRow struct {
	SiteClass      string   `json:"site_class"`
	Strategy       string   `json:"strategy"`
	Shape          Shape    `json:"shape"`
	Rows           int      `json:"rows"`
	Streams        int      `json:"streams"`
	StepMs         float64  `json:"step_ms"`
	TokensPerStep  float64  `json:"tokens_per_step"`
	TokensAStep    float64  `json:"tokens_a_step"`
	Class          string   `json:"class"`
	AcceptanceCost float64  `json:"acceptance_cost"`
	RunRecordIDs   []string `json:"run_record_ids"`
}

// UnmarshalJSON handles both nested shape and flat rows/streams, as well as token count aliases.
func (r *StrategyRow) UnmarshalJSON(data []byte) error {
	var raw unmarshalStrategyRow
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	r.SiteClass = raw.SiteClass
	r.Strategy = raw.Strategy
	r.Shape = raw.Shape
	if r.Shape.Rows == 0 && raw.Rows != 0 {
		r.Shape.Rows = raw.Rows
	}
	if r.Shape.Streams == 0 && raw.Streams != 0 {
		r.Shape.Streams = raw.Streams
	}
	r.StepMs = raw.StepMs
	r.TokensPerStep = raw.TokensPerStep
	if r.TokensPerStep == 0 && raw.TokensAStep != 0 {
		r.TokensPerStep = raw.TokensAStep
	}
	r.Class = raw.Class
	r.AcceptanceCost = raw.AcceptanceCost
	r.RunRecordIDs = raw.RunRecordIDs
	return nil
}

// LoadStrategyTable reads and parses a strategy table from a file.
func LoadStrategyTable(path string) (*StrategyTable, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read strategy table %q: %w", path, err)
	}
	return ParseStrategyTable(data)
}

// ParseStrategyTable parses a strategy table from JSON bytes.
func ParseStrategyTable(data []byte) (*StrategyTable, error) {
	var table StrategyTable
	if err := json.Unmarshal(data, &table); err != nil {
		return nil, fmt.Errorf("parse strategy table: %w", err)
	}
	if table.Format != "" && table.Format != StrategyTableFormat {
		return nil, fmt.Errorf("unsupported strategy table format %q, want %q", table.Format, StrategyTableFormat)
	}
	return &table, nil
}

// DefaultStrategy returns the fallback strategy for a known site class.
func DefaultStrategy(siteClass string) string {
	switch siteClass {
	case SiteClassSumWire, "allreduce", "sum":
		return StrategySumWireFP32
	case SiteClassExpert, "expert_kernel", "expert_form":
		return StrategyExpertGrouped
	default:
		return "default"
	}
}

// ChooseStrategy picks the best strategy for a requested shape and site class from table rows.
func ChooseStrategy(table *StrategyTable, req StrategyRequest, siteClass string) StrategyChoice {
	defaultStrat := DefaultStrategy(siteClass)

	if table == nil || len(table.Rows) == 0 {
		return StrategyChoice{
			SiteClass: siteClass,
			Strategy:  defaultStrat,
			Default:   true,
			Reason:    fmt.Sprintf("no strategy table rows available; falling back to default %s", defaultStrat),
		}
	}

	// Match rows by site class and stream count.
	var matchingRows []StrategyRow
	for _, row := range table.Rows {
		if row.SiteClass != siteClass {
			continue
		}
		if row.Shape.Streams != req.Shape.Streams {
			continue
		}
		if req.Shape.Rows > 0 && row.Shape.Rows > 0 && row.Shape.Rows != req.Shape.Rows {
			continue
		}
		matchingRows = append(matchingRows, row)
	}

	if len(matchingRows) == 0 {
		return StrategyChoice{
			SiteClass: siteClass,
			Strategy:  defaultStrat,
			Default:   true,
			Reason:    fmt.Sprintf("no measured rows for shape (rows=%d, streams=%d); falling back to default %s", req.Shape.Rows, req.Shape.Streams, defaultStrat),
		}
	}

	// Filter by allowed class.
	var allowedRows []StrategyRow
	for _, row := range matchingRows {
		if row.Class == ClassExact || row.Class == "" || row.AcceptanceCost == 0 {
			allowedRows = append(allowedRows, row)
			continue
		}
		if row.Class == ClassApproximation {
			if req.AllowApproximations && row.AcceptanceCost <= req.AcceptanceBudget {
				allowedRows = append(allowedRows, row)
			}
		}
	}

	if len(allowedRows) == 0 {
		return StrategyChoice{
			SiteClass: siteClass,
			Strategy:  defaultStrat,
			Default:   true,
			Reason: fmt.Sprintf("no allowed rows for shape (rows=%d, streams=%d) within acceptance budget (approximations=%v, budget=%.4f); falling back to default %s",
				req.Shape.Rows, req.Shape.Streams, req.AllowApproximations, req.AcceptanceBudget, defaultStrat),
			Rows: matchingRows,
		}
	}

	// Pick lowest step_ms among allowed rows.
	best := allowedRows[0]
	for _, row := range allowedRows[1:] {
		if row.StepMs < best.StepMs {
			best = row
		} else if row.StepMs == best.StepMs && row.AcceptanceCost < best.AcceptanceCost {
			best = row
		}
	}

	return StrategyChoice{
		SiteClass: siteClass,
		Strategy:  best.Strategy,
		Default:   best.Strategy == defaultStrat,
		Reason: fmt.Sprintf("chose %s (%.2f ms, %.2f tok/s, class %s, acceptance cost %.4f) from %d measured row(s)",
			best.Strategy, best.StepMs, best.TokensPerStep, best.Class, best.AcceptanceCost, len(matchingRows)),
		Rows: matchingRows,
	}
}

// ApplyStrategies chooses strategies for the requested site classes and records choices on plan.
func ApplyStrategies(plan *PipelinePlan, table *StrategyTable, req StrategyRequest, siteClasses ...string) {
	if len(siteClasses) == 0 {
		siteClasses = []string{SiteClassSumWire}
	}
	if plan.Strategies == nil {
		plan.Strategies = make(map[string]string)
	}
	for _, sc := range siteClasses {
		choice := ChooseStrategy(table, req, sc)
		plan.Strategies[sc] = choice.Strategy
		plan.StrategyChoices = append(plan.StrategyChoices, choice)
		plan.Explain = append(plan.Explain, fmt.Sprintf("strategy %s: %s (%s)", sc, choice.Strategy, choice.Reason))
	}
}
