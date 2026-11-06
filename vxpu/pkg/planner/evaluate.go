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

package planner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// LogitTolerance is the maximum allowable logit relative deviation (from checks.py).
const LogitTolerance = 0.025

// ErrMissingChecks is returned when a run record does not have the required checks field.
var ErrMissingChecks = errors.New("record missing required 'checks' field")

// CandidatePlan holds one candidate physical plan for evaluation.
type CandidatePlan struct {
	Index int             `json:"index"`
	Name  string          `json:"name,omitempty"`
	Plan  json.RawMessage `json:"plan"`
}

// Verdict describes the evaluation decision for one candidate.
type Verdict struct {
	Candidate  int             `json:"candidate"`
	Name       string          `json:"name,omitempty"`
	Verdict    string          `json:"verdict"` // "kept" or "rejected"
	Reason     string          `json:"reason,omitempty"`
	StepMs     float64         `json:"step_ms,omitempty"`
	BestStepMs float64         `json:"best_step_ms,omitempty"`
	Checks     any             `json:"checks,omitempty"`
	Plan       json.RawMessage `json:"plan,omitempty"`
	Timestamp  time.Time       `json:"timestamp"`
}

// Line returns the formatted verdict line for console output.
func (v Verdict) Line() string {
	label := fmt.Sprintf("candidate %d", v.Candidate)
	if v.Name != "" {
		label = fmt.Sprintf("candidate %d (%s)", v.Candidate, v.Name)
	}
	if v.Verdict == "kept" {
		if v.BestStepMs > 0 && v.StepMs > 0 {
			return fmt.Sprintf("%s: kept (%.2f ms vs %.2f ms)", label, v.StepMs, v.BestStepMs)
		}
		return fmt.Sprintf("%s: kept", label)
	}
	if v.Reason != "" {
		return fmt.Sprintf("%s: rejected: %s", label, v.Reason)
	}
	return fmt.Sprintf("%s: rejected", label)
}

// EvaluationOptions configures the evaluation loop.
type EvaluationOptions struct {
	CandidatesPath  string
	ControlPath     string
	CurrentPlanPath string
	LedgerPath      string
	ControlPlanPath string
	ExecutorCmd     string
	Log             func(format string, args ...any)
}

// GateResult holds the outcome of evaluating the standing checks gates.
type GateResult struct {
	Passed bool
	Reason string
}

// CheckGates evaluates the gates from the record's checks field and top-level checks data.
func CheckGates(checksVal any, rawRecord map[string]any) GateResult {
	if checksVal == nil || checksVal == "" {
		return GateResult{Passed: false, Reason: "record missing checks field"}
	}

	switch v := checksVal.(type) {
	case string:
		return checkGatesString(v, rawRecord)
	case map[string]any:
		return checkGatesMap(v, rawRecord)
	default:
		return GateResult{Passed: false, Reason: fmt.Sprintf("unsupported checks field type: %T", checksVal)}
	}
}

func checkGatesString(checksStr string, rawRecord map[string]any) GateResult {
	lower := strings.ToLower(checksStr)

	// A skipped check is a hard error.
	if strings.Contains(lower, "skipped:") || strings.Contains(lower, "skipped") || strings.Contains(lower, "(override)") {
		return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: skipped check in %q", checksStr)}
	}

	// Exactness check: drafted equals undrafted.
	if strings.Contains(lower, "exactness=failed") || strings.Contains(lower, "exactness: failed") || strings.Contains(lower, "drafted != undrafted") {
		return GateResult{Passed: false, Reason: "gate failed: exactness mismatch (drafted != undrafted)"}
	}

	// Explicit failure or mismatch in the checks line.
	if strings.Contains(lower, "failed:") || strings.Contains(lower, "mismatch") {
		return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: %s", checksStr)}
	}

	// Measured logit error against reference.
	reLogit := regexp.MustCompile(`(?i)(?:logit_error|logits_max_rel|logit_deviation|logit error)[=:\s]+([0-9.]+)`)
	if match := reLogit.FindStringSubmatch(checksStr); match != nil {
		val, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return GateResult{
				Passed: false,
				Reason: fmt.Sprintf("gate failed: invalid logit error %q: %v", match[1], err),
			}
		}
		if val > LogitTolerance {
			return GateResult{
				Passed: false,
				Reason: fmt.Sprintf("gate failed: logit error %.4f exceeds tolerance %.3f", val, LogitTolerance),
			}
		}
	}

	// Also verify any top-level exactness and logit deviation signals in the run record.
	if res := checkRecordFields(rawRecord); !res.Passed {
		return res
	}

	return GateResult{Passed: true}
}

func checkGatesMap(checksMap map[string]any, rawRecord map[string]any) GateResult {
	for k, v := range checksMap {
		strVal := fmt.Sprint(v)
		lower := strings.ToLower(strVal)
		if strings.Contains(lower, "skipped") || strings.Contains(lower, "override") {
			return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: check %s skipped (%s)", k, strVal)}
		}
		if strings.Contains(lower, "failed") || strings.Contains(lower, "mismatch") {
			return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: check %s failed (%s)", k, strVal)}
		}
	}

	if exactness, ok := checksMap["exactness"]; ok {
		if b, ok := exactness.(bool); ok && !b {
			return GateResult{Passed: false, Reason: "gate failed: exactness mismatch (drafted != undrafted)"}
		}
		if s, ok := exactness.(string); ok && (strings.EqualFold(s, "failed") || strings.EqualFold(s, "mismatch")) {
			return GateResult{Passed: false, Reason: "gate failed: exactness mismatch (drafted != undrafted)"}
		}
	}

	for _, key := range []string{"logit_error", "logits_max_rel", "logit_deviation"} {
		if v, ok := checksMap[key]; ok {
			val, err := toFloat(v)
			if err != nil {
				return GateResult{
					Passed: false,
					Reason: fmt.Sprintf("gate failed: invalid %s %v: %v", key, v, err),
				}
			}
			if val > LogitTolerance {
				return GateResult{
					Passed: false,
					Reason: fmt.Sprintf("gate failed: logit error %.4f exceeds tolerance %.3f", val, LogitTolerance),
				}
			}
		}
	}

	if res := checkRecordFields(rawRecord); !res.Passed {
		return res
	}

	return GateResult{Passed: true}
}

func checkRecordFields(rawRecord map[string]any) GateResult {
	if rawRecord == nil {
		return GateResult{Passed: true}
	}

	if val, ok := rawRecord["exactness"]; ok {
		if b, ok := val.(bool); ok && !b {
			return GateResult{Passed: false, Reason: "gate failed: exactness mismatch (drafted != undrafted)"}
		}
		if s, ok := val.(string); ok && (strings.EqualFold(s, "failed") || strings.EqualFold(s, "mismatch")) {
			return GateResult{Passed: false, Reason: "gate failed: exactness mismatch (drafted != undrafted)"}
		}
	}

	if matchVal, ok1 := rawRecord["match"]; ok1 {
		if ofVal, ok2 := rawRecord["of"]; ok2 {
			m, err1 := toFloat(matchVal)
			if err1 != nil {
				return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: invalid match value %v: %v", matchVal, err1)}
			}
			o, err2 := toFloat(ofVal)
			if err2 != nil {
				return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: invalid of value %v: %v", ofVal, err2)}
			}
			if o > 0 && m < o {
				return GateResult{
					Passed: false,
					Reason: fmt.Sprintf("gate failed: exactness mismatch (%.0f of %.0f prompts matched)", m, o),
				}
			}
		}
	}

	if sameRows, ok := rawRecord["same_rows"]; ok {
		if b, ok := sameRows.(bool); ok && !b {
			return GateResult{Passed: false, Reason: "gate failed: duplicate rows invariant failed"}
		}
	}

	for _, key := range []string{"logit_error", "logits_max_rel"} {
		if v, ok := rawRecord[key]; ok {
			val, err := toFloat(v)
			if err != nil {
				return GateResult{Passed: false, Reason: fmt.Sprintf("gate failed: invalid %s %v: %v", key, v, err)}
			}
			if val > LogitTolerance {
				return GateResult{
					Passed: false,
					Reason: fmt.Sprintf("gate failed: logit error %.4f exceeds tolerance %.3f", val, LogitTolerance),
				}
			}
		}
	}

	return GateResult{Passed: true}
}

func toFloat(val any) (float64, error) {
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case string:
		return strconv.ParseFloat(v, 64)
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", val)
	}
}

// ExtractStepMs extracts the median step time in milliseconds from a run record.
func ExtractStepMs(rec map[string]any) (float64, error) {
	for _, key := range []string{"median_step_ms", "step_ms", "step_time_ms", "ms_per_step", "time_ms", "time"} {
		if val, ok := rec[key]; ok && val != nil {
			f, err := toFloat(val)
			if err == nil && f > 0 {
				return f, nil
			}
		}
	}
	return 0, fmt.Errorf("record missing step time field (median_step_ms, step_ms)")
}

// ReadCandidates parses a candidates file.
func ReadCandidates(data []byte) ([]CandidatePlan, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, fmt.Errorf("candidates file is empty")
	}

	// First try JSON array of candidate objects or plans.
	if data[0] == '[' {
		var list []json.RawMessage
		if err := json.Unmarshal(data, &list); err != nil {
			return nil, fmt.Errorf("parsing candidates list: %w", err)
		}
		var out []CandidatePlan
		for i, item := range list {
			cand, err := parseCandidate(i, item)
			if err != nil {
				return nil, err
			}
			out = append(out, cand)
		}
		return out, nil
	}

	// Try object with a "candidates" array.
	var wrapper struct {
		Candidates []json.RawMessage `json:"candidates"`
	}
	if err := json.Unmarshal(data, &wrapper); err == nil && len(wrapper.Candidates) > 0 {
		var out []CandidatePlan
		for i, item := range wrapper.Candidates {
			cand, err := parseCandidate(i, item)
			if err != nil {
				return nil, err
			}
			out = append(out, cand)
		}
		return out, nil
	}

	// Try single candidate object or single plan.
	cand, err := parseCandidate(0, data)
	if err != nil {
		return nil, fmt.Errorf("parsing single candidate plan: %w", err)
	}
	return []CandidatePlan{cand}, nil
}

func parseCandidate(idx int, data []byte) (CandidatePlan, error) {
	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return CandidatePlan{}, err
	}

	name := ""
	if n, ok := obj["name"].(string); ok && n != "" {
		name = n
	} else if l, ok := obj["label"].(string); ok && l != "" {
		name = l
	} else if s, ok := obj["summary"].(string); ok && s != "" {
		name = s
	}

	// If candidate wraps a nested "plan" field, use that.
	if rawPlan, ok := obj["plan"]; ok && rawPlan != nil {
		planBytes, err := json.Marshal(rawPlan)
		if err != nil {
			return CandidatePlan{}, fmt.Errorf("marshaling plan for candidate %d: %w", idx, err)
		}
		return CandidatePlan{
			Index: idx,
			Name:  name,
			Plan:  planBytes,
		}, nil
	}

	// Otherwise the object itself is the plan.
	return CandidatePlan{
		Index: idx,
		Name:  name,
		Plan:  data,
	}, nil
}

// ReadRunRecord parses a run record and refuses it if the checks field is missing.
func ReadRunRecord(data []byte) (map[string]any, error) {
	var rec map[string]any
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("parsing run record JSON: %w", err)
	}
	checksVal, ok := rec["checks"]
	if !ok || checksVal == nil || checksVal == "" {
		return nil, ErrMissingChecks
	}
	return rec, nil
}

// RunExecutor executes the candidate plan through the executor subprocess and returns the record bytes.
func RunExecutor(ctx context.Context, executorCmd string, planBytes []byte) ([]byte, error) {
	planFile, err := os.CreateTemp("", "vxpu-plan-*.json")
	if err != nil {
		return nil, fmt.Errorf("creating temporary plan file: %w", err)
	}
	defer func() {
		if err := os.Remove(planFile.Name()); err != nil && !os.IsNotExist(err) {
			_ = err
		}
	}()

	if _, err := planFile.Write(planBytes); err != nil {
		return nil, errors.Join(fmt.Errorf("writing candidate plan to file: %w", err), planFile.Close())
	}
	if err := planFile.Close(); err != nil {
		return nil, err
	}

	recordFile, err := os.CreateTemp("", "vxpu-record-*.json")
	if err != nil {
		return nil, fmt.Errorf("creating temporary record file: %w", err)
	}
	recordPath := recordFile.Name()
	if err := recordFile.Close(); err != nil {
		return nil, err
	}
	defer func() {
		if err := os.Remove(recordPath); err != nil && !os.IsNotExist(err) {
			_ = err
		}
	}()

	fields := strings.Fields(executorCmd)
	if len(fields) == 0 {
		return nil, fmt.Errorf("empty executor command")
	}

	args := append(fields[1:], "--plan", planFile.Name(), "--out", recordPath)
	cmd := exec.CommandContext(ctx, fields[0], args...)
	cmd.Env = append(os.Environ(),
		"VXPU_PLAN="+planFile.Name(),
		"VXPU_PLAN_RECORD="+recordPath,
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	if runErr != nil {
		errOutput := strings.TrimSpace(stderr.String())
		if errOutput == "" {
			errOutput = strings.TrimSpace(stdout.String())
		}
		return nil, fmt.Errorf("build failed: %w: %s", runErr, errOutput)
	}

	// Check if the record file was populated.
	if fi, err := os.Stat(recordPath); err == nil && fi.Size() > 0 {
		data, err := os.ReadFile(recordPath)
		if err == nil && len(bytes.TrimSpace(data)) > 0 {
			return data, nil
		}
	}

	// Check stdout for RESULT lines or direct JSON.
	outStr := stdout.String()
	for _, line := range strings.Split(outStr, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "RESULT ") {
			jsonPart := strings.TrimPrefix(trimmed, "RESULT ")
			return []byte(jsonPart), nil
		}
	}

	trimmedOut := bytes.TrimSpace(stdout.Bytes())
	if len(trimmedOut) > 0 && trimmedOut[0] == '{' {
		return trimmedOut, nil
	}

	return nil, fmt.Errorf("build failed: executor produced no run record in %s or stdout", recordPath)
}

func writePlanFile(path string, planBytes []byte) error {
	var formatted bytes.Buffer
	err := json.Indent(&formatted, planBytes, "", "  ")
	if err != nil {
		return os.WriteFile(path, planBytes, 0o644)
	}
	return os.WriteFile(path, formatted.Bytes(), 0o644)
}

// Evaluate runs the evaluation loop over candidate plans against a control record.
func Evaluate(ctx context.Context, opts EvaluationOptions) ([]Verdict, error) {
	if opts.CandidatesPath == "" || opts.ControlPath == "" {
		return nil, fmt.Errorf("candidates and control paths are required")
	}

	controlData, err := os.ReadFile(opts.ControlPath)
	if err != nil {
		return nil, fmt.Errorf("reading control record from %s: %w", opts.ControlPath, err)
	}

	controlRecord, err := ReadRunRecord(controlData)
	if err != nil {
		return nil, fmt.Errorf("control record %s: %w", opts.ControlPath, err)
	}

	controlStepMs, err := ExtractStepMs(controlRecord)
	if err != nil {
		return nil, fmt.Errorf("control record %s: %w", opts.ControlPath, err)
	}

	candidatesData, err := os.ReadFile(opts.CandidatesPath)
	if err != nil {
		return nil, fmt.Errorf("reading candidates from %s: %w", opts.CandidatesPath, err)
	}

	candidates, err := ReadCandidates(candidatesData)
	if err != nil {
		return nil, fmt.Errorf("candidates file %s: %w", opts.CandidatesPath, err)
	}

	executor := opts.ExecutorCmd
	if executor == "" {
		executor = os.Getenv("VXPU_EXECUTOR")
	}
	if executor == "" {
		executor = "python dev/hero/batches.py"
	}

	var lastKeptPlan []byte
	if opts.ControlPlanPath != "" {
		data, err := os.ReadFile(opts.ControlPlanPath)
		if err != nil {
			return nil, fmt.Errorf("reading control plan %s: %w", opts.ControlPlanPath, err)
		}
		lastKeptPlan = data
	}

	bestStepMs := controlStepMs
	var verdicts []Verdict

	for _, cand := range candidates {
		v := Verdict{
			Candidate:  cand.Index,
			Name:       cand.Name,
			Plan:       cand.Plan,
			BestStepMs: bestStepMs,
			Timestamp:  time.Now().UTC(),
		}

		recordBytes, runErr := RunExecutor(ctx, executor, cand.Plan)
		if runErr != nil {
			v.Verdict = "rejected"
			v.Reason = runErr.Error()
			verdicts = append(verdicts, v)
			if err := appendLedger(opts.LedgerPath, v); err != nil {
				return verdicts, err
			}
			continue
		}

		rec, recErr := ReadRunRecord(recordBytes)
		if recErr != nil {
			// A record without a checks field is refused (stops the evaluation).
			if errors.Is(recErr, ErrMissingChecks) {
				return verdicts, fmt.Errorf("candidate %d record refused: %w", cand.Index, ErrMissingChecks)
			}
			v.Verdict = "rejected"
			v.Reason = fmt.Sprintf("build failed: invalid record: %v", recErr)
			verdicts = append(verdicts, v)
			if err := appendLedger(opts.LedgerPath, v); err != nil {
				return verdicts, err
			}
			continue
		}

		v.Checks = rec["checks"]

		gateResult := CheckGates(rec["checks"], rec)
		if !gateResult.Passed {
			v.Verdict = "rejected"
			v.Reason = gateResult.Reason
			verdicts = append(verdicts, v)
			if err := appendLedger(opts.LedgerPath, v); err != nil {
				return verdicts, err
			}
			continue
		}

		stepMs, timeErr := ExtractStepMs(rec)
		if timeErr != nil {
			v.Verdict = "rejected"
			v.Reason = fmt.Sprintf("build failed: %v", timeErr)
			verdicts = append(verdicts, v)
			if err := appendLedger(opts.LedgerPath, v); err != nil {
				return verdicts, err
			}
			continue
		}
		v.StepMs = stepMs

		if stepMs >= bestStepMs {
			v.Verdict = "rejected"
			v.Reason = fmt.Sprintf("slower (%.2f ms vs %.2f ms)", stepMs, bestStepMs)
			verdicts = append(verdicts, v)
			if err := appendLedger(opts.LedgerPath, v); err != nil {
				return verdicts, err
			}
			continue
		}

		// Gates passed and time improved: keep candidate plan.
		v.Verdict = "kept"
		v.Reason = ""
		bestStepMs = stepMs
		lastKeptPlan = cand.Plan

		if opts.CurrentPlanPath != "" {
			if err := writePlanFile(opts.CurrentPlanPath, cand.Plan); err != nil {
				return verdicts, err
			}
		}

		verdicts = append(verdicts, v)
		if err := appendLedger(opts.LedgerPath, v); err != nil {
			return verdicts, err
		}
	}

	// If a plan was kept and a plan file path is configured, ensure it is written.
	if opts.CurrentPlanPath != "" && len(lastKeptPlan) > 0 {
		if err := writePlanFile(opts.CurrentPlanPath, lastKeptPlan); err != nil {
			return verdicts, err
		}
	}

	return verdicts, nil
}

func appendLedger(path string, v Verdict) error {
	if path == "" {
		return nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshaling verdict for ledger: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("opening ledger %s: %w", path, err)
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		return errors.Join(fmt.Errorf("writing to ledger %s: %w", path, err), f.Close())
	}
	return f.Close()
}
