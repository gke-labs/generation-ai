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

// The measuring step (hero run 11): the GEMV templates' blocks per shape
// class, swept on the card and chosen by measurement.
//
// A shape class is what fixes a GEMV kernel's tiles: the template, the
// rows, the weight's shape and dtype, the prologue and epilogue flags
// (python/vxpu/triton_ops.py: gemv_class). The engine's kernel record
// lists the classes it launched with the blocks that ran
// (stats.gemv_classes). Measure enumerates the block sets each class can
// run (the tiles must divide the shape), times every one through the
// Python micro-benchmark (python -m vxpu.measure run: the kernel over
// weights cycled past the L2, captured in a CUDA graph), and writes the
// fastest per class into the table's "blocks" entries, with the bytes
// and the floor so the choice reads as an efficiency. A class the engine
// ran under the table's blocks checks against them (CheckBlocks).

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

// BlockEntry is the table's choice of blocks for one GEMV shape class.
type BlockEntry struct {
	Shapes string         `json:"shapes"` // the shape class (triton_ops.gemv_class)
	Blocks map[string]int `json:"blocks"` // BN, BK, KSPLIT, warps, stages
	// RowsBand, when set, is [lo, hi]: the entry also serves every class that
	// differs from Shapes only in its rows, for rows within the band (the exact
	// class wins). The engine reads it (plan.load_table); a class measured once
	// above 16 rows applies through the band without naming each row count.
	RowsBand []int `json:"rows_band,omitempty"`
	// What the measurement said: microseconds a launch, the bytes the
	// class moves, the floor at the card's bandwidth, and the efficiency
	// (floor / us). The engine does not read these.
	Us         float64 `json:"us,omitempty"`
	Bytes      int64   `json:"bytes,omitempty"`
	FloorUs    float64 `json:"floor_us,omitempty"`
	Efficiency float64 `json:"efficiency,omitempty"`
	// Why says how it was chosen: the candidates tried, what the
	// engine's own blocks measured.
	Why string `json:"why,omitempty"`
}

// ClassRun is a shape class as the record saw it: launches a step and
// the blocks that ran (stats.gemv_classes).
type ClassRun struct {
	Count     int            `json:"count"` // launches since the process started (warm-up and prefill included)
	Sites     int            `json:"sites"` // call sites the lowering traced: the launches a decode step
	Blocks    map[string]int `json:"blocks"`
	FromTable bool           `json:"from_table"`
}

// AStep is the class's launches a decode step: the traced sites, else the count.
func (c ClassRun) AStep() int {
	if c.Sites > 0 {
		return c.Sites
	}
	return c.Count
}

// Classes reads the GEMV shape classes out of a record's stats.
func Classes(r *Record) (map[string]ClassRun, error) {
	raw, ok := r.Stats["gemv_classes"]
	if !ok {
		return nil, fmt.Errorf("the record has no stats.gemv_classes: it was written by an engine before the measuring step existed")
	}
	var out map[string]ClassRun
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("stats.gemv_classes: %w", err)
	}
	return out, nil
}

// Class is a parsed shape class key.
type Class struct {
	Key      string
	Template string
	Rows     int
	N, K     int
	Fields   map[string]string // the flags, by name: w, native, epi, groups, out, split, hd, rd
}

// ParseClass parses "gemv_rows rows=1 n=5120 k=8192 w=fp8 native=1 epi=quant".
func ParseClass(key string) (Class, error) {
	parts := strings.Fields(key)
	if len(parts) < 4 {
		return Class{}, fmt.Errorf("shape class %q: want a template and rows=, n=, k=", key)
	}
	c := Class{Key: key, Template: parts[0], Fields: map[string]string{}}
	for _, p := range parts[1:] {
		name, value, ok := strings.Cut(p, "=")
		if !ok {
			return Class{}, fmt.Errorf("shape class %q: %q is not name=value", key, p)
		}
		c.Fields[name] = value
	}
	var err error
	for name, dst := range map[string]*int{"rows": &c.Rows, "n": &c.N, "k": &c.K} {
		v, ok := c.Fields[name]
		if !ok {
			return Class{}, fmt.Errorf("shape class %q: missing %s", key, name)
		}
		if *dst, err = strconv.Atoi(v); err != nil {
			return Class{}, fmt.Errorf("shape class %q: %s: %w", key, name, err)
		}
	}
	return c, nil
}

// field is a numeric flag of the class, 0 when absent or not a number (the flags
// the keys carry are written by gemv_class, so a bad one is a bad key, not an error here).
func (c Class) field(name string) int {
	n, err := strconv.Atoi(c.Fields[name])
	if err != nil {
		return 0
	}
	return n
}

// Candidates enumerates the block sets a class can run, the engine's own
// blocks first (so the choice never regresses a class the sweep finds
// nothing better for). The axes: weight rows per program BN (16, 32, 64;
// whole quantization blocks of 32 where the epilogue quantizes; even and
// dividing the head's unrotated part for the rotary epilogue), the K step
// BK (64 to 512 dividing the slice), the K split KSPLIT (1 to 16 where the
// caller takes partial sums, and only until the programs fill the card
// twice over: sms), warps (4, 8). Stages stay at the compiler's
// default; a second pass can sweep them on the winner.
func Candidates(c Class, ran map[string]int, sms int) []map[string]int {
	var bns, bks, splits, warps []int
	switch c.Template {
	case "dual_gemv_swiglu":
		bns = []int{16, 32, 64} // 16 only with a K split: the fused epilogue holds whole blocks of 32
	case "gemv_pe":
		bns = []int{16, 32}
	default:
		bns = []int{16, 32, 64}
		if c.Fields["epi"] == "quant" {
			bns = []int{32, 64}
		}
	}
	bks = []int{64, 128, 256, 512}
	splits = []int{1}
	if c.Template == "dual_gemv_swiglu" || c.Fields["split"] == "1" {
		splits = []int{1, 2, 4, 8, 16}
	}
	warps = []int{4, 8}
	small := int64(c.N)*int64(c.K) < 16<<20 // a weight under 16 MB: a few microseconds a launch, where fewer warps may ramp faster
	if small {
		warps = []int{2, 4, 8}
	}
	rowsPerGroup := c.N
	if g := c.field("groups"); g > 1 {
		rowsPerGroup = c.N / g
	}
	hd, rd := c.field("hd"), c.field("rd")
	var out []map[string]int
	seen := map[string]bool{}
	add := func(b map[string]int) {
		k := blockKey(b)
		if !seen[k] {
			seen[k] = true
			out = append(out, b)
		}
	}
	fits := func(bn, bk, ks int) bool {
		if c.N%bn != 0 || rowsPerGroup%bn != 0 || c.K%ks != 0 || (c.K/ks)%bk != 0 || bk > c.K {
			return false
		}
		if c.Template == "gemv_pe" && (bn%2 != 0 || (hd-rd)%bn != 0) {
			return false
		}
		if c.Template == "dual_gemv_swiglu" && bn%32 != 0 && ks == 1 {
			return false
		}
		if ks > 1 && sms > 0 && (c.N/bn)*(ks/2) >= 2*sms { // the split before this one already filled the card twice
			return false
		}
		return true
	}
	if ran != nil && fits(ran["BN"], ran["BK"], max(1, ran["KSPLIT"])) {
		add(map[string]int{"BN": ran["BN"], "BK": ran["BK"], "KSPLIT": max(1, ran["KSPLIT"]), "warps": orDefault(ran["warps"], 4), "stages": orDefault(ran["stages"], 3)})
	}
	for _, bn := range bns {
		for _, bk := range bks {
			for _, ks := range splits {
				if !fits(bn, bk, ks) {
					continue
				}
				for _, w := range warps {
					add(map[string]int{"BN": bn, "BK": bk, "KSPLIT": ks, "warps": w, "stages": 3})
				}
			}
		}
	}
	return out
}

func orDefault(v, d int) int {
	if v == 0 {
		return d
	}
	return v
}

func blockKey(b map[string]int) string {
	names := make([]string, 0, len(b))
	for k := range b {
		names = append(names, k)
	}
	sort.Strings(names)
	var sb strings.Builder
	for _, k := range names {
		fmt.Fprintf(&sb, "%s=%d ", k, b[k])
	}
	return strings.TrimSpace(sb.String())
}

// Job is one timing the runner performs: a class under blocks.
type Job struct {
	Shapes string         `json:"shapes"`
	Blocks map[string]int `json:"blocks"`
}

// Result is what the runner reports for a job (python -m vxpu.measure run's line).
type Result struct {
	Shapes        string         `json:"shapes"`
	Blocks        map[string]int `json:"blocks"`
	Us            float64        `json:"us"`
	UsMin         float64        `json:"us_min"`
	Bytes         int64          `json:"bytes"`
	GBps          float64        `json:"gbps"`
	FloorUs       float64        `json:"floor_us"`
	Efficiency    float64        `json:"efficiency"`
	BandwidthGBps float64        `json:"bandwidth_gbps"`
	GPU           string         `json:"gpu"`
	Error         string         `json:"error,omitempty"`
}

// Runner times jobs on a card.
type Runner interface {
	Run(ctx context.Context, jobs []Job) ([]Result, error)
}

// PythonRunner runs python -m vxpu.measure run over a jobs file.
type PythonRunner struct {
	Python string   // the interpreter (default "python")
	Env    []string // extra environment (PYTHONPATH, CUDA_VISIBLE_DEVICES)
	Device string   // the CUDA device (default cuda:0)
	Keep   string   // where to keep the runner's result lines (default: a temporary file)
	Log    func(format string, args ...any)
}

func (p PythonRunner) Run(ctx context.Context, jobs []Job) ([]Result, error) {
	dir, err := os.MkdirTemp("", "vxpu-measure-")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil && p.Log != nil {
			p.Log("leaving %s: %v", dir, err)
		}
	}()
	data, err := json.Marshal(jobs)
	if err != nil {
		return nil, err
	}
	jobsPath := dir + "/jobs.json"
	if err := os.WriteFile(jobsPath, data, 0o644); err != nil {
		return nil, err
	}
	python := p.Python
	if python == "" {
		python = "python"
	}
	device := p.Device
	if device == "" {
		device = "cuda:0"
	}
	// the runner appends a line per job to a file, so a run that dies keeps what it measured
	outPath := dir + "/results.jsonl"
	if p.Keep != "" {
		outPath = p.Keep
	}
	cmd := exec.CommandContext(ctx, python, "-m", "vxpu.measure", "run", jobsPath, "--device", device, "--out", outPath)
	cmd.Env = append(os.Environ(), p.Env...)
	var stderr bytes.Buffer
	cmd.Stdout = os.Stderr
	cmd.Stderr = &stderr
	runErr := cmd.Run()
	lines, readErr := os.ReadFile(outPath)
	if readErr != nil && runErr != nil {
		return nil, fmt.Errorf("python -m vxpu.measure run: %w: %s", runErr, lastLines(stderr.String(), 5))
	}
	results, parseErr := ParseResults(lines)
	if parseErr != nil {
		return nil, parseErr
	}
	if runErr != nil && len(results) == 0 {
		return nil, fmt.Errorf("python -m vxpu.measure run: %w: %s", runErr, lastLines(stderr.String(), 5))
	}
	if p.Log != nil {
		p.Log("measured %d of %d jobs%s", len(results), len(jobs), map[bool]string{true: " (the runner exited early: " + lastLines(stderr.String(), 2) + ")", false: ""}[runErr != nil])
	}
	return results, nil
}

// ParseResults reads the runner's output: one JSON object a line.
func ParseResults(out []byte) ([]Result, error) {
	var results []Result
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var r Result
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, fmt.Errorf("the measuring step's output: %w: %s", err, line)
		}
		results = append(results, r)
	}
	return results, sc.Err()
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, " | ")
}

// MeasureOptions shape a sweep.
type MeasureOptions struct {
	SMs       int      // the card's streaming multiprocessors (the K split's bound); 0: unknown, no bound
	Templates []string // only these templates (empty: all)
	MaxRows   int      // only classes of at most this many rows (0: all)
	NoStages  bool     // skip the second pass over the pipeline depth
}

func fastest(rs []Result) *Result {
	var best *Result
	for i := range rs {
		if rs[i].Error == "" && (best == nil || rs[i].Us < best.Us) {
			best = &rs[i]
		}
	}
	return best
}

// Measure sweeps the classes' blocks through the runner and returns the
// table's block entries (one per class, the fastest measured) and the
// explanation lines. A class whose every candidate failed keeps no entry
// and says so.
func Measure(ctx context.Context, classes map[string]ClassRun, runner Runner, opts MeasureOptions) ([]BlockEntry, []string, error) {
	keys := make([]string, 0, len(classes))
	for k := range classes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var jobs []Job
	parsed := map[string]Class{}
	for _, key := range keys {
		c, err := ParseClass(key)
		if err != nil {
			return nil, nil, err
		}
		if len(opts.Templates) > 0 && !contains(opts.Templates, c.Template) {
			continue
		}
		if opts.MaxRows > 0 && c.Rows > opts.MaxRows {
			continue
		}
		parsed[key] = c
		for _, b := range Candidates(c, classes[key].Blocks, opts.SMs) {
			jobs = append(jobs, Job{Shapes: key, Blocks: b})
		}
	}
	if len(jobs) == 0 {
		return nil, nil, fmt.Errorf("no classes to measure")
	}
	results, err := runner.Run(ctx, jobs)
	if err != nil {
		return nil, nil, err
	}
	byClass := map[string][]Result{}
	for _, r := range results {
		byClass[r.Shapes] = append(byClass[r.Shapes], r)
	}
	// the second pass: the pipeline depth (stages 2 and 4) on each class's winner
	if !opts.NoStages {
		var more []Job
		for _, key := range keys {
			if best := fastest(byClass[key]); best != nil {
				for _, st := range []int{2, 4} {
					b := map[string]int{}
					for k, v := range best.Blocks {
						b[k] = v
					}
					b["stages"] = st
					more = append(more, Job{Shapes: key, Blocks: b})
				}
			}
		}
		if len(more) > 0 {
			second, err := runner.Run(ctx, more)
			if err != nil {
				return nil, nil, err
			}
			for _, r := range second {
				byClass[r.Shapes] = append(byClass[r.Shapes], r)
			}
		}
	}
	var entries []BlockEntry
	var explain []string
	for _, key := range keys {
		c, ok := parsed[key]
		if !ok {
			continue
		}
		rs := byClass[key]
		var best *Result
		var ranUs float64
		failed := 0
		ranKey := ""
		if ran := classes[key].Blocks; ran != nil {
			ranKey = blockKey(map[string]int{"BN": ran["BN"], "BK": ran["BK"], "KSPLIT": max(1, ran["KSPLIT"]), "warps": orDefault(ran["warps"], 4), "stages": orDefault(ran["stages"], 3)})
		}
		for i := range rs {
			r := &rs[i]
			if r.Error != "" {
				failed++
				continue
			}
			if blockKey(r.Blocks) == ranKey {
				ranUs = r.Us
			}
			if best == nil || r.Us < best.Us {
				best = r
			}
		}
		if best == nil {
			explain = append(explain, fmt.Sprintf("%s: no candidate ran (%d failed); the engine's blocks stay", key, failed))
			continue
		}
		why := fmt.Sprintf("%d candidates, %d failed", len(rs), failed)
		if ranUs > 0 {
			why += fmt.Sprintf("; the engine's own blocks %.1f us", ranUs)
		}
		entries = append(entries, BlockEntry{Shapes: key, Blocks: best.Blocks, Us: best.Us, Bytes: best.Bytes,
			FloorUs: best.FloorUs, Efficiency: best.Efficiency, Why: why})
		gain := ""
		if ranUs > 0 {
			gain = fmt.Sprintf(", %.0f%% of the engine's %.1f us", 100*best.Us/ranUs, ranUs)
		}
		explain = append(explain, fmt.Sprintf("%s (%d a step): %s %.1f us, floor %.1f, %.0f%% of bandwidth%s, %d candidates",
			key, classes[key].AStep(), blockKey(best.Blocks), best.Us, best.FloorUs, 100*best.Efficiency, gain, len(rs)-failed))
		_ = c
	}
	return entries, explain, nil
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// StepFloor sums a record's classes: the bytes the GEMVs move a step and
// the microseconds the measured blocks take, from the table's entries
// (classes the table does not name count at no measured time).
func StepFloor(classes map[string]ClassRun, entries []BlockEntry) (bytesAStep int64, usAStep float64) {
	byKey := map[string]BlockEntry{}
	for _, e := range entries {
		byKey[e.Shapes] = e
	}
	for key, run := range classes {
		e, ok := byKey[key]
		if !ok {
			continue
		}
		bytesAStep += e.Bytes * int64(run.AStep())
		usAStep += e.Us * float64(run.AStep())
	}
	return bytesAStep, usAStep
}

// BlockMismatch is a class the record ran under other blocks than the table's.
type BlockMismatch struct {
	Shapes string
	Ran    map[string]int
	Want   map[string]int
}

func (m BlockMismatch) String() string {
	return fmt.Sprintf("%s ran %s, the table names %s", m.Shapes, blockKey(m.Ran), blockKey(m.Want))
}

// CheckBlocks compares the blocks a record ran per class against the
// table's entries (BN, BK, KSPLIT, warps, stages; BM is the rows' tile and
// not the table's to set).
func CheckBlocks(t *Table, r *Record) ([]BlockMismatch, error) {
	if len(t.Blocks) == 0 {
		return nil, nil
	}
	classes, err := Classes(r)
	if err != nil {
		return nil, err
	}
	var out []BlockMismatch
	for _, e := range t.Blocks {
		run, ok := classes[e.Shapes]
		if !ok {
			continue
		}
		for name, want := range e.Blocks {
			if run.Blocks[name] != want {
				out = append(out, BlockMismatch{Shapes: e.Shapes, Ran: run.Blocks, Want: e.Blocks})
				break
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Shapes < out[j].Shapes })
	return out, nil
}
