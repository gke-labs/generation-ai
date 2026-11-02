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

// Package kernelplan chooses the kernels: which implementation body each
// op call site of a model runs on a given GPU, and the kernels' knobs
// (hero run 8).
//
// The engine (Python) measures and records; this package decides. An
// engine run writes a Record (vxpu-kernel-record/1): per call site the
// candidate bodies, what the lowering measured, what it chose and why, and
// the kernels it inlined, with the GPU and the step's statistics. The
// planner reads records and writes a Table (vxpu-kernel-table/1): per call
// site the body to run, plus the knobs. An engine given a table applies it
// exactly and measures nothing (VXPU_PLAN_TABLE; python/vxpu/plan.py), and
// writes a record of what it ran, which Check compares against the table:
// a site that ran another body is the failure a planner cannot have
// silently.
//
// The policy today is the measured-fastest body per site, with a launch
// credited to a body the compiler fuses (as the engine's own costing
// does), the native-format knobs decided from the GPU's compute
// capability, and the other knobs from the record's sweeps where it has
// them. The search is a function of the record, so a costlier policy
// (regions costed whole, the graph cut in Go) slots in behind the same
// table.
package kernelplan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	TableFormat  = "vxpu-kernel-table/1"
	RecordFormat = "vxpu-kernel-record/1"
)

// Table is what the planner hands an engine.
type Table struct {
	Format string `json:"format"`
	// GPU names the card the table was planned for; an engine on another
	// card refuses it. Empty: any.
	GPU string `json:"gpu,omitempty"`
	// Knobs are triton_ops' hand-set numbers (GEMV_ROWS_MAX, FP4_DIRECT_MAX,
	// FP4_NATIVE, ...): integers and booleans, as the engine checks them.
	Knobs map[string]any `json:"knobs,omitempty"`
	Sites []Entry        `json:"sites"`
	// Blocks are the GEMV templates' tiles per shape class, chosen by the
	// measuring step (measure.go); the engine applies them per class
	// (triton_ops.gemv_blocks).
	Blocks []BlockEntry `json:"blocks,omitempty"`
	// Explain says why, line by line; the engine does not read it.
	Explain []string `json:"explain,omitempty"`
}

// Entry names the body for a call: a site (the module that calls the op,
// model.layers.14.engram.wkv), else the op at these shapes (the engine's
// shape key for the op), else the op everywhere. The engine takes the
// most specific entry that matches.
type Entry struct {
	Op     string `json:"op"`
	Site   string `json:"site,omitempty"`
	Shapes string `json:"shapes,omitempty"`
	Impl   string `json:"impl"`
	Why    string `json:"why,omitempty"`
	// The ledger's fields, filled by the measuring step (vxpu kernels
	// measure) or carried over from the record the table was planned
	// from: the bytes the site moves and the microseconds the chosen body
	// took on the table's GPU. Accepted is a recorded reason for a site
	// that runs under the check's efficiency threshold (a long-K GEMV
	// that cannot fill the card at one row); without one the check fails
	// the site.
	Bytes    *SiteBytes `json:"bytes,omitempty"`
	Us       float64    `json:"us,omitempty"`
	Accepted string     `json:"accepted,omitempty"`
}

// SiteBytes is what a call site must move, from the graph alone
// (python/vxpu/lower.py: site_bytes): the parameters it reads, the
// activations it reads and writes, and the state buffers it names (the
// caches, of which an op may read a window, so an upper bound kept apart).
// The floor is Weights plus Activations over the card's bandwidth.
type SiteBytes struct {
	Weights     int64 `json:"weights"`
	Activations int64 `json:"activations"`
	State       int64 `json:"state"`
}

// Floor returns the bytes the floor counts: the parameters and the
// activations, not the state (a window of it is read, not the whole).
func (b *SiteBytes) Floor() int64 {
	if b == nil {
		return 0
	}
	return b.Weights + b.Activations
}

// Record is what an engine writes about a run (plan.Plan.kernel_record).
type Record struct {
	Format   string         `json:"format"`
	Hardware Hardware       `json:"hardware"`
	Choices  map[string]any `json:"choices"`
	Knobs    map[string]any `json:"knobs"`
	Sites    []Site         `json:"sites"`
	// Stats are the step's statistics, by name: kernels a step, crossings,
	// the distinct experts a layer selects, and "sweeps": {knob: {value:
	// microseconds}} where a run swept a knob.
	Stats map[string]json.RawMessage `json:"stats"`
}

// Hardware is the record's machine, as the planner needs it.
type Hardware struct {
	Device        string `json:"device"`
	GPU           string `json:"gpu"`
	GPUCapability string `json:"gpu_capability"`
	Torch         string `json:"torch"`
	CUDA          string `json:"cuda"`
	Triton        string `json:"triton"`
	// BandwidthGBs is the card's memory bandwidth as the engine measured it
	// at load (a device-to-device copy; plan.gpu_bandwidth); the ledger's
	// floors are over it. Zero in an older record: the check takes
	// --bandwidth-gb-s or the card's known figure (knownBandwidth).
	BandwidthGBs float64 `json:"gpu_bandwidth_gb_s,omitempty"`
	// ReadBandwidthGBs is the same card's single-tensor read (a sum over
	// 1 GiB), the figure the ledger's floors use when present: the step's
	// kernels read weights and write little, and the head's GEMV was seen
	// above the copy figure (1,530 against 1,462 GB/s).
	ReadBandwidthGBs float64 `json:"gpu_read_bandwidth_gb_s,omitempty"`
}

// Site is one op call site of the lowered graph.
type Site struct {
	Op     string `json:"op"`
	Site   string `json:"site"`
	Node   string `json:"node"`
	Shapes string `json:"shapes"`
	Layer  *int   `json:"layer"`
	// Chosen is the body the engine inlined (or kept), Source why: "table",
	// "measured", "registry" (the registry's preference order, nothing
	// measured) or "kept" (not lowered; Why says what kept it).
	Chosen     string             `json:"chosen"`
	Source     string             `json:"source"`
	Why        string             `json:"why,omitempty"`
	Candidates []string           `json:"candidates"`
	Fused      []string           `json:"fused"` // candidates the compiler fuses: credited a launch
	MeasuredUs map[string]float64 `json:"measured_us"`
	// Bytes is what the site must move (lower.site_bytes); nil in a record
	// written before the ledger, or for a site without shapes.
	Bytes *SiteBytes `json:"bytes,omitempty"`
	// Kernels are the Triton kernels inlined for the body, as the traced
	// graph names them; empty for a fused body (the compiler's kernels),
	// nil for a kept call.
	Kernels []string `json:"kernels"`
}

// ReadRecord parses a record and checks its format.
func ReadRecord(data []byte) (*Record, error) {
	var r Record
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parsing the kernel record: %w", err)
	}
	if r.Format != RecordFormat {
		return nil, fmt.Errorf("kernel record: format %q, want %q", r.Format, RecordFormat)
	}
	return &r, nil
}

// ReadRecordFile reads a record from path.
func ReadRecordFile(path string) (*Record, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ReadRecord(data)
}

// ReadTable parses a table and checks it the way the engine does: the
// format, the knobs' types, and that no two entries name different bodies
// for the same key.
func ReadTable(data []byte) (*Table, error) {
	var t Table
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, fmt.Errorf("parsing the kernel table: %w", err)
	}
	if t.Format != TableFormat {
		return nil, fmt.Errorf("kernel table: format %q, want %q", t.Format, TableFormat)
	}
	for name, value := range t.Knobs {
		if err := checkKnob(name, value); err != nil {
			return nil, err
		}
	}
	for i, b := range t.Blocks {
		if b.Shapes == "" || len(b.Blocks) == 0 {
			return nil, fmt.Errorf("kernel table: blocks[%d] needs shapes and blocks", i)
		}
		for name, v := range b.Blocks {
			if !blockNames[name] || v < 1 {
				return nil, fmt.Errorf("kernel table: blocks[%d] (%s): %s=%d is not a block the engine knows (BN, BK, KSPLIT, warps, stages, FORM, BLOCK; positive)", i, b.Shapes, name, v)
			}
		}
	}
	seen := map[string]string{}
	for i, e := range t.Sites {
		if e.Op == "" || e.Impl == "" {
			return nil, fmt.Errorf("kernel table: sites[%d] needs op and impl", i)
		}
		k := e.key()
		if prev, ok := seen[k]; ok && prev != e.Impl {
			return nil, fmt.Errorf("kernel table: sites[%d] names %s where an earlier entry named %s for the same %s", i, e.Impl, prev, k)
		}
		seen[k] = e.Impl
	}
	return &t, nil
}

// ReadTableFile reads a table from path.
func ReadTableFile(path string) (*Table, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ReadTable(data)
}

// Write renders the table as the engine reads it, entries in a stable
// order (op, then site, then shapes).
func (t *Table) Write() ([]byte, error) {
	t.Format = TableFormat
	sort.SliceStable(t.Blocks, func(i, j int) bool { return t.Blocks[i].Shapes < t.Blocks[j].Shapes })
	sort.SliceStable(t.Sites, func(i, j int) bool {
		a, b := t.Sites[i], t.Sites[j]
		if a.Op != b.Op {
			return a.Op < b.Op
		}
		if a.Site != b.Site {
			return a.Site < b.Site
		}
		return a.Shapes < b.Shapes
	})
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // shape keys hold "->"
	enc.SetIndent("", " ")
	if err := enc.Encode(t); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Entry returns the table's entry for a call, by the engine's rule: the
// site's own entry, else the op's at these shapes, else the op's alone;
// nil when the table says nothing about it.
func (t *Table) Entry(op, site, shapes string) *Entry {
	var bySite, byShapes, byOp *Entry
	for i := range t.Sites {
		e := &t.Sites[i]
		if e.Op != op {
			continue
		}
		switch {
		case e.Site != "":
			if site != "" && e.Site == site {
				bySite = e
			}
		case e.Shapes != "":
			if shapes != "" && e.Shapes == shapes {
				byShapes = e
			}
		default:
			byOp = e
		}
	}
	if bySite != nil {
		return bySite
	}
	if byShapes != nil {
		return byShapes
	}
	return byOp
}

func (e Entry) key() string {
	switch {
	case e.Site != "":
		return e.Op + " at " + e.Site
	case e.Shapes != "":
		return e.Op + " shaped " + e.Shapes
	}
	return e.Op
}

// blockNames are the GEMV tiles a table may set per shape class (python/vxpu/triton_ops.py: GEMV_BLOCK_NAMES).
var blockNames = map[string]bool{"BN": true, "BK": true, "KSPLIT": true, "warps": true, "stages": true,
	"FORM": true, "BLOCK": true} // FORM, BLOCK: vxpu::allreduce's (1 one-shot, 2 reduce-scatter; elements a program moves)

// Knobs the engine knows, by type: "int" or "bool" (python/vxpu/plan.py:
// KNOBS). A table with another knob, or a wrong type, is refused there.
var knobTypes = map[string]string{
	"GEMV_ROWS_MAX": "int", "FP4_DIRECT_MAX": "int", "FFN_FRONT_EB": "int", "FFN_FRONT_BD": "int",
	"FRONT_SPLIT_BD": "int", "FRONT_SPLIT_EB": "int", "FP4_KSPLIT": "int", "FP4_NATIVE": "bool", "FP4_DIRECT": "bool",
	"MXFP8_RECODE": "bool",
	// the direct fp4 expert kernels' tiles: rows a program, K an iteration, warps, pipeline stages
	"FP4_GU_BN": "int", "FP4_GU_BK": "int", "FP4_GU_WARPS": "int", "FP4_GU_STAGES": "int",
	"FP4_DN_BN": "int", "FP4_DN_BK": "int", "FP4_DN_WARPS": "int", "FP4_DN_STAGES": "int",
	"FOLD_POST": "bool", "FOLD_POST_RB": "int",
	// the direct expert path as one launch a layer (hero run 12), and the programs it asks for per SM
	"FP4_ONE_LAUNCH": "bool", "FP4_ONE_PER_SM": "int", "FP4_ONE_WARPS": "int", "FP4_ONE_STAGES": "int",
	"FP4_PDL": "bool", // the two-launch path's down kernel as a programmatic dependent launch
	// the expert GEMM form (hero run 15): the pair count it starts at and its tiles for gate/up and down
	"FP4_GEMM_MIN_PAIRS": "int",
	"FP4_GG_BM":          "int", "FP4_GG_BN": "int", "FP4_GG_BK": "int", "FP4_GG_WARPS": "int", "FP4_GG_STAGES": "int",
	"FP4_GD_BM": "int", "FP4_GD_BN": "int", "FP4_GD_BK": "int", "FP4_GD_WARPS": "int", "FP4_GD_STAGES": "int",
	// vxpu::allreduce (hero run 14): the reduce-scatter form forced, the rows it starts at, the elements a program moves
	"ALLREDUCE_SCATTER": "bool", "ALLREDUCE_SCATTER_MIN_ROWS": "int", "ALLREDUCE_BLOCK": "int",
}

func checkKnob(name string, value any) error {
	kind, ok := knobTypes[name]
	if !ok {
		return fmt.Errorf("kernel table: unknown knob %q", name)
	}
	switch kind {
	case "bool":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("kernel table: knob %s must be true or false, got %v", name, value)
		}
	case "int":
		f, ok := value.(float64) // encoding/json's number
		if !ok || f != float64(int64(f)) {
			if _, isInt := value.(int); !isInt {
				return fmt.Errorf("kernel table: knob %s must be an integer, got %v", name, value)
			}
		}
	}
	return nil
}

// capabilityMajor parses "12.0" to 12; 0 when unknown.
func capabilityMajor(capability string) int {
	major, _, _ := strings.Cut(capability, ".")
	n, err := strconv.Atoi(major)
	if err != nil {
		return 0
	}
	return n
}
