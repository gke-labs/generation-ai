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

// Package kernelplan manages the planner's physical execution plans,
// per-site kernel strategies, and execution run records.
package kernelplan

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	PlanFormat          = "vxpu-plan/1"
	RecordFormat        = "vxpu-run-record/1"
	RecordSchemaVersion = 1

	ClassDenseGEMV = "dense GEMV"
	ClassExpertOp  = "expert op"
	ClassSumSite   = "sum site"
	ClassAttention = "attention"
	ClassDrafter   = "drafter"
)

// DefaultRowCounts is the standard row count ladder for plan emission.
var DefaultRowCounts = []int{1, 5, 8, 16, 32, 40, 64}

// Site represents a single call site in the model's computation graph.
type Site struct {
	ID        string `json:"id"`
	Class     string `json:"class"`
	Shape     []int  `json:"shape"`
	RowCounts []int  `json:"row_counts"`
	Strategy  string `json:"strategy"`
}

// Plan is the physical plan emitted for an artifact.
type Plan struct {
	Format      string            `json:"format"`
	ArtifactMD5 string            `json:"artifact_md5"`
	TreeCommit  string            `json:"tree_commit"`
	Sites       []Site            `json:"sites"`
	Strategies  map[string]string `json:"strategies"`
	Switches    map[string]string `json:"switches,omitempty"`
}

// RunRecord is the record written by the executor at the end of a run.
type RunRecord struct {
	Format            string             `json:"format"`
	SchemaVersion     int                `json:"schema_version"`
	PlanHash          string             `json:"plan_hash"`
	TreeCommit        string             `json:"tree_commit"`
	ArtifactMD5       string             `json:"artifact_md5"`
	KnobsInEffect     map[string]string  `json:"knobs_in_effect"`
	MedianStepMs      float64            `json:"median_step_ms"`
	TokensAStep       float64            `json:"tokens_a_step"`
	TokensPerStep     float64            `json:"tokens_per_step,omitempty"`
	Checks            map[string]any     `json:"checks"`
	PerCardPeakMemory map[string]int64   `json:"per_card_peak_memory,omitempty"`
	BuildTime         float64            `json:"build_time"`
	BuildTimeS        float64            `json:"build_time_s,omitempty"`
	PerClassTimes     map[string]float64 `json:"per_class_times,omitempty"`
	Timestamp         string             `json:"timestamp,omitempty"`
}

// ComputeArtifactMD5 computes a checksum over the artifact's manifest and binding.
func ComputeArtifactMD5(manifestBytes, bindingBytes []byte) string {
	h := md5.New()
	h.Write(manifestBytes)
	h.Write(bindingBytes)
	return hex.EncodeToString(h.Sum(nil))
}

// GetTreeCommit returns the git commit of the current working tree.
func GetTreeCommit() string {
	if v := os.Getenv("GIT_COMMIT"); v != "" {
		return v
	}
	if v := os.Getenv("TREE_COMMIT"); v != "" {
		return v
	}
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// HashPlan computes the canonical SHA-256 hash of a Plan.
func HashPlan(p *Plan) (string, error) {
	data, err := json.Marshal(p)
	if err != nil {
		return "", fmt.Errorf("marshal plan: %w", err)
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:16]), nil
}

// Emit creates a Plan by inspecting the artifact directory.
func Emit(artifactDir string, rowCounts []int) (*Plan, error) {
	manifestPath := filepath.Join(artifactDir, "manifest.json")
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}
	bindingPath := filepath.Join(artifactDir, "binding.json")
	bindingBytes, err := os.ReadFile(bindingPath)
	if err != nil {
		return nil, fmt.Errorf("read binding %s: %w", bindingPath, err)
	}
	return EmitFromArtifact(manifestBytes, bindingBytes, GetTreeCommit(), rowCounts)
}

// EmitFromArtifact parses manifest and binding JSON and builds a Plan.
func EmitFromArtifact(manifestBytes, bindingBytes []byte, treeCommit string, rowCounts []int) (*Plan, error) {
	if len(rowCounts) == 0 {
		rowCounts = append([]int{}, DefaultRowCounts...)
	}

	var manifest struct {
		Tensors map[string]struct {
			Length int64  `json:"length"`
			Shape  []int  `json:"shape"`
			DType  string `json:"dtype"`
		} `json:"tensors"`
		Config map[string]any `json:"config"`
	}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	var binding struct {
		Bound map[string]struct {
			Length int64  `json:"length"`
			Shape  []int  `json:"shape"`
			DType  string `json:"dtype"`
		} `json:"bound"`
		Stacked map[string]struct {
			Parts [][]string `json:"parts"`
		} `json:"stacked"`
		External struct {
			Experts map[string]map[string]struct {
				Parts [][]string `json:"parts"`
			} `json:"experts"`
			Engram map[string]any `json:"engram"`
		} `json:"external"`
	}
	if err := json.Unmarshal(bindingBytes, &binding); err != nil {
		return nil, fmt.Errorf("parse binding: %w", err)
	}

	artifactMD5 := ComputeArtifactMD5(manifestBytes, bindingBytes)
	sites := listSites(manifest.Config, manifest.Tensors, binding.Bound, binding.Stacked, binding.External.Experts, binding.External.Engram, rowCounts)

	strategies := make(map[string]string, len(sites))
	for _, s := range sites {
		strategies[s.ID] = s.Strategy
	}

	switches := map[string]string{
		"VXPU_FUSED_EXPERTS":              "1",
		"VXPU_ALLREDUCE_SCATTER_MIN_ROWS": "16",
		"VXPU_ALLREDUCE_PUSH_BF16":        "0",
	}

	return &Plan{
		Format:      PlanFormat,
		ArtifactMD5: artifactMD5,
		TreeCommit:  treeCommit,
		Sites:       sites,
		Strategies:  strategies,
		Switches:    switches,
	}, nil
}

func listSites(
	config map[string]any,
	manifestTensors map[string]struct {
		Length int64  `json:"length"`
		Shape  []int  `json:"shape"`
		DType  string `json:"dtype"`
	},
	bound map[string]struct {
		Length int64  `json:"length"`
		Shape  []int  `json:"shape"`
		DType  string `json:"dtype"`
	},
	stacked map[string]struct {
		Parts [][]string `json:"parts"`
	},
	experts map[string]map[string]struct {
		Parts [][]string `json:"parts"`
	},
	engram map[string]any,
	rowCounts []int,
) []Site {
	var sites []Site

	numLayers := getInt(config, "num_hidden_layers", "n_layers")
	hiddenSize := getInt(config, "hidden_size", "dim")
	numHeads := getInt(config, "num_attention_heads", "n_heads")
	headDim := getInt(config, "head_dim")
	if headDim == 0 && numHeads > 0 && hiddenSize > 0 {
		headDim = hiddenSize / numHeads
	}

	// 1. Dense GEMVs
	// Collect bound weight tensors that participate in linear ops.
	boundNames := make([]string, 0, len(bound))
	for name := range bound {
		boundNames = append(boundNames, name)
	}
	sort.Strings(boundNames)

	for _, name := range boundNames {
		b := bound[name]
		shape := b.Shape
		if len(shape) == 0 {
			if mt, ok := manifestTensors[name]; ok && len(mt.Shape) > 0 {
				shape = mt.Shape
			}
		}
		if len(shape) == 0 {
			shape = []int{hiddenSize, hiddenSize}
		}
		sites = append(sites, Site{
			ID:        name,
			Class:     ClassDenseGEMV,
			Shape:     shape,
			RowCounts: rowCounts,
			Strategy:  "default",
		})
	}

	// 2. Expert Ops
	// Expert layers from binding.external.experts
	expertLayers := make([]int, 0, len(experts))
	for k := range experts {
		l, err := strconv.Atoi(k)
		if err != nil {
			continue
		}
		expertLayers = append(expertLayers, l)
	}
	sort.Ints(expertLayers)

	numExperts := getInt(config, "num_routed_experts", "n_routed_experts")
	moeInterDim := getInt(config, "moe_inter_dim", "intermediate_size")
	for _, l := range expertLayers {
		shape := []int{numExperts, hiddenSize, moeInterDim}
		if numExperts == 0 {
			// Infer from parts count if available
			banks := experts[strconv.Itoa(l)]
			for _, spec := range banks {
				if len(spec.Parts) > 0 {
					shape[0] = len(spec.Parts)
					break
				}
			}
		}
		sites = append(sites, Site{
			ID:        fmt.Sprintf("layer.%d.experts", l),
			Class:     ClassExpertOp,
			Shape:     shape,
			RowCounts: rowCounts,
			Strategy:  "default",
		})
	}

	// 3. Attention Sites
	// If numLayers > 0, generate attention site per layer.
	// Otherwise look for attention-related tensors in bound.
	attnSeen := make(map[int]bool)
	if numLayers > 0 {
		for l := 0; l < numLayers; l++ {
			sites = append(sites, Site{
				ID:        fmt.Sprintf("model.layers.%d.self_attn", l),
				Class:     ClassAttention,
				Shape:     []int{numHeads, headDim},
				RowCounts: rowCounts,
				Strategy:  "default",
			})
			attnSeen[l] = true
		}
	} else {
		for _, name := range boundNames {
			if strings.Contains(name, "self_attn") || strings.Contains(name, "attn") {
				l := extractLayerIndex(name)
				if l >= 0 && !attnSeen[l] {
					sites = append(sites, Site{
						ID:        fmt.Sprintf("model.layers.%d.self_attn", l),
						Class:     ClassAttention,
						Shape:     []int{numHeads, headDim},
						RowCounts: rowCounts,
						Strategy:  "default",
					})
					attnSeen[l] = true
				}
			}
		}
	}

	// 4. Sum Sites
	// Attention sum and MLP sum per layer.
	sumSeen := make(map[int]bool)
	if numLayers > 0 {
		for l := 0; l < numLayers; l++ {
			sites = append(sites, Site{
				ID:        fmt.Sprintf("model.layers.%d.self_attn.sum", l),
				Class:     ClassSumSite,
				Shape:     []int{hiddenSize},
				RowCounts: rowCounts,
				Strategy:  "default",
			})
			sites = append(sites, Site{
				ID:        fmt.Sprintf("model.layers.%d.mlp.sum", l),
				Class:     ClassSumSite,
				Shape:     []int{hiddenSize},
				RowCounts: rowCounts,
				Strategy:  "default",
			})
			sumSeen[l] = true
		}
	} else {
		for l := range attnSeen {
			sites = append(sites, Site{
				ID:        fmt.Sprintf("model.layers.%d.self_attn.sum", l),
				Class:     ClassSumSite,
				Shape:     []int{hiddenSize},
				RowCounts: rowCounts,
				Strategy:  "default",
			})
			sites = append(sites, Site{
				ID:        fmt.Sprintf("model.layers.%d.mlp.sum", l),
				Class:     ClassSumSite,
				Shape:     []int{hiddenSize},
				RowCounts: rowCounts,
				Strategy:  "default",
			})
			sumSeen[l] = true
		}
	}

	// 5. Drafter Sites
	// Engram layers or speculative drafting.
	engramLayers := make([]int, 0, len(engram))
	for k := range engram {
		l, err := strconv.Atoi(k)
		if err != nil {
			continue
		}
		engramLayers = append(engramLayers, l)
	}
	sort.Ints(engramLayers)

	engramVocab := getInt(config, "engram_vocab_size", "engram_num_embeddings")
	for _, l := range engramLayers {
		sites = append(sites, Site{
			ID:        fmt.Sprintf("model.layers.%d.engram", l),
			Class:     ClassDrafter,
			Shape:     []int{engramVocab, hiddenSize},
			RowCounts: rowCounts,
			Strategy:  "default",
		})
	}

	return sites
}

func getInt(config map[string]any, keys ...string) int {
	if config == nil {
		return 0
	}
	for _, k := range keys {
		if v, ok := config[k]; ok {
			switch n := v.(type) {
			case float64:
				return int(n)
			case int:
				return n
			case int64:
				return int(n)
			case json.Number:
				i, err := n.Int64()
				if err != nil {
					return 0
				}
				return int(i)
			}
		}
	}
	return 0
}

func extractLayerIndex(name string) int {
	parts := strings.Split(name, ".")
	for i, p := range parts {
		if (p == "layers" || p == "layer") && i+1 < len(parts) {
			idx, err := strconv.Atoi(parts[i+1])
			if err != nil {
				continue
			}
			return idx
		}
	}
	return -1
}

// ValidateRecord validates a RunRecord ensuring all required fields are present.
func ValidateRecord(data []byte) (*RunRecord, error) {
	var rec RunRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("unmarshal run record: %w", err)
	}

	if rec.Format == "" && rec.SchemaVersion == 0 {
		return nil, errors.New("record schema is unversioned: missing format and schema_version")
	}
	if rec.Format != "" && rec.Format != RecordFormat {
		return nil, fmt.Errorf("unexpected record format %q, expected %q", rec.Format, RecordFormat)
	}
	if rec.PlanHash == "" {
		return nil, errors.New("missing plan_hash")
	}
	if rec.TreeCommit == "" {
		return nil, errors.New("missing tree_commit")
	}
	if rec.ArtifactMD5 == "" {
		return nil, errors.New("missing artifact_md5")
	}
	if rec.KnobsInEffect == nil {
		return nil, errors.New("missing knobs_in_effect")
	}
	if rec.Checks == nil {
		return nil, errors.New("missing checks field")
	}
	return &rec, nil
}

// AppendLedger appends a validated RunRecord to the specified ledger file as a single JSON line.
func AppendLedger(ledgerPath string, rec *RunRecord) error {
	if rec.Timestamp == "" {
		rec.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal run record: %w", err)
	}

	if dir := filepath.Dir(ledgerPath); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}

	f, err := os.OpenFile(ledgerPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open ledger %s: %w", ledgerPath, err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && !errors.Is(cerr, os.ErrClosed) {
			// file was already closed
		}
	}()

	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write ledger line: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close ledger %s: %w", ledgerPath, err)
	}
	return nil
}
