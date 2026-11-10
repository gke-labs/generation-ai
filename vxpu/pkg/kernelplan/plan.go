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

// Package kernelplan manages the planner's physical execution plans and
// per-site kernel strategies.
package kernelplan

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	// PlanFormat is the supported physical plan file format version.
	PlanFormat = "vxpu-plan/1"

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

func newSite(id, class string, shape, rowCounts []int) Site {
	return Site{
		ID:        id,
		Class:     class,
		Shape:     shape,
		RowCounts: rowCounts,
		Strategy:  "default",
	}
}

// Plan is the physical execution plan emitted for an artifact.
type Plan struct {
	Format      string `json:"format"`
	ArtifactMD5 string `json:"artifact_md5"`
	TreeCommit  string `json:"tree_commit"`
	Sites       []Site `json:"sites"`
}

// TensorMeta describes a tensor's dtype and shape.
type TensorMeta struct {
	DType string `json:"dtype,omitempty"`
	Shape []int  `json:"shape"`
}

// TensorParts represents constituent parts of a stacked tensor bank.
type TensorParts struct {
	Parts [][]string `json:"parts"`
}

// EngramMeta describes the dimensions of an external engram table.
type EngramMeta struct {
	NumEmbeddings int `json:"num_embeddings"`
	Dim           int `json:"dim"`
}

// Manifest represents the structure of manifest.json needed by the planner.
type Manifest struct {
	Tensors map[string]TensorMeta `json:"tensors"`
}

// Binding represents the structure of binding.json needed by the planner.
type Binding struct {
	Bound    map[string]TensorMeta  `json:"bound"`
	Stacked  map[string]TensorParts `json:"stacked,omitempty"`
	External struct {
		Experts map[string]map[string]TensorParts `json:"experts"`
		Engram  map[string]EngramMeta             `json:"engram"`
	} `json:"external"`
}

func computeArtifactMD5(m, b []byte) string {
	h := md5.New()
	h.Write(m)
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

// Parse unmarshals JSON bytes and validates the plan format version.
func Parse(data []byte) (*Plan, error) {
	var p Plan
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("unmarshal plan: %w", err)
	}
	if p.Format != PlanFormat {
		return nil, fmt.Errorf("unsupported plan format %q, expected %q", p.Format, PlanFormat)
	}
	return &p, nil
}

// Emit creates a Plan by inspecting the artifact directory.
func Emit(artifactDir, treeCommit string, rowCounts []int) (*Plan, error) {
	manifestBytes, err := os.ReadFile(filepath.Join(artifactDir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	bindingBytes, err := os.ReadFile(filepath.Join(artifactDir, "binding.json"))
	if err != nil {
		return nil, fmt.Errorf("read binding: %w", err)
	}
	return EmitFromArtifact(manifestBytes, bindingBytes, treeCommit, rowCounts)
}

// EmitFromArtifact parses manifest and binding JSON and builds a Plan.
func EmitFromArtifact(manifestBytes, bindingBytes []byte, treeCommit string, rowCounts []int) (*Plan, error) {
	if treeCommit == "" || treeCommit == "unknown" {
		return nil, errors.New("tree commit is required (refusing \"unknown\")")
	}
	if len(rowCounts) == 0 {
		rowCounts = append([]int{}, DefaultRowCounts...)
	}

	var manifest Manifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	var binding Binding
	if err := json.Unmarshal(bindingBytes, &binding); err != nil {
		return nil, fmt.Errorf("parse binding: %w", err)
	}

	var sites []Site
	sites = append(sites, denseGEMVSites(binding.Bound, rowCounts)...)
	sites = append(sites, expertSites(binding.External.Experts, binding.Stacked, manifest.Tensors, rowCounts)...)
	sites = append(sites, attentionSites(binding.Bound, rowCounts)...)
	sites = append(sites, sumSites(binding.Bound, rowCounts)...)
	sites = append(sites, drafterSites(binding.External.Engram, rowCounts)...)

	return &Plan{
		Format:      PlanFormat,
		ArtifactMD5: computeArtifactMD5(manifestBytes, bindingBytes),
		TreeCommit:  treeCommit,
		Sites:       sites,
	}, nil
}

func isScale(name string, m TensorMeta) bool {
	return strings.Contains(name, "scale") || m.DType == "F8_E8M0"
}

func denseGEMVSites(bound map[string]TensorMeta, rowCounts []int) []Site {
	names := make([]string, 0, len(bound))
	for name, m := range bound {
		if len(m.Shape) == 2 && !isScale(name, m) && !strings.Contains(name, "embed") && !strings.Contains(name, "wte") {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	sites := make([]Site, 0, len(names))
	for _, name := range names {
		sites = append(sites, newSite(name, ClassDenseGEMV, bound[name].Shape, rowCounts))
	}
	return sites
}

func expertSites(externalExperts map[string]map[string]TensorParts, stacked map[string]TensorParts, manifestTensors map[string]TensorMeta, rowCounts []int) []Site {
	layerMap := make(map[int]TensorParts)
	for k, banks := range externalExperts {
		l, err := strconv.Atoi(k)
		if err != nil || l < 0 {
			continue
		}
		var nonScaleKeys []string
		for name := range banks {
			if !strings.Contains(name, "scale") {
				nonScaleKeys = append(nonScaleKeys, name)
			}
		}
		sort.Strings(nonScaleKeys)
		if len(nonScaleKeys) > 0 {
			layerMap[l] = banks[nonScaleKeys[0]]
		}
	}

	stackedByLayer := make(map[int][]string)
	for name := range stacked {
		if !strings.Contains(name, "scale") {
			if l := extractLayerIndex(name); l >= 0 {
				stackedByLayer[l] = append(stackedByLayer[l], name)
			}
		}
	}
	for l, names := range stackedByLayer {
		if _, ok := layerMap[l]; !ok {
			sort.Strings(names)
			layerMap[l] = stacked[names[0]]
		}
	}

	layers := sortedIntKeys(layerMap)
	sites := make([]Site, 0, len(layers))
	for _, l := range layers {
		spec := layerMap[l]
		var shape []int
		if len(spec.Parts) > 0 && len(spec.Parts[0]) > 0 {
			part := spec.Parts[0][0]
			shape = append([]int{len(spec.Parts)}, manifestTensors[part].Shape...)
		}
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.experts", l), ClassExpertOp, shape, rowCounts))
	}
	return sites
}

func attentionSites(bound map[string]TensorMeta, rowCounts []int) []Site {
	layers := boundLayers(bound)
	sites := make([]Site, 0, len(layers))
	for _, l := range layers {
		shape := findOutputProjection(bound, l)
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.attn", l), ClassAttention, shape, rowCounts))
	}
	return sites
}

func sumSites(bound map[string]TensorMeta, rowCounts []int) []Site {
	layers := boundLayers(bound)
	sites := make([]Site, 0, len(layers)*2)
	for _, l := range layers {
		shape := findOutputProjection(bound, l)
		var sumShape []int
		if len(shape) > 0 {
			sumShape = []int{shape[0]}
		}
		sites = append(sites,
			newSite(fmt.Sprintf("layers.%d.attn.sum", l), ClassSumSite, sumShape, rowCounts),
			newSite(fmt.Sprintf("layers.%d.ffn.sum", l), ClassSumSite, sumShape, rowCounts),
		)
	}
	return sites
}

func drafterSites(engram map[string]EngramMeta, rowCounts []int) []Site {
	layers := sortedStringIntKeys(engram)
	sites := make([]Site, 0, len(layers))
	for _, l := range layers {
		e := engram[strconv.Itoa(l)]
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.engram", l), ClassDrafter, []int{e.NumEmbeddings, e.Dim}, rowCounts))
	}
	return sites
}

func findOutputProjection(bound map[string]TensorMeta, layer int) []int {
	var cands []string
	for name, m := range bound {
		if extractLayerIndex(name) == layer && len(m.Shape) == 2 && !isScale(name, m) {
			if strings.Contains(name, "wo") || strings.Contains(name, "o_proj") || strings.Contains(name, "out_proj") {
				cands = append(cands, name)
			}
		}
	}
	if len(cands) == 0 {
		return nil
	}
	sort.Strings(cands)
	return bound[cands[len(cands)-1]].Shape
}

func boundLayers(bound map[string]TensorMeta) []int {
	seen := make(map[int]bool)
	for name := range bound {
		if l := extractLayerIndex(name); l >= 0 {
			seen[l] = true
		}
	}
	return sortedIntKeys(seen)
}

func sortedIntKeys[V any](m map[int]V) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}

func sortedStringIntKeys[V any](m map[string]V) []int {
	seen := make(map[int]bool)
	for k := range m {
		if l, err := strconv.Atoi(k); err == nil && l >= 0 {
			seen[l] = true
		}
	}
	return sortedIntKeys(seen)
}

func extractLayerIndex(name string) int {
	parts := strings.Split(name, ".")
	for i, p := range parts {
		if (p == "layers" || p == "layer") && i+1 < len(parts) {
			idx, err := strconv.Atoi(parts[i+1])
			if err != nil || idx < 0 {
				continue
			}
			return idx
		}
	}
	return -1
}
