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
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

// Supported PT2 archive format and versions.
const (
	ExpectedArchiveFormat  = "pt2"
	ExpectedArchiveVersion = "0"
	ExpectedSchemaMajor    = 8
	ExpectedSchemaMinor    = 20
)

// HostOps lists the canonical operator targets that cut a segment in the exported graph.
// The planner owns this set because partitioning the model into compiled units
// versus host execution is a structural planning decision.
var HostOps = map[string]bool{
	"vxpu::moe_experts": true,
	"vxpu::engram_rows": true,
}

// IsHostOp reports whether the canonical operator name cuts the graph into segments.
func IsHostOp(op string) bool {
	return HostOps[op]
}

// HostOpNames returns the sorted list of canonical operator names that cut segments.
func HostOpNames() []string {
	names := make([]string, 0, len(HostOps))
	for op := range HostOps {
		names = append(names, op)
	}
	sort.Strings(names)
	return names
}

// BuildModelTree constructs a ModelTree proto message from a model name and its segments.
func BuildModelTree(modelName string, segments []*planpb.Segment) *planpb.ModelTree {
	return &planpb.ModelTree{
		Model:    modelName,
		Segments: segments,
	}
}

type pt2Model struct {
	SchemaVersion struct {
		Major int `json:"major"`
		Minor int `json:"minor"`
	} `json:"schema_version"`
	GraphModule struct {
		Graph struct {
			Nodes []pt2Node `json:"nodes"`
		} `json:"graph"`
	} `json:"graph_module"`
}

type pt2Node struct {
	Name     string             `json:"name"`
	Target   string             `json:"target"`
	Inputs   []pt2NamedArgument `json:"inputs"`
	Metadata map[string]any     `json:"metadata"`
}

type pt2NamedArgument struct {
	Name string      `json:"name"`
	Arg  pt2Argument `json:"arg"`
}

type pt2Argument struct {
	AsGraph json.RawMessage `json:"as_graph"`
}

// ParseOperator extracts the canonical operator name from an exported graph node target.
// It returns an empty string if target is not a vxpu operator.
// If target has the vxpu prefix but an empty operator name, it returns an error.
func ParseOperator(target string) (string, error) {
	target = strings.TrimPrefix(target, "torch.ops.")
	if !strings.HasPrefix(target, "vxpu.") {
		return "", nil
	}
	parts := strings.Split(target, ".")
	if len(parts) < 2 || parts[1] == "" {
		return "", fmt.Errorf("malformed operator target %q: empty operator name", target)
	}
	return "vxpu::" + parts[1], nil
}

// ParseSite extracts the module path from a node's nn_module_stack metadata.
func ParseSite(stack string) string {
	if stack == "" {
		return ""
	}
	entries := strings.Split(stack, ";")
	for i := len(entries) - 1; i >= 0; i-- {
		entry := strings.TrimSpace(entries[i])
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ",")
		if len(parts) >= 2 {
			return parts[1]
		}
	}
	return ""
}

// ReadSegments opens an artifact directory and partitions its exported decode graph
// into segments and custom operator call sites.
func ReadSegments(artifactDir string) (_ []*planpb.Segment, err error) {
	pt2Path := filepath.Join(artifactDir, "decode.pt2")
	zr, err := zip.OpenReader(pt2Path)
	if err != nil {
		return nil, fmt.Errorf("open archive: %w", err)
	}
	defer func() {
		if cerr := zr.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close archive: %w", cerr)
		}
	}()

	var formatFile, versionFile, modelFile *zip.File
	for _, f := range zr.File {
		name := f.Name
		if name == "archive_format" || strings.HasSuffix(name, "/archive_format") {
			formatFile = f
		} else if name == "archive_version" || strings.HasSuffix(name, "/archive_version") {
			versionFile = f
		} else if name == "models/model.json" || strings.HasSuffix(name, "/models/model.json") {
			modelFile = f
		}
	}

	if formatFile == nil {
		return nil, fmt.Errorf("archive missing archive_format")
	}
	formatBytes, err := readZipEntry(formatFile)
	if err != nil {
		return nil, fmt.Errorf("read archive_format: %w", err)
	}
	format := strings.TrimSpace(string(formatBytes))
	if format != ExpectedArchiveFormat {
		return nil, fmt.Errorf("unsupported archive format %q (expected %q)", format, ExpectedArchiveFormat)
	}

	if versionFile == nil {
		return nil, fmt.Errorf("archive missing archive_version")
	}
	versionBytes, err := readZipEntry(versionFile)
	if err != nil {
		return nil, fmt.Errorf("read archive_version: %w", err)
	}
	version := strings.TrimSpace(string(versionBytes))
	if version != ExpectedArchiveVersion {
		return nil, fmt.Errorf("unsupported archive version %q (expected %q)", version, ExpectedArchiveVersion)
	}

	if modelFile == nil {
		return nil, fmt.Errorf("archive missing model json")
	}
	modelBytes, err := readZipEntry(modelFile)
	if err != nil {
		return nil, fmt.Errorf("read model json: %w", err)
	}
	var model pt2Model
	if err := json.Unmarshal(modelBytes, &model); err != nil {
		return nil, fmt.Errorf("parse model json: %w", err)
	}
	if model.SchemaVersion.Major != ExpectedSchemaMajor || model.SchemaVersion.Minor != ExpectedSchemaMinor {
		return nil, fmt.Errorf("unsupported schema version %d.%d (expected %d.%d)",
			model.SchemaVersion.Major, model.SchemaVersion.Minor, ExpectedSchemaMajor, ExpectedSchemaMinor)
	}

	var (
		segments     []*planpb.Segment
		segmentsByID = make(map[int]*planpb.Segment)
		hostCount    int
	)

	for _, node := range model.GraphModule.Graph.Nodes {
		for _, inp := range node.Inputs {
			if len(inp.Arg.AsGraph) > 0 && string(inp.Arg.AsGraph) != "null" {
				return nil, fmt.Errorf("node %q has nested graph: subgraphs are not supported", node.Name)
			}
		}

		op, err := ParseOperator(node.Target)
		if err != nil {
			return nil, fmt.Errorf("parse node %q operator: %w", node.Name, err)
		}

		var partID int
		if op != "" && IsHostOp(op) {
			partID = 2*hostCount + 1
			hostCount++
		} else {
			partID = 2 * hostCount
		}

		seg, ok := segmentsByID[partID]
		if !ok {
			seg = &planpb.Segment{
				Name: fmt.Sprintf("submod_%d", partID),
				Host: partID%2 == 1,
			}
			segmentsByID[partID] = seg
			segments = append(segments, seg)
		}

		if op != "" {
			var stack string
			if node.Metadata != nil {
				if s, ok := node.Metadata["nn_module_stack"].(string); ok {
					stack = s
				}
			}
			seg.CallSites = append(seg.CallSites, &planpb.CallSite{
				Op:   op,
				Node: node.Name,
				Site: ParseSite(stack),
			})
		}
	}

	return segments, nil
}

func readZipEntry(f *zip.File) (_ []byte, err error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer func() {
		if cerr := rc.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close entry: %w", cerr)
		}
	}()
	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, err
	}
	return data, nil
}
