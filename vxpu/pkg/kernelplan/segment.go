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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
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

// SaysPackage reports whether a segment's run mode requires an ahead-of-time compiled package.
// Both an ahead-of-time compiled package called directly (SEGMENT_RUN_PACKAGE) and a package
// captured into a CUDA graph (SEGMENT_RUN_CAPTURED) require a compiled package.
// Plain PyTorch execution (SEGMENT_RUN_PLAIN_TORCH) and JIT compilation (SEGMENT_RUN_JIT) require none.
func SaysPackage(run planpb.SegmentRun) bool {
	return run == planpb.SegmentRun_SEGMENT_RUN_PACKAGE || run == planpb.SegmentRun_SEGMENT_RUN_CAPTURED
}

// HasCapture reports whether any segment in segments runs captured into a CUDA graph.
func HasCapture(segments []*planpb.Segment) bool {
	for _, seg := range segments {
		if seg != nil && seg.Run == planpb.SegmentRun_SEGMENT_RUN_CAPTURED {
			return true
		}
	}
	return false
}

// DecideSegmentRun decides how a segment runs on the target machine for the requested goal.
//
// Host segments always run as plain PyTorch (SEGMENT_RUN_PLAIN_TORCH).
// Under PLAN_GOAL_PASS_THROUGH, a model with no custom vxpu operations runs every
// segment as plain PyTorch; for a model the planner has kernels for, each non-host
// segment runs from an ahead-of-time compiled package (captured into a CUDA graph
// on a GPU machine), matching the planner's standard plan.
// Under PLAN_GOAL_SPEED, every non-host segment of any model runs from an
// ahead-of-time compiled package, captured into a CUDA graph when the target
// machine has a GPU and called directly otherwise.
func DecideSegmentRun(seg *planpb.Segment, hasCustomOps bool, m *machines.Machine, goal planpb.PlanGoal) planpb.SegmentRun {
	if m != nil && m.GPU != nil && m.GPUCount > 0 {
		return decideSegmentRunForDevice(seg, hasCustomOps, "cuda", goal)
	}
	return decideSegmentRunForDevice(seg, hasCustomOps, "cpu", goal)
}

func decideSegmentRunForDevice(seg *planpb.Segment, hasCustomOps bool, device string, goal planpb.PlanGoal) planpb.SegmentRun {
	if seg.Host {
		return planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH
	}
	if goal == planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH && !hasCustomOps {
		return planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH
	}
	if device == "cuda" {
		return planpb.SegmentRun_SEGMENT_RUN_CAPTURED
	}
	return planpb.SegmentRun_SEGMENT_RUN_PACKAGE
}

// IsPassThrough reports whether a model plan is pass-through:
// every segment runs as plain torch and no call site names a body.
func IsPassThrough(segments []*planpb.Segment) bool {
	if len(segments) == 0 {
		return false
	}
	for _, seg := range segments {
		if seg == nil || seg.Run != planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH {
			return false
		}
		for _, cs := range seg.CallSites {
			if cs != nil && (cs.Body != "" || cs.GetPlanSite().GetBody() != "") {
				return false
			}
		}
	}
	return true
}

// BuildModelTree constructs a ModelTree proto message from a model name and its segments.
func BuildModelTree(modelName string, segments []*planpb.Segment) *planpb.ModelTree {
	var totalCallSites int32
	for _, seg := range segments {
		totalCallSites += int32(len(seg.CallSites))
	}
	return &planpb.ModelTree{
		Model:          modelName,
		Segments:       segments,
		TotalCallSites: totalCallSites,
	}
}

// siteMatches reports whether the planned kernel body matches the executed bodies recorded during a run.
func siteMatches(planBody string, exec []*planpb.ExecutedBody) bool {
	if len(exec) == 0 {
		return false
	}
	if len(exec) == 1 {
		return exec[0].Body == planBody
	}
	return false
}

// AttachPlan joins a physical execution plan and optional run record to the segments in a ModelTree by site id.
// It populates call site plan choices, execution measurements, segment and tree totals, slowest call sites,
// and unmatched plan sites and measurements. If plan or record provides an artifact checksum and artifactDir
// is non-empty, AttachPlan verifies that the checksum matches the artifact in artifactDir, refusing a mismatched
// plan or record with that reason.
func AttachPlan(tree *planpb.ModelTree, artifactDir string, plan *planpb.PlanFile, record *planpb.PlanFile) error {
	if tree == nil {
		return fmt.Errorf("tree is required")
	}

	var expectedMD5 string
	getExpectedMD5 := func() (string, error) {
		if expectedMD5 != "" {
			return expectedMD5, nil
		}
		if artifactDir == "" {
			return "", nil
		}
		md5Val, err := ArtifactMD5(artifactDir)
		if err != nil {
			return "", fmt.Errorf("compute artifact checksum: %w", err)
		}
		expectedMD5 = md5Val
		return expectedMD5, nil
	}

	if plan != nil && plan.ArtifactMd5 != "" && artifactDir != "" {
		expected, err := getExpectedMD5()
		if err != nil {
			return err
		}
		if plan.ArtifactMd5 != expected {
			return fmt.Errorf("plan artifact checksum %q does not match artifact %q checksum %q", plan.ArtifactMd5, artifactDir, expected)
		}
	}

	if record != nil && record.ArtifactMd5 != "" && artifactDir != "" {
		expected, err := getExpectedMD5()
		if err != nil {
			return err
		}
		if record.ArtifactMd5 != expected {
			return fmt.Errorf("record artifact checksum %q does not match artifact %q checksum %q", record.ArtifactMd5, artifactDir, expected)
		}
	}

	if plan == nil && record == nil {
		return nil
	}

	if plan != nil {
		tree.HasPlan = true
		tree.ArtifactMd5 = plan.ArtifactMd5
		tree.TreeCommit = plan.TreeCommit
		tree.Goal = plan.Goal

		if plan.Knobs != nil {
			m := plan.Knobs.ProtoReflect()
			var knobs []*planpb.PlanKnob
			var rangeErr error
			m.Range(func(fd protoreflect.FieldDescriptor, val protoreflect.Value) bool {
				compiled, err := IsCompiledInKnob(fd)
				if err != nil {
					rangeErr = fmt.Errorf("knob %s: %w", fd.Name(), err)
					return false
				}
				knobs = append(knobs, &planpb.PlanKnob{
					Name:       string(fd.Name()),
					Value:      fmt.Sprint(val.Interface()),
					CompiledIn: compiled,
				})
				return true
			})
			if rangeErr != nil {
				return rangeErr
			}
			sort.Slice(knobs, func(i, j int) bool {
				return knobs[i].Name < knobs[j].Name
			})
			tree.Knobs = knobs
		}
	}

	if record != nil {
		tree.HasMeasurements = true
		tree.Stats = record.Stats
		if tree.ArtifactMd5 == "" {
			tree.ArtifactMd5 = record.ArtifactMd5
		}
		if tree.TreeCommit == "" {
			tree.TreeCommit = record.TreeCommit
		}
	}

	planSitesByID := make(map[string]*planpb.PlanSite)
	if plan != nil {
		for _, s := range plan.Sites {
			planSitesByID[s.Id] = s
		}
	}

	recordSitesByID := make(map[string]*planpb.PlanSite)
	if record != nil {
		for _, s := range record.Sites {
			recordSitesByID[s.Id] = s
		}
	}

	type measuredSite struct {
		node    string
		op      string
		segment string
		timeUs  float64
	}

	var measuredSites []measuredSite
	var plannedCount int32
	var measuredCount int32
	var differingCount int32
	var unmeasuredCount int32
	var totalMeasuredTimeUs float64

	matchedPlanSites := make(map[string]bool)
	matchedRecordSites := make(map[string]bool)

	for _, seg := range tree.Segments {
		var segMeasuredUs float64
		var segHasMeas bool
		for _, cs := range seg.CallSites {
			// Plan choice comes strictly from plan
			if plan != nil {
				if ps, ok := planSitesByID[cs.Node]; ok {
					cs.PlanSite = ps
					cs.Body = ps.Body
					cs.BodyReason = ps.BodyReason
					matchedPlanSites[ps.Id] = true
					if ps.Body != "" {
						plannedCount++
					}
				} else {
					cs.PlanSite = nil
				}
			}

			// Measurements come strictly from record
			if record != nil {
				if rs, ok := recordSitesByID[cs.Node]; ok {
					matchedRecordSites[rs.Id] = true
					cs.InRecord = proto.Bool(true)
					cs.ExecutedBodies = rs.ExecutedBodies
					isMeasured := rs.Measured || len(rs.ExecutedBodies) > 0
					cs.Measured = isMeasured
					if !isMeasured {
						unmeasuredCount++
					} else {
						if plan != nil && cs.PlanSite != nil {
							if !siteMatches(cs.PlanSite.Body, rs.ExecutedBodies) {
								differingCount++
							}
						}
					}
					if rs.Stats != nil && rs.Stats.MeasuredUs != nil {
						cs.Stats = rs.Stats
						us := rs.Stats.GetMeasuredUs()
						segMeasuredUs += us
						segHasMeas = true
						totalMeasuredTimeUs += us
						measuredCount++
						measuredSites = append(measuredSites, measuredSite{
							node:    cs.Node,
							op:      cs.Op,
							segment: seg.Name,
							timeUs:  us,
						})
					}
				}
			}
		}
		if segHasMeas {
			seg.TotalMeasuredTimeUs = &segMeasuredUs
		}
	}
	tree.PlannedCallSites = plannedCount
	tree.MeasuredCallSites = measuredCount
	tree.DifferingCallSites = differingCount
	tree.UnmeasuredCallSites = unmeasuredCount

	// Attach segment run mode from plan segments.
	// A plan without segments is not interpreted.
	if plan != nil && len(plan.Segments) > 0 {
		planSegsByName := make(map[string]*planpb.Segment, len(plan.Segments))
		for _, ps := range plan.Segments {
			if ps != nil && ps.Name != "" {
				planSegsByName[ps.Name] = ps
			}
		}
		for _, seg := range tree.Segments {
			if seg == nil {
				continue
			}
			if ps, ok := planSegsByName[seg.Name]; ok {
				seg.Run = ps.Run
			}
		}
	}

	// Unmatched plan sites come strictly from plan
	if plan != nil {
		for _, s := range plan.Sites {
			if !matchedPlanSites[s.Id] {
				tree.UnmatchedPlanSites = append(tree.UnmatchedPlanSites, s)
			} else if record != nil {
				if _, ok := recordSitesByID[s.Id]; !ok {
					tree.UnmatchedPlanSites = append(tree.UnmatchedPlanSites, s)
				}
			}
		}
	}

	// Unmatched measurements come strictly from record
	if record != nil {
		for _, s := range record.Sites {
			if !matchedRecordSites[s.Id] {
				if len(s.ExecutedBodies) > 0 {
					s.Measured = true
				}
				tree.UnmatchedMeasurements = append(tree.UnmatchedMeasurements, s)
			}
		}
	}

	// Only set TotalMeasuredTimeUs if there are measured call sites on the tree
	if measuredCount > 0 {
		tree.TotalMeasuredTimeUs = &totalMeasuredTimeUs
	}

	if len(measuredSites) > 0 {
		sort.Slice(measuredSites, func(i, j int) bool {
			if measuredSites[i].timeUs != measuredSites[j].timeUs {
				return measuredSites[i].timeUs > measuredSites[j].timeUs
			}
			return measuredSites[i].node < measuredSites[j].node
		})
		limit := 5
		if len(measuredSites) < limit {
			limit = len(measuredSites)
		}
		slowest := make([]*planpb.SlowestCallSite, 0, limit)
		for i := 0; i < limit; i++ {
			ms := measuredSites[i]
			pct := 0.0
			if totalMeasuredTimeUs > 0 {
				pct = (ms.timeUs / totalMeasuredTimeUs) * 100.0
			}
			slowest = append(slowest, &planpb.SlowestCallSite{
				Rank:           int32(i + 1),
				Node:           ms.node,
				Op:             ms.op,
				Segment:        ms.segment,
				MeasuredTimeUs: ms.timeUs,
				Percent:        pct,
			})
		}
		tree.SlowestCallSites = slowest
	}

	return nil
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
		Signature pt2Signature `json:"signature"`
	} `json:"graph_module"`
}

type pt2TensorArg struct {
	Name string `json:"name"`
}

type pt2Signature struct {
	InputSpecs []struct {
		UserInput *struct {
			Arg pt2Argument `json:"arg"`
		} `json:"user_input"`
		Parameter *struct {
			Arg           pt2TensorArg `json:"arg"`
			ParameterName string       `json:"parameter_name"`
		} `json:"parameter"`
		Buffer *struct {
			Arg        pt2TensorArg `json:"arg"`
			BufferName string       `json:"buffer_name"`
		} `json:"buffer"`
		TensorConstant *struct {
			Arg                pt2TensorArg `json:"arg"`
			TensorConstantName string       `json:"tensor_constant_name"`
		} `json:"tensor_constant"`
	} `json:"input_specs"`
	OutputSpecs []struct {
		UserOutput *struct {
			Arg pt2Argument `json:"arg"`
		} `json:"user_output"`
		BufferMutation *struct {
			Arg        pt2TensorArg `json:"arg"`
			BufferName string       `json:"buffer_name"`
		} `json:"buffer_mutation"`
	} `json:"output_specs"`
}

type pt2Node struct {
	Name     string             `json:"name"`
	Target   string             `json:"target"`
	Inputs   []pt2NamedArgument `json:"inputs"`
	Outputs  []pt2Argument      `json:"outputs"`
	Metadata map[string]any     `json:"metadata"`
}

// nodeProducedValues returns the value identifiers declared in node.Outputs.
// Void operations (such as asserts) declare an empty outputs list and produce no values.
// A node that omits the outputs field altogether is invalid and returns an error.
func nodeProducedValues(node pt2Node) ([]string, error) {
	if node.Outputs == nil {
		return nil, fmt.Errorf("node %q missing outputs", node.Name)
	}
	var names []string
	for _, outArg := range node.Outputs {
		names = append(names, collectArgumentValues(outArg)...)
	}
	return names, nil
}

type pt2NamedArgument struct {
	Name string      `json:"name"`
	Arg  pt2Argument `json:"arg"`
}

type pt2SubGraph struct {
	Graph struct {
		Nodes []pt2Node `json:"nodes"`
	} `json:"graph"`
}

type pt2Argument struct {
	AsGraph   *pt2SubGraph    `json:"as_graph"`
	AsInt     *int64          `json:"as_int"`
	AsTensor  *pt2TensorArg   `json:"as_tensor"`
	AsTensors []*pt2TensorArg `json:"as_tensors"`
	AsSymInt  *struct {
		AsName string `json:"as_name"`
	} `json:"as_sym_int"`
}

func collectArgumentValues(arg pt2Argument) []string {
	var names []string
	if arg.AsTensor != nil && arg.AsTensor.Name != "" {
		names = append(names, arg.AsTensor.Name)
	}
	for _, t := range arg.AsTensors {
		if t != nil && t.Name != "" {
			names = append(names, t.Name)
		}
	}
	if arg.AsSymInt != nil && arg.AsSymInt.AsName != "" {
		names = append(names, arg.AsSymInt.AsName)
	}
	return names
}

func canonicalDtype(dt string) string {
	switch strings.ToLower(dt) {
	case "bf16", "bfloat16":
		return "bfloat16"
	case "f32", "fp32", "float32":
		return "float32"
	case "f16", "fp16", "float16":
		return "float16"
	default:
		return strings.ToLower(dt)
	}
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

	var mBytes, bBytes []byte
	if data, err := os.ReadFile(filepath.Join(artifactDir, "manifest.json")); err == nil {
		mBytes = data
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read manifest.json: %w", err)
	}
	if data, err := os.ReadFile(filepath.Join(artifactDir, "binding.json")); err == nil {
		bBytes = data
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read binding.json: %w", err)
	}
	return readSegmentsFromArchive(zr.File, mBytes, bBytes)
}

// ReadSegmentsFromBytes partitions an in-memory decode graph archive into segments.
func ReadSegmentsFromBytes(decodeGraph, manifestBytes, bindingBytes []byte) ([]*planpb.Segment, error) {
	zr, err := zip.NewReader(bytes.NewReader(decodeGraph), int64(len(decodeGraph)))
	if err != nil {
		return nil, fmt.Errorf("open archive: %w", err)
	}
	return readSegmentsFromArchive(zr.File, manifestBytes, bindingBytes)
}

// ReadSegmentsFromStore fetches decode.pt2, manifest.json, and binding.json from
// store for repository and tag and partitions the exported decode graph into segments.
func ReadSegmentsFromStore(ctx context.Context, store *weightstore.Client, repo, tag string) ([]*planpb.Segment, error) {
	decodeBytes, err := store.FetchFile(ctx, repo, tag, "decode.pt2")
	if err != nil {
		return nil, fmt.Errorf("fetch decode.pt2: %w", err)
	}
	var mBytes, bBytes []byte
	if data, err := store.FetchFile(ctx, repo, tag, "manifest.json"); err == nil {
		mBytes = data
	} else if !errors.Is(err, weightstore.ErrNotFound) {
		return nil, fmt.Errorf("fetch manifest.json: %w", err)
	}
	if data, err := store.FetchFile(ctx, repo, tag, "binding.json"); err == nil {
		bBytes = data
	} else if !errors.Is(err, weightstore.ErrNotFound) {
		return nil, fmt.Errorf("fetch binding.json: %w", err)
	}
	return ReadSegmentsFromBytes(decodeBytes, mBytes, bBytes)
}

func readSegmentsFromArchive(files []*zip.File, manifestBytes, bindingBytes []byte) ([]*planpb.Segment, error) {
	var formatFile, versionFile, modelFile *zip.File
	for _, f := range files {
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

	var manifest struct {
		Tensors map[string]struct {
			DType string `json:"dtype"`
			Shape []int  `json:"shape"`
		} `json:"tensors"`
	}
	if len(manifestBytes) > 0 {
		if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
			return nil, fmt.Errorf("parse manifest.json: %w", err)
		}
	}

	var binding struct {
		Bound map[string]struct {
			DType string `json:"dtype"`
			Shape []int  `json:"shape"`
		} `json:"bound"`
	}
	if len(bindingBytes) > 0 {
		if err := json.Unmarshal(bindingBytes, &binding); err != nil {
			return nil, fmt.Errorf("parse binding.json: %w", err)
		}
	}

	if len(model.GraphModule.Signature.InputSpecs) == 0 {
		return nil, fmt.Errorf("archive missing signature input specs")
	}
	if len(model.GraphModule.Signature.OutputSpecs) == 0 {
		return nil, fmt.Errorf("archive missing signature output specs")
	}

	residentSlot := make(map[string]string)
	stepInputSet := make(map[string]bool)

	for _, spec := range model.GraphModule.Signature.InputSpecs {
		if spec.UserInput != nil {
			for _, name := range collectArgumentValues(spec.UserInput.Arg) {
				stepInputSet[name] = true
			}
		}
		if spec.Parameter != nil && spec.Parameter.Arg.Name != "" {
			if spec.Parameter.ParameterName == "" {
				return nil, fmt.Errorf("parameter input %q missing parameter_name", spec.Parameter.Arg.Name)
			}
			residentSlot[spec.Parameter.Arg.Name] = spec.Parameter.ParameterName
		}
		if spec.Buffer != nil && spec.Buffer.Arg.Name != "" {
			if spec.Buffer.BufferName == "" {
				return nil, fmt.Errorf("buffer input %q missing buffer_name", spec.Buffer.Arg.Name)
			}
			residentSlot[spec.Buffer.Arg.Name] = spec.Buffer.BufferName
		}
		if spec.TensorConstant != nil && spec.TensorConstant.Arg.Name != "" {
			if spec.TensorConstant.TensorConstantName == "" {
				return nil, fmt.Errorf("tensor constant input %q missing tensor_constant_name", spec.TensorConstant.Arg.Name)
			}
			residentSlot[spec.TensorConstant.Arg.Name] = spec.TensorConstant.TensorConstantName
		}
	}

	userOutputSet := make(map[string]bool)
	bufferMutationSet := make(map[string]string)

	for _, spec := range model.GraphModule.Signature.OutputSpecs {
		if spec.UserOutput != nil {
			for _, name := range collectArgumentValues(spec.UserOutput.Arg) {
				userOutputSet[name] = true
			}
		}
		if spec.BufferMutation != nil && spec.BufferMutation.Arg.Name != "" {
			if spec.BufferMutation.BufferName == "" {
				return nil, fmt.Errorf("buffer mutation for %q missing buffer_name", spec.BufferMutation.Arg.Name)
			}
			bufferMutationSet[spec.BufferMutation.Arg.Name] = spec.BufferMutation.BufferName
		}
	}

	var (
		segments         []*planpb.Segment
		segmentsByID     = make(map[int]*planpb.Segment)
		segmentIDs       []int
		hostCount        int
		nodesByPartID    = make(map[int][]pt2Node)
		producedByPartID = make(map[int][]string)
		nodeNamesByPart  = make(map[int]map[string]bool)
		producerPartID   = make(map[string]int)
	)

	var processNode func(node pt2Node) error
	processNode = func(node pt2Node) error {
		if node.Target == "output" || node.Name == "output" {
			return nil
		}
		if node.Target == "placeholder" {
			return nil
		}

		op, err := ParseOperator(node.Target)
		if err != nil {
			return fmt.Errorf("parse node %q operator: %w", node.Name, err)
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
			segmentIDs = append(segmentIDs, partID)
		}

		produced, err := nodeProducedValues(node)
		if err != nil {
			return err
		}
		if nodeNamesByPart[partID] == nil {
			nodeNamesByPart[partID] = make(map[string]bool)
		}
		for _, v := range produced {
			nodeNamesByPart[partID][v] = true
			producerPartID[v] = partID
		}
		producedByPartID[partID] = append(producedByPartID[partID], produced...)
		nodesByPartID[partID] = append(nodesByPartID[partID], node)

		if op != "" {
			var stack string
			if node.Metadata != nil {
				if s, ok := node.Metadata["nn_module_stack"].(string); ok {
					stack = s
				}
			}
			sitePath := ParseSite(stack)
			cs := &planpb.CallSite{
				Op:   op,
				Node: node.Name,
				Site: sitePath,
			}

			// Read tensor metadata from binding.json or manifest.json from the node's site and inputs
			var weightTensor *struct {
				DType string `json:"dtype"`
				Shape []int  `json:"shape"`
			}
			var weightKeys []string
			if sitePath != "" {
				weightKeys = append(weightKeys,
					sitePath+".weight",
					strings.TrimPrefix(sitePath, "model.")+".weight",
					"model."+sitePath+".weight",
				)
			}
			for _, inp := range node.Inputs {
				if inp.Name == "weight" && inp.Arg.AsTensor != nil {
					tName := inp.Arg.AsTensor.Name
					weightKeys = append(weightKeys, tName, strings.TrimPrefix(tName, "p_"))
				}
			}

			for _, k := range weightKeys {
				if binding.Bound != nil {
					if t, ok := binding.Bound[k]; ok {
						weightTensor = &t
						break
					}
				}
				if manifest.Tensors != nil {
					if t, ok := manifest.Tensors[k]; ok {
						weightTensor = &t
						break
					}
				}
			}

			if weightTensor != nil {
				cs.Dtype = canonicalDtype(weightTensor.DType)
				if len(weightTensor.Shape) == 2 {
					if op == "vxpu::head_linear" {
						cs.Shapes = fmt.Sprintf("%d->%d %s", weightTensor.Shape[1], weightTensor.Shape[0], cs.Dtype)
					} else if op == "vxpu::fp8_linear" {
						cs.Shapes = fmt.Sprintf("%d->%d", weightTensor.Shape[1], weightTensor.Shape[0])
					}
				}
			}

			// Read facts from node input arguments
			for _, inp := range node.Inputs {
				if inp.Name == "hc_mult" && inp.Arg.AsInt != nil {
					mult := *inp.Arg.AsInt
					// Triton hc_split_sinkhorn requires power-of-two width for register block tiling (triton_ops.hc_split_sinkhorn_supported).
					cs.SinkhornSupported = mult > 0 && (mult&(mult-1)) == 0
					cs.Shapes = fmt.Sprintf("hc=%d", mult)
				} else if inp.Name == "table" && inp.Arg.AsInt != nil {
					cs.Shapes = fmt.Sprintf("table=%d", *inp.Arg.AsInt)
				}
			}
			seg.CallSites = append(seg.CallSites, cs)
		}

		for _, inp := range node.Inputs {
			if inp.Arg.AsGraph != nil {
				for _, subNode := range inp.Arg.AsGraph.Graph.Nodes {
					if err := processNode(subNode); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}

	for _, node := range model.GraphModule.Graph.Nodes {
		if err := processNode(node); err != nil {
			return nil, err
		}
	}

	// Compute Reads for each segment in execution order
	readsBySegIndex := make([]map[string]bool, len(segments))
	for idx, seg := range segments {
		partID := segmentIDs[idx]
		partNodes := nodeNamesByPart[partID]
		seenRead := make(map[string]bool)
		readsBySegIndex[idx] = make(map[string]bool)

		for _, node := range nodesByPartID[partID] {
			for _, inp := range node.Inputs {
				for _, v := range collectArgumentValues(inp.Arg) {
					if partNodes[v] {
						continue
					}
					if seenRead[v] {
						continue
					}
					seenRead[v] = true
					readsBySegIndex[idx][v] = true

					var kind planpb.ValueKind
					var resident string
					if prodPart, ok := producerPartID[v]; ok && prodPart < partID {
						if slot, isMut := bufferMutationSet[v]; isMut {
							kind = planpb.ValueKind_VALUE_KIND_RESIDENT
							resident = slot
						} else {
							kind = planpb.ValueKind_VALUE_KIND_INTERMEDIATE
						}
					} else if slot, ok := residentSlot[v]; ok {
						kind = planpb.ValueKind_VALUE_KIND_RESIDENT
						resident = slot
					} else if stepInputSet[v] {
						kind = planpb.ValueKind_VALUE_KIND_STEP_INPUT
					} else {
						return nil, fmt.Errorf("value %q read by segment %q is not a declared step input, not a resident value, and was not produced by an earlier segment", v, seg.Name)
					}

					seg.Reads = append(seg.Reads, &planpb.Value{
						Name:     v,
						Kind:     kind,
						Resident: resident,
					})
				}
			}
		}
	}

	// Compute Writes for each segment in execution order
	for idx, seg := range segments {
		partID := segmentIDs[idx]
		seenWrite := make(map[string]bool)

		for _, v := range producedByPartID[partID] {
			if seenWrite[v] {
				continue
			}

			isUserOut := userOutputSet[v]
			isReadLater := false
			for laterIdx := idx + 1; laterIdx < len(segments); laterIdx++ {
				if readsBySegIndex[laterIdx][v] {
					isReadLater = true
					break
				}
			}

			if isUserOut {
				seenWrite[v] = true
				seg.Writes = append(seg.Writes, &planpb.Value{
					Name: v,
					Kind: planpb.ValueKind_VALUE_KIND_STEP_OUTPUT,
				})
			} else if residentName, ok := bufferMutationSet[v]; ok {
				seenWrite[v] = true
				seg.Writes = append(seg.Writes, &planpb.Value{
					Name:     v,
					Kind:     planpb.ValueKind_VALUE_KIND_RESIDENT,
					Resident: residentName,
				})
			} else if isReadLater {
				seenWrite[v] = true
				seg.Writes = append(seg.Writes, &planpb.Value{
					Name: v,
					Kind: planpb.ValueKind_VALUE_KIND_INTERMEDIATE,
				})
			}
		}
	}

	return segments, nil
}

// ReadCallSites reads all custom operator call sites from an artifact's decode.pt2 in graph order.
func ReadCallSites(artifactDir string) ([]*planpb.CallSite, error) {
	segments, err := ReadSegments(artifactDir)
	if err != nil {
		return nil, err
	}
	var sites []*planpb.CallSite
	for _, seg := range segments {
		sites = append(sites, seg.CallSites...)
	}
	return sites, nil
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
