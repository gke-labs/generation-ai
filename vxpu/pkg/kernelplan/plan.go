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
	"bytes"
	"crypto/md5"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

// DefaultRowCounts is the standard row count ladder for plan emission.
var DefaultRowCounts = []int{1, 5, 8, 16, 32, 40, 64}

var compactOptions = protojson.MarshalOptions{
	UseProtoNames: true,
}

func toInt32Slice(s []int) []int32 {
	r := make([]int32, len(s))
	for i, v := range s {
		r[i] = int32(v)
	}
	return r
}

func newSite(id string, kind planpb.SiteKind, shape, rowCounts []int32) *planpb.PlanSite {
	return &planpb.PlanSite{
		Id:        id,
		Kind:      kind,
		Shape:     shape,
		RowCounts: rowCounts,
		Strategy:  "default",
	}
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

// ComputeArtifactMD5 computes the MD5 checksum over manifest and binding bytes.
func ComputeArtifactMD5(m, b []byte) string {
	h := md5.New()
	h.Write(m)
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

// ArtifactMD5 computes the MD5 checksum over manifest.json and binding.json bytes in artifactDir.
func ArtifactMD5(artifactDir string) (string, error) {
	manifestBytes, err := os.ReadFile(filepath.Join(artifactDir, "manifest.json"))
	if err != nil {
		return "", fmt.Errorf("read manifest: %w", err)
	}
	bindingBytes, err := os.ReadFile(filepath.Join(artifactDir, "binding.json"))
	if err != nil {
		return "", fmt.Errorf("read binding: %w", err)
	}
	return ComputeArtifactMD5(manifestBytes, bindingBytes), nil
}

func hasUnknownFields(m protoreflect.Message) bool {
	if len(m.GetUnknown()) > 0 {
		return true
	}
	var found bool
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		if fd.IsList() {
			if fd.Message() != nil {
				list := v.List()
				for i := 0; i < list.Len(); i++ {
					if hasUnknownFields(list.Get(i).Message()) {
						found = true
						return false
					}
				}
			}
		} else if fd.IsMap() {
			if fd.MapValue().Message() != nil {
				v.Map().Range(func(k protoreflect.MapKey, mv protoreflect.Value) bool {
					if hasUnknownFields(mv.Message()) {
						found = true
						return false
					}
					return true
				})
			}
		} else if fd.Message() != nil && m.Has(fd) {
			if hasUnknownFields(v.Message()) {
				found = true
				return false
			}
		}
		return !found
	})
	return found
}

// ReadDocument reads path and unmarshals it into msg.
// If path ends with ".json" (case-insensitive), it unmarshals as JSON using protojson;
// otherwise, as binary protobuf. Binary documents with unknown fields are rejected
// to prevent reading the wrong document type.
func ReadDocument(path string, msg proto.Message) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	if strings.HasSuffix(strings.ToLower(path), ".json") {
		if err := (protojson.UnmarshalOptions{DiscardUnknown: false}).Unmarshal(data, msg); err != nil {
			return fmt.Errorf("unmarshal %s as JSON: %w", path, err)
		}
		return nil
	}
	if err := proto.Unmarshal(data, msg); err != nil {
		return fmt.Errorf("unmarshal %s as binary: %w", path, err)
	}
	if hasUnknownFields(msg.ProtoReflect()) {
		return fmt.Errorf("unmarshal %s: document contains unknown fields (wrong document type or schema mismatch)", path)
	}
	return nil
}

// ReadPlan reads and unmarshals a PlanFile from a binary or JSON file.
func ReadPlan(path string) (*planpb.PlanFile, error) {
	var p planpb.PlanFile
	if err := ReadDocument(path, &p); err != nil {
		return nil, fmt.Errorf("read plan: %w", err)
	}
	return &p, nil
}

// PlanHash computes the SHA-256 digest representing the plan's identity in Go.
// It covers what the planner decided for this artifact and machine:
// artifact_md5, sites (excluding each site's stats), knobs, and segments.
// It excludes build provenance (tree_commit), execution measurements
// (PlanFile.stats and each PlanSite.stats), and any existing plan_hash.
func PlanHash(p *planpb.PlanFile) (string, error) {
	if p == nil {
		return "", errors.New("cannot compute hash of nil plan")
	}
	clone := proto.Clone(p).(*planpb.PlanFile)
	clone.TreeCommit = ""
	clone.Stats = nil
	clone.PlanHash = ""
	for _, s := range clone.Sites {
		s.Stats = nil
	}
	data, err := MarshalAsJSON(clone)
	if err != nil {
		return "", fmt.Errorf("marshal plan for hashing: %w", err)
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]), nil
}

// EnsurePlanIdentity computes the plan's identity in Go and sets PlanHash.
// If the plan already carries a PlanHash that does not match the computed
// identity, it returns an error refusing the mismatched hash.
func EnsurePlanIdentity(p *planpb.PlanFile) error {
	if p == nil {
		return nil
	}
	h, err := PlanHash(p)
	if err != nil {
		return err
	}
	if p.PlanHash != "" && p.PlanHash != h {
		return fmt.Errorf("plan hash %q does not match computed identity %q", p.PlanHash, h)
	}
	p.PlanHash = h
	return nil
}

// Marshal serializes a PlanFile to binary protobuf bytes.
func Marshal(p *planpb.PlanFile) ([]byte, error) {
	return proto.Marshal(p)
}

// MarshalAsJSON serializes any protobuf message to formatted JSON bytes using compact protojson
// indented deterministically with encoding/json.Indent.
func MarshalAsJSON(m proto.Message) ([]byte, error) {
	raw, err := compactOptions.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("marshal message as JSON: %w", err)
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "", "  "); err != nil {
		return nil, fmt.Errorf("indent message JSON: %w", err)
	}
	buf.WriteByte('\n')
	return buf.Bytes(), nil
}

// ParseKernelTable unmarshals JSON bytes using protojson into a KernelTable.
func ParseKernelTable(data []byte) (*planpb.KernelTable, error) {
	var kt planpb.KernelTable
	if err := protojson.Unmarshal(data, &kt); err != nil {
		return nil, fmt.Errorf("unmarshal kernel table: %w", err)
	}
	return &kt, nil
}

// MarshalKernelTableAsJSON serializes a KernelTable to formatted JSON bytes.
func MarshalKernelTableAsJSON(kt *planpb.KernelTable) ([]byte, error) {
	return MarshalAsJSON(kt)
}

// ReadKernelRecord reads and unmarshals a KernelRecord from a binary or JSON file.
func ReadKernelRecord(path string) (*planpb.KernelRecord, error) {
	var kr planpb.KernelRecord
	if err := ReadDocument(path, &kr); err != nil {
		return nil, fmt.Errorf("read kernel record: %w", err)
	}
	return &kr, nil
}

// ReadKernelCatalog reads and unmarshals a KernelCatalog from a binary or JSON file.
func ReadKernelCatalog(path string) (*planpb.KernelCatalog, error) {
	var cat planpb.KernelCatalog
	if err := ReadDocument(path, &cat); err != nil {
		return nil, fmt.Errorf("read kernel catalog: %w", err)
	}
	return &cat, nil
}

//go:embed catalog/kernel_catalog.json
var defaultKernelCatalogJSON []byte

// DefaultKernelCatalog returns the built-in catalog of kernel bodies.
func DefaultKernelCatalog() (*planpb.KernelCatalog, error) {
	var cat planpb.KernelCatalog
	if err := protojson.Unmarshal(defaultKernelCatalogJSON, &cat); err != nil {
		return nil, fmt.Errorf("unmarshal default kernel catalog: %w", err)
	}
	return &cat, nil
}

// ReadJSON reads and unmarshals a JSON file into a protobuf message.
func ReadJSON[M proto.Message](path string, msg M) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := protojson.Unmarshal(data, msg); err != nil {
		return fmt.Errorf("unmarshal %s: %w", filepath.Base(path), err)
	}
	return nil
}

type tableKey struct {
	op   string
	spec string
}

type tableMatch struct {
	impl  string
	index int
}

type tableIndex struct {
	bySite   map[tableKey]tableMatch
	byShapes map[tableKey]tableMatch
	byOp     map[string]tableMatch
}

func indexTable(table *planpb.KernelTable) (*tableIndex, error) {
	if table == nil {
		return nil, nil
	}
	idx := &tableIndex{
		bySite:   make(map[tableKey]tableMatch),
		byShapes: make(map[tableKey]tableMatch),
		byOp:     make(map[string]tableMatch),
	}
	for i, entry := range table.Sites {
		if entry.Op == "" || entry.Impl == "" {
			return nil, fmt.Errorf("kernel table: sites[%d] requires op and impl", i)
		}
		tm := tableMatch{impl: entry.Impl, index: i}
		if entry.GetSite() != "" {
			k := tableKey{op: entry.Op, spec: entry.GetSite()}
			if existing, ok := idx.bySite[k]; ok && existing.impl != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for site %q", existing.impl, entry.Impl, entry.GetSite())
			}
			idx.bySite[k] = tm
		} else if entry.GetShapes() != "" {
			k := tableKey{op: entry.Op, spec: entry.GetShapes()}
			if existing, ok := idx.byShapes[k]; ok && existing.impl != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for shapes %q", existing.impl, entry.Impl, entry.GetShapes())
			}
			idx.byShapes[k] = tm
		} else {
			if existing, ok := idx.byOp[entry.Op]; ok && existing.impl != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for op %q", existing.impl, entry.Impl, entry.Op)
			}
			idx.byOp[entry.Op] = tm
		}
	}
	return idx, nil
}

func (idx *tableIndex) match(op, site, shapes string, matchedEntries map[int]bool) (string, bool) {
	if idx == nil {
		return "", false
	}
	if site != "" {
		if tm, ok := idx.bySite[tableKey{op: op, spec: site}]; ok {
			if matchedEntries != nil {
				matchedEntries[tm.index] = true
			}
			return tm.impl, true
		}
	}
	if shapes != "" {
		if tm, ok := idx.byShapes[tableKey{op: op, spec: shapes}]; ok {
			if matchedEntries != nil {
				matchedEntries[tm.index] = true
			}
			return tm.impl, true
		}
	}
	if tm, ok := idx.byOp[op]; ok {
		if matchedEntries != nil {
			matchedEntries[tm.index] = true
		}
		return tm.impl, true
	}
	return "", false
}

type catalogIndex struct {
	bodiesByOp map[string][]*planpb.KernelBody
	kindByOp   map[string]planpb.SiteKind
}

func indexCatalog(cat *planpb.KernelCatalog) (*catalogIndex, error) {
	if cat == nil {
		return nil, fmt.Errorf("kernel catalog is required")
	}
	idx := &catalogIndex{
		bodiesByOp: make(map[string][]*planpb.KernelBody),
		kindByOp:   make(map[string]planpb.SiteKind),
	}
	for _, b := range cat.Bodies {
		if b.Op == "" || b.Name == "" {
			return nil, fmt.Errorf("kernel catalog: body requires op and name")
		}
		idx.bodiesByOp[b.Op] = append(idx.bodiesByOp[b.Op], b)
		if b.Kind != planpb.SiteKind_SITE_KIND_UNSPECIFIED {
			idx.kindByOp[b.Op] = b.Kind
		}
	}
	return idx, nil
}

func matchCondition(cond *planpb.KernelCondition, site *planpb.CallSite) bool {
	if cond == nil {
		return true
	}
	if cond.MaxRows != nil {
		if len(site.RowCounts) > 0 {
			for _, r := range site.RowCounts {
				if r > cond.GetMaxRows() {
					return false
				}
			}
		} else if site.Rows > 0 {
			if site.Rows > cond.GetMaxRows() {
				return false
			}
		} else {
			return false
		}
	}
	if cond.ExactRows != nil {
		if len(site.RowCounts) > 0 {
			for _, r := range site.RowCounts {
				if r != cond.GetExactRows() {
					return false
				}
			}
		} else if site.Rows > 0 {
			if site.Rows != cond.GetExactRows() {
				return false
			}
		} else {
			return false
		}
	}
	if cond.MinRows != nil {
		if len(site.RowCounts) > 0 {
			for _, r := range site.RowCounts {
				if r < cond.GetMinRows() {
					return false
				}
			}
		} else if site.Rows > 0 {
			if site.Rows < cond.GetMinRows() {
				return false
			}
		} else {
			return false
		}
	}
	if cond.Dtype != nil {
		if site.Dtype == "" || site.Dtype != cond.GetDtype() {
			return false
		}
	}
	if cond.RequireSinkhornSupported != nil && cond.GetRequireSinkhornSupported() {
		if !site.SinkhornSupported {
			return false
		}
	}
	return true
}

func isDeviceApplicable(b *planpb.KernelBody, deviceType string) bool {
	if slices.Contains(b.Devices, deviceType) {
		return true
	}
	return b.Host && slices.Contains(b.Devices, "cpu")
}

func matchSiteBody(site *planpb.CallSite, catIdx *catalogIndex, tblIdx *tableIndex, device string, matchedEntries map[int]bool) (string, planpb.SiteKind, error) {
	deviceType := strings.Split(device, ":")[0]
	if deviceType == "" {
		deviceType = "cpu"
	}
	siteIdent := site.Node
	if site.Site != "" {
		siteIdent = fmt.Sprintf("%s (%s)", site.Node, site.Site)
	}

	bodies, ok := catIdx.bodiesByOp[site.Op]
	if !ok || len(bodies) == 0 {
		// Stated rule: ops without catalog bodies execute their default implementation without planner substitution.
		return "", planpb.SiteKind_SITE_KIND_UNSPECIFIED, nil
	}
	kind := catIdx.kindByOp[site.Op]

	var applicable []*planpb.KernelBody
	for _, b := range bodies {
		if isDeviceApplicable(b, deviceType) && matchCondition(b.Condition, site) {
			applicable = append(applicable, b)
		}
	}

	if impl, matched := tblIdx.match(site.Op, site.Site, site.Shapes, matchedEntries); matched {
		for _, b := range applicable {
			if b.Name == impl {
				return impl, kind, nil
			}
		}
		for _, b := range bodies {
			if b.Name == impl {
				if !isDeviceApplicable(b, deviceType) {
					return "", 0, fmt.Errorf("site %q: table names body %q which does not apply for device %q", siteIdent, impl, deviceType)
				}
				if b.Condition != nil {
					if b.Condition.MaxRows != nil && site.Rows == 0 && len(site.RowCounts) == 0 {
						return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (requires max_rows %d fact not provided via --rows)", siteIdent, impl, b.Condition.GetMaxRows())
					}
					if b.Condition.ExactRows != nil && site.Rows == 0 && len(site.RowCounts) == 0 {
						return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (requires exact_rows %d fact not provided via --rows)", siteIdent, impl, b.Condition.GetExactRows())
					}
					if b.Condition.MinRows != nil && site.Rows == 0 && len(site.RowCounts) == 0 {
						return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (requires min_rows %d fact not provided via --rows)", siteIdent, impl, b.Condition.GetMinRows())
					}
					if b.Condition.Dtype != nil && site.Dtype == "" {
						return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (requires dtype %q fact not provided in artifact tensor metadata)", siteIdent, impl, b.Condition.GetDtype())
					}
					if b.Condition.RequireSinkhornSupported != nil && b.Condition.GetRequireSinkhornSupported() && !site.SinkhornSupported {
						return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (requires sinkhorn supported fact not satisfied by node hc_mult)", siteIdent, impl)
					}
				}
				return "", 0, fmt.Errorf("site %q: table names body %q which does not apply at this site", siteIdent, impl)
			}
		}
		return "", 0, fmt.Errorf("site %q: table names body %q which does not apply (not in catalog)", siteIdent, impl)
	}

	for _, b := range applicable {
		if b.Condition == nil || b.Condition.RequireExternalFact == nil {
			return b.Name, kind, nil
		}
	}

	if len(applicable) == 0 {
		return "", 0, fmt.Errorf("site %q: no kernel body applies for device %q", siteIdent, deviceType)
	}
	return "", 0, fmt.Errorf("site %q: all applicable bodies require external facts (%s) not provided in table", siteIdent, applicable[0].Condition.GetRequireExternalFact())
}

// PlanSites chooses a kernel body for each call site in sites according to cat, table, and device.
func PlanSites(
	sites []*planpb.CallSite,
	cat *planpb.KernelCatalog,
	table *planpb.KernelTable,
	device string,
) ([]*planpb.PlanSite, error) {
	if cat == nil {
		var err error
		cat, err = DefaultKernelCatalog()
		if err != nil {
			return nil, fmt.Errorf("load default kernel catalog: %w", err)
		}
	}
	catIdx, err := indexCatalog(cat)
	if err != nil {
		return nil, err
	}
	tblIdx, err := indexTable(table)
	if err != nil {
		return nil, err
	}

	matchedEntries := make(map[int]bool)
	planSites := make([]*planpb.PlanSite, len(sites))
	for i, s := range sites {
		body, kind, err := matchSiteBody(s, catIdx, tblIdx, device, matchedEntries)
		if err != nil {
			return nil, err
		}
		planSites[i] = &planpb.PlanSite{
			Id:   s.Node,
			Kind: kind,
			Body: body,
		}
	}

	if table != nil {
		for i, entry := range table.Sites {
			if !matchedEntries[i] {
				var desc []string
				if entry.Op != "" {
					desc = append(desc, fmt.Sprintf("op %q", entry.Op))
				}
				if entry.GetSite() != "" {
					desc = append(desc, fmt.Sprintf("site %q", entry.GetSite()))
				}
				if entry.GetShapes() != "" {
					desc = append(desc, fmt.Sprintf("shapes %q", entry.GetShapes()))
				}
				if entry.Impl != "" {
					desc = append(desc, fmt.Sprintf("impl %q", entry.Impl))
				}
				return nil, fmt.Errorf("kernel table entry applies to no call site: %s", strings.Join(desc, ", "))
			}
		}
	}

	return planSites, nil
}

// JoinPlan joins plan implementation choices to call sites in segments by graph node name.
// It sets Body on each CallSite. If any call site in the segments is not present in the plan,
// or if any call site entry in the plan is not present in the segments, it returns an error naming the unmatched site.
func JoinPlan(segments []*planpb.Segment, plan *planpb.PlanFile) error {
	if plan == nil {
		return nil
	}
	planSites := make(map[string]*planpb.PlanSite, len(plan.Sites))
	for _, ps := range plan.Sites {
		if _, exists := planSites[ps.Id]; exists {
			return fmt.Errorf("duplicate site in plan: %q", ps.Id)
		}
		planSites[ps.Id] = ps
	}

	matched := make(map[string]bool, len(plan.Sites))
	for _, seg := range segments {
		for _, cs := range seg.CallSites {
			ps, ok := planSites[cs.Node]
			if !ok {
				return fmt.Errorf("unmatched call site in graph: %q", cs.Node)
			}
			cs.Body = ps.Body
			matched[cs.Node] = true
		}
	}

	for id, ps := range planSites {
		// Artifact macroscopic sites from binding.json (parameter paths) have Body == "".
		// Planned call sites have Body != "" and must match a call site in the graph.
		if ps.Body != "" && !matched[id] {
			return fmt.Errorf("unmatched plan site: %q", id)
		}
	}

	if len(plan.Segments) > 0 {
		planSegsByName := make(map[string]*planpb.Segment, len(plan.Segments))
		for _, ps := range plan.Segments {
			if ps != nil && ps.Name != "" {
				planSegsByName[ps.Name] = ps
			}
		}
		for _, seg := range segments {
			if seg == nil {
				continue
			}
			if ps, ok := planSegsByName[seg.Name]; ok {
				seg.Run = ps.Run
			}
		}
	}

	return nil
}

// KnobTypes returns a map of knob names to their protobuf kinds, derived
// directly from the Knobs message descriptor.
func KnobTypes() map[string]protoreflect.Kind {
	fields := (&planpb.Knobs{}).ProtoReflect().Descriptor().Fields()
	types := make(map[string]protoreflect.Kind, fields.Len())
	for i := 0; i < fields.Len(); i++ {
		fd := fields.Get(i)
		types[string(fd.Name())] = fd.Kind()
	}
	return types
}

// ValidatePlanGoal reports an error if g is not a recognized PlanGoal value.
func ValidatePlanGoal(g planpb.PlanGoal) error {
	switch g {
	case planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH, planpb.PlanGoal_PLAN_GOAL_SPEED:
		return nil
	default:
		return fmt.Errorf("unknown plan goal %v: want pass-through or speed", g)
	}
}

// ParsePlanGoal parses a goal name into a PlanGoal enum value.
// An empty string parses as the default goal (PLAN_GOAL_PASS_THROUGH).
func ParsePlanGoal(s string) (planpb.PlanGoal, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "pass-through":
		return planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH, nil
	case "speed":
		return planpb.PlanGoal_PLAN_GOAL_SPEED, nil
	default:
		return planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH, fmt.Errorf("unknown goal %q: want \"pass-through\" or \"speed\"", s)
	}
}

// FormatPlanGoal returns the human-readable name for a PlanGoal.
func FormatPlanGoal(g planpb.PlanGoal) string {
	switch g {
	case planpb.PlanGoal_PLAN_GOAL_PASS_THROUGH:
		return "pass-through"
	case planpb.PlanGoal_PLAN_GOAL_SPEED:
		return "speed"
	default:
		return g.String()
	}
}

// EmitOptions configures PlanArtifact, the single planner entry point used by
// both `vxpu plan emit` and the router.
type EmitOptions struct {
	// ManifestBytes is the raw manifest.json content of the artifact.
	ManifestBytes []byte
	// BindingBytes is the raw binding.json content of the artifact.
	BindingBytes []byte
	// DecodeGraph is the optional raw decode.pt2 archive content of the artifact.
	DecodeGraph []byte
	// TreeCommit is the optional git commit hash of the build tree.
	TreeCommit string
	// RowCounts overrides DefaultRowCounts when non-empty.
	RowCounts []int
	// MachineKind is the target hardware identifier produced by MachineKind,
	// MachineKindFromName, or MachineKindFromProbe.
	MachineKind string
	// Goal states what the caller wants from the plan (default: pass-through).
	Goal planpb.PlanGoal
	// Catalog is the optional kernel catalog; when nil, DefaultKernelCatalog is used.
	Catalog *planpb.KernelCatalog
	// Table is the optional kernel table overriding catalog choices and knobs.
	Table *planpb.KernelTable
}

// PlanArtifact builds a physical execution plan for an artifact on the target
// machine and computes its canonical plan_hash. Both `vxpu plan emit` and the
// router call this function so the same artifact, machine, and goal always
// produce the same plan and identity.
func PlanArtifact(opts EmitOptions) (*planpb.PlanFile, error) {
	if opts.TreeCommit == "unknown" {
		return nil, errors.New("tree commit is required (refusing \"unknown\")")
	}
	if err := ValidatePlanGoal(opts.Goal); err != nil {
		return nil, err
	}
	device, err := DeviceForMachineKind(opts.MachineKind)
	if err != nil {
		return nil, err
	}

	plan, err := emitBase(opts.ManifestBytes, opts.BindingBytes, opts.TreeCommit, opts.RowCounts)
	if err != nil {
		return nil, err
	}
	plan.Goal = opts.Goal

	if len(opts.DecodeGraph) > 0 {
		segments, err := ReadSegmentsFromBytes(opts.DecodeGraph, opts.ManifestBytes, opts.BindingBytes)
		if err != nil {
			return nil, fmt.Errorf("read segments: %w", err)
		}
		var callSites []*planpb.CallSite
		for _, seg := range segments {
			if seg != nil {
				callSites = append(callSites, seg.CallSites...)
			}
		}
		hasCustomOps := len(callSites) > 0
		for _, seg := range segments {
			if seg != nil {
				seg.Run = decideSegmentRunForDevice(seg, hasCustomOps, device, opts.Goal)
			}
		}
		if len(opts.RowCounts) > 0 {
			rowCounts32 := toInt32Slice(opts.RowCounts)
			for _, cs := range callSites {
				cs.Rows = rowCounts32[0]
				cs.RowCounts = rowCounts32
			}
		}
		cat := opts.Catalog
		if cat == nil {
			cat, err = DefaultKernelCatalog()
			if err != nil {
				return nil, fmt.Errorf("default kernel catalog: %w", err)
			}
		}
		sites, err := PlanSites(callSites, cat, opts.Table, device)
		if err != nil {
			return nil, fmt.Errorf("plan call sites: %w", err)
		}
		plan.Sites = append(plan.Sites, sites...)
		plan.Segments = segments
		if opts.Table != nil && opts.Table.Knobs != nil {
			plan.Knobs = opts.Table.Knobs
		}
	}

	if err := EnsurePlanIdentity(plan); err != nil {
		return nil, fmt.Errorf("ensure plan identity: %w", err)
	}
	return plan, nil
}

// Emit creates a PlanFile by inspecting the artifact directory.
func Emit(artifactDir, treeCommit string, rowCounts []int) (*planpb.PlanFile, error) {
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

// EmitFromArtifact parses manifest and binding JSON and builds a PlanFile.
func EmitFromArtifact(manifestBytes, bindingBytes []byte, treeCommit string, rowCounts []int) (*planpb.PlanFile, error) {
	if treeCommit == "" || treeCommit == "unknown" {
		return nil, errors.New("tree commit is required (refusing \"unknown\")")
	}
	return emitBase(manifestBytes, bindingBytes, treeCommit, rowCounts)
}

func emitBase(manifestBytes, bindingBytes []byte, treeCommit string, rowCounts []int) (*planpb.PlanFile, error) {
	if len(rowCounts) == 0 {
		rowCounts = append([]int{}, DefaultRowCounts...)
	}
	rc32 := toInt32Slice(rowCounts)

	var manifest Manifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	var binding Binding
	if len(bindingBytes) > 0 {
		if err := json.Unmarshal(bindingBytes, &binding); err != nil {
			return nil, fmt.Errorf("parse binding: %w", err)
		}
	}

	var sites []*planpb.PlanSite
	sites = append(sites, denseGEMVSites(binding.Bound, rc32)...)
	sites = append(sites, expertSites(binding.External.Experts, binding.Stacked, manifest.Tensors, rc32)...)
	sites = append(sites, attentionSites(binding.Bound, rc32)...)
	sites = append(sites, sumSites(binding.Bound, rc32)...)
	sites = append(sites, drafterSites(binding.External.Engram, rc32)...)

	return &planpb.PlanFile{
		ArtifactMd5: ComputeArtifactMD5(manifestBytes, bindingBytes),
		TreeCommit:  treeCommit,
		Sites:       sites,
	}, nil
}

func isScale(name string, m TensorMeta) bool {
	return strings.Contains(name, "scale") || m.DType == "F8_E8M0"
}

func denseGEMVSites(bound map[string]TensorMeta, rowCounts []int32) []*planpb.PlanSite {
	names := make([]string, 0, len(bound))
	for name, m := range bound {
		if len(m.Shape) == 2 && !isScale(name, m) && !strings.Contains(name, "embed") && !strings.Contains(name, "wte") {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	sites := make([]*planpb.PlanSite, 0, len(names))
	for _, name := range names {
		sites = append(sites, newSite(name, planpb.SiteKind_SITE_KIND_DENSE_GEMV, toInt32Slice(bound[name].Shape), rowCounts))
	}
	return sites
}

func expertSites(externalExperts map[string]map[string]TensorParts, stacked map[string]TensorParts, manifestTensors map[string]TensorMeta, rowCounts []int32) []*planpb.PlanSite {
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
	sites := make([]*planpb.PlanSite, 0, len(layers))
	for _, l := range layers {
		spec := layerMap[l]
		var shape []int32
		if len(spec.Parts) > 0 && len(spec.Parts[0]) > 0 {
			part := spec.Parts[0][0]
			shape = append([]int32{int32(len(spec.Parts))}, toInt32Slice(manifestTensors[part].Shape)...)
		}
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.experts", l), planpb.SiteKind_SITE_KIND_EXPERT_OP, shape, rowCounts))
	}
	return sites
}

func attentionSites(bound map[string]TensorMeta, rowCounts []int32) []*planpb.PlanSite {
	layers := boundLayers(bound)
	sites := make([]*planpb.PlanSite, 0, len(layers))
	for _, l := range layers {
		shape := toInt32Slice(findOutputProjection(bound, l))
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.attn", l), planpb.SiteKind_SITE_KIND_ATTENTION, shape, rowCounts))
	}
	return sites
}

func sumSites(bound map[string]TensorMeta, rowCounts []int32) []*planpb.PlanSite {
	layers := boundLayers(bound)
	sites := make([]*planpb.PlanSite, 0, len(layers)*2)
	for _, l := range layers {
		shape := findOutputProjection(bound, l)
		var sumShape []int32
		if len(shape) > 0 {
			sumShape = []int32{int32(shape[0])}
		}
		sites = append(sites,
			newSite(fmt.Sprintf("layers.%d.attn.sum", l), planpb.SiteKind_SITE_KIND_SUM_SITE, sumShape, rowCounts),
			newSite(fmt.Sprintf("layers.%d.ffn.sum", l), planpb.SiteKind_SITE_KIND_SUM_SITE, sumShape, rowCounts),
		)
	}
	return sites
}

func drafterSites(engram map[string]EngramMeta, rowCounts []int32) []*planpb.PlanSite {
	layers := sortedStringIntKeys(engram)
	sites := make([]*planpb.PlanSite, 0, len(layers))
	for _, l := range layers {
		e := engram[strconv.Itoa(l)]
		sites = append(sites, newSite(fmt.Sprintf("layers.%d.engram", l), planpb.SiteKind_SITE_KIND_DRAFTER, []int32{int32(e.NumEmbeddings), int32(e.Dim)}, rowCounts))
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
