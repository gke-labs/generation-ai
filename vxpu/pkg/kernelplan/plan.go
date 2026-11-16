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

func computeArtifactMD5(m, b []byte) string {
	h := md5.New()
	h.Write(m)
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
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

type tableIndex struct {
	bySite   map[tableKey]string
	byShapes map[tableKey]string
	byOp     map[string]string
}

func indexTable(table *planpb.KernelTable) (*tableIndex, error) {
	if table == nil {
		return nil, nil
	}
	idx := &tableIndex{
		bySite:   make(map[tableKey]string),
		byShapes: make(map[tableKey]string),
		byOp:     make(map[string]string),
	}
	for i, entry := range table.Sites {
		if entry.Op == "" || entry.Impl == "" {
			return nil, fmt.Errorf("kernel table: sites[%d] requires op and impl", i)
		}
		if entry.GetSite() != "" {
			k := tableKey{op: entry.Op, spec: entry.GetSite()}
			if existing, ok := idx.bySite[k]; ok && existing != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for site %q", existing, entry.Impl, entry.GetSite())
			}
			idx.bySite[k] = entry.Impl
		} else if entry.GetShapes() != "" {
			k := tableKey{op: entry.Op, spec: entry.GetShapes()}
			if existing, ok := idx.byShapes[k]; ok && existing != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for shapes %q", existing, entry.Impl, entry.GetShapes())
			}
			idx.byShapes[k] = entry.Impl
		} else {
			if existing, ok := idx.byOp[entry.Op]; ok && existing != entry.Impl {
				return nil, fmt.Errorf("kernel table: conflicting impl %q vs %q for op %q", existing, entry.Impl, entry.Op)
			}
			idx.byOp[entry.Op] = entry.Impl
		}
	}
	return idx, nil
}

func (idx *tableIndex) match(op, site, shapes string) (string, bool) {
	if idx == nil {
		return "", false
	}
	if site != "" {
		if impl, ok := idx.bySite[tableKey{op: op, spec: site}]; ok {
			return impl, true
		}
	}
	if shapes != "" {
		if impl, ok := idx.byShapes[tableKey{op: op, spec: shapes}]; ok {
			return impl, true
		}
	}
	if impl, ok := idx.byOp[op]; ok {
		return impl, true
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

func matchCondition(cond *planpb.KernelCondition, site *planpb.SiteRecord) bool {
	if cond == nil {
		return true
	}
	if cond.MaxRows != nil {
		if site.Rows == nil || site.GetRows() > cond.GetMaxRows() {
			return false
		}
	}
	if cond.ExactRows != nil {
		if site.Rows == nil || site.GetRows() != cond.GetExactRows() {
			return false
		}
	}
	if cond.MinRows != nil {
		if site.Rows == nil || site.GetRows() < cond.GetMinRows() {
			return false
		}
	}
	if cond.Dtype != nil {
		if site.Dtype == nil || site.GetDtype() != cond.GetDtype() {
			return false
		}
	}
	if cond.RequireSinkhornSupported != nil && cond.GetRequireSinkhornSupported() {
		if site.SinkhornSupported == nil || !site.GetSinkhornSupported() {
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

func matchSiteBody(site *planpb.SiteRecord, catIdx *catalogIndex, tblIdx *tableIndex, device string) (string, planpb.SiteKind, error) {
	deviceType := strings.Split(device, ":")[0]
	if deviceType == "" {
		deviceType = "cpu"
	}
	siteIdent := site.Node
	if site.GetSite() != "" {
		siteIdent = fmt.Sprintf("%s (%s)", site.Node, site.GetSite())
	}

	bodies, ok := catIdx.bodiesByOp[site.Op]
	if !ok || len(bodies) == 0 {
		return "", 0, fmt.Errorf("site %q: unknown op %q", siteIdent, site.Op)
	}
	kind := catIdx.kindByOp[site.Op]

	var applicable []*planpb.KernelBody
	for _, b := range bodies {
		if isDeviceApplicable(b, deviceType) && matchCondition(b.Condition, site) {
			applicable = append(applicable, b)
		}
	}

	if impl, matched := tblIdx.match(site.Op, site.GetSite(), site.GetShapes()); matched {
		for _, b := range applicable {
			if b.Name == impl {
				return impl, kind, nil
			}
		}
		return "", 0, fmt.Errorf("site %q: table names body %q which does not apply at this site", siteIdent, impl)
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

// PlanSites chooses a kernel body for each call site in desc according to cat, table, and device.
func PlanSites(
	desc *planpb.KernelRecord,
	cat *planpb.KernelCatalog,
	table *planpb.KernelTable,
	device string,
) ([]*planpb.PlanSite, error) {
	if desc == nil {
		return nil, fmt.Errorf("kernel record description is required")
	}
	catIdx, err := indexCatalog(cat)
	if err != nil {
		return nil, err
	}
	tblIdx, err := indexTable(table)
	if err != nil {
		return nil, err
	}

	sites := make([]*planpb.PlanSite, len(desc.Sites))
	for i, s := range desc.Sites {
		body, kind, err := matchSiteBody(s, catIdx, tblIdx, device)
		if err != nil {
			return nil, err
		}
		sites[i] = &planpb.PlanSite{
			Id:   s.Node,
			Kind: kind,
			Body: body,
		}
	}
	return sites, nil
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
	if len(rowCounts) == 0 {
		rowCounts = append([]int{}, DefaultRowCounts...)
	}
	rc32 := toInt32Slice(rowCounts)

	var manifest Manifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	var binding Binding
	if err := json.Unmarshal(bindingBytes, &binding); err != nil {
		return nil, fmt.Errorf("parse binding: %w", err)
	}

	var sites []*planpb.PlanSite
	sites = append(sites, denseGEMVSites(binding.Bound, rc32)...)
	sites = append(sites, expertSites(binding.External.Experts, binding.Stacked, manifest.Tensors, rc32)...)
	sites = append(sites, attentionSites(binding.Bound, rc32)...)
	sites = append(sites, sumSites(binding.Bound, rc32)...)
	sites = append(sites, drafterSites(binding.External.Engram, rc32)...)

	return &planpb.PlanFile{
		ArtifactMd5: computeArtifactMD5(manifestBytes, bindingBytes),
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
