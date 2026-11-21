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

package kernelplan_test

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

type standInStore struct {
	packages map[string]bool
	failures map[string]*planpb.CompileFailure
}

func newStandInStore() *standInStore {
	return &standInStore{
		packages: make(map[string]bool),
		failures: make(map[string]*planpb.CompileFailure),
	}
}

func (s *standInStore) HasPackage(ctx context.Context, model, key string) (bool, error) {
	return s.packages[key], nil
}

func (s *standInStore) GetCompileFailure(ctx context.Context, model, key string) (*planpb.CompileFailure, error) {
	return s.failures[key], nil
}

func (s *standInStore) RecordCompileFailure(ctx context.Context, model string, failure *planpb.CompileFailure) error {
	if failure != nil && failure.Package != nil && failure.Package.Key != "" {
		s.failures[failure.Package.Key] = failure
	}
	return nil
}

func (s *standInStore) ClearCompileFailure(ctx context.Context, model, key string) error {
	delete(s.failures, key)
	return nil
}

func baseSpec() kernelplan.PackageSpec {
	return kernelplan.PackageSpec{
		SoftwareVersion: "0123456789abcdef0123456789abcdef01234567",
		ArtifactMD5:     "cc56a287aa327e84649b8737b964449e",
		Segment:         "submod_0",
		Bodies:          []string{"reference", "reference", "reference"},
		MachineKind:     "gpu:nvidia l4",
	}
}

func createTestArtifact(t *testing.T, nodesJSON string) string {
	t.Helper()
	tmpDir := t.TempDir()
	pt2Path := filepath.Join(tmpDir, "decode.pt2")
	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	wFormat, err := zw.Create("decode/archive_format")
	if err != nil {
		t.Fatalf("create archive_format: %v", err)
	}
	if _, err := wFormat.Write([]byte("pt2")); err != nil {
		t.Fatalf("write archive_format: %v", err)
	}

	wVer, err := zw.Create("decode/archive_version")
	if err != nil {
		t.Fatalf("create archive_version: %v", err)
	}
	if _, err := wVer.Write([]byte("0")); err != nil {
		t.Fatalf("write archive_version: %v", err)
	}

	wModel, err := zw.Create("decode/models/model.json")
	if err != nil {
		t.Fatalf("create model.json: %v", err)
	}
	modelJSON := fmt.Sprintf(`{
		"schema_version": {"major": 8, "minor": 20},
		"graph_module": {
			"graph": {
				"nodes": %s
			}
		}
	}`, nodesJSON)
	if _, err := wModel.Write([]byte(modelJSON)); err != nil {
		t.Fatalf("write model.json: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	if err := os.WriteFile(pt2Path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write decode.pt2: %v", err)
	}
	return tmpDir
}

const smallGraphNodesJSON = `[
	{"name": "op0_0", "target": "torch.ops.vxpu.fp8_linear.default"},
	{"name": "op0_1", "target": "torch.ops.vxpu.fp8_linear.default"},
	{"name": "moe0", "target": "torch.ops.vxpu.moe_experts.default"},
	{"name": "op2_0", "target": "torch.ops.vxpu.fp8_linear.default"},
	{"name": "op2_1", "target": "torch.ops.vxpu.fp8_linear.default"},
	{"name": "engram0", "target": "torch.ops.vxpu.engram_rows.default"},
	{"name": "op4_0", "target": "torch.ops.vxpu.fp8_linear.default"}
]`

func newSmallGraphPlan() *planpb.PlanFile {
	return &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites: []*planpb.PlanSite{
			{Id: "op0_0", Body: "reference"},
			{Id: "op0_1", Body: "reference"},
			{Id: "moe0", Body: "reference"},
			{Id: "op2_0", Body: "reference"},
			{Id: "op2_1", Body: "reference"},
			{Id: "engram0", Body: "reference"},
			{Id: "op4_0", Body: "reference"},
		},
	}
}

func TestRequiredPackages_JoinValidation(t *testing.T) {
	artDir := createTestArtifact(t, smallGraphNodesJSON)
	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	machineKind := "gpu:nvidia l4"

	// 1. Plan site absent from graph is refused, naming the site
	planExtra := newSmallGraphPlan()
	planExtra.Sites = append(planExtra.Sites, &planpb.PlanSite{Id: "ghost_site", Body: "reference"})
	_, err = kernelplan.RequiredPackages(planExtra, segments, machineKind)
	if err == nil || !strings.Contains(err.Error(), `"ghost_site"`) {
		t.Errorf("expected error naming ghost_site, got: %v", err)
	}

	// 2. Plan whose only site is absent from graph is refused
	planOnlyGhost := &planpb.PlanFile{
		ArtifactMd5: "cc56a287aa327e84649b8737b964449e",
		TreeCommit:  "0123456789abcdef0123456789abcdef01234567",
		Sites:       []*planpb.PlanSite{{Id: "ghost_site", Body: "reference"}},
	}
	_, err = kernelplan.RequiredPackages(planOnlyGhost, segments, machineKind)
	if err == nil {
		t.Errorf("expected error for plan with only absent site, got nil")
	}

	// 3. Graph call site absent from plan is refused, naming the site
	planMissing := newSmallGraphPlan()
	planMissing.Sites = planMissing.Sites[:len(planMissing.Sites)-1] // drops op4_0
	_, err = kernelplan.RequiredPackages(planMissing, segments, machineKind)
	if err == nil || !strings.Contains(err.Error(), `"op4_0"`) {
		t.Errorf("expected error naming op4_0, got: %v", err)
	}
}

func TestRequiredPackages_PerSegmentNamingAndKnobs(t *testing.T) {
	artDir := createTestArtifact(t, smallGraphNodesJSON)
	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	machineKind := "gpu:nvidia l4"
	plan := newSmallGraphPlan()

	// With no segments given, returns nil
	pkgsNone, err := kernelplan.RequiredPackages(plan, nil, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages no segments: %v", err)
	}
	if len(pkgsNone) != 0 {
		t.Errorf("expected 0 packages when no segments given, got %d", len(pkgsNone))
	}

	// Host-op segments (submod_1, submod_3) produce no packages;
	// only dense segments (submod_0, submod_2, submod_4) are returned.
	pkgs, err := kernelplan.RequiredPackages(plan, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}
	if len(pkgs) != 3 {
		t.Fatalf("expected 3 packages, got %d", len(pkgs))
	}
	expectedNames := []string{"submod_0", "submod_2", "submod_4"}
	for i, p := range pkgs {
		if p.Name != expectedNames[i] {
			t.Errorf("package[%d] name = %q, want %q", i, p.Name, expectedNames[i])
		}
		if len(p.Key) != 32 {
			t.Errorf("package %s key %q has length %d, want 32", p.Name, p.Key, len(p.Key))
		}
	}

	// Changing one site's body in submod_2 renames submod_2 and no other segment (both halves).
	planModified := proto.Clone(plan).(*planpb.PlanFile)
	for _, s := range planModified.Sites {
		if s.Id == "op2_0" {
			s.Body = "gpu-fp8-gemv"
			break
		}
	}
	pkgsModified, err := kernelplan.RequiredPackages(planModified, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages modified: %v", err)
	}
	for i := range pkgs {
		if pkgs[i].Name == "submod_2" {
			// Half 1: submod_2 changed its package key.
			if pkgs[i].Key == pkgsModified[i].Key {
				t.Errorf("segment %s: package key did not change when body changed (%s)", pkgs[i].Name, pkgs[i].Key)
			}
		} else {
			// Half 2: all other segments retain their unchanged package keys.
			if pkgs[i].Key != pkgsModified[i].Key {
				t.Errorf("segment %s: package key changed unexpectedly from %s to %s", pkgs[i].Name, pkgs[i].Key, pkgsModified[i].Key)
			}
		}
	}

	// A plan site with an empty body is allowed (representing an unchosen body)
	// and names an empty body string in the package key.
	planEmptyBody := proto.Clone(plan).(*planpb.PlanFile)
	for _, s := range planEmptyBody.Sites {
		if s.Id == "op2_0" {
			s.Body = ""
			break
		}
	}
	pkgsEmptyBody, err := kernelplan.RequiredPackages(planEmptyBody, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages empty body: %v", err)
	}
	if pkgsEmptyBody[1].Key == pkgs[1].Key {
		t.Errorf("empty body should change package key compared to non-empty body")
	}

	// Outside-segment knobs on PlanFile.Knobs do not alter package keys.
	planWithOutside := proto.Clone(plan).(*planpb.PlanFile)
	planWithOutside.Knobs = &planpb.Knobs{
		GEMV_ROWS_MAX:  proto.Int32(32),
		FP4_NATIVE:     proto.Bool(true),
		ALLREDUCE_PULL: proto.Bool(true),
		FP4_DIRECT:     proto.Bool(true),
	}
	planCompiledOnly := proto.Clone(plan).(*planpb.PlanFile)
	planCompiledOnly.Knobs = &planpb.Knobs{
		GEMV_ROWS_MAX: proto.Int32(32),
		FP4_NATIVE:    proto.Bool(true),
	}
	pkgsWithOutside, err := kernelplan.RequiredPackages(planWithOutside, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages with outside knobs: %v", err)
	}
	pkgsCompiledOnly, err := kernelplan.RequiredPackages(planCompiledOnly, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages compiled only: %v", err)
	}
	for i := range pkgsWithOutside {
		if pkgsWithOutside[i].Key != pkgsCompiledOnly[i].Key {
			t.Errorf("outside knobs changed package key for %s: %s vs %s",
				pkgsWithOutside[i].Name, pkgsWithOutside[i].Key, pkgsCompiledOnly[i].Key)
		}
	}

	// Changing a compiled-in knob on PlanFile.Knobs renames every compiled package.
	planDifferent := proto.Clone(plan).(*planpb.PlanFile)
	planDifferent.Knobs = &planpb.Knobs{
		GEMV_ROWS_MAX: proto.Int32(64),
		FP4_NATIVE:    proto.Bool(true),
	}
	pkgsDifferentCompiled, err := kernelplan.RequiredPackages(planDifferent, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages different compiled: %v", err)
	}
	for i := range pkgs {
		if pkgs[i].Key == pkgsDifferentCompiled[i].Key {
			t.Errorf("compiled-in knob change did not change package key for %s (%s)",
				pkgs[i].Name, pkgs[i].Key)
		}
	}
}

func TestPackageKeyValidation(t *testing.T) {
	spec := baseSpec()

	// Missing software version
	s := spec
	s.SoftwareVersion = ""
	if _, err := kernelplan.PackageKey(s); err == nil {
		t.Error("expected error for empty SoftwareVersion, got nil")
	}

	// Missing artifact MD5
	s = spec
	s.ArtifactMD5 = ""
	if _, err := kernelplan.PackageKey(s); err == nil {
		t.Error("expected error for empty ArtifactMD5, got nil")
	}

	// Missing segment
	s = spec
	s.Segment = ""
	if _, err := kernelplan.PackageKey(s); err == nil {
		t.Error("expected error for empty Segment, got nil")
	}

	// Missing machine kind
	s = spec
	s.MachineKind = ""
	if _, err := kernelplan.PackageKey(s); err == nil {
		t.Error("expected error for empty MachineKind, got nil")
	}
}

func TestPackageKeyDeterminismAndLength(t *testing.T) {
	spec := baseSpec()
	key1, err := kernelplan.PackageKey(spec)
	if err != nil {
		t.Fatalf("PackageKey: %v", err)
	}
	key2, err := kernelplan.PackageKey(spec)
	if err != nil {
		t.Fatalf("PackageKey second call: %v", err)
	}
	if key1 != key2 {
		t.Errorf("PackageKey non-deterministic: %s != %s", key1, key2)
	}
	if len(key1) != 32 {
		t.Errorf("expected 32-character key, got length %d (%q)", len(key1), key1)
	}
}

func TestSoftwareVersionRenamesPackages(t *testing.T) {
	specA := baseSpec()
	specA.SoftwareVersion = "commit_aaaa"

	specB := baseSpec()
	specB.SoftwareVersion = "commit_bbbb"

	keyA, err := kernelplan.PackageKey(specA)
	if err != nil {
		t.Fatalf("PackageKey A: %v", err)
	}
	keyB, err := kernelplan.PackageKey(specB)
	if err != nil {
		t.Fatalf("PackageKey B: %v", err)
	}

	if keyA == keyB {
		t.Errorf("PackageKey did not change when SoftwareVersion changed (%s)", keyA)
	}
}

func TestArtifactMD5RenamesPackages(t *testing.T) {
	spec1 := baseSpec()
	spec1.ArtifactMD5 = "md5_model_1"

	spec2 := baseSpec()
	spec2.ArtifactMD5 = "md5_model_2"

	key1, err := kernelplan.PackageKey(spec1)
	if err != nil {
		t.Fatalf("PackageKey 1: %v", err)
	}
	key2, err := kernelplan.PackageKey(spec2)
	if err != nil {
		t.Fatalf("PackageKey 2: %v", err)
	}

	if key1 == key2 {
		t.Errorf("PackageKey did not change when ArtifactMD5 changed (%s)", key1)
	}
}

func TestSegmentNamesDistinguishPackages(t *testing.T) {
	spec0 := baseSpec()
	spec0.Segment = "submod_0"

	spec2 := baseSpec()
	spec2.Segment = "submod_2"

	key0, err := kernelplan.PackageKey(spec0)
	if err != nil {
		t.Fatalf("PackageKey submod_0: %v", err)
	}
	key2, err := kernelplan.PackageKey(spec2)
	if err != nil {
		t.Fatalf("PackageKey submod_2: %v", err)
	}

	if key0 == key2 {
		t.Errorf("PackageKey did not change when Segment changed (%s)", key0)
	}
}

func TestChosenBodiesChangePackageKey(t *testing.T) {
	specRef := baseSpec()
	specRef.Bodies = []string{"reference", "reference"}

	specGemv := baseSpec()
	specGemv.Bodies = []string{"reference", "gpu-fp8-gemv"}

	keyRef, err := kernelplan.PackageKey(specRef)
	if err != nil {
		t.Fatalf("PackageKey ref: %v", err)
	}
	keyGemv, err := kernelplan.PackageKey(specGemv)
	if err != nil {
		t.Fatalf("PackageKey gemv: %v", err)
	}

	if keyRef == keyGemv {
		t.Errorf("PackageKey did not change when chosen body changed (%s vs %s)", keyRef, keyGemv)
	}

	// Positional order: "x, none" vs "none, x"
	specOrder1 := baseSpec()
	specOrder1.Bodies = []string{"gpu-fp8-gemv", ""}

	specOrder2 := baseSpec()
	specOrder2.Bodies = []string{"", "gpu-fp8-gemv"}

	keyOrder1, _ := kernelplan.PackageKey(specOrder1)
	keyOrder2, _ := kernelplan.PackageKey(specOrder2)
	if keyOrder1 == keyOrder2 {
		t.Errorf("PackageKey collides when body positions differ (%s)", keyOrder1)
	}
}

func TestKnobsAffectPackageKey(t *testing.T) {
	specBase := baseSpec()
	keyBase, err := kernelplan.PackageKey(specBase)
	if err != nil {
		t.Fatalf("PackageKey base: %v", err)
	}

	// In-segment knob FOLD_POST changes key
	specFold := baseSpec()
	specFold.Knobs = &planpb.Knobs{FOLD_POST: proto.Bool(true)}
	keyFold, err := kernelplan.PackageKey(specFold)
	if err != nil {
		t.Fatalf("PackageKey fold: %v", err)
	}
	if keyFold == keyBase {
		t.Errorf("PackageKey did not change when in-segment knob FOLD_POST changed")
	}

	// In-segment knob GEMV_ROWS_MAX changes key
	specGemv := baseSpec()
	specGemv.Knobs = &planpb.Knobs{GEMV_ROWS_MAX: proto.Int32(32)}
	keyGemv, err := kernelplan.PackageKey(specGemv)
	if err != nil {
		t.Fatalf("PackageKey gemv: %v", err)
	}
	if keyGemv == keyBase {
		t.Errorf("PackageKey did not change when in-segment knob GEMV_ROWS_MAX changed")
	}

	// In-segment knob FFN_FRONT_EB changes key
	specFront := baseSpec()
	specFront.Knobs = &planpb.Knobs{FFN_FRONT_EB: proto.Int32(64)}
	keyFront, err := kernelplan.PackageKey(specFront)
	if err != nil {
		t.Fatalf("PackageKey front: %v", err)
	}
	if keyFront == keyBase {
		t.Errorf("PackageKey did not change when in-segment knob FFN_FRONT_EB changed")
	}

	// Outside-segment knobs do NOT change key
	specOutside := baseSpec()
	specOutside.Knobs = &planpb.Knobs{
		ALLREDUCE_PULL: proto.Bool(true),
		FP4_DIRECT:     proto.Bool(true),
		GEMV_PUSH_BN:   proto.Int32(8),
		FP4_GU_BN:      proto.Int32(64),
		FP4_DN_BK:      proto.Int32(128),
	}
	keyOutside, err := kernelplan.PackageKey(specOutside)
	if err != nil {
		t.Fatalf("PackageKey outside: %v", err)
	}
	if keyOutside != keyBase {
		t.Errorf("PackageKey changed when outside-segment knobs changed (got %s, want %s)", keyOutside, keyBase)
	}
}

func TestMachineKind(t *testing.T) {
	// Nil machine returns error
	if _, err := kernelplan.MachineKind(nil); err == nil {
		t.Error("expected error for MachineKind(nil), got nil")
	}

	// Nil probe returns error
	if _, err := kernelplan.MachineKindFromProbe(nil); err == nil {
		t.Error("expected error for MachineKindFromProbe(nil), got nil")
	}

	mL4, err := machines.Lookup("g2-standard-48")
	if err != nil {
		t.Fatalf("lookup g2-standard-48: %v", err)
	}
	mRTX, err := machines.Lookup("g4-standard-48")
	if err != nil {
		t.Fatalf("lookup g4-standard-48: %v", err)
	}
	mCPU, err := machines.Lookup("c3d-standard-60")
	if err != nil {
		t.Fatalf("lookup c3d-standard-60: %v", err)
	}
	mL4Sibling, err := machines.Lookup("g2-standard-24")
	if err != nil {
		t.Fatalf("lookup g2-standard-24: %v", err)
	}

	kindL4, err := kernelplan.MachineKind(mL4)
	if err != nil || kindL4 != "gpu:nvidia l4" {
		t.Errorf("mL4 machineKind = %q, want %q, err=%v", kindL4, "gpu:nvidia l4", err)
	}
	kindRTX, err := kernelplan.MachineKind(mRTX)
	if err != nil || kindRTX != "gpu:nvidia rtx pro 6000" {
		t.Errorf("mRTX machineKind = %q, want %q, err=%v", kindRTX, "gpu:nvidia rtx pro 6000", err)
	}
	kindL4Sibling, err := kernelplan.MachineKind(mL4Sibling)
	if err != nil || kindL4Sibling != "gpu:nvidia l4" {
		t.Errorf("mL4Sibling machineKind = %q, want %q, err=%v", kindL4Sibling, "gpu:nvidia l4", err)
	}

	// MachineKindFromName
	nameKind, err := kernelplan.MachineKindFromName("g2-standard-48")
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}
	if nameKind != "gpu:nvidia l4" {
		t.Errorf("MachineKindFromName = %q, want %q", nameKind, "gpu:nvidia l4")
	}

	// MachineKindFromProbe
	pGPU := &v1alpha1.Probe{Gpu: "NVIDIA L4"}
	probeGPU, err := kernelplan.MachineKindFromProbe(pGPU)
	if err != nil || probeGPU != "gpu:nvidia l4" {
		t.Errorf("probe GPU got %q, want %q, err=%v", probeGPU, "gpu:nvidia l4", err)
	}
	pCPU := &v1alpha1.Probe{Cpu: "AMD EPYC", Simd: "avx512"}
	probeCPU, err := kernelplan.MachineKindFromProbe(pCPU)
	if err != nil || probeCPU != "cpu:amd epyc:avx512" {
		t.Errorf("probe CPU got %q, want %q, err=%v", probeCPU, "cpu:amd epyc:avx512", err)
	}

	// Different machine kinds change package keys
	specL4 := baseSpec()
	specL4.MachineKind = kindL4
	specRTX := baseSpec()
	specRTX.MachineKind = kindRTX
	specCPU := baseSpec()
	specCPU.MachineKind, _ = kernelplan.MachineKind(mCPU)

	keyL4, _ := kernelplan.PackageKey(specL4)
	keyRTX, _ := kernelplan.PackageKey(specRTX)
	keyCPU, _ := kernelplan.PackageKey(specCPU)

	if keyL4 == keyRTX {
		t.Errorf("L4 and RTX produced identical package keys: %s", keyL4)
	}
	if keyL4 == keyCPU {
		t.Errorf("L4 and CPU produced identical package keys: %s", keyL4)
	}
}

func TestCheckPackages(t *testing.T) {
	ctx := t.Context()
	artDir := createTestArtifact(t, smallGraphNodesJSON)
	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	machineKind := "gpu:nvidia l4"
	plan := newSmallGraphPlan()

	pkgs, err := kernelplan.RequiredPackages(plan, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}
	if len(pkgs) != 3 {
		t.Fatalf("expected 3 packages, got %d", len(pkgs))
	}

	// 1. Empty store: all missing
	emptyStore := newStandInStore()
	repEmpty, err := kernelplan.CheckPackages(ctx, emptyStore, "test-model", plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages empty: %v", err)
	}
	if len(repEmpty.Required) != 3 || len(repEmpty.Missing) != 3 || len(repEmpty.Present) != 0 {
		t.Errorf("empty report mismatch: %+v", repEmpty)
	}

	// 2. Partial store: reports present and missing accurately
	partialStore := newStandInStore()
	partialStore.packages[pkgs[0].Key] = true
	partialStore.packages[pkgs[2].Key] = true
	repPartial, err := kernelplan.CheckPackages(ctx, partialStore, "test-model", plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages partial: %v", err)
	}
	if len(repPartial.Required) != 3 || len(repPartial.Present) != 2 || len(repPartial.Missing) != 1 {
		t.Errorf("partial report mismatch: %+v", repPartial)
	}
	if repPartial.Missing[0].Name != "submod_2" {
		t.Errorf("expected missing submod_2, got %s", repPartial.Missing[0].Name)
	}

	// CheckPackagesFromArtifact discovers segments from the artifact directory
	repArtifact, err := kernelplan.CheckPackagesFromArtifact(ctx, partialStore, "test-model", plan, artDir, machineKind)
	if err != nil {
		t.Fatalf("CheckPackagesFromArtifact: %v", err)
	}
	if len(repArtifact.Required) != 3 || len(repArtifact.Present) != 2 || len(repArtifact.Missing) != 1 {
		t.Errorf("artifact report mismatch: %+v", repArtifact)
	}
}

func TestPackageCompileFailureThreeCases(t *testing.T) {
	ctx := t.Context()
	artDir := createTestArtifact(t, smallGraphNodesJSON)
	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	machineKind := "gpu:nvidia l4"
	plan := newSmallGraphPlan()

	pkgs, err := kernelplan.RequiredPackages(plan, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}
	if len(pkgs) != 3 {
		t.Fatalf("expected 3 packages, got %d", len(pkgs))
	}

	store := newStandInStore()
	model := "test-model"

	// Case 1: After a compile task reports a failure, the store holds a record of it.
	// Asking which packages are present, missing or known to fail for the same plan,
	// machine kind and software version shows the package as known to fail, with the
	// recorded reason.
	failSubmod2 := &planpb.CompileFailure{
		Package:         pkgs[1], // submod_2
		MachineKind:     machineKind,
		SoftwareVersion: plan.TreeCommit,
		Reason:          "Float8_e8m0fnu in namespace at does not name a type",
	}
	if err := store.RecordCompileFailure(ctx, model, failSubmod2); err != nil {
		t.Fatalf("RecordCompileFailure: %v", err)
	}

	rep1, err := kernelplan.CheckPackages(ctx, store, model, plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages Case 1: %v", err)
	}
	if len(rep1.Failed) != 1 {
		t.Fatalf("Case 1: got %d failed packages, want 1", len(rep1.Failed))
	}
	if rep1.Failed[0].Package.Name != "submod_2" {
		t.Errorf("Case 1: failed package name = %q, want submod_2", rep1.Failed[0].Package.Name)
	}
	if rep1.Failed[0].Reason != failSubmod2.Reason {
		t.Errorf("Case 1: failed reason = %q, want %q", rep1.Failed[0].Reason, failSubmod2.Reason)
	}
	for _, m := range rep1.Missing {
		if m.Name == "submod_2" {
			t.Errorf("Case 1: submod_2 should not be in missing packages")
		}
	}
	if len(rep1.Missing) != 2 {
		t.Errorf("Case 1: got %d missing packages, want 2", len(rep1.Missing))
	}

	// Case 2: A second request to compile that plan does not ask for that package again;
	// a different software version, or a different machine kind, does ask.
	rep2Same, err := kernelplan.CheckPackages(ctx, store, model, plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages Case 2 same: %v", err)
	}
	for _, m := range rep2Same.Missing {
		if m.Name == "submod_2" {
			t.Errorf("Case 2: second request asked for submod_2 again in missing")
		}
	}
	if len(rep2Same.Failed) != 1 || rep2Same.Failed[0].Package.Name != "submod_2" {
		t.Errorf("Case 2: expected submod_2 in failed, got %+v", rep2Same.Failed)
	}

	// Different software version DOES ask
	planNewVer := proto.Clone(plan).(*planpb.PlanFile)
	planNewVer.TreeCommit = "new_software_commit_999"
	rep2NewVer, err := kernelplan.CheckPackages(ctx, store, model, planNewVer, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages Case 2 new version: %v", err)
	}
	foundInMissing := false
	for _, m := range rep2NewVer.Missing {
		if m.Name == "submod_2" {
			foundInMissing = true
			break
		}
	}
	if !foundInMissing {
		t.Errorf("Case 2: different software version should ask for submod_2 in missing")
	}
	if len(rep2NewVer.Failed) != 0 {
		t.Errorf("Case 2: different software version should have 0 failed, got %d", len(rep2NewVer.Failed))
	}

	// Different machine kind DOES ask
	machineKindCPU := "cpu:intel xeon:avx-512"
	rep2NewMachine, err := kernelplan.CheckPackages(ctx, store, model, plan, segments, machineKindCPU)
	if err != nil {
		t.Fatalf("CheckPackages Case 2 new machine kind: %v", err)
	}
	foundInMissingCPU := false
	for _, m := range rep2NewMachine.Missing {
		if m.Name == "submod_2" {
			foundInMissingCPU = true
			break
		}
	}
	if !foundInMissingCPU {
		t.Errorf("Case 2: different machine kind should ask for submod_2 in missing")
	}
	if len(rep2NewMachine.Failed) != 0 {
		t.Errorf("Case 2: different machine kind should have 0 failed, got %d", len(rep2NewMachine.Failed))
	}

	// Case 3: A record can be cleared (a deliberate retry), and a later
	// successful build of the package replaces it.
	// 3a. Cleared
	if err := store.ClearCompileFailure(ctx, model, pkgs[1].Key); err != nil {
		t.Fatalf("ClearCompileFailure: %v", err)
	}
	rep3Cleared, err := kernelplan.CheckPackages(ctx, store, model, plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages Case 3 cleared: %v", err)
	}
	foundInMissingCleared := false
	for _, m := range rep3Cleared.Missing {
		if m.Name == "submod_2" {
			foundInMissingCleared = true
			break
		}
	}
	if !foundInMissingCleared {
		t.Errorf("Case 3: cleared record should ask for submod_2 in missing")
	}
	if len(rep3Cleared.Failed) != 0 {
		t.Errorf("Case 3: cleared record should have 0 failed, got %d", len(rep3Cleared.Failed))
	}

	// 3b. A later successful build replaces it
	// Re-record the failure to demonstrate replacement
	if err := store.RecordCompileFailure(ctx, model, failSubmod2); err != nil {
		t.Fatalf("RecordCompileFailure: %v", err)
	}
	// Simulate successful build of submod_2 placed into store
	store.packages[pkgs[1].Key] = true
	rep3Success, err := kernelplan.CheckPackages(ctx, store, model, plan, segments, machineKind)
	if err != nil {
		t.Fatalf("CheckPackages Case 3 success: %v", err)
	}
	foundInPresent := false
	for _, p := range rep3Success.Present {
		if p.Name == "submod_2" {
			foundInPresent = true
			break
		}
	}
	if !foundInPresent {
		t.Errorf("Case 3: successful build should be in present packages")
	}
	if len(rep3Success.Failed) != 0 {
		t.Errorf("Case 3: successful build should replace failure, got %d failed", len(rep3Success.Failed))
	}
	for _, m := range rep3Success.Missing {
		if m.Name == "submod_2" {
			t.Errorf("Case 3: successful build should not be in missing packages")
		}
	}
}

func TestKnobsDeclareCompiledIn(t *testing.T) {
	fields := (&planpb.Knobs{}).ProtoReflect().Descriptor().Fields()
	if fields.Len() == 0 {
		t.Fatal("planpb.Knobs has no fields")
	}
	for i := 0; i < fields.Len(); i++ {
		fd := fields.Get(i)
		_, err := kernelplan.IsCompiledInKnob(fd)
		if err != nil {
			t.Errorf("knob %s does not declare whether it is compiled in: %v", fd.Name(), err)
		}
	}
}

func TestIsCompiledInKnob_MissingDeclarationFails(t *testing.T) {
	fd := (&planpb.SiteEntry{}).ProtoReflect().Descriptor().Fields().ByName("impl")
	if fd == nil {
		t.Fatal("SiteEntry.impl field descriptor not found")
	}
	_, err := kernelplan.IsCompiledInKnob(fd)
	if err == nil {
		t.Fatal("expected error for field lacking compiled_in option, got nil")
	}
}

func TestPackageNamesPreserved(t *testing.T) {
	artDir := createTestArtifact(t, smallGraphNodesJSON)
	segments, err := kernelplan.ReadSegments(artDir)
	if err != nil {
		t.Fatalf("ReadSegments: %v", err)
	}
	machineKind := "gpu:nvidia l4"
	plan := newSmallGraphPlan()

	pkgs, err := kernelplan.RequiredPackages(plan, segments, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}

	wantKeys := map[string]string{
		"submod_0": "dd581feca2b00d45fe243dda571ffe1f",
		"submod_2": "d9a5ffa4a8886f85ec4ccd83c9ea00ec",
		"submod_4": "23212f5636800be5f95d2e6d0c00543d",
	}
	for _, p := range pkgs {
		want, ok := wantKeys[p.Name]
		if !ok {
			t.Errorf("unexpected package %s", p.Name)
			continue
		}
		if p.Key != want {
			t.Errorf("package %s key changed: got %s, want %s", p.Name, p.Key, want)
		}
	}
}
