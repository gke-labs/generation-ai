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
	"context"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

type standInStore map[string]bool

func (s standInStore) HasPackage(ctx context.Context, model, key string) (bool, error) {
	return s[key], nil
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

func loadTestPlan(t *testing.T) *planpb.PlanFile {
	t.Helper()
	fixturePath := filepath.Join("testdata", "golden_call_sites_plan.json")
	var plan planpb.PlanFile
	if err := kernelplan.ReadJSON(fixturePath, &plan); err != nil {
		t.Fatalf("failed to read plan fixture %s: %v", fixturePath, err)
	}
	return &plan
}

func TestTinyDeepSeekRequiredPackages(t *testing.T) {
	plan := loadTestPlan(t)
	machineKind, err := kernelplan.MachineKindFromName("g2-standard-48")
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}

	// With no segments given, RequiredPackages returns nil
	pkgsNone, err := kernelplan.RequiredPackages(plan, nil, machineKind)
	if err != nil {
		t.Fatalf("RequiredPackages no segments: %v", err)
	}
	if len(pkgsNone) != 0 {
		t.Errorf("expected 0 packages when no segments given, got %d", len(pkgsNone))
	}

	segments := []string{"submod_0", "submod_2", "submod_4", "submod_6", "submod_8", "submod_10"}
	pkgs, err := kernelplan.RequiredPackages(plan, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}
	if len(pkgs) != len(segments) {
		t.Fatalf("expected %d packages, got %d", len(segments), len(pkgs))
	}

	seenKeys := make(map[string]string)
	for _, p := range pkgs {
		if p.Key == "" {
			t.Errorf("package %s has empty key", p.Name)
		}
		if len(p.Key) != 32 {
			t.Errorf("package %s key %q has length %d, want 32", p.Name, p.Key, len(p.Key))
		}
		if prev, ok := seenKeys[p.Key]; ok {
			t.Errorf("duplicate package key %s for %s and %s", p.Key, prev, p.Name)
		}
		seenKeys[p.Key] = p.Name
	}

	// Changing one site's body in the plan renames the packages.
	planModified := proto.Clone(plan).(*planpb.PlanFile)
	changed := false
	for _, s := range planModified.Sites {
		if s.Id == "fp8_linear_4" {
			s.Body = "gpu-fp8-gemv"
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("site fp8_linear_4 not found in plan")
	}

	pkgsModified, err := kernelplan.RequiredPackages(planModified, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages modified: %v", err)
	}
	for i := range pkgs {
		if pkgs[i].Key == pkgsModified[i].Key {
			t.Errorf("segment %s: package key did not change when body changed (%s)", pkgs[i].Name, pkgs[i].Key)
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
	plan := loadTestPlan(t)
	machineKind := "gpu:nvidia l4"
	segments := []string{"submod_0", "submod_2", "submod_4"}

	pkgs, err := kernelplan.RequiredPackages(plan, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
	}

	// 1. Empty store: all missing
	emptyStore := standInStore{}
	repEmpty, err := kernelplan.CheckPackages(ctx, emptyStore, "tiny-deepseek", plan, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("CheckPackages empty: %v", err)
	}
	if len(repEmpty.Required) != 3 || len(repEmpty.Missing) != 3 || len(repEmpty.Present) != 0 {
		t.Errorf("empty report mismatch: %+v", repEmpty)
	}

	// 2. Partial store: reports present and missing accurately
	partialStore := standInStore{pkgs[0].Key: true, pkgs[2].Key: true}
	repPartial, err := kernelplan.CheckPackages(ctx, partialStore, "tiny-deepseek", plan, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("CheckPackages partial: %v", err)
	}
	if len(repPartial.Required) != 3 || len(repPartial.Present) != 2 || len(repPartial.Missing) != 1 {
		t.Errorf("partial report mismatch: %+v", repPartial)
	}
	if repPartial.Missing[0].Name != "submod_2" {
		t.Errorf("expected missing submod_2, got %s", repPartial.Missing[0].Name)
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

func TestTinyDeepSeekPackageNamesPreserved(t *testing.T) {
	plan := loadTestPlan(t)
	machineKind, err := kernelplan.MachineKindFromName("g2-standard-48")
	if err != nil {
		t.Fatalf("MachineKindFromName: %v", err)
	}

	segments := []string{"submod_0", "submod_2", "submod_4", "submod_6", "submod_8", "submod_10"}

	// Expected package keys for the tiny DeepSeek plan before this move:
	wantKeys := map[string]string{
		"submod_0":  "26cf3dc17c3641315dce303c717ccc9c",
		"submod_2":  "66cf00e22dfbbfa175f85dfd55a67a18",
		"submod_4":  "f3d25c15df6c3fb2bbeacf850679c763",
		"submod_6":  "f40056fc0db3495952cab6781bbf89e3",
		"submod_8":  "6942bb8d3a973eb9dfb8001f13dd477c",
		"submod_10": "a878b49047f200b3a4dc12a61d365316",
	}

	pkgs, err := kernelplan.RequiredPackages(plan, nil, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages: %v", err)
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

	// Outside-segment knobs do not alter package keys.
	knobsWithOutside := &planpb.Knobs{
		GEMV_ROWS_MAX:  proto.Int32(32),
		FP4_NATIVE:     proto.Bool(true),
		ALLREDUCE_PULL: proto.Bool(true),
		FP4_DIRECT:     proto.Bool(true),
	}
	knobsCompiledOnly := &planpb.Knobs{
		GEMV_ROWS_MAX: proto.Int32(32),
		FP4_NATIVE:    proto.Bool(true),
	}
	pkgsWithOutside, err := kernelplan.RequiredPackages(plan, knobsWithOutside, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages with outside knobs: %v", err)
	}
	pkgsCompiledOnly, err := kernelplan.RequiredPackages(plan, knobsCompiledOnly, machineKind, segments...)
	if err != nil {
		t.Fatalf("RequiredPackages compiled only: %v", err)
	}
	for i := range pkgsWithOutside {
		if pkgsWithOutside[i].Key != pkgsCompiledOnly[i].Key {
			t.Errorf("outside knobs changed package key for %s: %s vs %s",
				pkgsWithOutside[i].Name, pkgsWithOutside[i].Key, pkgsCompiledOnly[i].Key)
		}
	}
}
