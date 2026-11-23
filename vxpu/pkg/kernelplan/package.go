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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/descriptorpb"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// Package is one compiled package required by a plan.
type Package = planpb.Package

// PackageReport records which compiled packages a plan requires, and
// which exist in or are missing from the package store.
type PackageReport = planpb.PackageReport

// PackageStore answers whether compiled packages exist or have failed to compile, and clears failures.
type PackageStore interface {
	// HasPackage reports whether model contains a compiled package for key.
	HasPackage(ctx context.Context, model, key string) (bool, error)
	// GetCompileFailure returns any recorded compilation failure for key under model.
	GetCompileFailure(ctx context.Context, model, key string) (*planpb.CompileFailure, error)
	// RecordCompileFailure records a compilation failure under model.
	RecordCompileFailure(ctx context.Context, model string, failure *planpb.CompileFailure) error
	// ClearCompileFailure clears recorded compilation failure for package key under model.
	ClearCompileFailure(ctx context.Context, model, key string) error

	// GetBodyCompileOutcome returns the last observed compilation outcome for op and body on machineKind and softwareVersion.
	// Returns (nil, nil) if no outcome has been observed (not known).
	GetBodyCompileOutcome(ctx context.Context, op, body, machineKind, softwareVersion string) (*planpb.BodyCompileOutcome, error)
	// RecordBodyCompileOutcome records the compilation outcome for a body on machineKind and softwareVersion,
	// replacing any previous record under the same identity.
	RecordBodyCompileOutcome(ctx context.Context, outcome *planpb.BodyCompileOutcome) error
	// ListBodyCompileOutcomes lists observed body compilation outcomes for machineKind.
	ListBodyCompileOutcomes(ctx context.Context, machineKind string) ([]*planpb.BodyCompileOutcome, error)
}

// MachineKind derives the hardware target identifier from a catalog machine.
func MachineKind(m *machines.Machine) (string, error) {
	if m == nil {
		return "", errors.New("machine is required")
	}
	if m.GPU != nil && m.GPUCount > 0 && m.GPU.Name != "" {
		return "gpu:" + strings.ToLower(strings.TrimSpace(m.GPU.Name)), nil
	}
	return fmt.Sprintf("cpu:%s:%s",
		strings.ToLower(strings.TrimSpace(m.CPU.Model)),
		strings.ToLower(strings.TrimSpace(m.CPU.SIMD)),
	), nil
}

// MachineKindFromProbe derives the hardware target identifier from an executor probe.
func MachineKindFromProbe(p *v1alpha1.Probe) (string, error) {
	if p == nil {
		return "", errors.New("probe is required")
	}
	if p.Gpu != "" {
		return "gpu:" + strings.ToLower(strings.TrimSpace(p.Gpu)), nil
	}
	return fmt.Sprintf("cpu:%s:%s",
		strings.ToLower(strings.TrimSpace(p.Cpu)),
		strings.ToLower(strings.TrimSpace(p.Simd)),
	), nil
}

// MachineKindFromName resolves name in the machine catalog.
func MachineKindFromName(name string) (string, error) {
	m, err := machines.Lookup(name)
	if err != nil {
		return "", err
	}
	return MachineKind(m)
}

// IsCompiledInKnob reports whether fd compiles into a package segment.
// It returns an error if fd does not declare whether it is compiled in.
func IsCompiledInKnob(fd protoreflect.FieldDescriptor) (bool, error) {
	opts, ok := fd.Options().(*descriptorpb.FieldOptions)
	if !ok || opts == nil || !proto.HasExtension(opts, planpb.E_CompiledIn) {
		return false, fmt.Errorf("knob %s does not declare whether it is compiled into a package", fd.Name())
	}
	return proto.GetExtension(opts, planpb.E_CompiledIn).(bool), nil
}

// InSegmentKnobs formats compiled-in tuning knobs in deterministic sorted order.
func InSegmentKnobs(knobs *planpb.Knobs) (string, error) {
	if knobs == nil {
		return "", nil
	}
	m := knobs.ProtoReflect()
	var pairs []string
	var err error
	m.Range(func(fd protoreflect.FieldDescriptor, val protoreflect.Value) bool {
		compiled, kerr := IsCompiledInKnob(fd)
		if kerr != nil {
			err = kerr
			return false
		}
		if compiled {
			pairs = append(pairs, fmt.Sprintf("%s=%v;", fd.Name(), val.Interface()))
		}
		return true
	})
	if err != nil {
		return "", err
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ""), nil
}

// PackageSpec specifies the planner inputs that determine a compiled package.
type PackageSpec struct {
	// SoftwareVersion is the executor software version (image digest or tree commit).
	SoftwareVersion string
	// ArtifactMD5 is the checksum of the model artifact's manifest and binding bytes.
	ArtifactMD5 string
	// Segment identifies which segment of the model is compiled (e.g. "submod_0").
	Segment string
	// Bodies names the kernel implementation bodies chosen for the segment in execution order.
	Bodies []string
	// Knobs are the tuning knobs planned for the model, set by the planner before execution.
	Knobs *planpb.Knobs
	// MachineKind is the hardware compilation target.
	MachineKind string
}

// PackageKey computes the deterministic content key for a compiled segment package.
func PackageKey(spec PackageSpec) (string, error) {
	if spec.SoftwareVersion == "" {
		return "", errors.New("software version is required")
	}
	if spec.ArtifactMD5 == "" {
		return "", errors.New("artifact MD5 is required")
	}
	if spec.Segment == "" {
		return "", errors.New("segment is required")
	}
	if spec.MachineKind == "" {
		return "", errors.New("machine kind is required")
	}

	h := sha256.New()
	writeField := func(s string) {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}

	writeField(spec.SoftwareVersion)
	writeField(spec.ArtifactMD5)
	writeField(spec.Segment)
	writeField(spec.MachineKind)

	if len(spec.Bodies) > 0 {
		for i, b := range spec.Bodies {
			writeField(fmt.Sprintf("B%d=%s", i, b))
		}
	} else {
		writeField("B-")
	}

	inSegment, err := InSegmentKnobs(spec.Knobs)
	if err != nil {
		return "", err
	}
	writeField(inSegment)

	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:16]), nil
}

// RequiredPackages lists the compiled packages plan requires for machineKind across segments.
// Host-op segments do not produce packages. Each compiled package is named from its own
// segment: the kernel implementation bodies chosen for the call sites in that segment,
// and the compiled-in tuning knobs from plan.Knobs. A plan site with an empty body is
// allowed (for example, when an operator's choice falls back to default unoptimized execution
// or where no specific kernel body was selected); it is named with an empty body string ("")
// at that call site's positional index. Until knobs are partitioned by segment, a changed
// compiled-in knob renames every compiled package; outside knobs leave all package names unchanged.
func RequiredPackages(
	plan *planpb.PlanFile,
	segments []*planpb.Segment,
	machineKind string,
) ([]*Package, error) {
	if plan == nil {
		return nil, errors.New("plan is required")
	}
	if machineKind == "" {
		return nil, errors.New("machine kind is required")
	}
	if len(segments) == 0 {
		return nil, nil
	}

	if err := JoinPlan(segments, plan); err != nil {
		return nil, err
	}

	var pkgs []*Package
	for _, seg := range segments {
		if seg == nil || seg.Host {
			continue
		}
		var bodies []string
		for _, cs := range seg.CallSites {
			if cs == nil || cs.Node == "" {
				continue
			}
			bodies = append(bodies, cs.Body)
		}
		key, err := PackageKey(PackageSpec{
			SoftwareVersion: plan.TreeCommit,
			ArtifactMD5:     plan.ArtifactMd5,
			Segment:         seg.Name,
			Bodies:          bodies,
			Knobs:           plan.Knobs,
			MachineKind:     machineKind,
		})
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, &Package{
			Name: seg.Name,
			Key:  key,
		})
	}
	return pkgs, nil
}

// RequiredPackagesFromArtifact reads the exported graph segments from artifactDir and
// lists the compiled packages plan requires for machineKind.
func RequiredPackagesFromArtifact(
	plan *planpb.PlanFile,
	artifactDir string,
	machineKind string,
) ([]*Package, error) {
	segments, err := ReadSegments(artifactDir)
	if err != nil {
		return nil, fmt.Errorf("read segments: %w", err)
	}
	return RequiredPackages(plan, segments, machineKind)
}

// CheckPackages reports which compiled packages plan requires for machineKind across segments,
// and which of those are present in or missing from store, or known to fail.
func CheckPackages(
	ctx context.Context,
	store PackageStore,
	model string,
	plan *planpb.PlanFile,
	segments []*planpb.Segment,
	machineKind string,
) (*PackageReport, error) {
	required, err := RequiredPackages(plan, segments, machineKind)
	if err != nil {
		return nil, fmt.Errorf("computing required packages: %w", err)
	}
	report := &PackageReport{
		Required:     required,
		Present:      make([]*Package, 0),
		Missing:      make([]*Package, 0),
		Failed:       make([]*planpb.CompileFailure, 0),
		BodyOutcomes: make([]*planpb.BodyCompileOutcome, 0),
		MachineKind:  machineKind,
	}
	if store == nil {
		report.Missing = append(report.Missing, required...)
		return report, nil
	}
	bodyOutcomes, err := store.ListBodyCompileOutcomes(ctx, machineKind)
	if err != nil {
		return nil, fmt.Errorf("listing body compile outcomes for %s: %w", machineKind, err)
	}
	if len(bodyOutcomes) > 0 {
		report.BodyOutcomes = append(report.BodyOutcomes, bodyOutcomes...)
	}
	for _, pkg := range required {
		exists, err := store.HasPackage(ctx, model, pkg.Key)
		if err != nil {
			return nil, fmt.Errorf("checking package %s (%s): %w", pkg.Name, pkg.Key, err)
		}
		if exists {
			report.Present = append(report.Present, pkg)
			continue
		}
		fail, err := store.GetCompileFailure(ctx, model, pkg.Key)
		if err != nil {
			return nil, fmt.Errorf("checking failure for package %s (%s): %w", pkg.Name, pkg.Key, err)
		}
		if fail != nil {
			report.Failed = append(report.Failed, fail)
		} else {
			report.Missing = append(report.Missing, pkg)
		}
	}
	return report, nil
}

// CheckPackagesFromArtifact discovers segments from the artifact directory and reports
// which compiled packages plan requires for machineKind.
func CheckPackagesFromArtifact(
	ctx context.Context,
	store PackageStore,
	model string,
	plan *planpb.PlanFile,
	artifactDir string,
	machineKind string,
) (*PackageReport, error) {
	segments, err := ReadSegments(artifactDir)
	if err != nil {
		return nil, fmt.Errorf("read segments: %w", err)
	}
	return CheckPackages(ctx, store, model, plan, segments, machineKind)
}
