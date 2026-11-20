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

	"google.golang.org/protobuf/reflect/protoreflect"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// Package is one compiled package required by a plan.
type Package = planpb.Package

// PackageReport records which compiled packages a plan requires, and
// which exist in or are missing from the package store.
type PackageReport = planpb.PackageReport

// PackageStore answers whether a compiled package exists for a key.
type PackageStore interface {
	// HasPackage reports whether model contains a compiled package for key.
	HasPackage(ctx context.Context, model, key string) (bool, error)
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

// outsideKnobs explicitly enumerates the tuning knobs that execute outside
// dense compiled segments (routed expert kernels and collectives).
var outsideKnobs = map[string]bool{
	"ALLREDUCE_BLOCK":            true,
	"ALLREDUCE_PULL":             true,
	"ALLREDUCE_PUSH_BF16":        true,
	"ALLREDUCE_SCATTER":          true,
	"ALLREDUCE_SCATTER_MIN_ROWS": true,
	"DRAFT_ALLREDUCE_PULL":       true,
	"FP4_DIRECT":                 true,
	"FP4_DIRECT_COMPACT":         true,
	"FP4_DIRECT_FIRST":           true,
	"FP4_DIRECT_GROUP":           true,
	"FP4_DIRECT_MAX":             true,
	"FP4_DN_BK":                  true,
	"FP4_DN_BN":                  true,
	"FP4_DN_KS":                  true,
	"FP4_DN_STAGES":              true,
	"FP4_DN_WARPS":               true,
	"FP4_FUSE_EPI":               true,
	"FP4_GD_BK":                  true,
	"FP4_GD_BM":                  true,
	"FP4_GD_BN":                  true,
	"FP4_GD_STAGES":              true,
	"FP4_GD_WARPS":               true,
	"FP4_GEMM_MIN_PAIRS":         true,
	"FP4_GG_BK":                  true,
	"FP4_GG_BM":                  true,
	"FP4_GG_BN":                  true,
	"FP4_GG_STAGES":              true,
	"FP4_GG_WARPS":               true,
	"FP4_GU_BK":                  true,
	"FP4_GU_BN":                  true,
	"FP4_GU_KS":                  true,
	"FP4_GU_STAGES":              true,
	"FP4_GU_WARPS":               true,
	"FP4_ONE_LAUNCH":             true,
	"FP4_ONE_PER_SM":             true,
	"FP4_ONE_STAGES":             true,
	"FP4_ONE_WARPS":              true,
	"FP4_PDL":                    true,
	"GEMV_PUSH_BN":               true,
}

func isCompiledInKnob(name string) bool {
	return !outsideKnobs[name]
}

// InSegmentKnobs formats compiled-in tuning knobs in deterministic sorted order.
func InSegmentKnobs(knobs *planpb.Knobs) string {
	if knobs == nil {
		return ""
	}
	m := knobs.ProtoReflect()
	var pairs []string
	m.Range(func(fd protoreflect.FieldDescriptor, val protoreflect.Value) bool {
		name := string(fd.Name())
		if isCompiledInKnob(name) {
			pairs = append(pairs, fmt.Sprintf("%s=%v;", name, val.Interface()))
		}
		return true
	})
	sort.Strings(pairs)
	return strings.Join(pairs, "")
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
	// Knobs are the tuning knobs applied to the plan.
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

	writeField(InSegmentKnobs(spec.Knobs))

	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:16]), nil
}

// RequiredPackages lists the compiled packages plan requires for machineKind across segments.
// Until the planner reads segment structure from the artifact, each segment's package is
// named from all of the plan's chosen bodies and compiled-in knobs.
func RequiredPackages(
	plan *planpb.PlanFile,
	knobs *planpb.Knobs,
	machineKind string,
	segments ...string,
) ([]*Package, error) {
	if plan == nil {
		return nil, errors.New("plan is required")
	}
	if machineKind == "" {
		return nil, errors.New("machine kind is required")
	}
	var bodies []string
	for _, s := range plan.Sites {
		if s != nil && s.Body != "" {
			bodies = append(bodies, fmt.Sprintf("%s=%s", s.Id, s.Body))
		}
	}
	sort.Strings(bodies)

	if len(segments) == 0 {
		return nil, nil
	}

	pkgs := make([]*Package, 0, len(segments))
	for _, seg := range segments {
		key, err := PackageKey(PackageSpec{
			SoftwareVersion: plan.TreeCommit,
			ArtifactMD5:     plan.ArtifactMd5,
			Segment:         seg,
			Bodies:          bodies,
			Knobs:           knobs,
			MachineKind:     machineKind,
		})
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, &Package{
			Name: seg,
			Key:  key,
		})
	}
	return pkgs, nil
}

// CheckPackages reports which compiled packages plan requires for machineKind,
// and which of those are present in or missing from store.
func CheckPackages(
	ctx context.Context,
	store PackageStore,
	model string,
	plan *planpb.PlanFile,
	knobs *planpb.Knobs,
	machineKind string,
	segments ...string,
) (*PackageReport, error) {
	required, err := RequiredPackages(plan, knobs, machineKind, segments...)
	if err != nil {
		return nil, fmt.Errorf("computing required packages: %w", err)
	}
	report := &PackageReport{
		Required: required,
		Present:  make([]*Package, 0),
		Missing:  make([]*Package, 0),
	}
	if store == nil {
		report.Missing = append(report.Missing, required...)
		return report, nil
	}
	for _, pkg := range required {
		exists, err := store.HasPackage(ctx, model, pkg.Key)
		if err != nil {
			return nil, fmt.Errorf("checking package %s (%s): %w", pkg.Name, pkg.Key, err)
		}
		if exists {
			report.Present = append(report.Present, pkg)
		} else {
			report.Missing = append(report.Missing, pkg)
		}
	}
	return report, nil
}
