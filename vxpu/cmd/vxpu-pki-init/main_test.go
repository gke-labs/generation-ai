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

package main

import (
	"bytes"
	"crypto/x509"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func TestRunOutDir(t *testing.T) {
	// Setting KUBECONFIG to a nonexistent file verifies --out-dir never touches Kubernetes.
	t.Setenv("KUBECONFIG", filepath.Join(t.TempDir(), "does-not-exist"))
	outDir := filepath.Join(t.TempDir(), "tls")

	if err := run(t.Context(), []string{"--out-dir", outDir}); err != nil {
		t.Fatalf("first run --out-dir: %v", err)
	}

	caCertPEM, err := os.ReadFile(filepath.Join(outDir, "ca/ca.crt"))
	if err != nil {
		t.Fatalf("reading ca.crt: %v", err)
	}
	caKeyPEM, err := os.ReadFile(filepath.Join(outDir, "ca/ca.key"))
	if err != nil {
		t.Fatalf("reading ca.key: %v", err)
	}
	ca, err := pki.ParseKeyPair(caCertPEM, caKeyPEM)
	if err != nil {
		t.Fatalf("parsing CA: %v", err)
	}

	coordCertPEM, err := os.ReadFile(filepath.Join(outDir, "coordinator/tls.crt"))
	if err != nil {
		t.Fatalf("reading coordinator/tls.crt: %v", err)
	}
	coordKeyPEM, err := os.ReadFile(filepath.Join(outDir, "coordinator/tls.key"))
	if err != nil {
		t.Fatalf("reading coordinator/tls.key: %v", err)
	}
	if _, err := pki.VerifyLeaf(ca.Cert, coordCertPEM, coordKeyPEM,
		pki.CoordinatorSPIFFEID(pki.DefaultTrustDomain, pki.DefaultWorkloadNamespace),
		x509.ExtKeyUsageClientAuth, time.Now().UTC()); err != nil {
		t.Fatalf("verifying coordinator leaf: %v", err)
	}

	if err := run(t.Context(), []string{"--out-dir", outDir}); err != nil {
		t.Fatalf("second run --out-dir: %v", err)
	}
	coordCertPEM2, err := os.ReadFile(filepath.Join(outDir, "coordinator/tls.crt"))
	if err != nil {
		t.Fatalf("reading coordinator/tls.crt after second run: %v", err)
	}
	if !bytes.Equal(coordCertPEM, coordCertPEM2) {
		t.Errorf("second run changed coordinator/tls.crt")
	}
}

func TestRunRejectsInvalidFlagCombinations(t *testing.T) {
	t.Setenv("KUBECONFIG", filepath.Join(t.TempDir(), "does-not-exist"))

	for _, args := range [][]string{
		{"--out-dir", ""},
		{"--out-dir="},
	} {
		err := run(t.Context(), args)
		if err == nil {
			t.Errorf("run(%v): expected error for empty --out-dir", args)
		} else if !strings.Contains(err.Error(), "output directory must not be empty") {
			t.Errorf("run(%v): got %q, want empty output directory error (must not fall through to Kubernetes)", args, err)
		}
	}

	if err := run(t.Context(), []string{"--out-dir", "", "--kubeconfig", "/tmp/kc"}); err == nil || !strings.Contains(err.Error(), "cannot be used together") {
		t.Errorf("expected conflicting-flags error for empty --out-dir with --kubeconfig, got %v", err)
	}
	if err := run(t.Context(), []string{"--out-dir", t.TempDir(), "--kubeconfig", "/tmp/kc"}); err == nil {
		t.Errorf("expected error when both --out-dir and --kubeconfig are set")
	}
	if err := run(t.Context(), []string{"unexpected-arg"}); err == nil {
		t.Errorf("expected error for unexpected positional argument")
	}
}
