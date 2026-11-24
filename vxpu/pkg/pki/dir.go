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

package pki

import (
	"bytes"
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/fileutil"
)

const caStagingPrefix = ".ca.tmp-"

// EnsureDir writes the CA, coordinator, and agent certificates into a local
// directory tree for running vXPU on one machine without a cluster. It places
// ca.crt and ca.key under <dir>/ca, and places ca.crt, tls.crt, and tls.key
// under <dir>/coordinator and <dir>/agent. It keeps valid files unchanged,
// updates stale CA bundles to the current CA, mints any missing, expired, or
// non-verifying leaf from the CA, and refuses a malformed or incomplete CA
// without overwriting existing files.
func EnsureDir(ctx context.Context, dir string, opts Options) error {
	if dir == "" {
		return fmt.Errorf("output directory must not be empty")
	}
	opts = opts.withDefaults()
	ca, err := ensureCADir(ctx, filepath.Join(dir, "ca"), opts.CATTL)
	if err != nil {
		return err
	}
	for _, leaf := range []struct {
		subdir   string
		name     string
		spiffeID string
		eku      x509.ExtKeyUsage
	}{
		{"coordinator", CoordinatorSecretName, CoordinatorSPIFFEID(opts.TrustDomain, opts.WorkloadNamespace), x509.ExtKeyUsageClientAuth},
		{"agent", AgentSecretName, AgentSPIFFEID(opts.TrustDomain, opts.WorkloadNamespace), x509.ExtKeyUsageServerAuth},
	} {
		leafDir := filepath.Join(dir, leaf.subdir)
		if err := ensureCABundleFile(ctx, leafDir, leaf.subdir, ca.CertPEM); err != nil {
			return err
		}
		if err := ensureLeafDir(ctx, leafDir, leaf.name, ca, leaf.spiffeID, leaf.eku, opts.LeafTTL); err != nil {
			return err
		}
	}
	return nil
}

func ensureCADir(ctx context.Context, caDir string, ttl time.Duration) (*KeyPair, error) {
	certPath := filepath.Join(caDir, "ca.crt")
	keyPath := filepath.Join(caDir, "ca.key")
	certPEM, certErr := os.ReadFile(certPath)
	keyPEM, keyErr := os.ReadFile(keyPath)
	if certErr != nil && !errors.Is(certErr, os.ErrNotExist) {
		return nil, fmt.Errorf("reading %s: %w", certPath, certErr)
	}
	if keyErr != nil && !errors.Is(keyErr, os.ErrNotExist) {
		return nil, fmt.Errorf("reading %s: %w", keyPath, keyErr)
	}
	if certErr == nil && keyErr == nil {
		ca, err := ParseKeyPair(certPEM, keyPEM)
		if err != nil {
			return nil, fmt.Errorf("parsing CA in %s: %w", caDir, err)
		}
		if !ca.Cert.IsCA || time.Now().UTC().After(ca.Cert.NotAfter) {
			return nil, fmt.Errorf("CA in %s is not a valid unexpired CA", caDir)
		}
		if err := cleanStaleCAStagingDirs(ctx, filepath.Dir(caDir)); err != nil {
			return nil, err
		}
		logCert(ctx, CASecretName, ca.Cert, "kept", "")
		return ca, nil
	}
	if certErr == nil || keyErr == nil {
		return nil, fmt.Errorf("incomplete CA in %s: both ca.crt and ca.key must exist or both must be absent", caDir)
	}

	if err := cleanStaleCAStagingDirs(ctx, filepath.Dir(caDir)); err != nil {
		return nil, err
	}
	ca, err := MintCA(ttl)
	if err != nil {
		return nil, err
	}
	if err := stageCADir(ctx, caDir, ca); err != nil {
		return nil, err
	}
	logCert(ctx, CASecretName, ca.Cert, "created", "")
	return ca, nil
}

func cleanStaleCAStagingDirs(ctx context.Context, parent string) error {
	entries, err := os.ReadDir(parent)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("reading %s: %w", parent, err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), caStagingPrefix) {
			continue
		}
		stale := filepath.Join(parent, entry.Name())
		if err := os.RemoveAll(stale); err != nil {
			return fmt.Errorf("removing stale CA staging directory %s: %w", stale, err)
		}
		klog.FromContext(ctx).Info("Removed stale CA staging directory", "name", entry.Name())
	}
	return nil
}

func stageCADir(ctx context.Context, caDir string, ca *KeyPair) error {
	parent := filepath.Dir(caDir)
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return fmt.Errorf("creating directory %s: %w", parent, err)
	}
	stageDir, err := os.MkdirTemp(parent, caStagingPrefix+"*")
	if err != nil {
		return fmt.Errorf("creating staging directory for %s: %w", caDir, err)
	}
	renamed := false
	defer func() {
		if !renamed {
			if err := os.RemoveAll(stageDir); err != nil {
				klog.FromContext(ctx).V(2).Info("removing CA staging directory during cleanup", "path", stageDir, "error", err)
			}
		}
	}()
	if err := writeKeyPairFiles(stageDir, "ca.crt", "ca.key", ca); err != nil {
		return err
	}
	if err := os.Rename(stageDir, caDir); err != nil {
		info, statErr := os.Lstat(caDir)
		if statErr != nil || !info.IsDir() {
			return fmt.Errorf("placing CA directory %s: %w", caDir, err)
		}
		if removeErr := os.Remove(caDir); removeErr != nil {
			return fmt.Errorf("placing CA directory %s: %w (removing existing directory: %w)", caDir, err, removeErr)
		}
		if err := os.Rename(stageDir, caDir); err != nil {
			return fmt.Errorf("placing CA directory %s: %w", caDir, err)
		}
	}
	renamed = true
	return nil
}

func ensureCABundleFile(ctx context.Context, leafDir, subdir string, caCertPEM []byte) error {
	bundlePath := filepath.Join(leafDir, "ca.crt")
	bundleName := filepath.Join(subdir, "ca.crt")
	existing, err := os.ReadFile(bundlePath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("reading %s: %w", bundlePath, err)
	}
	if err == nil && bytes.Equal(existing, caCertPEM) {
		klog.FromContext(ctx).Info("CA bundle ready", "name", bundleName, "status", "kept")
		return nil
	}
	created := errors.Is(err, os.ErrNotExist)
	if err := os.MkdirAll(leafDir, 0o700); err != nil {
		return fmt.Errorf("creating directory %s: %w", leafDir, err)
	}
	if err := fileutil.WriteAtomic(bundlePath, caCertPEM, 0o644); err != nil {
		return err
	}
	if created {
		klog.FromContext(ctx).Info("CA bundle ready", "name", bundleName, "status", "created")
		return nil
	}
	klog.FromContext(ctx).Info("CA bundle ready", "name", bundleName, "status", "updated", "reason", "CA certificate changed")
	return nil
}

func ensureLeafDir(ctx context.Context, leafDir, name string, ca *KeyPair,
	spiffeID string, eku x509.ExtKeyUsage, ttl time.Duration) error {
	certPath := filepath.Join(leafDir, "tls.crt")
	keyPath := filepath.Join(leafDir, "tls.key")
	certPEM, certErr := os.ReadFile(certPath)
	keyPEM, keyErr := os.ReadFile(keyPath)
	if certErr != nil && !errors.Is(certErr, os.ErrNotExist) {
		return fmt.Errorf("reading %s: %w", certPath, certErr)
	}
	if keyErr != nil && !errors.Is(keyErr, os.ErrNotExist) {
		return fmt.Errorf("reading %s: %w", keyPath, keyErr)
	}

	status := "created"
	reason := ""
	switch {
	case certErr == nil && keyErr == nil:
		cert, verifyErr := VerifyLeaf(ca.Cert, certPEM, keyPEM, spiffeID, eku, time.Now().UTC())
		if verifyErr == nil {
			logCert(ctx, name, cert, "kept", "")
			return nil
		}
		status = "re-minted"
		reason = verifyErr.Error()
	case certErr == nil:
		status = "re-minted"
		reason = "missing tls.key"
	case keyErr == nil:
		status = "re-minted"
		reason = "missing tls.crt"
	}

	leaf, err := MintLeaf(ca, spiffeID, eku, ttl)
	if err != nil {
		return err
	}
	if err := writeKeyPairFiles(leafDir, "tls.crt", "tls.key", leaf); err != nil {
		return err
	}
	logCert(ctx, name, leaf.Cert, status, reason)
	return nil
}

func writeKeyPairFiles(dir, certFile, keyFile string, kp *KeyPair) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("creating directory %s: %w", dir, err)
	}
	if err := fileutil.WriteAtomic(filepath.Join(dir, certFile), kp.CertPEM, 0o644); err != nil {
		return err
	}
	return fileutil.WriteAtomic(filepath.Join(dir, keyFile), kp.KeyPEM, 0o600)
}
