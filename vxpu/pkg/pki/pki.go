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

// Package pki mints the vXPU mutual TLS certificate authority and SPIFFE leaf
// certificates and reconciles them into Kubernetes objects.
package pki

import (
	"cmp"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/url"
	"time"

	"k8s.io/klog/v2"
)

const (
	// DefaultCANamespace holds the CA private key apart from workload namespaces.
	DefaultCANamespace = "vxpu-pki"
	// DefaultWorkloadNamespace holds workload bundles and leaf Secrets.
	DefaultWorkloadNamespace = "default"
	// DefaultTrustDomain is the SPIFFE trust domain written into leaf SANs.
	DefaultTrustDomain = "cluster.local"
	// DefaultCATTL is the default lifetime of the CA certificate.
	DefaultCATTL = 365 * 24 * time.Hour
	// DefaultLeafTTL is the default lifetime of coordinator and agent leaf certificates.
	DefaultLeafTTL = 30 * 24 * time.Hour

	// CASecretName is the Secret storing ca.crt and ca.key in the CA namespace.
	CASecretName = "vxpu-ca"
	// CABundleConfigMapName is the ConfigMap storing ca.crt in the workload namespace.
	CABundleConfigMapName = "vxpu-ca-bundle"
	// CoordinatorSecretName is the TLS Secret for the coordinator client certificate.
	CoordinatorSecretName = "vxpu-coordinator-tls"
	// AgentSecretName is the TLS Secret for the pod agent server certificate.
	AgentSecretName = "vxpu-agent-tls"
)

// Options configures certificate lifetimes, SPIFFE identities, and target namespaces.
type Options struct {
	// CANamespace is the Kubernetes namespace that holds the vxpu-ca Secret.
	CANamespace string
	// WorkloadNamespace is the namespace that holds the CA bundle and workload Secrets.
	WorkloadNamespace string
	// TrustDomain is the SPIFFE trust domain used in leaf URI SANs.
	TrustDomain string
	// CATTL is the validity duration of a newly minted CA certificate.
	CATTL time.Duration
	// LeafTTL is the validity duration of newly minted leaf certificates.
	LeafTTL time.Duration
}

func (o Options) withDefaults() Options {
	o.CANamespace = cmp.Or(o.CANamespace, DefaultCANamespace)
	o.WorkloadNamespace = cmp.Or(o.WorkloadNamespace, DefaultWorkloadNamespace)
	o.TrustDomain = cmp.Or(o.TrustDomain, DefaultTrustDomain)
	o.CATTL = cmp.Or(o.CATTL, DefaultCATTL)
	o.LeafTTL = cmp.Or(o.LeafTTL, DefaultLeafTTL)
	return o
}

// KeyPair holds a parsed X.509 certificate, its ECDSA private key, and both PEM encodings.
type KeyPair struct {
	// Cert is the parsed X.509 certificate.
	Cert *x509.Certificate
	// Key is the ECDSA P-256 private key matching Cert.
	Key *ecdsa.PrivateKey
	// CertPEM is the PEM-encoded certificate.
	CertPEM []byte
	// KeyPEM is the PKCS#8 PEM-encoded private key.
	KeyPEM []byte
}

// CoordinatorSPIFFEID returns the SPIFFE URI for the coordinator in the given namespace.
func CoordinatorSPIFFEID(trustDomain, namespace string) string {
	return fmt.Sprintf("spiffe://%s/ns/%s/sa/vxpu-router", trustDomain, namespace)
}

// AgentSPIFFEID returns the SPIFFE URI for pod agents in the given namespace.
func AgentSPIFFEID(trustDomain, namespace string) string {
	return fmt.Sprintf("spiffe://%s/ns/%s/sa/vxpu-agent", trustDomain, namespace)
}

// MintCA generates a self-signed ECDSA P-256 certificate authority.
func MintCA(ttl time.Duration) (*KeyPair, error) {
	if ttl <= 0 {
		return nil, fmt.Errorf("CA TTL must be positive, got %v", ttl)
	}
	now := time.Now().UTC()
	return signCert(&x509.Certificate{
		Subject:               pkix.Name{CommonName: CASecretName},
		NotBefore:             now.Add(-5 * time.Minute),
		NotAfter:              now.Add(ttl),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
		MaxPathLenZero:        true,
	}, nil, nil)
}

// MintLeaf signs an ECDSA P-256 leaf certificate carrying a single SPIFFE URI SAN
// and one extended key usage, with no DNS or IP SANs.
func MintLeaf(ca *KeyPair, spiffeID string, eku x509.ExtKeyUsage, ttl time.Duration) (*KeyPair, error) {
	if ttl <= 0 {
		return nil, fmt.Errorf("leaf TTL must be positive, got %v", ttl)
	}
	uri, err := url.Parse(spiffeID)
	if err != nil || uri.Scheme != "spiffe" || uri.Host == "" || uri.Path == "" {
		return nil, fmt.Errorf("invalid SPIFFE ID %q", spiffeID)
	}
	now := time.Now().UTC()
	return signCert(&x509.Certificate{
		NotBefore:             now.Add(-5 * time.Minute),
		NotAfter:              now.Add(ttl),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{eku},
		BasicConstraintsValid: true,
		URIs:                  []*url.URL{uri},
	}, ca.Cert, ca.Key)
}

// ParseKeyPair decodes a PEM-encoded certificate and PKCS#8 ECDSA private key and checks that they match.
func ParseKeyPair(certPEM, keyPEM []byte) (*KeyPair, error) {
	block, _ := pem.Decode(certPEM)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, fmt.Errorf("invalid certificate PEM")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing certificate: %w", err)
	}
	keyBlock, _ := pem.Decode(keyPEM)
	if keyBlock == nil || keyBlock.Type != "PRIVATE KEY" {
		return nil, fmt.Errorf("invalid PKCS#8 private key PEM")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(keyBlock.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing PKCS#8 private key: %w", err)
	}
	key, ok := parsed.(*ecdsa.PrivateKey)
	if !ok || !key.PublicKey.Equal(cert.PublicKey) {
		return nil, fmt.Errorf("private key does not match certificate ECDSA public key")
	}
	return &KeyPair{Cert: cert, Key: key, CertPEM: certPEM, KeyPEM: keyPEM}, nil
}

// VerifyLeaf checks that certPEM and keyPEM form a matching leaf key pair that
// chains to ca, is valid at now, carries only spiffeID with no DNS or IP SANs,
// and carries only eku.
func VerifyLeaf(ca *x509.Certificate, certPEM, keyPEM []byte, spiffeID string, eku x509.ExtKeyUsage, now time.Time) (*x509.Certificate, error) {
	kp, err := ParseKeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, err
	}
	c := kp.Cert
	if c.IsCA || len(c.DNSNames) != 0 || len(c.IPAddresses) != 0 || len(c.URIs) != 1 || c.URIs[0].String() != spiffeID {
		return nil, fmt.Errorf("certificate constraints or SANs do not match %s", spiffeID)
	}
	if c.KeyUsage != x509.KeyUsageDigitalSignature || len(c.ExtKeyUsage) != 1 || c.ExtKeyUsage[0] != eku {
		return nil, fmt.Errorf("key usage does not match expected usage")
	}
	if now.After(c.NotAfter) {
		return nil, fmt.Errorf("certificate expired at %s", c.NotAfter.UTC().Format(time.RFC3339))
	}
	roots := x509.NewCertPool()
	roots.AddCert(ca)
	if _, err := c.Verify(x509.VerifyOptions{Roots: roots, CurrentTime: now, KeyUsages: []x509.ExtKeyUsage{eku}}); err != nil {
		return nil, fmt.Errorf("does not verify against current CA: %w", err)
	}
	return c, nil
}

func signCert(tmpl, parent *x509.Certificate, signer *ecdsa.PrivateKey) (*KeyPair, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generating ECDSA key: %w", err)
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("generating serial number: %w", err)
	}
	if serial.Sign() == 0 {
		serial.SetInt64(1)
	}
	tmpl.SerialNumber = serial
	if parent == nil {
		parent, signer = tmpl, key
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, parent, &key.PublicKey, signer)
	if err != nil {
		return nil, fmt.Errorf("creating certificate: %w", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, fmt.Errorf("parsing minted certificate: %w", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return nil, fmt.Errorf("encoding private key: %w", err)
	}
	return &KeyPair{
		Cert:    cert,
		Key:     key,
		CertPEM: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		KeyPEM:  pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}),
	}, nil
}

func logCert(ctx context.Context, name string, cert *x509.Certificate, status, reason string) {
	identity := cert.Subject.CommonName
	if len(cert.URIs) > 0 {
		identity = cert.URIs[0].String()
	}
	args := []any{"name", name, "identity", identity, "expires", cert.NotAfter.UTC().Format(time.RFC3339), "status", status}
	if reason != "" {
		args = append(args, "reason", reason)
	}
	klog.FromContext(ctx).Info("Certificate ready", args...)
}
