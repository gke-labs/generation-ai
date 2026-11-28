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

// Package mtls loads vXPU mutual TLS credentials from a certificate directory
// and verifies SPIFFE URI Subject Alternative Names over TLS 1.3.
package mtls

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc/credentials"
	"k8s.io/klog/v2"
)

const (
	caFileName   = "ca.crt"
	certFileName = "tls.crt"
	keyFileName  = "tls.key"
)

type dirBundle struct {
	// caPool holds the trusted CA certificate pool loaded from ca.crt.
	caPool *x509.CertPool
	// cert holds the loaded TLS certificate, private key, and parsed leaf.
	cert tls.Certificate
	// trustDomain is the SPIFFE trust domain parsed from the local leaf.
	trustDomain string
	// namespace is the SPIFFE namespace parsed from the local leaf.
	namespace string
	// identity is the local leaf's SPIFFE URI string.
	identity string
}

type peerIdentityError struct {
	// uri is the URI SAN presented by the peer, or "none" if absent.
	uri string
	// err is the underlying verification failure.
	err error
}

func (e *peerIdentityError) Error() string {
	return e.err.Error()
}

func (e *peerIdentityError) Unwrap() error {
	return e.err
}

type serverCreds struct {
	// inner is the standard gRPC TLS credentials performing the TLS 1.3 handshake.
	inner credentials.TransportCredentials
	// log records structured refusal events when handshakes fail.
	log klog.Logger
}

// LoadAgentServerCredentials loads ca.crt, tls.crt, and tls.key from tlsDir
// and returns gRPC server credentials that require TLS 1.3 mutual TLS from the
// coordinator in the same trust domain and namespace.
func LoadAgentServerCredentials(ctx context.Context, tlsDir string) (credentials.TransportCredentials, error) {
	bundle, err := loadDirBundle(tlsDir, x509.ExtKeyUsageServerAuth, isAgentIdentity)
	if err != nil {
		return nil, err
	}
	expectedCoordinator := fmt.Sprintf("spiffe://%s/ns/%s/sa/vxpu-router", bundle.trustDomain, bundle.namespace)
	tlsCfg := &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{bundle.cert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    bundle.caPool,
		VerifyConnection: func(cs tls.ConnectionState) error {
			return verifyCoordinatorPeer(cs.PeerCertificates, expectedCoordinator)
		},
	}
	log := klog.FromContext(ctx)
	log.Info("Loaded agent mTLS credentials",
		"identity", bundle.identity,
		"expectedCoordinator", expectedCoordinator,
		"expires", bundle.cert.Leaf.NotAfter.UTC().Format(time.RFC3339),
	)
	return &serverCreds{
		inner: credentials.NewTLS(tlsCfg),
		log:   log,
	}, nil
}

// LoadCoordinatorClientCredentials loads ca.crt, tls.crt, and tls.key from
// tlsDir and returns gRPC client credentials that present the coordinator
// certificate over TLS 1.3 and verify that the peer is a pod or VM agent in
// the same trust domain and namespace.
func LoadCoordinatorClientCredentials(ctx context.Context, tlsDir string) (credentials.TransportCredentials, error) {
	bundle, err := loadDirBundle(tlsDir, x509.ExtKeyUsageClientAuth, isCoordinatorIdentity)
	if err != nil {
		return nil, err
	}
	tlsCfg := &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{bundle.cert},
		RootCAs:      bundle.caPool,
		// SPIFFE certificates carry only a URI SAN and no DNS or IP SANs, so
		// standard hostname matching cannot verify dial targets. VerifyConnection
		// validates the X.509 chain, ServerAuth EKU, validity window, and SPIFFE ID.
		InsecureSkipVerify: true,
		VerifyConnection: func(cs tls.ConnectionState) error {
			return verifyAgentPeer(cs.PeerCertificates, bundle.caPool, bundle.trustDomain, bundle.namespace)
		},
	}
	klog.FromContext(ctx).Info("Loaded coordinator mTLS credentials",
		"identity", bundle.identity,
		"expires", bundle.cert.Leaf.NotAfter.UTC().Format(time.RFC3339),
	)
	return credentials.NewTLS(tlsCfg), nil
}

func loadDirBundle(tlsDir string, expectedEKU x509.ExtKeyUsage, checkIdentity func(kind, name string) bool) (*dirBundle, error) {
	if tlsDir == "" {
		return nil, errors.New("TLS credential directory must not be empty")
	}
	caPath := filepath.Join(tlsDir, caFileName)
	caPEM, err := os.ReadFile(caPath)
	if err != nil {
		return nil, fmt.Errorf("reading CA bundle %s: %w", caPath, err)
	}
	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caPEM) {
		return nil, fmt.Errorf("parsing CA bundle %s: no valid PEM certificates found", caPath)
	}

	certPath := filepath.Join(tlsDir, certFileName)
	keyPath := filepath.Join(tlsDir, keyFileName)
	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("loading TLS key pair from %s: %w", tlsDir, err)
	}
	if len(cert.Certificate) == 0 {
		return nil, fmt.Errorf("certificate file %s is empty", certPath)
	}
	leaf, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return nil, fmt.Errorf("parsing leaf certificate %s: %w", certPath, err)
	}
	cert.Leaf = leaf

	intermediates, err := parseIntermediates(cert.Certificate[1:])
	if err != nil {
		return nil, fmt.Errorf("parsing intermediate certificates in %s: %w", certPath, err)
	}
	if _, err := leaf.Verify(x509.VerifyOptions{
		Roots:         caPool,
		Intermediates: intermediates,
		KeyUsages:     []x509.ExtKeyUsage{expectedEKU},
	}); err != nil {
		return nil, fmt.Errorf("verifying leaf certificate %s against CA bundle: %w", certPath, err)
	}
	if leaf.IsCA {
		return nil, fmt.Errorf("leaf certificate %s must not be a CA", certPath)
	}
	if len(leaf.URIs) != 1 {
		return nil, fmt.Errorf("leaf certificate %s must carry exactly 1 SPIFFE URI SAN, got %d", certPath, len(leaf.URIs))
	}
	trustDomain, namespace, kind, name, ok := parseSPIFFEID(leaf.URIs[0])
	if !ok || !checkIdentity(kind, name) {
		return nil, fmt.Errorf("leaf certificate %s has unexpected SPIFFE ID %q", certPath, leaf.URIs[0].String())
	}

	return &dirBundle{
		caPool:      caPool,
		cert:        cert,
		trustDomain: trustDomain,
		namespace:   namespace,
		identity:    leaf.URIs[0].String(),
	}, nil
}

func parseIntermediates(raw [][]byte) (*x509.CertPool, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	pool := x509.NewCertPool()
	for _, der := range raw {
		c, err := x509.ParseCertificate(der)
		if err != nil {
			return nil, err
		}
		pool.AddCert(c)
	}
	return pool, nil
}

func verifyCoordinatorPeer(certs []*x509.Certificate, expectedSPIFFEID string) error {
	if len(certs) == 0 {
		return &peerIdentityError{uri: "none", err: errors.New("peer presented no certificate")}
	}
	leaf := certs[0]
	if leaf.IsCA {
		return &peerIdentityError{uri: presentedURI(leaf), err: errors.New("peer certificate must not be a CA")}
	}
	if len(leaf.URIs) != 1 {
		return &peerIdentityError{
			uri: presentedURI(leaf),
			err: fmt.Errorf("peer certificate must carry exactly 1 URI SAN, got %d", len(leaf.URIs)),
		}
	}
	got := leaf.URIs[0].String()
	if got != expectedSPIFFEID {
		return &peerIdentityError{
			uri: got,
			err: fmt.Errorf("unexpected coordinator SPIFFE ID %q, want %q", got, expectedSPIFFEID),
		}
	}
	return nil
}

func verifyAgentPeer(certs []*x509.Certificate, roots *x509.CertPool, trustDomain, namespace string) error {
	if len(certs) == 0 {
		return errors.New("agent presented no certificate")
	}
	leaf := certs[0]
	var intermediates *x509.CertPool
	if len(certs) > 1 {
		intermediates = x509.NewCertPool()
		for _, c := range certs[1:] {
			intermediates.AddCert(c)
		}
	}
	if _, err := leaf.Verify(x509.VerifyOptions{
		Roots:         roots,
		Intermediates: intermediates,
		KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}); err != nil {
		return fmt.Errorf("verifying agent certificate: %w", err)
	}
	if leaf.IsCA {
		return errors.New("agent certificate must not be a CA")
	}
	if len(leaf.URIs) != 1 {
		return fmt.Errorf("agent certificate must carry exactly 1 URI SAN, got %d", len(leaf.URIs))
	}
	td, ns, kind, name, ok := parseSPIFFEID(leaf.URIs[0])
	if !ok || td != trustDomain || ns != namespace || !isAgentIdentity(kind, name) {
		return fmt.Errorf("unexpected agent SPIFFE ID %q for trust domain %q and namespace %q",
			leaf.URIs[0].String(), trustDomain, namespace)
	}
	return nil
}

func parseSPIFFEID(u *url.URL) (trustDomain, namespace, kind, name string, ok bool) {
	if u == nil || u.Scheme != "spiffe" || u.Host == "" || strings.Contains(u.Host, ":") {
		return "", "", "", "", false
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", "", "", "", false
	}
	if u.String() != "spiffe://"+u.Host+u.Path {
		return "", "", "", "", false
	}
	if !strings.HasPrefix(u.Path, "/") {
		return "", "", "", "", false
	}
	parts := strings.Split(strings.TrimPrefix(u.Path, "/"), "/")
	if len(parts) != 4 || parts[0] != "ns" || !validSPIFFESegment(parts[1]) {
		return "", "", "", "", false
	}
	if (parts[2] != "sa" && parts[2] != "vm") || !validSPIFFESegment(parts[3]) {
		return "", "", "", "", false
	}
	return u.Host, parts[1], parts[2], parts[3], true
}

func validSPIFFESegment(s string) bool {
	return s != "" && s != "." && s != ".."
}

func isCoordinatorIdentity(kind, name string) bool {
	return kind == "sa" && name == "vxpu-router"
}

func isAgentIdentity(kind, name string) bool {
	return (kind == "sa" && name == "vxpu-agent") || kind == "vm"
}

func presentedURI(cert *x509.Certificate) string {
	if cert == nil || len(cert.URIs) == 0 {
		return "none"
	}
	if len(cert.URIs) == 1 {
		return cert.URIs[0].String()
	}
	uris := make([]string, len(cert.URIs))
	for i, u := range cert.URIs {
		uris[i] = u.String()
	}
	return strings.Join(uris, ",")
}

func (c *serverCreds) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return c.inner.ClientHandshake(ctx, authority, rawConn)
}

func (c *serverCreds) ServerHandshake(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	conn, authInfo, err := c.inner.ServerHandshake(rawConn)
	if err != nil {
		peerAddr := "unknown"
		if rawConn.RemoteAddr() != nil {
			peerAddr = rawConn.RemoteAddr().String()
		}
		uri := "none"
		var pe *peerIdentityError
		if errors.As(err, &pe) {
			uri = pe.uri
		}
		c.log.Info("Refused agent mTLS connection", "peer", peerAddr, "uri", uri, "reason", err.Error())
		return nil, nil, err
	}
	return conn, authInfo, nil
}

func (c *serverCreds) Info() credentials.ProtocolInfo {
	return c.inner.Info()
}

func (c *serverCreds) Clone() credentials.TransportCredentials {
	return &serverCreds{
		inner: c.inner.Clone(),
		log:   c.log,
	}
}

func (c *serverCreds) OverrideServerName(serverNameOverride string) error {
	return c.inner.OverrideServerName(serverNameOverride)
}
