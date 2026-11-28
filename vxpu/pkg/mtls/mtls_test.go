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

package mtls_test

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"math/big"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/mtls"
	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func (b *lockedBuffer) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf.Reset()
}

func newLogContext(t *testing.T) (context.Context, *lockedBuffer) {
	t.Helper()
	var buf lockedBuffer
	ctx := klog.NewContext(t.Context(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf))))
	return ctx, &buf
}

func waitForLogContains(t *testing.T, buf *lockedBuffer, substr string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(buf.String(), substr) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Errorf("expected log to contain %q, got: %s", substr, buf.String())
}

func writeDirBundle(t *testing.T, dir string, caCertPEM, certPEM, keyPEM []byte) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ca.crt"), caCertPEM, 0o644); err != nil {
		t.Fatalf("writing ca.crt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tls.crt"), certPEM, 0o644); err != nil {
		t.Fatalf("writing tls.crt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tls.key"), keyPEM, 0o600); err != nil {
		t.Fatalf("writing tls.key: %v", err)
	}
}

func startAgentGRPC(t *testing.T, creds credentials.TransportCredentials) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	var opts []grpc.ServerOption
	if creds != nil {
		opts = append(opts, grpc.Creds(creds))
	}
	gs := grpc.NewServer(opts...)
	srv := agent.NewServer(t.TempDir(), t.TempDir(), nil)
	pb.RegisterAgentServer(gs, srv)
	go func() {
		_ = gs.Serve(lis)
	}()
	t.Cleanup(func() {
		gs.Stop()
		_ = lis.Close()
	})
	return lis.Addr().String()
}

func dialAndCallDescribe(t *testing.T, addr string, creds credentials.TransportCredentials) error {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(creds))
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()

	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	_, err = pb.NewAgentClient(conn).DescribeMachine(ctx, &pb.DescribeMachineRequest{})
	return err
}

func mintExpiredLeaf(t *testing.T, ca *pki.KeyPair, spiffeID string, eku x509.ExtKeyUsage) *pki.KeyPair {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generating key: %v", err)
	}
	uri, err := url.Parse(spiffeID)
	if err != nil {
		t.Fatalf("parsing URI: %v", err)
	}
	now := time.Now().UTC()
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(42),
		NotBefore:             now.Add(-2 * time.Hour),
		NotAfter:              now.Add(-1 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{eku},
		BasicConstraintsValid: true,
		URIs:                  []*url.URL{uri},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.Cert, &key.PublicKey, ca.Key)
	if err != nil {
		t.Fatalf("creating expired cert: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("parsing expired cert: %v", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("marshaling key: %v", err)
	}
	return &pki.KeyPair{
		Cert:    cert,
		Key:     key,
		CertPEM: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		KeyPEM:  pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}),
	}
}

func rawClientTLS(t *testing.T, caCertPEM []byte, cert *pki.KeyPair) credentials.TransportCredentials {
	t.Helper()
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caCertPEM) {
		t.Fatalf("parsing CA PEM")
	}
	cfg := &tls.Config{
		MinVersion:         tls.VersionTLS13,
		RootCAs:            pool,
		InsecureSkipVerify: true,
	}
	if cert != nil {
		kp, err := tls.X509KeyPair(cert.CertPEM, cert.KeyPEM)
		if err != nil {
			t.Fatalf("loading client key pair: %v", err)
		}
		cfg.Certificates = []tls.Certificate{kp}
	}
	return credentials.NewTLS(cfg)
}

func TestAgentServerMTLS(t *testing.T) {
	pkiDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), pkiDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir: %v", err)
	}

	caCertPEM, err := os.ReadFile(filepath.Join(pkiDir, "ca/ca.crt"))
	if err != nil {
		t.Fatalf("reading ca.crt: %v", err)
	}
	caKeyPEM, err := os.ReadFile(filepath.Join(pkiDir, "ca/ca.key"))
	if err != nil {
		t.Fatalf("reading ca.key: %v", err)
	}
	ca, err := pki.ParseKeyPair(caCertPEM, caKeyPEM)
	if err != nil {
		t.Fatalf("parsing CA: %v", err)
	}

	serverCtx, serverLogs := newLogContext(t)
	serverCreds, err := mtls.LoadAgentServerCredentials(serverCtx, filepath.Join(pkiDir, "agent"))
	if err != nil {
		t.Fatalf("LoadAgentServerCredentials: %v", err)
	}
	addr := startAgentGRPC(t, serverCreds)

	// 1. Valid coordinator credentials succeed on multiple RPCs.
	clientCtx, clientLogs := newLogContext(t)
	coordCreds, err := mtls.LoadCoordinatorClientCredentials(clientCtx, filepath.Join(pkiDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials: %v", err)
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(coordCreds))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer func() { _ = conn.Close() }()
	client := pb.NewAgentClient(conn)

	callCtx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if _, err := client.DescribeMachine(callCtx, &pb.DescribeMachineRequest{}); err != nil {
		t.Fatalf("valid coordinator DescribeMachine failed: %v", err)
	}
	if _, err := client.ListBlobs(callCtx, &pb.ListBlobsRequest{}); err != nil {
		t.Fatalf("valid coordinator ListBlobs failed: %v", err)
	}

	// 2. Plaintext caller is rejected and logged.
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, insecure.NewCredentials()); err == nil {
		t.Fatalf("expected plaintext client to be rejected")
	}
	waitForLogContains(t, serverLogs, "Refused agent mTLS connection")

	// 3. TLS caller with no client certificate is rejected and logged.
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, rawClientTLS(t, caCertPEM, nil)); err == nil {
		t.Fatalf("expected no-client-cert caller to be rejected")
	}
	waitForLogContains(t, serverLogs, `uri="none"`)

	// 4. Caller signed by a different CA is rejected and logged.
	otherCA, err := pki.MintCA(time.Hour)
	if err != nil {
		t.Fatalf("MintCA: %v", err)
	}
	otherCoord, err := pki.MintLeaf(otherCA, pki.CoordinatorSPIFFEID(pki.DefaultTrustDomain, pki.DefaultWorkloadNamespace), x509.ExtKeyUsageClientAuth, time.Hour)
	if err != nil {
		t.Fatalf("MintLeaf(otherCA): %v", err)
	}
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, rawClientTLS(t, caCertPEM, otherCoord)); err == nil {
		t.Fatalf("expected wrong-CA client to be rejected")
	}
	waitForLogContains(t, serverLogs, "Refused agent mTLS connection")

	// 5. Caller with an expired coordinator certificate is rejected.
	expiredCoord := mintExpiredLeaf(t, ca, pki.CoordinatorSPIFFEID(pki.DefaultTrustDomain, pki.DefaultWorkloadNamespace), x509.ExtKeyUsageClientAuth)
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, rawClientTLS(t, caCertPEM, expiredCoord)); err == nil {
		t.Fatalf("expected expired client cert to be rejected")
	}
	waitForLogContains(t, serverLogs, "Refused agent mTLS connection")

	// 6. Caller using an agent ServerAuth cert as a client cert is rejected.
	agentLeaf, err := pki.MintLeaf(ca, pki.AgentSPIFFEID(pki.DefaultTrustDomain, pki.DefaultWorkloadNamespace), x509.ExtKeyUsageServerAuth, time.Hour)
	if err != nil {
		t.Fatalf("MintLeaf(agent): %v", err)
	}
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, rawClientTLS(t, caCertPEM, agentLeaf)); err == nil {
		t.Fatalf("expected agent ServerAuth cert to be rejected as client")
	}
	waitForLogContains(t, serverLogs, "Refused agent mTLS connection")

	// 7. Caller with a CA-signed ClientAuth cert carrying the wrong SPIFFE URI is rejected and logs its presented URI.
	wrongURIs := []string{
		"spiffe://cluster.local/ns/default/sa/vxpu-agent",
		"spiffe://cluster.local/ns/other/sa/vxpu-router",
		"spiffe://other.domain/ns/default/sa/vxpu-router",
	}
	for _, badURI := range wrongURIs {
		badLeaf, err := pki.MintLeaf(ca, badURI, x509.ExtKeyUsageClientAuth, time.Hour)
		if err != nil {
			t.Fatalf("MintLeaf(%s): %v", badURI, err)
		}
		serverLogs.Reset()
		if err := dialAndCallDescribe(t, addr, rawClientTLS(t, caCertPEM, badLeaf)); err == nil {
			t.Fatalf("expected client with URI %q to be rejected", badURI)
		}
		waitForLogContains(t, serverLogs, badURI)
	}

	// 8. TLS 1.2 client is rejected because TLS 1.3 is required.
	coordCert, err := tls.LoadX509KeyPair(filepath.Join(pkiDir, "coordinator/tls.crt"), filepath.Join(pkiDir, "coordinator/tls.key"))
	if err != nil {
		t.Fatalf("LoadX509KeyPair: %v", err)
	}
	tls12Creds := credentials.NewTLS(&tls.Config{
		MinVersion:         tls.VersionTLS12,
		MaxVersion:         tls.VersionTLS12,
		Certificates:       []tls.Certificate{coordCert},
		InsecureSkipVerify: true,
	})
	serverLogs.Reset()
	if err := dialAndCallDescribe(t, addr, tls12Creds); err == nil {
		t.Fatalf("expected TLS 1.2 client to be rejected")
	}
	waitForLogContains(t, serverLogs, "Refused agent mTLS connection")

	// Confirm no private key material ever appears in server or client logs.
	if strings.Contains(serverLogs.String(), "PRIVATE KEY") || strings.Contains(clientLogs.String(), "PRIVATE KEY") {
		t.Errorf("private key material must never be logged")
	}
}

func TestCoordinatorVerifiesAgentIdentity(t *testing.T) {
	ca, err := pki.MintCA(time.Hour)
	if err != nil {
		t.Fatalf("MintCA: %v", err)
	}
	coordLeaf, err := pki.MintLeaf(ca, "spiffe://example.org/ns/prod/sa/vxpu-router", x509.ExtKeyUsageClientAuth, time.Hour)
	if err != nil {
		t.Fatalf("MintLeaf(coordinator): %v", err)
	}
	coordDir := filepath.Join(t.TempDir(), "coordinator")
	writeDirBundle(t, coordDir, ca.CertPEM, coordLeaf.CertPEM, coordLeaf.KeyPEM)

	coordCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), coordDir)
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials: %v", err)
	}

	// 1. VM agent identity in the same trust domain and namespace succeeds.
	vmLeaf, err := pki.MintLeaf(ca, "spiffe://example.org/ns/prod/vm/gpu-vm-01", x509.ExtKeyUsageServerAuth, time.Hour)
	if err != nil {
		t.Fatalf("MintLeaf(vm): %v", err)
	}
	vmDir := filepath.Join(t.TempDir(), "vm-agent")
	writeDirBundle(t, vmDir, ca.CertPEM, vmLeaf.CertPEM, vmLeaf.KeyPEM)
	vmCreds, err := mtls.LoadAgentServerCredentials(t.Context(), vmDir)
	if err != nil {
		t.Fatalf("LoadAgentServerCredentials(vm): %v", err)
	}
	vmAddr := startAgentGRPC(t, vmCreds)
	if err := dialAndCallDescribe(t, vmAddr, coordCreds); err != nil {
		t.Fatalf("coordinator dialing VM agent failed: %v", err)
	}

	// 2. Server presenting a wrong-CA certificate is rejected by the coordinator.
	otherCA, err := pki.MintCA(time.Hour)
	if err != nil {
		t.Fatalf("MintCA(other): %v", err)
	}
	otherAgent, err := pki.MintLeaf(otherCA, "spiffe://example.org/ns/prod/sa/vxpu-agent", x509.ExtKeyUsageServerAuth, time.Hour)
	if err != nil {
		t.Fatalf("MintLeaf(otherAgent): %v", err)
	}
	otherDir := filepath.Join(t.TempDir(), "other-agent")
	writeDirBundle(t, otherDir, otherCA.CertPEM, otherAgent.CertPEM, otherAgent.KeyPEM)
	otherCreds, err := mtls.LoadAgentServerCredentials(t.Context(), otherDir)
	if err != nil {
		t.Fatalf("LoadAgentServerCredentials(other): %v", err)
	}
	otherAddr := startAgentGRPC(t, otherCreds)
	if err := dialAndCallDescribe(t, otherAddr, coordCreds); err == nil {
		t.Fatalf("expected coordinator to reject agent signed by wrong CA")
	}

	// 3. Server presenting a CA-signed ServerAuth cert with a non-agent SPIFFE ID is rejected.
	for _, badURI := range []string{
		"spiffe://example.org/ns/prod/sa/vxpu-router",
		"spiffe://example.org/ns/other/sa/vxpu-agent",
		"spiffe://other.org/ns/prod/sa/vxpu-agent",
		"spiffe://example.org/ns/prod/sa/other",
	} {
		badServerLeaf, err := pki.MintLeaf(ca, badURI, x509.ExtKeyUsageServerAuth, time.Hour)
		if err != nil {
			t.Fatalf("MintLeaf(%s): %v", badURI, err)
		}
		kp, err := tls.X509KeyPair(badServerLeaf.CertPEM, badServerLeaf.KeyPEM)
		if err != nil {
			t.Fatalf("X509KeyPair: %v", err)
		}
		pool := x509.NewCertPool()
		pool.AppendCertsFromPEM(ca.CertPEM)
		badSrvCreds := credentials.NewTLS(&tls.Config{
			MinVersion:   tls.VersionTLS13,
			Certificates: []tls.Certificate{kp},
			ClientAuth:   tls.RequireAndVerifyClientCert,
			ClientCAs:    pool,
		})
		badAddr := startAgentGRPC(t, badSrvCreds)
		if err := dialAndCallDescribe(t, badAddr, coordCreds); err == nil {
			t.Fatalf("expected coordinator to reject server with URI %q", badURI)
		}
	}
}

func TestLoadCredentialValidationErrors(t *testing.T) {
	pkiDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), pkiDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir: %v", err)
	}

	if _, err := mtls.LoadAgentServerCredentials(t.Context(), ""); err == nil {
		t.Errorf("expected error for empty agent tlsDir")
	}
	if _, err := mtls.LoadCoordinatorClientCredentials(t.Context(), ""); err == nil {
		t.Errorf("expected error for empty coordinator tlsDir")
	}
	if _, err := mtls.LoadAgentServerCredentials(t.Context(), filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Errorf("expected error for missing agent tlsDir")
	}
	// Swapped directories must fail startup validation (wrong EKU and SPIFFE ID).
	if _, err := mtls.LoadAgentServerCredentials(t.Context(), filepath.Join(pkiDir, "coordinator")); err == nil {
		t.Errorf("expected error when loading coordinator certs as agent server credentials")
	}
	if _, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(pkiDir, "agent")); err == nil {
		t.Errorf("expected error when loading agent certs as coordinator client credentials")
	}

	// Expired local certificate must fail startup validation.
	ca, err := pki.MintCA(time.Hour)
	if err != nil {
		t.Fatalf("MintCA: %v", err)
	}
	expiredAgent := mintExpiredLeaf(t, ca, pki.AgentSPIFFEID(pki.DefaultTrustDomain, pki.DefaultWorkloadNamespace), x509.ExtKeyUsageServerAuth)
	expDir := filepath.Join(t.TempDir(), "expired-agent")
	writeDirBundle(t, expDir, ca.CertPEM, expiredAgent.CertPEM, expiredAgent.KeyPEM)
	if _, err := mtls.LoadAgentServerCredentials(t.Context(), expDir); err == nil {
		t.Errorf("expected error when loading expired agent certificate")
	}
}
