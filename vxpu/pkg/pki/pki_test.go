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
	"crypto/x509"
	"crypto/x509/pkix"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

func TestEnsureK8sLifecycleAndConsistency(t *testing.T) {
	client := fake.NewClientset()
	var logBuf bytes.Buffer
	ctx := klog.NewContext(t.Context(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&logBuf))))

	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("first EnsureK8s: %v", err)
	}

	caSec := mustGetSecret(t, client, DefaultCANamespace, CASecretName)
	bundleCM := mustGetConfigMap(t, client, DefaultWorkloadNamespace, CABundleConfigMapName)
	coordSec := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	agentSec := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)

	if len(bundleCM.Data) != 1 || bundleCM.Data["ca.crt"] != string(caSec.Data["ca.crt"]) {
		t.Fatalf("CA bundle must contain only ca.crt matching vxpu-ca, got keys: %v", bundleCM.Data)
	}
	if strings.Contains(bundleCM.Data["ca.crt"], "PRIVATE KEY") {
		t.Fatalf("CA bundle must not contain a private key")
	}

	caCert := verifyCA(t, caSec.Data["ca.crt"], caSec.Data["ca.key"])
	coordCert := verifyLeafStrict(t, caCert, coordSec.Data[corev1.TLSCertKey], coordSec.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert := verifyLeafStrict(t, caCert, agentSec.Data[corev1.TLSCertKey], agentSec.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	checkLogOutput(t, logBuf.String(), caCert, coordCert, agentCert)

	// Re-running with all valid objects must leave every certificate unchanged and log status="kept".
	logBuf.Reset()
	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("second EnsureK8s: %v", err)
	}
	caSec2 := mustGetSecret(t, client, DefaultCANamespace, CASecretName)
	coordSec2 := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	agentSec2 := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)
	if !bytes.Equal(caSec.Data["ca.crt"], caSec2.Data["ca.crt"]) || !bytes.Equal(caSec.Data["ca.key"], caSec2.Data["ca.key"]) {
		t.Errorf("second run modified CA secret")
	}
	if !bytes.Equal(coordSec.Data[corev1.TLSCertKey], coordSec2.Data[corev1.TLSCertKey]) {
		t.Errorf("second run modified coordinator secret")
	}
	if !bytes.Equal(agentSec.Data[corev1.TLSCertKey], agentSec2.Data[corev1.TLSCertKey]) {
		t.Errorf("second run modified agent secret")
	}
	if !strings.Contains(logBuf.String(), `status="kept"`) {
		t.Errorf("second run logs missing status=kept: %s", logBuf.String())
	}
	checkLogOutput(t, logBuf.String(), caCert, coordCert, agentCert)

	// Deleting one leaf and re-running keeps CA and coordinator intact and creates a fresh agent leaf.
	if err := client.CoreV1().Secrets(DefaultWorkloadNamespace).Delete(ctx, AgentSecretName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("deleting agent secret: %v", err)
	}
	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("EnsureK8s after deleting agent secret: %v", err)
	}
	coordSec3 := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	agentSec3 := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)
	if !bytes.Equal(coordSec.Data[corev1.TLSCertKey], coordSec3.Data[corev1.TLSCertKey]) {
		t.Errorf("re-creating agent modified coordinator secret")
	}
	if bytes.Equal(agentSec.Data[corev1.TLSCertKey], agentSec3.Data[corev1.TLSCertKey]) {
		t.Errorf("re-created agent secret has identical certificate to deleted one")
	}
	verifyLeafStrict(t, caCert, agentSec3.Data[corev1.TLSCertKey], agentSec3.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)

	// Re-creating vxpu-ca while the old bundle and leaves remain must update the bundle and re-mint both leaves.
	if err := client.CoreV1().Secrets(DefaultCANamespace).Delete(ctx, CASecretName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("deleting CA secret: %v", err)
	}
	logBuf.Reset()
	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("EnsureK8s after deleting CA secret: %v", err)
	}
	caSec4 := mustGetSecret(t, client, DefaultCANamespace, CASecretName)
	bundleCM4 := mustGetConfigMap(t, client, DefaultWorkloadNamespace, CABundleConfigMapName)
	coordSec4 := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	agentSec4 := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)
	if bytes.Equal(caSec.Data["ca.crt"], caSec4.Data["ca.crt"]) {
		t.Fatalf("expected a new CA certificate after deleting vxpu-ca")
	}
	if bundleCM4.Data["ca.crt"] != string(caSec4.Data["ca.crt"]) {
		t.Errorf("CA bundle was not updated to the new CA")
	}
	caCert4 := verifyCA(t, caSec4.Data["ca.crt"], caSec4.Data["ca.key"])
	coordCert4 := verifyLeafStrict(t, caCert4, coordSec4.Data[corev1.TLSCertKey], coordSec4.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert4 := verifyLeafStrict(t, caCert4, agentSec4.Data[corev1.TLSCertKey], agentSec4.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	if !strings.Contains(logBuf.String(), `status="re-minted"`) || !strings.Contains(logBuf.String(), `reason=`) {
		t.Errorf("expected re-minted status and reason in logs after CA change: %s", logBuf.String())
	}
	if !strings.Contains(logBuf.String(), `status="updated"`) {
		t.Errorf("expected bundle updated status in logs after CA change: %s", logBuf.String())
	}
	checkLogOutput(t, logBuf.String(), caCert4, coordCert4, agentCert4)
}

func TestEnsureK8sRemintsExpiredAndMismatchedLeaves(t *testing.T) {
	client := fake.NewClientset()
	var logBuf bytes.Buffer
	ctx := klog.NewContext(t.Context(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&logBuf))))

	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("initial EnsureK8s: %v", err)
	}
	caSec := mustGetSecret(t, client, DefaultCANamespace, CASecretName)
	ca, err := ParseKeyPair(caSec.Data["ca.crt"], caSec.Data["ca.key"])
	if err != nil {
		t.Fatalf("parsing CA: %v", err)
	}

	// Replace coordinator secret with an expired certificate signed by the current CA.
	uri, _ := url.Parse(CoordinatorSPIFFEID(DefaultTrustDomain, DefaultWorkloadNamespace))
	expiredLeaf, err := signCert(&x509.Certificate{
		NotBefore:             time.Now().UTC().Add(-48 * time.Hour),
		NotAfter:              time.Now().UTC().Add(-1 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		URIs:                  []*url.URL{uri},
	}, ca.Cert, ca.Key)
	if err != nil {
		t.Fatalf("minting expired leaf: %v", err)
	}
	coordSec := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	coordSec.Data[corev1.TLSCertKey] = expiredLeaf.CertPEM
	coordSec.Data[corev1.TLSPrivateKeyKey] = expiredLeaf.KeyPEM
	if _, err := client.CoreV1().Secrets(DefaultWorkloadNamespace).Update(ctx, coordSec, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("updating coordinator secret with expired cert: %v", err)
	}

	// Corrupt agent secret's private key with a mismatched key.
	otherLeaf, err := MintLeaf(ca, AgentSPIFFEID(DefaultTrustDomain, DefaultWorkloadNamespace), x509.ExtKeyUsageServerAuth, DefaultLeafTTL)
	if err != nil {
		t.Fatalf("minting other leaf: %v", err)
	}
	agentSec := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)
	agentSec.Data[corev1.TLSPrivateKeyKey] = otherLeaf.KeyPEM
	if _, err := client.CoreV1().Secrets(DefaultWorkloadNamespace).Update(ctx, agentSec, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("corrupting agent secret key: %v", err)
	}

	logBuf.Reset()
	if err := EnsureK8s(ctx, client, Options{}); err != nil {
		t.Fatalf("EnsureK8s repairing leaves: %v", err)
	}

	repairedCoord := mustGetSecret(t, client, DefaultWorkloadNamespace, CoordinatorSecretName)
	repairedAgent := mustGetSecret(t, client, DefaultWorkloadNamespace, AgentSecretName)
	coordCert := verifyLeafStrict(t, ca.Cert, repairedCoord.Data[corev1.TLSCertKey], repairedCoord.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert := verifyLeafStrict(t, ca.Cert, repairedAgent.Data[corev1.TLSCertKey], repairedAgent.Data[corev1.TLSPrivateKeyKey],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)

	logs := logBuf.String()
	if !strings.Contains(logs, "expired") {
		t.Errorf("expected expired reason in logs: %s", logs)
	}
	if !strings.Contains(logs, "does not match") {
		t.Errorf("expected key mismatch reason in logs: %s", logs)
	}
	checkLogOutput(t, logs, ca.Cert, coordCert, agentCert)
}

func TestEnsureK8sRejectsMalformedCAAndSharedNamespace(t *testing.T) {
	client := fake.NewClientset()
	if err := EnsureK8s(t.Context(), client, Options{CANamespace: "default", WorkloadNamespace: "default"}); err == nil {
		t.Fatalf("expected error when CA namespace equals workload namespace")
	}
	_, err := client.CoreV1().Secrets(DefaultCANamespace).Create(t.Context(), &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: CASecretName, Namespace: DefaultCANamespace},
		Data:       map[string][]byte{"ca.crt": []byte("not-a-cert"), "ca.key": []byte("not-a-key")},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("creating malformed CA secret: %v", err)
	}
	if err := EnsureK8s(t.Context(), client, Options{}); err == nil {
		t.Fatalf("expected error for malformed CA secret")
	}
}

func mustGetSecret(t *testing.T, client *fake.Clientset, namespace, name string) *corev1.Secret {
	t.Helper()
	sec, err := client.CoreV1().Secrets(namespace).Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting secret %s/%s: %v", namespace, name, err)
	}
	return sec
}

func mustGetConfigMap(t *testing.T, client *fake.Clientset, namespace, name string) *corev1.ConfigMap {
	t.Helper()
	cm, err := client.CoreV1().ConfigMaps(namespace).Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting configmap %s/%s: %v", namespace, name, err)
	}
	return cm
}

func verifyCA(t *testing.T, certPEM, keyPEM []byte) *x509.Certificate {
	t.Helper()
	kp, err := ParseKeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("parsing CA key pair: %v", err)
	}
	if !kp.Cert.IsCA || !kp.Cert.BasicConstraintsValid || !kp.Cert.MaxPathLenZero || kp.Cert.MaxPathLen != 0 {
		t.Errorf("CA cert missing expected basic constraints")
	}
	if kp.Cert.KeyUsage != (x509.KeyUsageCertSign | x509.KeyUsageCRLSign) {
		t.Errorf("CA KeyUsage = %v, want CertSign|CRLSign", kp.Cert.KeyUsage)
	}
	if len(kp.Cert.DNSNames) != 0 || len(kp.Cert.IPAddresses) != 0 || len(kp.Cert.URIs) != 0 {
		t.Errorf("CA cert should have no SANs, got dns=%v ip=%v uri=%v", kp.Cert.DNSNames, kp.Cert.IPAddresses, kp.Cert.URIs)
	}
	return kp.Cert
}

func verifyLeafStrict(t *testing.T, ca *x509.Certificate, certPEM, keyPEM []byte, wantSPIFFE string, wantEKU x509.ExtKeyUsage) *x509.Certificate {
	t.Helper()
	cert, err := VerifyLeaf(ca, certPEM, keyPEM, wantSPIFFE, wantEKU, time.Now().UTC())
	if err != nil {
		t.Fatalf("VerifyLeaf failed: %v", err)
	}
	wrongEKU := x509.ExtKeyUsageServerAuth
	if wantEKU == x509.ExtKeyUsageServerAuth {
		wrongEKU = x509.ExtKeyUsageClientAuth
	}
	if _, err := VerifyLeaf(ca, certPEM, keyPEM, wantSPIFFE, wrongEKU, time.Now().UTC()); err == nil {
		t.Errorf("leaf unexpectedly verified for wrong EKU %v", wrongEKU)
	}
	return cert
}

func checkLogOutput(t *testing.T, logs string, certs ...*x509.Certificate) {
	t.Helper()
	if strings.Contains(logs, "PRIVATE KEY") {
		t.Errorf("logs contain private key material: %s", logs)
	}
	for _, c := range certs {
		id := c.Subject.CommonName
		if len(c.URIs) > 0 {
			id = c.URIs[0].String()
		}
		if !strings.Contains(logs, id) {
			t.Errorf("logs missing identity %q: %s", id, logs)
		}
		exp := c.NotAfter.UTC().Format(time.RFC3339)
		if !strings.Contains(logs, exp) {
			t.Errorf("logs missing expiry %q for %s: %s", exp, id, logs)
		}
	}
}

func TestEnsureDirLifecycleAndConsistency(t *testing.T) {
	dir := t.TempDir()
	var logBuf bytes.Buffer
	ctx := klog.NewContext(t.Context(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&logBuf))))

	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("first EnsureDir: %v", err)
	}

	files1 := snapshotDirFiles(t, dir)
	checkDirPermsAndLayout(t, dir)

	if !bytes.Equal(files1["coordinator/ca.crt"], files1["ca/ca.crt"]) {
		t.Errorf("coordinator/ca.crt does not match ca/ca.crt")
	}
	if !bytes.Equal(files1["agent/ca.crt"], files1["ca/ca.crt"]) {
		t.Errorf("agent/ca.crt does not match ca/ca.crt")
	}
	if bytes.Contains(files1["coordinator/ca.crt"], []byte("PRIVATE KEY")) || bytes.Contains(files1["agent/ca.crt"], []byte("PRIVATE KEY")) {
		t.Fatalf("workload ca.crt must not contain private key material")
	}

	caCert := verifyCA(t, files1["ca/ca.crt"], files1["ca/ca.key"])
	coordCert := verifyLeafStrict(t, caCert, files1["coordinator/tls.crt"], files1["coordinator/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert := verifyLeafStrict(t, caCert, files1["agent/tls.crt"], files1["agent/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	checkLogOutput(t, logBuf.String(), caCert, coordCert, agentCert)

	// Second run must leave every file byte-for-byte unchanged and log only status="kept".
	logBuf.Reset()
	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("second EnsureDir: %v", err)
	}
	files2 := snapshotDirFiles(t, dir)
	for rel, want := range files1 {
		if !bytes.Equal(want, files2[rel]) {
			t.Errorf("second run modified %s", rel)
		}
	}
	logs2 := logBuf.String()
	if !strings.Contains(logs2, `status="kept"`) || strings.Contains(logs2, `status="created"`) || strings.Contains(logs2, `status="re-minted"`) {
		t.Errorf("unexpected second run logs: %s", logs2)
	}
	checkLogOutput(t, logs2, caCert, coordCert, agentCert)

	// Deleting one leaf key keeps CA and coordinator unchanged and re-mints the agent leaf.
	if err := os.Remove(filepath.Join(dir, "agent/tls.key")); err != nil {
		t.Fatalf("removing agent/tls.key: %v", err)
	}
	logBuf.Reset()
	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("EnsureDir after deleting agent/tls.key: %v", err)
	}
	files3 := snapshotDirFiles(t, dir)
	checkDirPermsAndLayout(t, dir)
	if !bytes.Equal(files1["ca/ca.crt"], files3["ca/ca.crt"]) || !bytes.Equal(files1["coordinator/tls.crt"], files3["coordinator/tls.crt"]) {
		t.Errorf("re-minting agent modified CA or coordinator")
	}
	if bytes.Equal(files1["agent/tls.crt"], files3["agent/tls.crt"]) {
		t.Errorf("agent/tls.crt was not re-minted after deleting agent/tls.key")
	}
	agentCert3 := verifyLeafStrict(t, caCert, files3["agent/tls.crt"], files3["agent/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	if !strings.Contains(logBuf.String(), `status="re-minted"`) || !strings.Contains(logBuf.String(), "missing tls.key") {
		t.Errorf("expected re-minted log with missing tls.key reason: %s", logBuf.String())
	}
	checkLogOutput(t, logBuf.String(), caCert, coordCert, agentCert3)

	// Replacing the CA with a different valid CA must update both ca.crt bundles and re-mint both leaves.
	newCA, err := MintCA(DefaultCATTL)
	if err != nil {
		t.Fatalf("minting replacement CA: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ca/ca.crt"), newCA.CertPEM, 0o644); err != nil {
		t.Fatalf("writing replacement ca.crt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ca/ca.key"), newCA.KeyPEM, 0o600); err != nil {
		t.Fatalf("writing replacement ca.key: %v", err)
	}
	logBuf.Reset()
	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("EnsureDir after replacing CA: %v", err)
	}
	files4 := snapshotDirFiles(t, dir)
	checkDirPermsAndLayout(t, dir)
	if !bytes.Equal(files4["ca/ca.crt"], newCA.CertPEM) {
		t.Fatalf("EnsureDir overwrote valid replacement CA")
	}
	if !bytes.Equal(files4["coordinator/ca.crt"], newCA.CertPEM) || !bytes.Equal(files4["agent/ca.crt"], newCA.CertPEM) {
		t.Errorf("workload ca.crt bundles were not updated to replacement CA")
	}
	coordCert4 := verifyLeafStrict(t, newCA.Cert, files4["coordinator/tls.crt"], files4["coordinator/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert4 := verifyLeafStrict(t, newCA.Cert, files4["agent/tls.crt"], files4["agent/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	logs4 := logBuf.String()
	if !strings.Contains(logs4, `status="re-minted"`) || !strings.Contains(logs4, `status="updated"`) {
		t.Errorf("expected re-minted and updated statuses after CA replacement: %s", logs4)
	}
	checkLogOutput(t, logs4, newCA.Cert, coordCert4, agentCert4)

	// Deleting the CA directory mints a new CA, updates both bundles, and re-mints both leaves.
	if err := os.RemoveAll(filepath.Join(dir, "ca")); err != nil {
		t.Fatalf("removing ca directory: %v", err)
	}
	logBuf.Reset()
	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("EnsureDir after deleting ca directory: %v", err)
	}
	files5 := snapshotDirFiles(t, dir)
	checkDirPermsAndLayout(t, dir)
	if bytes.Equal(files5["ca/ca.crt"], newCA.CertPEM) {
		t.Fatalf("expected a freshly minted CA after deleting ca directory")
	}
	caCert5 := verifyCA(t, files5["ca/ca.crt"], files5["ca/ca.key"])
	if !bytes.Equal(files5["coordinator/ca.crt"], files5["ca/ca.crt"]) || !bytes.Equal(files5["agent/ca.crt"], files5["ca/ca.crt"]) {
		t.Errorf("workload ca.crt bundles were not updated to newly minted CA")
	}
	coordCert5 := verifyLeafStrict(t, caCert5, files5["coordinator/tls.crt"], files5["coordinator/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert5 := verifyLeafStrict(t, caCert5, files5["agent/tls.crt"], files5["agent/tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	checkLogOutput(t, logBuf.String(), caCert5, coordCert5, agentCert5)
}

func TestEnsureDirRemintsExpiredAndMismatchedLeaves(t *testing.T) {
	dir := t.TempDir()
	var logBuf bytes.Buffer
	ctx := klog.NewContext(t.Context(), textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&logBuf))))

	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("initial EnsureDir: %v", err)
	}
	ca, err := ParseKeyPair(mustReadFile(t, filepath.Join(dir, "ca/ca.crt")), mustReadFile(t, filepath.Join(dir, "ca/ca.key")))
	if err != nil {
		t.Fatalf("parsing CA: %v", err)
	}

	// Replace coordinator files with an expired certificate signed by the current CA.
	uri, _ := url.Parse(CoordinatorSPIFFEID(DefaultTrustDomain, DefaultWorkloadNamespace))
	expiredLeaf, err := signCert(&x509.Certificate{
		NotBefore:             time.Now().UTC().Add(-48 * time.Hour),
		NotAfter:              time.Now().UTC().Add(-1 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		URIs:                  []*url.URL{uri},
	}, ca.Cert, ca.Key)
	if err != nil {
		t.Fatalf("minting expired leaf: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "coordinator/tls.crt"), expiredLeaf.CertPEM, 0o644); err != nil {
		t.Fatalf("writing expired coordinator cert: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "coordinator/tls.key"), expiredLeaf.KeyPEM, 0o600); err != nil {
		t.Fatalf("writing expired coordinator key: %v", err)
	}

	// Replace agent private key with a mismatched key.
	otherLeaf, err := MintLeaf(ca, AgentSPIFFEID(DefaultTrustDomain, DefaultWorkloadNamespace), x509.ExtKeyUsageServerAuth, DefaultLeafTTL)
	if err != nil {
		t.Fatalf("minting other leaf: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "agent/tls.key"), otherLeaf.KeyPEM, 0o600); err != nil {
		t.Fatalf("writing mismatched agent key: %v", err)
	}

	logBuf.Reset()
	if err := EnsureDir(ctx, dir, Options{}); err != nil {
		t.Fatalf("EnsureDir repairing leaves: %v", err)
	}
	checkDirPermsAndLayout(t, dir)

	coordCert := verifyLeafStrict(t, ca.Cert, mustReadFile(t, filepath.Join(dir, "coordinator/tls.crt")),
		mustReadFile(t, filepath.Join(dir, "coordinator/tls.key")),
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert := verifyLeafStrict(t, ca.Cert, mustReadFile(t, filepath.Join(dir, "agent/tls.crt")),
		mustReadFile(t, filepath.Join(dir, "agent/tls.key")),
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)

	logs := logBuf.String()
	if !strings.Contains(logs, "expired") {
		t.Errorf("expected expired reason in logs: %s", logs)
	}
	if !strings.Contains(logs, "does not match") {
		t.Errorf("expected key mismatch reason in logs: %s", logs)
	}
	checkLogOutput(t, logs, ca.Cert, coordCert, agentCert)
}

func TestEnsureDirRefusesMalformedCAWithoutOverwriting(t *testing.T) {
	if err := EnsureDir(t.Context(), "", Options{}); err == nil {
		t.Fatalf("expected error for empty output directory")
	}

	otherCA, err := MintCA(DefaultCATTL)
	if err != nil {
		t.Fatalf("minting other CA: %v", err)
	}
	expiredCA, err := signCert(&x509.Certificate{
		Subject:               pkix.Name{CommonName: CASecretName},
		NotBefore:             time.Now().UTC().Add(-48 * time.Hour),
		NotAfter:              time.Now().UTC().Add(-1 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
		MaxPathLenZero:        true,
	}, nil, nil)
	if err != nil {
		t.Fatalf("minting expired CA: %v", err)
	}
	nonCA, err := MintLeaf(otherCA, CoordinatorSPIFFEID(DefaultTrustDomain, DefaultWorkloadNamespace), x509.ExtKeyUsageClientAuth, DefaultLeafTTL)
	if err != nil {
		t.Fatalf("minting non-CA leaf: %v", err)
	}

	cases := []struct {
		name  string
		setup func(t *testing.T, dir string)
	}{
		{
			name: "corrupt PEM",
			setup: func(t *testing.T, dir string) {
				writeTestFile(t, filepath.Join(dir, "ca/ca.crt"), []byte("not-a-cert"), 0o644)
				writeTestFile(t, filepath.Join(dir, "ca/ca.key"), []byte("not-a-key"), 0o600)
			},
		},
		{
			name: "mismatched key",
			setup: func(t *testing.T, dir string) {
				writeTestFile(t, filepath.Join(dir, "ca/ca.key"), otherCA.KeyPEM, 0o600)
			},
		},
		{
			name: "expired CA",
			setup: func(t *testing.T, dir string) {
				writeTestFile(t, filepath.Join(dir, "ca/ca.crt"), expiredCA.CertPEM, 0o644)
				writeTestFile(t, filepath.Join(dir, "ca/ca.key"), expiredCA.KeyPEM, 0o600)
			},
		},
		{
			name: "non-CA certificate",
			setup: func(t *testing.T, dir string) {
				writeTestFile(t, filepath.Join(dir, "ca/ca.crt"), nonCA.CertPEM, 0o644)
				writeTestFile(t, filepath.Join(dir, "ca/ca.key"), nonCA.KeyPEM, 0o600)
			},
		},
		{
			name: "missing ca.key",
			setup: func(t *testing.T, dir string) {
				if err := os.Remove(filepath.Join(dir, "ca/ca.key")); err != nil {
					t.Fatalf("removing ca.key: %v", err)
				}
			},
		},
		{
			name: "missing ca.crt",
			setup: func(t *testing.T, dir string) {
				if err := os.Remove(filepath.Join(dir, "ca/ca.crt")); err != nil {
					t.Fatalf("removing ca.crt: %v", err)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := EnsureDir(t.Context(), dir, Options{}); err != nil {
				t.Fatalf("initial EnsureDir: %v", err)
			}
			tc.setup(t, dir)
			before := snapshotExistingFiles(t, dir)

			if err := EnsureDir(t.Context(), dir, Options{}); err == nil {
				t.Fatalf("expected EnsureDir to refuse %s", tc.name)
			}

			after := snapshotExistingFiles(t, dir)
			if len(before) != len(after) {
				t.Fatalf("file count changed on refused CA: before=%d after=%d", len(before), len(after))
			}
			for rel, want := range before {
				if !bytes.Equal(want, after[rel]) {
					t.Errorf("EnsureDir overwrote %s on refused CA", rel)
				}
			}
		})
	}
}

func TestEnsureDirRecoversFromInterruptedFirstRun(t *testing.T) {
	t.Run("failed CA placement leaves no partial CA and cleans staging dir", func(t *testing.T) {
		dir := t.TempDir()
		caDir := filepath.Join(dir, "ca")
		if err := os.MkdirAll(caDir, 0o700); err != nil {
			t.Fatalf("creating caDir: %v", err)
		}
		blocker := filepath.Join(caDir, "blocker")
		writeTestFile(t, blocker, []byte("x"), 0o600)

		if err := EnsureDir(t.Context(), dir, Options{}); err == nil {
			t.Fatalf("expected EnsureDir to fail when ca directory cannot be replaced")
		}
		for _, rel := range []string{"ca/ca.crt", "ca/ca.key"} {
			if _, err := os.Stat(filepath.Join(dir, rel)); !os.IsNotExist(err) {
				t.Errorf("failed first run left %s behind", rel)
			}
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("reading %s: %v", dir, err)
		}
		for _, e := range entries {
			if strings.HasPrefix(e.Name(), ".ca.tmp-") {
				t.Errorf("failed first run left staging directory %s behind", e.Name())
			}
		}

		// Clearing the blocker leaves an empty ca directory; a clean follow-up run must succeed.
		if err := os.Remove(blocker); err != nil {
			t.Fatalf("removing blocker: %v", err)
		}
		if err := EnsureDir(t.Context(), dir, Options{}); err != nil {
			t.Fatalf("clean EnsureDir after failed first run: %v", err)
		}
		checkDirPermsAndLayout(t, dir)
		files := snapshotDirFiles(t, dir)
		caCert := verifyCA(t, files["ca/ca.crt"], files["ca/ca.key"])
		verifyLeafStrict(t, caCert, files["coordinator/tls.crt"], files["coordinator/tls.key"],
			"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
		verifyLeafStrict(t, caCert, files["agent/tls.crt"], files["agent/tls.key"],
			"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	})

	t.Run("killed during staging leaves stale key dir that next run removes", func(t *testing.T) {
		dir := t.TempDir()
		abandonedStage := filepath.Join(dir, ".ca.tmp-interrupted")
		if err := os.MkdirAll(abandonedStage, 0o700); err != nil {
			t.Fatalf("creating simulated staging dir: %v", err)
		}
		writeTestFile(t, filepath.Join(abandonedStage, "ca.crt"), []byte("partial-cert"), 0o644)
		writeTestFile(t, filepath.Join(abandonedStage, "ca.key"), []byte("orphaned-private-key"), 0o600)

		if err := EnsureDir(t.Context(), dir, Options{}); err != nil {
			t.Fatalf("EnsureDir after interrupted staging: %v", err)
		}
		if _, err := os.Stat(abandonedStage); !os.IsNotExist(err) {
			t.Fatalf("stale staging directory %s was not removed", abandonedStage)
		}
		checkDirPermsAndLayout(t, dir)
		files := snapshotDirFiles(t, dir)
		caCert := verifyCA(t, files["ca/ca.crt"], files["ca/ca.key"])
		verifyLeafStrict(t, caCert, files["coordinator/tls.crt"], files["coordinator/tls.key"],
			"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
		verifyLeafStrict(t, caCert, files["agent/tls.crt"], files["agent/tls.key"],
			"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)

		// A stale staging directory appearing alongside an existing valid CA is also cleaned without changing the CA.
		staleAgain := filepath.Join(dir, ".ca.tmp-leftover")
		if err := os.MkdirAll(staleAgain, 0o700); err != nil {
			t.Fatalf("creating stale staging dir: %v", err)
		}
		writeTestFile(t, filepath.Join(staleAgain, "ca.key"), []byte("orphaned-private-key"), 0o600)

		if err := EnsureDir(t.Context(), dir, Options{}); err != nil {
			t.Fatalf("EnsureDir cleaning stale staging dir alongside valid CA: %v", err)
		}
		if _, err := os.Stat(staleAgain); !os.IsNotExist(err) {
			t.Fatalf("stale staging directory %s was not removed alongside valid CA", staleAgain)
		}
		filesAfter := snapshotDirFiles(t, dir)
		for rel, want := range files {
			if !bytes.Equal(want, filesAfter[rel]) {
				t.Errorf("cleaning stale staging dir modified %s", rel)
			}
		}
	})
}

func checkDirPermsAndLayout(t *testing.T, dir string) {
	t.Helper()
	wantPerms := map[string]os.FileMode{
		"ca/ca.crt":           0o644,
		"ca/ca.key":           0o600,
		"coordinator/ca.crt":  0o644,
		"coordinator/tls.crt": 0o644,
		"coordinator/tls.key": 0o600,
		"agent/ca.crt":        0o644,
		"agent/tls.crt":       0o644,
		"agent/tls.key":       0o600,
	}
	for rel, want := range wantPerms {
		info, err := os.Stat(filepath.Join(dir, rel))
		if err != nil {
			t.Fatalf("stat %s: %v", rel, err)
		}
		if got := info.Mode().Perm(); got != want {
			t.Errorf("%s perm = %#o, want %#o", rel, got, want)
		}
	}
	for _, subdir := range []string{"coordinator", "agent"} {
		if _, err := os.Stat(filepath.Join(dir, subdir, "ca.key")); !os.IsNotExist(err) {
			t.Errorf("%s must not contain ca.key", subdir)
		}
	}
	for _, subdir := range []string{"ca", "coordinator", "agent"} {
		entries, err := os.ReadDir(filepath.Join(dir, subdir))
		if err != nil {
			t.Fatalf("reading %s: %v", subdir, err)
		}
		for _, e := range entries {
			if strings.Contains(e.Name(), ".tmp-") {
				t.Errorf("leftover temp file in %s: %s", subdir, e.Name())
			}
		}
	}
}

func snapshotDirFiles(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	rels := []string{
		"ca/ca.crt", "ca/ca.key",
		"coordinator/ca.crt", "coordinator/tls.crt", "coordinator/tls.key",
		"agent/ca.crt", "agent/tls.crt", "agent/tls.key",
	}
	out := make(map[string][]byte, len(rels))
	for _, rel := range rels {
		out[rel] = mustReadFile(t, filepath.Join(dir, rel))
	}
	return out
}

func snapshotExistingFiles(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		out[rel] = mustReadFile(t, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walking %s: %v", dir, err)
	}
	return out
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return data
}

func writeTestFile(t *testing.T, path string, data []byte, perm os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, data, perm); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}
