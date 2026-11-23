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
	"net/url"
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
