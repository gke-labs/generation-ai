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

package e2e

import (
	"bytes"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

// testPKIInit exercises the vxpu-pki-init Job in kind: minting the CA and
// coordinator/agent certificates, verifying chains, SPIFFE identities, EKUs,
// CA key isolation from router and executor service accounts, non-secret logs,
// idempotency with deleted-leaf recreation, and consistency after CA replacement.
func testPKIInit(t *testing.T, h *Harness, vxpuRoot string) {
	manifestPath := filepath.Join(vxpuRoot, "k8s/pki-init.yaml")
	rawManifest, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("reading %s: %v", manifestPath, err)
	}
	manifest := strings.ReplaceAll(string(rawManifest), "image: vxpu-pki-init:latest", "image: vxpu-pki-init:e2e")
	manifest = strings.ReplaceAll(manifest, "imagePullPolicy: IfNotPresent", "imagePullPolicy: Never")

	// Run 1: apply the manifest and wait for the Job to finish.
	h.KubectlApplyContent("vxpu-pki-init-run1", manifest)
	waitForJobComplete(t, h, "vxpu-pki-init", pki.DefaultCANamespace)

	logs1 := jobLogs(t, "vxpu-pki-init", pki.DefaultCANamespace)
	t.Logf("vxpu-pki-init run 1 logs: %s", logs1)

	caData1 := getSecretData(t, pki.DefaultCANamespace, pki.CASecretName)
	bundleData1 := getConfigMapData(t, pki.DefaultWorkloadNamespace, pki.CABundleConfigMapName)
	coordData1 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.CoordinatorSecretName)
	agentData1 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.AgentSecretName)

	if len(bundleData1) != 1 || bundleData1["ca.crt"] != string(caData1["ca.crt"]) {
		t.Fatalf("vxpu-ca-bundle must contain only ca.crt matching vxpu-ca, got keys=%v", mapKeys(bundleData1))
	}
	if strings.Contains(bundleData1["ca.crt"], "PRIVATE KEY") {
		t.Fatalf("vxpu-ca-bundle contains private key material")
	}

	caCert := checkCACert(t, caData1["ca.crt"], caData1["ca.key"])
	coordCert := checkLeafCert(t, caCert, coordData1["tls.crt"], coordData1["tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert := checkLeafCert(t, caCert, agentData1["tls.crt"], agentData1["tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	checkPKILogs(t, logs1, caCert, coordCert, agentCert)

	// Confirm the CA private key lives in a separate namespace that workloads cannot
	// mount Secrets from, and that neither the router nor executor service accounts
	// can read vxpu-ca through Kubernetes RBAC.
	if pki.DefaultCANamespace == pki.DefaultWorkloadNamespace {
		t.Fatalf("CA namespace must differ from workload namespace")
	}
	for _, sa := range []string{
		"system:serviceaccount:default:vxpu-router",
		"system:serviceaccount:default:vxpu-agent",
		"system:serviceaccount:default:default",
	} {
		if canReadCASecret(t, sa) {
			t.Errorf("%s can read secret/%s in %s; expected RBAC denial", sa, pki.CASecretName, pki.DefaultCANamespace)
		}
	}
	if !canReadCASecret(t, "system:serviceaccount:vxpu-pki:vxpu-pki-init") {
		t.Errorf("vxpu-pki-init service account cannot read secret/%s in %s", pki.CASecretName, pki.DefaultCANamespace)
	}

	// Run 2: delete one leaf Secret and re-run; existing certs stay unchanged, deleted leaf is re-created.
	h.RunCommand("kubectl", "delete", "secret", pki.AgentSecretName, "-n", pki.DefaultWorkloadNamespace)
	h.RunCommand("kubectl", "delete", "job", "vxpu-pki-init", "-n", pki.DefaultCANamespace, "--wait=true")
	h.KubectlApplyContent("vxpu-pki-init-run2", manifest)
	waitForJobComplete(t, h, "vxpu-pki-init", pki.DefaultCANamespace)

	logs2 := jobLogs(t, "vxpu-pki-init", pki.DefaultCANamespace)
	t.Logf("vxpu-pki-init run 2 logs: %s", logs2)

	caData2 := getSecretData(t, pki.DefaultCANamespace, pki.CASecretName)
	bundleData2 := getConfigMapData(t, pki.DefaultWorkloadNamespace, pki.CABundleConfigMapName)
	coordData2 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.CoordinatorSecretName)
	agentData2 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.AgentSecretName)

	if !bytes.Equal(caData1["ca.crt"], caData2["ca.crt"]) || !bytes.Equal(caData1["ca.key"], caData2["ca.key"]) {
		t.Errorf("second run changed vxpu-ca")
	}
	if bundleData1["ca.crt"] != bundleData2["ca.crt"] {
		t.Errorf("second run changed vxpu-ca-bundle")
	}
	if !bytes.Equal(coordData1["tls.crt"], coordData2["tls.crt"]) || !bytes.Equal(coordData1["tls.key"], coordData2["tls.key"]) {
		t.Errorf("second run changed vxpu-coordinator-tls")
	}
	if bytes.Equal(agentData1["tls.crt"], agentData2["tls.crt"]) {
		t.Errorf("second run did not mint a new certificate for deleted vxpu-agent-tls")
	}
	agentCert2 := checkLeafCert(t, caCert, agentData2["tls.crt"], agentData2["tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	checkPKILogs(t, logs2, caCert, coordCert, agentCert2)

	// Run 3: delete vxpu-ca while old bundle and leaves remain; re-running must update
	// the bundle and re-mint both leaves so everything chains to the new CA.
	h.RunCommand("kubectl", "delete", "secret", pki.CASecretName, "-n", pki.DefaultCANamespace)
	h.RunCommand("kubectl", "delete", "job", "vxpu-pki-init", "-n", pki.DefaultCANamespace, "--wait=true")
	h.KubectlApplyContent("vxpu-pki-init-run3", manifest)
	waitForJobComplete(t, h, "vxpu-pki-init", pki.DefaultCANamespace)

	logs3 := jobLogs(t, "vxpu-pki-init", pki.DefaultCANamespace)
	t.Logf("vxpu-pki-init run 3 logs: %s", logs3)

	caData3 := getSecretData(t, pki.DefaultCANamespace, pki.CASecretName)
	bundleData3 := getConfigMapData(t, pki.DefaultWorkloadNamespace, pki.CABundleConfigMapName)
	coordData3 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.CoordinatorSecretName)
	agentData3 := getSecretData(t, pki.DefaultWorkloadNamespace, pki.AgentSecretName)

	if bytes.Equal(caData1["ca.crt"], caData3["ca.crt"]) {
		t.Errorf("third run did not mint a new CA after deleting vxpu-ca")
	}
	if len(bundleData3) != 1 || bundleData3["ca.crt"] != string(caData3["ca.crt"]) {
		t.Errorf("third run did not update vxpu-ca-bundle to the new CA")
	}
	caCert3 := checkCACert(t, caData3["ca.crt"], caData3["ca.key"])
	coordCert3 := checkLeafCert(t, caCert3, coordData3["tls.crt"], coordData3["tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-router", x509.ExtKeyUsageClientAuth)
	agentCert3 := checkLeafCert(t, caCert3, agentData3["tls.crt"], agentData3["tls.key"],
		"spiffe://cluster.local/ns/default/sa/vxpu-agent", x509.ExtKeyUsageServerAuth)
	if !strings.Contains(logs3, `status="re-minted"`) {
		t.Errorf("third run logs missing status=re-minted: %s", logs3)
	}
	checkPKILogs(t, logs3, caCert3, coordCert3, agentCert3)
}

func waitForJobComplete(t *testing.T, h *Harness, name, namespace string) {
	t.Helper()
	cmd := exec.Command("kubectl", "wait", "--for=condition=complete", "job/"+name, "-n", namespace, "--timeout=2m")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("Job logs: %s", h.GetPodLogs("job-name="+name, namespace))
		t.Logf("Events: %s", h.GetEvents(namespace))
		t.Fatalf("waiting for job %s/%s: %v (%s)", namespace, name, err, out)
	}
}

func jobLogs(t *testing.T, name, namespace string) string {
	t.Helper()
	out, err := exec.Command("kubectl", "logs", "job/"+name, "-n", namespace).CombinedOutput()
	if err != nil {
		t.Fatalf("getting logs for job %s/%s: %v (%s)", namespace, name, err, out)
	}
	return string(out)
}

func getSecretData(t *testing.T, namespace, name string) map[string][]byte {
	t.Helper()
	out, err := exec.Command("kubectl", "get", "secret", name, "-n", namespace, "-o", "json").CombinedOutput()
	if err != nil {
		t.Fatalf("getting secret %s/%s: %v (%s)", namespace, name, err, out)
	}
	var obj struct {
		Data map[string]string `json:"data"`
	}
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("unmarshaling secret %s/%s: %v", namespace, name, err)
	}
	decoded := make(map[string][]byte, len(obj.Data))
	for k, v := range obj.Data {
		b, err := base64.StdEncoding.DecodeString(v)
		if err != nil {
			t.Fatalf("decoding secret %s/%s key %q: %v", namespace, name, k, err)
		}
		decoded[k] = b
	}
	return decoded
}

func getConfigMapData(t *testing.T, namespace, name string) map[string]string {
	t.Helper()
	out, err := exec.Command("kubectl", "get", "configmap", name, "-n", namespace, "-o", "json").CombinedOutput()
	if err != nil {
		t.Fatalf("getting configmap %s/%s: %v (%s)", namespace, name, err, out)
	}
	var obj struct {
		Data map[string]string `json:"data"`
	}
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("unmarshaling configmap %s/%s: %v", namespace, name, err)
	}
	return obj.Data
}

func canReadCASecret(t *testing.T, subject string) bool {
	t.Helper()
	cmd := exec.Command("kubectl", "auth", "can-i", "get", "secret/"+pki.CASecretName,
		"-n", pki.DefaultCANamespace, "--as", subject)
	out, err := cmd.CombinedOutput()
	ans := strings.TrimSpace(string(out))
	if err == nil && ans == "yes" {
		return true
	}
	if ans == "no" {
		return false
	}
	t.Fatalf("unexpected kubectl auth can-i output for %s: err=%v out=%q", subject, err, ans)
	return false
}

func checkCACert(t *testing.T, certPEM, keyPEM []byte) *x509.Certificate {
	t.Helper()
	kp, err := pki.ParseKeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("parsing CA key pair: %v", err)
	}
	if !kp.Cert.IsCA || !kp.Cert.BasicConstraintsValid || !kp.Cert.MaxPathLenZero || kp.Cert.MaxPathLen != 0 {
		t.Errorf("CA cert missing expected basic constraints")
	}
	if kp.Cert.KeyUsage != (x509.KeyUsageCertSign | x509.KeyUsageCRLSign) {
		t.Errorf("CA KeyUsage = %v, want CertSign|CRLSign", kp.Cert.KeyUsage)
	}
	return kp.Cert
}

func checkLeafCert(t *testing.T, ca *x509.Certificate, certPEM, keyPEM []byte, wantSPIFFE string, wantEKU x509.ExtKeyUsage) *x509.Certificate {
	t.Helper()
	cert, err := pki.VerifyLeaf(ca, certPEM, keyPEM, wantSPIFFE, wantEKU, time.Now().UTC())
	if err != nil {
		t.Fatalf("VerifyLeaf failed: %v", err)
	}
	wrongEKU := x509.ExtKeyUsageServerAuth
	if wantEKU == x509.ExtKeyUsageServerAuth {
		wrongEKU = x509.ExtKeyUsageClientAuth
	}
	if _, err := pki.VerifyLeaf(ca, certPEM, keyPEM, wantSPIFFE, wrongEKU, time.Now().UTC()); err == nil {
		t.Errorf("leaf unexpectedly verified for wrong EKU %v", wrongEKU)
	}
	return cert
}

func checkPKILogs(t *testing.T, logs string, certs ...*x509.Certificate) {
	t.Helper()
	if strings.Contains(logs, "PRIVATE KEY") {
		t.Errorf("Job logs contain private key material: %s", logs)
	}
	for _, c := range certs {
		id := c.Subject.CommonName
		if len(c.URIs) > 0 {
			id = c.URIs[0].String()
		}
		if !strings.Contains(logs, id) {
			t.Errorf("Job logs missing identity %q: %s", id, logs)
		}
		exp := c.NotAfter.UTC().Format(time.RFC3339)
		if !strings.Contains(logs, exp) {
			t.Errorf("Job logs missing expiry %q for %s: %s", exp, id, logs)
		}
	}
}

func mapKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
