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
	"context"
	"crypto/x509"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

var pkiLabels = map[string]string{"app.kubernetes.io/part-of": "vxpu-pki"}

// EnsureK8s ensures the CA Secret, CA bundle ConfigMap, coordinator TLS Secret,
// and agent TLS Secret exist and form a consistent set in Kubernetes. Valid
// objects remain unchanged; a stale CA bundle is updated to the current CA, and
// any missing, expired, or non-verifying leaf Secret is minted from the CA.
func EnsureK8s(ctx context.Context, client kubernetes.Interface, opts Options) error {
	opts = opts.withDefaults()
	if opts.CANamespace == opts.WorkloadNamespace {
		return fmt.Errorf("CA namespace %q must differ from workload namespace so workloads cannot mount the CA key", opts.CANamespace)
	}
	ca, err := ensureCASecret(ctx, client, opts)
	if err != nil {
		return err
	}
	if err := ensureCABundle(ctx, client, opts.WorkloadNamespace, ca.CertPEM); err != nil {
		return err
	}
	if err := ensureLeafSecret(ctx, client, opts.WorkloadNamespace, CoordinatorSecretName, ca,
		CoordinatorSPIFFEID(opts.TrustDomain, opts.WorkloadNamespace), x509.ExtKeyUsageClientAuth, opts.LeafTTL); err != nil {
		return err
	}
	return ensureLeafSecret(ctx, client, opts.WorkloadNamespace, AgentSecretName, ca,
		AgentSPIFFEID(opts.TrustDomain, opts.WorkloadNamespace), x509.ExtKeyUsageServerAuth, opts.LeafTTL)
}

func ensureCASecret(ctx context.Context, client kubernetes.Interface, opts Options) (*KeyPair, error) {
	secrets := client.CoreV1().Secrets(opts.CANamespace)
	status := "kept"
	sec, err := secrets.Get(ctx, CASecretName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		ca, mintErr := MintCA(opts.CATTL)
		if mintErr != nil {
			return nil, mintErr
		}
		sec = &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: CASecretName, Namespace: opts.CANamespace, Labels: pkiLabels},
			Type:       corev1.SecretTypeOpaque,
			Data:       map[string][]byte{"ca.crt": ca.CertPEM, "ca.key": ca.KeyPEM},
		}
		if sec, err = secrets.Create(ctx, sec, metav1.CreateOptions{}); apierrors.IsAlreadyExists(err) {
			sec, err = secrets.Get(ctx, CASecretName, metav1.GetOptions{})
		} else if err == nil {
			status = "created"
		}
	}
	if err != nil {
		return nil, fmt.Errorf("ensuring Secret %s/%s: %w", opts.CANamespace, CASecretName, err)
	}
	ca, err := ParseKeyPair(sec.Data["ca.crt"], sec.Data["ca.key"])
	if err != nil {
		return nil, fmt.Errorf("parsing Secret %s/%s: %w", opts.CANamespace, CASecretName, err)
	}
	if !ca.Cert.IsCA || time.Now().UTC().After(ca.Cert.NotAfter) {
		return nil, fmt.Errorf("Secret %s/%s is not a valid unexpired CA", opts.CANamespace, CASecretName)
	}
	logCert(ctx, CASecretName, ca.Cert, status, "")
	return ca, nil
}

func ensureCABundle(ctx context.Context, client kubernetes.Interface, namespace string, caCertPEM []byte) error {
	cms := client.CoreV1().ConfigMaps(namespace)
	want := string(caCertPEM)
	existing, err := cms.Get(ctx, CABundleConfigMapName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: CABundleConfigMapName, Namespace: namespace, Labels: pkiLabels},
			Data:       map[string]string{"ca.crt": want},
		}
		if _, err = cms.Create(ctx, cm, metav1.CreateOptions{}); err == nil {
			klog.FromContext(ctx).Info("CA bundle ready", "name", CABundleConfigMapName, "status", "created")
			return nil
		}
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("creating ConfigMap %s/%s: %w", namespace, CABundleConfigMapName, err)
		}
		existing, err = cms.Get(ctx, CABundleConfigMapName, metav1.GetOptions{})
	}
	if err != nil {
		return fmt.Errorf("getting ConfigMap %s/%s: %w", namespace, CABundleConfigMapName, err)
	}
	if len(existing.Data) == 1 && existing.Data["ca.crt"] == want {
		klog.FromContext(ctx).Info("CA bundle ready", "name", CABundleConfigMapName, "status", "kept")
		return nil
	}
	updated := existing.DeepCopy()
	updated.Data = map[string]string{"ca.crt": want}
	if _, err := cms.Update(ctx, updated, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("updating ConfigMap %s/%s: %w", namespace, CABundleConfigMapName, err)
	}
	klog.FromContext(ctx).Info("CA bundle ready", "name", CABundleConfigMapName, "status", "updated", "reason", "CA certificate changed")
	return nil
}

func ensureLeafSecret(ctx context.Context, client kubernetes.Interface, namespace, name string,
	ca *KeyPair, spiffeID string, eku x509.ExtKeyUsage, ttl time.Duration) error {
	secrets := client.CoreV1().Secrets(namespace)
	existing, err := secrets.Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		leaf, mintErr := MintLeaf(ca, spiffeID, eku, ttl)
		if mintErr != nil {
			return mintErr
		}
		sec := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: pkiLabels},
			Type:       corev1.SecretTypeTLS,
			Data:       map[string][]byte{corev1.TLSCertKey: leaf.CertPEM, corev1.TLSPrivateKeyKey: leaf.KeyPEM},
		}
		if _, err = secrets.Create(ctx, sec, metav1.CreateOptions{}); err == nil {
			logCert(ctx, name, leaf.Cert, "created", "")
			return nil
		}
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("creating Secret %s/%s: %w", namespace, name, err)
		}
		existing, err = secrets.Get(ctx, name, metav1.GetOptions{})
	}
	if err != nil {
		return fmt.Errorf("getting Secret %s/%s: %w", namespace, name, err)
	}
	cert, verifyErr := VerifyLeaf(ca.Cert, existing.Data[corev1.TLSCertKey], existing.Data[corev1.TLSPrivateKeyKey], spiffeID, eku, time.Now().UTC())
	if verifyErr == nil {
		logCert(ctx, name, cert, "kept", "")
		return nil
	}
	leaf, err := MintLeaf(ca, spiffeID, eku, ttl)
	if err != nil {
		return err
	}
	updated := existing.DeepCopy()
	updated.Data = map[string][]byte{corev1.TLSCertKey: leaf.CertPEM, corev1.TLSPrivateKeyKey: leaf.KeyPEM}
	if _, err := secrets.Update(ctx, updated, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("updating Secret %s/%s: %w", namespace, name, err)
	}
	logCert(ctx, name, leaf.Cert, "re-minted", verifyErr.Error())
	return nil
}
