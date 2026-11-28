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
	"io"
	"slices"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"

	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func TestApplyManifest_MissingKindOrName(t *testing.T) {
	orig := getK8sClient
	t.Cleanup(func() { getK8sClient = orig })
	getK8sClient = func() (*k8sClient, error) { return &k8sClient{}, nil }

	manifestWithoutKind := `
apiVersion: v1
metadata:
  name: test-name
`
	err := applyManifest(t.Context(), manifestWithoutKind)
	if err == nil {
		t.Fatal("expected error for manifest without kind, got nil")
	}
	if !strings.Contains(err.Error(), "manifest document 0 has no kind or name") {
		t.Errorf("unexpected error message: %v", err)
	}

	manifestWithoutName := `
apiVersion: v1
kind: ServiceAccount
metadata:
  namespace: test-ns
`
	err = applyManifest(t.Context(), manifestWithoutName)
	if err == nil {
		t.Fatal("expected error for manifest without name, got nil")
	}
	if !strings.Contains(err.Error(), "manifest document 0 has no kind or name") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestRouterManifest_AgentMTLS(t *testing.T) {
	dec := yaml.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(routerManifest)), 4096)
	var agentSA *corev1.ServiceAccount
	var routerPod *corev1.Pod
	for {
		var raw map[string]any
		err := dec.Decode(&raw)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("decoding router.yaml: %v", err)
		}
		if len(raw) == 0 {
			continue
		}
		u := unstructured.Unstructured{Object: raw}
		switch u.GetKind() {
		case "ServiceAccount":
			if u.GetName() == "vxpu-agent" {
				var sa corev1.ServiceAccount
				if err := runtime.DefaultUnstructuredConverter.FromUnstructured(raw, &sa); err != nil {
					t.Fatalf("converting vxpu-agent ServiceAccount: %v", err)
				}
				agentSA = &sa
			}
		case "Pod":
			var pod corev1.Pod
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(raw, &pod); err != nil {
				t.Fatalf("converting router Pod: %v", err)
			}
			routerPod = &pod
		}
	}

	if agentSA == nil {
		t.Fatalf("router.yaml missing ServiceAccount vxpu-agent")
	}
	if agentSA.AutomountServiceAccountToken == nil || *agentSA.AutomountServiceAccountToken {
		t.Errorf("vxpu-agent automountServiceAccountToken = %v, want false", agentSA.AutomountServiceAccountToken)
	}
	if routerPod == nil {
		t.Fatalf("router.yaml missing router Pod")
	}

	const wantTLSDir = "/var/run/vxpu/tls"
	c := routerPod.Spec.Containers[0]
	if !slices.Contains(c.Args, "--tls-dir="+wantTLSDir) {
		t.Errorf("router args = %v, want --tls-dir=%s", c.Args, wantTLSDir)
	}

	var foundMount bool
	for _, vm := range c.VolumeMounts {
		if vm.Name == "vxpu-tls" {
			foundMount = true
			if vm.MountPath != wantTLSDir || !vm.ReadOnly {
				t.Errorf("router TLS volumeMount = %+v, want mountPath=%s readOnly=true", vm, wantTLSDir)
			}
		}
	}
	if !foundMount {
		t.Fatalf("router.yaml missing vxpu-tls volumeMount")
	}

	var proj *corev1.ProjectedVolumeSource
	for _, v := range routerPod.Spec.Volumes {
		if v.Name == "vxpu-tls" {
			proj = v.Projected
		}
	}
	if proj == nil || len(proj.Sources) != 2 {
		t.Fatalf("router projected TLS volume = %+v, want 2 sources", proj)
	}
	cm := proj.Sources[0].ConfigMap
	if cm == nil || cm.Name != pki.CABundleConfigMapName || len(cm.Items) != 1 || cm.Items[0].Key != "ca.crt" || cm.Items[0].Path != "ca.crt" {
		t.Errorf("router CA bundle projection = %+v, want ConfigMap %s with ca.crt", cm, pki.CABundleConfigMapName)
	}
	sec := proj.Sources[1].Secret
	if sec == nil || sec.Name != pki.CoordinatorSecretName || len(sec.Items) != 2 {
		t.Fatalf("router coordinator TLS projection = %+v, want Secret %s with 2 items", sec, pki.CoordinatorSecretName)
	}
	for _, item := range sec.Items {
		if item.Key != item.Path || (item.Key != corev1.TLSCertKey && item.Key != corev1.TLSPrivateKeyKey) {
			t.Errorf("unexpected secret item %+v", item)
		}
		if item.Mode == nil || *item.Mode != 0o400 {
			t.Errorf("secret item %s mode = %v, want 0400 (%d)", item.Key, item.Mode, 0o400)
		}
	}
}
