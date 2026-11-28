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

package router

import (
	"context"
	"net"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/mtls"
	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func TestServerAgentMTLS(t *testing.T) {
	pkiDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), pkiDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir: %v", err)
	}

	serverCreds, err := mtls.LoadAgentServerCredentials(t.Context(), filepath.Join(pkiDir, "agent"))
	if err != nil {
		t.Fatalf("LoadAgentServerCredentials: %v", err)
	}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	gs := grpc.NewServer(grpc.Creds(serverCreds))
	pb.RegisterAgentServer(gs, agent.NewServer(t.TempDir(), t.TempDir(), nil))
	go func() {
		_ = gs.Serve(lis)
	}()
	t.Cleanup(func() {
		gs.Stop()
		_ = lis.Close()
	})

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("SplitHostPort: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("Atoi: %v", err)
	}

	// Valid coordinator credentials connect to the mTLS agent.
	coordCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(pkiDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials: %v", err)
	}
	srv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = srv.Close() })
	srv.agentPort = port
	srv.UseAgentCredentials(coordCreds)

	conn, err := srv.agentConn(host)
	if err != nil {
		t.Fatalf("agentConn: %v", err)
	}
	callCtx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if _, err := pb.NewAgentClient(conn).DescribeMachine(callCtx, &pb.DescribeMachineRequest{}); err != nil {
		t.Fatalf("DescribeMachine over router mTLS failed: %v", err)
	}

	// Plaintext router (no agent credentials) is rejected by the mTLS agent.
	plainSrv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = plainSrv.Close() })
	plainSrv.agentPort = port
	plainConn, err := plainSrv.agentConn(host)
	if err != nil {
		t.Fatalf("plain agentConn: %v", err)
	}
	plainCtx, plainCancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer plainCancel()
	if _, err := pb.NewAgentClient(plainConn).DescribeMachine(plainCtx, &pb.DescribeMachineRequest{}); err == nil {
		t.Fatalf("expected plaintext router connection to be rejected by mTLS agent")
	}

	// Router with wrong-CA coordinator credentials is rejected by the mTLS agent.
	otherPKIDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), otherPKIDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir(other): %v", err)
	}
	wrongCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(otherPKIDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials(other): %v", err)
	}
	wrongSrv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = wrongSrv.Close() })
	wrongSrv.agentPort = port
	wrongSrv.UseAgentCredentials(wrongCreds)
	wrongConn, err := wrongSrv.agentConn(host)
	if err != nil {
		t.Fatalf("wrong agentConn: %v", err)
	}
	wrongCtx, wrongCancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer wrongCancel()
	if _, err := pb.NewAgentClient(wrongConn).DescribeMachine(wrongCtx, &pb.DescribeMachineRequest{}); err == nil {
		t.Fatalf("expected wrong-CA router connection to be rejected by mTLS agent")
	}
}

func TestGetOrStartPod_AgentMTLS(t *testing.T) {
	pkiDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), pkiDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir: %v", err)
	}
	coordCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(pkiDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials: %v", err)
	}

	cs := fake.NewClientset()
	cs.PrependReactor("create", "pods", func(action k8stesting.Action) (bool, runtime.Object, error) {
		pod := action.(k8stesting.CreateAction).GetObject().(*corev1.Pod)
		pod.Status = corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "10.0.0.20"}
		return false, nil, nil
	})
	want := PlanPlacement("", 1_000_000_000)

	// Start without mTLS: pod binds vxpu-agent SA but omits TLS volume and --tls-dir.
	srv := NewServer(cs, "ns", "img:v1", "")
	t.Cleanup(func() { _ = srv.Close() })
	if _, err := srv.getOrStartPod(t.Context(), "m-mtls", &want); err != nil {
		t.Fatalf("getOrStartPod without mTLS: %v", err)
	}
	pod, err := cs.CoreV1().Pods("ns").Get(t.Context(), "vxpu-executor-m-mtls", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if pod.Spec.ServiceAccountName != agentServiceAccountName {
		t.Errorf("ServiceAccountName = %q, want %q", pod.Spec.ServiceAccountName, agentServiceAccountName)
	}
	if pod.Spec.AutomountServiceAccountToken == nil || *pod.Spec.AutomountServiceAccountToken {
		t.Errorf("AutomountServiceAccountToken = %v, want false", pod.Spec.AutomountServiceAccountToken)
	}
	if slices.Contains(pod.Spec.Containers[0].Args, "--tls-dir="+agentTLSMountPath) {
		t.Errorf("unexpected --tls-dir arg when agentCreds is nil: %v", pod.Spec.Containers[0].Args)
	}

	// Enabling mTLS recreates the pod with projected /var/run/vxpu/tls and --tls-dir.
	srv.UseAgentCredentials(coordCreds)
	if _, err := srv.getOrStartPod(t.Context(), "m-mtls", &want); err != nil {
		t.Fatalf("getOrStartPod with mTLS: %v", err)
	}
	pod, err = cs.CoreV1().Pods("ns").Get(t.Context(), "vxpu-executor-m-mtls", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if pod.Spec.ServiceAccountName != agentServiceAccountName {
		t.Errorf("ServiceAccountName = %q, want %q", pod.Spec.ServiceAccountName, agentServiceAccountName)
	}
	if pod.Spec.AutomountServiceAccountToken == nil || *pod.Spec.AutomountServiceAccountToken {
		t.Errorf("AutomountServiceAccountToken = %v, want false", pod.Spec.AutomountServiceAccountToken)
	}
	if !slices.Contains(pod.Spec.Containers[0].Args, "--tls-dir="+agentTLSMountPath) {
		t.Errorf("args = %v, want --tls-dir=%s", pod.Spec.Containers[0].Args, agentTLSMountPath)
	}

	var foundMount bool
	for _, vm := range pod.Spec.Containers[0].VolumeMounts {
		if vm.Name == agentTLSVolumeName {
			foundMount = true
			if vm.MountPath != agentTLSMountPath || !vm.ReadOnly {
				t.Errorf("TLS volumeMount = %+v, want mountPath=%s readOnly=true", vm, agentTLSMountPath)
			}
		}
	}
	if !foundMount {
		t.Fatalf("missing %q volumeMount on executor container", agentTLSVolumeName)
	}

	var proj *corev1.ProjectedVolumeSource
	for _, v := range pod.Spec.Volumes {
		if v.Name == agentTLSVolumeName {
			proj = v.Projected
		}
	}
	if proj == nil || len(proj.Sources) != 2 {
		t.Fatalf("projected TLS volume = %+v, want 2 sources", proj)
	}
	cm := proj.Sources[0].ConfigMap
	if cm == nil || cm.Name != pki.CABundleConfigMapName || len(cm.Items) != 1 || cm.Items[0].Key != "ca.crt" || cm.Items[0].Path != "ca.crt" {
		t.Errorf("CA bundle projection = %+v, want ConfigMap %s with ca.crt", cm, pki.CABundleConfigMapName)
	}
	sec := proj.Sources[1].Secret
	if sec == nil || sec.Name != pki.AgentSecretName || len(sec.Items) != 2 {
		t.Fatalf("agent TLS projection = %+v, want Secret %s with 2 items", sec, pki.AgentSecretName)
	}
	for _, item := range sec.Items {
		if item.Key != item.Path || (item.Key != corev1.TLSCertKey && item.Key != corev1.TLSPrivateKeyKey) {
			t.Errorf("unexpected secret item %+v", item)
		}
		if item.Mode == nil || *item.Mode != 0o400 {
			t.Errorf("secret item %s mode = %v, want 0400", item.Key, item.Mode)
		}
	}
}
