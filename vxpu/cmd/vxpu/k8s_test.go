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
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
)

func setTestK8sClient(t *testing.T, client *k8sClient) {
	orig := getK8sClient
	t.Cleanup(func() { getK8sClient = orig })
	getK8sClient = func() (*k8sClient, error) { return client, nil }
}

func newTestK8sClient() (*k8sClient, *fake.Clientset) {
	clientset := fake.NewSimpleClientset()
	client := &k8sClient{
		clientset: clientset,
		namespace: "test-ns",
	}
	return client, clientset
}

func TestPodSettingsAndPhase(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	// When pod does not exist
	settings, err := podSettings(t.Context(), "vxpu-router")
	if err != nil {
		t.Fatalf("podSettings for missing pod failed: %v", err)
	}
	if settings != nil {
		t.Errorf("expected nil settings for missing pod, got %v", settings)
	}

	phase, err := podPhase(t.Context(), "vxpu-router")
	if err != nil {
		t.Fatalf("podPhase for missing pod failed: %v", err)
	}
	if phase != "" {
		t.Errorf("expected empty phase for missing pod, got %q", phase)
	}

	// Create a pod
	p := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-router",
			Namespace: client.namespace,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "router",
					Image: "router:v1",
					Env: []corev1.EnvVar{
						{Name: "VXPU_EXECUTOR_IMAGE", Value: "exec:v1"},
						{Name: "VXPU_ACCELERATOR", Value: "nvidia-l4"},
						{Name: "VXPU_MODELSTORE_URL", Value: "http://store"},
						{Name: "VXPU_EXTERNAL_EXECUTORS", Value: "ext:1234"},
						{Name: "VXPU_MODELSTORE_EXTERNAL_URL", Value: "http://ext-store"},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}
	if _, err := clientset.CoreV1().Pods(client.namespace).Create(t.Context(), p, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating pod: %v", err)
	}

	settings, err = podSettings(t.Context(), "vxpu-router")
	if err != nil {
		t.Fatalf("podSettings failed: %v", err)
	}
	if settings == nil {
		t.Fatal("expected settings, got nil")
	}
	if settings.image != "router:v1" || settings.executorImage != "exec:v1" ||
		settings.accelerator != "nvidia-l4" || settings.modelstore != "http://store" ||
		settings.externalExecutors != "ext:1234" || settings.externalModelstore != "http://ext-store" {
		t.Errorf("unexpected settings: %+v", settings)
	}

	phase, err = podPhase(t.Context(), "vxpu-router")
	if err != nil {
		t.Fatalf("podPhase failed: %v", err)
	}
	if phase != "Running" {
		t.Errorf("expected phase Running, got %q", phase)
	}
}

func TestDeletePod(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	p := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-router",
			Namespace: client.namespace,
		},
	}
	if _, err := clientset.CoreV1().Pods(client.namespace).Create(t.Context(), p, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating pod: %v", err)
	}

	if err := deletePod(t.Context(), "vxpu-router"); err != nil {
		t.Fatalf("deletePod failed: %v", err)
	}

	phase, err := podPhase(t.Context(), "vxpu-router")
	if err != nil {
		t.Fatalf("podPhase failed: %v", err)
	}
	if phase != "" {
		t.Errorf("expected pod to be gone, got phase %q", phase)
	}
}

func TestWaitReady(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	p := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vxpu-router",
			Namespace: client.namespace,
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{
					Type:   corev1.PodReady,
					Status: corev1.ConditionTrue,
				},
			},
		},
	}
	if _, err := clientset.CoreV1().Pods(client.namespace).Create(t.Context(), p, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating pod: %v", err)
	}

	if err := waitReady(t.Context(), "vxpu-router"); err != nil {
		t.Fatalf("waitReady failed: %v", err)
	}
}

func TestDeleteRouterResources(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	if err := deleteRouterResources(t.Context(), "vxpu-router"); err != nil {
		t.Fatalf("deleteRouterResources failed: %v", err)
	}

	collections := map[string]bool{}
	deleted := map[string]bool{}
	for _, action := range clientset.Actions() {
		if action.GetVerb() == "delete-collection" {
			collections[action.GetResource().Resource] = true
		}
		if action.GetVerb() == "delete" {
			deleted[action.GetResource().Resource] = true
		}
	}
	for _, res := range []string{"pods", "serviceaccounts", "roles", "rolebindings"} {
		if !collections[res] {
			t.Errorf("expected delete-collection on %s, actions: %v", res, clientset.Actions())
		}
	}
	if !deleted["services"] {
		t.Errorf("expected delete on services, actions: %v", clientset.Actions())
	}
}

func TestPortForwardPod_NilConfig(t *testing.T) {
	client, _ := newTestK8sClient()
	_, _, err := portForwardPod(t.Context(), client, client.namespace, "vxpu-router", 50051)
	if err == nil {
		t.Fatal("expected error with nil config, got nil")
	}
	if !strings.Contains(err.Error(), "without rest.Config") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPortForwardService_MissingService(t *testing.T) {
	client, _ := newTestK8sClient()
	setTestK8sClient(t, client)

	_, _, err := portForwardService(t.Context(), client.namespace, "nonexistent", 80)
	if err == nil {
		t.Fatal("expected error for nonexistent service, got nil")
	}
}

func TestPortForwardService_NoRunningPod(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-svc",
			Namespace: client.namespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": "test"},
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt32(8080),
				},
			},
		},
	}
	if _, err := clientset.CoreV1().Services(client.namespace).Create(t.Context(), svc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating service: %v", err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	_, _, err := portForwardService(ctx, client.namespace, "test-svc", 80)
	if err == nil {
		t.Fatal("expected error when no pods match selector, got nil")
	}
}

func TestPortForwardService_PrefersReadyPod(t *testing.T) {
	client, clientset := newTestK8sClient()
	setTestK8sClient(t, client)

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-svc",
			Namespace: client.namespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": "test"},
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt32(8080),
				},
			},
		},
	}
	if _, err := clientset.CoreV1().Services(client.namespace).Create(t.Context(), svc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating service: %v", err)
	}

	podUnready := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-unready",
			Namespace: client.namespace,
			Labels:    map[string]string{"app": "test"},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "c", Ports: []corev1.ContainerPort{{ContainerPort: 8080}}},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionFalse},
			},
		},
	}
	podReady := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-ready",
			Namespace: client.namespace,
			Labels:    map[string]string{"app": "test"},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "c", Ports: []corev1.ContainerPort{{ContainerPort: 8080}}},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
	if _, err := clientset.CoreV1().Pods(client.namespace).Create(t.Context(), podUnready, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating unready pod: %v", err)
	}
	if _, err := clientset.CoreV1().Pods(client.namespace).Create(t.Context(), podReady, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating ready pod: %v", err)
	}

	// portForwardService should find pod-ready and fail on portForwardPod (nil rest.Config)
	// rather than timing out looking for a pod.
	_, _, err := portForwardService(t.Context(), client.namespace, "test-svc", 80)
	if err == nil || !strings.Contains(err.Error(), "without rest.Config") {
		t.Fatalf("expected error from portForwardPod with nil config, got %v", err)
	}
}
