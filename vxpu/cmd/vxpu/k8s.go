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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
	"k8s.io/klog/v2"
	sigyaml "sigs.k8s.io/yaml"
)

// errPodNeedsRecreate signals that the API server rejected an in-place pod update
// with an invalid error, indicating that an immutable pod field changed.
var errPodNeedsRecreate = errors.New("router pod cannot be updated in place")

// k8sClient wraps a Kubernetes clientset, REST configuration, and target namespace.
type k8sClient struct {
	clientset kubernetes.Interface
	config    *rest.Config
	namespace string
}

// newK8sClient initializes a Kubernetes client following standard clientcmd loading
// rules, which fall back to in-cluster credentials and namespace automatically.
func newK8sClient() (*k8sClient, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})
	config, err := clientConfig.ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("loading kubeconfig: %w", err)
	}
	namespace, _, err := clientConfig.Namespace()
	if err != nil || namespace == "" {
		namespace = "default"
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("creating kubernetes client: %w", err)
	}
	return &k8sClient{
		clientset: clientset,
		config:    config,
		namespace: namespace,
	}, nil
}

// getK8sClient returns the process-wide Kubernetes client, initializing it on first use.
var getK8sClient = sync.OnceValues(newK8sClient)

// podSettings reads a running router's image, executor image, accelerator, and
// modelstore configuration. It returns nil when the pod does not exist.
func podSettings(ctx context.Context, pod string) (*routerSettings, error) {
	client, err := getK8sClient()
	if err != nil {
		return nil, err
	}
	p, err := client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting pod %s: %w", pod, err)
	}
	if len(p.Spec.Containers) == 0 {
		return nil, nil
	}
	c := p.Spec.Containers[0]
	settings := &routerSettings{image: c.Image}
	for _, env := range c.Env {
		switch env.Name {
		case "VXPU_EXECUTOR_IMAGE":
			settings.executorImage = env.Value
		case "VXPU_ACCELERATOR":
			settings.accelerator = env.Value
		case "VXPU_MODELSTORE_URL":
			settings.modelstore = env.Value
		case "VXPU_EXTERNAL_EXECUTORS":
			settings.externalExecutors = env.Value
		case "VXPU_MODELSTORE_EXTERNAL_URL":
			settings.externalModelstore = env.Value
		}
	}
	return settings, nil
}

// podPhase returns the phase of the router pod, or empty string when the pod does not exist.
func podPhase(ctx context.Context, pod string) (string, error) {
	client, err := getK8sClient()
	if err != nil {
		return "", err
	}
	p, err := client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("getting pod %s: %w", pod, err)
	}
	return string(p.Status.Phase), nil
}

// deletePod removes the named pod and waits until it is gone from the cluster.
func deletePod(ctx context.Context, pod string) error {
	client, err := getK8sClient()
	if err != nil {
		return err
	}
	err = client.clientset.CoreV1().Pods(client.namespace).Delete(ctx, pod, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting pod %s: %w", pod, err)
	}
	waitCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if err := wait.PollUntilContextCancel(waitCtx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		_, err := client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		return false, nil
	}); err != nil {
		return fmt.Errorf("waiting for pod %s to be deleted: %w", pod, err)
	}
	return nil
}

// waitReady polls until the specified pod is Ready. If the pod does not
// become ready within 900 seconds, waitReady prints diagnostics (pod
// YAML and container logs) to standard error and returns the timeout error.
func waitReady(ctx context.Context, pod string) error {
	log := klog.FromContext(ctx)
	client, err := getK8sClient()
	if err != nil {
		return err
	}
	waitCtx, cancel := context.WithTimeout(ctx, 900*time.Second)
	defer cancel()

	err = wait.PollUntilContextCancel(waitCtx, time.Second, true, func(ctx context.Context) (bool, error) {
		p, err := client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		for _, cond := range p.Status.Conditions {
			if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "pod %s failed to become ready. Printing pod details and logs:\n", pod)
		p, getErr := client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
		if getErr != nil {
			log.Error(getErr, "collecting diagnostics: getting pod", "pod", pod)
		} else {
			y, yerr := sigyaml.Marshal(p)
			if yerr != nil {
				log.Error(yerr, "marshaling pod yaml", "pod", pod)
			} else {
				fmt.Fprintf(os.Stderr, "$ pod details\n%s\n", string(y))
			}
			tailLines := int64(50)
			var containers []corev1.Container
			containers = append(containers, p.Spec.InitContainers...)
			containers = append(containers, p.Spec.Containers...)
			for _, c := range containers {
				req := client.clientset.CoreV1().Pods(client.namespace).GetLogs(pod, &corev1.PodLogOptions{
					Container: c.Name,
					TailLines: &tailLines,
				})
				raw, logErr := req.DoRaw(ctx)
				if logErr != nil {
					log.Error(logErr, "collecting diagnostics: getting logs", "pod", pod, "container", c.Name)
				} else {
					fmt.Fprintf(os.Stderr, "$ logs %s (container %s)\n%s\n", pod, c.Name, string(raw))
				}
			}
		}
		return fmt.Errorf("waiting for pod %s to be ready: %w", pod, err)
	}
	return nil
}

// applyManifest parses and applies each document in manifest using server-side apply.
func applyManifest(ctx context.Context, manifest string) error {
	client, err := getK8sClient()
	if err != nil {
		return err
	}
	reader := yaml.NewYAMLReader(bufio.NewReader(strings.NewReader(manifest)))
	for {
		doc, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("reading manifest: %w", err)
		}
		doc = bytes.TrimSpace(doc)
		if len(doc) == 0 {
			continue
		}
		var unstr unstructured.Unstructured
		if err := yaml.Unmarshal(doc, &unstr.Object); err != nil {
			return fmt.Errorf("unmarshaling manifest document: %w", err)
		}
		kind := unstr.GetKind()
		name := unstr.GetName()
		if kind == "" || name == "" {
			continue
		}

		// A RoleBinding's ServiceAccount subject needs a namespace if not specified.
		if kind == "RoleBinding" {
			subjects, found, err := unstructured.NestedSlice(unstr.Object, "subjects")
			if err != nil {
				return fmt.Errorf("reading rolebinding subjects: %w", err)
			}
			if found {
				for _, s := range subjects {
					if sm, ok := s.(map[string]any); ok {
						if sm["kind"] == "ServiceAccount" && (sm["namespace"] == nil || sm["namespace"] == "") {
							sm["namespace"] = client.namespace
						}
					}
				}
				if err := unstructured.SetNestedSlice(unstr.Object, subjects, "subjects"); err != nil {
					return fmt.Errorf("setting rolebinding subjects: %w", err)
				}
			}
		}

		data, err := json.Marshal(unstr.Object)
		if err != nil {
			return fmt.Errorf("marshaling %s %s: %w", kind, name, err)
		}

		force := true
		opts := metav1.PatchOptions{
			FieldManager: "vxpu",
			Force:        &force,
		}

		switch kind {
		case "ServiceAccount":
			_, err = client.clientset.CoreV1().ServiceAccounts(client.namespace).Patch(ctx, name, types.ApplyPatchType, data, opts)
		case "Role":
			_, err = client.clientset.RbacV1().Roles(client.namespace).Patch(ctx, name, types.ApplyPatchType, data, opts)
		case "RoleBinding":
			_, err = client.clientset.RbacV1().RoleBindings(client.namespace).Patch(ctx, name, types.ApplyPatchType, data, opts)
		case "Service":
			_, err = client.clientset.CoreV1().Services(client.namespace).Patch(ctx, name, types.ApplyPatchType, data, opts)
		case "Pod":
			_, err = client.clientset.CoreV1().Pods(client.namespace).Patch(ctx, name, types.ApplyPatchType, data, opts)
			if err != nil && apierrors.IsInvalid(err) {
				return fmt.Errorf("%w: %w", errPodNeedsRecreate, err)
			}
		default:
			return fmt.Errorf("unsupported resource kind: %s", kind)
		}
		if err != nil {
			return fmt.Errorf("applying %s %s: %w", kind, name, err)
		}
	}
	return nil
}

// deleteRouterResources removes the router pod, its service, and all
// resources labeled with app.kubernetes.io/part-of=vxpu-router.
func deleteRouterResources(ctx context.Context, pod string) error {
	client, err := getK8sClient()
	if err != nil {
		return err
	}
	// Delete named pod and service (covers routers created before objects were labelled).
	if err := client.clientset.CoreV1().Pods(client.namespace).Delete(ctx, pod, metav1.DeleteOptions{}); err == nil {
		fmt.Printf("pod %q deleted\n", pod)
	} else if !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting pod %s: %w", pod, err)
	}
	if err := client.clientset.CoreV1().Services(client.namespace).Delete(ctx, pod, metav1.DeleteOptions{}); err == nil {
		fmt.Printf("service %q deleted\n", pod)
	} else if !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting service %s: %w", pod, err)
	}

	partOfSelector := "app.kubernetes.io/part-of=vxpu-router"
	listOpts := metav1.ListOptions{LabelSelector: partOfSelector}
	delOpts := metav1.DeleteOptions{}

	if err := client.clientset.CoreV1().Pods(client.namespace).DeleteCollection(ctx, delOpts, listOpts); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting router pods: %w", err)
	}
	if err := client.clientset.CoreV1().ServiceAccounts(client.namespace).DeleteCollection(ctx, delOpts, listOpts); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting router serviceaccounts: %w", err)
	}
	if err := client.clientset.RbacV1().Roles(client.namespace).DeleteCollection(ctx, delOpts, listOpts); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting router roles: %w", err)
	}
	if err := client.clientset.RbacV1().RoleBindings(client.namespace).DeleteCollection(ctx, delOpts, listOpts); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("deleting router rolebindings: %w", err)
	}

	// Service does not support DeleteCollection; list and delete each matching service.
	svcs, err := client.clientset.CoreV1().Services(client.namespace).List(ctx, listOpts)
	if err != nil {
		return fmt.Errorf("listing router services: %w", err)
	}
	for _, s := range svcs.Items {
		if s.Name != pod {
			if err := client.clientset.CoreV1().Services(client.namespace).Delete(ctx, s.Name, delOpts); err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("deleting router service %s: %w", s.Name, err)
			}
			fmt.Printf("service %q deleted\n", s.Name)
		}
	}

	waitCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if err := wait.PollUntilContextCancel(waitCtx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		remainingPods, err := client.clientset.CoreV1().Pods(client.namespace).List(ctx, listOpts)
		if err != nil {
			return false, err
		}
		if len(remainingPods.Items) > 0 {
			return false, nil
		}
		_, err = client.clientset.CoreV1().Pods(client.namespace).Get(ctx, pod, metav1.GetOptions{})
		if err == nil {
			return false, nil
		}
		if !apierrors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}); err != nil {
		return fmt.Errorf("waiting for router pods to be deleted: %w", err)
	}
	return nil
}

// portForward tunnels an ephemeral local port to port 50051 of the router pod.
func portForward(ctx context.Context, pod string) (string, func(), error) {
	client, err := getK8sClient()
	if err != nil {
		return "", nil, err
	}
	return portForwardPod(ctx, client, client.namespace, pod, 50051)
}

// portForwardService tunnels an ephemeral local port to a Service port by finding
// a ready running pod backing the Service and mapping the service port to the container port.
func portForwardService(ctx context.Context, namespace, service string, port int) (string, func(), error) {
	client, err := getK8sClient()
	if err != nil {
		return "", nil, err
	}
	svc, err := client.clientset.CoreV1().Services(namespace).Get(ctx, service, metav1.GetOptions{})
	if err != nil {
		return "", nil, fmt.Errorf("getting service %s/%s: %w", namespace, service, err)
	}
	if len(svc.Spec.Selector) == 0 {
		return "", nil, fmt.Errorf("service %s/%s has no selector", namespace, service)
	}
	var targetPort intstr.IntOrString
	var foundPort bool
	for _, sp := range svc.Spec.Ports {
		if int(sp.Port) == port {
			targetPort = sp.TargetPort
			foundPort = true
			break
		}
	}
	if !foundPort {
		return "", nil, fmt.Errorf("service %s/%s has no port %d", namespace, service, port)
	}

	selector := labels.SelectorFromSet(svc.Spec.Selector).String()
	var targetPod *corev1.Pod
	waitCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err = wait.PollUntilContextCancel(waitCtx, time.Second, true, func(ctx context.Context) (bool, error) {
		pods, err := client.clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
		if err != nil {
			return false, err
		}
		// Prefer a Ready running pod, as kubectl does.
		for i := range pods.Items {
			p := &pods.Items[i]
			if p.Status.Phase == corev1.PodRunning && p.DeletionTimestamp == nil {
				for _, cond := range p.Status.Conditions {
					if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
						targetPod = p
						return true, nil
					}
				}
			}
		}
		// Fall back to any running pod.
		for i := range pods.Items {
			p := &pods.Items[i]
			if p.Status.Phase == corev1.PodRunning && p.DeletionTimestamp == nil {
				targetPod = p
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil || targetPod == nil {
		return "", nil, fmt.Errorf("waiting for pod for service %s/%s: %w", namespace, service, err)
	}

	var remotePort int
	if targetPort.Type == intstr.Int && targetPort.IntVal != 0 {
		remotePort = int(targetPort.IntVal)
	} else if targetPort.Type == intstr.String && targetPort.StrVal != "" {
		for _, container := range targetPod.Spec.Containers {
			for _, cp := range container.Ports {
				if cp.Name == targetPort.StrVal {
					remotePort = int(cp.ContainerPort)
					break
				}
			}
			if remotePort != 0 {
				break
			}
		}
		if remotePort == 0 {
			return "", nil, fmt.Errorf("port %s not found on pod %s", targetPort.StrVal, targetPod.Name)
		}
	} else {
		remotePort = port
	}

	return portForwardPod(ctx, client, namespace, targetPod.Name, remotePort)
}

// portForwardPod establishes a SPDY port-forward connection to the given pod on remotePort.
func portForwardPod(ctx context.Context, client *k8sClient, namespace, pod string, remotePort int) (string, func(), error) {
	if client.config == nil {
		return "", nil, errors.New("cannot port-forward without rest.Config")
	}
	transport, upgrader, err := spdy.RoundTripperFor(client.config)
	if err != nil {
		return "", nil, fmt.Errorf("creating spdy roundtripper: %w", err)
	}

	reqURL := client.clientset.CoreV1().RESTClient().Post().
		Resource("pods").
		Namespace(namespace).
		Name(pod).
		SubResource("portforward").
		URL()

	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, http.MethodPost, reqURL)

	stopChan := make(chan struct{})
	readyChan := make(chan struct{})

	ports := []string{fmt.Sprintf("0:%d", remotePort)}

	forwarder, err := portforward.NewOnAddresses(dialer, []string{"127.0.0.1"}, ports, stopChan, readyChan, io.Discard, os.Stderr)
	if err != nil {
		return "", nil, fmt.Errorf("creating port forwarder: %w", err)
	}

	errChan := make(chan error, 1)
	go func() {
		errChan <- forwarder.ForwardPorts()
	}()

	select {
	case <-readyChan:
	case err := <-errChan:
		close(stopChan)
		return "", nil, fmt.Errorf("port-forward to %s failed: %w", pod, err)
	case <-ctx.Done():
		close(stopChan)
		return "", nil, ctx.Err()
	}

	forwardedPorts, err := forwarder.GetPorts()
	if err != nil {
		close(stopChan)
		return "", nil, fmt.Errorf("getting forwarded ports: %w", err)
	}
	if len(forwardedPorts) == 0 {
		close(stopChan)
		return "", nil, fmt.Errorf("no ports forwarded to %s", pod)
	}

	localPort := forwardedPorts[0].Local
	addr := fmt.Sprintf("127.0.0.1:%d", localPort)

	var once sync.Once
	stop := func() {
		once.Do(func() {
			close(stopChan)
		})
	}

	go func() {
		select {
		case <-ctx.Done():
			stop()
		case <-stopChan:
		}
	}()

	return addr, stop, nil
}
