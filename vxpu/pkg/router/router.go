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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/blobserver"
	"github.com/gke-labs/generation-ai/vxpu/pkg/collections"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

type Server struct {
	pb.UnimplementedEngineServer
	pb.UnimplementedExecutorServer
	clientset   kubernetes.Interface
	namespace   string
	imageName   string
	accelerator string
	blobServer  *blobserver.Server
	// weightStore, when set, holds the weights (modelstore in a
	// bucket) and executors read it directly; otherwise the router
	// caches weights itself and serves them through blobServer.
	weightStore *weightstore.Client
	// executorPort is the executor pods' gRPC port (overridable for tests).
	executorPort int
	// agentPort is the agent pods' gRPC port (overridable for tests).
	agentPort int
	// agentDialer customizes gRPC dialing to the agent for tests.
	agentDialer func(addr string) (*grpc.ClientConn, error)

	mu sync.RWMutex
	// peerToModel maps each client peer address (IP:port) to its active model key.
	peerToModel map[string]string

	// conns holds pooled connections to executors, keyed by address (host:port).
	conns map[string]*grpc.ClientConn

	// externalExecutors are executor addresses on machines the router
	// does not manage (#357: GPU VMs, Macs); models loaded by name go
	// there when any are configured (external.go).
	externalExecutors []string

	// The fields below hold catalog state for models served by name (catalog.go), guarded by mu.

	// externalModel maps each external executor address to the model key it holds.
	externalModel map[string]string
	// modelEndpoint maps each model key to its external executor address.
	modelEndpoint map[string]string
	// loadedTags maps each compiled artifact tag to the model key loaded from it.
	loadedTags map[string]string
	// hints maps each model's statistics key to its registered plan, encoded as one comparable string by hintsKey, as last stored or read.
	hints map[string]string
	// keyModel maps each model key to the named model reference it serves.
	keyModel map[string]modelRef
	// executors maps each executor's endpoint to what it runs on: its probe and the catalog machine the probe matched.
	executors map[string]executorInfo
	// expertHolders maps each external executor address to the address of the engine whose experts it serves.
	expertHolders map[string]string
	// plans maps each model's statistics key to the plan chosen for its last load.
	plans map[string]modelPlan
	// goldens maps each model's statistics key to its registered golden reference file.
	goldens map[string]*planpb.GoldenFile
	// vocabs maps each model key to its cached vocabulary response.
	vocabs map[string]*pb.GetVocabResponse
	// stats records statistics for models served by name, or is nil when modelstore is not configured.
	stats *statsStore
	// externalModelstoreURL is modelstore's address for external executors
	// (VMs outside the cluster, which cannot resolve its Service).
	externalModelstoreURL string
	// flights deduplicates concurrent model registrations and model loads.
	flights *collections.Deduplicator[string, string]
	// models records the verification standing, refusal reason, and registered plan and golden facts for each modelKey.
	models map[string]modelState
	// compileResources specifies the resource requests and limits for compilation pods.
	compileResources corev1.ResourceRequirements
	// compilePoll is how often RegisterModel polls a compilation job.
	compilePoll time.Duration
	// compileUnschedulable is how long a compile pod may stay unschedulable
	// before RegisterModel gives up and says why.
	compileUnschedulable time.Duration
	// reconciler drives model loading through reconciliation.
	reconciler *ModelReconciler
}

// errNoExecutor: no executor pod exists for the model on this router
// and the caller may not create one (only LoadModel, which knows the
// model's size, creates pods).
var errNoExecutor = stderrors.New("no executor for this model; call LoadModel")

const (
	annotationWeightBytes = "vxpu.gke-labs.dev/weight-bytes"
	annotationAccelerator = "vxpu.gke-labs.dev/accelerator"
	annotationImage       = "vxpu.gke-labs.dev/image"
)

func NewServer(clientset kubernetes.Interface, namespace, imageName, accelerator string) *Server {
	bs, err := blobserver.NewServer("/tmp/vxpu-cache", 8080)
	if err != nil {
		klog.Errorf("Failed to initialize blobserver: %v", err)
	} else {
		bs.Start()
	}

	s := &Server{
		clientset:            clientset,
		namespace:            namespace,
		imageName:            imageName,
		accelerator:          accelerator,
		peerToModel:          make(map[string]string),
		conns:                make(map[string]*grpc.ClientConn),
		blobServer:           bs,
		executorPort:         50051,
		agentPort:            50052,
		externalModel:        make(map[string]string),
		modelEndpoint:        make(map[string]string),
		loadedTags:           make(map[string]string),
		keyModel:             make(map[string]modelRef),
		models:               make(map[string]modelState),
		vocabs:               make(map[string]*pb.GetVocabResponse),
		flights:              collections.NewDeduplicator[string, string](),
		compileResources:     defaultCompileResources(),
		compilePoll:          5 * time.Second,
		compileUnschedulable: 5 * time.Minute,
	}
	s.reconciler = NewModelReconciler(s)
	return s
}

// modelStanding is the verification standing of a model on the router.
type modelStanding int

const (
	standingUnchecked modelStanding = iota
	standingPassed
	standingRefused
)

func (s modelStanding) String() string {
	switch s {
	case standingPassed:
		return "passed"
	case standingRefused:
		return "refused"
	default:
		return "unchecked"
	}
}

// modelState records a model's verification standing, refusal reason, and registered plan and golden facts.
type modelState struct {
	// standing is the verification standing of the model.
	standing modelStanding

	// err is the token mismatch error recorded when standing is standingRefused.
	err error

	// hints is the registered plan key used when the model was loaded.
	hints string

	// planHash is the plan's identity hash used when the model was loaded.
	planHash string

	// golden is the golden file digest used when the model was loaded.
	golden string
}

func (s *Server) modelStanding(modelKey string) modelState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.models[modelKey]
}

func (s *Server) setModelStanding(modelKey string, standing modelStanding, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.models == nil {
		s.models = make(map[string]modelState)
	}
	st := s.models[modelKey]
	st.standing = standing
	st.err = err
	s.models[modelKey] = st
}

// UseWeightStore makes the router keep weights in modelstore instead of
// its own cache; nil keeps the router's cache.
func (s *Server) UseWeightStore(ws *weightstore.Client) {
	s.weightStore = ws
}

// rewriteViaWeightStore makes every file of the manifest present in
// modelstore (which fetches what it lacks from the manifest's sources)
// and points the sources at modelstore's blob URLs. Files are handled
// concurrently; modelstore de-duplicates and serializes the actual
// downloads. The manifest goes through the typed Manifest, whose config
// and tensors are raw JSON, so every byte the router does not touch is
// reproduced verbatim (a generic decode would turn 30.0 into 30, which
// transformers rejects where it expects a float).
func (s *Server) rewriteViaWeightStore(ctx context.Context, manifestJSON string, external bool) (string, error) {
	var manifest pb.Manifest
	if err := json.Unmarshal([]byte(manifestJSON), &manifest); err != nil {
		return "", fmt.Errorf("parsing manifest: %w", err)
	}
	if len(manifest.Files) == 0 {
		return "", fmt.Errorf("manifest has no files")
	}
	type result struct {
		sha, url string
		err      error
	}
	results := make(chan result, len(manifest.Files))
	for sha, file := range manifest.Files {
		go func(sha, src string) {
			url, err := s.weightStore.Ensure(ctx, sha, src)
			results <- result{sha, url, err}
		}(sha, file.Source)
	}
	var errs []string
	for range manifest.Files {
		r := <-results
		if r.err != nil {
			errs = append(errs, r.err.Error())
			continue
		}
		url := r.url
		if external && s.externalModelstoreURL != "" {
			// an executor outside the cluster cannot reach modelstore's
			// Service: the same blob at the address it can reach
			url = s.externalModelstoreURL + strings.TrimPrefix(url, s.weightStore.URL())
		}
		file := manifest.Files[r.sha]
		file.Source = url
		manifest.Files[r.sha] = file
	}
	if len(errs) > 0 {
		return "", fmt.Errorf("weight store: %s", strings.Join(errs, "; "))
	}
	out, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("serializing manifest: %w", err)
	}
	return string(out), nil
}

// executorConn returns a pooled connection to the executor at endpoint.
func (s *Server) executorConn(endpoint string) (*grpc.ClientConn, error) {
	if endpoint == "" {
		return nil, status.Errorf(codes.InvalidArgument, "executor address cannot be empty")
	}
	s.mu.RLock()
	conn, ok := s.conns[endpoint]
	s.mu.RUnlock()
	if ok {
		return conn, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if conn, ok := s.conns[endpoint]; ok {
		return conn, nil
	}
	conn, err := dialExecutor(endpoint)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	if s.conns == nil {
		s.conns = make(map[string]*grpc.ClientConn)
	}
	s.conns[endpoint] = conn
	return conn, nil
}

// executorClient returns an ExecutorClient for the executor serving modelKey,
// reusing the router's pooled connection.
func (s *Server) executorClient(ctx context.Context, modelKey string) (pb.ExecutorClient, error) {
	endpoint, err := s.executorEndpoint(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	conn, err := s.executorConn(endpoint)
	if err != nil {
		return nil, err
	}
	return pb.NewExecutorClient(conn), nil
}

// agentConn returns a pooled connection to the agent on the pod at host.
func (s *Server) agentConn(host string) (*grpc.ClientConn, error) {
	if host == "" {
		return nil, status.Errorf(codes.InvalidArgument, "agent host cannot be empty")
	}
	endpoint := fmt.Sprintf("%s:%d", host, s.agentPort)
	s.mu.RLock()
	conn, ok := s.conns[endpoint]
	s.mu.RUnlock()
	if ok {
		return conn, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if conn, ok := s.conns[endpoint]; ok {
		return conn, nil
	}
	conn, err := s.dialAgent(endpoint)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to agent at %s: %v", endpoint, err)
	}
	if s.conns == nil {
		s.conns = make(map[string]*grpc.ClientConn)
	}
	s.conns[endpoint] = conn
	return conn, nil
}

func (s *Server) dialAgent(endpoint string) (*grpc.ClientConn, error) {
	if s.agentDialer != nil {
		return s.agentDialer(endpoint)
	}
	return grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
}

// executorKeepAliveSeconds returns how long an executor keeps an idle model
// loaded (seconds, default 3600), read from VXPU_EXECUTOR_KEEP_ALIVE.
func executorKeepAliveSeconds() int32 {
	if v := os.Getenv("VXPU_EXECUTOR_KEEP_ALIVE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return int32(n)
		}
	}
	return 3600
}

// startRuntime asks the agent at host to start the executor runtime,
// and returns the published endpoint (host:port) from the process table.
func (s *Server) startRuntime(ctx context.Context, host string) (string, error) {
	conn, err := s.agentConn(host)
	if err != nil {
		return "", err
	}
	client := pb.NewAgentClient(conn)
	keepAlive := executorKeepAliveSeconds()
	resp, err := client.StartProcess(ctx, &pb.StartProcessRequest{
		Id:               "executor",
		Process:          "executor",
		KeepAliveSeconds: &keepAlive,
	})
	if err != nil {
		return "", err
	}
	endpoint := resp.GetStatus().GetEndpoint()
	if endpoint == "" {
		return "", fmt.Errorf("agent at %s started serving process %q but published an empty endpoint", host, "executor")
	}
	return endpoint, nil
}

// RemoveExecutor closes the connection to the executor at endpoint and drops
// what the router knows about it.
func (s *Server) RemoveExecutor(ctx context.Context, endpoint string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeExecutorLocked(ctx, endpoint)
}

func (s *Server) removeExecutorLocked(ctx context.Context, endpoint string) {
	if conn, ok := s.conns[endpoint]; ok {
		if err := conn.Close(); err != nil {
			klog.FromContext(ctx).Error(err, "closing executor connection", "endpoint", endpoint)
		}
		delete(s.conns, endpoint)
	}
	host, _, err := net.SplitHostPort(endpoint)
	if err != nil {
		host = endpoint
	}
	agentEndpoint := fmt.Sprintf("%s:%d", host, s.agentPort)
	if aConn, ok := s.conns[agentEndpoint]; ok {
		if err := aConn.Close(); err != nil {
			klog.FromContext(ctx).Error(err, "closing agent connection", "endpoint", agentEndpoint)
		}
		delete(s.conns, agentEndpoint)
	}
	delete(s.executors, endpoint)
	delete(s.expertHolders, endpoint)
}

// Close closes all pooled connections to executors and stops background reconciler workers.
func (s *Server) Close() error {
	if s.reconciler != nil {
		s.reconciler.Stop()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var errs []error
	for endpoint, conn := range s.conns {
		if err := conn.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing executor connection to %s: %w", endpoint, err))
		}
		delete(s.conns, endpoint)
	}
	return stderrors.Join(errs...)
}

// Reconciler returns the model reconciler managing loading for this server.
func (s *Server) Reconciler() *ModelReconciler {
	return s.reconciler
}

// InstanceStatus reports the current reconciliation status of an instance key.
func (s *Server) InstanceStatus(key string) InstanceStatus {
	if s.reconciler == nil {
		return InstanceStatus{Phase: InstancePhaseNotLoaded}
	}
	return s.reconciler.Status(key)
}

// ModelStatus reports the current reconciliation status of modelKey.
func (s *Server) ModelStatus(modelKey string) InstanceStatus {
	return s.InstanceStatus(modelKey)
}

// splitSessionID parses the routed "modelKey:backendSessionID" form.
func splitSessionID(sessionID string) (string, string, error) {
	parts := strings.SplitN(sessionID, ":", 2)
	if len(parts) != 2 {
		return "", "", status.Errorf(codes.InvalidArgument, "invalid routed session ID format; expected 'modelKey:sessionID'")
	}
	return parts[0], parts[1], nil
}

func getPeerKey(ctx context.Context) string {
	if p, ok := peer.FromContext(ctx); ok {
		return p.Addr.String()
	}
	return "unknown"
}

// placementAnnotations records what a pod was sized for, so a later
// LoadModel can tell whether an existing pod fits the model and runs the expected image.
func placementAnnotations(p Placement, image string) map[string]string {
	m := map[string]string{
		annotationWeightBytes: strconv.FormatInt(p.WeightBytes, 10),
		annotationAccelerator: p.Accelerator,
	}
	if image != "" {
		m[annotationImage] = image
	}
	return m
}

// placementMatches reports whether an existing pod was created for
// this placement and image. Pods without annotations (created before they were
// recorded) are kept.
func placementMatches(pod *corev1.Pod, p Placement, image string) bool {
	weight, ok := pod.Annotations[annotationWeightBytes]
	if !ok {
		return true
	}
	if weight != strconv.FormatInt(p.WeightBytes, 10) ||
		pod.Annotations[annotationAccelerator] != p.Accelerator {
		return false
	}
	if image != "" {
		recordedImage, ok := pod.Annotations[annotationImage]
		if !ok || recordedImage != image {
			return false
		}
	}
	return true
}

// waitDeleted blocks until the pod is gone (or ctx ends).
func (s *Server) waitDeleted(ctx context.Context, podName string) error {
	for i := 0; i < 60; i++ {
		_, err := s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
	return fmt.Errorf("pod %s still exists after deletion", podName)
}

// getOrStartPod returns the IP of the executor serving modelID. With a
// placement (LoadModel) it creates the pod if absent and recreates it
// if the existing pod was sized for a different placement or image; without one
// (NewSession/Chat) it only finds an existing pod and otherwise returns
// errNoExecutor.
func (s *Server) getOrStartPod(ctx context.Context, modelID string, want *Placement) (string, error) {
	log := klog.FromContext(ctx)
	if modelID == "" {
		return "", fmt.Errorf("modelID cannot be empty")
	}

	if s.imageName == "" {
		return "", fmt.Errorf("no executor image configured on router")
	}

	podName := "vxpu-executor-" + modelID

	// Check if pod already exists
	pod, err := s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if err == nil {
		log.Info("Found existing pod", "pod", podName, "phase", pod.Status.Phase)
		recreate := ""
		switch {
		case want != nil && !placementMatches(pod, *want, s.imageName):
			recreate = "existing pod was sized for a different placement or image"
		case pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodUnknown:
			recreate = "existing pod is in bad state"
		case pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "":
			return pod.Status.PodIP, nil
		}
		if recreate != "" {
			log.Info("Deleting executor pod", "pod", podName, "phase", pod.Status.Phase, "reason", recreate)
			if pod.Status.PodIP != "" {
				s.RemoveExecutor(ctx, fmt.Sprintf("%s:%d", pod.Status.PodIP, s.executorPort))
			}
			if err := s.clientset.CoreV1().Pods(s.namespace).Delete(ctx, podName, metav1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
				return "", fmt.Errorf("failed to delete pod: %w", err)
			}
			if err := s.waitDeleted(ctx, podName); err != nil {
				return "", err
			}
		}
	} else if !errors.IsNotFound(err) {
		return "", fmt.Errorf("failed to get pod: %w", err)
	}

	// Create a new Pod if it didn't exist or was deleted
	_, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if errors.IsNotFound(err) && want == nil {
		return "", errNoExecutor
	}
	if errors.IsNotFound(err) {
		placement := *want
		log.Info("Creating a new executor pod", "pod", podName, "image", s.imageName,
			"accelerator", placement.Accelerator, "weightBytes", placement.WeightBytes,
			"cpu", placement.CPU.String(), "memory", placement.Memory.String(),
			"reason", placement.Reason)

		// Build pod spec from the placement decision.
		var nodeSelector map[string]string
		var resources corev1.ResourceRequirements
		var containerEnv []corev1.EnvVar

		if placement.IsAccelerator() {
			nodeSelector = map[string]string{
				"cloud.google.com/gke-accelerator": placement.Accelerator,
			}
			resources = corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:              placement.CPU,
					corev1.ResourceMemory:           placement.Memory,
					corev1.ResourceEphemeralStorage: resource.MustParse("60Gi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceMemory:                 placement.MemoryLimit,
					corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
				},
			}
			containerEnv = []corev1.EnvVar{
				{
					Name:  "LD_LIBRARY_PATH",
					Value: "/usr/local/nvidia/lib64",
				},
			}
		} else {
			// CPU executor: weights are resident in RAM, so memory is
			// sized from the manifest; no CPU limit so the bandwidth-
			// bound decode loop can use what the node has spare.
			resources = corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    placement.CPU,
					corev1.ResourceMemory: placement.Memory,
				},
				Limits: corev1.ResourceList{
					corev1.ResourceMemory: placement.MemoryLimit,
				},
			}
			containerEnv = []corev1.EnvVar{
				{
					Name:  "OMP_NUM_THREADS",
					Value: fmt.Sprintf("%d", placement.Threads),
				},
			}
		}
		containerEnv = append(containerEnv, corev1.EnvVar{
			Name: "POD_IP",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		})

		newPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: s.namespace,
				Labels: map[string]string{
					"app":      "vxpu-executor",
					"model-id": modelID,
				},
				Annotations: placementAnnotations(placement, s.imageName),
			},
			Spec: corev1.PodSpec{
				NodeSelector: nodeSelector,
				Volumes: []corev1.Volume{
					{
						Name: "work",
						VolumeSource: corev1.VolumeSource{
							EmptyDir: &corev1.EmptyDirVolumeSource{
								SizeLimit: func() *resource.Quantity {
									q := resource.MustParse("80Gi")
									return &q
								}(),
							},
						},
					},
				},
				Containers: []corev1.Container{
					{
						Name:  "executor",
						Image: s.imageName,
						Args:  []string{"--host=$(POD_IP)"},
						Ports: []corev1.ContainerPort{
							{
								Name:          "executor",
								ContainerPort: int32(s.executorPort),
							},
							{
								Name:          "agent",
								ContainerPort: int32(s.agentPort),
							},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{
									Port: intstr.FromInt32(int32(s.agentPort)),
								},
							},
							InitialDelaySeconds: 1,
							PeriodSeconds:       2,
						},
						VolumeMounts: []corev1.VolumeMount{
							{
								Name:      "work",
								MountPath: "/tmp/vxpu",
							},
						},
						Env:       containerEnv,
						Resources: resources,
					},
				},
				RestartPolicy: corev1.RestartPolicyNever,
			},
		}

		_, err = s.clientset.CoreV1().Pods(s.namespace).Create(ctx, newPod, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("failed to create pod: %w", err)
		}
	}

	// Wait for the Pod to be running and have an IP
	log.Info("Waiting for pod to be running and get an IP", "pod", podName)
	var podIP string
	for i := 0; i < 300; i++ {
		pod, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			log.Error(err, "failed to get pod status, retrying", "pod", podName)
		} else {
			if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
				podIP = pod.Status.PodIP
				break
			}
			if pod.Status.Phase == corev1.PodFailed {
				return "", fmt.Errorf("pod failed to start")
			}
		}
		time.Sleep(2 * time.Second)
	}

	if podIP == "" {
		return "", fmt.Errorf("timed out waiting for pod to get an IP")
	}

	return podIP, nil
}

func (s *Server) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	for _, owned := range routerOwnedSwitches {
		if _, set := req.Plan[owned]; set {
			return nil, status.Errorf(codes.InvalidArgument, "%s is the router's to set (it wires the executors it starts), not a client's to set", owned)
		}
	}
	if len(req.Plan) > 0 { // sessions find their executor by the artifact alone (modelKey), so one plan per artifact
		return nil, status.Error(codes.InvalidArgument, "a plan comes with a model loaded by name (vxpu register --plan), not with an artifact")
	}
	if req.PlanFile != nil {
		if err := kernelplan.EnsurePlanIdentity(req.PlanFile); err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "plan identity: %v", err)
		}
	}
	modelKey := artifactKey(req.ManifestJson)
	peerKey := getPeerKey(ctx)
	if peerKey != "unknown" {
		s.mu.Lock()
		s.peerToModel[peerKey] = modelKey
		s.mu.Unlock()
	}
	if _, err := s.loadArtifact(ctx, req, false, nil); err != nil {
		return nil, err
	}
	return &pb.LoadModelResponse{Success: true}, nil
}

// artifactKey identifies an artifact by its manifest (as shipped).
func artifactKey(manifestJSON string) string {
	h := sha256.Sum256([]byte(manifestJSON))
	return hex.EncodeToString(h[:16])
}

// isExecutorReady reports whether the executor at endpoint is ready to serve.
func (s *Server) isExecutorReady(ctx context.Context, endpoint string) (bool, error) {
	conn, err := s.executorConn(endpoint)
	if err != nil {
		return false, err
	}
	client := pb.NewExecutorClient(conn)
	loadResp, err := client.GetLoad(ctx, &pb.GetLoadRequest{})
	if err != nil {
		return false, err
	}
	return loadResp.State == pb.LoadState_LOAD_STATE_READY, nil
}

// checkServingModel checks whether a model in standingPassed is already ready on its executor.
// It returns (true, nil) if the model is ready and reload can be skipped.
// If the executor is gone or no longer holds the model, it forgets the model and returns (false, nil).
// If a transport error or context error occurs while communicating with the executor or apiserver,
// it returns that error without forgetting the model.
func (s *Server) checkServingModel(ctx context.Context, modelKey string) (bool, error) {
	if st := s.modelStanding(modelKey); st.standing != standingPassed {
		return false, nil
	}
	ep, err := s.servingEndpoint(ctx, modelKey)
	if stderrors.Is(err, errNoExecutor) {
		s.forgetModel(modelKey, "")
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if ep == "" {
		return false, nil
	}
	ready, err := s.isExecutorReady(ctx, ep)
	if err != nil {
		// Transport or context error: leave standing unchanged and do not forget the model.
		return false, err
	}
	if ready {
		return true, nil
	}
	// The executor is reachable but no longer holds the model (e.g. idle keep-alive eviction).
	s.forgetModel(modelKey, ep)
	return false, nil
}

// loadArtifact makes the artifact's weights reachable (modelstore, or
// the router's own cache), places an executor for it and ships it there.
// On an external executor when external is set and the router has any
// (#357), else on an executor pod the router starts. Returns the model
// key sessions are routed by.
//
// planFor, when set, chooses the plan sent with the load, knowing what
// the executor runs on (nil: it did not say) and given the manifest as
// sent, its weights' sources rewritten to where executors read them;
// otherwise req's plan is sent.
func (s *Server) loadArtifact(ctx context.Context, req *pb.LoadModelRequest, external bool,
	planFor func(context.Context, *executorInfo, string) map[string]string) (string, error) {
	modelKey := artifactKey(req.ManifestJson)

	if serving, err := s.checkServingModel(ctx, modelKey); err != nil {
		if ctx.Err() != nil {
			return "", status.FromContextError(ctx.Err()).Err()
		}
		return "", err
	} else if serving {
		st := s.modelStanding(modelKey)
		incomingPlanHash := ""
		if req.PlanFile != nil {
			incomingPlanHash = req.PlanFile.PlanHash
		}
		if incomingPlanHash != st.planHash {
			return "", status.Errorf(codes.FailedPrecondition, "model %s is already serving under a different plan; reloading with a different plan is not supported", modelKey)
		}
		klog.FromContext(ctx).Info("Model already passed verification and is serving; skipping reload", "model_key", modelKey)
		return modelKey, nil
	}

	// Loads of one model do not overlap.
	loadedKey, err := s.flights.Do(ctx, "load-artifact/"+modelKey, func() (string, bool, error) {
		if serving, err := s.checkServingModel(ctx, modelKey); err != nil {
			return "", false, err
		} else if serving {
			st := s.modelStanding(modelKey)
			incomingPlanHash := ""
			if req.PlanFile != nil {
				incomingPlanHash = req.PlanFile.PlanHash
			}
			if incomingPlanHash != st.planHash {
				return "", false, status.Errorf(codes.FailedPrecondition, "model %s is already serving under a different plan; reloading with a different plan is not supported", modelKey)
			}
			return modelKey, true, nil
		}
		k, err := s.loadArtifactInternal(ctx, req, external, planFor)
		if err != nil {
			// Only a golden check mismatch that marks the model refused is a
			// definitive verdict to share across concurrent callers; other errors
			// (including other FailedPrecondition errors from the executor or
			// transport failures) allow waiting callers to make their own attempt.
			isVerdict := s.modelStanding(modelKey).standing == standingRefused
			return "", isVerdict, err
		}
		return k, true, nil
	})
	if err != nil {
		if stderrors.Is(err, context.Canceled) || stderrors.Is(err, context.DeadlineExceeded) {
			return "", status.FromContextError(err).Err()
		}
		return "", err
	}
	return loadedKey, nil
}

func (s *Server) loadArtifactInternal(ctx context.Context, req *pb.LoadModelRequest, external bool,
	planFor func(context.Context, *executorInfo, string) map[string]string) (string, error) {
	log := klog.FromContext(ctx)
	modelKey := artifactKey(req.ManifestJson)

	if serving, err := s.checkServingModel(ctx, modelKey); err != nil {
		if ctx.Err() != nil {
			return "", status.FromContextError(ctx.Err()).Err()
		}
		return "", err
	} else if serving {
		return modelKey, nil
	}

	log.Info("Loading artifact", "model_key", modelKey, "external", external, "plan", planSwitches(req.Plan))

	if s.blobServer == nil {
		return "", status.Errorf(codes.Internal, "blobserver is not initialized")
	}

	// Caching tens of GB outlives impatient clients: a dropped
	// connection must not abort a download that the retry would only
	// restart from zero.
	var rewrittenManifestJSON string
	var err error
	if s.weightStore != nil {
		rewrittenManifestJSON, err = s.rewriteViaWeightStore(context.WithoutCancel(ctx), req.ManifestJson,
			external && len(s.externalExecutors) > 0)
		if err != nil {
			return "", status.Errorf(codes.Internal, "weight store: %v", err)
		}
	} else {
		rewrittenManifestJSON, err = s.blobServer.CacheAndRewriteManifest(
			context.WithoutCancel(ctx), req.ManifestJson, getRouterIP())
		if err != nil {
			return "", status.Errorf(codes.Internal, "failed to cache and rewrite manifest: %v", err)
		}
	}

	// Prepare rewritten request
	reqCopy := &pb.LoadModelRequest{
		ManifestJson: rewrittenManifestJSON,
		BindingJson:  req.BindingJson,
		PrefillGraph: req.PrefillGraph,
		DecodeGraph:  req.DecodeGraph,
		Plan:         req.Plan,
		PlanFile:     req.PlanFile,
	}

	var endpoint string
	if external && len(s.externalExecutors) > 0 {
		endpoint = s.pickExternalExecutor(modelKey)
		log.Info("Placing on an external executor", "model_key", modelKey, "address", endpoint)
		s.mu.RLock()
		previous := s.externalModel[endpoint]
		s.mu.RUnlock()
		if previous != "" && previous != modelKey {
			log.Info("Evicting previous model from external executor", "previous", previous, "new_model", modelKey, "address", endpoint)
			s.forgetModel(previous, endpoint)
		}
	} else {
		bytes, err := weightBytes(req.ManifestJson)
		if err != nil {
			return "", status.Errorf(codes.InvalidArgument, "invalid manifest: %v", err)
		}
		placement := PlanPlacement(s.accelerator, bytes)
		log.Info("Planned executor placement", "model_key", modelKey,
			"weightBytes", bytes, "accelerator", placement.Accelerator,
			"cpu", placement.CPU.String(), "memory", placement.Memory.String(),
			"reason", placement.Reason)
		podIP, err := s.getOrStartPod(ctx, modelKey, &placement)
		if err != nil {
			return "", status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
		}
		ep, err := s.startRuntime(ctx, podIP)
		if err != nil {
			if ctx.Err() != nil {
				return "", status.FromContextError(ctx.Err()).Err()
			}
			return "", status.Errorf(codes.Internal, "failed to start runtime: %v", err)
		}
		endpoint = ep
		s.mu.Lock()
		delete(s.executors, endpoint)
		s.mu.Unlock()
	}

	conn, err := s.executorConn(endpoint)
	if err != nil {
		return "", status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}

	client := pb.NewExecutorClient(conn)
	info, err := s.waitForExecutor(ctx, client, endpoint)
	if err != nil {
		return "", err
	}
	if planFor != nil {
		reqCopy.Plan = planFor(ctx, info, reqCopy.ManifestJson)
	}

	var loadErr error
	for i := 0; i < 30; i++ {
		loadCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, loadErr = client.LoadModel(loadCtx, reqCopy)
		cancel()
		if loadErr == nil || status.Code(loadErr) == codes.InvalidArgument || ctx.Err() != nil { // a refused plan stays refused
			break
		}
		log.Info("Waiting for executor gRPC server to accept LoadModel", "attempt", i, "error", loadErr.Error())
		time.Sleep(2 * time.Second)
	}

	if loadErr != nil {
		return "", loadErr
	}

	modelName := modelFromManifest(req.ManifestJson)
	golden, err := s.getGolden(ctx, modelName)
	if err != nil {
		return "", status.Errorf(codes.Unavailable, "reading golden file for %s: %v", modelName, err)
	}
	if golden != nil {
		if err := s.verifyGolden(ctx, client, modelKey, golden); err != nil {
			if ctx.Err() != nil {
				return "", status.FromContextError(ctx.Err()).Err()
			}
			var mismatch *kernelplan.TokenMismatchError
			if stderrors.As(err, &mismatch) {
				log.Error(err, "Golden check failed", "model", modelName)
				s.setModelStanding(modelKey, standingRefused, err)
				return "", status.Errorf(codes.FailedPrecondition, "golden check failed: %v", err)
			}
			log.Error(err, "Golden check execution error", "model", modelName)
			return "", status.Errorf(codes.Unavailable, "running golden check: %v", err)
		}
		log.Info("Golden check passed", "model", modelName)
	} else {
		log.Info("No golden file for model; skipping verification", "model", modelName)
	}

	s.mu.Lock()
	if s.models == nil {
		s.models = make(map[string]modelState)
	}
	st := s.models[modelKey]
	st.standing = standingPassed
	st.err = nil
	if reqCopy.PlanFile != nil {
		st.planHash = reqCopy.PlanFile.PlanHash
	} else {
		st.planHash = ""
	}
	s.models[modelKey] = st
	s.mu.Unlock()

	if external && len(s.externalExecutors) > 0 {
		s.placedExternally(modelKey, endpoint)
	}
	log.Info("Successfully loaded model on executor", "model_key", modelKey, "endpoint", endpoint)
	return modelKey, nil
}

// modelFromManifest extracts the model name or repository from a manifest JSON.
func modelFromManifest(manifestJSON string) string {
	var m struct {
		Source struct {
			Repo string `json:"repo"`
		} `json:"source"`
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(manifestJSON), &m); err != nil {
		return ""
	}
	if m.Source.Repo != "" {
		return m.Source.Repo
	}
	return m.Model
}

// waitForReady waits until client.GetLoad reports LOAD_STATE_READY or failure.
func (s *Server) waitForReady(ctx context.Context, client pb.ExecutorClient) error {
	log := klog.FromContext(ctx)
	deadline := time.Now().Add(10 * time.Minute)
	for time.Now().Before(deadline) {
		resp, err := client.GetLoad(ctx, &pb.GetLoadRequest{})
		if err != nil {
			log.V(2).Info("Waiting for executor GetLoad", "error", err)
		} else {
			switch resp.State {
			case pb.LoadState_LOAD_STATE_READY:
				return nil
			case pb.LoadState_LOAD_STATE_FAILED:
				return fmt.Errorf("model load failed on executor: %s", resp.Error)
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	return fmt.Errorf("timed out waiting for executor to become ready")
}

// verifyGolden sends each golden prompt through the executor's ordinary serving
// call (NewSession and streaming Execute), collects the produced tokens, and compares
// them against golden. A mismatch returns an error naming the prompt and position.
func (s *Server) verifyGolden(ctx context.Context, client pb.ExecutorClient, modelKey string, golden *planpb.GoldenFile) error {
	if err := s.waitForReady(ctx, client); err != nil {
		return err
	}
	var produced [][]int32
	for i, prompt := range golden.Prompts {
		sessResp, err := client.NewSession(ctx, &pb.NewSessionRequest{})
		if err != nil {
			return fmt.Errorf("golden prompt %d: new session: %w", i, err)
		}
		n := int32(golden.TokensPerPrompt)
		if n <= 0 {
			n = 32
		}
		doSample := false
		temp := float32(0)
		stream, err := client.Execute(ctx, &pb.ExecuteRequest{
			Request: &pb.GenerateRequest{
				SessionId:    sessResp.GetSessionId(),
				InputIds:     prompt.Ids,
				MaxNewTokens: &n,
				DoSample:     &doSample,
				Temperature:  &temp,
			},
		})
		if err != nil {
			return fmt.Errorf("golden prompt %d: execute: %w", i, err)
		}
		var tokens []int32
		for {
			resp, err := stream.Recv()
			if err != nil {
				if stderrors.Is(err, io.EOF) {
					break
				}
				return fmt.Errorf("golden prompt %d: stream: %w", i, err)
			}
			if r := resp.GetResponse(); r != nil {
				tokens = append(tokens, r.GetTokenIds()...)
				if r.GetDone() {
					break
				}
			}
		}
		produced = append(produced, tokens)
	}
	return kernelplan.CompareTokens(golden, produced)
}

func (s *Server) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	log := klog.FromContext(ctx)
	var modelKey string
	if req.Model != "" {
		key, err := s.loadByName(ctx, req.Model, req.Revision)
		if err != nil {
			return nil, err
		}
		modelKey = key
	} else {
		peerKey := getPeerKey(ctx)
		s.mu.RLock()
		key, exists := s.peerToModel[peerKey]
		s.mu.RUnlock()
		if !exists {
			return nil, status.Errorf(codes.FailedPrecondition, "no model has been loaded for this connection")
		}
		modelKey = key
	}

	log.Info("NewSession request for", "model_key", modelKey, "model", req.Model)

	endpoint, err := s.executorEndpoint(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	conn, err := s.executorConn(endpoint)
	if err != nil {
		return nil, err
	}

	client := pb.NewExecutorClient(conn)
	sessResp, err := client.NewSession(ctx, &pb.NewSessionRequest{})
	if err != nil {
		loadResp, loadErr := client.GetLoad(ctx, &pb.GetLoadRequest{})
		if loadErr != nil {
			log.Info("Failed to get load state after NewSession failure", "error", loadErr.Error())
			return nil, err
		}
		switch loadResp.State {
		case pb.LoadState_LOAD_STATE_NONE:
			// The executor evicted the model (idle keep-alive) or restarted:
			// forget where it was, so the caller's retry loads it again.
			s.forgetModel(modelKey, endpoint)
			if req.Model != "" {
				return nil, status.Errorf(codes.FailedPrecondition, "%s is being reloaded (the executor no longer holds it)", req.Model)
			}
			return nil, status.Errorf(codes.FailedPrecondition, "call LoadModel first")
		case pb.LoadState_LOAD_STATE_LOADING:
			if req.Model != "" {
				return nil, status.Errorf(codes.FailedPrecondition, "%s is still loading", req.Model)
			}
			return nil, status.Errorf(codes.FailedPrecondition, "model still loading")
		case pb.LoadState_LOAD_STATE_FAILED:
			// The executor's asynchronous load failed (it keeps answering so
			// until it is asked to load again): forget the placement, so the
			// next request loads the model again.
			s.forgetModel(modelKey, endpoint)
			if req.Model != "" {
				return nil, status.Errorf(codes.Unavailable, "%s failed to load on its executor; the next request loads it again: %s",
					req.Model, loadResp.Error)
			}
			return nil, status.Errorf(codes.Internal, "model load failed: %s", loadResp.Error)
		case pb.LoadState_LOAD_STATE_READY:
			// The model finished loading between NewSession and GetLoad: retry once.
			if sessResp, err = client.NewSession(ctx, &pb.NewSessionRequest{}); err != nil {
				return nil, err
			}
		default:
			return nil, err
		}
	}

	wrappedSessionID := fmt.Sprintf("%s:%s", modelKey, sessResp.SessionId)
	log.Info("Successfully created routed session", "wrapped_session_id", wrappedSessionID)
	return &pb.NewSessionResponse{
		SessionId: wrappedSessionID,
	}, nil
}

// proxyChat forwards a ChatRequest to the executor via ExecuteChat, sampling telemetry
// for stats if configured.
func (s *Server) proxyChat(ctx context.Context, modelKey, backendSessionID string, chatReq *pb.ChatRequest, telemetry pb.Telemetry) (*pb.ExecuteChatResponse, error) {
	endpoint, err := s.executorEndpoint(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	conn, err := s.executorConn(endpoint)
	if err != nil {
		return nil, err
	}
	client := pb.NewExecutorClient(conn)

	backend := &pb.ExecuteChatRequest{
		Request:   proto.Clone(chatReq).(*pb.ChatRequest),
		Telemetry: telemetry,
	}
	backend.Request.SessionId = backendSessionID
	ref, executor, recorded := s.statsTarget(modelKey)
	asked := backend.GetTelemetry() != pb.Telemetry_TELEMETRY_NONE
	if recorded && !asked && s.stats.sampleTelemetry() {
		backend.Telemetry = pb.Telemetry_TELEMETRY_SUMMARY
	}
	resp, err := client.ExecuteChat(ctx, backend)
	if err != nil {
		return nil, err
	}
	if !recorded {
		return resp, nil
	}
	r := newRecord(ref, executor, "chat")
	innerResp := resp.GetResponse()
	if innerResp != nil {
		r.PromptTokens, r.PrefilledTokens = innerResp.SessionTokens-innerResp.Generated, innerResp.NewPromptTokens
		r.Generated, r.PrefillMs, r.MsPerToken = innerResp.Generated, innerResp.PrefillMs, innerResp.MsPerToken
	}
	if resp.TelemetryJson != "" {
		r.Telemetry = json.RawMessage(resp.TelemetryJson)
	}
	s.stats.record(ctx, r)
	return resp, nil
}

// Chat runs one conversation turn in a session (Engine service).
func (s *Server) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Chat request received", "session_id", req.SessionId)

	modelKey, backendSessionID, err := splitSessionID(req.SessionId)
	if err != nil {
		return nil, err
	}
	resp, err := s.proxyChat(ctx, modelKey, backendSessionID, req, pb.Telemetry_TELEMETRY_NONE)
	if err != nil {
		return nil, err
	}
	return resp.GetResponse(), nil
}

// ExecuteChat runs one conversation turn on the router's admin port (Executor service),
// allowing internal/admin callers to receive telemetry.
func (s *Server) ExecuteChat(ctx context.Context, req *pb.ExecuteChatRequest) (*pb.ExecuteChatResponse, error) {
	log := klog.FromContext(ctx)
	reqInner := req.GetRequest()
	if reqInner == nil {
		return nil, status.Error(codes.InvalidArgument, "missing request in ExecuteChatRequest")
	}
	log.Info("ExecuteChat request received", "session_id", reqInner.SessionId)

	modelKey, backendSessionID, err := splitSessionID(reqInner.SessionId)
	if err != nil {
		return nil, err
	}
	resp, err := s.proxyChat(ctx, modelKey, backendSessionID, reqInner, req.Telemetry)
	if err != nil {
		return nil, err
	}
	if req.GetTelemetry() == pb.Telemetry_TELEMETRY_NONE {
		resp.TelemetryJson = ""
	}
	return resp, nil
}

// proxyGenerate forwards a GenerateRequest to the executor via Execute, sampling telemetry
// for stats if configured, and yielding each ExecuteResponse to send.
func (s *Server) proxyGenerate(ctx context.Context, modelKey, backendSessionID string, genReq *pb.GenerateRequest, telemetry pb.Telemetry, send func(*pb.ExecuteResponse) error) error {
	endpoint, err := s.executorEndpoint(ctx, modelKey)
	if err != nil {
		return err
	}
	conn, err := s.executorConn(endpoint)
	if err != nil {
		return err
	}
	client := pb.NewExecutorClient(conn)

	backend := &pb.ExecuteRequest{
		Request:   proto.Clone(genReq).(*pb.GenerateRequest),
		Telemetry: telemetry,
	}
	backend.Request.SessionId = backendSessionID
	ref, executor, recorded := s.statsTarget(modelKey)
	asked := backend.GetTelemetry() != pb.Telemetry_TELEMETRY_NONE
	if recorded && !asked && s.stats.sampleTelemetry() {
		backend.Telemetry = pb.Telemetry_TELEMETRY_SUMMARY
	}
	upstream, err := client.Execute(ctx, backend)
	if err != nil {
		return err
	}
	for {
		resp, err := upstream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		innerResp := resp.GetResponse()
		if innerResp != nil && innerResp.Done && recorded {
			r := newRecord(ref, executor, "generate")
			r.PromptTokens, r.PrefilledTokens, r.Generated = innerResp.PromptTokens, innerResp.PrefilledTokens, innerResp.Generated
			r.PrefillMs, r.MsPerToken, r.FinishReason = innerResp.PrefillMs, innerResp.MsPerToken, innerResp.FinishReason
			if resp.TelemetryJson != "" {
				r.Telemetry = json.RawMessage(resp.TelemetryJson)
			}
			s.stats.record(ctx, r)
		}
		if err := send(resp); err != nil {
			return err
		}
	}
}

// Generate implements the Engine service: proxies the executor's token stream to the client.
func (s *Server) Generate(req *pb.GenerateRequest, stream grpc.ServerStreamingServer[pb.GenerateResponse]) error {
	ctx := stream.Context()
	log := klog.FromContext(ctx)
	log.Info("Generate request received", "session_id", req.SessionId, "prompt_tokens", len(req.InputIds))

	modelKey, backendSessionID, err := splitSessionID(req.SessionId)
	if err != nil {
		return err
	}
	return s.proxyGenerate(ctx, modelKey, backendSessionID, req, pb.Telemetry_TELEMETRY_NONE, func(resp *pb.ExecuteResponse) error {
		return stream.Send(resp.GetResponse())
	})
}

// Execute implements the Executor service on the router for internal or
// admin callers (e.g. vectors_test record --producer vxpu-router) needing telemetry.
func (s *Server) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	ctx := stream.Context()
	log := klog.FromContext(ctx)
	reqInner := req.GetRequest()
	if reqInner == nil {
		return status.Error(codes.InvalidArgument, "missing request in ExecuteRequest")
	}
	log.Info("Execute request received", "session_id", reqInner.SessionId, "prompt_tokens", len(reqInner.InputIds))

	modelKey, backendSessionID, err := splitSessionID(reqInner.SessionId)
	if err != nil {
		return err
	}
	asked := req.GetTelemetry() != pb.Telemetry_TELEMETRY_NONE
	return s.proxyGenerate(ctx, modelKey, backendSessionID, reqInner, req.Telemetry, func(resp *pb.ExecuteResponse) error {
		if !asked {
			resp.TelemetryJson = ""
		}
		return stream.Send(resp)
	})
}

// PeerMappingMethods helper for tests to directly query and inject mapping
func (s *Server) GetPeerToModel(peerKey string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	modelKey, exists := s.peerToModel[peerKey]
	return modelKey, exists
}

func (s *Server) SetPeerToModel(peerKey, modelKey string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.peerToModel[peerKey] = modelKey
}

func getRouterIP() string {
	if ip := os.Getenv("POD_IP"); ip != "" {
		return ip
	}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		klog.Warningf("listing interface addresses (falling back to loopback): %v", err)
		return "127.0.0.1"
	}
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String()
			}
		}
	}
	return "127.0.0.1"
}
