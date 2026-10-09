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
	"fmt"
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

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/blobserver"
)

type Server struct {
	pb.UnimplementedExecutorServer
	clientset   kubernetes.Interface
	namespace   string
	imageName   string
	accelerator string
	blobServer  *blobserver.Server

	mu          sync.RWMutex
	peerToModel map[string]string    // maps peer IP:port -> modelKey
	placements  map[string]Placement // modelKey -> where/how big its executor is
}

func NewServer(clientset kubernetes.Interface, namespace, imageName, accelerator string) *Server {
	bs, err := blobserver.NewServer("/tmp/vxpu-cache", 8080)
	if err != nil {
		klog.Errorf("Failed to initialize blobserver: %v", err)
	} else {
		bs.Start()
	}

	s := &Server{
		clientset:   clientset,
		namespace:   namespace,
		imageName:   imageName,
		accelerator: accelerator,
		peerToModel: make(map[string]string),
		placements:  make(map[string]Placement),
		blobServer:  bs,
	}
	return s
}

func getPeerKey(ctx context.Context) string {
	if p, ok := peer.FromContext(ctx); ok {
		return p.Addr.String()
	}
	return "unknown"
}

// placementFor returns the placement recorded by LoadModel, or a
// default for the configured accelerator when the model's size is
// unknown (a pod recreated after eviction, before any LoadModel).
func (s *Server) placementFor(modelID string) Placement {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if p, ok := s.placements[modelID]; ok {
		return p
	}
	return PlanPlacement(s.accelerator, 0)
}

func (s *Server) getOrStartPod(ctx context.Context, modelID string) (string, error) {
	log := klog.FromContext(ctx)
	if modelID == "" {
		return "", fmt.Errorf("modelID cannot be empty")
	}

	podName := "vxpu-executor-" + modelID

	// Check if pod already exists
	pod, err := s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if err == nil {
		log.Info("Found existing pod", "pod", podName, "phase", pod.Status.Phase)
		if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
			return pod.Status.PodIP, nil
		}
		if pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodUnknown {
			log.Info("Existing pod is in bad state, deleting", "pod", podName, "phase", pod.Status.Phase)
			_ = s.clientset.CoreV1().Pods(s.namespace).Delete(ctx, podName, metav1.DeleteOptions{})
			// Give Kubernetes a moment to clean up
			time.Sleep(2 * time.Second)
		}
	} else if !errors.IsNotFound(err) {
		return "", fmt.Errorf("failed to get pod: %w", err)
	}

	// Create a new Pod if it didn't exist or was deleted
	_, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		placement := s.placementFor(modelID)
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

		newPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: s.namespace,
				Labels: map[string]string{
					"app":      "vxpu-executor",
					"model-id": modelID,
				},
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
						// Keep an idle model resident long enough for
						// interactive use (a notebook pauses between
						// cells); eviction means a full rehydrate.
						Args: []string{"--keep-alive", executorKeepAliveSeconds()},
						Ports: []corev1.ContainerPort{
							{
								ContainerPort: 50051,
							},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{
									Port: intstr.FromInt32(50051),
								},
							},
							InitialDelaySeconds: 3,
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
	log := klog.FromContext(ctx)

	// Derive modelKey from manifest_json (using original ManifestJson for identity)
	h := sha256.Sum256([]byte(req.ManifestJson))
	modelKey := hex.EncodeToString(h[:16])
	log.Info("LoadModel request received", "model_key", modelKey)

	if s.blobServer == nil {
		return nil, status.Errorf(codes.Internal, "blobserver is not initialized")
	}

	routerIP := getRouterIP()
	// Caching tens of GB outlives impatient clients: a dropped
	// connection must not abort a download that the retry would only
	// restart from zero.
	rewrittenManifestJSON, err := s.blobServer.CacheAndRewriteManifest(
		context.WithoutCancel(ctx), req.ManifestJson, routerIP)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to cache and rewrite manifest: %v", err)
	}

	// Prepare rewritten request
	reqCopy := &pb.LoadModelRequest{
		ManifestJson: rewrittenManifestJSON,
		BindingJson:  req.BindingJson,
		PrefillGraph: req.PrefillGraph,
		DecodeGraph:  req.DecodeGraph,
	}

	bytes, err := weightBytes(req.ManifestJson)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid manifest: %v", err)
	}
	placement := PlanPlacement(s.accelerator, bytes)
	log.Info("Planned executor placement", "model_key", modelKey,
		"weightBytes", bytes, "accelerator", placement.Accelerator,
		"cpu", placement.CPU.String(), "memory", placement.Memory.String(),
		"reason", placement.Reason)

	peerKey := getPeerKey(ctx)
	s.mu.Lock()
	if peerKey != "unknown" {
		s.peerToModel[peerKey] = modelKey
	}
	s.placements[modelKey] = placement
	s.mu.Unlock()

	podIP, err := s.getOrStartPod(ctx, modelKey)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewExecutorClient(conn)

	var loadErr error
	for i := 0; i < 30; i++ {
		loadCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, loadErr = client.LoadModel(loadCtx, reqCopy)
		cancel()
		if loadErr == nil {
			break
		}
		log.Info("Waiting for executor gRPC server to accept LoadModel", "attempt", i, "error", loadErr.Error())
		time.Sleep(2 * time.Second)
	}

	if loadErr != nil {
		// Propagate backend status error instead of collapsing to codes.Internal
		return nil, loadErr
	}

	log.Info("Successfully loaded model on executor", "model_key", modelKey)
	return &pb.LoadModelResponse{Success: true}, nil
}

func (s *Server) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	log := klog.FromContext(ctx)
	peerKey := getPeerKey(ctx)

	s.mu.RLock()
	modelKey, exists := s.peerToModel[peerKey]
	s.mu.RUnlock()

	if !exists {
		return nil, status.Errorf(codes.FailedPrecondition, "no model has been loaded for this connection")
	}

	log.Info("NewSession request for", "model_key", modelKey)

	podIP, err := s.getOrStartPod(ctx, modelKey)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewExecutorClient(conn)
	sessResp, err := client.NewSession(ctx, req)
	if err != nil {
		// Propagate the exact backend status (including FAILED_PRECONDITION) instead of collapsing to codes.Internal
		return nil, err
	}

	wrappedSessionID := fmt.Sprintf("%s:%s", modelKey, sessResp.SessionId)
	log.Info("Successfully created routed session", "wrapped_session_id", wrappedSessionID)
	return &pb.NewSessionResponse{
		SessionId: wrappedSessionID,
	}, nil
}

func (s *Server) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Chat request received", "session_id", req.SessionId)

	parts := strings.SplitN(req.SessionId, ":", 2)
	if len(parts) != 2 {
		return nil, status.Errorf(codes.InvalidArgument, "invalid routed session ID format; expected 'modelKey:sessionID'")
	}
	modelKey, backendSessionID := parts[0], parts[1]

	podIP, err := s.getOrStartPod(ctx, modelKey)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewExecutorClient(conn)
	resp, err := client.Chat(ctx, backendChatRequest(req, backendSessionID))
	if err != nil {
		// Propagate original backend status error
		return nil, err
	}

	return resp, nil
}

// backendChatRequest is the client's request addressed to the executor's
// own session id. Everything else is forwarded untouched, so new
// ChatRequest fields reach the executor without router changes.
func backendChatRequest(req *pb.ChatRequest, backendSessionID string) *pb.ChatRequest {
	backend := proto.Clone(req).(*pb.ChatRequest)
	backend.SessionId = backendSessionID
	return backend
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

// executorKeepAliveSeconds is how long an executor keeps an idle model
// loaded (VXPU_EXECUTOR_KEEP_ALIVE, seconds; default one hour).
func executorKeepAliveSeconds() string {
	if v := os.Getenv("VXPU_EXECUTOR_KEEP_ALIVE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return strconv.Itoa(n)
		}
	}
	return "3600"
}

func getRouterIP() string {
	if ip := os.Getenv("POD_IP"); ip != "" {
		return ip
	}
	addrs, err := net.InterfaceAddrs()
	if err == nil {
		for _, addr := range addrs {
			if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
				if ipnet.IP.To4() != nil {
					return ipnet.IP.String()
				}
			}
		}
	}
	return "127.0.0.1"
}
