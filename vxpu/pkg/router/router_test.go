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
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestGetOrStartPod_ExistingRunningPod(t *testing.T) {
	ctx := t.Context()
	modelKey := "test-model-key"
	podName := "vxpu-executor-" + modelKey
	namespace := "test-ns"

	existingPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels: map[string]string{
				"app":      "vxpu-executor",
				"model-id": modelKey,
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "10.244.1.100",
		},
	}

	clientset := fake.NewSimpleClientset(existingPod)

	s := NewServer(clientset, namespace, "test-image", "")

	ip, err := s.getOrStartPod(ctx, modelKey)
	if err != nil {
		t.Fatalf("getOrStartPod failed: %v", err)
	}

	expectedIP := "10.244.1.100"
	if ip != expectedIP {
		t.Errorf("Expected IP %q, got %q", expectedIP, ip)
	}
}

func TestWrappedSessionIDParsing(t *testing.T) {
	// Let's test splitting the wrapped SessionId we use in Chat
	req := &pb.ChatRequest{
		SessionId: "abcdef123456:session-789",
		Text:      "Hello",
	}

	parts := strings.SplitN(req.SessionId, ":", 2)
	if len(parts) != 2 {
		t.Fatalf("Expected 2 parts, got %d", len(parts))
	}

	modelKey, backendSessionID := parts[0], parts[1]
	if modelKey != "abcdef123456" {
		t.Errorf("Expected modelKey 'abcdef123456', got %q", modelKey)
	}
	if backendSessionID != "session-789" {
		t.Errorf("Expected backendSessionID 'session-789', got %q", backendSessionID)
	}
}

func TestBackendChatRequest_ForwardsAllFields(t *testing.T) {
	req := &pb.ChatRequest{
		SessionId:    "abcdef123456:s7",
		Text:         "<bos><|turn>user\nhi<turn|>",
		MaxNewTokens: 42,
		RawPrompt:    true,
	}
	got := backendChatRequest(req, "s7")
	if got.SessionId != "s7" {
		t.Errorf("expected backend session id s7, got %q", got.SessionId)
	}
	if got.Text != req.Text || got.MaxNewTokens != 42 || !got.RawPrompt {
		t.Errorf("request fields not forwarded: %+v", got)
	}
	if req.SessionId != "abcdef123456:s7" {
		t.Error("client request must not be mutated")
	}
}

func TestGetPeerKey(t *testing.T) {
	// Test without peer info
	ctx := t.Context()
	key := getPeerKey(ctx)
	if key != "unknown" {
		t.Errorf("Expected 'unknown', got %q", key)
	}

	// Test with fake peer info
	fakePeer := &peer.Peer{
		Addr: fakeAddr{netType: "tcp", addrStr: "127.0.0.1:12345"},
	}
	ctxWithPeer := peer.NewContext(ctx, fakePeer)
	keyWithPeer := getPeerKey(ctxWithPeer)
	if keyWithPeer != "127.0.0.1:12345" {
		t.Errorf("Expected '127.0.0.1:12345', got %q", keyWithPeer)
	}
}

func TestLoadModel_PeerMapping(t *testing.T) {
	manifest := `{"model": "gemma"}`
	h := sha256.Sum256([]byte(manifest))
	expectedKey := hex.EncodeToString(h[:16])

	s := NewServer(nil, "default", "test-image", "")

	fakePeer := &peer.Peer{
		Addr: fakeAddr{netType: "tcp", addrStr: "127.0.0.1:12345"},
	}
	ctx := peer.NewContext(t.Context(), fakePeer)

	peerKey := getPeerKey(ctx)
	s.SetPeerToModel(peerKey, expectedKey)

	mapped, exists := s.GetPeerToModel("127.0.0.1:12345")

	if !exists {
		t.Fatalf("Expected peer mapping to exist")
	}
	if mapped != expectedKey {
		t.Errorf("Expected mapped key %q, got %q", expectedKey, mapped)
	}
}

func TestNewSession_NoModelLoaded(t *testing.T) {
	s := NewServer(nil, "default", "test-image", "")

	ctx := t.Context()
	_, err := s.NewSession(ctx, &pb.NewSessionRequest{})
	if err == nil {
		t.Fatalf("Expected error when no model is loaded")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("Expected status error")
	}
	if st.Code() != codes.FailedPrecondition {
		t.Errorf("Expected FailedPrecondition code, got %v", st.Code())
	}
}

type fakeAddr struct {
	netType string
	addrStr string
}

func (f fakeAddr) Network() string { return f.netType }
func (f fakeAddr) String() string  { return f.addrStr }
