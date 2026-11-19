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
	"reflect"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

type vocabExecutor struct {
	pb.UnimplementedExecutorServer
	vocabCalls     int
	tokenizeCalls  int
	lastMessages   []*pb.ChatMessage
	lastKwargsJSON string
}

func (v *vocabExecutor) TokenizeChat(_ context.Context, req *pb.TokenizeChatRequest) (*pb.TokenizeChatResponse, error) {
	v.tokenizeCalls++
	v.lastMessages = req.GetMessages()
	v.lastKwargsJSON = req.GetChatTemplateKwargsJson()
	return &pb.TokenizeChatResponse{
		InputIds: []int32{1, 10, 20, 2},
	}, nil
}

func (v *vocabExecutor) GetVocab(_ context.Context, _ *pb.GetVocabRequest) (*pb.GetVocabResponse, error) {
	v.vocabCalls++
	return &pb.GetVocabResponse{
		Pieces: [][]byte{
			[]byte(""),
			[]byte("Hello"),
			[]byte(" world"),
		},
		SpecialTokenIds: []int32{0},
	}, nil
}

func TestRouter_TokenizeChatAndVocab(t *testing.T) {
	ctx := t.Context()

	// Start in-process executor
	execLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	exec := &vocabExecutor{}
	execSrv := grpc.NewServer()
	pb.RegisterExecutorServer(execSrv, exec)
	go func() { _ = execSrv.Serve(execLis) }()
	t.Cleanup(execSrv.Stop)

	router := NewServer(fake.NewSimpleClientset(), "default", "img", "")
	modelKey := "test-model-key"
	router.placedExternally(modelKey, execLis.Addr().String())
	router.setModelStanding(modelKey, standingPassed, nil)

	// Test TokenizeChat forwards request and JSON kwargs
	tokResp, err := router.TokenizeChat(ctx, &pb.TokenizeChatRequest{
		Model: modelKey,
		Messages: []*pb.ChatMessage{
			{Role: "user", Content: "Hello"},
		},
		ChatTemplateKwargsJson: `{"enable_thinking":false,"style":"concise"}`,
	})
	if err != nil {
		t.Fatalf("TokenizeChat failed: %v", err)
	}
	wantIDs := []int32{1, 10, 20, 2}
	if !reflect.DeepEqual(tokResp.GetInputIds(), wantIDs) {
		t.Errorf("expected input ids %v, got %v", wantIDs, tokResp.GetInputIds())
	}
	if exec.tokenizeCalls != 1 {
		t.Errorf("expected 1 tokenize call, got %d", exec.tokenizeCalls)
	}
	if exec.lastKwargsJSON != `{"enable_thinking":false,"style":"concise"}` {
		t.Errorf("expected kwargs JSON forwarded, got %q", exec.lastKwargsJSON)
	}

	// Test GetVocab (first call fetches from executor)
	vocabResp, err := router.GetVocab(ctx, &pb.GetVocabRequest{Model: modelKey})
	if err != nil {
		t.Fatalf("GetVocab failed: %v", err)
	}
	if exec.vocabCalls != 1 {
		t.Errorf("expected 1 vocab call, got %d", exec.vocabCalls)
	}
	if len(vocabResp.GetPieces()) != 3 {
		t.Errorf("expected 3 pieces, got %d", len(vocabResp.GetPieces()))
	}
	if !reflect.DeepEqual(vocabResp.GetSpecialTokenIds(), []int32{0}) {
		t.Errorf("special token ids mismatch: %v", vocabResp.GetSpecialTokenIds())
	}

	// Test caching: second call returns cached response without second RPC
	vocabResp2, err := router.GetVocab(ctx, &pb.GetVocabRequest{Model: modelKey})
	if err != nil {
		t.Fatalf("GetVocab (cached) failed: %v", err)
	}
	if exec.vocabCalls != 1 {
		t.Errorf("expected cached vocab to avoid second executor RPC, got %d calls", exec.vocabCalls)
	}
	if !reflect.DeepEqual(vocabResp2.GetPieces(), vocabResp.GetPieces()) {
		t.Errorf("cached response mismatch")
	}

	// Test error when model is not loaded
	_, err = router.GetVocab(ctx, &pb.GetVocabRequest{})
	if status.Code(err) != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition for unassociated connection, got %v", err)
	}
}
