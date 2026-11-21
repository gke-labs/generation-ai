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

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func (s *Server) resolveModelKey(ctx context.Context, model string) (string, error) {
	if model != "" {
		if s.weightStore != nil {
			return s.loadByName(ctx, model, "")
		}
		return model, nil
	}
	peerKey := getPeerKey(ctx)
	s.mu.RLock()
	key, exists := s.peerToModel[peerKey]
	s.mu.RUnlock()
	if !exists {
		return "", status.Errorf(codes.FailedPrecondition, "no model has been loaded for this connection")
	}
	return key, nil
}

// tokenizeChatByKey renders chat messages into prompt token IDs on the executor serving modelKey.
func (s *Server) tokenizeChatByKey(ctx context.Context, modelKey string, req *pb.TokenizeChatRequest) (*pb.TokenizeChatResponse, error) {
	client, err := s.executorClient(ctx, modelKey)
	if err != nil {
		return nil, err
	}

	log := klog.FromContext(ctx)
	log.Info("TokenizeChat request received", "model_key", modelKey, "messages", len(req.GetMessages()))
	return client.TokenizeChat(ctx, req)
}

// TokenizeChat turns chat messages into prompt token ids using the model's chat template.
func (s *Server) TokenizeChat(ctx context.Context, req *pb.TokenizeChatRequest) (*pb.TokenizeChatResponse, error) {
	modelKey, err := s.resolveModelKey(ctx, req.GetModel())
	if err != nil {
		return nil, err
	}
	return s.tokenizeChatByKey(ctx, modelKey, req)
}

// getVocabByKey reports the model's vocabulary pieces and special tokens for an already-resolved model key.
func (s *Server) getVocabByKey(ctx context.Context, modelKey string) (*pb.GetVocabResponse, error) {
	s.mu.RLock()
	cached, ok := s.vocabs[modelKey]
	s.mu.RUnlock()
	if ok {
		return cached, nil
	}

	client, err := s.executorClient(ctx, modelKey)
	if err != nil {
		return nil, err
	}

	log := klog.FromContext(ctx)
	log.Info("Fetching vocabulary from executor", "model_key", modelKey)
	resp, err := client.GetVocab(ctx, &pb.GetVocabRequest{})
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	if s.vocabs == nil {
		s.vocabs = make(map[string]*pb.GetVocabResponse)
	}
	s.vocabs[modelKey] = resp
	s.mu.Unlock()

	return resp, nil
}

// GetVocab reports the model's vocabulary pieces and special tokens.
func (s *Server) GetVocab(ctx context.Context, req *pb.GetVocabRequest) (*pb.GetVocabResponse, error) {
	modelKey, err := s.resolveModelKey(ctx, req.GetModel())
	if err != nil {
		return nil, err
	}
	return s.getVocabByKey(ctx, modelKey)
}
