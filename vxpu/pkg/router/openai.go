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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
	"unicode/utf8"

	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// openAIChatRequest defines the fields accepted by POST /v1/chat/completions.
type openAIChatRequest struct {
	// Model specifies the name of the model to serve.
	Model string `json:"model"`

	// Messages contains the list of conversation turns.
	Messages []openAIMessage `json:"messages"`

	// MaxTokens is the maximum number of tokens to generate.
	MaxTokens *int `json:"max_tokens"`

	// MaxCompletionTokens is an alias for MaxTokens in newer OpenAI clients.
	MaxCompletionTokens *int `json:"max_completion_tokens"`

	// Temperature controls randomness; 0 means greedy decoding.
	Temperature *float32 `json:"temperature"`

	// TopP is the nucleus sampling cutoff probability.
	TopP *float32 `json:"top_p"`

	// Stream must be true; non-streaming is refused.
	Stream *bool `json:"stream"`

	// StreamOptions configures stream features such as usage reporting.
	StreamOptions *openAIStreamOpts `json:"stream_options"`

	// ChatTemplateKwargs carries template arguments for the model's tokenizer.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`

	// MinTokens specifies the minimum number of tokens to generate.
	MinTokens *int `json:"min_tokens"`

	// IgnoreEOS instructs the generator to continue past end-of-sequence tokens.
	IgnoreEOS *bool `json:"ignore_eos"`

	// Stop specifies custom stop sequences.
	Stop *[]string `json:"stop"`

	// N is the number of completions requested; n != 1 is refused.
	N *int `json:"n"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChunkChoice `json:"choices"`
	Usage   *chatUsage        `json:"usage,omitempty"`
}

type chatChunkChoice struct {
	Index        int            `json:"index"`
	Delta        chatChunkDelta `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

type chatChunkDelta struct {
	Content string `json:"content,omitempty"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// HTTPHandler returns an http.Handler serving the OpenAI-compatible API.
func (s *Server) HTTPHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	return mux
}

func validateOpenAIRequest(raw map[string]json.RawMessage, req *openAIChatRequest) (int, error) {
	for k := range raw {
		switch k {
		case "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
			"top_p", "stream", "stream_options", "chat_template_kwargs", "min_tokens",
			"ignore_eos", "stop", "n":
		default:
			return 0, fmt.Errorf("unsupported option: %s", k)
		}
	}
	if req.Stream == nil || !*req.Stream {
		return 0, errors.New("unsupported option: stream=false (only streaming is supported)")
	}
	if req.Model == "" {
		return 0, errors.New("model is required")
	}
	if req.N != nil && *req.N != 1 {
		return 0, errors.New("unsupported option: n (several choices not supported)")
	}
	if req.MaxTokens != nil && req.MaxCompletionTokens != nil && *req.MaxTokens != *req.MaxCompletionTokens {
		return 0, errors.New("conflicting options: max_tokens and max_completion_tokens must match when both are provided")
	}
	var maxTok *int
	if req.MaxTokens != nil {
		maxTok = req.MaxTokens
	} else if req.MaxCompletionTokens != nil {
		maxTok = req.MaxCompletionTokens
	}
	if maxTok == nil {
		return 0, errors.New("unsupported option: max_tokens is required")
	}
	if *maxTok <= 0 {
		return 0, errors.New("unsupported option: max_tokens must be positive")
	}
	if *maxTok > 4096 {
		return 0, errors.New("unsupported option: max_tokens exceeds maximum allowed limit of 4096")
	}
	if req.Temperature != nil && *req.Temperature < 0 {
		return 0, errors.New("unsupported option: temperature must be non-negative")
	}
	if req.TopP != nil && (*req.TopP <= 0 || *req.TopP > 1) {
		return 0, errors.New("unsupported option: top_p must be between 0 and 1")
	}
	if req.Stop != nil && len(*req.Stop) > 0 {
		return 0, errors.New("unsupported option: stop with entries is not supported")
	}
	if req.MinTokens != nil && *req.MinTokens > 0 {
		if req.IgnoreEOS == nil || !*req.IgnoreEOS || *req.MinTokens != *maxTok {
			return 0, errors.New("unsupported option: min_tokens is only supported together with ignore_eos=true and equal to max_tokens")
		}
	}
	var rawMsgs []map[string]json.RawMessage
	if err := json.Unmarshal(raw["messages"], &rawMsgs); err != nil || len(rawMsgs) == 0 {
		return 0, errors.New("messages cannot be empty")
	}
	for i, rm := range rawMsgs {
		for k := range rm {
			if k != "role" && k != "content" {
				return 0, fmt.Errorf("unsupported option in message %d: %s", i, k)
			}
		}
		var msg openAIMessage
		data, err := json.Marshal(rm)
		if err != nil {
			return 0, fmt.Errorf("marshalling message %d: %w", i, err)
		}
		if err := json.Unmarshal(data, &msg); err != nil {
			return 0, fmt.Errorf("unmarshalling message %d: %w", i, err)
		}
		if msg.Role != "user" && msg.Role != "assistant" && msg.Role != "system" {
			return 0, fmt.Errorf("unsupported role in message %d: %s", i, msg.Role)
		}
	}
	for k := range req.ChatTemplateKwargs {
		if k == "chat_template" || k == "tokenize" || k == "add_generation_prompt" {
			return 0, fmt.Errorf("unsupported chat_template_kwargs: %s", k)
		}
	}
	return *maxTok, nil
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	log := klog.FromContext(ctx)

	const maxBodySize = 10 * 1024 * 1024
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeClientError(ctx, w, http.StatusRequestEntityTooLarge, "request body too large (exceeds 10 MB limit)")
			return
		}
		writeClientError(ctx, w, http.StatusBadRequest, fmt.Sprintf("reading request body: %v", err))
		return
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(bodyBytes, &raw); err != nil {
		writeClientError(ctx, w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	var req openAIChatRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		writeClientError(ctx, w, http.StatusBadRequest, fmt.Sprintf("invalid request structure: %v", err))
		return
	}
	maxTokens, err := validateOpenAIRequest(raw, &req)
	if err != nil {
		writeClientError(ctx, w, http.StatusBadRequest, err.Error())
		return
	}

	// Resolve model once per request and confirm executor availability before any work
	modelKey, err := s.resolveModelKey(ctx, req.Model)
	if err != nil {
		writeClientError(ctx, w, http.StatusNotFound, fmt.Sprintf("The model '%s' does not exist", req.Model))
		return
	}
	if _, err := s.executorEndpoint(ctx, modelKey); err != nil {
		writeClientError(ctx, w, http.StatusNotFound, fmt.Sprintf("The model '%s' does not exist", req.Model))
		return
	}

	vocab, err := s.getVocabByKey(ctx, modelKey)
	if err != nil {
		writeServerError(ctx, w, fmt.Sprintf("failed to get vocab: %v", err))
		return
	}

	backendSessionID, err := s.newSessionOnExecutor(ctx, modelKey)
	if err != nil {
		writeServerError(ctx, w, fmt.Sprintf("failed to allocate session: %v", err))
		return
	}

	var chatMsgs []*pb.ChatMessage
	for _, m := range req.Messages {
		chatMsgs = append(chatMsgs, &pb.ChatMessage{Role: m.Role, Content: m.Content})
	}
	kwargsJSON := ""
	if len(req.ChatTemplateKwargs) > 0 {
		data, err := json.Marshal(req.ChatTemplateKwargs)
		if err != nil {
			writeClientError(ctx, w, http.StatusBadRequest, fmt.Sprintf("invalid chat_template_kwargs: %v", err))
			return
		}
		kwargsJSON = string(data)
	}

	tokResp, err := s.tokenizeChatByKey(ctx, modelKey, &pb.TokenizeChatRequest{
		Messages:               chatMsgs,
		ChatTemplateKwargsJson: kwargsJSON,
	})
	if err != nil {
		writeServerError(ctx, w, fmt.Sprintf("failed to tokenize chat: %v", err))
		return
	}

	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if flusher != nil {
		flusher.Flush()
	}

	genReq := &pb.GenerateRequest{
		SessionId:    backendSessionID,
		InputIds:     tokResp.GetInputIds(),
		MaxNewTokens: proto.Int32(int32(maxTokens)),
	}
	if req.Temperature != nil {
		if *req.Temperature > 0 {
			genReq.Temperature = req.Temperature
			genReq.DoSample = proto.Bool(true)
		} else {
			genReq.DoSample = proto.Bool(false)
		}
	}
	if req.TopP != nil {
		genReq.TopP = req.TopP
	}
	if req.IgnoreEOS != nil {
		genReq.IgnoreEos = req.IgnoreEOS
	}

	streamID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	decoder := newStreamDecoder(vocab)
	var finalResp *pb.GenerateResponse

	sendChunk := func(delta string, finishReason *string, usage *chatUsage) error {
		choices := []chatChunkChoice{}
		if finishReason != nil || delta != "" {
			choices = append(choices, chatChunkChoice{
				Index:        0,
				Delta:        chatChunkDelta{Content: delta},
				FinishReason: finishReason,
			})
		}
		return writeSSEChunk(w, flusher, chatChunk{
			ID:      streamID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: choices,
			Usage:   usage,
		})
	}

	err = s.proxyGenerate(ctx, modelKey, backendSessionID, genReq, pb.Telemetry_TELEMETRY_NONE, func(resp *pb.ExecuteResponse) error {
		inner := resp.GetResponse()
		if inner == nil {
			return nil
		}
		if len(inner.GetTokenIds()) > 0 {
			if text := decoder.decode(inner.GetTokenIds()); text != "" {
				if err := sendChunk(text, nil, nil); err != nil {
					return err
				}
			}
		}
		if inner.GetDone() {
			finalResp = inner
		}
		return nil
	})

	if err != nil {
		if ctx.Err() == nil {
			log.Error(err, "error during generation stream", "model_key", modelKey)
			writeSSEError(ctx, w, flusher, err.Error())
		}
		return
	}

	if remaining := decoder.flush(); remaining != "" {
		if err := sendChunk(remaining, nil, nil); err != nil {
			log.Error(err, "failed to write remaining chunk")
			return
		}
	}

	finishReason := "stop"
	if finalResp != nil && finalResp.GetFinishReason() == "length" {
		finishReason = "length"
	}
	if err := sendChunk("", &finishReason, nil); err != nil {
		log.Error(err, "failed to write finish chunk")
		return
	}

	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage && finalResp != nil {
		pTok, cTok := int(finalResp.GetPromptTokens()), int(finalResp.GetGenerated())
		if err := sendChunk("", nil, &chatUsage{PromptTokens: pTok, CompletionTokens: cTok, TotalTokens: pTok + cTok}); err != nil {
			log.Error(err, "failed to write usage chunk")
			return
		}
	}

	if err := writeSSEDone(w, flusher); err != nil {
		log.Error(err, "failed to write [DONE]")
	}
}

// streamDecoder reconstructs UTF-8 text from vocabulary pieces across streaming token arrivals.
type streamDecoder struct {
	vocab    [][]byte
	specials map[int32]bool
	pending  []byte
}

func newStreamDecoder(vocabResp *pb.GetVocabResponse) *streamDecoder {
	specials := make(map[int32]bool, len(vocabResp.GetSpecialTokenIds()))
	for _, id := range vocabResp.GetSpecialTokenIds() {
		specials[id] = true
	}
	return &streamDecoder{
		vocab:    vocabResp.GetPieces(),
		specials: specials,
	}
}

func (d *streamDecoder) decode(tokenIDs []int32) string {
	for _, id := range tokenIDs {
		if d.specials[id] || id < 0 || int(id) >= len(d.vocab) {
			continue
		}
		d.pending = append(d.pending, d.vocab[id]...)
	}
	var decoded []byte
	for len(d.pending) > 0 && utf8.FullRune(d.pending) {
		_, size := utf8.DecodeRune(d.pending)
		decoded = append(decoded, d.pending[:size]...)
		d.pending = d.pending[size:]
	}
	return string(decoded)
}

func (d *streamDecoder) flush() string {
	if len(d.pending) == 0 {
		return ""
	}
	rem := string(d.pending)
	d.pending = nil
	return rem
}

func writeSSEChunk(w http.ResponseWriter, flusher http.Flusher, chunk any) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

func writeSSEDone(w http.ResponseWriter, flusher http.Flusher) error {
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

func writeSSEError(ctx context.Context, w http.ResponseWriter, flusher http.Flusher, msg string) {
	log := klog.FromContext(ctx)
	errPayload, err := json.Marshal(map[string]any{
		"error": map[string]string{"message": msg},
	})
	if err != nil {
		errPayload = []byte(`{"error":{"message":"internal error"}}`)
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", errPayload); err != nil {
		log.Error(err, "writing SSE error")
	}
	if flusher != nil {
		flusher.Flush()
	}
}

func writeClientError(ctx context.Context, w http.ResponseWriter, status int, msg string) {
	log := klog.FromContext(ctx)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "invalid_request_error",
		},
	}); err != nil {
		log.Error(err, "writing client error")
	}
}

func writeServerError(ctx context.Context, w http.ResponseWriter, msg string) {
	log := klog.FromContext(ctx)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	if err := json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "server_error",
		},
	}); err != nil {
		log.Error(err, "writing server error")
	}
}
