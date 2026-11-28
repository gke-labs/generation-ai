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
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

type mockExecutor struct {
	pb.UnimplementedExecutorServer

	mu sync.Mutex

	// sessions tracks active sessions allocated on this executor.
	sessions map[string]bool

	// nextSessionID provides monotonic session ids.
	nextSessionID int

	// vocab is returned by GetVocab.
	vocab *pb.GetVocabResponse

	// tokenStream defines token IDs emitted on Execute.
	tokenStream [][]int32

	// streamErr is emitted by Execute if non-nil.
	streamErr error

	// pauseStream blocks Execute until closed.
	pauseStream chan struct{}

	// tokenSent notifies when the first token was sent.
	tokenSent chan struct{}
}

func newMockExecutor() *mockExecutor {
	return &mockExecutor{
		sessions: make(map[string]bool),
	}
}

func (m *mockExecutor) NewSession(_ context.Context, _ *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nextSessionID++
	id := fmt.Sprintf("sess-%d", m.nextSessionID)
	m.sessions[id] = true
	return &pb.NewSessionResponse{SessionId: id}, nil
}

func (m *mockExecutor) TokenizeChat(_ context.Context, _ *pb.TokenizeChatRequest) (*pb.TokenizeChatResponse, error) {
	return &pb.TokenizeChatResponse{InputIds: []int32{1, 2, 3}}, nil
}

func (m *mockExecutor) GetVocab(_ context.Context, _ *pb.GetVocabRequest) (*pb.GetVocabResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.vocab != nil {
		return m.vocab, nil
	}
	return &pb.GetVocabResponse{}, nil
}

func (m *mockExecutor) Execute(req *pb.ExecuteRequest, stream grpc.ServerStreamingServer[pb.ExecuteResponse]) error {
	m.mu.Lock()
	tokens := m.tokenStream
	streamErr := m.streamErr
	pause := m.pauseStream
	tokenSent := m.tokenSent
	m.mu.Unlock()

	for _, batch := range tokens {
		if pause != nil {
			if tokenSent != nil {
				select {
				case <-tokenSent:
				default:
					close(tokenSent)
				}
			}
			select {
			case <-pause:
			case <-stream.Context().Done():
				return stream.Context().Err()
			}
		}
		if err := stream.Send(&pb.ExecuteResponse{
			Response: &pb.GenerateResponse{TokenIds: batch},
		}); err != nil {
			return err
		}
	}

	if streamErr != nil {
		return streamErr
	}

	totalGenerated := 0
	for _, b := range tokens {
		totalGenerated += len(b)
	}

	return stream.Send(&pb.ExecuteResponse{
		Response: &pb.GenerateResponse{
			Done:         true,
			FinishReason: "stop",
			PromptTokens: 5,
			Generated:    int32(totalGenerated),
		},
	})
}

func setupTestRouter(t *testing.T, exec *mockExecutor) (*httptest.Server, *Server) {
	t.Helper()
	execLis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	execSrv := grpc.NewServer()
	pb.RegisterExecutorServer(execSrv, exec)
	go func() { _ = execSrv.Serve(execLis) }()
	t.Cleanup(execSrv.Stop)

	router := NewServer(fake.NewSimpleClientset(), "default", "img", "")
	router.placedExternally("test-model", execLis.Addr().String())

	ts := httptest.NewServer(router.HTTPHandler())
	t.Cleanup(ts.Close)
	return ts, router
}

func TestOpenAI_StreamFramingAndMultiByteCharacters(t *testing.T) {
	exec := newMockExecutor()
	pieces := [][]byte{
		[]byte(""),
		[]byte("Hello"),
		[]byte(" "),
		{0xC3}, {0xA9},
		{0xE2}, {0x80}, {0x94},
		{0xF0}, {0x9F}, {0x8C}, {0x9F},
	}
	exec.vocab = &pb.GetVocabResponse{
		Pieces:          pieces,
		SpecialTokenIds: []int32{0},
	}
	exec.tokenStream = [][]int32{
		{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11},
	}

	ts, _ := setupTestRouter(t, exec)

	reqBody := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"max_tokens": 100,
		"stream": true,
		"stream_options": {"include_usage": true}
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(b))
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %q", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var fullText string
	var seenDone bool
	var seenFinishReason string
	var seenUsage *chatUsage
	var seenUsageRaw string

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			seenDone = true
			break
		}

		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("unmarshal chunk error: %v, raw: %s", err, payload)
		}

		if chunk.Usage != nil {
			seenUsage = chunk.Usage
			seenUsageRaw = payload
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if strings.ContainsRune(choice.Delta.Content, '\uFFFD') {
					t.Errorf("unexpected unicode replacement character in chunk: %q", choice.Delta.Content)
				}
				fullText += choice.Delta.Content
			}
			if choice.FinishReason != nil {
				seenFinishReason = *choice.FinishReason
			}
		}
	}

	wantText := "Hello é—🌟"
	if fullText != wantText {
		t.Errorf("expected full text %q, got %q", wantText, fullText)
	}
	if !seenDone {
		t.Errorf("expected to see [DONE] end marker")
	}
	if seenFinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", seenFinishReason)
	}
	if seenUsage == nil {
		t.Fatal("expected usage chunk, got nil")
	}
	if seenUsage.PromptTokens != 5 || seenUsage.CompletionTokens != 11 || seenUsage.TotalTokens != 16 {
		t.Errorf("unexpected usage counts: %+v", seenUsage)
	}

	// Assert the raw bytes of the usage chunk contain "choices":[] and not null
	if !strings.Contains(seenUsageRaw, `"choices":[]`) {
		t.Errorf("expected usage chunk to have raw '\"choices\":[]', got %s", seenUsageRaw)
	}
	if strings.Contains(seenUsageRaw, `"choices":null`) {
		t.Errorf("usage chunk sent choices as null: %s", seenUsageRaw)
	}
}

func TestOpenAI_VocabDecodingEquality_692(t *testing.T) {
	exec := newMockExecutor()
	pieces := make([][]byte, 30000)
	pieces[10994] = []byte("Hello")
	pieces[3186] = []byte(" world")
	pieces[29991] = []byte("!")
	pieces[29871] = []byte(" ")
	pieces[243] = []byte{0xF0}
	pieces[162] = []byte{0x9F}
	pieces[143] = []byte{0x8C}
	pieces[2] = []byte("") // EOS

	exec.vocab = &pb.GetVocabResponse{
		Pieces:          pieces,
		SpecialTokenIds: []int32{2},
	}
	exec.tokenStream = [][]int32{
		{10994}, {3186}, {29991}, {29871}, {243}, {162}, {143}, {162}, {2},
	}

	ts, _ := setupTestRouter(t, exec)

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "stream": true}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var fullText string
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			fullText += c.Delta.Content
		}
	}

	want := "Hello world! 🌟"
	if fullText != want {
		t.Errorf("vocab decoding equality mismatch: want %q, got %q", want, fullText)
	}
}

func TestOpenAI_Refusals(t *testing.T) {
	exec := newMockExecutor()
	ts, _ := setupTestRouter(t, exec)

	tests := []struct {
		name       string
		body       string
		wantSubstr string
	}{
		{
			name:       "stream false",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": false}`,
			wantSubstr: "stream=false",
		},
		{
			name:       "missing stream",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}`,
			wantSubstr: "stream=false",
		},
		{
			name:       "missing max_tokens",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": true}`,
			wantSubstr: "max_tokens is required",
		},
		{
			name:       "negative max_tokens",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5, "stream": true}`,
			wantSubstr: "max_tokens must be positive",
		},
		{
			name:       "max_tokens exceeds 4096",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000, "stream": true}`,
			wantSubstr: "4096",
		},
		{
			name:       "conflicting max_tokens and max_completion_tokens",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "max_completion_tokens": 200, "stream": true}`,
			wantSubstr: "conflicting options",
		},
		{
			name:       "negative temperature",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "temperature": -0.5, "stream": true}`,
			wantSubstr: "temperature must be non-negative",
		},
		{
			name:       "top_p zero or negative",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "top_p": 0, "stream": true}`,
			wantSubstr: "top_p must be between 0 and 1",
		},
		{
			name:       "top_p greater than 1",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "top_p": 1.5, "stream": true}`,
			wantSubstr: "top_p must be between 0 and 1",
		},
		{
			name:       "stop with entries",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stop": ["\n"], "stream": true}`,
			wantSubstr: "stop with entries is not supported",
		},
		{
			name:       "min_tokens without ignore_eos",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "min_tokens": 10, "stream": true}`,
			wantSubstr: "min_tokens",
		},
		{
			name:       "min_tokens not equal to max_tokens",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20, "min_tokens": 10, "ignore_eos": true, "stream": true}`,
			wantSubstr: "min_tokens",
		},
		{
			name:       "n != 1 refused (n=0)",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "n": 0, "stream": true}`,
			wantSubstr: "n",
		},
		{
			name:       "n != 1 refused (n=2)",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "n": 2, "stream": true}`,
			wantSubstr: "n",
		},
		{
			name:       "tool role in message refused",
			body:       `{"model": "test-model", "messages": [{"role": "tool", "content": "result"}], "max_tokens": 10, "stream": true}`,
			wantSubstr: "tool",
		},
		{
			name:       "unknown option in message refused",
			body:       `{"model": "test-model", "messages": [{"role": "assistant", "content": "", "tool_calls": []}], "max_tokens": 10, "stream": true}`,
			wantSubstr: "tool_calls",
		},
		{
			name:       "tools unsupported",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "tools": []}`,
			wantSubstr: "tools",
		},
		{
			name:       "tool_choice unsupported",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "tool_choice": "auto"}`,
			wantSubstr: "tool_choice",
		},
		{
			name:       "logprobs unsupported",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "logprobs": true}`,
			wantSubstr: "logprobs",
		},
		{
			name:       "presence_penalty unsupported",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "presence_penalty": 0.5}`,
			wantSubstr: "presence_penalty",
		},
		{
			name:       "frequency_penalty unsupported",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "frequency_penalty": 0.5}`,
			wantSubstr: "frequency_penalty",
		},
		{
			name:       "chat_template swap refused",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "chat_template_kwargs": {"chat_template": "custom"}}`,
			wantSubstr: "chat_template",
		},
		{
			name:       "tokenize option refused",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "chat_template_kwargs": {"tokenize": false}}`,
			wantSubstr: "tokenize",
		},
		{
			name:       "add_generation_prompt option refused",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "chat_template_kwargs": {"add_generation_prompt": true}}`,
			wantSubstr: "add_generation_prompt",
		},
		{
			name:       "arbitrary unsupported option",
			body:       `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": true, "user": "alice"}`,
			wantSubstr: "user",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("expected HTTP 400 Bad Request, got %d", resp.StatusCode)
			}
			b, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(b), tc.wantSubstr) {
				t.Errorf("expected error message to contain %q, got %s", tc.wantSubstr, string(b))
			}
		})
	}
}

func TestOpenAI_UnknownModelAnswers404(t *testing.T) {
	exec := newMockExecutor()
	ts, _ := setupTestRouter(t, exec)

	reqBody := `{"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "stream": true}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected HTTP 404 Not Found, got %d", resp.StatusCode)
	}
	b, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(b), "The model 'nonexistent-model' does not exist") {
		t.Errorf("expected 404 message to name model, got %s", string(b))
	}
}

func TestOpenAI_BenchOptionsAccepted(t *testing.T) {
	exec := newMockExecutor()
	exec.vocab = &pb.GetVocabResponse{
		Pieces:          [][]byte{[]byte(""), []byte("tok")},
		SpecialTokenIds: []int32{0},
	}
	exec.tokenStream = [][]int32{{1}}
	ts, _ := setupTestRouter(t, exec)

	// Test the exact request format vxpu bench sends
	reqBody := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"max_tokens": 400,
		"max_completion_tokens": 400,
		"min_tokens": 400,
		"ignore_eos": true,
		"stop": [],
		"temperature": 0,
		"top_p": 1,
		"stream": true,
		"stream_options": {"include_usage": true},
		"chat_template_kwargs": {"enable_thinking": false, "thinking": false, "thinking_mode": "disabled"}
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(b))
	}
}

func TestOpenAI_Cancellation(t *testing.T) {
	exec := newMockExecutor()
	exec.vocab = &pb.GetVocabResponse{
		Pieces: [][]byte{[]byte(""), []byte("tok")},
	}
	exec.tokenStream = [][]int32{{1}}
	exec.pauseStream = make(chan struct{})
	exec.tokenSent = make(chan struct{})

	ts, _ := setupTestRouter(t, exec)

	ctx, cancel := context.WithCancel(t.Context())
	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "stream": true}`
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	select {
	case <-exec.tokenSent:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for tokenSent")
	}

	cancel()
	_ = resp.Body.Close()
}

func TestOpenAI_ErrorDuringStream(t *testing.T) {
	exec := newMockExecutor()
	exec.vocab = &pb.GetVocabResponse{
		Pieces: [][]byte{[]byte(""), []byte("tok")},
	}
	exec.tokenStream = [][]int32{{1}}
	exec.streamErr = status.Error(codes.Internal, "simulated executor failure")

	ts, _ := setupTestRouter(t, exec)

	reqBody := `{"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "stream": true}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var seenError bool
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "simulated executor failure") {
			seenError = true
			break
		}
	}

	if !seenError {
		t.Errorf("expected stream to receive error event with failure message")
	}
}

func TestStreamDecoder_UnitMultiByteRuneBuffering(t *testing.T) {
	vocab := &pb.GetVocabResponse{
		Pieces: [][]byte{
			[]byte(""),
			{0xC3}, {0xA9}, // 'é'
			{0xE2}, {0x80}, {0x94}, // '—'
			{0xF0}, {0x9F}, {0x8C}, {0x9F}, // '🌟'
		},
		SpecialTokenIds: []int32{0},
	}
	d := newStreamDecoder(vocab)

	// 2-byte rune across 2 calls
	if s := d.decode([]int32{1}); s != "" {
		t.Errorf("expected empty string for partial 2-byte rune, got %q", s)
	}
	if s := d.decode([]int32{2}); s != "é" {
		t.Errorf("expected 'é' on second byte, got %q", s)
	}

	// 3-byte rune across 3 calls
	if s := d.decode([]int32{3}); s != "" {
		t.Errorf("expected empty string for byte 1 of 3, got %q", s)
	}
	if s := d.decode([]int32{4}); s != "" {
		t.Errorf("expected empty string for byte 2 of 3, got %q", s)
	}
	if s := d.decode([]int32{5}); s != "—" {
		t.Errorf("expected '—' on byte 3, got %q", s)
	}

	// 4-byte rune across 4 calls
	if s := d.decode([]int32{6}); s != "" {
		t.Errorf("expected empty string for byte 1 of 4, got %q", s)
	}
	if s := d.decode([]int32{7}); s != "" {
		t.Errorf("expected empty string for byte 2 of 4, got %q", s)
	}
	if s := d.decode([]int32{8}); s != "" {
		t.Errorf("expected empty string for byte 3 of 4, got %q", s)
	}
	if s := d.decode([]int32{9}); s != "🌟" {
		t.Errorf("expected '🌟' on byte 4, got %q", s)
	}

	// Flush empty
	if s := d.flush(); s != "" {
		t.Errorf("expected empty flush, got %q", s)
	}
}
