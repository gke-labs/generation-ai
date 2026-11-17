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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func sampleTestPrompts() *promptConfig {
	var cfg promptConfig
	cfg.Prose.Base = "Describe the water cycle and weather patterns across different climates."
	cfg.Code.Tasks = make([]string, 16)
	for i := 0; i < 16; i++ {
		cfg.Code.Tasks[i] = fmt.Sprintf("task_%02d\ndef task_%02d(): pass", i, i)
	}
	cfg.Code.Tail = "Return runnable Python with no markdown fences."
	cfg.Code.WarmupTask = "echo_value\ndef echo_value(x): return x"
	return &cfg
}

func TestFormulas(t *testing.T) {
	// Worked example from published data:
	// 400 completion tokens, first token at 226.45 ms, last at 1811.68 ms.
	// Throughput = 399 / (1811.68 - 226.45) ms = 399 / 1.58523 s = 251.70167... tok/s.
	base := time.Date(2026, 11, 16, 0, 0, 0, 0, time.UTC)
	tFirst := base.Add(226450 * time.Microsecond)
	tLast := base.Add(1811680 * time.Microsecond)

	tokPerSec, err := streamThroughput(400, tFirst, tLast)
	if err != nil {
		t.Fatalf("streamThroughput failed: %v", err)
	}
	expected := 399.0 / (1.81168 - 0.22645)
	if math.Abs(tokPerSec-expected) > 1e-9 {
		t.Errorf("streamThroughput = %f, want %f", tokPerSec, expected)
	}

	// Wave throughput for 2 streams:
	// stream 0: tFirst=100ms, tLast=1100ms, 400 tokens -> 399 tokens
	// stream 1: tFirst=200ms, tLast=1200ms, 400 tokens -> 399 tokens
	// sum tokens = 798
	// wave span = 1200ms - 100ms = 1100ms = 1.1s
	// aggregate = 798 / 1.1 = 725.454545...
	stream0 := streamTimings{
		StreamIndex:      0,
		TFirst:           base.Add(100 * time.Millisecond),
		TLast:            base.Add(1100 * time.Millisecond),
		CompletionTokens: 400,
	}
	stream1 := streamTimings{
		StreamIndex:      1,
		TFirst:           base.Add(200 * time.Millisecond),
		TLast:            base.Add(1200 * time.Millisecond),
		CompletionTokens: 400,
	}
	agg, err := waveThroughput([]streamTimings{stream0, stream1})
	if err != nil {
		t.Fatalf("waveThroughput failed: %v", err)
	}
	expectedAgg := 798.0 / 1.1
	if math.Abs(agg-expectedAgg) > 1e-9 {
		t.Errorf("waveThroughput = %f, want %f", agg, expectedAgg)
	}

	if _, err := streamThroughput(1, tFirst, tLast); err == nil {
		t.Errorf("streamThroughput with 1 token expected error, got nil")
	}
	if _, err := streamThroughput(400, tLast, tFirst); err == nil {
		t.Errorf("streamThroughput with inverted times expected error, got nil")
	}

	odd := []float64{300.0, 100.0, 200.0}
	if m := median(odd); m != 200.0 {
		t.Errorf("median(odd) = %f, want 200.0", m)
	}
	even := []float64{100.0, 300.0, 200.0, 400.0}
	if m := median(even); m != 250.0 {
		t.Errorf("median(even) = %f, want 250.0", m)
	}
}

func TestPromptConstruction(t *testing.T) {
	cfg := sampleTestPrompts()

	p1 := buildPrompt(cfg, "prose", 0, 1)
	if p1 != "Describe the water cycle and weather patterns across different climates." {
		t.Errorf("prose 1-stream prompt = %q, want base alone", p1)
	}
	p2_0 := buildPrompt(cfg, "prose", 0, 2)
	if p2_0 != "Describe the water cycle and weather patterns across different climates. (stream 1/2)" {
		t.Errorf("prose 2-stream stream 0 prompt = %q", p2_0)
	}
	p2_1 := buildPrompt(cfg, "prose", 1, 2)
	if p2_1 != "Describe the water cycle and weather patterns across different climates. (stream 2/2)" {
		t.Errorf("prose 2-stream stream 1 prompt = %q", p2_1)
	}

	c0 := buildPrompt(cfg, "code", 0, 1)
	if c0 != "task_00\ndef task_00(): pass\nReturn runnable Python with no markdown fences." {
		t.Errorf("code stream 0 prompt = %q", c0)
	}

	wp := buildWarmupPrompt(cfg, "prose")
	if wp != "Describe the water cycle and weather patterns across different climates." {
		t.Errorf("warmup prose prompt = %q", wp)
	}
	wc := buildWarmupPrompt(cfg, "code")
	if wc != "echo_value\ndef echo_value(x): return x\nReturn runnable Python with no markdown fences." {
		t.Errorf("warmup code prompt = %q", wc)
	}
}

func TestPromptValidation(t *testing.T) {
	// Prose-only config without code tasks succeeds for --set prose
	var proseOnly promptConfig
	proseOnly.Prose.Base = "Some prose prompt"
	if err := validatePrompts(&proseOnly, []string{"prose"}, []int{1, 2, 4}); err != nil {
		t.Errorf("prose-only validation failed: %v", err)
	}

	// Code-only config without prose.base succeeds for --set code
	var codeOnly promptConfig
	codeOnly.Code.Tasks = make([]string, 16)
	codeOnly.Code.Tail = "tail"
	codeOnly.Code.WarmupTask = "warmup"
	if err := validatePrompts(&codeOnly, []string{"code"}, []int{1, 2, 4, 8, 16}); err != nil {
		t.Errorf("code-only validation failed: %v", err)
	}

	// Code level exceeding task count fails immediately and names the level and count
	err := validatePrompts(&codeOnly, []string{"code"}, []int{1, 32})
	if err == nil {
		t.Fatalf("expected error for level 32 with 16 tasks, got nil")
	}
	if !strings.Contains(err.Error(), "level 32") || !strings.Contains(err.Error(), "16") {
		t.Errorf("error %q should name level 32 and task count 16", err)
	}
}

type capturedRequest struct {
	Method  string
	Path    string
	Body    chatRequest
	RawBody map[string]any
}

func setupStubServer(t *testing.T, setClock func(time.Time), tok1Read, tok400Read chan struct{}, tFirst, tLast time.Time, expectedTokens int, failMode string) (*httptest.Server, *[]capturedRequest) {
	var mu sync.Mutex
	var captured []capturedRequest

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var raw map[string]any
		if err := json.Unmarshal(bodyBytes, &raw); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var req chatRequest
		if err := json.Unmarshal(bodyBytes, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		captured = append(captured, capturedRequest{Method: r.Method, Path: r.URL.Path, Body: req, RawBody: raw})
		reqCount := len(captured)
		mu.Unlock()

		// Warmup request handling: sends 10 tokens (< 32), which must succeed
		if req.MaxTokens == 32 {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, ": keep-alive comment\n")
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"warmup\"}}]}\n\n")
			fmt.Fprintf(w, "data: {\"usage\":{\"completion_tokens\":10}}\n\n")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return
		}

		if failMode == "status_500" {
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		if failMode == "stream_early" {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"early\"}}]}\n\n")
			fmt.Fprintf(w, "data: {\"usage\":{\"completion_tokens\":200}}\n\n")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return
		}
		if failMode == "omit_usage" {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return
		}
		if failMode == "fail_stream_1" && reqCount > 2 {
			http.Error(w, "stream 1 failed", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		// Comments should be ignored by the SSE client
		fmt.Fprintf(w, ": server comment\n")

		if setClock != nil {
			setClock(tFirst)
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok1\"}}]}\n\n")
		w.(http.Flusher).Flush()

		if tok1Read != nil {
			<-tok1Read
		}

		if setClock != nil {
			setClock(tLast)
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok400\"}}]}\n\n")
		w.(http.Flusher).Flush()

		if tok400Read != nil {
			<-tok400Read
		}

		fmt.Fprintf(w, "data: {\"usage\":{\"completion_tokens\":%d}}\n\n", expectedTokens)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	})

	server := httptest.NewServer(handler)
	return server, &captured
}

func TestBenchAgainstStub_KnownTimesAndFormulas(t *testing.T) {
	base := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	tStart := base
	tFirst := base.Add(226450 * time.Microsecond)
	tLast := base.Add(1811680 * time.Microsecond)

	tok1Read := make(chan struct{})
	tok400Read := make(chan struct{})

	var clockMu sync.Mutex
	currentClock := tStart
	setClock := func(tm time.Time) {
		clockMu.Lock()
		defer clockMu.Unlock()
		currentClock = tm
	}

	server, captured := setupStubServer(t, setClock, tok1Read, tok400Read, tFirst, tLast, 400, "")
	defer server.Close()

	oldTimeNow := timeNow
	timeNow = func() time.Time {
		clockMu.Lock()
		tm := currentClock
		clockMu.Unlock()
		if tm.Equal(tFirst) {
			select {
			case tok1Read <- struct{}{}:
			default:
			}
		} else if tm.Equal(tLast) {
			select {
			case tok400Read <- struct{}{}:
			default:
			}
		}
		return tm
	}
	defer func() { timeNow = oldTimeNow }()

	prompts := sampleTestPrompts()
	var buf bytes.Buffer
	ctx := t.Context()
	metrics, err := runBench(ctx, benchOptions{
		Endpoint:   server.URL,
		Model:      "test-model",
		Prompts:    prompts,
		Levels:     []int{1},
		Sets:       []string{"prose"},
		Timeout:    5 * time.Second,
		AsJSON:     true,
		Out:        &buf,
		HTTPClient: &http.Client{},
	})
	if err != nil {
		t.Fatalf("runBench failed: %v", err)
	}

	if len(metrics) != 1 {
		t.Fatalf("expected 1 metric wave, got %d", len(metrics))
	}
	m := metrics[0]

	expectedThroughput := 399.0 / (1.81168 - 0.22645)
	if math.Abs(m.Aggregate-expectedThroughput) > 1e-6 {
		t.Errorf("m.Aggregate = %f, want %f", m.Aggregate, expectedThroughput)
	}
	if len(m.StreamTokSec) != 1 || math.Abs(m.StreamTokSec[0]-expectedThroughput) > 1e-6 {
		t.Errorf("m.StreamTokSec[0] = %f, want %f", m.StreamTokSec[0], expectedThroughput)
	}
	if math.Abs(m.MedianTokSec-expectedThroughput) > 1e-6 {
		t.Errorf("m.MedianTokSec = %f, want %f", m.MedianTokSec, expectedThroughput)
	}
	expectedTTFTMs := 226.45
	if math.Abs(m.MedianTTFTMs-expectedTTFTMs) > 1e-3 {
		t.Errorf("m.MedianTTFTMs = %f, want %f", m.MedianTTFTMs, expectedTTFTMs)
	}

	var parsedJSON waveMetrics
	if err := json.Unmarshal(buf.Bytes(), &parsedJSON); err != nil {
		t.Fatalf("unmarshaling output JSON: %v", err)
	}
	if math.Abs(parsedJSON.Aggregate-expectedThroughput) > 1e-6 {
		t.Errorf("parsed JSON aggregate = %f, want %f", parsedJSON.Aggregate, expectedThroughput)
	}

	if len(*captured) < 2 {
		t.Fatalf("expected at least 2 requests, got %d", len(*captured))
	}

	// Warmup request should have exact 8 keys, with no min_tokens, ignore_eos, or stop
	warmupRaw := (*captured)[0].RawBody
	wantWarmupKeys := []string{"chat_template_kwargs", "max_tokens", "messages", "model", "stream", "stream_options", "temperature", "top_p"}
	var gotWarmupKeys []string
	for k := range warmupRaw {
		gotWarmupKeys = append(gotWarmupKeys, k)
	}
	sort.Strings(gotWarmupKeys)
	if strings.Join(gotWarmupKeys, ",") != strings.Join(wantWarmupKeys, ",") {
		t.Errorf("warmup body keys = %v, want %v", gotWarmupKeys, wantWarmupKeys)
	}
	if warmupRaw["max_tokens"] != float64(32) {
		t.Errorf("warmup max_tokens = %v, want 32", warmupRaw["max_tokens"])
	}

	// Measured request should match published method key-for-key, including "stop": []
	measuredRaw := (*captured)[1].RawBody
	wantMeasuredKeys := []string{"chat_template_kwargs", "ignore_eos", "max_tokens", "messages", "min_tokens", "model", "stop", "stream", "stream_options", "temperature", "top_p"}
	var gotMeasuredKeys []string
	for k := range measuredRaw {
		gotMeasuredKeys = append(gotMeasuredKeys, k)
	}
	sort.Strings(gotMeasuredKeys)
	if strings.Join(gotMeasuredKeys, ",") != strings.Join(wantMeasuredKeys, ",") {
		t.Errorf("measured body keys = %v, want %v", gotMeasuredKeys, wantMeasuredKeys)
	}
	if measuredRaw["model"] != "test-model" {
		t.Errorf("measured model = %v, want test-model", measuredRaw["model"])
	}
	if measuredRaw["max_tokens"] != float64(400) || measuredRaw["min_tokens"] != float64(400) {
		t.Errorf("measured tokens = %v/%v, want 400/400", measuredRaw["max_tokens"], measuredRaw["min_tokens"])
	}
	if measuredRaw["ignore_eos"] != true {
		t.Errorf("measured ignore_eos = %v, want true", measuredRaw["ignore_eos"])
	}
	stopSlice, ok := measuredRaw["stop"].([]any)
	if !ok || len(stopSlice) != 0 {
		t.Errorf("measured stop = %v (type %T), want empty array []", measuredRaw["stop"], measuredRaw["stop"])
	}
	if measuredRaw["temperature"] != float64(0) || measuredRaw["top_p"] != float64(1) {
		t.Errorf("measured temp/top_p = %v/%v, want 0/1", measuredRaw["temperature"], measuredRaw["top_p"])
	}
	if measuredRaw["stream"] != true {
		t.Errorf("measured stream = %v, want true", measuredRaw["stream"])
	}
	ctk, ok := measuredRaw["chat_template_kwargs"].(map[string]any)
	if !ok || ctk["thinking_mode"] != "disabled" {
		t.Errorf("chat_template_kwargs = %v, want thinking_mode: disabled", measuredRaw["chat_template_kwargs"])
	}
}

func TestBenchRequestsAndPrompts_MultiStream(t *testing.T) {
	base := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	server, captured := setupStubServer(t, nil, nil, nil, base.Add(100*time.Millisecond), base.Add(1100*time.Millisecond), 400, "")
	defer server.Close()

	prompts := sampleTestPrompts()
	dir := t.TempDir()
	promptsPath := filepath.Join(dir, "prompts.json")
	pData, _ := json.Marshal(prompts)
	_ = os.WriteFile(promptsPath, pData, 0644)

	ctx := t.Context()
	err := cmdBench(ctx, []string{
		"--endpoint", server.URL,
		"--model", "test-model",
		"--prompts", promptsPath,
		"--levels", "1,2",
		"--set", "all",
	})
	if err != nil {
		t.Fatalf("cmdBench failed: %v", err)
	}

	var prosePrompts []string
	var codePrompts []string
	for _, c := range *captured {
		content := c.Body.Messages[0].Content
		if strings.HasPrefix(content, "Describe") {
			prosePrompts = append(prosePrompts, content)
		} else {
			codePrompts = append(codePrompts, content)
		}
	}

	if len(prosePrompts) != 4 {
		t.Errorf("got %d prose prompts, want 4: %v", len(prosePrompts), prosePrompts)
	} else {
		if prosePrompts[0] != prompts.Prose.Base || prosePrompts[1] != prompts.Prose.Base {
			t.Errorf("unexpected prose prompt warmup/lvl1: %v", prosePrompts[:2])
		}
		p2_found1 := prosePrompts[2] == prompts.Prose.Base+" (stream 1/2)" || prosePrompts[3] == prompts.Prose.Base+" (stream 1/2)"
		p2_found2 := prosePrompts[2] == prompts.Prose.Base+" (stream 2/2)" || prosePrompts[3] == prompts.Prose.Base+" (stream 2/2)"
		if !p2_found1 || !p2_found2 {
			t.Errorf("prose lvl 2 prompts mismatch: %v", prosePrompts[2:])
		}
	}

	if len(codePrompts) != 4 {
		t.Errorf("got %d code prompts, want 4: %v", len(codePrompts), codePrompts)
	} else {
		if !strings.HasPrefix(codePrompts[0], "echo_value") {
			t.Errorf("code warmup prompt = %q", codePrompts[0])
		}
		if !strings.HasPrefix(codePrompts[1], "task_00") {
			t.Errorf("code lvl 1 prompt = %q", codePrompts[1])
		}
	}
}

func TestBenchFailures_NamedStreamNoNumber(t *testing.T) {
	base := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	prompts := sampleTestPrompts()
	ctx := t.Context()

	// 1. Server ends stream early (200 tokens instead of 400)
	{
		server, _ := setupStubServer(t, nil, nil, nil, base.Add(100*time.Millisecond), base.Add(1100*time.Millisecond), 400, "stream_early")
		var out bytes.Buffer
		metrics, err := runBench(ctx, benchOptions{
			Endpoint:   server.URL,
			Model:      "test-model",
			Prompts:    prompts,
			Levels:     []int{1},
			Sets:       []string{"prose"},
			Out:        &out,
			HTTPClient: &http.Client{},
		})
		server.Close()
		if err == nil {
			t.Fatalf("expected error for stream ending early, got nil")
		}
		if !strings.Contains(err.Error(), "stream 0") || !strings.Contains(err.Error(), "ended early") {
			t.Errorf("error %q should name stream 0 and say ended early", err)
		}
		if len(metrics) != 0 || out.Len() != 0 {
			t.Errorf("expected no metrics/numbers on failure, got %v / %q", metrics, out.String())
		}
	}

	// 2. Server omits usage
	{
		server, _ := setupStubServer(t, nil, nil, nil, base.Add(100*time.Millisecond), base.Add(1100*time.Millisecond), 400, "omit_usage")
		var out bytes.Buffer
		metrics, err := runBench(ctx, benchOptions{
			Endpoint:   server.URL,
			Model:      "test-model",
			Prompts:    prompts,
			Levels:     []int{1},
			Sets:       []string{"prose"},
			Out:        &out,
			HTTPClient: &http.Client{},
		})
		server.Close()
		if err == nil {
			t.Fatalf("expected error for omitted usage, got nil")
		}
		if !strings.Contains(err.Error(), "stream 0") || !strings.Contains(err.Error(), "omitted usage") {
			t.Errorf("error %q should name stream 0 and mention omitted usage", err)
		}
		if len(metrics) != 0 || out.Len() != 0 {
			t.Errorf("expected no metrics/numbers on failure, got %v / %q", metrics, out.String())
		}
	}

	// 3. Server returns error (HTTP 500)
	{
		server, _ := setupStubServer(t, nil, nil, nil, base.Add(100*time.Millisecond), base.Add(1100*time.Millisecond), 400, "status_500")
		var out bytes.Buffer
		metrics, err := runBench(ctx, benchOptions{
			Endpoint:   server.URL,
			Model:      "test-model",
			Prompts:    prompts,
			Levels:     []int{1},
			Sets:       []string{"prose"},
			Out:        &out,
			HTTPClient: &http.Client{},
		})
		server.Close()
		if err == nil {
			t.Fatalf("expected error for status 500, got nil")
		}
		if !strings.Contains(err.Error(), "stream 0") || !strings.Contains(err.Error(), "status 500") {
			t.Errorf("error %q should name stream 0 and mention status 500", err)
		}
		if len(metrics) != 0 || out.Len() != 0 {
			t.Errorf("expected no metrics/numbers on failure, got %v / %q", metrics, out.String())
		}
	}

	// 4. Stream 1 of 2 fails
	{
		server, _ := setupStubServer(t, nil, nil, nil, base.Add(100*time.Millisecond), base.Add(1100*time.Millisecond), 400, "fail_stream_1")
		var out bytes.Buffer
		metrics, err := runBench(ctx, benchOptions{
			Endpoint:   server.URL,
			Model:      "test-model",
			Prompts:    prompts,
			Levels:     []int{2},
			Sets:       []string{"prose"},
			Out:        &out,
			HTTPClient: &http.Client{},
		})
		server.Close()
		if err == nil {
			t.Fatalf("expected error when stream 1 fails, got nil")
		}
		if !strings.Contains(err.Error(), "stream 1") {
			t.Errorf("error %q should name stream 1", err)
		}
		if len(metrics) != 0 || out.Len() != 0 {
			t.Errorf("expected no metrics/numbers on failure, got %v / %q", metrics, out.String())
		}
	}
}
