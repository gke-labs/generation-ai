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
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

var timeNow = time.Now

type chatRequest struct {
	Model              string         `json:"model"`
	Messages           []chatMessage  `json:"messages"`
	MaxTokens          int            `json:"max_tokens"`
	Temperature        float64        `json:"temperature"`
	TopP               float64        `json:"top_p"`
	Stream             bool           `json:"stream"`
	StreamOptions      map[string]any `json:"stream_options"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`
	MinTokens          *int           `json:"min_tokens,omitempty"`
	IgnoreEOS          *bool          `json:"ignore_eos,omitempty"`
	Stop               *[]string      `json:"stop,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type promptConfig struct {
	Prose struct {
		Base string `json:"base"`
	} `json:"prose"`
	Code struct {
		Tasks      []string `json:"tasks"`
		Tail       string   `json:"tail"`
		WarmupTask string   `json:"warmup_task"`
	} `json:"code"`
}

type streamTimings struct {
	StreamIndex, CompletionTokens int
	TStart, TFirst, TLast         time.Time
	TokensPerSec                  float64
	TTFT                          time.Duration
}

type waveMetrics struct {
	Set          string    `json:"set"`
	Streams      int       `json:"streams"`
	Aggregate    float64   `json:"aggregate_tok_s"`
	StreamTokSec []float64 `json:"stream_tok_s"`
	MedianTokSec float64   `json:"median_tok_s"`
	MedianTTFTMs float64   `json:"median_ttft_ms"`
}

// streamThroughput calculates tokens per second for one stream:
// (completion_tokens - 1) / (tLast - tFirst).
// Prefill and the first token are excluded.
func streamThroughput(completionTokens int, tFirst, tLast time.Time) (float64, error) {
	if completionTokens <= 1 {
		return 0, fmt.Errorf("insufficient completion tokens (%d)", completionTokens)
	}
	dur := tLast.Sub(tFirst).Seconds()
	if dur <= 0 {
		return 0, fmt.Errorf("non-positive stream duration (%v)", dur)
	}
	return float64(completionTokens-1) / dur, nil
}

// waveThroughput calculates aggregate tokens per second across a wave of streams:
// sum over successful streams of (completion_tokens - 1), divided by
// (latest tLast - earliest tFirst) across the wave.
func waveThroughput(streams []streamTimings) (float64, error) {
	if len(streams) == 0 {
		return 0, errors.New("empty wave")
	}
	var sumTokens int
	var earliestFirst, latestLast time.Time
	for i, s := range streams {
		if s.CompletionTokens <= 1 {
			return 0, fmt.Errorf("stream %d: insufficient completion tokens (%d)", s.StreamIndex, s.CompletionTokens)
		}
		sumTokens += s.CompletionTokens - 1
		if i == 0 || s.TFirst.Before(earliestFirst) {
			earliestFirst = s.TFirst
		}
		if i == 0 || s.TLast.After(latestLast) {
			latestLast = s.TLast
		}
	}
	dur := latestLast.Sub(earliestFirst).Seconds()
	if dur <= 0 {
		return 0, fmt.Errorf("non-positive wave span (%v)", dur)
	}
	return float64(sumTokens) / dur, nil
}

func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2.0
}

func loadPrompts(path string) (*promptConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading prompt file: %w", err)
	}
	var cfg promptConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing prompt file: %w", err)
	}
	return &cfg, nil
}

func validatePrompts(cfg *promptConfig, sets []string, levels []int) error {
	maxLevel := 0
	for _, l := range levels {
		if l > maxLevel {
			maxLevel = l
		}
	}
	for _, s := range sets {
		if s == "prose" && strings.TrimSpace(cfg.Prose.Base) == "" {
			return errors.New("prompt file missing prose.base")
		}
		if s == "code" {
			if strings.TrimSpace(cfg.Code.Tail) == "" || strings.TrimSpace(cfg.Code.WarmupTask) == "" {
				return errors.New("prompt file missing code.tail or code.warmup_task")
			}
			if len(cfg.Code.Tasks) < maxLevel {
				return fmt.Errorf("code prompts require at least %d tasks for level %d, found %d", maxLevel, maxLevel, len(cfg.Code.Tasks))
			}
		}
	}
	return nil
}

func buildPrompt(cfg *promptConfig, set string, streamIdx, concurrency int) string {
	if set == "prose" {
		if concurrency <= 1 {
			return cfg.Prose.Base
		}
		return fmt.Sprintf("%s (stream %d/%d)", cfg.Prose.Base, streamIdx+1, concurrency)
	}
	return cfg.Code.Tasks[streamIdx] + "\n" + cfg.Code.Tail
}

func buildWarmupPrompt(cfg *promptConfig, set string) string {
	if set == "prose" {
		return cfg.Prose.Base
	}
	return cfg.Code.WarmupTask + "\n" + cfg.Code.Tail
}

func buildChatRequest(model, prompt string, isWarmup bool) chatRequest {
	req := chatRequest{
		Model:              model,
		Messages:           []chatMessage{{Role: "user", Content: prompt}},
		MaxTokens:          400,
		Temperature:        0,
		TopP:               1,
		Stream:             true,
		StreamOptions:      map[string]any{"include_usage": true},
		ChatTemplateKwargs: map[string]any{"enable_thinking": false, "thinking": false, "thinking_mode": "disabled"},
	}
	if isWarmup {
		req.MaxTokens = 32
	} else {
		minTok := 400
		emptyStop := []string{}
		req.MinTokens, req.Stop = &minTok, &emptyStop
		ignore := true
		req.IgnoreEOS = &ignore
	}
	return req
}

func parseStream(ctx context.Context, body io.ReadCloser, streamIdx, maxTokens int, isWarmup bool, tStart time.Time) (streamTimings, error) {
	defer func() {
		if err := body.Close(); err != nil {
			klog.FromContext(ctx).Error(err, "closing stream body")
		}
	}()
	scanner := bufio.NewScanner(body)
	var tFirst, tLast time.Time
	var completionTokens int
	var seenDone bool

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			seenDone = true
			break
		}
		if payload == "" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct{ Content string }
			}
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			}
			Error *struct{ Message string }
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return streamTimings{}, fmt.Errorf("stream %d: parsing event JSON: %w", streamIdx, err)
		}
		if chunk.Error != nil && chunk.Error.Message != "" {
			return streamTimings{}, fmt.Errorf("stream %d: server error: %s", streamIdx, chunk.Error.Message)
		}
		hasContent := false
		for _, c := range chunk.Choices {
			if c.Delta.Content != "" {
				hasContent = true
				break
			}
		}
		if hasContent {
			arrivalTime := timeNow()
			if tFirst.IsZero() {
				tFirst = arrivalTime
			}
			tLast = arrivalTime
		}
		if chunk.Usage != nil && chunk.Usage.CompletionTokens > 0 {
			completionTokens = chunk.Usage.CompletionTokens
		}
	}
	if err := scanner.Err(); err != nil {
		return streamTimings{}, fmt.Errorf("stream %d: reading stream: %w", streamIdx, err)
	}
	if !seenDone {
		return streamTimings{}, fmt.Errorf("stream %d ended early: missing [DONE]", streamIdx)
	}
	if completionTokens == 0 {
		return streamTimings{}, fmt.Errorf("stream %d: server omitted usage in stream", streamIdx)
	}
	if isWarmup {
		return streamTimings{StreamIndex: streamIdx, CompletionTokens: completionTokens}, nil
	}
	if completionTokens < maxTokens {
		return streamTimings{}, fmt.Errorf("stream %d ended early: got %d completion tokens, want %d", streamIdx, completionTokens, maxTokens)
	}
	if tFirst.IsZero() {
		return streamTimings{}, fmt.Errorf("stream %d ended early: no content tokens received", streamIdx)
	}
	tokPerSec, err := streamThroughput(completionTokens, tFirst, tLast)
	if err != nil {
		return streamTimings{}, fmt.Errorf("stream %d: %w", streamIdx, err)
	}
	return streamTimings{
		StreamIndex:      streamIdx,
		TStart:           tStart,
		TFirst:           tFirst,
		TLast:            tLast,
		CompletionTokens: completionTokens,
		TokensPerSec:     tokPerSec,
		TTFT:             tFirst.Sub(tStart),
	}, nil
}

func runStream(ctx context.Context, client *http.Client, url, model, prompt string, streamIdx int, isWarmup bool) (streamTimings, error) {
	reqBody := buildChatRequest(model, prompt, isWarmup)
	data, err := json.Marshal(reqBody)
	if err != nil {
		return streamTimings{}, fmt.Errorf("stream %d: encoding request: %w", streamIdx, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return streamTimings{}, fmt.Errorf("stream %d: creating request: %w", streamIdx, err)
	}
	req.Header.Set("Content-Type", "application/json")

	tStart := timeNow()
	resp, err := client.Do(req)
	if err != nil {
		return streamTimings{}, fmt.Errorf("stream %d: request failed: %w", streamIdx, err)
	}
	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(io.LimitReader(resp.Body, 2048))
		if err != nil {
			body = []byte("(unreadable body)")
		}
		if closeErr := resp.Body.Close(); closeErr != nil {
			klog.FromContext(ctx).Error(closeErr, "closing error response body")
		}
		return streamTimings{}, fmt.Errorf("stream %d: server returned status %d: %s", streamIdx, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return parseStream(ctx, resp.Body, streamIdx, reqBody.MaxTokens, isWarmup, tStart)
}

func runWave(ctx context.Context, client *http.Client, url, model string, prompts []string, timeout time.Duration) ([]streamTimings, error) {
	n := len(prompts)
	results := make([]streamTimings, n)
	errCh := make(chan error, n)
	ready := make(chan struct{})
	var wg sync.WaitGroup

	waveCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-ready
			res, err := runStream(waveCtx, client, url, model, prompts[idx], idx, false)
			if err != nil {
				errCh <- err
				return
			}
			results[idx] = res
		}(i)
	}
	close(ready)
	wg.Wait()
	close(errCh)

	if len(errCh) > 0 {
		return nil, <-errCh
	}
	return results, nil
}

type benchOptions struct {
	Endpoint   string
	Model      string
	Prompts    *promptConfig
	Levels     []int
	Sets       []string
	Timeout    time.Duration
	AsJSON     bool
	Out        io.Writer
	HTTPClient *http.Client
}

func parseLevels(s string) ([]int, error) {
	var levels []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		val, err := strconv.Atoi(p)
		if err != nil || val <= 0 {
			return nil, fmt.Errorf("invalid level %q", p)
		}
		levels = append(levels, val)
	}
	if len(levels) == 0 {
		return nil, errors.New("no concurrency levels specified")
	}
	return levels, nil
}

func runBench(ctx context.Context, opts benchOptions) ([]waveMetrics, error) {
	log := klog.FromContext(ctx)
	if err := validatePrompts(opts.Prompts, opts.Sets, opts.Levels); err != nil {
		return nil, err
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 360 * time.Second
	}
	url := strings.TrimRight(opts.Endpoint, "/") + "/v1/chat/completions"
	var allMetrics []waveMetrics

	for _, set := range opts.Sets {
		// Warm-up request with max_tokens 32. Failure logs an error and does not fail the job.
		warmupPrompt := buildWarmupPrompt(opts.Prompts, set)
		warmupCtx, cancelWarmup := context.WithTimeout(ctx, timeout)
		_, warmupErr := runStream(warmupCtx, opts.HTTPClient, url, opts.Model, warmupPrompt, 0, true)
		cancelWarmup()
		if warmupErr != nil {
			log.Error(warmupErr, "warmup request failed, continuing", "set", set)
		}

		for _, concurrency := range opts.Levels {
			prompts := make([]string, concurrency)
			for i := 0; i < concurrency; i++ {
				prompts[i] = buildPrompt(opts.Prompts, set, i, concurrency)
			}

			timings, err := runWave(ctx, opts.HTTPClient, url, opts.Model, prompts, timeout)
			if err != nil {
				return nil, fmt.Errorf("%s level %d (%d streams) failed: %w", set, concurrency, concurrency, err)
			}

			aggTokSec, err := waveThroughput(timings)
			if err != nil {
				return nil, err
			}
			streamTokSec := make([]float64, concurrency)
			streamTTFTs := make([]float64, concurrency)
			for i, t := range timings {
				streamTokSec[i] = t.TokensPerSec
				streamTTFTs[i] = float64(t.TTFT.Microseconds()) / 1000.0
			}
			m := waveMetrics{
				Set:          set,
				Streams:      concurrency,
				Aggregate:    aggTokSec,
				StreamTokSec: streamTokSec,
				MedianTokSec: median(streamTokSec),
				MedianTTFTMs: median(streamTTFTs),
			}
			allMetrics = append(allMetrics, m)

			if opts.AsJSON {
				line, err := json.Marshal(m)
				if err != nil {
					return nil, fmt.Errorf("marshaling metrics JSON: %w", err)
				}
				if _, err := fmt.Fprintf(opts.Out, "%s\n", line); err != nil {
					return nil, err
				}
			} else {
				if _, err := fmt.Fprintf(opts.Out, "%s level %d (%d streams): aggregate %.1f tok/s, median stream %.1f tok/s, median TTFT %.2f ms\n",
					strings.ToUpper(set), concurrency, concurrency, m.Aggregate, m.MedianTokSec, m.MedianTTFTMs); err != nil {
					return nil, err
				}
				for _, s := range timings {
					ttftMs := float64(s.TTFT.Microseconds()) / 1000.0
					if _, err := fmt.Fprintf(opts.Out, "  stream %d: %.1f tok/s, TTFT %.2f ms\n", s.StreamIndex, s.TokensPerSec, ttftMs); err != nil {
						return nil, err
					}
				}
			}
		}
	}
	return allMetrics, nil
}

// cmdBench runs the benchmark against an OpenAI-compatible endpoint.
func cmdBench(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("bench", flag.ContinueOnError)
	endpoint := flags.String("endpoint", "http://localhost:8000", "base URL of OpenAI-compatible endpoint")
	model := flags.String("model", "", "model name to request")
	promptsPath := flags.String("prompts", "", "path to prompt JSON file")
	setFlag := flags.String("set", "all", "prompt set to run: prose, code, or all")
	levelsFlag := flags.String("levels", "1,2,4,8,16", "comma-separated stream concurrency levels")
	timeout := flags.Duration("timeout", 360*time.Second, "timeout for each stream and wave")
	asJSON := flags.Bool("json", false, "output results as JSON lines")

	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *model == "" {
		return errors.New("vxpu bench requires --model")
	}
	if *promptsPath == "" {
		return errors.New("vxpu bench requires --prompts")
	}
	levels, err := parseLevels(*levelsFlag)
	if err != nil {
		return err
	}
	var sets []string
	switch strings.ToLower(*setFlag) {
	case "all":
		sets = []string{"prose", "code"}
	case "prose":
		sets = []string{"prose"}
	case "code":
		sets = []string{"code"}
	default:
		return fmt.Errorf("unknown --set %q (must be prose, code, or all)", *setFlag)
	}

	cfg, err := loadPrompts(*promptsPath)
	if err != nil {
		return err
	}

	_, err = runBench(ctx, benchOptions{
		Endpoint:   *endpoint,
		Model:      *model,
		Prompts:    cfg,
		Levels:     levels,
		Sets:       sets,
		Timeout:    *timeout,
		AsJSON:     *asJSON,
		Out:        os.Stdout,
		HTTPClient: &http.Client{},
	})
	return err
}
