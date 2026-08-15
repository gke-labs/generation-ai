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

// ask: inference against an exported model artifact with zero PyTorch
// on the client. Ships manifest + binding + weightless graphs to a
// ModelExecutor over gRPC (auto port-forwarding to the pod), opens a
// session, and asks a question. The executor rehydrates weights from
// content-addressed storage on its side; this binary never touches
// them.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/gke-labs/generation-ai/experiments/model-manifest/pkg/api/v1alpha1"
)

const maxMsgSize = 128 * 1024 * 1024

func main() {
	pod := flag.String("pod", "manifest-executor-grpc", "executor pod to port-forward to")
	addr := flag.String("addr", "", "executor address (skips port-forward)")
	manifest := flag.String("manifest", "HuggingFaceTB--SmolLM2-135M-Instruct.manifest.json", "manifest json")
	binding := flag.String("binding", "cached.binding.json", "binding json")
	prefill := flag.String("prefill", "prefill.pt2", "prefill graph")
	decode := flag.String("decode", "decode.pt2", "decode graph")
	prompt := flag.String("prompt", "Is the sky blue?", "prompt")
	maxNewTokens := flag.Int("max-new-tokens", 96, "max new tokens")
	flag.Parse()

	if *addr == "" {
		forwarded, stop, err := portForward(*pod)
		if err != nil {
			log.Fatalf("port-forward: %v", err)
		}
		defer stop()
		*addr = forwarded
	}

	conn, err := grpc.NewClient(*addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(maxMsgSize),
			grpc.MaxCallRecvMsgSize(maxMsgSize),
		))
	if err != nil {
		log.Fatalf("dial %s: %v", *addr, err)
	}
	defer conn.Close()
	client := pb.NewModelExecutorClient(conn)

	manifestJSON := mustRead(*manifest)
	bindingJSON := mustRead(*binding)
	prefillGraph := mustRead(*prefill)
	decodeGraph := mustRead(*decode)
	fmt.Printf("shipping artifact: %d KB manifest, %d KB binding, %d MB graphs\n",
		len(manifestJSON)/1024, len(bindingJSON)/1024,
		(len(prefillGraph)+len(decodeGraph))/(1024*1024))

	// LoadModel triggers weight rehydration executor-side (range
	// requests into its CAS): give it time on a cold cache.
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	start := time.Now()
	if _, err := client.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: string(manifestJSON),
		BindingJson:  string(bindingJSON),
		PrefillGraph: prefillGraph,
		DecodeGraph:  decodeGraph,
	}); err != nil {
		log.Fatalf("LoadModel: %v", err)
	}
	fmt.Printf("model loaded in %.1fs\n", time.Since(start).Seconds())

	session, err := client.NewSession(ctx, &pb.NewSessionRequest{})
	if err != nil {
		log.Fatalf("NewSession: %v", err)
	}

	ask := func(text string) {
		start := time.Now()
		reply, err := client.Chat(ctx, &pb.ChatRequest{
			SessionId:    session.SessionId,
			Text:         text,
			MaxNewTokens: int32(*maxNewTokens),
		})
		if err != nil {
			log.Fatalf("Chat: %v", err)
		}
		fmt.Printf("\n>>> %s\n%s\n\n", text, reply.Text)
		fmt.Printf("(%d tokens, %.1f ms/token on the executor, %.1fs round trip)\n",
			reply.Generated, reply.MsPerToken, time.Since(start).Seconds())
	}
	ask(*prompt)
	// Same session: exercises the KV cache (incremental prefill) and
	// shows steady-state speed once the decode graph is compiled.
	ask("Summarize your answer in one short sentence.")
}

func mustRead(path string) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	return data
}

// portForward starts `kubectl port-forward` on an ephemeral local port
// and returns the local address once the tunnel is ready.
func portForward(pod string) (string, func(), error) {
	cmd := exec.Command("kubectl", "port-forward", "pod/"+pod, ":50051")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return "", nil, err
	}
	stop := func() { _ = cmd.Process.Kill() }

	re := regexp.MustCompile(`Forwarding from 127\.0\.0\.1:(\d+)`)
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		if m := re.FindStringSubmatch(scanner.Text()); m != nil {
			addr := "127.0.0.1:" + m[1]
			fmt.Printf("port-forward ready: %s -> %s:50051\n", addr, pod)
			return addr, stop, nil
		}
	}
	stop()
	return "", nil, fmt.Errorf("port-forward to %s never became ready", pod)
}
