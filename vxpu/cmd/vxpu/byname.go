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
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// Models by name (#357): `vxpu register` makes a model servable by name
// (the router imports its weights into modelstore and compiles it), and
// `vxpu ask --model` opens a session on it by name.

// connectRouter dials the router at addr, or ensures the router pod
// exists (creating it with settings if needed) and port-forwards to it.
// done closes the connection and the tunnel.
func connectRouter(ctx context.Context, addr, pod string, settings routerSettings) (pb.EngineClient, func(), error) {
	log := klog.FromContext(ctx)
	stop := func() {}
	if addr == "" {
		// A running router is used as it is: its settings were decided
		// by `vxpu up`; these only apply when the router has to be created.
		if err := ensurePod(ctx, pod, settings, false); err != nil {
			return nil, nil, fmt.Errorf("ensuring router pod: %w", err)
		}
		var err error
		if addr, stop, err = portForward(ctx, pod); err != nil {
			return nil, nil, fmt.Errorf("port-forward: %w", err)
		}
	}
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(maxMessageBytes),
			grpc.MaxCallRecvMsgSize(maxMessageBytes)))
	if err != nil {
		stop()
		return nil, nil, fmt.Errorf("dialing %s: %w", addr, err)
	}
	return pb.NewEngineClient(conn), func() {
		if err := conn.Close(); err != nil {
			log.Error(err, "closing router connection", "address", addr)
		}
		stop()
	}, nil
}

// splitRevision splits REPO[@REVISION].
func splitRevision(ref string) (string, string) {
	if repo, revision, ok := strings.Cut(ref, "@"); ok {
		return repo, revision
	}
	return ref, ""
}

func cmdRegister(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("register", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod name")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips the pod and port-forward")
	golden := flags.String("golden", "", "optional golden tokens file to verify against")
	timeout := flags.Duration("timeout", 2*time.Hour, "how long to wait (importing weights and compiling can take a while)")
	plan := map[string]string{}
	flags.Func("plan", "VXPU_NAME=VALUE: a switch the router sends with every load of the model, as the "+
		"environment would set it (repeatable; replaces the model's plan)", func(kv string) error {
		k, v, ok := strings.Cut(kv, "=")
		if !ok || !strings.HasPrefix(k, "VXPU_") {
			return fmt.Errorf("%q: expected VXPU_NAME=VALUE", kv)
		}
		plan[k] = v
		return nil
	})
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu register [flags] REPO[@REVISION]")
	}
	repo, revision := splitRevision(flags.Arg(0))
	var goldenJSON string
	if *golden != "" {
		data, err := os.ReadFile(*golden)
		if err != nil {
			return fmt.Errorf("reading golden file: %w", err)
		}
		goldenJSON = string(data)
	}
	client, done, err := connectRouter(ctx, *routerAddr, *pod, envRouterSettings())
	if err != nil {
		return err
	}
	defer done()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	fmt.Printf("registering %s (importing weights into modelstore, compiling if needed)...\n", flags.Arg(0))
	started := time.Now()
	resp, err := client.RegisterModel(ctx, &pb.RegisterModelRequest{Model: repo, Revision: revision, Plan: plan, GoldenJson: goldenJSON}, grpc.WaitForReady(true))
	if err != nil {
		return fmt.Errorf("RegisterModel: %w", err)
	}
	how := "already compiled for this engine"
	if resp.Compiled {
		how = "compiled"
	}
	fmt.Printf("registered %s@%s (commit %s): %s, %.1f GB of weights, artifact %s (%.0fs)\n",
		resp.Model, resp.Revision, resp.Commit, how, float64(resp.WeightBytes)/1e9, resp.Artifact, time.Since(started).Seconds())
	if len(plan) > 0 {
		fmt.Printf("plan: %v (sent with every load of the model)\n", plan)
	}
	return nil
}

// envRouterSettings are the router settings the environment gives, for a
// command that creates the router if there is none (as `vxpu ask` does).
func envRouterSettings() routerSettings {
	return routerSettings{image: os.Getenv("VXPU_ROUTER_IMAGE"), executorImage: os.Getenv("VXPU_EXECUTOR_IMAGE"),
		accelerator: "nvidia-l4", modelstore: os.Getenv("VXPU_MODELSTORE_URL"),
		externalExecutors: os.Getenv("VXPU_EXTERNAL_EXECUTORS"), externalModelstore: os.Getenv("VXPU_MODELSTORE_EXTERNAL_URL")}
}

// loadPoll is how often askByName asks again while a model loads.
var loadPoll = 15 * time.Second

// stillLoading reports whether NewSession's error is a state that passes
// on its own: the model loading on its executor, being reloaded after an
// eviction, or loading again after a failed load. Every other error (not
// registered, not compiled, no modelstore, ...) is final.
func stillLoading(err error) bool {
	msg := status.Convert(err).Message()
	switch status.Code(err) {
	case codes.FailedPrecondition:
		return strings.Contains(msg, "still loading") || strings.Contains(msg, "being reloaded")
	case codes.Unavailable:
		return strings.Contains(msg, "loads it again")
	}
	return false
}

// askByName opens a session on a registered model by name (the router
// loads it the first time; NewSession reports it loading until it is)
// and runs one chat turn.
func askByName(ctx context.Context, client pb.EngineClient, repo, revision, prompt string, maxNewTokens int) error {
	started := time.Now()
	var session *pb.NewSessionResponse
	for {
		var err error
		session, err = client.NewSession(ctx, &pb.NewSessionRequest{Model: repo, Revision: revision}, grpc.WaitForReady(true))
		if err == nil {
			break
		}
		if !stillLoading(err) {
			return fmt.Errorf("NewSession: %w", err)
		}
		if ctx.Err() != nil {
			return fmt.Errorf("timed out waiting for %s to load: %w", repo, err)
		}
		fmt.Printf("  loading %s... (%.0fs: %s)\n", repo, time.Since(started).Seconds(), status.Convert(err).Message())
		time.Sleep(loadPoll)
	}
	fmt.Printf("session on %s ready in %.1fs\n", repo, time.Since(started).Seconds())
	started = time.Now()
	reply, err := client.Chat(ctx, &pb.ChatRequest{SessionId: session.SessionId, Text: prompt, MaxNewTokens: int32(maxNewTokens)})
	if err != nil {
		return fmt.Errorf("Chat: %w", err)
	}
	fmt.Printf("\n>>> %s\n%s\n\n(%d tokens, %.1f ms/token on the executor, %.1fs round trip)\n",
		prompt, reply.Text, reply.Generated, reply.MsPerToken, time.Since(started).Seconds())
	return nil
}

// cmdStats prints the engine's statistics for a model served by name,
// and with --records writes every stored record (JSON lines; readable by
// `python -m vxpu.expert_stats`).
func cmdStats(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("stats", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod name")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips the pod and port-forward")
	records := flags.String("records", "", "also write the stored records to this file (JSON lines; the newest 64 MiB)")
	timeout := flags.Duration("timeout", 5*time.Minute, "how long to wait for the summary")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu stats [flags] REPO")
	}
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	client, done, err := connectRouter(ctx, *routerAddr, *pod, envRouterSettings())
	if err != nil {
		return err
	}
	defer done()
	resp, err := client.GetModelStats(ctx, &pb.GetModelStatsRequest{Model: flags.Arg(0), Records: *records != ""})
	if err != nil {
		return fmt.Errorf("GetModelStats: %w", err)
	}
	fmt.Println(resp.SummaryJson)
	if *records != "" {
		if err := os.WriteFile(*records, []byte(resp.RecordsJsonl+"\n"), 0o644); err != nil {
			return err
		}
		fmt.Printf("records written to %s\n", *records)
	}
	return nil
}

// cmdExplain shows the plan the coordinator chose for a model's last
// load by name, and why (#394).
func cmdExplain(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("explain", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod name")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips the pod and port-forward")
	artifact := flags.String("artifact", "", "with --machines: an artifact directory (manifest.json, binding.json) to plan offline")
	fleet := flags.String("machines", "", "plan offline on these machines, from the catalog: TYPE[:COUNT],... (e.g. g2-standard-48:2,c3d-standard-60:2)")
	records := flags.String("records", "", "with --machines: request records (vxpu stats --records) whose expert selections place GPU experts")
	asJSON := flags.Bool("json", false, "with --machines: every candidate in full, as JSON")
	serve := flags.String("serve", "", "with --machines: show the planner searching layouts live, on this address (e.g. :8080)")
	cuts := flags.String("cuts", "", "with --machines: where the step can be cut into ranges (python -m vxpu.cuts ARTIFACT > FILE; default: the artifact's cuts.json, if any)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *fleet != "" {
		return explainOffline(ctx, *artifact, *fleet, *records, *cuts, *asJSON, *serve)
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu explain [flags] REPO | vxpu explain --artifact DIR --machines TYPE[:COUNT],...")
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	client, done, err := connectRouter(ctx, *routerAddr, *pod, envRouterSettings())
	if err != nil {
		return err
	}
	defer done()
	resp, err := client.ExplainModel(ctx, &pb.ExplainModelRequest{Model: flags.Arg(0)})
	if err != nil {
		return fmt.Errorf("ExplainModel: %w", err)
	}
	fmt.Println(resp.PlanJson)
	return nil
}
