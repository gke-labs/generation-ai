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

// vxpu runs vXPU artifacts on Kubernetes accelerators.
//
// The client is deliberately free of PyTorch, Python, and CUDA: it
// ships an artifact directory (manifest + weightless graphs, tens of
// MB) to an executor pod — creating the pod on demand — and chats over
// gRPC. Weights never pass through this machine; the executor
// rehydrates them from the manifest's content-addressed references.
//
//	vxpu ask --artifact ./gemma-e4b "Is the sky blue?"
//	vxpu up      # just the router, for notebooks/clients to connect to
//	vxpu down
package main

import (
	"bufio"
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

//go:embed router.yaml
var routerManifest string

const maxMessageBytes = 128 * 1024 * 1024

func main() {
	ctx := context.Background()
	log := klog.FromContext(ctx)
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr,
			"usage: vxpu ask [flags] PROMPT | vxpu up [flags] | vxpu down [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "ask":
		err = cmdAsk(ctx, os.Args[2:])
	case "up":
		err = cmdUp(ctx, os.Args[2:])
	case "down":
		err = cmdDown(ctx, os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", os.Args[1])
		os.Exit(2)
	}
	if err != nil {
		log.Error(err, "command failed", "command", os.Args[1])
		os.Exit(1)
	}
}

func cmdAsk(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("ask", flag.ContinueOnError)
	artifact := flags.String("artifact", ".",
		"artifact directory (manifest.json, binding.json, *.pt2)")
	pod := flags.String("pod", "vxpu-router", "router pod name")
	image := flags.String("image",
		os.Getenv("VXPU_ROUTER_IMAGE"), "router image")
	executorImage := flags.String("executor-image",
		os.Getenv("VXPU_EXECUTOR_IMAGE"), "executor image")
	accelerator := flags.String("accelerator", "nvidia-l4",
		"GKE accelerator label for the executor pod")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips direct pod creation")
	maxNewTokens := flags.Int("max-new-tokens", 96, "tokens per reply")
	timeout := flags.Duration("timeout", 20*time.Minute,
		"end-to-end timeout (cold loads rehydrate all weights)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	prompt := strings.Join(flags.Args(), " ")
	if prompt == "" {
		return errors.New("usage: vxpu ask [flags] PROMPT")
	}

	var addr string
	var stop func()
	if *routerAddr != "" {
		addr = *routerAddr
		stop = func() {}
	} else {
		if err := ensurePod(ctx, *pod, *image, *executorImage, *accelerator); err != nil {
			return fmt.Errorf("ensuring router pod: %w", err)
		}
		var err error
		addr, stop, err = portForward(ctx, *pod)
		if err != nil {
			return fmt.Errorf("port-forward: %w", err)
		}
	}
	defer stop()

	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(maxMessageBytes),
			grpc.MaxCallRecvMsgSize(maxMessageBytes)))
	if err != nil {
		return fmt.Errorf("dialing %s: %w", addr, err)
	}
	defer func() {
		if err := conn.Close(); err != nil {
			log.Error(err, "closing router connection", "address", addr)
		}
	}()
	client := pb.NewExecutorClient(conn)
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	files := map[string][]byte{}
	for _, name := range []string{"manifest.json", "binding.json", "prefill.pt2", "decode.pt2"} {
		data, err := os.ReadFile(filepath.Join(*artifact, name))
		if err != nil {
			return fmt.Errorf("reading artifact: %w", err)
		}
		files[name] = data
	}
	fmt.Printf("shipping artifact %s (%d MB graphs; weights stay remote)\n",
		*artifact, (len(files["prefill.pt2"])+len(files["decode.pt2"]))/(1024*1024))

	started := time.Now()
	// LoadModel accepts the artifact and loads asynchronously (it is
	// idempotent by content digest); NewSession doubles as the
	// readiness poll — no long-held RPC, so tunnels never sit idle.
	if _, err := client.LoadModel(ctx, &pb.LoadModelRequest{
		ManifestJson: string(files["manifest.json"]),
		BindingJson:  string(files["binding.json"]),
		PrefillGraph: files["prefill.pt2"],
		DecodeGraph:  files["decode.pt2"],
	}, grpc.WaitForReady(true)); err != nil {
		return fmt.Errorf("LoadModel: %w", err)
	}

	var session *pb.NewSessionResponse
	for {
		var err error
		session, err = client.NewSession(ctx, &pb.NewSessionRequest{})
		if err == nil {
			break
		}
		if status.Code(err) != codes.FailedPrecondition {
			return fmt.Errorf("NewSession: %w", err)
		}
		if ctx.Err() != nil {
			return fmt.Errorf("timed out waiting for model load: %w", err)
		}
		fmt.Printf("  loading... (%.0fs)\n", time.Since(started).Seconds())
		time.Sleep(15 * time.Second)
	}
	fmt.Printf("model loaded in %.1fs\n", time.Since(started).Seconds())
	started = time.Now()
	reply, err := client.Chat(ctx, &pb.ChatRequest{
		SessionId:    session.SessionId,
		Text:         prompt,
		MaxNewTokens: int32(*maxNewTokens),
	})
	if err != nil {
		return fmt.Errorf("Chat: %w", err)
	}
	fmt.Printf("\n>>> %s\n%s\n\n(%d tokens, %.1f ms/token on the executor, "+
		"%.1fs round trip)\n", prompt, reply.Text, reply.Generated,
		reply.MsPerToken, time.Since(started).Seconds())
	return nil
}

// cmdUp ensures the router is running without loading a model, so
// other clients (e.g. the Python client from a notebook) can connect
// via port-forward or the in-cluster Service.
func cmdUp(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("up", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod name")
	image := flags.String("image",
		os.Getenv("VXPU_ROUTER_IMAGE"), "router image")
	executorImage := flags.String("executor-image",
		os.Getenv("VXPU_EXECUTOR_IMAGE"), "executor image")
	accelerator := flags.String("accelerator", "nvidia-l4",
		"GKE accelerator label for the executor pod")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if err := ensurePod(ctx, *pod, *image, *executorImage, *accelerator); err != nil {
		return fmt.Errorf("ensuring router pod: %w", err)
	}
	fmt.Printf("router %s is ready\n", *pod)
	fmt.Printf("  from outside the cluster: kubectl port-forward pod/%s 50051:50051\n", *pod)
	fmt.Printf("  from inside the cluster:  %s:50051\n", *pod)
	return nil
}

func cmdDown(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("down", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod (and service) name")
	if err := flags.Parse(args); err != nil {
		return err
	}
	// The named pod/service (covers routers created before objects were
	// labelled), then everything the manifest labels as part of the
	// router: ServiceAccount, Role, RoleBinding, Service.
	for _, args := range [][]string{
		{"delete", "pod,service", *pod, "--ignore-not-found"},
		{"delete", "pod,service,serviceaccount,role,rolebinding",
			"-l", "app.kubernetes.io/part-of=vxpu-router", "--ignore-not-found"},
	} {
		out, err := exec.CommandContext(ctx, "kubectl", args...).CombinedOutput()
		fmt.Print(string(out))
		if err != nil {
			return fmt.Errorf("kubectl %s: %w", strings.Join(args, " "), err)
		}
	}
	return nil
}

// podPhase returns the router pod's phase, or "" if there is no such
// pod. --ignore-not-found makes absence a success with empty output,
// so any error from kubectl is a real one.
func podPhase(ctx context.Context, pod string) (string, error) {
	out, err := exec.CommandContext(ctx, "kubectl", "get", "pod", pod,
		"--ignore-not-found", "-o", "jsonpath={.status.phase}").CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("kubectl get pod %s: %w: %s", pod, err,
			strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// ensurePod applies the router manifest (ServiceAccount, Role,
// RoleBinding, Service, Pod) and waits until the pod is Ready. Applying
// every time keeps an existing router's Service/RBAC current; if the
// pod itself cannot be updated in place (an immutable field changed, or
// it has exited), it is recreated.
func ensurePod(ctx context.Context, pod, image, executorImage, accelerator string) error {
	phase, err := podPhase(ctx, pod)
	if err != nil {
		return err
	}
	if image == "" {
		if phase != "" {
			return waitReady(ctx, pod)
		}
		return fmt.Errorf(
			"pod %q not found and no --image/VXPU_ROUTER_IMAGE set",
			pod)
	}
	if phase == "Failed" || phase == "Succeeded" {
		fmt.Printf("router pod %s has exited (%s); recreating\n", pod, phase)
		if err := deletePod(ctx, pod); err != nil {
			return err
		}
	}
	fmt.Printf("applying router %s (image %s, executor-image %s, accelerator %s)\n",
		pod, image, executorImage, accelerator)
	manifest := strings.NewReplacer(
		`"NAME"`, fmt.Sprintf("%q", pod),
		`"IMAGE"`, fmt.Sprintf("%q", image),
		`"EXECUTOR_IMAGE"`, fmt.Sprintf("%q", executorImage),
		`"ACCELERATOR"`, fmt.Sprintf("%q", accelerator),
	).Replace(routerManifest)
	if err := kubectlApply(ctx, manifest); err != nil {
		// Pods are immutable apart from a few fields; recreate.
		fmt.Printf("router pod %s cannot be updated in place; recreating\n", pod)
		if derr := deletePod(ctx, pod); derr != nil {
			return derr
		}
		if err := kubectlApply(ctx, manifest); err != nil {
			return err
		}
	}
	return waitReady(ctx, pod)
}

func kubectlApply(ctx context.Context, manifest string) error {
	apply := exec.CommandContext(ctx, "kubectl", "apply", "-f", "-")
	apply.Stdin = strings.NewReader(manifest)
	apply.Stdout, apply.Stderr = os.Stdout, os.Stderr
	return apply.Run()
}

func deletePod(ctx context.Context, pod string) error {
	del := exec.CommandContext(ctx, "kubectl", "delete", "pod", pod,
		"--ignore-not-found", "--wait=true")
	del.Stdout, del.Stderr = os.Stdout, os.Stderr
	return del.Run()
}

func waitReady(ctx context.Context, pod string) error {
	log := klog.FromContext(ctx)
	wait := exec.CommandContext(ctx, "kubectl", "wait", "--for=condition=Ready",
		"pod/"+pod, "--timeout=900s")
	wait.Stdout, wait.Stderr = os.Stdout, os.Stderr
	err := wait.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "pod %s failed to become ready. Printing pod details and logs:\n", pod)
		for _, diag := range [][]string{
			{"get", "pod", pod, "-o", "yaml"},
			{"logs", pod, "--all-containers", "--tail=50"},
		} {
			out, derr := exec.CommandContext(ctx, "kubectl", diag...).CombinedOutput()
			if derr != nil {
				log.Error(derr, "collecting diagnostics", "command", "kubectl "+strings.Join(diag, " "))
			}
			fmt.Fprintf(os.Stderr, "$ kubectl %s\n%s\n", strings.Join(diag, " "), string(out))
		}
	}
	return err
}

// portForward tunnels an ephemeral local port to the router.
func portForward(ctx context.Context, pod string) (string, func(), error) {
	log := klog.FromContext(ctx)
	cmd := exec.CommandContext(ctx, "kubectl", "port-forward", "pod/"+pod, ":50051")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return "", nil, err
	}
	stop := func() {
		if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			log.Error(err, "stopping port-forward", "pod", pod)
		}
	}

	re := regexp.MustCompile(`Forwarding from (?:127\.0\.0\.1|\[::1\]):(\d+)`)
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		if m := re.FindStringSubmatch(scanner.Text()); m != nil {
			return "127.0.0.1:" + m[1], stop, nil
		}
	}
	stop()
	return "", nil, fmt.Errorf("port-forward to %s never became ready", pod)
}
