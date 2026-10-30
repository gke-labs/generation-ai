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
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

//go:embed router.yaml
var routerManifest string

const maxMessageBytes = 128 * 1024 * 1024

func main() {
	ctx := context.Background()
	log := klog.FromContext(ctx)
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr,
			"usage: vxpu ask [flags] PROMPT | vxpu register [flags] REPO[@REVISION] | vxpu stats [flags] REPO | vxpu up [flags] | vxpu down [flags] | vxpu mirror REPO[@REVISION] | vxpu machines [NAME] | vxpu explain [flags] REPO | vxpu disk build|attach|detach [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "ask":
		err = cmdAsk(ctx, os.Args[2:])
	case "register":
		err = cmdRegister(ctx, os.Args[2:])
	case "stats":
		err = cmdStats(ctx, os.Args[2:])
	case "up":
		err = cmdUp(ctx, os.Args[2:])
	case "down":
		err = cmdDown(ctx, os.Args[2:])
	case "mirror":
		err = cmdMirror(ctx, os.Args[2:])
	case "machines":
		err = cmdMachines(os.Args[2:])
	case "explain":
		err = cmdExplain(ctx, os.Args[2:])
	case "disk":
		err = cmdDisk(ctx, os.Args[2:])
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
		os.Getenv("VXPU_ROUTER_IMAGE"), "router image (used only if the router has to be created)")
	executorImage := flags.String("executor-image",
		os.Getenv("VXPU_EXECUTOR_IMAGE"), "executor image (used only if the router has to be created)")
	accelerator := flags.String("accelerator", "nvidia-l4",
		"GKE accelerator label for the executor pod (used only if the router has to be created; 'vxpu up' changes a running router)")
	modelstore := flags.String("modelstore", os.Getenv("VXPU_MODELSTORE_URL"),
		"modelstore URL for weights (used only if the router has to be created; empty: the router finds modelstore.modelstore or caches weights itself)")
	externalExecutors := flags.String("external-executor", os.Getenv("VXPU_EXTERNAL_EXECUTORS"),
		"comma-separated executor addresses outside Kubernetes (used only if the router has to be created)")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips direct pod creation")
	model := flags.String("model", "", "a registered model (REPO[@REVISION], see `vxpu register`) instead of --artifact")
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

	client, done, err := connectRouter(ctx, *routerAddr, *pod, routerSettings{image: *image, executorImage: *executorImage,
		accelerator: *accelerator, modelstore: *modelstore, externalExecutors: *externalExecutors})
	if err != nil {
		return err
	}
	defer done()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	if *model != "" {
		repo, revision := splitRevision(*model)
		return askByName(ctx, client, repo, revision, prompt, *maxNewTokens)
	}
	log.V(2).Info("shipping the artifact", "artifact", *artifact)

	files := map[string][]byte{}
	for _, name := range []string{"manifest.json", "binding.json", "prefill.pt2", "decode.pt2"} {
		data, err := os.ReadFile(filepath.Join(*artifact, name))
		if err != nil {
			return fmt.Errorf("reading artifact: %w", err)
		}
		files[name] = data
	}
	if err := checkShippable(files["manifest.json"]); err != nil {
		return err
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
	modelstore := flags.String("modelstore", os.Getenv("VXPU_MODELSTORE_URL"),
		"modelstore URL to keep weights in (empty: the router finds modelstore.modelstore if deployed, else caches weights itself)")
	externalExecutors := flags.String("external-executor", os.Getenv("VXPU_EXTERNAL_EXECUTORS"),
		"comma-separated executor addresses outside Kubernetes (GPU VMs, Macs) that models registered by name are placed on")
	externalModelstore := flags.String("modelstore-external-url", os.Getenv("VXPU_MODELSTORE_EXTERNAL_URL"),
		"modelstore's address for external executors, which cannot resolve its in-cluster Service (empty: they get --modelstore's URL)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	// A running router keeps the settings it was started with unless
	// a flag names a new value: `vxpu up` on its own means "make sure
	// it is running", not "reset it to the defaults".
	set := map[string]bool{}
	flags.Visit(func(f *flag.Flag) { set[f.Name] = true })
	if current, err := podSettings(ctx, *pod); err != nil {
		return err
	} else if current != nil {
		if !set["image"] && current.image != "" {
			*image = current.image
		}
		if !set["executor-image"] && current.executorImage != "" {
			*executorImage = current.executorImage
		}
		if !set["accelerator"] && current.accelerator != "" {
			*accelerator = current.accelerator
		}
		if !set["modelstore"] {
			*modelstore = current.modelstore
		}
		if !set["external-executor"] {
			*externalExecutors = current.externalExecutors
		}
		if !set["modelstore-external-url"] {
			*externalModelstore = current.externalModelstore
		}
	}
	if err := ensurePod(ctx, *pod, routerSettings{image: *image, executorImage: *executorImage, accelerator: *accelerator,
		modelstore: *modelstore, externalExecutors: *externalExecutors, externalModelstore: *externalModelstore}, true); err != nil {
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
	return deleteRouterResources(ctx, *pod)
}

// routerSettings are the values a running router was created with.
type routerSettings struct {
	image, executorImage, accelerator, modelstore string
	externalExecutors                             string // comma-separated host:port
	externalModelstore                            string // modelstore's URL for external executors
}

// ensurePod applies the router manifest (ServiceAccount, Role,
// RoleBinding, Service, Pod) and waits until the pod is Ready. Applying
// every time keeps an existing router's Service/RBAC current; if the
// pod itself cannot be updated in place (an immutable field changed, or
// it has exited), it is recreated.
// ensurePod makes sure the router pod is running. With reconfigure set
// (`vxpu up`) the manifest is applied, and a pod whose settings differ is
// recreated; without it (`vxpu ask`) a live router is used as it is,
// because replacing it would reset the accelerator chosen with `up` and
// drop its weight cache (issue #221).
func ensurePod(ctx context.Context, pod string, want routerSettings, reconfigure bool) error {
	image, executorImage, accelerator := want.image, want.executorImage, want.accelerator
	phase, err := podPhase(ctx, pod)
	if err != nil {
		return err
	}
	if image == "" {
		if phase != "" {
			return waitReady(ctx, pod)
		}
		return fmt.Errorf(
			"router pod %q not found: start it with `vxpu up` (or set VXPU_ROUTER_IMAGE, or pass --router)",
			pod)
	}
	if phase == "Failed" || phase == "Succeeded" {
		fmt.Printf("router pod %s has exited (%s); recreating\n", pod, phase)
		if err := deletePod(ctx, pod); err != nil {
			return err
		}
		phase = ""
	}
	if phase != "" && !reconfigure {
		fmt.Printf("using router %s as it is (run `vxpu up` to change its image or accelerator)\n", pod)
		return waitReady(ctx, pod)
	}
	fmt.Printf("applying router %s (image %s, executor-image %s, accelerator %s, modelstore %q, external executors %q, "+
		"modelstore for them %q)\n", pod, image, executorImage, accelerator, want.modelstore, want.externalExecutors,
		want.externalModelstore)
	manifest := strings.NewReplacer(
		`"NAME"`, fmt.Sprintf("%q", pod),
		`"IMAGE"`, fmt.Sprintf("%q", image),
		`"EXECUTOR_IMAGE"`, fmt.Sprintf("%q", executorImage),
		`"ACCELERATOR"`, fmt.Sprintf("%q", accelerator),
		`"MODELSTORE"`, fmt.Sprintf("%q", want.modelstore),
		`"EXTERNAL_EXECUTORS"`, fmt.Sprintf("%q", want.externalExecutors),
		`"MODELSTORE_EXTERNAL_URL"`, fmt.Sprintf("%q", want.externalModelstore),
	).Replace(routerManifest)
	if err := applyManifest(ctx, manifest); err != nil {
		if errors.Is(err, errPodNeedsRecreate) {
			// Pods are immutable apart from a few fields; recreate.
			fmt.Printf("router pod %s cannot be updated in place; recreating\n", pod)
			if derr := deletePod(ctx, pod); derr != nil {
				return derr
			}
			if err := applyManifest(ctx, manifest); err != nil {
				return err
			}
		} else {
			return err
		}
	}
	return waitReady(ctx, pod)
}

// checkShippable refuses an artifact exported from a local checkpoint
// directory: its manifest binds weights to file:// paths on the
// exporting machine, which an executor elsewhere cannot read. Catching
// it here is better than on the executor, which would already have
// evicted the model it was serving to make room.
func checkShippable(manifestJSON []byte) error {
	var manifest struct {
		Files map[string]struct {
			Source string `json:"source"`
		} `json:"files"`
	}
	if err := json.Unmarshal(manifestJSON, &manifest); err != nil {
		return fmt.Errorf("parsing manifest.json: %w", err)
	}
	for _, f := range manifest.Files {
		if strings.HasPrefix(f.Source, "file://") {
			return fmt.Errorf("artifact binds weights to local files, which the executor cannot read (%s); export from a Hub repo id to ship it, or run it in-process", f.Source)
		}
	}
	return nil
}

// cmdMirror asks modelstore (the cluster's weight store) to import a
// Hugging Face repository, so that its weights are pulled from the Hub
// once and every executor, reference run and vLLM pod reads them from
// the bucket. It returns when the import is done; the Model object
// (`kubectl get models`) shows progress meanwhile.
func cmdMirror(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("mirror", flag.ContinueOnError)
	namespace := flags.String("namespace", "modelstore", "namespace of the modelstore Service")
	service := flags.String("service", "modelstore", "name of the modelstore Service")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("usage: vxpu mirror [flags] REPO[@REVISION]")
	}
	repo, revision, _ := strings.Cut(flags.Arg(0), "@")
	if revision == "" {
		revision = "main"
	}
	addr, stop, err := portForwardService(ctx, *namespace, *service, 80)
	if err != nil {
		return fmt.Errorf("port-forward to modelstore: %w", err)
	}
	defer stop()
	store := weightstore.New("http://" + addr)
	name, err := store.ImportRepo(ctx, repo, revision)
	if err != nil {
		return err
	}
	fmt.Printf("importing %s@%s into modelstore as Model %s (kubectl get models -w)\n", repo, revision, name)
	last := weightstore.ModelStatus{}
	for {
		st, err := store.GetModelStatus(ctx, name)
		if err != nil {
			return err
		}
		if st != last {
			fmt.Printf("  %s: %d/%d files, %.1f GB\n", st.Phase, st.FilesImported, st.FilesTotal, float64(st.BytesImported)/1e9)
			last = st
		}
		switch st.Phase {
		case "Ready":
			fmt.Printf("%s@%s is in the weight store\n", repo, revision)
			return nil
		case "Failed":
			return fmt.Errorf("import failed: %s", st.Message)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
}
