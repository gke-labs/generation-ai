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
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// The model catalog (#357): models served by name.
//
// RegisterModel imports a model's weights into modelstore and compiles
// it for the router's executor image: a Kubernetes Job runs the executor
// image's `python -m vxpu.compile`, which exports the artifact (reading
// the model through modelstore's Hub API, never the weights) and pushes
// it to modelstore as vxpu-artifacts/<model>:<tag>. The tag names the
// commit and the executor image, so it is the catalog entry: whether a
// model is registered for this engine is whether the tag exists, and the
// catalog lives in modelstore, not in the router. NewSession with a model
// name resolves the tag, fetches the artifact from modelstore and loads
// it (on an external executor when the router has one, else on an
// executor pod), once; later sessions reuse it.

// compileMaxCacheLen is the KV cache length artifacts are compiled for.
const compileMaxCacheLen = 1024

var tagUnsafe = regexp.MustCompile(`[^A-Za-z0-9._-]`)

// artifactTag is the catalog entry for a model commit compiled by an
// executor image: the commit's first 12 characters and imageKey, as a
// valid registry tag.
func artifactTag(commit, image string) string {
	short := commit
	if len(short) > 12 {
		short = short[:12]
	}
	tag := tagUnsafe.ReplaceAllString(short+"_"+imageKey(image), "-")
	if len(tag) > 128 {
		tag = tag[:128]
	}
	return tag
}

// imageKey names the executor image in the catalog: the first 12 hex
// digits of its digest when the reference has one, else its repository's
// last path element and tag ("vxpu-executor-v34"). Only a digest names
// fixed contents: a tag such as :latest that is rebuilt keeps its key, so
// artifacts compiled by the previous build are reused (the router warns
// at startup when the image is not pinned by digest).
func imageKey(image string) string {
	if i := strings.LastIndex(image, "@sha256:"); i >= 0 {
		digest := image[i+len("@sha256:"):]
		if len(digest) > 12 {
			digest = digest[:12]
		}
		return digest
	}
	repo, tag := image, "latest"
	if i := strings.LastIndex(image, ":"); i > strings.LastIndex(image, "/") {
		repo, tag = image[:i], image[i+1:]
	}
	return repo[strings.LastIndex(repo, "/")+1:] + "-" + tag
}

// PinnedByDigest reports whether an image reference names a digest.
func PinnedByDigest(image string) bool {
	return strings.Contains(image, "@sha256:")
}

// defaultCompileResources: export runs on the meta device, so a compile
// needs little; the request fits a single kind node, and the limit lets a
// large model's export (real buffers such as rotary tables) burst.
func defaultCompileResources() corev1.ResourceRequirements {
	return corev1.ResourceRequirements{
		Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("500m"), corev1.ResourceMemory: resource.MustParse("2Gi")},
		Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("32Gi")},
	}
}

// RegisterModel makes a model servable by name. See the file comment.
func (s *Server) RegisterModel(ctx context.Context, req *pb.RegisterModelRequest) (*pb.RegisterModelResponse, error) {
	log := klog.FromContext(ctx)
	if req.Model == "" || strings.ContainsAny(req.Model, " \t\n") {
		return nil, status.Errorf(codes.InvalidArgument, "model %q is not a Hugging Face repository id", req.Model)
	}
	if s.weightStore == nil {
		return nil, status.Error(codes.FailedPrecondition, "registering models by name needs modelstore (deploy it, or start the router with --modelstore)")
	}
	for k, v := range req.Plan { // every load would refuse it: say so now, before anything is imported
		if strings.TrimSpace(v) == "" {
			return nil, status.Errorf(codes.InvalidArgument, "plan switch %s is empty: leave it out instead", k)
		}
	}
	revision := req.Revision
	if revision == "" {
		revision = "main"
	}
	log.Info("RegisterModel", "model", req.Model, "revision", revision)
	for _, owned := range routerOwnedSwitches {
		if _, set := req.Plan[owned]; set {
			return nil, status.Errorf(codes.InvalidArgument, "%s is the router's to set (it wires the executors it starts), not a registered plan's", owned)
		}
	}

	// The weights first (quick when modelstore already has them): export
	// reads the model's config and headers from modelstore, and
	// executors read the weights from it (or its bucket).
	name, err := s.weightStore.ImportRepo(ctx, req.Model, revision)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "importing %s into modelstore: %v", req.Model, err)
	}
	if _, err := s.weightStore.WaitForImport(ctx, name, 5*time.Second); err != nil {
		return nil, status.Errorf(codes.Unavailable, "%v", err)
	}
	commit, err := s.weightStore.ResolveCommit(ctx, req.Model, revision)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "resolving %s@%s: %v", req.Model, revision, err)
	}
	repo := weightstore.ArtifactRepository(req.Model)
	tag := artifactTag(commit, s.imageName)
	if len(req.Plan) > 0 {
		if err := s.putPlan(ctx, req.Model, req.Plan); err != nil {
			return nil, status.Errorf(codes.Unavailable, "storing the plan of %s: %v", req.Model, err)
		}
		// the next session loads it under the new plan (loadByName
		// compares it with the one a loaded tag was loaded under)
		s.setHints(req.Model, req.Plan, true)
		log.Info("Stored the model's plan", "model", req.Model, "switches", planSwitches(req.Plan))
	}
	if req.Golden != nil {
		if req.Golden.Format != "vxpu-golden/1" {
			return nil, status.Errorf(codes.InvalidArgument, "unknown golden format %q, expected vxpu-golden/1", req.Golden.Format)
		}
		if err := s.putGolden(ctx, req.Model, req.Golden); err != nil {
			return nil, status.Errorf(codes.Unavailable, "storing golden file for %s: %v", req.Model, err)
		}
		log.Info("Stored the model's golden reference document", "model", req.Model)
	}
	resp := &pb.RegisterModelResponse{Model: req.Model, Revision: revision, Commit: commit, Artifact: repo + ":" + tag}

	exists, err := s.weightStore.ManifestExists(ctx, repo, tag)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "%v", err)
	}
	if !exists {
		// A compile outlives an impatient client: a retry attaches to it.
		if _, err := s.flights.Do(ctx, "compile/"+repo+":"+tag, func() (string, bool, error) {
			err := s.runCompileJob(context.WithoutCancel(ctx), req.Model, commit, repo, tag)
			return "", false, err
		}); err != nil {
			if stderrors.Is(err, context.Canceled) || stderrors.Is(err, context.DeadlineExceeded) {
				return nil, status.FromContextError(err).Err()
			}
			return nil, err
		}
		resp.Compiled = true
	}
	manifest, err := s.weightStore.FetchFile(ctx, repo, tag, "manifest.json")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "reading the compiled artifact: %v", err)
	}
	if resp.WeightBytes, err = weightBytes(string(manifest)); err != nil {
		return nil, status.Errorf(codes.Internal, "compiled artifact %s: %v", resp.Artifact, err)
	}
	key := artifactKey(string(manifest))
	s.mu.Lock()
	if s.models == nil {
		s.models = make(map[string]modelState)
	}
	if s.keyModel == nil {
		s.keyModel = make(map[string]modelRef)
	}
	if _, ok := s.models[key]; !ok {
		s.models[key] = modelState{standing: standingUnchecked}
	}
	s.keyModel[key] = modelRef{Model: req.Model, Commit: commit, Artifact: resp.Artifact}
	s.mu.Unlock()
	log.Info("Registered model", "model", req.Model, "commit", commit, "artifact", resp.Artifact, "compiled", resp.Compiled)
	return resp, nil
}

// runCompileJob runs vxpu.compile for model@commit in a Kubernetes Job
// (the executor image, on whatever CPU node fits) and waits for it. The
// Job's name is derived from the artifact, so a second router call (or a
// restarted router) attaches to a Job already running.
func (s *Server) runCompileJob(ctx context.Context, model, commit, repo, tag string) error {
	log := klog.FromContext(ctx)
	name := compileJobName(repo, tag)
	backoff, deadline, ttl := int32(1), int64(3600), int32(86400)
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: s.namespace, Labels: map[string]string{"app": "vxpu-compile"},
			Annotations: map[string]string{"vxpu.gke-labs.dev/model": model, "vxpu.gke-labs.dev/artifact": repo + ":" + tag}},
		Spec: batchv1.JobSpec{
			BackoffLimit:            &backoff,
			ActiveDeadlineSeconds:   &deadline,
			TTLSecondsAfterFinished: &ttl,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "vxpu-compile", "vxpu.gke-labs.dev/compile": name}},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:    "compile",
						Image:   s.imageName,
						Command: []string{"python", "-m", "vxpu.compile"},
						Args: []string{model, "--modelstore", s.weightStore.URL(), "--tag", tag, "--revision", commit,
							"--max-cache-len", fmt.Sprint(compileMaxCacheLen)},
						Resources: s.compileResources,
					}},
				},
			},
		},
	}
	if err := s.createCompileJob(ctx, job); err != nil {
		return err
	}
	log.Info("Compiling model", "model", model, "commit", commit, "job", name, "artifact", repo+":"+tag)
	for {
		current, err := s.clientset.BatchV1().Jobs(s.namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return status.Errorf(codes.Internal, "reading compile job %s: %v", name, err)
		}
		if current.Status.Succeeded > 0 {
			log.Info("Compile job succeeded", "job", name, "summary", s.compileSummary(ctx, name))
			return nil
		}
		for _, cond := range current.Status.Conditions {
			if cond.Type == batchv1.JobFailed && cond.Status == corev1.ConditionTrue {
				return status.Errorf(codes.Internal, "compiling %s failed (job %s: %s): %s", model, name, cond.Reason, s.compileLogTail(ctx, name, 20))
			}
		}
		if why, stuck := s.compileUnschedulableFor(ctx, name); stuck {
			return status.Errorf(codes.ResourceExhausted, "compiling %s: job %s's pod cannot be scheduled: %s", model, name, why)
		}
		select {
		case <-ctx.Done():
			return status.FromContextError(ctx.Err()).Err()
		case <-time.After(s.compilePoll):
		}
	}
}

// compileJobName names the compile Job after the artifact it produces, so
// a second router call (or a restarted router) attaches to it.
func compileJobName(repo, tag string) string {
	h := sha256.Sum256([]byte(repo + ":" + tag))
	return "vxpu-compile-" + hex.EncodeToString(h[:5])
}

// createCompileJob creates the compile Job, or attaches to one already
// running under its name. The caller only compiles when the artifact is
// missing, so a Job of that name that has already finished is stale: a
// failure that should be retried, or a success whose artifact has since
// left modelstore. It is deleted and the Job created again, rather than
// its outcome answering every register until its TTL expires.
func (s *Server) createCompileJob(ctx context.Context, job *batchv1.Job) error {
	jobs := s.clientset.BatchV1().Jobs(s.namespace)
	for attempt := 0; ; attempt++ {
		_, err := jobs.Create(ctx, job, metav1.CreateOptions{})
		if err == nil {
			return nil
		}
		if !errors.IsAlreadyExists(err) {
			return status.Errorf(codes.Internal, "creating compile job: %v", err)
		}
		current, err := jobs.Get(ctx, job.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			continue // deleted in between: create it
		}
		if err != nil {
			return status.Errorf(codes.Internal, "reading compile job %s: %v", job.Name, err)
		}
		if !jobFinished(current) {
			return nil // running: attach
		}
		if attempt > 0 {
			return status.Errorf(codes.Internal, "compile job %s finished again before it could be replaced", job.Name)
		}
		klog.FromContext(ctx).Info("Replacing a finished compile job: the artifact it was for is missing", "job", job.Name,
			"succeeded", current.Status.Succeeded)
		background := metav1.DeletePropagationBackground
		if err := jobs.Delete(ctx, job.Name, metav1.DeleteOptions{PropagationPolicy: &background}); err != nil && !errors.IsNotFound(err) {
			return status.Errorf(codes.Internal, "deleting finished compile job %s: %v", job.Name, err)
		}
		for {
			if _, err := jobs.Get(ctx, job.Name, metav1.GetOptions{}); errors.IsNotFound(err) {
				break
			}
			select {
			case <-ctx.Done():
				return status.FromContextError(ctx.Err()).Err()
			case <-time.After(s.compilePoll):
			}
		}
	}
}

// jobFinished reports whether a Job has completed or failed.
func jobFinished(job *batchv1.Job) bool {
	if job.Status.Succeeded > 0 {
		return true
	}
	for _, cond := range job.Status.Conditions {
		if (cond.Type == batchv1.JobFailed || cond.Type == batchv1.JobComplete) && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// compileUnschedulableFor reports whether the compile job's pod has been
// unschedulable (no node with the resources it requests) for longer than
// s.compileUnschedulable, and the scheduler's reason. Otherwise the Job
// would wait out its hour-long deadline in silence.
func (s *Server) compileUnschedulableFor(ctx context.Context, job string) (string, bool) {
	pods, err := s.clientset.CoreV1().Pods(s.namespace).List(ctx, metav1.ListOptions{LabelSelector: "vxpu.gke-labs.dev/compile=" + job})
	if err != nil {
		return "", false
	}
	for _, p := range pods.Items {
		for _, c := range p.Status.Conditions {
			if c.Type == corev1.PodScheduled && c.Status == corev1.ConditionFalse && c.Reason == corev1.PodReasonUnschedulable &&
				time.Since(c.LastTransitionTime.Time) > s.compileUnschedulable {
				return c.Message, true
			}
		}
	}
	return "", false
}

// compileLogTail is the end of the compile job's newest pod's log.
func (s *Server) compileLogTail(ctx context.Context, job string, lines int64) string {
	pods, err := s.clientset.CoreV1().Pods(s.namespace).List(ctx, metav1.ListOptions{LabelSelector: "vxpu.gke-labs.dev/compile=" + job})
	if err != nil || len(pods.Items) == 0 {
		return fmt.Sprintf("(no pod log: %v)", err)
	}
	newest := pods.Items[0]
	for _, p := range pods.Items[1:] {
		if p.CreationTimestamp.After(newest.CreationTimestamp.Time) {
			newest = p
		}
	}
	raw, err := s.clientset.CoreV1().Pods(s.namespace).GetLogs(newest.Name, &corev1.PodLogOptions{TailLines: &lines}).DoRaw(ctx)
	if err != nil {
		return fmt.Sprintf("(reading log of %s: %v)", newest.Name, err)
	}
	return strings.TrimSpace(string(raw))
}

// compileSummary is the JSON summary vxpu.compile prints last, if found.
func (s *Server) compileSummary(ctx context.Context, job string) string {
	tail := s.compileLogTail(ctx, job, 5)
	var last string
	sc := bufio.NewScanner(bytes.NewReader([]byte(tail)))
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); strings.HasPrefix(line, "{") && json.Valid([]byte(line)) {
			last = line
		}
	}
	return last
}

// goldenKey returns a hash of the golden reference document for staleness comparison.
func goldenKey(gf *planpb.GoldenFile) (string, error) {
	if gf == nil {
		return "", nil
	}
	data, err := kernelplan.WriteGoldenJSON(gf)
	if err != nil {
		return "", fmt.Errorf("encoding golden json: %w", err)
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:16]), nil
}

// loadByName returns the model key of a registered model, loading its
// compiled artifact on an executor the first time (and again if that
// executor went away). Unregistered models fail with FAILED_PRECONDITION.
func (s *Server) loadByName(ctx context.Context, model, revision string) (string, error) {
	if s.weightStore == nil {
		return "", status.Error(codes.FailedPrecondition, "models by name need modelstore")
	}
	if revision == "" {
		revision = "main"
	}
	commit, err := s.weightStore.ResolveCommit(ctx, model, revision)
	if stderrors.Is(err, weightstore.ErrNotFound) {
		return "", status.Errorf(codes.FailedPrecondition, "model %s@%s is not registered: call RegisterModel", model, revision)
	}
	if err != nil {
		return "", status.Errorf(codes.Unavailable, "resolving %s@%s: %v", model, revision, err)
	}
	repo := weightstore.ArtifactRepository(model)
	tag := artifactTag(commit, s.imageName)

	gf, err := s.getGolden(ctx, model)
	if err != nil {
		return "", status.Errorf(codes.Unavailable, "reading golden file for %s: %v", model, err)
	}
	currentGolden, err := goldenKey(gf)
	if err != nil {
		return "", status.Errorf(codes.Internal, "golden key for %s: %v", model, err)
	}

	s.mu.RLock()
	key, loaded := s.loadedTags[tag]
	current, known := s.hints[statsKey(model)]
	mState := s.models[key]
	stale := loaded && ((known && mState.hints != current) || mState.golden != currentGolden)
	s.mu.RUnlock()
	if loaded && stale {
		klog.FromContext(ctx).Info("The model's plan or golden document changed; loading it again", "model", model, "model_key", key)
		s.forgetModel(key, "")
	} else if loaded {
		if mState.standing == standingRefused {
			return "", status.Errorf(codes.FailedPrecondition, "golden check failed: %v", mState.err)
		}
		_, err := s.executorEndpoint(ctx, key)
		if err == nil {
			return key, nil
		}
		if status.Code(err) != codes.FailedPrecondition {
			return "", err
		}
		klog.FromContext(ctx).Info("Loaded model's executor is gone; loading again", "model", model, "model_key", key)
		s.forgetModel(key, "")
	}
	loadedKey, err := s.flights.Do(ctx, "load/"+repo+":"+tag, func() (string, bool, error) {
		// the flight is shared: the caller that started it going away
		// must not fail it for the others, as loadArtifact below already
		// is not
		ctx := context.WithoutCancel(ctx)
		exists, err := s.weightStore.ManifestExists(ctx, repo, tag)
		if err != nil {
			return "", false, status.Errorf(codes.Unavailable, "%v", err)
		}
		if !exists {
			return "", true, status.Errorf(codes.FailedPrecondition, "model %s@%s is not compiled for executor image %s: call RegisterModel", model, revision, s.imageName)
		}
		hints, err := s.getPlan(ctx, model)
		if err != nil {
			return "", false, status.Errorf(codes.Unavailable, "reading the plan of %s: %v", model, err)
		}
		s.setHints(model, hints, false)
		files := map[string][]byte{}
		for _, f := range []string{"manifest.json", "binding.json", "prefill.pt2", "decode.pt2"} {
			data, err := s.weightStore.FetchFile(ctx, repo, tag, f)
			if err != nil {
				return "", false, status.Errorf(codes.Internal, "reading artifact %s:%s: %v", repo, tag, err)
			}
			files[f] = data
		}
		modelKey := artifactKey(string(files["manifest.json"]))
		key, err := s.loadArtifact(context.WithoutCancel(ctx), &pb.LoadModelRequest{
			ManifestJson: string(files["manifest.json"]),
			BindingJson:  string(files["binding.json"]),
			PrefillGraph: files["prefill.pt2"],
			DecodeGraph:  files["decode.pt2"],
		}, true, func(ctx context.Context, info *executorInfo, manifest string) map[string]string {
			// the manifest as sent: the expert executors read their
			// weights from where the engine does
			return s.planLoad(ctx, model, []byte(manifest), files["binding.json"], files["prefill.pt2"], files["decode.pt2"], hints, info)
		})
		if err != nil {
			if status.Code(err) == codes.FailedPrecondition && s.modelStanding(modelKey).standing == standingRefused {
				s.mu.Lock()
				s.loadedTags[tag] = modelKey
				st := s.models[modelKey]
				st.hints = hintsKey(hints)
				st.golden = currentGolden
				s.models[modelKey] = st
				s.keyModel[modelKey] = modelRef{Model: model, Commit: commit, Artifact: repo + ":" + tag}
				s.mu.Unlock()
				return "", true, err
			}
			return "", false, err
		}
		s.mu.Lock()
		s.loadedTags[tag] = key
		st := s.models[key]
		st.hints = hintsKey(hints)
		st.golden = currentGolden
		s.models[key] = st
		s.keyModel[key] = modelRef{Model: model, Commit: commit, Artifact: repo + ":" + tag}
		s.mu.Unlock()
		return key, true, nil
	})
	if err != nil {
		if stderrors.Is(err, context.Canceled) || stderrors.Is(err, context.DeadlineExceeded) {
			return "", status.FromContextError(err).Err()
		}
		return "", err
	}
	return loadedKey, nil
}
