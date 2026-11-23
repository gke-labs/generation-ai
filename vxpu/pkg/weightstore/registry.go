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

package weightstore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

// The part of modelstore's registry and Hub APIs the router uses to keep
// models by name (#357): imported weights are a repository named after
// the Hub id (lowercased), tagged with the revision and its commit, and
// compiled artifacts are kept under ArtifactRepository.

const (
	mediaTypeManifest   = "application/vnd.oci.image.manifest.v1+json"
	annotationHubCommit = "vnd.gke-labs.huggingface.commit"
)

// ErrNotFound: modelstore holds no such repository or tag.
var ErrNotFound = errors.New("not found in modelstore")

// RepositoryName is modelstore's name for a Hub repository.
func RepositoryName(repo string) string { return strings.ToLower(repo) }

// ArtifactRepository is where vxpu keeps a model's compiled artifacts
// (vxpu.compile pushes them there).
func ArtifactRepository(repo string) string { return "vxpu-artifacts/" + strings.ToLower(repo) }

// PackageRepository is where vxpu keeps a model's compiled packages in modelstore.
func PackageRepository(repo string) string { return "vxpu-packages/" + strings.ToLower(repo) }

// FailureRepository is where vxpu keeps a model's compile failure records in modelstore.
func FailureRepository(repo string) string { return "vxpu-failures/" + strings.ToLower(repo) }

// BodyOutcomeRepository is where vxpu keeps kernel body compile outcome records in modelstore.
// Body outcomes are facts about a kernel on a machine kind, holding across all models and packages.
func BodyOutcomeRepository() string { return "vxpu-body-outcomes" }

const (
	failureArtifactType        = "application/vnd.gke-labs.vxpu.compile-failure.v1+json"
	failureClearedArtifactType = "application/vnd.gke-labs.vxpu.compile-failure.cleared.v1+json"
	outcomeArtifactType        = "application/vnd.gke-labs.vxpu.body-outcome.v1+json"
)

// BodyOutcomeTag computes the repository tag for a body compile outcome on a machine kind and software version.
func BodyOutcomeTag(op, body, machineKind, softwareVersion string) string {
	raw := fmt.Sprintf("%s--%s--%s--%s", op, body, machineKind, softwareVersion)
	h := sha256.Sum256([]byte(raw))
	prefix := sanitizeTagPart(fmt.Sprintf("%s-%s", op, body))
	if len(prefix) > 40 {
		prefix = prefix[:40]
	}
	return fmt.Sprintf("body-%s-%s", prefix, hex.EncodeToString(h[:16]))
}

func sanitizeTagPart(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' || r == '.' {
			sb.WriteRune(r)
		} else {
			sb.WriteRune('_')
		}
	}
	return sb.String()
}

// HasPackage reports whether modelstore holds a compiled package for key under model.
func (c *Client) HasPackage(ctx context.Context, model, key string) (bool, error) {
	return c.ManifestExists(ctx, PackageRepository(model), key)
}

// RecordCompileFailure records a toolchain compilation failure for a package in modelstore.
func (c *Client) RecordCompileFailure(ctx context.Context, model string, failure *planpb.CompileFailure) error {
	if failure == nil || failure.Package == nil || failure.Package.Key == "" {
		return errors.New("package key is required to record compile failure")
	}
	data, err := protojson.Marshal(failure)
	if err != nil {
		return fmt.Errorf("marshal compile failure: %w", err)
	}
	name := FailureRepository(model)
	digest, err := c.PutBlob(ctx, name, data)
	if err != nil {
		return err
	}
	manifest := &Manifest{
		SchemaVersion: 2,
		MediaType:     mediaTypeManifest,
		ArtifactType:  failureArtifactType,
		Config:        EmptyConfig,
		Layers: []Layer{{
			MediaType:   MediaTypeFile,
			Digest:      digest,
			Size:        int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": "compile-failure.json"},
		}},
		Annotations: map[string]string{
			"vnd.gke-labs.vxpu.package.name": failure.Package.Name,
			"vnd.gke-labs.vxpu.package.key":  failure.Package.Key,
		},
	}
	return c.PutManifest(ctx, name, failure.Package.Key, manifest)
}

// GetCompileFailure returns any recorded compilation failure for key under model.
// A failure manifest with no layers is an error.
func (c *Client) GetCompileFailure(ctx context.Context, model, key string) (*planpb.CompileFailure, error) {
	name := FailureRepository(model)
	m, err := c.GetManifest(ctx, name, key)
	if errors.Is(err, ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if m.ArtifactType == failureClearedArtifactType {
		return nil, nil
	}
	if len(m.Layers) == 0 {
		return nil, fmt.Errorf("modelstore failure manifest %s:%s has no layers", name, key)
	}
	data, err := c.FetchBlob(ctx, name, m.Layers[0].Digest)
	if err != nil {
		return nil, err
	}
	var failure planpb.CompileFailure
	if err := protojson.Unmarshal(data, &failure); err != nil {
		return nil, fmt.Errorf("decoding compile failure: %w", err)
	}
	return &failure, nil
}

// ClearCompileFailure clears recorded compilation failure for package key under model.
func (c *Client) ClearCompileFailure(ctx context.Context, model, key string) error {
	repo := FailureRepository(model)
	m, err := c.GetManifest(ctx, repo, key)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if m.ArtifactType == failureClearedArtifactType {
		return nil
	}
	return c.putClearedManifest(ctx, repo, key)
}

// RecordBodyCompileOutcome records the compilation outcome for a kernel body on a machine kind and software version.
// Writing an outcome replaces any previous outcome recorded under the same identity.
func (c *Client) RecordBodyCompileOutcome(ctx context.Context, outcome *planpb.BodyCompileOutcome) error {
	if outcome == nil {
		return errors.New("body compile outcome is required")
	}
	if outcome.Op == "" {
		return errors.New("operator name is required to record body compile outcome")
	}
	if outcome.Body == "" {
		return errors.New("body name is required to record body compile outcome")
	}
	if outcome.MachineKind == "" {
		return errors.New("machine kind is required to record body compile outcome")
	}
	if outcome.SoftwareVersion == "" {
		return errors.New("software version is required to record body compile outcome")
	}
	if !outcome.Compiled && outcome.Reason == "" {
		return errors.New("reason is required when body compilation failed")
	}
	if outcome.Compiled && outcome.Reason != "" {
		outcome = proto.Clone(outcome).(*planpb.BodyCompileOutcome)
		outcome.Reason = ""
	}
	data, err := protojson.Marshal(outcome)
	if err != nil {
		return fmt.Errorf("marshal body compile outcome: %w", err)
	}
	name := BodyOutcomeRepository()
	digest, err := c.PutBlob(ctx, name, data)
	if err != nil {
		return err
	}
	manifest := &Manifest{
		SchemaVersion: 2,
		MediaType:     mediaTypeManifest,
		ArtifactType:  outcomeArtifactType,
		Config:        EmptyConfig,
		Layers: []Layer{{
			MediaType:   MediaTypeFile,
			Digest:      digest,
			Size:        int64(len(data)),
			Annotations: map[string]string{"org.opencontainers.image.title": "body-compile-outcome.json"},
		}},
		Annotations: map[string]string{
			"vnd.gke-labs.vxpu.op.name":          outcome.Op,
			"vnd.gke-labs.vxpu.body.name":        outcome.Body,
			"vnd.gke-labs.vxpu.machine.kind":     outcome.MachineKind,
			"vnd.gke-labs.vxpu.software.version": outcome.SoftwareVersion,
			"vnd.gke-labs.vxpu.body.compiled":    fmt.Sprintf("%t", outcome.Compiled),
		},
	}
	tag := BodyOutcomeTag(outcome.Op, outcome.Body, outcome.MachineKind, outcome.SoftwareVersion)
	return c.PutManifest(ctx, name, tag, manifest)
}

// GetBodyCompileOutcome returns the recorded compilation outcome for op and body on machineKind and softwareVersion.
// Returns (nil, nil) if no outcome has been recorded (not known).
func (c *Client) GetBodyCompileOutcome(ctx context.Context, op, body, machineKind, softwareVersion string) (*planpb.BodyCompileOutcome, error) {
	name := BodyOutcomeRepository()
	tag := BodyOutcomeTag(op, body, machineKind, softwareVersion)
	m, err := c.GetManifest(ctx, name, tag)
	if errors.Is(err, ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if m.ArtifactType != outcomeArtifactType {
		return nil, nil
	}
	if len(m.Layers) == 0 {
		return nil, fmt.Errorf("modelstore outcome manifest %s:%s has no layers", name, tag)
	}
	data, err := c.FetchBlob(ctx, name, m.Layers[0].Digest)
	if err != nil {
		return nil, err
	}
	var outcome planpb.BodyCompileOutcome
	if err := protojson.Unmarshal(data, &outcome); err != nil {
		return nil, fmt.Errorf("decoding body compile outcome: %w", err)
	}
	if outcome.Op != op || outcome.Body != body || outcome.MachineKind != machineKind || outcome.SoftwareVersion != softwareVersion {
		return nil, nil
	}
	return &outcome, nil
}

// ListBodyCompileOutcomes returns all active body compile outcomes recorded for machineKind.
func (c *Client) ListBodyCompileOutcomes(ctx context.Context, machineKind string) ([]*planpb.BodyCompileOutcome, error) {
	repo := BodyOutcomeRepository()
	tags, err := c.ListTags(ctx, repo)
	if errors.Is(err, ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var outcomes []*planpb.BodyCompileOutcome
	for _, tag := range tags {
		m, err := c.GetManifest(ctx, repo, tag)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if m.ArtifactType != outcomeArtifactType {
			continue
		}
		if len(m.Layers) == 0 {
			return nil, fmt.Errorf("modelstore outcome manifest %s:%s has no layers", repo, tag)
		}
		data, err := c.FetchBlob(ctx, repo, m.Layers[0].Digest)
		if err != nil {
			return nil, err
		}
		var outcome planpb.BodyCompileOutcome
		if err := protojson.Unmarshal(data, &outcome); err != nil {
			return nil, fmt.Errorf("decoding body compile outcome from %s: %w", tag, err)
		}
		if machineKind == "" || outcome.MachineKind == machineKind {
			outcomes = append(outcomes, &outcome)
		}
	}
	sort.Slice(outcomes, func(i, j int) bool {
		if outcomes[i].Op != outcomes[j].Op {
			return outcomes[i].Op < outcomes[j].Op
		}
		if outcomes[i].Body != outcomes[j].Body {
			return outcomes[i].Body < outcomes[j].Body
		}
		if outcomes[i].MachineKind != outcomes[j].MachineKind {
			return outcomes[i].MachineKind < outcomes[j].MachineKind
		}
		return outcomes[i].SoftwareVersion < outcomes[j].SoftwareVersion
	})
	return outcomes, nil
}

func (c *Client) putClearedManifest(ctx context.Context, repo, tag string) error {
	emptyData := []byte("{}")
	digest, err := c.PutBlob(ctx, repo, emptyData)
	if err != nil {
		return err
	}
	return c.PutManifest(ctx, repo, tag, &Manifest{
		SchemaVersion: 2,
		MediaType:     mediaTypeManifest,
		ArtifactType:  failureClearedArtifactType,
		Config:        EmptyConfig,
		Layers: []Layer{{
			MediaType:   MediaTypeFile,
			Digest:      digest,
			Size:        int64(len(emptyData)),
			Annotations: map[string]string{"org.opencontainers.image.title": "compile-failure.json"},
		}},
	})
}

// WaitForImport polls the Model object of an import until it is Ready,
// and fails if the import failed.
func (c *Client) WaitForImport(ctx context.Context, name string, every time.Duration) (ModelStatus, error) {
	for {
		status, err := c.GetModelStatus(ctx, name)
		if err != nil {
			return status, err
		}
		switch status.Phase {
		case "Ready":
			return status, nil
		case "Failed":
			return status, fmt.Errorf("modelstore import of %s failed: %s", name, status.Message)
		}
		select {
		case <-ctx.Done():
			return status, ctx.Err()
		case <-time.After(every):
		}
	}
}

// ResolveCommit returns the commit an imported repository's revision
// resolved to (the annotation modelstore's import records), or
// ErrNotFound if the revision was never imported.
func (c *Client) ResolveCommit(ctx context.Context, repo, revision string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/v2/%s/manifests/%s", c.base, RepositoryName(repo), revision), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", mediaTypeManifest)
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("modelstore manifest %s:%s: %w", repo, revision, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode == http.StatusNotFound {
		return "", ErrNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("modelstore manifest %s:%s: %s: %s", repo, revision, resp.Status, errorBody(resp))
	}
	var manifest struct {
		Annotations map[string]string `json:"annotations"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return "", fmt.Errorf("decoding manifest %s:%s: %w", repo, revision, err)
	}
	commit := manifest.Annotations[annotationHubCommit]
	if commit == "" {
		return "", fmt.Errorf("manifest %s:%s records no commit", repo, revision)
	}
	return commit, nil
}

// ManifestExists reports whether repository name has tag.
func (c *Client) ManifestExists(ctx context.Context, name, tag string) (bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, fmt.Sprintf("%s/v2/%s/manifests/%s", c.base, name, tag), nil)
	if err != nil {
		return false, err
	}
	req.Header.Set("Accept", mediaTypeManifest)
	resp, err := c.http.Do(req)
	if err != nil {
		return false, fmt.Errorf("modelstore manifest %s:%s: %w", name, tag, err)
	}
	defer closeBody(ctx, resp)
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	}
	return false, fmt.Errorf("modelstore manifest %s:%s: %s", name, tag, resp.Status)
}

// FetchFile reads one file of a stored repository through modelstore's
// Hub-compatible route (/<name>/resolve/<tag>/<path>).
func (c *Client) FetchFile(ctx context.Context, name, tag, path string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/%s/resolve/%s/%s", c.base, name, tag, path), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelstore %s:%s/%s: %w", name, tag, path, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("modelstore %s:%s/%s: %w", name, tag, path, ErrNotFound)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("modelstore %s:%s/%s: %s: %s", name, tag, path, resp.Status, errorBody(resp))
	}
	return io.ReadAll(resp.Body)
}

// Layer is one file of a stored repository.
type Layer struct {
	MediaType   string            `json:"mediaType"`
	Digest      string            `json:"digest"`
	Size        int64             `json:"size"`
	Annotations map[string]string `json:"annotations,omitempty"`
}

// Manifest is a stored repository's manifest (the fields vxpu uses).
type Manifest struct {
	SchemaVersion int               `json:"schemaVersion"`
	MediaType     string            `json:"mediaType"`
	ArtifactType  string            `json:"artifactType,omitempty"`
	Config        Layer             `json:"config"`
	Layers        []Layer           `json:"layers"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// MediaTypeFile is the media type of a file layer, as modelstore writes them.
const MediaTypeFile = "application/vnd.gke-labs.model.file.v1"

// EmptyConfig is the OCI empty config every vxpu manifest carries.
var EmptyConfig = Layer{MediaType: "application/vnd.oci.empty.v1+json",
	Digest: "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a", Size: 2}

// PutBlob uploads data as a blob of repository name (a no-op when
// modelstore already holds it) and returns its digest.
func (c *Client) PutBlob(ctx context.Context, name string, data []byte) (string, error) {
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/v2/%s/blobs/uploads/?digest=%s", c.base, name, digest), bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("modelstore blob upload to %s: %w", name, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("modelstore blob upload to %s: %s: %s", name, resp.Status, errorBody(resp))
	}
	return digest, nil
}

// GetManifest reads repository name's manifest at tag, or ErrNotFound.
func (c *Client) GetManifest(ctx context.Context, name, tag string) (*Manifest, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/v2/%s/manifests/%s", c.base, name, tag), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", mediaTypeManifest)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelstore manifest %s:%s: %w", name, tag, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("modelstore manifest %s:%s: %s: %s", name, tag, resp.Status, errorBody(resp))
	}
	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, fmt.Errorf("decoding manifest %s:%s: %w", name, tag, err)
	}
	return &m, nil
}

// PutManifest writes repository name's manifest at tag (every layer must
// already be uploaded).
func (c *Client) PutManifest(ctx context.Context, name, tag string, m *Manifest) error {
	body, err := json.Marshal(m)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", c.base, name, tag), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", mediaTypeManifest)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("modelstore manifest %s:%s: %w", name, tag, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("modelstore manifest %s:%s: %s: %s", name, tag, resp.Status, errorBody(resp))
	}
	return nil
}

// ListTags lists repository name's tags (none if it does not exist).
func (c *Client) ListTags(ctx context.Context, name string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/v2/%s/tags/list", c.base, name), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelstore tags of %s: %w", name, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("modelstore tags of %s: %s: %s", name, resp.Status, errorBody(resp))
	}
	var list struct {
		Tags []string `json:"tags"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	return list.Tags, nil
}

// FetchBlob reads a blob by digest.
func (c *Client) FetchBlob(ctx context.Context, name, digest string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/v2/%s/blobs/%s", c.base, name, digest), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelstore blob %s: %w", digest, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("modelstore blob %s: %s: %s", digest, resp.Status, errorBody(resp))
	}
	return io.ReadAll(resp.Body)
}
