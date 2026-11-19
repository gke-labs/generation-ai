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
	"strings"
	"time"
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

// HasPackage reports whether modelstore holds a compiled package for key under model.
func (c *Client) HasPackage(ctx context.Context, model, key string) (bool, error) {
	return c.ManifestExists(ctx, PackageRepository(model), key)
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
