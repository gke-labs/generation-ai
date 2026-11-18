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

// Package weightstore is the router's client for modelstore, the
// cluster's content-addressed weight store (modelstore/ in this repo).
//
// The router never moves weight bytes itself: it asks modelstore to
// make sure each file of a manifest is present (modelstore streams it
// from the Hub into its bucket once), and rewrites the manifest's
// sources to modelstore's blob URLs, which executors range-read.
// modelstore is the only endpoint executors know; whether it proxies
// the bucket or redirects to it is its decision, not theirs.
package weightstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"k8s.io/klog/v2"
)

// DefaultURL is where modelstore is reachable when deployed from
// modelstore/k8s/manifest.yaml.
const DefaultURL = "http://modelstore.modelstore"

// Client talks to one modelstore.
type Client struct {
	base string
	http *http.Client
}

// New returns a client for the modelstore at base (e.g. DefaultURL).
func New(base string) *Client {
	return &Client{base: strings.TrimSuffix(base, "/"), http: &http.Client{}}
}

// Discover returns a client for the modelstore named by url, or, when
// url is empty, for DefaultURL if its host resolves in this cluster,
// else nil (the router then caches weights itself).
func Discover(ctx context.Context, url string) *Client {
	log := klog.FromContext(ctx)
	if url != "" {
		return New(url)
	}
	host := strings.TrimPrefix(DefaultURL, "http://")
	// The answer decides the router's behaviour for its lifetime, so a
	// transient resolver fault (cluster DNS not answering yet at pod
	// start) is retried; only a definite "no such host" means no store.
	var err error
	for attempt := 0; attempt < 10; attempt++ {
		lookup, cancel := context.WithTimeout(ctx, 3*time.Second)
		_, err = net.DefaultResolver.LookupHost(lookup, host)
		cancel()
		var dnsErr *net.DNSError
		if err == nil || (errors.As(err, &dnsErr) && dnsErr.IsNotFound) || ctx.Err() != nil {
			break
		}
		log.Info("Resolving modelstore failed, retrying", "host", host, "error", err.Error())
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		log.Info("No modelstore in this cluster; the router caches weights itself", "host", host, "error", err.Error())
		return nil
	}
	log.Info("Using modelstore as the weight store", "url", DefaultURL)
	return New(DefaultURL)
}

// URL is the modelstore base URL.
func (c *Client) URL() string { return c.base }

// Ensure makes blob sha present in the store, importing it from srcURL
// if needed, and returns the modelstore URL executors read it from. It
// blocks until the blob is present; pass a context that outlives the
// caller's request for large files.
func (c *Client) Ensure(ctx context.Context, sha, srcURL string) (string, error) {
	body, err := json.Marshal(map[string]string{"url": srcURL})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/blobs/%s/import?wait=true", c.base, sha), bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("modelstore import of %s: %w", sha, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode == http.StatusNotFound {
		// An older modelstore (before blob import) answers 404 here; say
		// so rather than letting every LoadModel fail with a bare status.
		return "", fmt.Errorf("modelstore at %s has no blob import route (deploy the modelstore from this tree): %s", c.base, errorBody(resp))
	}
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("modelstore import of %s: %s: %s", sha, resp.Status, errorBody(resp))
	}
	return c.BlobURL(sha), nil
}

// BlobURL is where blob sha is read from (GET/HEAD with Range).
func (c *Client) BlobURL(sha string) string {
	return fmt.Sprintf("%s/blobs/%s", c.base, sha)
}

// OpenBlob streams a blob's bytes from the store. It returns an io.ReadCloser
// for reading the blob body and its total size in bytes. The caller must close
// the returned reader when finished.
func (c *Client) OpenBlob(ctx context.Context, sha string) (io.ReadCloser, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BlobURL(sha), nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("modelstore blob %s: %w", sha, err)
	}
	if resp.StatusCode == http.StatusNotFound {
		closeBody(ctx, resp)
		return nil, 0, fmt.Errorf("modelstore blob %s: %w", sha, ErrNotFound)
	}
	if resp.StatusCode != http.StatusOK {
		defer closeBody(ctx, resp)
		return nil, 0, fmt.Errorf("modelstore blob %s: %s: %s", sha, resp.Status, errorBody(resp))
	}
	return resp.Body, resp.ContentLength, nil
}

// ImportRepo asks modelstore to import a Hub repository revision in the
// background and returns the Model object's name.
func (c *Client) ImportRepo(ctx context.Context, repo, revision string) (string, error) {
	body, err := json.Marshal(map[string]string{"repo": repo, "revision": revision})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/models/import", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("modelstore import of %s: %w", repo, err)
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("modelstore import of %s: %s: %s", repo, resp.Status, errorBody(resp))
	}
	var model struct {
		Metadata struct {
			Name string `json:"name"`
		} `json:"metadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&model); err != nil {
		return "", fmt.Errorf("decoding modelstore response: %w", err)
	}
	return model.Metadata.Name, nil
}

// ModelStatus is the progress of an import.
type ModelStatus struct {
	Phase         string `json:"phase"`
	FilesTotal    int    `json:"filesTotal"`
	FilesImported int    `json:"filesImported"`
	BytesImported int64  `json:"bytesImported"`
	Message       string `json:"message"`
}

// GetModelStatus reads a Model's status.
func (c *Client) GetModelStatus(ctx context.Context, name string) (ModelStatus, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/models/"+name, nil)
	if err != nil {
		return ModelStatus{}, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return ModelStatus{}, err
	}
	defer closeBody(ctx, resp)
	if resp.StatusCode != http.StatusOK {
		return ModelStatus{}, fmt.Errorf("modelstore GET %s: %s", name, resp.Status)
	}
	var model struct {
		Status ModelStatus `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&model); err != nil {
		return ModelStatus{}, err
	}
	return model.Status, nil
}

// errorBody is the start of an error response's body, for messages.
func errorBody(resp *http.Response) string {
	msg, err := io.ReadAll(io.LimitReader(resp.Body, 2048))
	if err != nil {
		return "(unreadable body: " + err.Error() + ")"
	}
	return strings.TrimSpace(string(msg))
}

func closeBody(ctx context.Context, resp *http.Response) {
	if err := resp.Body.Close(); err != nil {
		klog.FromContext(ctx).Error(err, "closing modelstore response")
	}
}
