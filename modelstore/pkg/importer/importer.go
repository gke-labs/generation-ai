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

// Package importer brings files into the blob store from the Hugging
// Face Hub (or any URL) without the bytes passing through a client:
// modelstore streams them itself, verifying the sha256 on the way.
package importer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"k8s.io/klog/v2"
)

// Importer streams blobs into a Store, de-duplicating concurrent
// requests for the same blob.
type Importer struct {
	store      blobstore.Store
	hfEndpoint string
	hfToken    string
	client     *http.Client

	mu     sync.Mutex
	jobs   map[string]*job    // by sha
	failed map[string]failure // last failed import by sha, for failureTTL
}

type job struct {
	url  string
	done chan struct{}
	err  error
	n    int64
}

// failure remembers why an import failed, so the next request for the
// same blob from the same url is answered with the reason instead of
// starting the same download again.
type failure struct {
	url string
	err error
	at  time.Time
}

const failureTTL = 10 * time.Minute

// ErrRecentFailure wraps the error of an import that failed within
// failureTTL when the same blob is requested from the same url again.
var ErrRecentFailure = errors.New("a recent import of this blob from this url failed")

// New creates an Importer. hfEndpoint is the Hub base URL (default
// https://huggingface.co); the HF_TOKEN environment variable, when set,
// is sent to it for gated repositories.
func New(store blobstore.Store, hfEndpoint string) *Importer {
	if hfEndpoint == "" {
		hfEndpoint = "https://huggingface.co"
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = 60 * time.Second
	return &Importer{
		store:      store,
		hfEndpoint: strings.TrimSuffix(hfEndpoint, "/"),
		hfToken:    os.Getenv("HF_TOKEN"),
		// No overall timeout (multi-GB downloads); a stalled body is
		// caught by the per-job progress watchdog in fetch.
		client: &http.Client{Transport: transport},
		jobs:   map[string]*job{},
	}
}

// StallTimeout is how long a download may go without delivering a byte
// before its job is cancelled and reported failed.
var StallTimeout = 5 * time.Minute

// ErrBadSource marks a URL modelstore refuses to fetch (a client error).
var ErrBadSource = errors.New("bad source")

// CheckSource says whether modelstore will fetch srcURL: only https
// URLs on the configured Hub host. The import endpoints are reachable
// by anything in the cluster, and this pod holds the bucket
// credentials, so it is not a general-purpose fetcher.
func (im *Importer) CheckSource(srcURL string) error {
	u, err := url.Parse(srcURL)
	if err != nil {
		return fmt.Errorf("%w: invalid url %q: %v", ErrBadSource, srcURL, err)
	}
	h, err := url.Parse(im.hfEndpoint)
	if err != nil {
		return fmt.Errorf("invalid hf endpoint %q: %w", im.hfEndpoint, err)
	}
	if u.Scheme != h.Scheme || u.Host != h.Host {
		return fmt.Errorf("%w: refusing to fetch %q, imports come from %s only", ErrBadSource, srcURL, im.hfEndpoint)
	}
	return nil
}

// Status of a blob: Present with its size, InFlight, or Absent.
type Status int

const (
	Absent Status = iota
	InFlight
	Present
)

// Stat reports whether a blob is present, being imported, or absent.
func (im *Importer) Stat(ctx context.Context, sha string) (Status, int64, error) {
	if size, err := im.store.Stat(ctx, sha); err == nil {
		return Present, size, nil
	} else if !errors.Is(err, blobstore.ErrNotFound) {
		return Absent, 0, err
	}
	im.mu.Lock()
	_, running := im.jobs[sha]
	im.mu.Unlock()
	if running {
		return InFlight, 0, nil
	}
	return Absent, 0, nil
}

// EnsureBlob makes sure blob sha is in the store, fetching it from
// srcURL if needed. It returns as soon as the blob is present or its
// import has started; Wait blocks for completion. Idempotent: a second
// call for a blob in flight joins the existing import.
func (im *Importer) EnsureBlob(ctx context.Context, sha, srcURL string) (Status, error) {
	if !blobstore.ValidSHA(sha) {
		return Absent, fmt.Errorf("invalid sha256 %q", sha)
	}
	st, _, err := im.Stat(ctx, sha)
	if err != nil || st != Absent {
		return st, err
	}
	if err := im.CheckSource(srcURL); err != nil {
		return Absent, err
	}
	im.mu.Lock()
	if _, running := im.jobs[sha]; running {
		im.mu.Unlock()
		return InFlight, nil
	}
	if f, ok := im.failed[sha]; ok && f.url == srcURL && time.Since(f.at) < failureTTL {
		im.mu.Unlock()
		return Absent, fmt.Errorf("%w: %v", ErrRecentFailure, f.err)
	}
	j := &job{url: srcURL, done: make(chan struct{})}
	im.jobs[sha] = j
	im.mu.Unlock()
	go im.run(sha, srcURL, j)
	return InFlight, nil
}

// WaitFor is Wait for a caller that asked for srcURL. An import keyed
// by digest alone could be in flight from another caller's url; if that
// one fails, this caller's url is tried once rather than inheriting a
// failure it never asked for.
func (im *Importer) WaitFor(ctx context.Context, sha, srcURL string) (int64, error) {
	im.mu.Lock()
	j, running := im.jobs[sha]
	im.mu.Unlock()
	n, err := im.Wait(ctx, sha)
	if err == nil || !running || j.url == srcURL || ctx.Err() != nil {
		return n, err
	}
	if _, err := im.EnsureBlob(ctx, sha, srcURL); err != nil {
		return 0, err
	}
	return im.Wait(ctx, sha)
}

// Wait blocks until the import of sha (if any) finishes and returns
// its error. Present blobs return immediately.
func (im *Importer) Wait(ctx context.Context, sha string) (int64, error) {
	im.mu.Lock()
	j, running := im.jobs[sha]
	im.mu.Unlock()
	if !running {
		size, err := im.store.Stat(ctx, sha)
		return size, err
	}
	select {
	case <-j.done:
		return j.n, j.err
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}

func (im *Importer) run(sha, srcURL string, j *job) {
	// The import outlives the request that started it: a dropped client
	// must not abort a 50 GB download that a retry would only restart.
	// It is cancelled only by its own stall watchdog (see fetch).
	ctx := context.Background()
	log := klog.FromContext(ctx).WithValues("sha", sha, "url", srcURL)
	start := time.Now()
	_, n, err := im.fetch(ctx, sha, srcURL)
	j.n, j.err = n, err
	if err != nil {
		log.Error(err, "import failed", "bytes", n)
	} else {
		log.Info("imported blob", "bytes", n, "seconds", int(time.Since(start).Seconds()))
	}
	im.mu.Lock()
	delete(im.jobs, sha)
	if err != nil {
		if im.failed == nil {
			im.failed = map[string]failure{}
		}
		im.failed[sha] = failure{url: srcURL, err: err, at: time.Now()}
	} else {
		delete(im.failed, sha)
	}
	im.mu.Unlock()
	close(j.done)
}

// fetch streams srcURL into the store as blob sha ("" names it by
// content). A download that delivers no byte for StallTimeout is
// cancelled, so a silent peer cannot pin a job forever.
func (im *Importer) fetch(ctx context.Context, sha, srcURL string) (string, int64, error) {
	if err := im.CheckSource(srcURL); err != nil {
		return "", 0, err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srcURL, nil)
	if err != nil {
		return "", 0, err
	}
	if im.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+im.hfToken) // CheckSource: https on the Hub host only
	}
	resp, err := im.client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("fetching %s: %w", srcURL, err)
	}
	defer closeBody(ctx, resp, srcURL)
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("fetching %s: %s", srcURL, resp.Status)
	}
	body := newStallWatch(resp.Body, StallTimeout, func() {
		cancel(fmt.Errorf("no data for %s", StallTimeout))
	})
	defer body.stop()
	got, n, err := im.store.Put(ctx, sha, body)
	if err != nil {
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) {
			err = fmt.Errorf("%w (%v)", err, cause)
		}
		return got, n, fmt.Errorf("storing %s: %w", srcURL, err)
	}
	return got, n, nil
}

func closeBody(ctx context.Context, resp *http.Response, url string) {
	if err := resp.Body.Close(); err != nil {
		klog.FromContext(ctx).Error(err, "closing response body", "url", url)
	}
}

// stallWatch is a reader that calls onStall when no Read has returned
// data for the timeout; each Read that delivers data re-arms it.
type stallWatch struct {
	io.Reader
	timer   *time.Timer
	timeout time.Duration
	stopped atomic.Bool
}

func newStallWatch(r io.Reader, timeout time.Duration, onStall func()) *stallWatch {
	w := &stallWatch{Reader: r, timeout: timeout}
	w.timer = time.AfterFunc(timeout, func() {
		if !w.stopped.Load() {
			onStall()
		}
	})
	return w
}

func (w *stallWatch) Read(p []byte) (int, error) {
	n, err := w.Reader.Read(p)
	if n > 0 {
		w.timer.Reset(w.timeout)
	}
	return n, err
}

func (w *stallWatch) stop() {
	w.stopped.Store(true)
	w.timer.Stop()
}

// File is an imported file: its path in the repository, its sha256
// and its size.
type File struct {
	Path   string
	SHA256 string
	Size   int64
}

// HubFile is one file of a Hub repository at a commit.
type HubFile struct {
	Path   string
	SHA256 string // empty for files the Hub stores outside LFS
	Size   int64
	URL    string
}

// ListRepo resolves a Hub repository revision to its commit and files.
func (im *Importer) ListRepo(ctx context.Context, repo, revision string) (string, []HubFile, error) {
	if revision == "" {
		revision = "main"
	}
	api := im.hfEndpoint + "/api/models/" + escapePath(repo) + "/revision/" + url.PathEscape(revision) + "?blobs=true"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, api, nil)
	if err != nil {
		return "", nil, err
	}
	if im.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+im.hfToken)
	}
	resp, err := im.client.Do(req)
	if err != nil {
		return "", nil, fmt.Errorf("listing %s@%s: %w", repo, revision, err)
	}
	defer closeBody(ctx, resp, api)
	if resp.StatusCode != http.StatusOK {
		body, rerr := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if rerr != nil {
			body = []byte("(unreadable body: " + rerr.Error() + ")")
		}
		return "", nil, fmt.Errorf("listing %s@%s: %s: %s", repo, revision, resp.Status, strings.TrimSpace(string(body)))
	}
	var info struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Rfilename string `json:"rfilename"`
			Size      int64  `json:"size"`
			LFS       *struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", nil, fmt.Errorf("decoding listing of %s@%s: %w", repo, revision, err)
	}
	commit := info.SHA
	if commit == "" {
		commit = revision
	}
	var files []HubFile
	for _, s := range info.Siblings {
		f := HubFile{Path: s.Rfilename, Size: s.Size,
			URL: im.hfEndpoint + "/" + escapePath(repo) + "/resolve/" + url.PathEscape(commit) + "/" + escapePath(s.Rfilename)}
		if s.LFS != nil {
			f.SHA256, f.Size = s.LFS.SHA256, s.LFS.Size
		}
		files = append(files, f)
	}
	return commit, files, nil
}

// escapePath escapes each segment of a slash-separated path (a repo id
// or a file name), keeping the slashes: "#", "?" and "%" in a file name
// must not be read as URL syntax.
func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

// Progress is reported by ImportRepo after each file.
type Progress struct {
	FilesTotal, FilesImported int
	BytesImported             int64
}

// ImportRepo brings every file of a Hub repository revision into the
// store and returns the Model files (with digests) for it. LFS files
// are fetched by their published sha256 and verified; other files
// (configs, tokenizers) are hashed as they stream. report, if set, is
// called after each file.
func (im *Importer) ImportRepo(ctx context.Context, repo, revision string, report func(Progress)) (string, []File, error) {
	commit, files, err := im.ListRepo(ctx, repo, revision)
	if err != nil {
		return "", nil, err
	}
	var out []File
	prog := Progress{FilesTotal: len(files)}
	for _, f := range files {
		sha := f.SHA256
		var n int64
		if sha != "" {
			if _, err := im.EnsureBlob(ctx, sha, f.URL); err != nil {
				return commit, out, err
			}
			var err error
			if n, err = im.WaitFor(ctx, sha, f.URL); err != nil {
				return commit, out, fmt.Errorf("%s: %w", f.Path, err)
			}
		} else {
			got, fetched, err := im.fetch(ctx, "", f.URL)
			if err != nil {
				return commit, out, fmt.Errorf("%s: %w", f.Path, err)
			}
			sha, n = got, fetched
		}
		prog.BytesImported += n
		size := f.Size
		if size == 0 {
			size = n
		}
		out = append(out, File{Path: f.Path, SHA256: sha, Size: size})
		prog.FilesImported++
		if report != nil {
			report(prog)
		}
	}
	return commit, out, nil
}
