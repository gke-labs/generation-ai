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

// Package proxy is modelstore's HTTP front: the OCI distribution API
// (pkg/registry) for blobs, manifests and tags; imports from the
// Hugging Face Hub that end as OCI artifacts; a Hub-compatible read API
// served from manifest annotations so that transformers and vLLM load
// through it with HF_ENDPOINT set; and the plain /blobs/<sha> routes
// the vxpu router uses today.
package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"strings"
	"sync"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	"github.com/gke-labs/generation-ai/modelstore/pkg/registry"
	"k8s.io/klog/v2"
)

// Import phases.
const (
	PhaseImporting = "Importing"
	PhaseReady     = "Ready"
	PhaseFailed    = "Failed"
)

// ImportStatus is the progress of a repository import, kept in memory
// for the process that runs it; the durable result is the manifest and
// its tags in the registry.
type ImportStatus struct {
	Phase         string `json:"phase"`
	Repo          string `json:"repo,omitempty"`
	Revision      string `json:"revision,omitempty"`
	FilesTotal    int    `json:"filesTotal,omitempty"`
	FilesImported int    `json:"filesImported,omitempty"`
	BytesImported int64  `json:"bytesImported,omitempty"`
	Message       string `json:"message,omitempty"`
	Digest        string `json:"digest,omitempty"`
}

type Proxy struct {
	reg      *registry.Registry
	importer *importer.Importer

	mu      sync.Mutex
	imports map[string]*ImportStatus // repository imports in this process, by name
}

// NewProxy serves the registry and imports through imp.
func NewProxy(reg *registry.Registry, imp *importer.Importer) *Proxy {
	return &Proxy{reg: reg, importer: imp, imports: map[string]*ImportStatus{}}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rw := &responseWriter{ResponseWriter: w}
	err := p.serve(rw, r)
	if err == nil {
		return
	}
	log := klog.FromContext(r.Context())
	if rw.started {
		// Headers are out: the error can only be reported by cutting the
		// body short. Most of these are clients closing a large download
		// early, which is not a server failure.
		log.V(2).Info("response cut short", "path", r.URL.Path, "error", err.Error())
		return
	}
	log.Error(err, "request failed", "path", r.URL.Path, "method", r.Method)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// responseWriter remembers whether a status line has been written.
type responseWriter struct {
	http.ResponseWriter
	started bool
}

func (w *responseWriter) WriteHeader(code int) {
	w.started = true
	w.ResponseWriter.WriteHeader(code)
}

func (w *responseWriter) Write(b []byte) (int, error) {
	w.started = true
	return w.ResponseWriter.Write(b)
}

func (p *Proxy) serve(w http.ResponseWriter, r *http.Request) error {
	if strings.HasPrefix(r.URL.Path, "/debug/pprof") {
		http.DefaultServeMux.ServeHTTP(w, r)
		return nil
	}

	if handled, err := p.reg.Handle(w, r); handled {
		return err
	}

	// Plain blob routes: what the vxpu router rewrites sources to today.
	if rest, ok := strings.CutPrefix(r.URL.Path, "/blobs/"); ok {
		sha, action, _ := strings.Cut(rest, "/")
		if !blobstore.ValidSHA(sha) {
			http.Error(w, "blob names are lowercase hex sha256 digests", http.StatusBadRequest)
			return nil
		}
		switch {
		case action == "" && r.Method == http.MethodPut:
			return p.handleBlobPut(w, r, sha)
		case action == "" && (r.Method == http.MethodGet || r.Method == http.MethodHead):
			return p.reg.ServeBlob(w, r, sha)
		case action == "import" && r.Method == http.MethodPost:
			return p.handleBlobImport(w, r, sha)
		}
		http.NotFound(w, r)
		return nil
	}

	if r.URL.Path == "/models" || strings.HasPrefix(r.URL.Path, "/models/") {
		switch {
		case r.URL.Path == "/models/import" && r.Method == http.MethodPost:
			return p.handleModelImport(w, r)
		case r.Method == http.MethodGet && (r.URL.Path == "/models" || r.URL.Path == "/models/"):
			return p.handleModelList(w, r)
		case r.Method == http.MethodGet:
			return p.handleModelGet(w, r)
		}
		http.NotFound(w, r)
		return nil
	}

	// Hugging Face Hub compatible reads: metadata, tree, then downloads.
	if strings.HasPrefix(r.URL.Path, "/api/models/") {
		rest := strings.TrimPrefix(r.URL.Path, "/api/models/")
		if i := strings.Index(rest, "/tree/"); i >= 0 {
			return p.handleHFTree(w, r, rest[:i], rest[i+len("/tree/"):])
		}
		if i := strings.Index(rest, "/paths-info/"); i >= 0 {
			return p.handleHFPathsInfo(w, r, rest[:i], rest[i+len("/paths-info/"):])
		}
		return p.handleHFMetadata(w, r)
	}
	if strings.Contains(r.URL.Path, "/resolve/") && (r.Method == http.MethodGet || r.Method == http.MethodHead) {
		return p.handleHFDownload(w, r)
	}

	http.NotFound(w, r)
	return nil
}

// RepositoryName is the registry name for a Hub repo id: lowercase,
// with "/" kept and anything the OCI name grammar rejects collapsed to
// "-" ("Qwen/Qwen3-8B-FP8" becomes "qwen/qwen3-8b-fp8"). The exact id
// is kept in the manifest's annotations.
func RepositoryName(repoID string) (string, error) {
	var b strings.Builder
	for _, c := range strings.ToLower(repoID) {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '.', c == '_', c == '-', c == '/':
			b.WriteRune(c)
		default:
			b.WriteRune('-')
		}
	}
	name := b.String()
	if !registry.ValidName(name) {
		return "", fmt.Errorf("repository %q gives an invalid registry name %q", repoID, name)
	}
	return name, nil
}

// ---- Hub-compatible reads

// artifactFiles resolves a repository reference (tag or digest) to the
// files of its weights manifest. An index resolves to the manifest
// whose artifactType is the weights one (or the first manifest).
func (p *Proxy) artifactFiles(ctx context.Context, name, ref string) (*registry.Manifest, string, error) {
	body, mediaType, digest, err := p.reg.GetManifest(ctx, name, ref)
	if errors.Is(err, blobstore.ErrNotFound) && ref == "main" {
		// Hub clients ask for the default branch, "main"; registry
		// pushes (modelstore upload, oras, docker) tag "latest" by
		// default. A name with no "main" answers for its "latest".
		body, mediaType, digest, err = p.reg.GetManifest(ctx, name, "latest")
	}
	if err != nil {
		return nil, "", err
	}
	if mediaType == registry.MediaTypeIndex {
		var idx registry.Index
		if err := json.Unmarshal(body, &idx); err != nil {
			return nil, "", fmt.Errorf("index %s: %w", digest, err)
		}
		pick := ""
		for _, d := range idx.Manifests {
			if d.ArtifactType == registry.ArtifactWeights || pick == "" {
				pick = d.Digest
			}
		}
		if pick == "" {
			return nil, "", blobstore.ErrNotFound
		}
		body, _, digest, err = p.reg.GetManifest(ctx, name, pick)
		if err != nil {
			return nil, "", err
		}
	}
	var m registry.Manifest
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, "", fmt.Errorf("manifest %s: %w", digest, err)
	}
	return &m, digest, nil
}

func revisionOf(m *registry.Manifest) string {
	if c := m.Annotations[registry.AnnotationHubSHA]; c != "" {
		return c
	}
	return "main"
}

// hubNotFound answers 404 the way the Hub does. huggingface_hub reads
// X-Error-Code to tell a missing repository (RepoNotFound) or revision
// (RevisionNotFound) from a missing file (EntryNotFound); transformers
// treats the last as an optional file that is absent, and anything
// else as a server it could not reach.
func hubNotFound(w http.ResponseWriter, code string) {
	w.Header().Set("X-Error-Code", code)
	http.Error(w, code, http.StatusNotFound)
}

// hubMissing is hubNotFound for a repository the registry does not
// have at that revision: RevisionNotFound when the name has tags,
// RepoNotFound when it has none.
func (p *Proxy) hubMissing(ctx context.Context, w http.ResponseWriter, name string) error {
	tags, err := p.reg.Tags(ctx, name)
	if err != nil {
		return err
	}
	if len(tags) > 0 {
		hubNotFound(w, "RevisionNotFound")
	} else {
		hubNotFound(w, "RepoNotFound")
	}
	return nil
}

func (p *Proxy) handleHFMetadata(w http.ResponseWriter, r *http.Request) error {
	repoID := strings.TrimPrefix(r.URL.Path, "/api/models/")
	revision := "main"
	if i := strings.Index(repoID, "/revision/"); i >= 0 {
		repoID, revision = repoID[:i], repoID[i+len("/revision/"):]
	}
	name, err := RepositoryName(repoID)
	if err != nil {
		hubNotFound(w, "RepoNotFound")
		return nil
	}
	m, _, err := p.artifactFiles(r.Context(), name, revision)
	if errors.Is(err, blobstore.ErrNotFound) {
		return p.hubMissing(r.Context(), w, name)
	}
	if err != nil {
		return err
	}
	// Every file is reported as an LFS file whose sha256 is its blob
	// digest (what the Hub does for weights, and what model_info(
	// files_metadata=True) callers such as vxpu's manifest builder read).
	siblings := make([]map[string]any, 0, len(m.Layers))
	for _, l := range m.Layers {
		sha, _ := registry.ParseDigest(l.Digest)
		s := map[string]any{"rfilename": l.Annotations[registry.AnnotationTitle], "size": l.Size, "blobId": sha,
			"lfs": map[string]any{"sha256": sha, "size": l.Size, "pointerSize": 134}}
		siblings = append(siblings, s)
	}
	resp := map[string]any{"modelId": repoID, "id": repoID, "siblings": siblings, "sha": revisionOf(m)}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// hfEntry is one file in the Hub's tree / paths-info responses.
type hfEntry struct {
	Type string `json:"type"`
	Oid  string `json:"oid"`
	Size int64  `json:"size"`
	Path string `json:"path"`
	LFS  *struct {
		Oid         string `json:"oid"`
		Size        int64  `json:"size"`
		PointerSize int    `json:"pointerSize"`
	} `json:"lfs,omitempty"`
}

func hfEntries(m *registry.Manifest) []hfEntry {
	entries := make([]hfEntry, 0, len(m.Layers))
	for _, l := range m.Layers {
		sha, _ := registry.ParseDigest(l.Digest)
		e := hfEntry{Type: "file", Oid: sha, Size: l.Size, Path: l.Annotations[registry.AnnotationTitle]}
		e.LFS = &struct {
			Oid         string `json:"oid"`
			Size        int64  `json:"size"`
			PointerSize int    `json:"pointerSize"`
		}{Oid: sha, Size: l.Size, PointerSize: 134}
		entries = append(entries, e)
	}
	return entries
}

// handleHFTree answers GET /api/models/<repo>/tree/<revision>[/<path>],
// which huggingface_hub uses to list a repository (transformers lists
// before it fetches sharded checkpoints). Every file is reported as an
// LFS file whose oid is its sha256, which is what the Hub does for
// weights and is harmless for small files.
func (p *Proxy) handleHFTree(w http.ResponseWriter, r *http.Request, repoID, revAndPath string) error {
	revision, sub, _ := strings.Cut(revAndPath, "/")
	name, err := RepositoryName(repoID)
	if err != nil {
		hubNotFound(w, "RepoNotFound")
		return nil
	}
	m, _, err := p.artifactFiles(r.Context(), name, revision)
	if errors.Is(err, blobstore.ErrNotFound) {
		return p.hubMissing(r.Context(), w, name)
	}
	if err != nil {
		return err
	}
	var out []hfEntry
	for _, e := range hfEntries(m) {
		if sub == "" || strings.HasPrefix(e.Path, strings.TrimSuffix(sub, "/")+"/") {
			out = append(out, e)
		}
	}
	if out == nil {
		out = []hfEntry{}
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(out)
}

// handleHFPathsInfo answers POST /api/models/<repo>/paths-info/<revision>
// with form field paths=...: the entries for the named files.
func (p *Proxy) handleHFPathsInfo(w http.ResponseWriter, r *http.Request, repoID, revision string) error {
	if r.Method != http.MethodPost {
		http.NotFound(w, r)
		return nil
	}
	name, err := RepositoryName(repoID)
	if err != nil {
		hubNotFound(w, "RepoNotFound")
		return nil
	}
	m, _, err := p.artifactFiles(r.Context(), name, revision)
	if errors.Is(err, blobstore.ErrNotFound) {
		return p.hubMissing(r.Context(), w, name)
	}
	if err != nil {
		return err
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil
	}
	want := map[string]bool{}
	for _, path := range r.Form["paths"] {
		want[path] = true
	}
	out := []hfEntry{}
	for _, e := range hfEntries(m) {
		if want[e.Path] {
			out = append(out, e)
		}
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(out)
}

func (p *Proxy) handleHFDownload(w http.ResponseWriter, r *http.Request) error {
	before, revAndPath, _ := strings.Cut(r.URL.Path, "/resolve/")
	repoID := strings.TrimPrefix(before, "/")
	revision, path, ok := strings.Cut(revAndPath, "/")
	if repoID == "" || !ok || path == "" {
		http.NotFound(w, r)
		return nil
	}
	name, err := RepositoryName(repoID)
	if err != nil {
		hubNotFound(w, "RepoNotFound")
		return nil
	}
	m, _, err := p.artifactFiles(r.Context(), name, revision)
	if errors.Is(err, blobstore.ErrNotFound) {
		return p.hubMissing(r.Context(), w, name)
	}
	if err != nil {
		return err
	}
	for _, l := range m.Layers {
		if l.Annotations[registry.AnnotationTitle] == path {
			sha, ok := registry.ParseDigest(l.Digest)
			if !ok {
				return fmt.Errorf("bad digest %q in manifest for %s", l.Digest, name)
			}
			// huggingface_hub reads these to decide on caching.
			w.Header().Set("X-Repo-Commit", revisionOf(m))
			w.Header().Set("X-Linked-Etag", `"`+sha+`"`)
			w.Header().Set("ETag", `"`+sha+`"`)
			return p.reg.ServeBlob(w, r, sha)
		}
	}
	hubNotFound(w, "EntryNotFound")
	return nil
}

// ---- plain blob routes

func (p *Proxy) handleBlobPut(w http.ResponseWriter, r *http.Request, sha string) error {
	log := klog.FromContext(r.Context())
	_, n, err := p.reg.Store().Put(r.Context(), sha, r.Body)
	if err != nil {
		if strings.Contains(err.Error(), "sha256 mismatch") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return nil
		}
		return err
	}
	log.Info("stored blob", "sha", sha, "bytes", n)
	w.WriteHeader(http.StatusCreated)
	return nil
}

// BlobImportRequest asks modelstore to fetch a blob itself.
type BlobImportRequest struct {
	URL string `json:"url"`
}

// handleBlobImport starts (or joins) the import of one blob from a URL.
// 201 when the blob is present, 202 while it is being fetched; the
// client polls HEAD /blobs/<sha> or re-POSTs (both idempotent).
func (p *Proxy) handleBlobImport(w http.ResponseWriter, r *http.Request, sha string) error {
	var req BlobImportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.URL == "" {
		http.Error(w, `body must be {"url": "..."}`, http.StatusBadRequest)
		return nil
	}
	st, err := p.importer.EnsureBlob(r.Context(), sha, req.URL)
	if errors.Is(err, importer.ErrBadSource) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil
	}
	if errors.Is(err, importer.ErrRecentFailure) {
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
		return nil
	}
	if err != nil {
		return err
	}
	if r.URL.Query().Get("wait") == "true" {
		if _, err := p.importer.WaitFor(r.Context(), sha, req.URL); err != nil {
			// The fetch the client asked for did not produce the blob it
			// named: the client's problem (422) or the source's (502),
			// not this server's.
			code := http.StatusBadGateway
			if strings.Contains(err.Error(), "sha256 mismatch") || errors.Is(err, importer.ErrRecentFailure) {
				code = http.StatusUnprocessableEntity
			}
			http.Error(w, err.Error(), code)
			return nil
		}
		st = importer.Present
	}
	if st == importer.Present {
		w.WriteHeader(http.StatusCreated)
	} else {
		w.WriteHeader(http.StatusAccepted)
	}
	return nil
}

// ---- repository imports

// ModelImportRequest asks modelstore to import a Hub repository.
type ModelImportRequest struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision,omitempty"`
}

// modelView is the response shape of the import routes: the registry
// name, the status, and the tags the artifact is reachable by.
type modelView struct {
	Metadata struct {
		Name string `json:"name"`
	} `json:"metadata"`
	Status ImportStatus `json:"status"`
	Tags   []string     `json:"tags,omitempty"`
}

func (p *Proxy) view(ctx context.Context, name string) (modelView, error) {
	var v modelView
	v.Metadata.Name = name
	tags, err := p.reg.Tags(ctx, name)
	if err != nil {
		return v, err
	}
	v.Tags = tags
	p.mu.Lock()
	st := p.imports[name]
	p.mu.Unlock()
	switch {
	case st != nil:
		v.Status = *st
	case len(tags) > 0:
		v.Status = ImportStatus{Phase: PhaseReady}
	}
	return v, nil
}

// handleModelImport imports a Hub repository revision as a weights
// artifact: every file becomes a blob, the manifest lists them with
// their names, and the tags <revision> and <commit> point at it. The
// import runs in the background; 202 returns at once and GET
// /models/<name> reports progress. Re-importing an existing revision
// skips the blobs already present.
func (p *Proxy) handleModelImport(w http.ResponseWriter, r *http.Request) error {
	var req ModelImportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Repo == "" {
		http.Error(w, `body must be {"repo": "org/name", "revision": "main"}`, http.StatusBadRequest)
		return nil
	}
	if req.Revision == "" {
		req.Revision = "main"
	}
	if !registry.ValidTag(req.Revision) {
		// checked before a download, not after: the revision becomes the tag
		http.Error(w, fmt.Sprintf("revision %q is not a valid tag (a Hub revision with '/' such as refs/pr/1 is not supported; use the commit sha)", req.Revision), http.StatusBadRequest)
		return nil
	}
	name, err := RepositoryName(req.Repo)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return nil
	}
	// Names are lossy (case folded): never let one repository write a
	// manifest under a name another repository already owns.
	if owner, err := p.owner(r.Context(), name); err != nil {
		return err
	} else if owner != "" && owner != req.Repo {
		http.Error(w, fmt.Sprintf("registry name %s already holds %s; cannot import %s under it", name, owner, req.Repo), http.StatusConflict)
		return nil
	}
	p.mu.Lock()
	if st := p.imports[name]; st != nil && st.Phase == PhaseImporting {
		if st.Repo != req.Repo || st.Revision != req.Revision {
			// one import per name at a time; a second revision would
			// otherwise be answered 202 and never run
			p.mu.Unlock()
			http.Error(w, fmt.Sprintf("import of %s@%s is in progress under %s; retry when it is Ready", st.Repo, st.Revision, name), http.StatusConflict)
			return nil
		}
		p.mu.Unlock()
		v, err := p.view(r.Context(), name)
		if err != nil {
			return err
		}
		w.WriteHeader(http.StatusAccepted)
		return json.NewEncoder(w).Encode(v)
	}
	st := &ImportStatus{Phase: PhaseImporting, Repo: req.Repo, Revision: req.Revision}
	p.imports[name] = st
	p.mu.Unlock()
	go p.runModelImport(name, req, st)
	v, err := p.view(r.Context(), name)
	if err != nil {
		return err
	}
	w.WriteHeader(http.StatusAccepted)
	return json.NewEncoder(w).Encode(v)
}

// owner returns the Hub repo id recorded on the name's manifests, if any.
func (p *Proxy) owner(ctx context.Context, name string) (string, error) {
	tags, err := p.reg.Tags(ctx, name)
	if err != nil || len(tags) == 0 {
		return "", err
	}
	m, _, err := p.artifactFiles(ctx, name, tags[0])
	if errors.Is(err, blobstore.ErrNotFound) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return m.Annotations[registry.AnnotationHubRepo], nil
}

func (p *Proxy) runModelImport(name string, req ModelImportRequest, st *ImportStatus) {
	ctx := context.Background()
	log := klog.FromContext(ctx).WithValues("name", name, "repo", req.Repo, "revision", req.Revision)
	set := func(f func(*ImportStatus)) {
		p.mu.Lock()
		f(st)
		p.mu.Unlock()
	}
	report := func(prog importer.Progress) {
		set(func(s *ImportStatus) {
			s.FilesTotal, s.FilesImported, s.BytesImported = prog.FilesTotal, prog.FilesImported, prog.BytesImported
		})
	}
	start := time.Now()
	commit, files, err := p.importer.ImportRepo(ctx, req.Repo, req.Revision, report)
	if err == nil {
		var digest string
		digest, err = p.writeWeightsArtifact(ctx, name, req, commit, files)
		if err == nil {
			set(func(s *ImportStatus) { s.Phase, s.Digest = PhaseReady, digest })
			log.Info("imported repository", "commit", commit, "files", len(files), "digest", digest,
				"seconds", int(time.Since(start).Seconds()))
			return
		}
	}
	log.Error(err, "import failed")
	set(func(s *ImportStatus) { s.Phase, s.Message = PhaseFailed, err.Error() })
}

// writeWeightsArtifact records an imported repository as an OCI
// artifact: one layer per file (annotated with its name), tagged with
// the revision asked for and the commit it resolved to.
func (p *Proxy) writeWeightsArtifact(ctx context.Context, name string, req ModelImportRequest, commit string, files []importer.File) (string, error) {
	m := registry.Manifest{
		SchemaVersion: 2,
		MediaType:     registry.MediaTypeManifest,
		ArtifactType:  registry.ArtifactWeights,
		Config:        registry.EmptyConfig,
		Annotations: map[string]string{
			registry.AnnotationHubRepo: req.Repo,
			registry.AnnotationHubRev:  req.Revision,
			registry.AnnotationHubSHA:  commit,
		},
	}
	for _, f := range files {
		m.Layers = append(m.Layers, registry.Descriptor{
			MediaType:   registry.MediaTypeFile,
			Digest:      "sha256:" + f.SHA256,
			Size:        f.Size,
			Annotations: map[string]string{registry.AnnotationTitle: f.Path},
		})
	}
	body, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	// The commit tag first: the revision tag is what makes the artifact
	// Ready and what the facade's "sha" points clients at, so it must
	// never exist without the commit it names.
	if commit != "" && commit != req.Revision {
		if _, err := p.reg.PutManifest(ctx, name, commit, registry.MediaTypeManifest, body); err != nil {
			return "", fmt.Errorf("tagging commit: %w", err)
		}
	}
	digest, err := p.reg.PutManifest(ctx, name, req.Revision, registry.MediaTypeManifest, body)
	if err != nil {
		return "", fmt.Errorf("writing manifest: %w", err)
	}
	return digest, nil
}

func (p *Proxy) handleModelList(w http.ResponseWriter, r *http.Request) error {
	names, err := p.reg.Repositories(r.Context())
	if err != nil {
		return err
	}
	p.mu.Lock()
	for name, st := range p.imports {
		if st.Phase == PhaseImporting {
			names = append(names, name)
		}
	}
	p.mu.Unlock()
	seen := map[string]bool{}
	var items []modelView
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		v, err := p.view(r.Context(), name)
		if err != nil {
			return err
		}
		items = append(items, v)
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(map[string]any{"items": items})
}

func (p *Proxy) handleModelGet(w http.ResponseWriter, r *http.Request) error {
	name := strings.TrimPrefix(r.URL.Path, "/models/")
	if !registry.ValidName(name) {
		http.NotFound(w, r)
		return nil
	}
	v, err := p.view(r.Context(), name)
	if err != nil {
		return err
	}
	if v.Status.Phase == "" {
		http.NotFound(w, r)
		return nil
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(v)
}
