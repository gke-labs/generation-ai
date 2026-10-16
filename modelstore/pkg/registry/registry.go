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

// Package registry serves the OCI distribution API over a blob store:
// a model is an OCI artifact whose blobs are its weight files and its
// vxpu graphs, addressed by the same sha256 the Hub and the vxpu
// manifest use, and an index per repository and revision names one
// manifest per executor method.
//
// This is deliberately a subset of the specification, enough for
// `oras`/`crane` to push and for executors to range-read: blob HEAD/GET
// with a single Range; the two-step monolithic upload (POST for a
// session, PUT with the digest and the body) and the single-POST form;
// manifest PUT/GET/HEAD for image manifests and indexes, with every
// referenced blob checked to exist; tag listing. Chunked uploads and
// deletes are refused with 405.
package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/google/uuid"
	"k8s.io/klog/v2"
)

// Media types used by modelstore's own artifacts.
const (
	MediaTypeManifest  = "application/vnd.oci.image.manifest.v1+json"
	MediaTypeIndex     = "application/vnd.oci.image.index.v1+json"
	MediaTypeEmpty     = "application/vnd.oci.empty.v1+json"
	MediaTypeFile      = "application/vnd.gke-labs.model.file.v1"
	ArtifactWeights    = "application/vnd.gke-labs.model.weights.v1+json"
	AnnotationTitle    = "org.opencontainers.image.title"
	AnnotationHubRepo  = "vnd.gke-labs.huggingface.repo"
	AnnotationHubRev   = "vnd.gke-labs.huggingface.revision"
	AnnotationHubSHA   = "vnd.gke-labs.huggingface.commit"
	maxManifestBytes   = 4 << 20
	headerDigest       = "Docker-Content-Digest"
	headerDirectURL    = "X-Modelstore-Direct-Url"
	emptyDescriptorSHA = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" // sha256("{}")
)

// ValidTag reports whether ref is a tag the registry accepts (the
// distribution grammar; a Hub revision such as "refs/pr/1" is not).
func ValidTag(ref string) bool { return tagRe.MatchString(ref) }

// Descriptor is an OCI content descriptor.
type Descriptor struct {
	MediaType    string            `json:"mediaType"`
	Digest       string            `json:"digest"`
	Size         int64             `json:"size"`
	ArtifactType string            `json:"artifactType,omitempty"`
	Annotations  map[string]string `json:"annotations,omitempty"`
}

// Manifest is an OCI image manifest (artifact form).
type Manifest struct {
	SchemaVersion int               `json:"schemaVersion"`
	MediaType     string            `json:"mediaType"`
	ArtifactType  string            `json:"artifactType,omitempty"`
	Config        Descriptor        `json:"config"`
	Layers        []Descriptor      `json:"layers"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// Index is an OCI image index.
type Index struct {
	SchemaVersion int               `json:"schemaVersion"`
	MediaType     string            `json:"mediaType"`
	ArtifactType  string            `json:"artifactType,omitempty"`
	Manifests     []Descriptor      `json:"manifests"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// EmptyConfig is the descriptor artifacts use when they have no config.
var EmptyConfig = Descriptor{MediaType: MediaTypeEmpty, Digest: "sha256:" + emptyDescriptorSHA, Size: 2}

var (
	nameRe = regexp.MustCompile(`^[a-z0-9]+(?:(?:\.|_|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:\.|_|__|-+)[a-z0-9]+)*)*$`)
	tagRe  = regexp.MustCompile(`^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$`)
)

// ValidName reports whether name is a valid repository name.
func ValidName(name string) bool { return len(name) <= 255 && nameRe.MatchString(name) }

// ParseDigest returns the hex of a sha256:<hex> digest.
func ParseDigest(d string) (string, bool) {
	hex, ok := strings.CutPrefix(d, "sha256:")
	return hex, ok && blobstore.ValidSHA(hex)
}

// Registry is the server.
type Registry struct {
	store blobstore.Store

	mu       sync.Mutex
	sessions map[string]uploadSession // two-step uploads awaiting their PUT
}

// New serves the distribution API over store.
func New(store blobstore.Store) *Registry {
	return &Registry{store: store, sessions: map[string]uploadSession{}}
}

// uploadSession is a two-step upload between its POST and its PUT.
// Sessions a client abandons are dropped after sessionTTL.
type uploadSession struct {
	name    string
	created time.Time
}

const sessionTTL = time.Hour

// Store is the registry's blob store.
func (r *Registry) Store() blobstore.Store { return r.store }

// Handle serves a /v2/ request. It returns false if the path is not
// the registry's.
func (reg *Registry) Handle(w http.ResponseWriter, r *http.Request) (bool, error) {
	if r.URL.Path == "/v2" || r.URL.Path == "/v2/" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Docker-Distribution-API-Version", "registry/2.0")
		_, err := w.Write([]byte("{}"))
		return true, err
	}
	rest, ok := strings.CutPrefix(r.URL.Path, "/v2/")
	if !ok {
		return false, nil
	}
	// <name>/{blobs,manifests,tags}/...: the name may contain slashes,
	// and even the reserved words as components ("someone/blobs"), but
	// the argument (a digest, tag or session id) never contains a slash,
	// so the route is the reserved component nearest the end of the path.
	kind, at := "", -1
	for _, k := range []string{"/blobs/uploads/", "/blobs/", "/manifests/", "/tags/list"} {
		i := strings.LastIndex(rest, k)
		if i < 0 || (k == "/tags/list" && i+len(k) != len(rest)) { // tags/list takes no argument
			continue
		}
		if i > at || (i == at && k == "/blobs/uploads/") {
			kind, at = k, i
		}
	}
	if kind == "" {
		return true, reg.fail(w, http.StatusNotFound, "UNSUPPORTED", "no such route")
	}
	name, arg := rest[:at], rest[at+len(kind):]
	if !ValidName(name) {
		return true, reg.fail(w, http.StatusBadRequest, "NAME_INVALID", "invalid repository name %q", name)
	}
	switch kind {
	case "/blobs/uploads/":
		return true, reg.handleUpload(w, r, name, arg)
	case "/blobs/":
		return true, reg.handleBlob(w, r, name, arg)
	case "/manifests/":
		return true, reg.handleManifest(w, r, name, arg)
	}
	return true, reg.handleTags(w, r, name)
}

// ociError is the distribution API's error body.
type ociError struct {
	Errors []struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

func (reg *Registry) fail(w http.ResponseWriter, status int, code, format string, args ...any) error {
	var body ociError
	body.Errors = append(body.Errors, struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}{code, fmt.Sprintf(format, args...)})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	return json.NewEncoder(w).Encode(body)
}

// ---- blobs

func (reg *Registry) handleBlob(w http.ResponseWriter, r *http.Request, name, digest string) error {
	sha, ok := ParseDigest(digest)
	if !ok {
		return reg.fail(w, http.StatusBadRequest, "DIGEST_INVALID", "digest %q is not sha256:<hex>", digest)
	}
	switch r.Method {
	case http.MethodGet, http.MethodHead:
		w.Header().Set(headerDigest, digest)
		return reg.ServeBlob(w, r, sha)
	case http.MethodDelete:
		return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "blob deletion is not supported")
	}
	return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "%s not supported on blobs", r.Method)
}

// ServeBlob answers GET/HEAD for one blob by its hex sha256, with
// single-range Range support (executors read one tensor per request).
// A missing blob is a 404 in the registry's error format.
func (reg *Registry) ServeBlob(w http.ResponseWriter, r *http.Request, sha string) error {
	ctx := r.Context()
	size, err := reg.store.Stat(ctx, sha)
	if errors.Is(err, blobstore.ErrNotFound) {
		return reg.fail(w, http.StatusNotFound, "BLOB_UNKNOWN", "blob sha256:%s unknown", sha)
	}
	if err != nil {
		return fmt.Errorf("stat blob %s: %w", sha, err)
	}
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Content-Type", "application/octet-stream")
	u, err := reg.store.DirectURL(ctx, sha)
	if err != nil {
		klog.FromContext(ctx).Error(err, "generating direct URL for blob", "sha", sha)
	}
	if u != "" {
		w.Header().Set(headerDirectURL, u)
		if r.URL.Query().Get("redirect") == "1" {
			http.Redirect(w, r, u, http.StatusTemporaryRedirect)
			return nil
		}
	}
	offset, length := int64(0), size
	status := http.StatusOK
	if rng := r.Header.Get("Range"); rng != "" {
		o, l, ok := parseRange(rng, size)
		if !ok {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", size))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return nil
		}
		offset, length, status = o, l, http.StatusPartialContent
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", offset, offset+length-1, size))
	}
	w.Header().Set("Content-Length", strconv.FormatInt(length, 10))
	if r.Method == http.MethodHead {
		w.WriteHeader(status)
		return nil
	}
	rc, err := reg.store.Open(ctx, sha, offset, length)
	if err != nil {
		return fmt.Errorf("open blob %s: %w", sha, err)
	}
	defer func() {
		if err := rc.Close(); err != nil {
			klog.FromContext(ctx).Error(err, "closing blob", "sha", sha)
		}
	}()
	w.WriteHeader(status)
	_, err = io.CopyN(w, rc, length)
	return err
}

// parseRange handles a single byte range: bytes=a-b, bytes=a-, bytes=-n.
// Returns offset, length and whether the range is satisfiable.
func parseRange(h string, size int64) (int64, int64, bool) {
	spec, ok := strings.CutPrefix(strings.TrimSpace(h), "bytes=")
	if !ok || strings.Contains(spec, ",") {
		return 0, 0, false
	}
	first, last, ok := strings.Cut(spec, "-")
	if !ok {
		return 0, 0, false
	}
	first, last = strings.TrimSpace(first), strings.TrimSpace(last)
	if first == "" { // suffix: last n bytes
		n, err := strconv.ParseInt(last, 10, 64)
		if err != nil || n <= 0 {
			return 0, 0, false
		}
		if n > size {
			n = size
		}
		if n == 0 { // a suffix of an empty blob: nothing to satisfy
			return 0, 0, false
		}
		return size - n, n, true
	}
	start, err := strconv.ParseInt(first, 10, 64)
	if err != nil || start < 0 || start >= size {
		return 0, 0, false
	}
	end := size - 1
	if last != "" {
		end, err = strconv.ParseInt(last, 10, 64)
		if err != nil || end < start {
			return 0, 0, false
		}
		if end >= size {
			end = size - 1
		}
	}
	return start, end - start + 1, true
}

// ---- uploads

func (reg *Registry) handleUpload(w http.ResponseWriter, r *http.Request, name, session string) error {
	switch {
	case session == "" && r.Method == http.MethodPost:
		q := r.URL.Query()
		if mount := q.Get("mount"); mount != "" {
			// Blobs are shared by digest across names: a mount is a
			// no-op when the blob exists, else the client uploads.
			if sha, ok := ParseDigest(mount); ok {
				_, err := reg.store.Stat(r.Context(), sha)
				switch {
				case err == nil:
					w.Header().Set("Location", "/v2/"+name+"/blobs/"+mount)
					w.Header().Set(headerDigest, mount)
					w.WriteHeader(http.StatusCreated)
					return nil
				case !errors.Is(err, blobstore.ErrNotFound):
					return err
				}
			}
		}
		if digest := q.Get("digest"); digest != "" { // single-POST monolithic upload
			return reg.finishUpload(w, r, name, digest)
		}
		id := uuid.NewString()
		reg.mu.Lock()
		for old, s := range reg.sessions {
			if time.Since(s.created) > sessionTTL {
				delete(reg.sessions, old)
			}
		}
		reg.sessions[id] = uploadSession{name: name, created: time.Now()}
		reg.mu.Unlock()
		w.Header().Set("Location", "/v2/"+name+"/blobs/uploads/"+id)
		w.Header().Set("Range", "0-0")
		w.Header().Set("Docker-Upload-UUID", id)
		w.WriteHeader(http.StatusAccepted)
		return nil
	case session != "" && r.Method == http.MethodPut:
		reg.mu.Lock()
		owner, ok := reg.sessions[session]
		delete(reg.sessions, session)
		reg.mu.Unlock()
		if !ok || owner.name != name {
			return reg.fail(w, http.StatusNotFound, "BLOB_UPLOAD_UNKNOWN", "upload session %q unknown", session)
		}
		return reg.finishUpload(w, r, name, r.URL.Query().Get("digest"))
	case session != "" && r.Method == http.MethodPatch:
		return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "chunked uploads are not supported; PUT the whole blob with ?digest=")
	}
	return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "%s not supported on uploads", r.Method)
}

func (reg *Registry) finishUpload(w http.ResponseWriter, r *http.Request, name, digest string) error {
	sha, ok := ParseDigest(digest)
	if !ok {
		return reg.fail(w, http.StatusBadRequest, "DIGEST_INVALID", "digest %q is not sha256:<hex>", digest)
	}
	if _, n, err := reg.store.Put(r.Context(), sha, r.Body); err != nil {
		if strings.Contains(err.Error(), "sha256 mismatch") {
			return reg.fail(w, http.StatusBadRequest, "DIGEST_INVALID", "%v", err)
		}
		return err
	} else {
		klog.FromContext(r.Context()).Info("stored blob", "name", name, "sha", sha, "bytes", n)
	}
	w.Header().Set("Location", "/v2/"+name+"/blobs/"+digest)
	w.Header().Set(headerDigest, digest)
	w.WriteHeader(http.StatusCreated)
	return nil
}

// ---- manifests

func tagKey(name, tag string) string         { return "tags/" + name + "/" + tag }
func manifestKey(name, digest string) string { return "manifests/" + name + "/" + digest }

// manifestRecord is what a manifest ref stores: enough to serve GET
// without parsing the body.
type manifestRecord struct {
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
}

func (reg *Registry) handleManifest(w http.ResponseWriter, r *http.Request, name, ref string) error {
	switch r.Method {
	case http.MethodPut:
		return reg.putManifest(w, r, name, ref)
	case http.MethodGet, http.MethodHead:
		return reg.getManifest(w, r, name, ref)
	case http.MethodDelete:
		return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "manifest deletion is not supported")
	}
	return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "%s not supported on manifests", r.Method)
}

// Resolve returns the digest a tag or digest reference points to.
func (reg *Registry) Resolve(ctx context.Context, name, ref string) (string, error) {
	if _, ok := ParseDigest(ref); ok {
		if _, err := reg.store.GetRef(ctx, manifestKey(name, ref)); err != nil {
			return "", err
		}
		return ref, nil
	}
	if !tagRe.MatchString(ref) {
		return "", blobstore.ErrNotFound
	}
	b, err := reg.store.GetRef(ctx, tagKey(name, ref))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

// GetManifest returns a manifest's body and media type by tag or digest.
func (reg *Registry) GetManifest(ctx context.Context, name, ref string) ([]byte, string, string, error) {
	digest, err := reg.Resolve(ctx, name, ref)
	if err != nil {
		return nil, "", "", err
	}
	recb, err := reg.store.GetRef(ctx, manifestKey(name, digest))
	if err != nil {
		return nil, "", "", err
	}
	var rec manifestRecord
	if err := json.Unmarshal(recb, &rec); err != nil {
		return nil, "", "", fmt.Errorf("manifest record %s: %w", digest, err)
	}
	sha, _ := ParseDigest(digest)
	rc, err := reg.store.Open(ctx, sha, 0, -1)
	if err != nil {
		return nil, "", "", err
	}
	body, err := io.ReadAll(io.LimitReader(rc, maxManifestBytes))
	if cerr := rc.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, "", "", err
	}
	return body, rec.MediaType, digest, nil
}

func (reg *Registry) getManifest(w http.ResponseWriter, r *http.Request, name, ref string) error {
	body, mediaType, digest, err := reg.GetManifest(r.Context(), name, ref)
	if errors.Is(err, blobstore.ErrNotFound) {
		return reg.fail(w, http.StatusNotFound, "MANIFEST_UNKNOWN", "manifest %s:%s unknown", name, ref)
	}
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", mediaType)
	w.Header().Set(headerDigest, digest)
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return nil
	}
	_, err = w.Write(body)
	return err
}

// PutManifest stores a manifest body under name, verifying that every
// blob or manifest it references exists, and tags it if ref is a tag.
// Returns the manifest's digest.
func (reg *Registry) PutManifest(ctx context.Context, name, ref, mediaType string, body []byte) (string, error) {
	if len(body) > maxManifestBytes {
		return "", &clientError{"MANIFEST_INVALID", "manifest larger than 4 MB"}
	}
	var probe struct {
		MediaType string       `json:"mediaType"`
		Config    *Descriptor  `json:"config"`
		Layers    []Descriptor `json:"layers"`
		Manifests []Descriptor `json:"manifests"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return "", &clientError{"MANIFEST_INVALID", "manifest is not JSON: " + err.Error()}
	}
	if probe.MediaType != "" {
		mediaType = probe.MediaType
	}
	if mediaType == "" {
		mediaType = MediaTypeManifest
	}
	// Everything referenced must already be here.
	var refs []Descriptor
	if probe.Config != nil {
		refs = append(refs, *probe.Config)
	}
	refs = append(refs, probe.Layers...)
	for _, d := range refs {
		sha, ok := ParseDigest(d.Digest)
		if !ok {
			return "", &clientError{"MANIFEST_BLOB_UNKNOWN", "descriptor digest " + d.Digest + " is not sha256:<hex>"}
		}
		if sha == emptyDescriptorSHA {
			if err := reg.ensureEmptyBlob(ctx); err != nil {
				return "", err
			}
			continue
		}
		if _, err := reg.store.Stat(ctx, sha); errors.Is(err, blobstore.ErrNotFound) {
			return "", &clientError{"MANIFEST_BLOB_UNKNOWN", "blob " + d.Digest + " is not in the registry"}
		} else if err != nil {
			return "", err
		}
	}
	for _, d := range probe.Manifests {
		if _, err := reg.store.GetRef(ctx, manifestKey(name, d.Digest)); errors.Is(err, blobstore.ErrNotFound) {
			return "", &clientError{"MANIFEST_BLOB_UNKNOWN", "manifest " + d.Digest + " is not in " + name}
		} else if err != nil {
			return "", err
		}
	}
	sum := sha256.Sum256(body)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if d, ok := ParseDigest(ref); ok && "sha256:"+d != digest {
		return "", &clientError{"DIGEST_INVALID", "body digest " + digest + " does not match reference " + ref}
	} else if !ok && !ValidTag(ref) {
		// before anything is written: a refused manifest must leave nothing behind
		return "", &clientError{"TAG_INVALID", "invalid tag " + ref}
	}
	if _, _, err := reg.store.Put(ctx, hex.EncodeToString(sum[:]), strings.NewReader(string(body))); err != nil {
		return "", err
	}
	rec, err := json.Marshal(manifestRecord{MediaType: mediaType, Size: int64(len(body))})
	if err != nil {
		return "", err
	}
	if err := reg.store.PutRef(ctx, manifestKey(name, digest), rec); err != nil {
		return "", err
	}
	if _, isDigest := ParseDigest(ref); !isDigest {
		if err := reg.store.PutRef(ctx, tagKey(name, ref), []byte(digest)); err != nil {
			return "", err
		}
	}
	return digest, nil
}

// ensureEmptyBlob stores the two-byte "{}" config so artifacts can
// reference it without pushing it first.
func (reg *Registry) ensureEmptyBlob(ctx context.Context) error {
	_, err := reg.store.Stat(ctx, emptyDescriptorSHA)
	if err == nil {
		return nil
	}
	if !errors.Is(err, blobstore.ErrNotFound) {
		return err
	}
	_, _, err = reg.store.Put(ctx, emptyDescriptorSHA, strings.NewReader("{}"))
	return err
}

func (reg *Registry) putManifest(w http.ResponseWriter, r *http.Request, name, ref string) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxManifestBytes+1))
	if err != nil {
		return err
	}
	digest, err := reg.PutManifest(r.Context(), name, ref, r.Header.Get("Content-Type"), body)
	var ce *clientError
	if errors.As(err, &ce) {
		return reg.fail(w, http.StatusBadRequest, ce.code, "%s", ce.msg)
	}
	if err != nil {
		return err
	}
	w.Header().Set("Location", "/v2/"+name+"/manifests/"+digest)
	w.Header().Set(headerDigest, digest)
	w.WriteHeader(http.StatusCreated)
	return nil
}

type clientError struct{ code, msg string }

func (e *clientError) Error() string { return e.code + ": " + e.msg }

// ---- tags

// Tags lists a repository's tags.
func (reg *Registry) Tags(ctx context.Context, name string) ([]string, error) {
	keys, err := reg.store.ListRefs(ctx, "tags/"+name+"/")
	if err != nil {
		return nil, err
	}
	tags := make([]string, 0, len(keys))
	for _, k := range keys {
		tag := strings.TrimPrefix(k, "tags/"+name+"/")
		if !strings.Contains(tag, "/") { // not a tag of a longer name
			tags = append(tags, tag)
		}
	}
	return tags, nil
}

func (reg *Registry) handleTags(w http.ResponseWriter, r *http.Request, name string) error {
	if r.Method != http.MethodGet {
		return reg.fail(w, http.StatusMethodNotAllowed, "UNSUPPORTED", "%s not supported on tags", r.Method)
	}
	tags, err := reg.Tags(r.Context(), name)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(map[string]any{"name": name, "tags": tags})
}

// Repositories lists the names that have at least one tag.
func (reg *Registry) Repositories(ctx context.Context) ([]string, error) {
	keys, err := reg.store.ListRefs(ctx, "tags/")
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var names []string
	for _, k := range keys {
		rest := strings.TrimPrefix(k, "tags/")
		i := strings.LastIndex(rest, "/")
		if i <= 0 {
			continue
		}
		if name := rest[:i]; !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	return names, nil
}
