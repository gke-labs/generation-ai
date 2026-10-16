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

package blobstore

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"sort"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/google/uuid"
	"google.golang.org/api/iterator"
	"k8s.io/klog/v2"
)

// GCS keeps blobs as objects <prefix>/_blobs/<sha> in a bucket.
//
// Put streams into a temporary object under <prefix>/_tmp/, verifying
// the sha256 as the bytes pass, then copies it server-side to its
// final name and deletes the temporary: a reader listing or opening
// _blobs never sees a partial object, and a mismatch publishes nothing.
type GCS struct {
	client *storage.Client
	bucket string
	prefix string
	cache  *URLSignerCache
}

type gcsSigner struct {
	client         *storage.Client
	serviceAccount string
}

func (s *gcsSigner) SignURL(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
	opts := &storage.SignedURLOptions{
		GoogleAccessID: s.serviceAccount,
		Method:         http.MethodGet,
		Expires:        expires,
		Scheme:         storage.SigningSchemeV4,
	}
	return s.client.Bucket(bucket).SignedURL(object, opts)
}

// NewGCS opens gs://bucket[/prefix] with application default credentials
// (workload identity on GKE).
func NewGCS(ctx context.Context, location string, opts ...Option) (*GCS, error) {
	var options Options
	for _, opt := range opts {
		opt(&options)
	}

	rest := strings.TrimPrefix(location, "gs://")
	bucket, prefix, _ := strings.Cut(rest, "/")
	if bucket == "" {
		return nil, fmt.Errorf("invalid GCS location %q: want gs://bucket[/prefix]", location)
	}
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating GCS client: %w", err)
	}
	g := &GCS{client: client, bucket: bucket, prefix: strings.Trim(prefix, "/")}

	var signer URLSigner = options.URLSigner
	if signer == nil && options.SignerServiceAccount != "" {
		signer = &gcsSigner{
			client:         client,
			serviceAccount: options.SignerServiceAccount,
		}
	}
	if signer != nil {
		g.cache = NewURLSignerCache(bucket, signer, options.Validity, options.Now)
	}

	// Fail at startup, not on the first Put, if the bucket is not ours
	// to use (wrong name, missing grant). Listing one object needs only
	// the object roles the README grants; bucket metadata would need
	// more.
	it := client.Bucket(bucket).Objects(ctx, &storage.Query{Prefix: g.object("")})
	it.PageInfo().MaxSize = 1
	if _, err := it.Next(); err != nil && !errors.Is(err, iterator.Done) {
		return nil, fmt.Errorf("bucket gs://%s is not reachable with these credentials: %w", bucket, err)
	}
	return g, nil
}

func (g *GCS) object(sha string) string { return path.Join(g.prefix, "_blobs", sha) }

func (g *GCS) Stat(ctx context.Context, sha string) (int64, error) {
	attrs, err := g.client.Bucket(g.bucket).Object(g.object(sha)).Attrs(ctx)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, err
	}
	return attrs.Size, nil
}

func (g *GCS) Put(ctx context.Context, sha string, r io.Reader) (string, int64, error) {
	bkt := g.client.Bucket(g.bucket)
	tmpName := path.Join(g.prefix, "_tmp", uuid.NewString())
	tmp := bkt.Object(tmpName)
	w := tmp.NewWriter(ctx)
	w.ChunkSize = 64 << 20 // fewer resumable-upload round trips for multi-GB files
	// discard removes the temporary object after a failure; the
	// original error is what the caller sees.
	discard := func() {
		if err := tmp.Delete(context.WithoutCancel(ctx)); err != nil && !errors.Is(err, storage.ErrObjectNotExist) {
			klog.FromContext(ctx).Error(err, "removing temporary object", "object", tmpName)
		}
	}
	dw := newDigestWriter(w)
	if err := dw.copyFrom(r); err != nil {
		if cerr := w.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing failed upload", "object", tmpName, "error", cerr.Error())
		}
		discard()
		return "", dw.n, fmt.Errorf("uploading blob: %w", err)
	}
	if err := w.Close(); err != nil {
		discard()
		return "", dw.n, fmt.Errorf("finishing upload: %w", err)
	}
	got, err := dw.check(sha)
	if err != nil {
		discard()
		return got, dw.n, err
	}
	// Publish: server-side copy to the final name (no bytes move through
	// here), then drop the temporary.
	copier := bkt.Object(g.object(got)).CopierFrom(tmp)
	copier.Metadata = map[string]string{"sha256": got}
	if _, err := copier.Run(ctx); err != nil {
		discard()
		return got, dw.n, fmt.Errorf("publishing blob: %w", err)
	}
	// The blob is live; a failure to drop the temporary is not a failed
	// Put. The bucket's lifecycle rule (README) collects strays.
	if err := tmp.Delete(context.WithoutCancel(ctx)); err != nil && !errors.Is(err, storage.ErrObjectNotExist) {
		klog.FromContext(ctx).Error(err, "removing temporary object after publish", "object", tmpName)
	}
	return got, dw.n, nil
}

func (g *GCS) Open(ctx context.Context, sha string, offset, length int64) (io.ReadCloser, error) {
	rc, err := g.client.Bucket(g.bucket).Object(g.object(sha)).NewRangeReader(ctx, offset, length)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return nil, ErrNotFound
	}
	return rc, err
}

func (g *GCS) ref(key string) string { return path.Join(g.prefix, "_refs", key) }

func (g *GCS) PutRef(ctx context.Context, key string, value []byte) error {
	if !ValidRefKey(key) {
		return fmt.Errorf("invalid ref key %q", key)
	}
	w := g.client.Bucket(g.bucket).Object(g.ref(key)).NewWriter(ctx)
	w.ContentType = "application/octet-stream"
	if _, err := w.Write(value); err != nil {
		if cerr := w.Close(); cerr != nil {
			klog.FromContext(ctx).V(2).Info("closing failed ref write", "key", key, "error", cerr.Error())
		}
		return err
	}
	return w.Close()
}

func (g *GCS) GetRef(ctx context.Context, key string) ([]byte, error) {
	if !ValidRefKey(key) {
		return nil, ErrNotFound
	}
	rc, err := g.client.Bucket(g.bucket).Object(g.ref(key)).NewReader(ctx)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	defer func() {
		if cerr := rc.Close(); cerr != nil {
			klog.FromContext(ctx).Error(cerr, "closing ref", "key", key)
		}
	}()
	return io.ReadAll(rc)
}

func (g *GCS) ListRefs(ctx context.Context, prefix string) ([]string, error) {
	base := g.ref("") + "/"
	it := g.client.Bucket(g.bucket).Objects(ctx, &storage.Query{Prefix: base + prefix})
	var keys []string
	for {
		attrs, err := it.Next()
		if errors.Is(err, iterator.Done) {
			break
		}
		if err != nil {
			return nil, err
		}
		keys = append(keys, strings.TrimPrefix(attrs.Name, base))
	}
	sort.Strings(keys)
	return keys, nil
}

func (g *GCS) DirectURL(ctx context.Context, sha string) (string, error) {
	if g.cache == nil {
		return "", nil
	}
	return g.cache.Get(ctx, sha, g.object(sha))
}

func (g *GCS) String() string { return "gs://" + g.bucket + "/" + g.prefix }
