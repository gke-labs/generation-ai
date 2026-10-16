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
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
	"k8s.io/klog/v2"
)

// URLSigner generates a pre-signed URL for an object in a bucket.
type URLSigner interface {
	SignURL(ctx context.Context, bucket, object string, expires time.Time) (string, error)
}

// URLSignerFunc adapts a function to URLSigner.
type URLSignerFunc func(ctx context.Context, bucket, object string, expires time.Time) (string, error)

func (f URLSignerFunc) SignURL(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
	return f(ctx, bucket, object, expires)
}

type cachedURL struct {
	url       string
	expiresAt time.Time
	validity  time.Duration
}

// URLSignerCache caches pre-signed URLs in-process per blob and refreshes
// them when less than half their validity remains.
type URLSignerCache struct {
	bucket   string
	signer   URLSigner
	validity time.Duration
	now      func() time.Time

	mu     sync.RWMutex
	blobs  map[string]cachedURL
	flight singleflight.Group
}

// DefaultValidity is 7 days, the maximum validity for GCS V4 signed URLs.
const DefaultValidity = 7 * 24 * time.Hour

// NewURLSignerCache creates an in-process signed URL cache.
func NewURLSignerCache(bucket string, signer URLSigner, validity time.Duration, now func() time.Time) *URLSignerCache {
	if validity <= 0 {
		validity = DefaultValidity
	}
	if now == nil {
		now = time.Now
	}
	return &URLSignerCache{
		bucket:   bucket,
		signer:   signer,
		validity: validity,
		now:      now,
		blobs:    make(map[string]cachedURL),
	}
}

// Get returns a cached signed URL for object, or signs a new one if not
// cached or if less than half its validity remains.
func (c *URLSignerCache) Get(ctx context.Context, sha, object string) (string, error) {
	if c == nil || c.signer == nil {
		return "", nil
	}

	now := c.now()

	c.mu.RLock()
	entry, ok := c.blobs[sha]
	c.mu.RUnlock()

	if ok {
		remaining := entry.expiresAt.Sub(now)
		if remaining >= entry.validity/2 {
			return entry.url, nil
		}
	}

	res, err, _ := c.flight.Do(sha, func() (any, error) {
		now := c.now()
		c.mu.RLock()
		entry, ok := c.blobs[sha]
		c.mu.RUnlock()

		if ok {
			remaining := entry.expiresAt.Sub(now)
			if remaining >= entry.validity/2 {
				return entry.url, nil
			}
		}

		expires := now.Add(c.validity)
		url, err := c.signer.SignURL(ctx, c.bucket, object, expires)
		if err != nil {
			if ok && entry.expiresAt.After(now) {
				klog.FromContext(ctx).Error(err, "refreshing direct URL failed; using unexpired cached URL", "sha", sha)
				return entry.url, nil
			}
			return "", err
		}

		c.mu.Lock()
		c.blobs[sha] = cachedURL{
			url:       url,
			expiresAt: expires,
			validity:  c.validity,
		}
		c.mu.Unlock()

		return url, nil
	})

	if err != nil {
		return "", err
	}
	return res.(string), nil
}
