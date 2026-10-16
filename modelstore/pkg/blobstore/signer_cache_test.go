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
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestURLSignerCache_InitialSignAndCacheHit(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d&exp=%d", bucket, object, count, expires.Unix()), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)

	sha := "1111111111111111111111111111111111111111111111111111111111111111"
	obj := "_blobs/" + sha

	// Initial call signs and caches
	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}

	// 1 day later (6 days remaining >= 3.5 days): cache hit, no new sign call
	nowTime = nowTime.Add(24 * time.Hour)
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u1 != u2 {
		t.Fatalf("expected same URL, got %s vs %s", u1, u2)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}

	// 3 days from start (4 days remaining >= 3.5 days): still cache hit
	nowTime = nowTime.Add(48 * time.Hour)
	u3, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u1 != u3 {
		t.Fatalf("expected same URL, got %s vs %s", u1, u3)
	}
	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected 1 sign call, got %d", calls)
	}
}

func TestURLSignerCache_RefreshWhenLessThanHalfValidityRemains(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d&exp=%d", bucket, object, count, expires.Unix()), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "2222222222222222222222222222222222222222222222222222222222222222"
	obj := "_blobs/" + sha

	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if signCalls.Load() != 1 {
		t.Fatalf("expected 1 call, got %d", signCalls.Load())
	}

	// Advance time to 3.6 days later (3.4 days remaining < 3.5 days -> triggers refresh!)
	nowTime = nowTime.Add(3*24*time.Hour + 14*time.Hour + 24*time.Minute)
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected refresh to trigger sign call 2, got %d", signCalls.Load())
	}
	if u1 == u2 {
		t.Fatalf("expected refreshed URL to differ from initial URL")
	}

	// Immediately after refresh (remaining ~7 days >= 3.5 days): cache hit with new URL
	nowTime = nowTime.Add(time.Hour)
	u3, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u2 != u3 {
		t.Fatalf("expected same refreshed URL, got %s vs %s", u2, u3)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected still 2 calls, got %d", signCalls.Load())
	}
}

func TestURLSignerCache_ExpiredURL(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d", bucket, object, count), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "3333333333333333333333333333333333333333333333333333333333333333"
	obj := "_blobs/" + sha

	_, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}

	// Advance time past the 7 day expiration (e.g. 10 days later)
	nowTime = nowTime.Add(10 * 24 * time.Hour)
	_, err = cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}
	if signCalls.Load() != 2 {
		t.Fatalf("expected 2 calls, got %d", signCalls.Load())
	}
}

func TestURLSignerCache_ErrorFallbackToUnexpired(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	failSign := atomic.Bool{}

	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		signCalls.Add(1)
		if failSign.Load() {
			return "", errors.New("simulated IAM signing failure")
		}
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s", bucket, object), nil
	})

	nowTime := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return nowTime }

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nowFunc)
	sha := "4444444444444444444444444444444444444444444444444444444444444444"
	obj := "_blobs/" + sha

	u1, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatal(err)
	}

	// Advance time to 4 days later (remaining 3 days < 3.5 days, but not expired).
	// Enable failure.
	nowTime = nowTime.Add(4 * 24 * time.Hour)
	failSign.Store(true)

	// Cache should fallback to the still-valid cached URL rather than failing
	u2, err := cache.Get(ctx, sha, obj)
	if err != nil {
		t.Fatalf("expected fallback to unexpired cached URL, got error: %v", err)
	}
	if u1 != u2 {
		t.Fatalf("expected u1 fallback, got %s", u2)
	}

	// Advance time past 7 days (now fully expired)
	nowTime = nowTime.Add(4 * 24 * time.Hour)
	_, err = cache.Get(ctx, sha, obj)
	if err == nil {
		t.Fatalf("expected error when cached URL is expired and signer fails")
	}
}

func TestURLSignerCache_ConcurrentRequestsJoinOneSign(t *testing.T) {
	ctx := t.Context()
	var signCalls atomic.Int64
	fakeSigner := URLSignerFunc(func(ctx context.Context, bucket, object string, expires time.Time) (string, error) {
		time.Sleep(20 * time.Millisecond) // simulate IAM network latency
		count := signCalls.Add(1)
		return fmt.Sprintf("https://storage.googleapis.com/%s/%s?call=%d", bucket, object, count), nil
	})

	cache := NewURLSignerCache("test-bucket", fakeSigner, DefaultValidity, nil)
	sha := "5555555555555555555555555555555555555555555555555555555555555555"
	obj := "_blobs/" + sha

	const concurrency = 20
	var wg sync.WaitGroup
	urls := make([]string, concurrency)
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			u, err := cache.Get(ctx, sha, obj)
			if err != nil {
				t.Errorf("goroutine %d: %v", idx, err)
				return
			}
			urls[idx] = u
		}(i)
	}
	wg.Wait()

	if calls := signCalls.Load(); calls != 1 {
		t.Fatalf("expected exactly 1 sign call across %d concurrent requests, got %d", concurrency, calls)
	}
	for i, u := range urls {
		if u != urls[0] {
			t.Errorf("url[%d] (%s) != url[0] (%s)", i, u, urls[0])
		}
	}
}
