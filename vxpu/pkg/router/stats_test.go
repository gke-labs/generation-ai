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
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

func TestStats_RecordsNamedModelRequests(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(1)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, &loadingExecutor{})
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()
	s.UseExternalExecutors(ctx, []string{lis.Addr().String()})

	session, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	reply, err := s.Chat(ctx, &pb.ChatRequest{SessionId: session.SessionId, Text: "hi"})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if reply.Text != "yes" {
		t.Errorf("expected reply 'yes', got %q", reply.Text)
	}

	stats, err := s.GetModelStats(ctx, &pb.GetModelStatsRequest{Model: "org/M", Records: true})
	if err != nil {
		t.Fatalf("GetModelStats: %v", err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(stats.SummaryJson), &summary); err != nil {
		t.Fatalf("summary: %v", err)
	}
	if summary["requests"] != 1.0 || summary["telemetry_sampled"] != 1.0 || summary["generated_tokens"] != 10.0 {
		t.Errorf("unexpected summary: %s", stats.SummaryJson)
	}
	if byKind, _ := summary["by_kind_ms_per_step"].(map[string]any); byKind["dense"] != 20.0 {
		t.Errorf("segment times not summarised: %s", stats.SummaryJson)
	}
	experts, _ := summary["experts_decode"].(map[string]any)
	if experts["selections"] != 12.0 || experts["experts_per_layer_seen"] != 4.0 {
		t.Errorf("expert counts not summarised: %s", stats.SummaryJson)
	}
	if !strings.Contains(stats.RecordsJsonl, `"rpc":"chat"`) || !strings.Contains(stats.RecordsJsonl, `"vxpu-telemetry/1"`) ||
		!strings.Contains(stats.RecordsJsonl, `"commit":"`+testCommit+`"`) {
		t.Errorf("records lack the request, its telemetry or its commit: %s", stats.RecordsJsonl)
	}
}

// putBatch stores data as one statistics batch of model, as statsStore.write does.
func putBatch(t *testing.T, s *Server, model string, data []byte) {
	t.Helper()
	ctx, name := t.Context(), statsRepository(model)
	digest, err := s.weightStore.PutBlob(ctx, name, data)
	if err != nil {
		t.Fatal(err)
	}
	m, err := s.weightStore.GetManifest(ctx, name, "d20261019")
	if err == weightstore.ErrNotFound {
		m = &weightstore.Manifest{SchemaVersion: 2, MediaType: "application/vnd.oci.image.manifest.v1+json",
			ArtifactType: statsArtifactType, Config: weightstore.EmptyConfig}
	} else if err != nil {
		t.Fatal(err)
	}
	m.Layers = append(m.Layers, weightstore.Layer{MediaType: weightstore.MediaTypeFile, Digest: digest, Size: int64(len(data))})
	if err := s.weightStore.PutManifest(ctx, name, "d20261019", m); err != nil {
		t.Fatal(err)
	}
}

func recordLines(n int) []byte {
	var b bytes.Buffer
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, `{"format":"vxpu-request/1","time":"2026-10-19T00:00:%02dZ","model":"org/M","rpc":"chat","generated":4,"ms_per_token":10}`+"\n", i)
	}
	return b.Bytes()
}

func TestStats_SkipsUnreadableLines(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(0)
	putBatch(t, s, "org/M", append([]byte("not json\n{\"truncated\":\n"), recordLines(2)...))
	stats, err := s.GetModelStats(t.Context(), &pb.GetModelStatsRequest{Model: "org/M", Records: true})
	if err != nil {
		t.Fatalf("one bad batch line must not make the statistics unreadable: %v", err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(stats.SummaryJson), &summary); err != nil {
		t.Fatal(err)
	}
	if summary["requests"] != 2.0 || summary["skipped_lines"] != 2.0 {
		t.Errorf("want 2 requests and 2 skipped lines: %s", stats.SummaryJson)
	}
	if strings.Contains(stats.RecordsJsonl, "not json") {
		t.Errorf("the records should hold only records: %s", stats.RecordsJsonl)
	}
}

func TestStats_RecordsListingIsBounded(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(0)
	lines := recordLines(10)
	one := bytes.IndexByte(lines, '\n') + 1
	defer func(n int) { statsRecordsMaxBytes = n }(statsRecordsMaxBytes)
	statsRecordsMaxBytes = 3 * one // room for the newest three
	putBatch(t, s, "org/M", lines)
	stats, err := s.GetModelStats(t.Context(), &pb.GetModelStatsRequest{Model: "org/M", Records: true})
	if err != nil {
		t.Fatal(err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(stats.SummaryJson), &summary); err != nil {
		t.Fatal(err)
	}
	got := strings.Count(stats.RecordsJsonl, "\n") + 1
	if summary["requests"] != 10.0 || summary["records_omitted"] != 7.0 || got != 3 {
		t.Errorf("want all 10 summarised, the newest 3 listed and 7 omitted; got %d listed: %s", got, stats.SummaryJson)
	}
	if !strings.Contains(stats.RecordsJsonl, "00:00:09Z") || strings.Contains(stats.RecordsJsonl, "00:00:00Z") {
		t.Errorf("the listing should keep the newest records: %s", stats.RecordsJsonl)
	}
}

func TestStats_FlushIgnoresTheNamesCase(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(0)
	s.stats.record(t.Context(), requestRecord{Format: requestFormat, Model: "Org/M", RPC: "chat"})
	stats, err := s.GetModelStats(t.Context(), &pb.GetModelStatsRequest{Model: "org/m"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stats.SummaryJson, `"requests": 1`) {
		t.Errorf("a record buffered as Org/M should be flushed by a summary of org/m: %s", stats.SummaryJson)
	}
}

func TestStats_FlushStatsWritesEveryBuffer(t *testing.T) {
	store := &fakeModelstore{compiled: true}
	s, _ := newCatalogServer(t, store)
	s.UseStats(0)
	s.stats.record(t.Context(), requestRecord{Format: requestFormat, Model: "org/M", RPC: "chat"}) // below a batch
	if err := s.FlushStats(t.Context()); err != nil {
		t.Fatal(err)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	written := false
	for key := range store.manifests {
		written = written || strings.HasPrefix(key, "vxpu-stats/org/m:")
	}
	if !written {
		t.Errorf("FlushStats should have written the buffered record; manifests: %v", store.manifests)
	}
}

func TestStats_DropsARecordThatDoesNotEncode(t *testing.T) {
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	s.UseStats(0)
	s.stats.record(t.Context(), requestRecord{Format: requestFormat, Model: "org/m", RPC: "chat", Telemetry: json.RawMessage(`{"ms": NaN}`)})
	s.stats.record(t.Context(), requestRecord{Format: requestFormat, Model: "org/m", RPC: "chat"})
	stats, err := s.GetModelStats(t.Context(), &pb.GetModelStatsRequest{Model: "org/m"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stats.SummaryJson, `"requests": 1`) {
		t.Errorf("the good record should be written without the bad one: %s", stats.SummaryJson)
	}
}

func TestStats_BacksOffAfterAFailedWrite(t *testing.T) {
	var uploads atomic.Int64
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/blobs/uploads/") {
			uploads.Add(1)
		}
		http.Error(w, "modelstore is down", http.StatusServiceUnavailable)
	}))
	defer failing.Close()
	st := newStatsStore(weightstore.New(failing.URL), 0)
	for i := 0; i < statsBatch+20; i++ {
		st.record(t.Context(), requestRecord{Format: requestFormat, Model: "org/M", RPC: "chat"})
	}
	deadline := time.Now().Add(5 * time.Second)
	for uploads.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(100 * time.Millisecond) // the 20 records after the batch arrive while backing off
	if n := uploads.Load(); n != 1 {
		t.Errorf("after a failed write the buffer should wait for its backoff, not be uploaded per record: %d uploads", n)
	}
}
