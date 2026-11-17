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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

// Statistics about models served by name (#357): the engine's record of
// how it behaves in the real world, for us and, later, for the planner
// (#316, #349).
//
// Every Chat and Generate on a session opened by model name is recorded:
// the request's statistics (tokens, prefill, ms/token), which commit and
// artifact served it and on which executor, and, for a sampled share of
// requests (every Nth; --telemetry-every), the executor's execution
// telemetry (segment times, host-op phases, expert selections, the plan;
// #337). The router turns telemetry on for sampled requests and keeps the
// record to itself unless the client asked for it. Records are batched
// into JSON-lines blobs in modelstore, one tag per UTC day on the
// repository vxpu-stats/<model> (a day with many batches continues in
// dYYYYMMDD.1, .2, ...), so they outlive the router; GetModelStats
// summarises them, reading one batch at a time.
//
// Records are buffered per model (by its lowercased name, the
// repository's) until statsBatch of them or the next statsFlushEvery
// tick. A write has statsWriteTimeout; after a failed one the model waits
// (doubling, up to statsRetryMax) before the next attempt, instead of
// re-uploading its whole buffer on every new record. The router flushes
// every buffer when it shuts down (FlushStats); a process that dies
// without shutting down loses what was buffered.

const (
	requestFormat     = "vxpu-request/1"
	modelStatsFormat  = "vxpu-model-stats/1"
	statsArtifactType = "application/vnd.gke-labs.vxpu.stats.v1+json"
	statsBatch        = 50
	statsFlushEvery   = 30 * time.Second
	statsWriteTimeout = time.Minute
	statsRetryMax     = 10 * time.Minute
	// a day's manifest gains a layer per batch; the registry refuses
	// manifests over 4 MB, so a busy day continues on a new tag
	statsMaxLayersPerTag = 2000
)

// statsRecordsMaxBytes bounds the records GetModelStats returns (gRPC
// caps a message at 128 MB): the newest that fit, the oldest omitted.
var statsRecordsMaxBytes = 64 << 20

// modelRef names what serves a model key loaded by name.
type modelRef struct {
	Model, Commit, Artifact string
}

// requestRecord is one request's entry in the statistics.
type requestRecord struct {
	Format          string          `json:"format"`
	Time            string          `json:"time"`
	Model           string          `json:"model"`
	Commit          string          `json:"commit"`
	Artifact        string          `json:"artifact"`
	Executor        string          `json:"executor"`
	RPC             string          `json:"rpc"`
	PromptTokens    int32           `json:"prompt_tokens"`
	PrefilledTokens int32           `json:"prefilled_tokens"`
	Generated       int32           `json:"generated"`
	PrefillMs       float32         `json:"prefill_ms"`
	MsPerToken      float32         `json:"ms_per_token"`
	FinishReason    string          `json:"finish_reason,omitempty"`
	Telemetry       json.RawMessage `json:"telemetry,omitempty"`
}

// statsKey is how statistics name a model: Hub ids are case-insensitive,
// and the repository is lowercase, so records and flushes for "Org/M"
// and "org/m" are one model's.
func statsKey(model string) string { return strings.ToLower(model) }

func statsRepository(model string) string { return "vxpu-stats/" + statsKey(model) }

// statsStore batches records into modelstore.
type statsStore struct {
	ws             *weightstore.Client
	telemetryEvery int64
	requests       atomic.Int64
	records        chan requestRecord
	flushes        chan flushRequest
}

type flushRequest struct {
	// statsKey of the model to write now, or "" for every model
	key  string
	done chan error
}

func newStatsStore(ws *weightstore.Client, telemetryEvery int) *statsStore {
	st := &statsStore{ws: ws, telemetryEvery: int64(telemetryEvery),
		records: make(chan requestRecord, 1000), flushes: make(chan flushRequest)}
	go st.loop(context.Background())
	return st
}

// sampleTelemetry says whether this request records execution telemetry.
func (st *statsStore) sampleTelemetry() bool {
	if st.telemetryEvery <= 0 {
		return false
	}
	return (st.requests.Add(1)-1)%st.telemetryEvery == 0
}

func (st *statsStore) record(ctx context.Context, r requestRecord) {
	select {
	case st.records <- r:
	default:
		klog.FromContext(ctx).Info("Statistics buffer full; dropping a record", "model", r.Model)
	}
}

// flush writes model's buffered records now (GetModelStats does, so a
// summary includes the latest requests), or every model's for "".
func (st *statsStore) flush(ctx context.Context, model string) error {
	key := ""
	if model != "" {
		key = statsKey(model)
	}
	req := flushRequest{key: key, done: make(chan error, 1)}
	select {
	case st.flushes <- req:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-req.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (st *statsStore) loop(ctx context.Context) {
	log := klog.FromContext(ctx)
	buffered := map[string][]requestRecord{} // by statsKey
	retryAt, backoff := map[string]time.Time{}, map[string]time.Duration{}
	write := func(key string) error {
		recs := buffered[key]
		if len(recs) == 0 {
			return nil
		}
		wctx, cancel := context.WithTimeout(ctx, statsWriteTimeout)
		defer cancel()
		if err := st.write(wctx, key, recs); err != nil {
			if len(recs) > 10*statsBatch { // modelstore unreachable for long: keep the newest
				buffered[key] = recs[len(recs)-10*statsBatch:]
			}
			b := min(max(2*backoff[key], statsFlushEvery), statsRetryMax)
			backoff[key], retryAt[key] = b, time.Now().Add(b)
			return err
		}
		delete(buffered, key)
		delete(retryAt, key)
		delete(backoff, key)
		return nil
	}
	due := func(key string) bool { return !time.Now().Before(retryAt[key]) }
	tick := time.NewTicker(statsFlushEvery)
	defer tick.Stop()
	for {
		select {
		case r := <-st.records:
			key := statsKey(r.Model)
			buffered[key] = append(buffered[key], r)
			if len(buffered[key]) >= statsBatch && due(key) {
				if err := write(key); err != nil {
					log.Error(err, "writing statistics", "model", key)
				}
			}
		case req := <-st.flushes:
			for drained := false; !drained; { // records sent before the flush was asked for
				select {
				case r := <-st.records:
					key := statsKey(r.Model)
					buffered[key] = append(buffered[key], r)
				default:
					drained = true
				}
			}
			// asked for: written now, whatever the backoff
			if req.key != "" {
				req.done <- write(req.key)
				continue
			}
			var first error
			for key := range buffered {
				if err := write(key); err != nil && first == nil {
					first = err
				}
			}
			req.done <- first
		case <-tick.C:
			for key := range buffered {
				if !due(key) {
					continue
				}
				if err := write(key); err != nil {
					log.Error(err, "writing statistics", "model", key)
				}
			}
		}
	}
}

// write appends one batch to today's manifest of the model's statistics.
func (st *statsStore) write(ctx context.Context, model string, recs []requestRecord) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range recs {
		// One bad record (telemetry that is not JSON, say) is dropped; kept,
		// it would fail every write of the batch it is in.
		if err := enc.Encode(r); err != nil {
			klog.FromContext(ctx).Error(err, "Dropping a statistics record that does not encode", "model", model)
		}
	}
	if buf.Len() == 0 {
		return nil
	}
	name := statsRepository(model)
	digest, err := st.ws.PutBlob(ctx, name, buf.Bytes())
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	day := "d" + now.Format("20060102")
	var m *weightstore.Manifest
	tag := day
	for n := 1; ; n++ {
		m, err = st.ws.GetManifest(ctx, name, tag)
		if err == weightstore.ErrNotFound {
			m = &weightstore.Manifest{SchemaVersion: 2, MediaType: "application/vnd.oci.image.manifest.v1+json",
				ArtifactType: statsArtifactType, Config: weightstore.EmptyConfig,
				Annotations: map[string]string{"vnd.gke-labs.vxpu.model": model}}
			break
		}
		if err != nil {
			return err
		}
		if len(m.Layers) < statsMaxLayersPerTag {
			break
		}
		tag = fmt.Sprintf("%s.%d", day, n) // this day's manifest is full: continue on the next
	}
	m.Layers = append(m.Layers, weightstore.Layer{MediaType: weightstore.MediaTypeFile, Digest: digest, Size: int64(buf.Len()),
		Annotations: map[string]string{"org.opencontainers.image.title": "batch-" + strconv.FormatInt(now.UnixNano(), 10) + ".jsonl"}})
	return st.ws.PutManifest(ctx, name, tag, m)
}

// UseStats records statistics for models served by name, with execution
// telemetry on every Nth request (0: none). Needs modelstore.
func (s *Server) UseStats(telemetryEvery int) {
	if s.weightStore != nil {
		s.stats = newStatsStore(s.weightStore, telemetryEvery)
	}
}

// FlushStats writes every model's buffered statistics now: the router
// calls it when shutting down, so the last requests are not lost.
func (s *Server) FlushStats(ctx context.Context) error {
	if s.stats == nil {
		return nil
	}
	return s.stats.flush(ctx, "")
}

// statsTarget returns what serves modelKey if its requests are recorded.
func (s *Server) statsTarget(modelKey string) (modelRef, string, bool) {
	if s.stats == nil {
		return modelRef{}, "", false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref, ok := s.keyModel[modelKey]
	if !ok {
		return modelRef{}, "", false
	}
	executor := s.modelEndpoint[modelKey]
	if executor == "" {
		executor = "pod"
	}
	return ref, executor, true
}

func newRecord(ref modelRef, executor, rpc string) requestRecord {
	return requestRecord{Format: requestFormat, Time: time.Now().UTC().Format(time.RFC3339Nano),
		Model: ref.Model, Commit: ref.Commit, Artifact: ref.Artifact, Executor: executor, RPC: rpc}
}

// GetModelStats summarises a model's recorded statistics.
func (s *Server) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
	if s.stats == nil {
		return nil, status.Error(codes.FailedPrecondition, "statistics need modelstore")
	}
	if req.Model == "" {
		return nil, status.Error(codes.InvalidArgument, "which model's statistics?")
	}
	if err := s.stats.flush(ctx, req.Model); err != nil {
		klog.FromContext(ctx).Error(err, "flushing statistics before the summary", "model", req.Model)
	}
	acc := &statsAccumulator{}
	var kept [][]byte // the newest records, within statsRecordsMaxBytes
	keptBytes, start, omitted := 0, 0, 0
	unreadable, err := s.eachRecord(ctx, req.Model, func(line []byte) {
		if !acc.add(line) || !req.Records {
			return
		}
		kept = append(kept, append([]byte(nil), line...))
		keptBytes += len(line) + 1
		for keptBytes > statsRecordsMaxBytes && len(kept)-start > 1 {
			keptBytes -= len(kept[start]) + 1
			kept[start] = nil
			start++
			omitted++
		}
		if start > len(kept)/2 {
			kept, start = append([][]byte(nil), kept[start:]...), 0
		}
	})
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "%v", err)
	}
	acc.skipped += unreadable
	if acc.skipped > 0 {
		klog.FromContext(ctx).Info("Skipped unreadable statistics lines", "model", req.Model, "skipped", acc.skipped)
	}
	summary, err := acc.summary(req.Model, omitted)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}
	resp := &pb.GetModelStatsResponse{SummaryJson: summary}
	if req.Records {
		resp.RecordsJsonl = string(bytes.Join(kept[start:], []byte("\n")))
	}
	return resp, nil
}

// eachRecord calls fn with every recorded line of model's statistics,
// oldest first. It returns how many batches could not be read to the end.
func (s *Server) eachRecord(ctx context.Context, model string, fn func(line []byte)) (int, error) {
	name := statsRepository(model)
	tags, err := s.weightStore.ListTags(ctx, name)
	if err != nil {
		return 0, err
	}
	sort.Strings(tags) // dYYYYMMDD, dYYYYMMDD.1, ...: oldest first
	unreadable := 0
	for _, tag := range tags {
		m, err := s.weightStore.GetManifest(ctx, name, tag)
		if err != nil {
			return 0, err
		}
		for _, layer := range m.Layers {
			data, err := s.weightStore.FetchBlob(ctx, name, layer.Digest)
			if err != nil {
				return 0, err
			}
			sc := bufio.NewScanner(bytes.NewReader(data))
			sc.Buffer(make([]byte, 1<<20), 64<<20)
			for sc.Scan() {
				if line := bytes.TrimSpace(sc.Bytes()); len(line) > 0 {
					fn(line)
				}
			}
			if err := sc.Err(); err != nil { // the rest of this batch cannot be read
				unreadable++
				klog.FromContext(ctx).Error(err, "reading a statistics batch", "model", model, "tag", tag, "blob", layer.Digest)
			}
		}
	}
	return unreadable, nil
}

// expertCounts is model's recorded decode selections: layer -> expert ->
// how many times it was selected.
func (s *Server) expertCounts(ctx context.Context, model string) (map[int]map[int]int64, error) {
	acc := &statsAccumulator{}
	if _, err := s.eachRecord(ctx, model, func(line []byte) { acc.add(line) }); err != nil {
		return nil, err
	}
	counts := map[int]map[int]int64{}
	for key, c := range acc.experts {
		layer, err := strconv.Atoi(key)
		if err != nil {
			continue // not a layer index; the record is from another format
		}
		counts[layer] = map[int]int64{}
		for expert, n := range c {
			counts[layer][int(expert)] = n
		}
	}
	return counts, nil
}

// telemetrySummary is the part of a vxpu-telemetry/1 record the summary reads.
type telemetrySummary struct {
	ByKind  map[string]float64                       `json:"by_kind_ms_per_step"`
	HostOps map[string]map[string]map[string]float64 `json:"host_ops"`
	Experts map[string]map[string]map[string]int64   `json:"experts"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[int(math.Min(float64(len(sorted)-1), math.Floor(p*float64(len(sorted)))))]
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// statsAccumulator builds a vxpu-model-stats/1 summary one record at a
// time, keeping only the aggregates (and three numbers per request for
// the percentiles), so a model's whole history is never held at once.
type statsAccumulator struct {
	requests, sampled, skipped  int
	msPerToken, prefill, prompt []float64
	byRPC                       map[string]int
	commits, executors          map[string]bool
	byKind, moe                 map[string]float64
	// decode: layer -> expert -> count
	experts     map[string]map[int64]int64
	generated   int64
	first, last string
}

// add counts one record (a JSON line); false, and counted as skipped, if
// it is not one (a truncated line, a foreign blob, a later format).
func (a *statsAccumulator) add(line []byte) bool {
	var r requestRecord
	if err := json.Unmarshal(line, &r); err != nil {
		a.skipped++
		return false
	}
	if a.byRPC == nil {
		a.byRPC, a.commits, a.executors = map[string]int{}, map[string]bool{}, map[string]bool{}
		a.byKind, a.moe, a.experts = map[string]float64{}, map[string]float64{}, map[string]map[int64]int64{}
	}
	a.requests++
	if a.first == "" || r.Time < a.first {
		a.first = r.Time
	}
	if r.Time > a.last {
		a.last = r.Time
	}
	a.byRPC[r.RPC]++
	a.commits[r.Commit] = true
	a.executors[r.Executor] = true
	a.generated += int64(r.Generated)
	if r.Generated > 1 && r.MsPerToken > 0 {
		a.msPerToken = append(a.msPerToken, float64(r.MsPerToken))
	}
	a.prefill = append(a.prefill, float64(r.PrefillMs))
	a.prompt = append(a.prompt, float64(r.PromptTokens))
	if len(r.Telemetry) == 0 {
		return true
	}
	var t telemetrySummary
	if err := json.Unmarshal(r.Telemetry, &t); err != nil {
		return true
	}
	a.sampled++
	for k, v := range t.ByKind {
		a.byKind[k] += v
	}
	for k, v := range t.HostOps["decode"]["moe_experts"] {
		a.moe[k] += v
	}
	for layer, counts := range t.Experts["decode"] {
		if a.experts[layer] == nil {
			a.experts[layer] = map[int64]int64{}
		}
		for e, n := range counts {
			id, err := strconv.ParseInt(e, 10, 64)
			if err != nil {
				continue // not an expert index; the record is from another format
			}
			a.experts[layer][id] += n
		}
	}
	return true
}

// summary is the vxpu-model-stats/1 summary of what was added; omitted
// counts records left out of a records listing for its size.
func (a *statsAccumulator) summary(model string, omitted int) (string, error) {
	sort.Float64s(a.msPerToken)
	sort.Float64s(a.prefill)
	sort.Float64s(a.prompt)
	mean := func(xs []float64) float64 {
		if len(xs) == 0 {
			return 0
		}
		sum := 0.0
		for _, x := range xs {
			sum += x
		}
		return sum / float64(len(xs))
	}
	keys := func(m map[string]bool) []string {
		out := make([]string, 0, len(m))
		for k := range m {
			out = append(out, k)
		}
		sort.Strings(out)
		return out
	}
	out := map[string]any{
		"format": modelStatsFormat, "model": model, "requests": a.requests, "by_rpc": a.byRPC,
		"commits": keys(a.commits), "executors": keys(a.executors), "first": a.first, "last": a.last,
		"generated_tokens":  a.generated,
		"ms_per_token":      map[string]float64{"mean": round3(mean(a.msPerToken)), "p50": round3(percentile(a.msPerToken, 0.5)), "p90": round3(percentile(a.msPerToken, 0.9))},
		"prefill_ms":        map[string]float64{"p50": round3(percentile(a.prefill, 0.5)), "p90": round3(percentile(a.prefill, 0.9))},
		"prompt_tokens":     map[string]float64{"p50": percentile(a.prompt, 0.5), "max": percentile(a.prompt, 1)},
		"telemetry_sampled": a.sampled,
	}
	if a.skipped > 0 {
		out["skipped_lines"] = a.skipped
	}
	if omitted > 0 {
		out["records_omitted"] = omitted // the oldest, beyond the listing's size limit
	}
	if a.sampled > 0 {
		for k := range a.byKind {
			a.byKind[k] = round3(a.byKind[k] / float64(a.sampled))
		}
		for k := range a.moe {
			a.moe[k] = round3(a.moe[k] / float64(a.sampled))
		}
		out["by_kind_ms_per_step"] = a.byKind
		out["moe_experts_per_step"] = a.moe
		out["experts_decode"] = expertConcentration(a.experts)
	}
	b, err := json.MarshalIndent(out, "", " ")
	return string(b), err
}

// expertConcentration: how much of the decode traffic the hottest
// fraction of each layer's experts carried (an oracle cache's hit rate;
// vxpu.expert_stats has the full analysis).
func expertConcentration(experts map[string]map[int64]int64) map[string]any {
	var maxID, total int64
	for _, counts := range experts {
		for id, n := range counts {
			if id > maxID {
				maxID = id
			}
			total += n
		}
	}
	perLayer := maxID + 1
	out := map[string]any{"selections": total, "layers": len(experts), "experts_per_layer_seen": perLayer}
	for _, f := range []float64{0.25, 0.5} {
		var hit int64
		for _, counts := range experts {
			ns := make([]int64, 0, len(counts))
			for _, n := range counts {
				ns = append(ns, n)
			}
			sort.Slice(ns, func(i, j int) bool { return ns[i] > ns[j] })
			k := int(math.Ceil(f * float64(perLayer)))
			for i := 0; i < k && i < len(ns); i++ {
				hit += ns[i]
			}
		}
		if total > 0 {
			out[fmt.Sprintf("hottest_%d%%_share", int(f*100))] = round3(float64(hit) / float64(total))
		}
	}
	return out
}
