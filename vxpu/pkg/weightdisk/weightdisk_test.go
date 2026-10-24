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

package weightdisk

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// fakeGcloud records the commands and answers the factory's guest
// attributes: copying for the first poll, then done.
type fakeGcloud struct {
	calls []string
	polls int
	fail  map[string]error // command prefixes that fail
}

func (f *fakeGcloud) run(_ context.Context, args ...string) ([]byte, error) {
	cmd := strings.Join(args, " ")
	f.calls = append(f.calls, cmd)
	for prefix, err := range f.fail {
		if strings.HasPrefix(cmd, prefix) {
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("%s failed", prefix)
		}
	}
	if strings.HasPrefix(cmd, "compute instances get-guest-attributes") {
		f.polls++
		if f.polls == 1 {
			return []byte(`[{"namespace":"vxpu","key":"status","value":"copying"}]`), nil
		}
		return []byte(`[{"namespace":"vxpu","key":"status","value":"done"},{"namespace":"vxpu","key":"copy_seconds","value":"412"}]`), nil
	}
	return nil, nil
}

func (f *fakeGcloud) find(prefix string) string {
	for _, c := range f.calls {
		if strings.HasPrefix(c, prefix) {
			return c
		}
	}
	return ""
}

func TestBuild(t *testing.T) {
	g := &fakeGcloud{}
	res, err := Build(t.Context(), g.run, BuildOptions{
		Project: "p", Zone: "us-central1-a", Image: "ds-flash", Bucket: "gs://weights",
		Files: map[string]int64{"aa": 300e9, "bb": 200e9}, Poll: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Files != 2 || res.Bytes != 500e9 || res.CopySeconds != 412 {
		t.Errorf("result %+v", res)
	}
	create := g.find("compute instances create ds-flash-factory")
	for _, want := range []string{
		"--create-disk name=ds-flash-build,size=535,type=hyperdisk-balanced,device-name=vxpu-weights,auto-delete=no,provisioned-throughput=1200,provisioned-iops=4800",
		"--machine-type c3d-standard-30", "vxpu-bucket=gs://weights", "enable-guest-attributes=TRUE",
	} {
		if !strings.Contains(create, want) {
			t.Errorf("factory create %q lacks %q", create, want)
		}
	}
	// the VM goes before the image is made from its disk; the disk after
	order := []string{"compute instances create", "compute instances get-guest-attributes", "compute instances delete ds-flash-factory",
		"compute snapshots create ds-flash", "compute disks delete ds-flash-build"}
	at := 0
	for _, c := range g.calls {
		if at < len(order) && strings.HasPrefix(c, order[at]) {
			at++
		}
	}
	if at != len(order) {
		t.Errorf("commands out of order:\n%s", strings.Join(g.calls, "\n"))
	}
	if g.polls != 2 {
		t.Errorf("polled %d times, want 2", g.polls)
	}
}

func TestBuildDeletesTheFactoryWhenTheCopyFails(t *testing.T) {
	g := &fakeGcloud{}
	failed := func(ctx context.Context, args ...string) ([]byte, error) {
		if strings.HasPrefix(strings.Join(args, " "), "compute instances get-guest-attributes") {
			return []byte(`[{"key":"status","value":"failed: gcloud storage cp: 403"}]`), nil
		}
		return g.run(ctx, args...)
	}
	_, err := Build(t.Context(), failed, BuildOptions{Project: "p", Zone: "z", Image: "x", Bucket: "gs://b",
		Files: map[string]int64{"aa": 1}, Poll: time.Millisecond})
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("err %v, want the factory's failure", err)
	}
	if g.find("compute instances delete x-factory") == "" || g.find("compute snapshots create") != "" {
		t.Errorf("want the factory deleted and no image:\n%s", strings.Join(g.calls, "\n"))
	}
	if g.find("compute disks delete x-build") == "" {
		t.Errorf("want the factory's disk deleted too (it bills by the hour):\n%s", strings.Join(g.calls, "\n"))
	}
}
