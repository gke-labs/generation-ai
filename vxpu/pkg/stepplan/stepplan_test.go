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

package stepplan

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	planv1alpha1 "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	"google.golang.org/protobuf/encoding/protojson"
)

func edgeRef(node string) *planv1alpha1.ValueRef {
	return &planv1alpha1.ValueRef{
		Ref: &planv1alpha1.ValueRef_Edge{
			Edge: &planv1alpha1.EdgeRef{Node: node},
		},
	}
}

func residentRef(name string) *planv1alpha1.ValueRef {
	return &planv1alpha1.ValueRef{
		Ref: &planv1alpha1.ValueRef_Resident{
			Resident: name,
		},
	}
}

func inputRef(name string) *planv1alpha1.ValueRef {
	return &planv1alpha1.ValueRef{
		Ref: &planv1alpha1.ValueRef_Input{
			Input: name,
		},
	}
}

func hostCall(collection string, index int32) *planv1alpha1.HostCall {
	return &planv1alpha1.HostCall{
		Collection: collection,
		Index:      index,
	}
}

func TestWaitsAndPlanTable(t *testing.T) {
	tests := []struct {
		name        string
		desc        *StepGraphDescription
		opts        Options
		wantErr     bool
		wantWaits   map[string][]string
		wantStreams map[string]Stream
	}{
		{
			name: "chain where every node reads the previous one",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{inputRef("x")}},
					{Name: "n1", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("n0")}},
					{Name: "n2", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("n1")}},
				},
			},
			wantWaits: map[string][]string{
				"n0": {},
				"n1": {"n0"},
				"n2": {"n1"},
			},
			wantStreams: map[string]Stream{
				"n0": StreamMain,
				"n1": StreamMain,
				"n2": StreamMain,
			},
		},
		{
			name: "reader after a writer of one resident",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "w1", Kind: KindDense, Writes: []string{"kv_cache"}},
					{Name: "r1", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("kv_cache")}},
				},
			},
			wantWaits: map[string][]string{
				"w1": {},
				"r1": {"w1"},
			},
			wantStreams: map[string]Stream{
				"w1": StreamMain,
				"r1": StreamMain,
			},
		},
		{
			name: "writer after two readers",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "r1", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("kv_cache")}},
					{Name: "r2", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("kv_cache")}},
					{Name: "w1", Kind: KindDense, Writes: []string{"kv_cache"}},
				},
			},
			wantWaits: map[string][]string{
				"r1": {},
				"r2": {},
				"w1": {"r1", "r2"},
			},
			wantStreams: map[string]Stream{
				"r1": StreamMain,
				"r2": StreamParallel,
				"w1": StreamMain,
			},
		},
		{
			name: "two readers of one resident with no writer",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "r1", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("weights")}},
					{Name: "r2", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("weights")}},
				},
			},
			wantWaits: map[string][]string{
				"r1": {},
				"r2": {},
			},
			wantStreams: map[string]Stream{
				"r1": StreamMain,
				"r2": StreamParallel,
			},
		},
		{
			name: "the H, D1, D2, D3, D4 worked example from design note",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{
						Name:   "H",
						Kind:   "experts",
						Host:   hostCall("experts", 0),
						Writes: []string{"cache"},
					},
					{
						Name:  "D1",
						Kind:  KindDense,
						Reads: []*planv1alpha1.ValueRef{edgeRef("H")},
					},
					{
						Name:  "D2",
						Kind:  KindDense,
						Reads: []*planv1alpha1.ValueRef{edgeRef("D1"), residentRef("cache")},
					},
					{
						Name:  "D3",
						Kind:  KindDense,
						Reads: []*planv1alpha1.ValueRef{inputRef("step_input")},
					},
					{
						Name:   "D4",
						Kind:   KindDense,
						Writes: []string{"cache"},
					},
				},
			},
			wantWaits: map[string][]string{
				"H":  {},
				"D1": {"H"},
				"D2": {"D1", "H"},
				"D3": {},
				"D4": {"D2", "H"},
			},
			wantStreams: map[string]Stream{
				"H":  StreamMain,
				"D1": StreamMain,
				"D2": StreamMain,
				"D3": StreamParallel,
				"D4": StreamMain,
			},
		},
		{
			name: "Serialize option chains all nodes on main stream",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense},
					{Name: "n1", Kind: KindDense},
					{Name: "n2", Kind: KindDense},
				},
			},
			opts: Options{Serialize: true},
			wantWaits: map[string][]string{
				"n0": {},
				"n1": {"n0"},
				"n2": {"n1"},
			},
			wantStreams: map[string]Stream{
				"n0": StreamMain,
				"n1": StreamMain,
				"n2": StreamMain,
			},
		},
		{
			name: "NoParallel option keeps dataflow waits but uses only main stream",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense},
					{Name: "n1", Kind: KindDense},
				},
			},
			opts: Options{NoParallel: true},
			wantWaits: map[string][]string{
				"n0": {},
				"n1": {},
			},
			wantStreams: map[string]Stream{
				"n0": StreamMain,
				"n1": StreamMain,
			},
		},
		{
			name: "parallel dense island followed by node waiting on mainLast returns to main",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "dense0", Kind: KindDense},
					{Name: "island", Kind: KindDense},
					{Name: "dense1", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("dense0")}},
				},
			},
			wantWaits: map[string][]string{
				"dense0": {},
				"island": {},
				"dense1": {"dense0"},
			},
			wantStreams: map[string]Stream{
				"dense0": StreamMain,
				"island": StreamParallel,
				"dense1": StreamMain,
			},
		},
		{
			name:    "empty description refused",
			desc:    &StepGraphDescription{},
			wantErr: true,
		},
		{
			name:    "nil description refused",
			desc:    nil,
			wantErr: true,
		},
		{
			name: "empty node name refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "", Kind: KindDense},
				},
			},
			wantErr: true,
		},
		{
			name: "duplicate node name refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "dup", Kind: KindDense},
					{Name: "dup", Kind: KindDense},
				},
			},
			wantErr: true,
		},
		{
			name: "edge to non-existent node refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("ghost")}},
				},
			},
			wantErr: true,
		},
		{
			name: "edge to later node refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("n1")}},
					{Name: "n1", Kind: KindDense},
				},
			},
			wantErr: true,
		},
		{
			name: "edge from node to itself refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{edgeRef("n0")}},
				},
			},
			wantErr: true,
		},
		{
			name: "nil node entry refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{nil},
			},
			wantErr: true,
		},
		{
			name: "read with no variant set refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{{}}},
				},
			},
			wantErr: true,
		},
		{
			name: "resident with empty name in reads refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Reads: []*planv1alpha1.ValueRef{residentRef("")}},
				},
			},
			wantErr: true,
		},
		{
			name: "resident with empty name in writes refused",
			desc: &StepGraphDescription{
				Nodes: []*StepNode{
					{Name: "n0", Kind: KindDense, Writes: []string{""}},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.wantErr {
				if _, err := Waits(tc.desc); err == nil {
					t.Errorf("Waits expected error for %q, got nil", tc.name)
				}
				if _, err := Plan(tc.desc, tc.opts); err == nil {
					t.Errorf("Plan expected error for %q, got nil", tc.name)
				}
				return
			}

			if !tc.opts.Serialize {
				gotWaits, err := Waits(tc.desc)
				if err != nil {
					t.Fatalf("Waits failed: %v", err)
				}
				for nodeName, wantW := range tc.wantWaits {
					gotW := gotWaits[nodeName]
					if len(gotW) == 0 && len(wantW) == 0 {
						continue
					}
					if !reflect.DeepEqual(gotW, wantW) {
						t.Errorf("Waits node %q = %v, want %v", nodeName, gotW, wantW)
					}
				}
			}

			plan, err := Plan(tc.desc, tc.opts)
			if err != nil {
				t.Fatalf("Plan failed: %v", err)
			}
			if len(plan.Nodes) != len(tc.desc.Nodes) {
				t.Fatalf("Plan returned %d nodes, want %d", len(plan.Nodes), len(tc.desc.Nodes))
			}

			for _, np := range plan.Nodes {
				wantStream, ok := tc.wantStreams[np.Name]
				if !ok {
					t.Fatalf("unexpected node %q in plan", np.Name)
				}
				if np.Stream != wantStream {
					t.Errorf("node %q stream = %v, want %v", np.Name, np.Stream, wantStream)
				}

				wantW := tc.wantWaits[np.Name]
				if len(np.Waits) == 0 && len(wantW) == 0 {
					continue
				}
				if !reflect.DeepEqual(np.Waits, wantW) {
					t.Errorf("node %q waits = %v, want %v", np.Name, np.Waits, wantW)
				}
			}
		})
	}
}

func TestChainFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "stepgraph_chain.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("ReadFile fixture failed: %v", err)
	}

	desc, err := ParseDescription(data)
	if err != nil {
		t.Fatalf("ParseDescription failed: %v", err)
	}

	if len(desc.Nodes) != 11 {
		t.Fatalf("expected 11 nodes in fixture, got %d", len(desc.Nodes))
	}

	plan, err := Plan(desc, Options{})
	if err != nil {
		t.Fatalf("Plan failed: %v", err)
	}
	if len(plan.Nodes) != 11 {
		t.Fatalf("expected 11 nodes in plan, got %d", len(plan.Nodes))
	}

	wantWaits := map[string][]string{
		"submod_0":  {},
		"submod_1":  {"submod_0"},
		"submod_2":  {"submod_0", "submod_1"},
		"submod_3":  {"submod_2"},
		"submod_4":  {"submod_0", "submod_2", "submod_3"},
		"submod_5":  {"submod_4"},
		"submod_6":  {"submod_4", "submod_5"},
		"submod_7":  {"submod_6"},
		"submod_8":  {"submod_6", "submod_7"},
		"submod_9":  {"submod_8"},
		"submod_10": {"submod_8", "submod_9"},
	}

	for i, np := range plan.Nodes {
		if np.Stream != StreamMain {
			t.Errorf("node %d (%s) stream = %v, want StreamMain (no parallel stream on chain)", i, np.Name, np.Stream)
		}
		expectedWaits := wantWaits[np.Name]
		if len(np.Waits) == 0 && len(expectedWaits) == 0 {
			continue
		}
		if !reflect.DeepEqual(np.Waits, expectedWaits) {
			t.Errorf("node %s waits = %v, want %v", np.Name, np.Waits, expectedWaits)
		}
	}

	// Verify serialization round trip with stable JSON formatting
	jsonData, err := Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal plan failed: %v", err)
	}

	var parsedPlan planv1alpha1.StepPlan
	if err := protojson.Unmarshal(jsonData, &parsedPlan); err != nil {
		t.Fatalf("Unmarshal plan failed: %v", err)
	}

	if len(parsedPlan.Nodes) != len(plan.Nodes) {
		t.Errorf("parsed plan node count = %d, want %d", len(parsedPlan.Nodes), len(plan.Nodes))
	}
}
