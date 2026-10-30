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
	"context"
	"net"
	"sync"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
)

// rangeExecutor loads what it is sent and serves its range once loaded:
// the first ServeRange finds it still loading.
type rangeExecutor struct {
	pb.UnimplementedExecutorServer
	mu     sync.Mutex
	plan   map[string]string
	served int
}

func (e *rangeExecutor) LoadModel(_ context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.plan = req.Plan
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *rangeExecutor) ServeRange(context.Context, *pb.ServeRangeRequest) (*pb.ServeRangeResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.served++; e.served == 1 {
		return &pb.ServeRangeResponse{State: pb.LoadState_LOAD_STATE_LOADING}, nil
	}
	return &pb.ServeRangeResponse{Port: 7100, LayerRange: e.plan["VXPU_LAYER_RANGE"], State: pb.LoadState_LOAD_STATE_READY}, nil
}

func deepseekArtifact() planner.Artifact {
	a := planner.Artifact{DenseBytes: 9_800_000_000, Experts: map[int]planner.Layer{}, TopK: 6, EngramLayers: 2,
		Engram: []int{1, 14}, DecodeFP8Bytes: 6_836_715_520, DecodeOtherBytes: 1_686_144_448, Cuts: []int{1, 2, 8, 14, 20}}
	for layer := range 40 {
		a.Experts[layer] = planner.Layer{Experts: 384, BytesPerExpert: 18_800_000}
	}
	return a
}

// On two g2-standard-48s the cheapest layout runs layers 0-19 on the
// engine and 20-39 on the other: the router loads the other with its
// range, waits for it to serve, and wires the engine to it.
func TestChooseRanges_LoadsAndWiresTheOtherRange(t *testing.T) {
	ctx := t.Context()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	other := &rangeExecutor{}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, other)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	g2, err := machines.Lookup("g2-standard-48")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{}
	mp := modelPlan{}
	sw, _ := s.chooseRanges(ctx, "m", deepseekArtifact(), &executorInfo{Endpoint: "10.0.0.1:50051", Probe: machines.ProbeOf(g2)},
		[]planner.Executor{{Address: lis.Addr().String(), Probe: machines.ProbeOf(g2)}}, nil,
		map[string]string{"VXPU_SEGMENT_RUNTIME": "aoti"}, []byte("{}"), []byte("{}"), nil, nil, &mp)
	if sw == nil {
		t.Fatalf("no layer ranges: %v (candidates %+v)", mp.Reasons, mp.Candidates)
	}
	if sw["VXPU_LAYER_RANGE"] != "0:20" || sw["VXPU_REMOTE_RANGES"] != "20:=tcp://127.0.0.1:7100" || sw["VXPU_DATAFLOW"] != "1" {
		t.Errorf("engine switches %v", sw)
	}
	if other.plan["VXPU_LAYER_RANGE"] != "20:" || other.plan["VXPU_DATAFLOW"] != "1" ||
		other.plan["VXPU_SEGMENT_RUNTIME"] != "aoti" || other.plan["VXPU_STEP_GRAPH"] != "1" {
		t.Errorf("the other range's plan %v", other.plan)
	}
	if s.expertHolders[lis.Addr().String()] != "10.0.0.1:50051" {
		t.Errorf("the other executor is not claimed for the engine: %v", s.expertHolders)
	}
	if len(mp.Candidates) < 2 || mp.Candidates[0].Kind != "layer-ranges" {
		t.Errorf("candidates %+v", mp.Candidates)
	}
}

// Without knowing where the step can be cut, the router never ranges.
func TestChooseRanges_NotWithoutCuts(t *testing.T) {
	g2, err := machines.Lookup("g2-standard-48")
	if err != nil {
		t.Fatal(err)
	}
	a := deepseekArtifact()
	a.Cuts = nil
	s := &Server{}
	if sw, _ := s.chooseRanges(t.Context(), "m", a, &executorInfo{Endpoint: "e", Probe: machines.ProbeOf(g2)},
		[]planner.Executor{{Address: "o", Probe: machines.ProbeOf(g2)}}, nil, nil, nil, nil, nil, nil, &modelPlan{}); sw != nil {
		t.Errorf("ranged without cuts: %v", sw)
	}
}
