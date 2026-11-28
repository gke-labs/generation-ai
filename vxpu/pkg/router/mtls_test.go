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
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"google.golang.org/grpc"
	"k8s.io/client-go/kubernetes/fake"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/mtls"
	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func TestServerAgentMTLS(t *testing.T) {
	pkiDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), pkiDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir: %v", err)
	}

	serverCreds, err := mtls.LoadAgentServerCredentials(t.Context(), filepath.Join(pkiDir, "agent"))
	if err != nil {
		t.Fatalf("LoadAgentServerCredentials: %v", err)
	}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	gs := grpc.NewServer(grpc.Creds(serverCreds))
	pb.RegisterAgentServer(gs, agent.NewServer(t.TempDir(), t.TempDir(), nil))
	go func() {
		_ = gs.Serve(lis)
	}()
	t.Cleanup(func() {
		gs.Stop()
		_ = lis.Close()
	})

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("SplitHostPort: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("Atoi: %v", err)
	}

	// Valid coordinator credentials connect to the mTLS agent.
	coordCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(pkiDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials: %v", err)
	}
	srv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = srv.Close() })
	srv.agentPort = port
	srv.UseAgentCredentials(coordCreds)

	conn, err := srv.agentConn(host)
	if err != nil {
		t.Fatalf("agentConn: %v", err)
	}
	callCtx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if _, err := pb.NewAgentClient(conn).DescribeMachine(callCtx, &pb.DescribeMachineRequest{}); err != nil {
		t.Fatalf("DescribeMachine over router mTLS failed: %v", err)
	}

	// Plaintext router (no agent credentials) is rejected by the mTLS agent.
	plainSrv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = plainSrv.Close() })
	plainSrv.agentPort = port
	plainConn, err := plainSrv.agentConn(host)
	if err != nil {
		t.Fatalf("plain agentConn: %v", err)
	}
	plainCtx, plainCancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer plainCancel()
	if _, err := pb.NewAgentClient(plainConn).DescribeMachine(plainCtx, &pb.DescribeMachineRequest{}); err == nil {
		t.Fatalf("expected plaintext router connection to be rejected by mTLS agent")
	}

	// Router with wrong-CA coordinator credentials is rejected by the mTLS agent.
	otherPKIDir := t.TempDir()
	if err := pki.EnsureDir(t.Context(), otherPKIDir, pki.Options{}); err != nil {
		t.Fatalf("pki.EnsureDir(other): %v", err)
	}
	wrongCreds, err := mtls.LoadCoordinatorClientCredentials(t.Context(), filepath.Join(otherPKIDir, "coordinator"))
	if err != nil {
		t.Fatalf("LoadCoordinatorClientCredentials(other): %v", err)
	}
	wrongSrv := NewServer(fake.NewClientset(), "default", "img:v1", "")
	t.Cleanup(func() { _ = wrongSrv.Close() })
	wrongSrv.agentPort = port
	wrongSrv.UseAgentCredentials(wrongCreds)
	wrongConn, err := wrongSrv.agentConn(host)
	if err != nil {
		t.Fatalf("wrong agentConn: %v", err)
	}
	wrongCtx, wrongCancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer wrongCancel()
	if _, err := pb.NewAgentClient(wrongConn).DescribeMachine(wrongCtx, &pb.DescribeMachineRequest{}); err == nil {
		t.Fatalf("expected wrong-CA router connection to be rejected by mTLS agent")
	}
}
