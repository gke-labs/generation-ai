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

package main

import (
	"context"
	"net"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/gke-labs/generation-ai/vxpu/pkg/agent"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/mtls"
	"github.com/gke-labs/generation-ai/vxpu/pkg/pki"
)

func TestDialAgentMTLS(t *testing.T) {
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

	// Dialing with coordinator mTLS credentials succeeds.
	client, done, err := dialAgent(t.Context(), lis.Addr().String(), filepath.Join(pkiDir, "coordinator"))
	if err != nil {
		t.Fatalf("dialAgent(mTLS): %v", err)
	}
	defer done()
	callCtx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if _, err := client.DescribeMachine(callCtx, &pb.DescribeMachineRequest{}); err != nil {
		t.Fatalf("DescribeMachine over CLI mTLS failed: %v", err)
	}

	// Dialing without --tls-dir uses plaintext and is rejected by the mTLS agent.
	plainClient, plainDone, err := dialAgent(t.Context(), lis.Addr().String(), "")
	if err != nil {
		t.Fatalf("dialAgent(plaintext): %v", err)
	}
	defer plainDone()
	plainCtx, plainCancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer plainCancel()
	if _, err := plainClient.DescribeMachine(plainCtx, &pb.DescribeMachineRequest{}); err == nil {
		t.Fatalf("expected plaintext CLI dialAgent to be rejected by mTLS agent")
	}

	// Dialing with an invalid --tls-dir returns an error immediately.
	if _, _, err := dialAgent(t.Context(), lis.Addr().String(), filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatalf("expected error for missing --tls-dir")
	}
}
