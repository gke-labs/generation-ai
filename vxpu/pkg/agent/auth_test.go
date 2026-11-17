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

package agent

import (
	"fmt"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

func TestAgentAuthentication_AllMethods(t *testing.T) {
	ctx := t.Context()
	token := "valid-secret-token"

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer lis.Close()

	srv := testServer(t)
	grpcServer, err := NewGRPCServer(token, false)
	if err != nil {
		t.Fatalf("NewGRPCServer: %v", err)
	}
	pb.RegisterAgentServer(grpcServer, srv)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.GracefulStop()

	addr := lis.Addr().String()

	connNoAuth, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer connNoAuth.Close()

	connWrongAuth, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()), WithAuthToken("wrong-token"))
	if err != nil {
		t.Fatal(err)
	}
	defer connWrongAuth.Close()

	connValidAuth, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()), WithAuthToken(token))
	if err != nil {
		t.Fatal(err)
	}
	defer connValidAuth.Close()

	clientNoAuth := pb.NewAgentClient(connNoAuth)
	clientWrongAuth := pb.NewAgentClient(connWrongAuth)
	clientValidAuth := pb.NewAgentClient(connValidAuth)

	// 1. MountWeights
	reqMount := &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}
	if _, err := clientNoAuth.MountWeights(ctx, reqMount); status.Code(err) != codes.Unauthenticated {
		t.Errorf("MountWeights without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.MountWeights(ctx, reqMount); status.Code(err) != codes.Unauthenticated {
		t.Errorf("MountWeights with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.MountWeights(ctx, reqMount); status.Code(err) == codes.Unauthenticated {
		t.Errorf("MountWeights with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// 2. UnmountWeights
	reqUnmount := &pb.UnmountWeightsRequest{DeviceName: "vxpu-weights"}
	if _, err := clientNoAuth.UnmountWeights(ctx, reqUnmount); status.Code(err) != codes.Unauthenticated {
		t.Errorf("UnmountWeights without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.UnmountWeights(ctx, reqUnmount); status.Code(err) != codes.Unauthenticated {
		t.Errorf("UnmountWeights with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.UnmountWeights(ctx, reqUnmount); status.Code(err) == codes.Unauthenticated {
		t.Errorf("UnmountWeights with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// 3. StartProcess
	reqStart := &pb.StartProcessRequest{Id: "p1", Process: "loop"}
	if _, err := clientNoAuth.StartProcess(ctx, reqStart); status.Code(err) != codes.Unauthenticated {
		t.Errorf("StartProcess without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.StartProcess(ctx, reqStart); status.Code(err) != codes.Unauthenticated {
		t.Errorf("StartProcess with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.StartProcess(ctx, reqStart); status.Code(err) == codes.Unauthenticated {
		t.Errorf("StartProcess with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// 4. StopProcess
	reqStop := &pb.StopProcessRequest{Id: "p1"}
	if _, err := clientNoAuth.StopProcess(ctx, reqStop); status.Code(err) != codes.Unauthenticated {
		t.Errorf("StopProcess without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.StopProcess(ctx, reqStop); status.Code(err) != codes.Unauthenticated {
		t.Errorf("StopProcess with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.StopProcess(ctx, reqStop); status.Code(err) == codes.Unauthenticated {
		t.Errorf("StopProcess with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// 5. GetProcess
	reqGet := &pb.GetProcessRequest{Id: "p1"}
	if _, err := clientNoAuth.GetProcess(ctx, reqGet); status.Code(err) != codes.Unauthenticated {
		t.Errorf("GetProcess without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.GetProcess(ctx, reqGet); status.Code(err) != codes.Unauthenticated {
		t.Errorf("GetProcess with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.GetProcess(ctx, reqGet); status.Code(err) == codes.Unauthenticated {
		t.Errorf("GetProcess with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// 6. ListProcesses
	reqList := &pb.ListProcessesRequest{}
	if _, err := clientNoAuth.ListProcesses(ctx, reqList); status.Code(err) != codes.Unauthenticated {
		t.Errorf("ListProcesses without auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientWrongAuth.ListProcesses(ctx, reqList); status.Code(err) != codes.Unauthenticated {
		t.Errorf("ListProcesses with wrong auth: got status %v (%v), want Unauthenticated", status.Code(err), err)
	}
	if _, err := clientValidAuth.ListProcesses(ctx, reqList); status.Code(err) == codes.Unauthenticated {
		t.Errorf("ListProcesses with valid auth: unexpectedly refused with Unauthenticated: %v", err)
	}

	// Programmatic check: verify that every method registered in the Agent service descriptor
	// is intercepted and rejected with Unauthenticated before handler execution when called
	// without authentication or with wrong authentication. This covers any future RPC method
	// added to Agent service without its author needing to write auth checks in the method.
	for _, method := range pb.Agent_ServiceDesc.Methods {
		fullMethod := fmt.Sprintf("/%s/%s", pb.Agent_ServiceDesc.ServiceName, method.MethodName)
		var out any

		errNoAuth := connNoAuth.Invoke(ctx, fullMethod, &pb.ListProcessesRequest{}, out)
		if status.Code(errNoAuth) != codes.Unauthenticated {
			t.Errorf("method %s without auth: got status %v, want Unauthenticated", fullMethod, status.Code(errNoAuth))
		}

		errWrongAuth := connWrongAuth.Invoke(ctx, fullMethod, &pb.ListProcessesRequest{}, out)
		if status.Code(errWrongAuth) != codes.Unauthenticated {
			t.Errorf("method %s with wrong auth: got status %v, want Unauthenticated", fullMethod, status.Code(errWrongAuth))
		}

		// Valid auth: the interceptor admits the call, so it is not refused with Unauthenticated.
		errValid := connValidAuth.Invoke(ctx, fullMethod, &pb.ListProcessesRequest{}, out)
		if status.Code(errValid) == codes.Unauthenticated {
			t.Errorf("method %s with valid auth: unexpectedly refused with Unauthenticated: %v", fullMethod, errValid)
		}
	}
}

func TestAgentAuthentication_StartupSettings(t *testing.T) {
	// Refuses to start when no token is configured and allowAnonymous is false.
	_, err := NewGRPCServer("", false)
	if err == nil {
		t.Error("want error when starting without token and without allowAnonymous, got nil")
	}

	// Refuses to start when both token and allowAnonymous are configured.
	_, err = NewGRPCServer("some-token", true)
	if err == nil {
		t.Error("want error when starting with both token and allowAnonymous, got nil")
	}

	// Starts successfully when token is configured.
	sToken, err := NewGRPCServer("valid-token", false)
	if err != nil {
		t.Fatalf("unexpected error starting with valid token: %v", err)
	}
	sToken.Stop()

	// Starts successfully when allowAnonymous is explicitly set.
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer lis.Close()

	srv := testServer(t)
	sAnon, err := NewGRPCServer("", true)
	if err != nil {
		t.Fatalf("unexpected error starting with allowAnonymous: %v", err)
	}
	pb.RegisterAgentServer(sAnon, srv)

	go func() {
		_ = sAnon.Serve(lis)
	}()
	defer sAnon.GracefulStop()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	client := pb.NewAgentClient(conn)
	resp, err := client.ListProcesses(t.Context(), &pb.ListProcessesRequest{})
	if err != nil {
		t.Fatalf("anonymous ListProcesses: %v", err)
	}
	if resp == nil {
		t.Error("expected non-nil response for anonymous call")
	}
}

func TestAgentAuthentication_ContextWithToken(t *testing.T) {
	ctx := t.Context()
	token := "per-call-token"

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer lis.Close()

	srv := testServer(t)
	grpcServer, err := NewGRPCServer(token, false)
	if err != nil {
		t.Fatalf("NewGRPCServer: %v", err)
	}
	pb.RegisterAgentServer(grpcServer, srv)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.GracefulStop()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	client := pb.NewAgentClient(conn)

	// Call without token in context fails with Unauthenticated.
	if _, err := client.ListProcesses(ctx, &pb.ListProcessesRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Errorf("call without token: got %v, want Unauthenticated", status.Code(err))
	}

	// Call with wrong token in context fails with Unauthenticated.
	ctxWrong := ContextWithToken(ctx, "wrong-token")
	if _, err := client.ListProcesses(ctxWrong, &pb.ListProcessesRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Errorf("call with wrong token: got %v, want Unauthenticated", status.Code(err))
	}

	// Call with valid token in context succeeds.
	ctxValid := ContextWithToken(ctx, token)
	if _, err := client.ListProcesses(ctxValid, &pb.ListProcessesRequest{}); err != nil {
		t.Errorf("call with valid token: unexpected error %v", err)
	}
}
