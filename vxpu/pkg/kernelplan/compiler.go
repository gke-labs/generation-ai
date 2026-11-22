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

package kernelplan

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/klog/v2"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/collections"
)

// dialCompiler connects to the compiler endpoint.
// Connections will be authenticated with mTLS under #662; credentials plug in here.
func dialCompiler(endpoint string) (*grpc.ClientConn, pb.CompilerClient, error) {
	conn, err := grpc.NewClient(endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, fmt.Errorf("dialing compiler on %s: %w", endpoint, err)
	}
	return conn, pb.NewCompilerClient(conn), nil
}

// CompilerClient orchestrates package compilation by managing a compiler process
// on an agent and sending Compile gRPC requests to its published endpoint.
type CompilerClient struct {
	// agent manages compiler processes on the target machine.
	agent pb.AgentClient

	// store records compilation outcomes and clears successes.
	store PackageStore

	// dedup deduplicates concurrent requests for the same compilation inputs.
	dedup *collections.Deduplicator[string, *planpb.PackageReport]
}

// NewCompilerClient returns a CompilerClient that compiles packages using agent and records outcomes in store.
func NewCompilerClient(agent pb.AgentClient, store PackageStore) *CompilerClient {
	if agent == nil {
		panic("kernelplan: agent must not be nil")
	}
	if store == nil {
		panic("kernelplan: store must not be nil")
	}
	return &CompilerClient{
		agent: agent,
		store: store,
		dedup: collections.NewDeduplicator[string, *planpb.PackageReport](),
	}
}

// Compile compiles requested packages for an artifact under a plan by asking the agent
// for a serving compiler process, waiting until the service answers, sending the request,
// and stopping the process when done.
func (c *CompilerClient) Compile(ctx context.Context, model string, req *pb.CompileRequest) (*planpb.PackageReport, error) {
	if model == "" {
		return nil, errors.New("model name is required")
	}
	if req == nil {
		return nil, errors.New("compile request is required")
	}
	if len(req.GetPackages()) == 0 {
		return &planpb.PackageReport{
			MachineKind: req.GetMachineKind(),
		}, nil
	}

	key := compileDedupKey(model, req)
	var leaderCleanupErr error
	report, err := c.dedup.Do(ctx, key, func() (*planpb.PackageReport, bool, error) {
		rep, cleanupErr, compileErr := c.compileDirect(ctx, model, req)
		if cleanupErr != nil {
			leaderCleanupErr = cleanupErr
		}
		if compileErr != nil {
			return nil, false, compileErr
		}
		return rep, true, nil
	})
	if err != nil {
		return nil, err
	}
	if leaderCleanupErr != nil {
		return nil, leaderCleanupErr
	}
	return report, nil
}

func (c *CompilerClient) compileDirect(ctx context.Context, model string, req *pb.CompileRequest) (report *planpb.PackageReport, cleanupErr error, compileErr error) {
	procID := newProcessID()

	// 1. Ask agent for a serving compile process.
	_, err := c.agent.StartProcess(ctx, &pb.StartProcessRequest{
		Id:      procID,
		Process: "compiler",
	})
	if err != nil {
		return nil, nil, fmt.Errorf("starting compiler process: %w", err)
	}

	// 2. Stop the process when work completes, or caller fails or is cancelled.
	defer func() {
		stopCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		if _, stopErr := c.agent.StopProcess(stopCtx, &pb.StopProcessRequest{Id: procID}); stopErr != nil {
			log := klog.FromContext(ctx)
			log.Error(stopErr, "stopping compiler process", "id", procID)
			if cleanupErr == nil {
				cleanupErr = fmt.Errorf("stopping compiler process %s: %w", procID, stopErr)
			}
		}
	}()

	// 3. Find the endpoint in the process table.
	statusResp, err := c.agent.GetProcess(ctx, &pb.GetProcessRequest{Id: procID})
	if err != nil {
		return nil, cleanupErr, fmt.Errorf("getting compiler process status: %w", err)
	}
	st := statusResp.GetStatus()
	if st == nil {
		return nil, cleanupErr, errors.New("compiler process status is nil")
	}
	endpoint := st.GetEndpoint()
	if endpoint == "" {
		if st.GetState() != "running" || st.GetExitCode() != 0 || st.GetSignal() != 0 {
			return nil, cleanupErr, formatExitedError(st)
		}
		return nil, cleanupErr, fmt.Errorf("compiler process %s has no published endpoint", procID)
	}

	// 4. Connect to published endpoint.
	conn, compClient, err := dialCompiler(endpoint)
	if err != nil {
		return nil, cleanupErr, fmt.Errorf("dialing compiler on %s: %w", endpoint, err)
	}
	defer func() {
		if closeErr := conn.Close(); closeErr != nil {
			log := klog.FromContext(ctx)
			log.Error(closeErr, "closing compiler connection", "endpoint", endpoint)
			if cleanupErr == nil {
				cleanupErr = fmt.Errorf("closing compiler connection to %s: %w", endpoint, closeErr)
			}
		}
	}()

	// 5. Establish readiness over the connection.
	if err := c.waitForReady(ctx, procID, conn); err != nil {
		return nil, cleanupErr, err
	}

	// 6. Send the compile request.
	resp, err := compClient.Compile(ctx, req)
	if err != nil {
		return nil, cleanupErr, fmt.Errorf("compile RPC: %w", err)
	}
	rep := resp.GetReport()
	if rep == nil {
		return nil, cleanupErr, errors.New("compile service returned nil PackageReport")
	}

	// 7. Record outcomes in the package store.
	for _, f := range rep.GetFailed() {
		if err := c.store.RecordCompileFailure(ctx, model, f); err != nil {
			return nil, cleanupErr, fmt.Errorf("recording compile failure: %w", err)
		}
	}
	for _, p := range rep.GetPresent() {
		if err := c.store.ClearCompileFailure(ctx, model, p.GetKey()); err != nil {
			return nil, cleanupErr, fmt.Errorf("clearing compile failure for present package %s: %w", p.GetName(), err)
		}
	}

	return rep, cleanupErr, nil
}

func (c *CompilerClient) waitForReady(ctx context.Context, procID string, conn *grpc.ClientConn) error {
	healthClient := grpc_health_v1.NewHealthClient(conn)

	for {
		if ctx.Err() != nil {
			return c.checkExitedOrCtxErr(ctx, procID)
		}

		// Check whether the process exited before serving.
		statusResp, err := c.agent.GetProcess(ctx, &pb.GetProcessRequest{Id: procID})
		if err != nil {
			if ctx.Err() != nil {
				return c.checkExitedOrCtxErr(ctx, procID)
			}
			return fmt.Errorf("checking compiler process status: %w", err)
		}
		st := statusResp.GetStatus()
		if st == nil {
			return errors.New("compiler process status is nil")
		}
		if st.GetState() != "running" || st.GetExitCode() != 0 || st.GetSignal() != 0 {
			return formatExitedError(st)
		}

		// Check health over the connection.
		resp, checkErr := healthClient.Check(ctx, &grpc_health_v1.HealthCheckRequest{})
		if checkErr == nil && resp.GetStatus() == grpc_health_v1.HealthCheckResponse_SERVING {
			return nil
		}

		select {
		case <-ctx.Done():
			return c.checkExitedOrCtxErr(ctx, procID)
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func (c *CompilerClient) checkExitedOrCtxErr(ctx context.Context, procID string) error {
	lookupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if statusResp, err := c.agent.GetProcess(lookupCtx, &pb.GetProcessRequest{Id: procID}); err == nil && statusResp.GetStatus() != nil {
		st := statusResp.GetStatus()
		if st.GetState() != "running" || st.GetExitCode() != 0 || st.GetSignal() != 0 {
			return formatExitedError(st)
		}
	}
	return ctx.Err()
}

func formatExitedError(st *pb.ProcessStatus) error {
	out := strings.TrimSpace(st.GetOutput())
	if out != "" {
		return fmt.Errorf("compiler process %s (exit code %d, signal %d):\n%s",
			st.GetState(), st.GetExitCode(), st.GetSignal(), out)
	}
	return fmt.Errorf("compiler process %s (exit code %d, signal %d)",
		st.GetState(), st.GetExitCode(), st.GetSignal())
}

func compileDedupKey(model string, req *pb.CompileRequest) string {
	pkgs := make([]string, len(req.GetPackages()))
	for i, p := range req.GetPackages() {
		pkgs[i] = p.GetName() + "=" + p.GetKey()
	}
	sort.Strings(pkgs)
	return fmt.Sprintf("model=%s;art=%s;plan=%s;out=%s;mk=%s;ver=%s;dev=%s;par=%d;pkgs=%s",
		model,
		req.GetArtifact(),
		req.GetPlanFile(),
		req.GetOutDir(),
		req.GetMachineKind(),
		req.GetSoftwareVersion(),
		req.GetDevice(),
		req.GetParallelBuilds(),
		strings.Join(pkgs, ","),
	)
}

func newProcessID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("compiler-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("compiler-%s", hex.EncodeToString(b))
}
