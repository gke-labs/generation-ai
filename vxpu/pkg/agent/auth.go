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
	"context"
	"crypto/subtle"
	"errors"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"
)

// tokenCredentials attaches a bearer token to outgoing RPCs.
type tokenCredentials struct {
	token string
}

func (t tokenCredentials) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{
		"authorization": "Bearer " + t.token,
	}, nil
}

func (t tokenCredentials) RequireTransportSecurity() bool {
	return false
}

// WithAuthToken returns a grpc.DialOption that attaches the bearer token to every RPC on the connection.
func WithAuthToken(token string) grpc.DialOption {
	return grpc.WithPerRPCCredentials(tokenCredentials{token: token})
}

// ContextWithToken returns a context with the bearer token added to outgoing metadata.
func ContextWithToken(ctx context.Context, token string) context.Context {
	if token == "" {
		return ctx
	}
	return metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+token)
}

// extractToken extracts the bearer token from incoming metadata.
func extractToken(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	vals := md.Get("authorization")
	if len(vals) == 0 {
		return ""
	}
	val := strings.TrimSpace(vals[0])
	if strings.HasPrefix(strings.ToLower(val), "bearer ") {
		return strings.TrimSpace(val[7:])
	}
	return val
}

// checkAuth checks that the incoming call carries the expected authorization token.
func checkAuth(ctx context.Context, expectedToken string, allowAnonymous bool, method string) error {
	if allowAnonymous {
		return nil
	}
	log := klog.FromContext(ctx)
	peerAddr := "unknown"
	if p, ok := peer.FromContext(ctx); ok && p.Addr != nil {
		peerAddr = p.Addr.String()
	}

	gotToken := extractToken(ctx)
	if gotToken == "" {
		log.Info("agent call refused", "method", method, "peer", peerAddr, "reason", "missing authorization token")
		return status.Errorf(codes.Unauthenticated, "unauthenticated: missing authorization token")
	}
	if subtle.ConstantTimeCompare([]byte(gotToken), []byte(expectedToken)) != 1 {
		log.Info("agent call refused", "method", method, "peer", peerAddr, "reason", "invalid authorization token")
		return status.Errorf(codes.Unauthenticated, "unauthenticated: invalid authorization token")
	}
	return nil
}

// UnaryAuthInterceptor returns a gRPC unary server interceptor that validates
// the caller's authorization token on every method before invoking the handler.
func UnaryAuthInterceptor(expectedToken string, allowAnonymous bool) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if err := checkAuth(ctx, expectedToken, allowAnonymous, info.FullMethod); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// StreamAuthInterceptor returns a gRPC stream server interceptor that validates
// the caller's authorization token on every stream before invoking the handler.
func StreamAuthInterceptor(expectedToken string, allowAnonymous bool) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := checkAuth(ss.Context(), expectedToken, allowAnonymous, info.FullMethod); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}

// ServerAuthOptions returns gRPC server options that enforce coordinator authentication.
// It returns an error if neither an auth token nor allowAnonymous is specified, or
// if both are specified.
func ServerAuthOptions(token string, allowAnonymous bool) ([]grpc.ServerOption, error) {
	if !allowAnonymous && token == "" {
		return nil, errors.New("refusing to start: vxpu-agent requires an authentication token (--auth-token, --auth-token-file, or VXPU_AGENT_TOKEN); specify --insecure-allow-anonymous to allow unauthenticated calls on test or development machines")
	}
	if allowAnonymous && token != "" {
		return nil, errors.New("cannot specify both an authentication token and --insecure-allow-anonymous")
	}
	return []grpc.ServerOption{
		grpc.UnaryInterceptor(UnaryAuthInterceptor(token, allowAnonymous)),
		grpc.StreamInterceptor(StreamAuthInterceptor(token, allowAnonymous)),
	}, nil
}

// NewGRPCServer returns a new *grpc.Server configured with authentication interceptors.
func NewGRPCServer(token string, allowAnonymous bool, opts ...grpc.ServerOption) (*grpc.Server, error) {
	authOpts, err := ServerAuthOptions(token, allowAnonymous)
	if err != nil {
		return nil, err
	}
	allOpts := append(authOpts, opts...)
	return grpc.NewServer(allOpts...), nil
}
