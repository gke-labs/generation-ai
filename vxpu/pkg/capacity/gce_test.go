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

package capacity

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/api/option"
)

func TestGCEBackend_Create(t *testing.T) {
	tests := []struct {
		name         string
		insertStatus int
		insertBody   string
		opBody       string
		wantErr      bool
		wantUnsupp   bool
		wantQuoErr   bool
	}{
		{
			name:         "success",
			insertStatus: http.StatusOK,
			insertBody:   `{"name":"op-success","status":"RUNNING"}`,
			opBody:       `{"name":"op-success","status":"DONE"}`,
		},
		{
			name:         "quota error on operation",
			insertStatus: http.StatusOK,
			insertBody:   `{"name":"op-quota","status":"RUNNING"}`,
			opBody:       `{"name":"op-quota","status":"DONE","httpErrorStatusCode":403,"httpErrorMessage":"Quota exceeded","error":{"errors":[{"code":"ZONE_RESOURCE_POOL_EXHAUSTED","message":"Quota exceeded"}]}}`,
			wantErr:      true,
			wantQuoErr:   true,
		},
		{
			name:         "unsupported zone on operation with invalid machine type",
			insertStatus: http.StatusOK,
			insertBody:   `{"name":"op-unsupp","status":"RUNNING"}`,
			opBody:       `{"name":"op-unsupp","status":"DONE","httpErrorStatusCode":400,"httpErrorMessage":"Invalid machine type","error":{"errors":[{"code":"INVALID","location":"resource.machineType","message":"Machine type not available in zone"}]}}`,
			wantErr:      true,
			wantUnsupp:   true,
		},
		{
			name:         "unsupported zone on insert with invalidUsage",
			insertStatus: http.StatusBadRequest,
			insertBody:   `{"error":{"code":400,"message":"Machine type not available in zone","errors":[{"reason":"invalidUsage"}]}}`,
			wantErr:      true,
			wantUnsupp:   true,
		},
		{
			name:         "unsupported zone on insert with invalid machineType location",
			insertStatus: http.StatusBadRequest,
			insertBody:   `{"error":{"code":400,"message":"Invalid value for field resource.machineType","errors":[{"reason":"invalid","location":"resource.machineType"}]}}`,
			wantErr:      true,
			wantUnsupp:   true,
		},
		{
			name:         "bare 400 is not unsupported zone",
			insertStatus: http.StatusBadRequest,
			insertBody:   `{"error":{"code":400,"message":"Malformed request body","errors":[]}}`,
			wantErr:      true,
			wantUnsupp:   false,
		},
		{
			name:         "invalid disk type is not unsupported zone",
			insertStatus: http.StatusBadRequest,
			insertBody:   `{"error":{"code":400,"message":"Invalid disk type","errors":[{"reason":"invalid","location":"resource.disks[0].initializeParams.diskType"}]}}`,
			wantErr:      true,
			wantUnsupp:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if strings.Contains(r.URL.Path, "/operations/") {
					w.WriteHeader(http.StatusOK)
					if _, err := w.Write([]byte(tc.opBody)); err != nil {
						t.Errorf("write op: %v", err)
					}
					return
				}
				w.WriteHeader(tc.insertStatus)
				if _, err := w.Write([]byte(tc.insertBody)); err != nil {
					t.Errorf("write insert: %v", err)
				}
			}))
			defer server.Close()

			backend, err := NewGCEBackend(t.Context(), "test-proj",
				option.WithEndpoint(server.URL),
				option.WithoutAuthentication(),
			)
			if err != nil {
				t.Fatalf("NewGCEBackend: %v", err)
			}
			defer func() {
				if err := backend.Close(); err != nil {
					t.Errorf("backend.Close: %v", err)
				}
			}()

			spec := MachineSpec{
				Name:        "test-vm",
				MachineType: "g4-standard-192",
				Accelerators: []AcceleratorSpec{
					{Type: "nvidia-rtx-pro-6000", Count: 4},
				},
				MaxRunDuration:       2 * time.Hour,
				RequestValidDuration: 1 * time.Hour,
			}

			err = backend.Create(t.Context(), "us-central1-a", spec)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.wantUnsupp && !errors.Is(err, ErrUnsupportedZone) {
					t.Fatalf("expected ErrUnsupportedZone, got %v", err)
				}
				if !tc.wantUnsupp && errors.Is(err, ErrUnsupportedZone) {
					t.Fatalf("error must not be ErrUnsupportedZone: %v", err)
				}
				if tc.wantQuoErr && errors.Is(err, ErrUnsupportedZone) {
					t.Fatalf("quota error must not be ErrUnsupportedZone: %v", err)
				}
				if errors.Is(err, ErrNotFound) {
					t.Fatalf("create error must not be ErrNotFound: %v", err)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestGCEBackend_Describe(t *testing.T) {
	tests := []struct {
		name         string
		status       int
		body         string
		wantErr      bool
		wantNotFound bool
		wantStatus   InstanceStatus
		wantType     string
	}{
		{
			name:       "running instance",
			status:     http.StatusOK,
			body:       `{"name":"test-vm","status":"RUNNING","machineType":"zones/us-central1-a/machineTypes/g4-standard-192","creationTimestamp":"2026-11-10T12:00:00Z"}`,
			wantStatus: StatusRunning,
			wantType:   "zones/us-central1-a/machineTypes/g4-standard-192",
		},
		{
			name:         "not found",
			status:       http.StatusNotFound,
			body:         `{"error":{"code":404,"message":"Not Found","errors":[{"reason":"notFound"}]}}`,
			wantErr:      true,
			wantNotFound: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Errorf("write: %v", err)
				}
			}))
			defer server.Close()

			backend, err := NewGCEBackend(t.Context(), "test-proj",
				option.WithEndpoint(server.URL),
				option.WithoutAuthentication(),
			)
			if err != nil {
				t.Fatalf("NewGCEBackend: %v", err)
			}
			defer func() {
				if err := backend.Close(); err != nil {
					t.Errorf("backend.Close: %v", err)
				}
			}()

			inst, err := backend.Describe(t.Context(), "us-central1-a", "test-vm")
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.wantNotFound && !errors.Is(err, ErrNotFound) {
					t.Fatalf("expected ErrNotFound, got %v", err)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if inst.Status != tc.wantStatus {
					t.Errorf("status = %v, want %v", inst.Status, tc.wantStatus)
				}
				if inst.MachineType != tc.wantType {
					t.Errorf("machineType = %v, want %v", inst.MachineType, tc.wantType)
				}
				if inst.CreationTimestamp.IsZero() {
					t.Error("expected non-zero creation timestamp")
				}
			}
		})
	}
}

func TestGCEBackend_Delete(t *testing.T) {
	tests := []struct {
		name         string
		status       int
		body         string
		wantErr      bool
		wantNotFound bool
	}{
		{
			name:   "successful delete",
			status: http.StatusOK,
			body:   `{"name":"op-del","status":"DONE"}`,
		},
		{
			name:         "404 on delete",
			status:       http.StatusNotFound,
			body:         `{"error":{"code":404,"message":"Not Found","errors":[{"reason":"notFound"}]}}`,
			wantErr:      true,
			wantNotFound: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Errorf("write: %v", err)
				}
			}))
			defer server.Close()

			backend, err := NewGCEBackend(t.Context(), "test-proj",
				option.WithEndpoint(server.URL),
				option.WithoutAuthentication(),
			)
			if err != nil {
				t.Fatalf("NewGCEBackend: %v", err)
			}
			defer func() {
				if err := backend.Close(); err != nil {
					t.Errorf("backend.Close: %v", err)
				}
			}()

			err = backend.Delete(t.Context(), "us-central1-a", "test-vm")
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.wantNotFound && !errors.Is(err, ErrNotFound) {
					t.Fatalf("expected ErrNotFound, got %v", err)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
