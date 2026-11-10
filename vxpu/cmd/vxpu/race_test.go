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
	"slices"
	"strings"
	"testing"
)

func TestParseZones(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"us-central1-b", []string{"us-central1-b"}},
		{"us-central1-b,us-central1-c,us-east4-b", []string{"us-central1-b", "us-central1-c", "us-east4-b"}},
		{" us-central1-b ,  us-central1-c , us-east4-b ", []string{"us-central1-b", "us-central1-c", "us-east4-b"}},
		{"us-central1-b,us-central1-c,us-central1-b", []string{"us-central1-b", "us-central1-c"}},
		{"", nil},
		{",,,", nil},
	}

	for _, tc := range tests {
		got := parseZones(tc.input)
		if !slices.Equal(got, tc.want) {
			t.Errorf("parseZones(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestParseAccelerator(t *testing.T) {
	tests := []struct {
		input     string
		wantType  string
		wantCount int
		wantErr   bool
	}{
		{
			input:     "type=nvidia-rtx-pro-6000,count=4",
			wantType:  "nvidia-rtx-pro-6000",
			wantCount: 4,
		},
		{
			input:     "type=nvidia-rtx-pro-6000",
			wantType:  "nvidia-rtx-pro-6000",
			wantCount: 1,
		},
		{
			input:     "type=nvidia-l4,count=1",
			wantType:  "nvidia-l4",
			wantCount: 1,
		},
		{
			input:     "type = nvidia-rtx-pro-6000 , count = 2",
			wantType:  "nvidia-rtx-pro-6000",
			wantCount: 2,
		},
		{
			input:   "nvidia-rtx-pro-6000:4",
			wantErr: true,
		},
		{
			input:   "nvidia-rtx-pro-6000",
			wantErr: true,
		},
		{
			input:   "count=4",
			wantErr: true,
		},
		{
			input:   "type=nvidia-rtx-pro-6000,count=0",
			wantErr: true,
		},
		{
			input:   "type=nvidia-rtx-pro-6000,count=-1",
			wantErr: true,
		},
		{
			input:   "type=nvidia-rtx-pro-6000,count=abc",
			wantErr: true,
		},
		{
			input:   "type=nvidia-rtx-pro-6000,unknown=val",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		got, err := parseAccelerator(tc.input)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseAccelerator(%q) expected error, got nil", tc.input)
			}
		} else {
			if err != nil {
				t.Errorf("parseAccelerator(%q) unexpected error: %v", tc.input, err)
			}
			if got.Type != tc.wantType || got.Count != tc.wantCount {
				t.Errorf("parseAccelerator(%q) = %+v, want Type=%q Count=%d", tc.input, got, tc.wantType, tc.wantCount)
			}
		}
	}
}

func TestCmdRace_FlagValidation(t *testing.T) {
	ctx := t.Context()

	tests := []struct {
		name    string
		args    []string
		env     map[string]string
		wantErr string
	}{
		{
			name:    "missing machine-type",
			args:    []string{"--zones", "us-central1-b"},
			wantErr: "requires --machine-type and --zones",
		},
		{
			name:    "missing zones",
			args:    []string{"--machine-type", "g4-standard-192"},
			wantErr: "requires --machine-type and --zones",
		},
		{
			name:    "empty zones",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", " , "},
			wantErr: "at least one valid zone",
		},
		{
			name:    "missing project",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", "us-central1-b"},
			wantErr: "GCP project not specified",
		},
		{
			name:    "invalid accelerator",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", "us-central1-b", "--project", "test-p", "--accelerator", "bad:spec"},
			wantErr: "invalid accelerator item",
		},
		{
			name:    "timeout flag rejected in favor of time-budget",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", "us-central1-b", "--project", "test-p", "--timeout", "10m"},
			wantErr: "flag provided but not defined: -timeout",
		},
		{
			name:    "invalid time-budget syntax",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", "us-central1-b", "--project", "test-p", "--time-budget", "invalid"},
			wantErr: "invalid value \"invalid\" for flag -time-budget",
		},
		{
			name:    "invalid max-run-duration syntax",
			args:    []string{"--machine-type", "g4-standard-192", "--zones", "us-central1-b", "--project", "test-p", "--max-run-duration", "invalid"},
			wantErr: "invalid value \"invalid\" for flag -max-run-duration",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("PROJECT", "")
			t.Setenv("GOOGLE_CLOUD_PROJECT", "")
			for k, v := range tc.env {
				t.Setenv(k, v)
			}

			err := cmdRace(ctx, tc.args)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}
