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
	"strings"
	"testing"
)

func TestApplyManifest_MissingKindOrName(t *testing.T) {
	orig := getK8sClient
	t.Cleanup(func() { getK8sClient = orig })
	getK8sClient = func() (*k8sClient, error) { return &k8sClient{}, nil }

	manifestWithoutKind := `
apiVersion: v1
metadata:
  name: test-name
`
	err := applyManifest(t.Context(), manifestWithoutKind)
	if err == nil {
		t.Fatal("expected error for manifest without kind, got nil")
	}
	if !strings.Contains(err.Error(), "manifest document 0 has no kind or name") {
		t.Errorf("unexpected error message: %v", err)
	}

	manifestWithoutName := `
apiVersion: v1
kind: ServiceAccount
metadata:
  namespace: test-ns
`
	err = applyManifest(t.Context(), manifestWithoutName)
	if err == nil {
		t.Fatal("expected error for manifest without name, got nil")
	}
	if !strings.Contains(err.Error(), "manifest document 0 has no kind or name") {
		t.Errorf("unexpected error message: %v", err)
	}
}
