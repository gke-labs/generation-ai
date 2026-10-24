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

package vm

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func runVMCommand(t *testing.T, env []string, script string) string {
	t.Helper()
	tmpDir := t.TempDir()
	gcloudPath := filepath.Join(tmpDir, "gcloud")
	gcloudScript := "#!/bin/sh\necho \"$@\"\n"
	if err := os.WriteFile(gcloudPath, []byte(gcloudScript), 0755); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("bash", "-c", script)
	cmd.Env = append(os.Environ(), "PATH="+tmpDir+":"+os.Getenv("PATH"), "ZONE=test-zone", "VM=test-vm", "PROJECT=test-proj")
	cmd.Env = append(cmd.Env, env...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %v, output: %s", err, string(out))
	}
	return strings.TrimSpace(string(out))
}

func TestTunnelThroughIAPOptional(t *testing.T) {
	tests := []struct {
		name       string
		env        []string
		wantTunnel bool
	}{
		{
			name:       "default connects directly",
			env:        []string{"VXPU_DEV_IAP="},
			wantTunnel: false,
		},
		{
			name:       "zero connects directly",
			env:        []string{"VXPU_DEV_IAP=0"},
			wantTunnel: false,
		},
		{
			name:       "one tunnels through IAP",
			env:        []string{"VXPU_DEV_IAP=1"},
			wantTunnel: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name+" (vm_ssh)", func(t *testing.T) {
			out := runVMCommand(t, tc.env, ". ./common.sh && vm_ssh 'echo hi'")
			hasTunnel := strings.Contains(out, "--tunnel-through-iap")
			if hasTunnel != tc.wantTunnel {
				t.Errorf("vm_ssh output %q has tunnel-through-iap = %v, want %v", out, hasTunnel, tc.wantTunnel)
			}
		})

		t.Run(tc.name+" (vm_scp)", func(t *testing.T) {
			out := runVMCommand(t, tc.env, ". ./common.sh && vm_scp 'file.txt' '/dest'")
			hasTunnel := strings.Contains(out, "--tunnel-through-iap")
			if hasTunnel != tc.wantTunnel {
				t.Errorf("vm_scp output %q has tunnel-through-iap = %v, want %v", out, hasTunnel, tc.wantTunnel)
			}
		})
	}
}
