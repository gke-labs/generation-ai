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

package planner

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestReadArtifactDraftFacts(t *testing.T) {
	manifestJSON := []byte(`{
		"config": {"num_experts_per_tok": 2},
		"tensors": {}
	}`)
	bindingJSON := []byte(`{
		"bound": {},
		"step_tokens": 4,
		"draft": {
			"block": 3
		}
	}`)
	artifact, err := ReadArtifact(manifestJSON, bindingJSON)
	if err != nil {
		t.Fatalf("ReadArtifact failed: %v", err)
	}
	if artifact.StepTokens != 4 {
		t.Errorf("got step_tokens %d, want 4", artifact.StepTokens)
	}
	if artifact.DraftBlock != 3 {
		t.Errorf("got draft block %d, want 3", artifact.DraftBlock)
	}
}

func TestReadArtifactDraftFactsFromPython(t *testing.T) {
	pythonExe := "python3"
	if p, err := exec.LookPath(".venv/bin/python"); err == nil {
		pythonExe = p
	} else if p, err := exec.LookPath("../../.venv/bin/python"); err == nil {
		pythonExe = p
	} else if p, err := exec.LookPath("../../../.venv/bin/python"); err == nil {
		pythonExe = p
	}

	checkCmd := exec.Command(pythonExe, "-c", "import vxpu.export, vxpu.fixtures.tiny_deepseek_v41")
	if err := checkCmd.Run(); err != nil {
		t.Skipf("python with vxpu export not available (%s): %v", pythonExe, err)
	}

	td := t.TempDir()
	script := `
import os, sys
from vxpu.fixtures import tiny_deepseek_v41
from vxpu.export import export_artifact

base = sys.argv[1]
model_dir = tiny_deepseek_v41.write(os.path.join(base, "model"), drafter=True)
art_dir = os.path.join(base, "artifact")
export_artifact(model_dir, art_dir, draft=True)
`
	cmd := exec.Command(pythonExe, "-c", script, td)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("python export failed: %v\nOutput: %s", err, string(out))
	}

	manifestBytes, err := os.ReadFile(filepath.Join(td, "artifact", "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest.json: %v", err)
	}
	bindingBytes, err := os.ReadFile(filepath.Join(td, "artifact", "binding.json"))
	if err != nil {
		t.Fatalf("read binding.json: %v", err)
	}

	artifact, err := ReadArtifact(manifestBytes, bindingBytes)
	if err != nil {
		t.Fatalf("ReadArtifact failed: %v", err)
	}

	if artifact.StepTokens != 4 {
		t.Errorf("got step_tokens %d, want 4", artifact.StepTokens)
	}
	if artifact.DraftBlock != 3 {
		t.Errorf("got draft block %d, want 3", artifact.DraftBlock)
	}
}
