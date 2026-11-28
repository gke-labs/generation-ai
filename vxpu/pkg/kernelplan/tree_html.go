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
	"bytes"
	_ "embed"
	"fmt"
	"html/template"

	planpb "github.com/gke-labs/generation-ai/vxpu/pkg/api/plan/v1alpha1"
)

//go:embed treeui/tree.html
var treeHTML string

var treeTemplate = template.Must(template.New("tree").Funcs(template.FuncMap{
	"formatFloat": func(f float64, decimals int) string {
		return fmt.Sprintf(fmt.Sprintf("%%.%df", decimals), f)
	},
	"formatBytes": func(b int64) string {
		if b >= 1024*1024*1024 {
			return fmt.Sprintf("%.2f GB (%d bytes)", float64(b)/(1024*1024*1024), b)
		} else if b >= 1024*1024 {
			return fmt.Sprintf("%.1f MB (%d bytes)", float64(b)/(1024*1024), b)
		}
		return fmt.Sprintf("%d bytes", b)
	},
	"formatSegmentRun": func(r planpb.SegmentRun) string {
		switch r {
		case planpb.SegmentRun_SEGMENT_RUN_PLAIN_TORCH:
			return "plain torch"
		case planpb.SegmentRun_SEGMENT_RUN_JIT:
			return "jit"
		case planpb.SegmentRun_SEGMENT_RUN_PACKAGE:
			return "package"
		case planpb.SegmentRun_SEGMENT_RUN_CAPTURED:
			return "captured"
		default:
			return ""
		}
	},
	"formatGoal": func(g planpb.PlanGoal) string {
		return FormatPlanGoal(g)
	},
	"isPassThrough": func(segments []*planpb.Segment) bool {
		return IsPassThrough(segments)
	},
}).Parse(treeHTML))

// RenderTreeHTML renders tree as a self-contained HTML page showing the model's
// segments, call sites, and any attached execution plan or measurements.
func RenderTreeHTML(tree *planpb.ModelTree) ([]byte, error) {
	var buf bytes.Buffer
	if err := treeTemplate.Execute(&buf, tree); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
