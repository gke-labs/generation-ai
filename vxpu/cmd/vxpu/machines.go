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
	"encoding/json"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// cmdMachines shows the machine database (#394): every machine, or one
// in full as JSON, or a card's memory bandwidth in GB/s with --card.
func cmdMachines(args []string) error {
	if len(args) > 0 && args[0] == "--card" {
		if len(args) != 2 {
			return fmt.Errorf("usage: vxpu machines [--card CARD | NAME]")
		}
		bw, err := machines.LookupCard(args[1])
		if err != nil {
			return err
		}
		fmt.Printf("%.0f\n", bw)
		return nil
	}
	if len(args) > 1 {
		return fmt.Errorf("usage: vxpu machines [--card CARD | NAME]")
	}
	if len(args) == 1 {
		m, err := machines.Lookup(args[0])
		if err != nil {
			return err
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(m)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	if _, err := fmt.Fprintln(w, "NAME\tGPUS\tCPU THREADS\tRAM GB\tSPOT $/H"); err != nil {
		return err
	}
	for _, name := range machines.Names() {
		m, err := machines.Lookup(name)
		if err != nil {
			return err
		}
		gpus, spot := "-", "-"
		if m.GPU != nil {
			if bw := m.GPUMemoryBandwidthGBs(); bw > 0 {
				gpus = fmt.Sprintf("%dx %s (%.0f GB, %.0f GB/s)", m.GPUCount, m.GPU.Name, float64(m.GPU.MemoryBytes)/1e9, bw)
			} else {
				gpus = fmt.Sprintf("%dx %s (%.0f GB)", m.GPUCount, m.GPU.Name, float64(m.GPU.MemoryBytes)/1e9)
			}
		}
		if m.Price != nil {
			spot = fmt.Sprintf("%.2f", m.Price.SpotUSDPerHour)
		}
		if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%.0f\t%s\n", name, gpus, m.CPU.Threads, float64(m.RAMBytes)/1e9, spot); err != nil {
			return err
		}
	}
	return w.Flush()
}
