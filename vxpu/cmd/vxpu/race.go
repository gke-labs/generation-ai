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
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/capacity"
)

// cmdRace runs the GPU capacity racer across multiple zones.
func cmdRace(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("race", flag.ContinueOnError)
	machineType := flags.String("machine-type", "", "GCE machine type (e.g. g4-standard-192)")
	zonesArg := flags.String("zones", "", "comma-separated list of zones to race in")
	project := flags.String("project", "", "GCP project ID (defaults to PROJECT or GOOGLE_CLOUD_PROJECT env)")
	accelerator := flags.String("accelerator", "", "accelerator configuration (type=TYPE,count=COUNT)")
	serviceAccount := flags.String("service-account", "", "service account email (defaults to project default)")
	timeBudget := flags.Duration("time-budget", 2*time.Hour, "maximum time to wait for capacity")
	maxRunDuration := flags.Duration("max-run-duration", 6*time.Hour, "maximum duration the VM is allowed to run")

	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *machineType == "" || *zonesArg == "" {
		return errors.New("vxpu race requires --machine-type and --zones")
	}
	zones := parseZones(*zonesArg)
	if len(zones) == 0 {
		return errors.New("at least one valid zone must be provided in --zones")
	}
	proj := *project
	if proj == "" {
		if proj = os.Getenv("PROJECT"); proj == "" {
			proj = os.Getenv("GOOGLE_CLOUD_PROJECT")
		}
	}
	if proj == "" {
		return errors.New("GCP project not specified: use --project or set PROJECT environment variable")
	}

	var accelerators []capacity.AcceleratorSpec
	if *accelerator != "" {
		acc, err := parseAccelerator(*accelerator)
		if err != nil {
			return err
		}
		accelerators = append(accelerators, acc)
	}

	backend, err := capacity.NewGCEBackend(ctx, proj)
	if err != nil {
		return fmt.Errorf("connecting to GCE API: %w", err)
	}
	defer func() {
		if cerr := backend.Close(); cerr != nil {
			log.Error(cerr, "closing GCE backend", "project", proj)
		}
	}()

	sigCtx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	winner, raceErr := capacity.Race(sigCtx, backend, capacity.RaceOptions{
		Spec: capacity.MachineSpec{
			MachineType:          *machineType,
			Accelerators:         accelerators,
			ServiceAccount:       *serviceAccount,
			MaxRunDuration:       *maxRunDuration,
			RequestValidDuration: *timeBudget,
		},
		Zones:   zones,
		Timeout: *timeBudget,
	})
	if winner == nil {
		return fmt.Errorf("capacity race failed: %w", raceErr)
	}
	if raceErr != nil {
		log.Error(raceErr, "cleaning up non-winning requests", "winner", winner.Name, "zone", winner.Zone)
	}

	fmt.Printf("%s %s %s\n", winner.Name, winner.Zone, winner.Wait.Round(time.Second))

	var cmdErr error
	if cmdArgs := flags.Args(); len(cmdArgs) > 0 {
		cmd := exec.CommandContext(sigCtx, cmdArgs[0], cmdArgs[1:]...)
		cmd.Env = append(os.Environ(), "NAME="+winner.Name, "ZONE="+winner.Zone, "PROJECT="+proj)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		cmdErr = cmd.Run()
	}
	if cmdErr != nil {
		return cmdErr
	}
	return raceErr
}

func parseZones(s string) []string {
	var zones []string
	seen := make(map[string]bool)
	for _, z := range strings.Split(s, ",") {
		if z = strings.TrimSpace(z); z != "" && !seen[z] {
			seen[z] = true
			zones = append(zones, z)
		}
	}
	return zones
}

func parseAccelerator(s string) (capacity.AcceleratorSpec, error) {
	if s == "" {
		return capacity.AcceleratorSpec{}, nil
	}
	var accType string
	count := 1
	for _, p := range strings.Split(s, ",") {
		kv := strings.Split(p, "=")
		if len(kv) != 2 {
			return capacity.AcceleratorSpec{}, fmt.Errorf("invalid accelerator item %q: must be key=value", p)
		}
		switch key, val := strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]); key {
		case "type":
			accType = val
		case "count":
			c, err := strconv.Atoi(val)
			if err != nil || c <= 0 || c > math.MaxInt32 {
				return capacity.AcceleratorSpec{}, fmt.Errorf("invalid accelerator count %q: must be between 1 and %d", val, math.MaxInt32)
			}
			count = c
		default:
			return capacity.AcceleratorSpec{}, fmt.Errorf("unknown accelerator key %q", key)
		}
	}
	if accType == "" {
		return capacity.AcceleratorSpec{}, fmt.Errorf("accelerator spec %q must include type", s)
	}
	return capacity.AcceleratorSpec{Type: accType, Count: count}, nil
}
