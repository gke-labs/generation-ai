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
	"context"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

// DeterministicName generates a consistent instance name for a given zone.
func DeterministicName(baseName, zone string) string {
	if baseName == "" {
		baseName = "vxpu-race"
	}
	return fmt.Sprintf("%s-%s", strings.TrimSuffix(baseName, "-"), zone)
}

// racer executes the lifecycle of a flex-start request in one zone.
type racer struct {
	backend        Backend
	zone           string
	name           string
	spec           MachineSpec
	pollInterval   time.Duration
	cleanupTimeout time.Duration
}

// Race executes a capacity race across the configured zones using racing goroutines.
// Race returns the Winner together with the cleanup error when a loser's delete fails
// after a grant; the Winner is valid whenever it is non-nil.
func Race(ctx context.Context, backend Backend, opts RaceOptions) (*Winner, error) {
	if len(opts.Zones) == 0 {
		return nil, errors.New("capacity race requires at least one zone")
	}

	raceCtx, stopRace := context.WithCancel(ctx)
	defer stopRace()

	if opts.Timeout > 0 {
		var cancelTimeout context.CancelFunc
		raceCtx, cancelTimeout = context.WithTimeout(raceCtx, opts.Timeout)
		defer cancelTimeout()
	}

	pollInterval := opts.PollInterval
	if pollInterval <= 0 {
		pollInterval = 2 * time.Second
	}

	cleanupTimeout := opts.CleanupTimeout
	if cleanupTimeout <= 0 {
		cleanupTimeout = 2 * time.Minute
	}

	racers := make([]*racer, len(opts.Zones))
	for i, zone := range opts.Zones {
		spec := opts.Spec
		name := DeterministicName(spec.Name, zone)
		spec.Name = name
		racers[i] = &racer{
			backend:        backend,
			zone:           zone,
			name:           name,
			spec:           spec,
			pollInterval:   pollInterval,
			cleanupTimeout: cleanupTimeout,
		}
	}

	winCh := make(chan *Winner)
	errs := make([]error, len(racers))
	var wg sync.WaitGroup
	for i, r := range racers {
		wg.Add(1)
		go func(index int, r *racer) {
			defer wg.Done()
			errs[index] = r.run(raceCtx, winCh)
		}(i, r)
	}

	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()

	var winner *Winner
	var raceErr error

	select {
	case winner = <-winCh:
		stopRace()
	case <-doneCh:
		raceErr = errors.New("no active flex-start requests could be started in the given zones")
	case <-raceCtx.Done():
		raceErr = raceCtx.Err()
	}

	<-doneCh

	cancelErr := errors.Join(errs...)
	if raceErr != nil {
		return nil, errors.Join(raceErr, cancelErr)
	}
	return winner, cancelErr
}

// run executes the self-contained polling loop for one zone.
func (r *racer) run(ctx context.Context, winCh chan<- *Winner) error {
	log := klog.FromContext(ctx)
	var requestTime time.Time
	var lastState InstanceStatus
	var attempted bool

	cleanup := func() error {
		if !attempted {
			return nil
		}
		cleanupCtx, cleanupCancel := context.WithTimeout(context.WithoutCancel(ctx), r.cleanupTimeout)
		defer cleanupCancel()
		return deleteAndConfirm(cleanupCtx, r.backend, r.zone, r.name)
	}

	instance, err := r.backend.Describe(ctx, r.zone, r.name)
	if err == nil {
		if r.spec.MachineType != "" && instance.MachineType != "" && path.Base(r.spec.MachineType) != path.Base(instance.MachineType) {
			log.Info("Existing instance machine type does not match spec; skipping adoption",
				"zone", r.zone, "name", r.name, "want", path.Base(r.spec.MachineType), "got", path.Base(instance.MachineType))
			return nil
		}
		attempted = true
		requestTime = time.Now()
		if !instance.CreationTimestamp.IsZero() {
			requestTime = instance.CreationTimestamp
		}
		log.Info("Adopted existing flex-start request", "zone", r.zone, "name", r.name, "status", string(instance.Status))
	} else {
		if !errors.Is(err, ErrNotFound) {
			log.Error(err, "Failed to describe instance before creation", "zone", r.zone, "name", r.name)
		}
		if !r.create(ctx, &requestTime, &lastState, &attempted) {
			return nil
		}
	}

	ticker := time.NewTicker(r.pollInterval)
	defer ticker.Stop()

	for {
		instance, err := r.backend.Describe(ctx, r.zone, r.name)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				log.Info("Flex-start request not found; recreating", "zone", r.zone, "name", r.name)
				if !r.create(ctx, &requestTime, &lastState, &attempted) {
					return cleanup()
				}
			}
		} else {
			if instance.Status == StatusRunning {
				select {
				case winCh <- &Winner{Name: r.name, Zone: r.zone, Wait: time.Since(requestTime)}:
					log.Info("Flex-start request granted", "zone", r.zone, "name", r.name, "state", "granted", "waited", time.Since(requestTime))
					return nil
				case <-ctx.Done():
					return cleanup()
				}
			}
			if instance.Status == StatusTerminated && lastState != StatusTerminated {
				lastState = StatusTerminated
				log.Info("Flex-start request expired; deleting", "zone", r.zone, "name", r.name, "reason", instance.StatusMessage)
				if deleteErr := r.backend.Delete(ctx, r.zone, r.name); deleteErr != nil && !errors.Is(deleteErr, ErrNotFound) {
					log.Error(deleteErr, "Failed to delete expired instance", "zone", r.zone, "name", r.name)
				}
			} else if instance.Status == StatusPending && lastState != StatusPending {
				lastState = StatusPending
				log.Info("Flex-start request pending", "zone", r.zone, "name", r.name, "state", "pending", "waited", time.Since(requestTime).Round(time.Second))
			}
		}

		select {
		case <-ctx.Done():
			return cleanup()
		case <-ticker.C:
		}
	}
}

// create requests a new flex-start instance. Returns false if the zone is unsupported.
func (r *racer) create(ctx context.Context, requestTime *time.Time, lastState *InstanceStatus, attempted *bool) bool {
	log := klog.FromContext(ctx)
	*attempted = true
	*requestTime = time.Now()
	err := r.backend.Create(ctx, r.zone, r.spec)
	if err != nil {
		if errors.Is(err, ErrUnsupportedZone) {
			*attempted = false
			log.Info("Zone does not offer machine type; dropping zone from race", "zone", r.zone, "error", err.Error())
			return false
		}
		log.Error(err, "Failed to create flex-start request", "zone", r.zone, "name", r.name)
		return true
	}
	*lastState = StatusPending
	log.Info("Requested flex-start instance", "zone", r.zone, "name", r.name, "state", "requested")
	return true
}

// deleteAndConfirm deletes the instance and polls Describe until GCE reports it gone.
func deleteAndConfirm(ctx context.Context, backend Backend, zone, name string) error {
	log := klog.FromContext(ctx)
	var deleted bool
	for {
		if !deleted {
			err := backend.Delete(ctx, zone, name)
			if errors.Is(err, ErrNotFound) {
				log.Info("Flex-start request deleted", "zone", zone, "name", name, "state", "deleted")
				return nil
			}
			if err != nil {
				log.Error(err, "Delete failed, waiting for removal", "zone", zone, "name", name)
			} else {
				deleted = true
			}
		}

		if deleted {
			instance, err := backend.Describe(ctx, zone, name)
			if errors.Is(err, ErrNotFound) {
				log.Info("Flex-start request deleted", "zone", zone, "name", name, "state", "deleted")
				return nil
			}
			if err != nil {
				log.Error(err, "Describe failed while confirming deletion", "zone", zone, "name", name)
			} else if instance.Status == StatusRunning {
				deleted = false
			}
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("deleting instance %s in %s: %w", name, zone, ctx.Err())
		case <-time.After(300 * time.Millisecond):
		}
	}
}
