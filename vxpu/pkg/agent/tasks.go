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
	"errors"
	"sync"
)

// Task is a function that performs work under a context and reports any error.
type Task func(ctx context.Context) error

// TaskOption configures the execution of tasks in RunTasks.
type TaskOption func(*taskConfig)

type taskConfig struct {
	// continueOnError indicates whether execution continues when a task fails.
	continueOnError bool
}

// ContinueOnError returns an option configuring RunTasks to run all tasks to
// completion even when some tasks fail, rather than stopping on the first error.
// Any errors encountered are combined and returned.
func ContinueOnError() TaskOption {
	return func(c *taskConfig) {
		c.continueOnError = true
	}
}

// RunTasks executes tasks concurrently with bounded parallelism.
// It limits active tasks to at most parallelism. If parallelism is less than
// or equal to zero, a parallelism of 1 is used.
//
// By default, RunTasks returns early on the first error, cancelling remaining
// tasks and returning that error. Use ContinueOnError to run all tasks to
// completion and collect every error.
func RunTasks(ctx context.Context, tasks []Task, parallelism int, opts ...TaskOption) error {
	if len(tasks) == 0 {
		return nil
	}
	if parallelism <= 0 {
		parallelism = 1
	}
	if parallelism > len(tasks) {
		parallelism = len(tasks)
	}

	var cfg taskConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	stopOnFirstError := !cfg.continueOnError

	runCtx := ctx
	var cancel context.CancelFunc
	if stopOnFirstError {
		runCtx, cancel = context.WithCancel(ctx)
		defer cancel()
	}

	taskCh := make(chan Task)
	var (
		wg       sync.WaitGroup
		firstErr error
		errOnce  sync.Once
		errsMu   sync.Mutex
		errs     []error
	)

	recordErr := func(err error) {
		if err == nil {
			return
		}
		if stopOnFirstError {
			errOnce.Do(func() {
				firstErr = err
				cancel()
			})
		} else {
			errsMu.Lock()
			errs = append(errs, err)
			errsMu.Unlock()
		}
	}

	for i := 0; i < parallelism; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskCh {
				if stopOnFirstError && runCtx.Err() != nil {
					return
				}
				if err := task(runCtx); err != nil {
					recordErr(err)
					if stopOnFirstError {
						return
					}
				}
			}
		}()
	}

dispatcher:
	for _, task := range tasks {
		if stopOnFirstError && runCtx.Err() != nil {
			break
		}
		select {
		case <-runCtx.Done():
			break dispatcher
		case taskCh <- task:
		}
	}
	close(taskCh)
	wg.Wait()

	if stopOnFirstError && firstErr != nil {
		return firstErr
	}
	if !stopOnFirstError && len(errs) > 0 {
		return errors.Join(errs...)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}
