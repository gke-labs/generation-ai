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

package weightdisk

import (
	"context"
	"net/http"

	"cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/api/googleapi"
	"google.golang.org/protobuf/proto"
)

type fakeOp struct {
	err error
}

func (f *fakeOp) Wait(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return f.err
}

type fakeClient struct {
	calls []any
	polls int
	fail  map[string]error

	// Responses for specific operations:
	guestAttrs []*computepb.GuestAttributes
	disk       *computepb.Disk
	diskErr    error
	instance   *computepb.Instance
	instErr    error
}

func (f *fakeClient) InsertInstance(_ context.Context, req *computepb.InsertInstanceRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["InsertInstance"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["InsertInstance.Wait"]}, nil
}

func (f *fakeClient) GetGuestAttributes(_ context.Context, req *computepb.GetGuestAttributesInstanceRequest) (*computepb.GuestAttributes, error) {
	f.calls = append(f.calls, req)
	f.polls++
	if err := f.fail["GetGuestAttributes"]; err != nil {
		return nil, err
	}
	if len(f.guestAttrs) > 0 {
		idx := min(f.polls-1, len(f.guestAttrs)-1)
		return f.guestAttrs[idx], nil
	}
	if f.polls == 1 {
		return &computepb.GuestAttributes{
			QueryValue: &computepb.GuestAttributesValue{
				Items: []*computepb.GuestAttributesEntry{
					{Namespace: proto.String("vxpu"), Key: proto.String("status"), Value: proto.String("copying")},
				},
			},
		}, nil
	}
	return &computepb.GuestAttributes{
		QueryValue: &computepb.GuestAttributesValue{
			Items: []*computepb.GuestAttributesEntry{
				{Namespace: proto.String("vxpu"), Key: proto.String("status"), Value: proto.String("done")},
				{Namespace: proto.String("vxpu"), Key: proto.String("copy_seconds"), Value: proto.String("412")},
			},
		},
	}, nil
}

func (f *fakeClient) DeleteInstance(_ context.Context, req *computepb.DeleteInstanceRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["DeleteInstance"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["DeleteInstance.Wait"]}, nil
}

func (f *fakeClient) GetInstance(_ context.Context, req *computepb.GetInstanceRequest) (*computepb.Instance, error) {
	f.calls = append(f.calls, req)
	if f.instErr != nil {
		return nil, f.instErr
	}
	if f.instance != nil {
		return f.instance, nil
	}
	return &computepb.Instance{Name: proto.String(req.GetInstance())}, nil
}

func (f *fakeClient) AttachDisk(_ context.Context, req *computepb.AttachDiskInstanceRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["AttachDisk"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["AttachDisk.Wait"]}, nil
}

func (f *fakeClient) DetachDisk(_ context.Context, req *computepb.DetachDiskInstanceRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["DetachDisk"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["DetachDisk.Wait"]}, nil
}

func (f *fakeClient) InsertDisk(_ context.Context, req *computepb.InsertDiskRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["InsertDisk"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["InsertDisk.Wait"]}, nil
}

func (f *fakeClient) GetDisk(_ context.Context, req *computepb.GetDiskRequest) (*computepb.Disk, error) {
	f.calls = append(f.calls, req)
	if f.diskErr != nil {
		return nil, f.diskErr
	}
	if f.disk != nil {
		return f.disk, nil
	}
	return nil, &googleapi.Error{Code: http.StatusNotFound, Message: "not found"}
}

func (f *fakeClient) DeleteDisk(_ context.Context, req *computepb.DeleteDiskRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["DeleteDisk"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["DeleteDisk.Wait"]}, nil
}

func (f *fakeClient) InsertSnapshot(_ context.Context, req *computepb.InsertSnapshotRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["InsertSnapshot"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["InsertSnapshot.Wait"]}, nil
}

func (f *fakeClient) InsertImage(_ context.Context, req *computepb.InsertImageRequest) (Operation, error) {
	f.calls = append(f.calls, req)
	if err := f.fail["InsertImage"]; err != nil {
		return nil, err
	}
	return &fakeOp{err: f.fail["InsertImage.Wait"]}, nil
}

func (f *fakeClient) Close() error {
	return nil
}

func findCall[T any](f *fakeClient) (T, bool) {
	var zero T
	for _, c := range f.calls {
		if req, ok := c.(T); ok {
			return req, true
		}
	}
	return zero, false
}
