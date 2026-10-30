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


// Python's view of the message rings (ring.h): laying a ring out in a
// tensor's memory, and measuring them. ring_echo answers every message on
// one ring with the same payload on another; ring_ping sends messages and
// times each round trip. With the two in different threads or processes
// (the regions in shared memory), the round trip is the cost of a mailbox
// crossing with no work.

#include <torch/extension.h>

#include <cstddef>
#include <vector>

#include "ring.h"

namespace {

vxpu::Ring view(const torch::Tensor& region) {
  TORCH_CHECK(region.dtype() == torch::kUInt8 && region.is_contiguous() && region.device().is_cpu(),
              "a ring lives in a contiguous uint8 CPU tensor (pinned or shared memory)");
  return vxpu::Ring::attach(region.data_ptr());
}

void ring_init(torch::Tensor region, int64_t slots, int64_t slot_bytes) {
  TORCH_CHECK(region.dtype() == torch::kUInt8 && region.is_contiguous() && region.device().is_cpu(),
              "a ring lives in a contiguous uint8 CPU tensor (pinned or shared memory)");
  TORCH_CHECK(static_cast<uint64_t>(region.numel()) >= vxpu::Ring::bytes_for(slots, slot_bytes),
              "the region is smaller than ring_bytes(slots, slot_bytes)");
  vxpu::Ring::create(region.data_ptr(), static_cast<uint32_t>(slots), static_cast<uint64_t>(slot_bytes));
}

// Answer `messages` messages from `in` with the same bytes on `out`.
void ring_echo(torch::Tensor in, torch::Tensor out, int64_t messages) {
  vxpu::Ring a = view(in), b = view(out);
  pybind11::gil_scoped_release nogil;
  for (int64_t m = 0; m < messages; ++m) {
    uint64_t bytes = 0;
    const uint8_t* src = a.next(&bytes);
    uint8_t* dst = b.reserve();
    std::memcpy(dst, src, bytes);
    a.release();
    b.publish(bytes);
  }
}

// Send `messages` messages of `bytes` bytes on `out`, each waiting for its
// reply on `in`: the round trips, in ns.
std::vector<int64_t> ring_ping(torch::Tensor out, torch::Tensor in, int64_t messages, int64_t bytes) {
  vxpu::Ring b = view(out), a = view(in);
  TORCH_CHECK(static_cast<uint64_t>(bytes) <= b.slot_bytes(), "the message is larger than a slot");
  std::vector<int64_t> rtt(messages);
  pybind11::gil_scoped_release nogil;
  for (int64_t m = 0; m < messages; ++m) {
    const int64_t t0 = vxpu::ring_now_ns();
    uint8_t* dst = b.reserve();
    std::memset(dst, static_cast<int>(m & 0xFF), bytes);
    b.publish(bytes);
    uint64_t got = 0;
    const uint8_t* src = a.next(&got);
    TORCH_CHECK(got == static_cast<uint64_t>(bytes) && (bytes == 0 || src[0] == (m & 0xFF)), "a reply that is not the message");
    a.release();
    rtt[m] = vxpu::ring_now_ns() - t0;
  }
  return rtt;
}

}  // namespace

void register_rings(pybind11::module& m) {
  m.def("ring_bytes", [](int64_t slots, int64_t slot_bytes) { return vxpu::Ring::bytes_for(slots, slot_bytes); },
        "bytes a ring of `slots` slots of `slot_bytes` payload takes");
  m.def("ring_init", &ring_init, "lay out an empty ring in a uint8 tensor's memory");
  m.def("ring_layout", []() {
        // byte offsets, for a GPU producer or consumer that addresses the ring itself
        pybind11::dict d;
        d["control"] = sizeof(vxpu::RingControl);
        d["head"] = offsetof(vxpu::RingControl, head);
        d["tail"] = offsetof(vxpu::RingControl, tail);
        d["slot_header"] = sizeof(vxpu::SlotHeader);
        d["seq"] = offsetof(vxpu::SlotHeader, seq);
        d["bytes"] = offsetof(vxpu::SlotHeader, bytes);
        return d;
      }, "the ring's layout: the control block's size, head and tail offsets, the slot header's size and fields");
  m.def("ring_echo", &ring_echo, "answer messages from one ring with the same bytes on another (releases the GIL)");
  m.def("ring_ping", &ring_ping, "send messages on one ring and time each reply on another, in ns");
}
