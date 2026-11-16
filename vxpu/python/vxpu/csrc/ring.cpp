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
// tensor's memory, writing and reading messages, and measuring them.
// ring_echo answers every message on one ring with the same payload on
// another; ring_ping sends messages and times each round trip. With the
// two in different processes (the regions in shared memory), the round
// trip is the cost of a mailbox crossing with no work.

#include <torch/extension.h>

#include <cstddef>
#include <string_view>
#include <vector>

#include "ring.h"

namespace {

vxpu::Ring view(const torch::Tensor& region) {
  TORCH_CHECK(region.dtype() == torch::kUInt8 && region.is_contiguous() && region.device().is_cpu(),
              "a ring lives in a contiguous uint8 CPU tensor (pinned or shared memory)");
  TORCH_CHECK(static_cast<uint64_t>(region.numel()) >= sizeof(vxpu::RingControl),
              "the region is smaller than RingControl");
  vxpu::Ring r = vxpu::Ring::attach(region.data_ptr());
  TORCH_CHECK(static_cast<uint64_t>(region.numel()) >= vxpu::Ring::bytes_for(r.slots(), r.slot_bytes()),
              "the region is smaller than ring_bytes(slots, slot_bytes)");
  return r;
}

int64_t ring_bytes(int64_t slots, int64_t slot_bytes) {
  TORCH_CHECK(slots > 0 && slots <= UINT32_MAX, "slots must be between 1 and UINT32_MAX");
  TORCH_CHECK(slot_bytes > 0 && slot_bytes <= static_cast<int64_t>(UINT64_MAX / 2),
              "slot_bytes must be positive");
  return static_cast<int64_t>(vxpu::Ring::bytes_for(static_cast<uint32_t>(slots), static_cast<uint64_t>(slot_bytes)));
}

void ring_init(torch::Tensor region, int64_t slots, int64_t slot_bytes) {
  TORCH_CHECK(region.dtype() == torch::kUInt8 && region.is_contiguous() && region.device().is_cpu(),
              "a ring lives in a contiguous uint8 CPU tensor (pinned or shared memory)");
  TORCH_CHECK(slots > 0 && slots <= UINT32_MAX, "slots must be between 1 and UINT32_MAX");
  TORCH_CHECK(slot_bytes > 0 && slot_bytes <= static_cast<int64_t>(UINT64_MAX / 2),
              "slot_bytes must be positive");
  const uint64_t needed = vxpu::Ring::bytes_for(static_cast<uint32_t>(slots), static_cast<uint64_t>(slot_bytes));
  TORCH_CHECK(static_cast<uint64_t>(region.numel()) >= needed,
              "the region is smaller than ring_bytes(slots, slot_bytes)");
  vxpu::Ring::create(region.data_ptr(), static_cast<uint32_t>(slots), static_cast<uint64_t>(slot_bytes));
}

void ring_echo(torch::Tensor in, torch::Tensor out, int64_t messages, int64_t timeout_ns) {
  vxpu::Ring a = view(in), b = view(out);
  pybind11::gil_scoped_release nogil;
  for (int64_t m = 0; m < messages; ++m) {
    uint64_t bytes = 0;
    const uint8_t* src = a.next(&bytes, nullptr, nullptr, timeout_ns);
    TORCH_CHECK(bytes <= b.slot_bytes(), "message exceeds destination slot_bytes");
    uint8_t* dst = b.reserve(timeout_ns);
    std::memcpy(dst, src, bytes);
    a.release();
    b.publish(bytes);
  }
}

std::vector<int64_t> ring_ping(torch::Tensor out, torch::Tensor in, int64_t messages, int64_t bytes,
                               int64_t timeout_ns) {
  vxpu::Ring b = view(out), a = view(in);
  TORCH_CHECK(static_cast<uint64_t>(bytes) <= b.slot_bytes(), "the message is larger than a slot");
  std::vector<int64_t> rtt(messages);
  pybind11::gil_scoped_release nogil;
  for (int64_t m = 0; m < messages; ++m) {
    const int64_t t0 = vxpu::ring_now_ns();
    uint8_t* dst = b.reserve(timeout_ns);
    std::memset(dst, static_cast<int>(m & 0xFF), bytes);
    b.publish(bytes);
    uint64_t got = 0;
    const uint8_t* src = a.next(&got, nullptr, nullptr, timeout_ns);
    TORCH_CHECK(got == static_cast<uint64_t>(bytes) && (bytes == 0 || src[0] == (m & 0xFF)),
                "a reply that is not the message");
    a.release();
    rtt[m] = vxpu::ring_now_ns() - t0;
  }
  return rtt;
}

void ring_write(torch::Tensor region, pybind11::bytes data, int64_t timeout_ns) {
  vxpu::Ring ring = view(region);
  std::string_view sv = data;
  TORCH_CHECK(static_cast<uint64_t>(sv.size()) <= ring.slot_bytes(), "message exceeds slot_bytes");
  uint8_t* dst = nullptr;
  {
    pybind11::gil_scoped_release nogil;
    dst = ring.reserve(timeout_ns);
  }
  std::memcpy(dst, sv.data(), sv.size());
  ring.publish(sv.size());
}

pybind11::tuple ring_read(torch::Tensor region, int64_t timeout_ns) {
  vxpu::Ring ring = view(region);
  uint64_t bytes = 0;
  int64_t written_ns = 0, read_ns = 0;
  const uint8_t* src = nullptr;
  {
    pybind11::gil_scoped_release nogil;
    src = ring.next(&bytes, &written_ns, &read_ns, timeout_ns);
  }
  pybind11::bytes payload(reinterpret_cast<const char*>(src), bytes);
  ring.release();
  return pybind11::make_tuple(payload, written_ns, read_ns);
}

}  // namespace

void register_rings(pybind11::module& m) {
  m.def("ring_bytes", &ring_bytes, "bytes a ring of `slots` slots of `slot_bytes` payload takes");
  m.def("ring_init", &ring_init, "lay out an empty ring in a uint8 tensor's memory");
  m.def("ring_slots", [](torch::Tensor region) { return view(region).slots(); }, "number of slots in ring");
  m.def("ring_slot_bytes", [](torch::Tensor region) { return view(region).slot_bytes(); }, "slot payload bytes in ring");
  m.def("ring_layout", []() {
    pybind11::dict d;
    d["control"] = sizeof(vxpu::RingControl);
    d["head"] = offsetof(vxpu::RingControl, head);
    d["tail"] = offsetof(vxpu::RingControl, tail);
    d["writer_pid"] = offsetof(vxpu::RingControl, writer_pid);
    d["reader_pid"] = offsetof(vxpu::RingControl, reader_pid);
    d["closed"] = offsetof(vxpu::RingControl, closed);
    d["slot_header"] = sizeof(vxpu::SlotHeader);
    d["seq"] = offsetof(vxpu::SlotHeader, seq);
    d["bytes"] = offsetof(vxpu::SlotHeader, bytes);
    d["written_ns"] = offsetof(vxpu::SlotHeader, written_ns);
    d["read_ns"] = offsetof(vxpu::SlotHeader, read_ns);
    return d;
  }, "the ring layout offsets and sizes");
  m.def("ring_echo", &ring_echo, pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("messages"),
        pybind11::arg("timeout_ns") = -1, "answer messages from one ring with the same bytes on another");
  m.def("ring_ping", &ring_ping, pybind11::arg("out"), pybind11::arg("in"), pybind11::arg("messages"),
        pybind11::arg("bytes"), pybind11::arg("timeout_ns") = -1,
        "send messages on one ring and time each reply on another");
  m.def("ring_write", &ring_write, pybind11::arg("region"), pybind11::arg("data"),
        pybind11::arg("timeout_ns") = -1, "write payload bytes into ring, waiting for free space");
  m.def("ring_read", &ring_read, pybind11::arg("region"), pybind11::arg("timeout_ns") = -1,
        "read next payload bytes from ring, returning (bytes, written_ns, read_ns)");
  m.def("ring_set_writer_pid", [](torch::Tensor region, int64_t pid) { view(region).set_writer_pid(pid); },
        "set writer PID for peer liveness checking");
  m.def("ring_set_reader_pid", [](torch::Tensor region, int64_t pid) { view(region).set_reader_pid(pid); },
        "set reader PID for peer liveness checking");
  m.def("ring_attach_writer", [](torch::Tensor region, int64_t peer_pid) { view(region).attach_writer(peer_pid); },
        pybind11::arg("region"), pybind11::arg("peer_pid") = 0, "attach as writer with optional peer PID");
  m.def("ring_attach_reader", [](torch::Tensor region, int64_t peer_pid) { view(region).attach_reader(peer_pid); },
        pybind11::arg("region"), pybind11::arg("peer_pid") = 0, "attach as reader with optional peer PID");
  m.def("ring_close", [](torch::Tensor region) { view(region).close(); }, "mark ring closed");
}
