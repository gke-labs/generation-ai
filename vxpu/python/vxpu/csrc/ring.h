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


// A mailbox as a message ring (design.md, "mailboxes as message rings"): a
// single-producer, single-consumer queue of fixed-size slots in one region
// of memory. The region can be pinned host memory (a GPU and the CPU),
// shared memory (two processes) or registered memory (two machines); the
// code is the same for all of them.
//
// Layout: a RingControl, then `slots` slots, each a SlotHeader followed by
// `slot_bytes` of payload. Sequence numbers count messages from 0. The
// producer fills slot n % slots, then publishes it by storing n + 1 into
// the slot's seq (release). The consumer waits for that value (acquire),
// reads the payload, and frees the slot by storing n + 1 into tail. The
// producer waits while head - tail == slots. Each header carries the time
// the message was written and read, so every crossing is timed.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vxpu {

constexpr uint32_t kRingMagic = 0x31525856;  // "VXR1"

struct alignas(64) RingControl {
  uint32_t magic;
  uint32_t slots;
  uint64_t slot_bytes;
  alignas(64) std::atomic<uint64_t> head;  // the next message the producer writes (written by the producer)
  alignas(64) std::atomic<uint64_t> tail;  // the next message the consumer reads (written by the consumer)
};

struct alignas(64) SlotHeader {
  std::atomic<uint64_t> seq;  // n + 1 once message n is in the slot
  uint64_t bytes;             // the payload's length
  int64_t written_ns;         // when the producer published it (steady clock)
  int64_t read_ns;            // when the consumer saw it
};

inline int64_t ring_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Spin briefly, then yield: a ring's peer answers in microseconds when it is
// busy polling, and a waiting thread must not take a core from it otherwise.
inline void ring_pause(uint64_t& spins) {
  if (++spins < 4096) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
  } else {
    std::this_thread::yield();
  }
}

class Ring {
 public:
  static uint64_t stride(uint64_t slot_bytes) { return sizeof(SlotHeader) + ((slot_bytes + 63) / 64) * 64; }
  static uint64_t bytes_for(uint32_t slots, uint64_t slot_bytes) {
    return sizeof(RingControl) + slots * stride(slot_bytes);
  }

  // Lay out an empty ring in base (bytes_for(slots, slot_bytes) bytes).
  static Ring create(void* base, uint32_t slots, uint64_t slot_bytes) {
    if (slots == 0 || slot_bytes == 0) throw std::invalid_argument("a ring needs slots and payload room");
    std::memset(base, 0, bytes_for(slots, slot_bytes));
    auto* c = static_cast<RingControl*>(base);
    c->slots = slots;
    c->slot_bytes = slot_bytes;
    c->head.store(0, std::memory_order_relaxed);
    c->tail.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    c->magic = kRingMagic;
    return Ring(base);
  }

  // A ring another party created in base.
  static Ring attach(void* base) {
    if (static_cast<RingControl*>(base)->magic != kRingMagic) throw std::runtime_error("not a ring (no magic)");
    return Ring(base);
  }

  uint32_t slots() const { return c_->slots; }
  uint64_t slot_bytes() const { return c_->slot_bytes; }

  // Producer: the payload of the next free slot, waiting while the ring is
  // full (nullptr after timeout_ns; a negative timeout waits for ever).
  uint8_t* reserve(int64_t timeout_ns = -1) {
    const uint64_t n = c_->head.load(std::memory_order_relaxed);
    uint64_t spins = 0;
    const int64_t until = timeout_ns < 0 ? -1 : ring_now_ns() + timeout_ns;
    while (n - c_->tail.load(std::memory_order_acquire) >= c_->slots) {
      if (until >= 0 && ring_now_ns() > until) return nullptr;
      ring_pause(spins);
    }
    return payload(n);
  }

  // Producer: publish the slot reserve() returned, with `bytes` of payload.
  void publish(uint64_t bytes) {
    const uint64_t n = c_->head.load(std::memory_order_relaxed);
    SlotHeader* h = header(n);
    h->bytes = bytes;
    h->written_ns = ring_now_ns();
    h->seq.store(n + 1, std::memory_order_release);
    c_->head.store(n + 1, std::memory_order_release);
  }

  // Consumer: the next message's payload and length, waiting for it
  // (nullptr after timeout_ns).
  const uint8_t* next(uint64_t* bytes, int64_t timeout_ns = -1) {
    const uint64_t n = c_->tail.load(std::memory_order_relaxed);
    SlotHeader* h = header(n);
    uint64_t spins = 0;
    const int64_t until = timeout_ns < 0 ? -1 : ring_now_ns() + timeout_ns;
    while (h->seq.load(std::memory_order_acquire) != n + 1) {
      if (until >= 0 && ring_now_ns() > until) return nullptr;
      ring_pause(spins);
    }
    h->read_ns = ring_now_ns();
    *bytes = h->bytes;
    return payload(n);
  }

  // Consumer: done with the message next() returned; its slot is free.
  void release() { c_->tail.store(c_->tail.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

  // Message n's slot: for a GPU publisher, which copies the payload in and
  // then writes n + 1 to seq itself (a system-scope store, after the copy).
  SlotHeader* header(uint64_t n) const {
    return reinterpret_cast<SlotHeader*>(reinterpret_cast<uint8_t*>(c_ + 1) + (n % c_->slots) * stride(c_->slot_bytes));
  }
  uint8_t* payload(uint64_t n) const { return reinterpret_cast<uint8_t*>(header(n) + 1); }

 private:
  explicit Ring(void* base) : c_(static_cast<RingControl*>(base)) {}
  RingControl* c_;
};

}  // namespace vxpu
