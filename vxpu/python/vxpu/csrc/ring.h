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

// A message ring between processes: a single-producer, single-consumer queue
// of fixed-size slots in shared memory. Every message records when it was
// written and read. When a peer dies, the survivor detects it and fails.
//
// On Linux, is_pid_alive checks /proc/<pid>/stat to detect whether a terminated
// process is a zombie ('Z') before it has been reaped by its parent. On macOS,
// an unreaped child still answers kill(pid, 0) as alive until reaped.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vxpu {

constexpr uint32_t kRingMagic = 0x31525856;  // "VXR1"

struct alignas(64) RingControl {
  uint32_t magic, slots;
  uint64_t slot_bytes;
  alignas(64) std::atomic<uint64_t> head;
  alignas(64) std::atomic<uint64_t> tail;
  alignas(64) std::atomic<int64_t> writer_pid;
  alignas(64) std::atomic<int64_t> reader_pid;
  alignas(64) std::atomic<uint32_t> closed;
};

struct alignas(64) SlotHeader {
  std::atomic<uint64_t> seq;
  uint64_t bytes;
  int64_t written_ns;
  int64_t read_ns;
};

inline int64_t ring_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

inline bool is_pid_alive(int64_t pid) {
  if (pid <= 0) return true;
  if (kill(static_cast<pid_t>(pid), 0) == -1) return errno != ESRCH;
#if defined(__linux__)
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/%lld/stat", static_cast<long long>(pid));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return errno != ENOENT;
  char buf[256];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (n > 0) {
    buf[n] = '\0';
    const char* rp = std::strrchr(buf, ')');
    if (rp && *(rp + 1) == ' ' && (*(rp + 2) == 'Z' || *(rp + 2) == 'X')) return false;
  }
#endif
  return true;
}

class Ring {
 public:
  static uint64_t stride(uint64_t slot_bytes) {
    if (slot_bytes == 0 || slot_bytes > (UINT64_MAX - sizeof(SlotHeader) - 63)) {
      throw std::invalid_argument("slot_bytes out of range");
    }
    return sizeof(SlotHeader) + ((slot_bytes + 63) / 64) * 64;
  }

  static uint64_t bytes_for(uint32_t slots, uint64_t slot_bytes) {
    if (slots == 0) throw std::invalid_argument("slots must be positive");
    const uint64_t s = stride(slot_bytes);
    if (s > (UINT64_MAX - sizeof(RingControl)) / slots) {
      throw std::invalid_argument("ring size overflows");
    }
    return sizeof(RingControl) + slots * s;
  }

  static Ring create(void* base, uint32_t slots, uint64_t slot_bytes, int64_t writer_pid = 0, int64_t reader_pid = 0) {
    const uint64_t needed = bytes_for(slots, slot_bytes);
    std::memset(base, 0, needed);
    auto* c = static_cast<RingControl*>(base);
    c->slots = slots;
    c->slot_bytes = slot_bytes;
    c->head.store(0, std::memory_order_relaxed);
    c->tail.store(0, std::memory_order_relaxed);
    c->writer_pid.store(writer_pid, std::memory_order_relaxed);
    c->reader_pid.store(reader_pid, std::memory_order_relaxed);
    c->closed.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    c->magic = kRingMagic;
    return Ring(base);
  }

  static Ring attach(void* base) {
    if (static_cast<RingControl*>(base)->magic != kRingMagic) throw std::runtime_error("not a ring (no magic)");
    return Ring(base);
  }

  uint32_t slots() const { return c_->slots; }
  uint64_t slot_bytes() const { return c_->slot_bytes; }
  void set_writer_pid(int64_t pid) { c_->writer_pid.store(pid, std::memory_order_release); }
  void set_reader_pid(int64_t pid) { c_->reader_pid.store(pid, std::memory_order_release); }
  int64_t writer_pid() const { return c_->writer_pid.load(std::memory_order_acquire); }
  int64_t reader_pid() const { return c_->reader_pid.load(std::memory_order_acquire); }
  void close() { c_->closed.store(1, std::memory_order_release); }
  bool is_closed() const { return c_->closed.load(std::memory_order_acquire) != 0; }

  void attach_writer(int64_t peer_pid = 0) {
    c_->writer_pid.store(getpid(), std::memory_order_release);
    if (peer_pid > 0 && c_->reader_pid.load(std::memory_order_relaxed) == 0) {
      c_->reader_pid.store(peer_pid, std::memory_order_release);
    }
  }

  void attach_reader(int64_t peer_pid = 0) {
    c_->reader_pid.store(getpid(), std::memory_order_release);
    if (peer_pid > 0 && c_->writer_pid.load(std::memory_order_relaxed) == 0) {
      c_->writer_pid.store(peer_pid, std::memory_order_release);
    }
  }

  uint8_t* reserve(int64_t timeout_ns = -1) {
    const uint64_t n = c_->head.load(std::memory_order_relaxed);
    if (c_->writer_pid.load(std::memory_order_relaxed) == 0) c_->writer_pid.store(getpid(), std::memory_order_release);
    uint64_t spins = 0;
    const int64_t until = timeout_ns < 0 ? -1 : ring_now_ns() + timeout_ns;
    while (n - c_->tail.load(std::memory_order_acquire) >= c_->slots) {
      ring_pause(spins);
      if ((spins & 4095) == 0) check_peer(c_->reader_pid.load(std::memory_order_relaxed), "reader", until);
    }
    return payload(n);
  }

  void publish(uint64_t bytes) {
    const uint64_t n = c_->head.load(std::memory_order_relaxed);
    SlotHeader* h = header(n);
    h->bytes = bytes;
    h->written_ns = ring_now_ns();
    h->seq.store(n + 1, std::memory_order_release);
    c_->head.store(n + 1, std::memory_order_release);
  }

  const uint8_t* next(uint64_t* bytes, int64_t* written_ns = nullptr, int64_t* read_ns = nullptr, int64_t timeout_ns = -1) {
    const uint64_t n = c_->tail.load(std::memory_order_relaxed);
    if (c_->reader_pid.load(std::memory_order_relaxed) == 0) c_->reader_pid.store(getpid(), std::memory_order_release);
    SlotHeader* h = header(n);
    uint64_t spins = 0;
    const int64_t until = timeout_ns < 0 ? -1 : ring_now_ns() + timeout_ns;
    while (h->seq.load(std::memory_order_acquire) != n + 1) {
      ring_pause(spins);
      if ((spins & 4095) == 0) check_peer(c_->writer_pid.load(std::memory_order_relaxed), "writer", until);
    }
    const int64_t r_ns = ring_now_ns();
    h->read_ns = r_ns;
    if (bytes) *bytes = h->bytes;
    if (written_ns) *written_ns = h->written_ns;
    if (read_ns) *read_ns = r_ns;
    return payload(n);
  }

  void release() { c_->tail.store(c_->tail.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

  SlotHeader* header(uint64_t n) const {
    return reinterpret_cast<SlotHeader*>(reinterpret_cast<uint8_t*>(c_ + 1) + (n % c_->slots) * stride(c_->slot_bytes));
  }
  uint8_t* payload(uint64_t n) const { return reinterpret_cast<uint8_t*>(header(n) + 1); }

 private:
  explicit Ring(void* base) : c_(static_cast<RingControl*>(base)) {}
  void check_peer(int64_t pid, const char* role, int64_t until) {
    if (c_->closed.load(std::memory_order_relaxed) != 0) throw std::runtime_error("ring: closed by peer");
    if (pid != 0) {
      if (!is_pid_alive(pid)) throw std::runtime_error(std::string("ring: ") + role + " process died");
    }
    if (until >= 0 && ring_now_ns() > until) throw std::runtime_error(std::string("ring: ") + role + " timed out");
  }
  RingControl* c_;
};

}  // namespace vxpu
