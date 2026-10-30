# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GPU's end of a message ring (csrc/ring.h; design.md, "mailboxes as
message rings").

A ring lives in pinned host memory, laid out by the native code, and the
CPU's end (ring.h) produces and consumes with release stores and acquire
loads. The GPU's end runs inside a captured CUDA graph, which replays
constant values, so it cannot pass per-message sequence numbers or slot
addresses to cuStreamWriteValue or cudaMemcpyAsync. Each direction is
three small kernels instead, with its sequence counter in device memory:
one program waits for the slot and writes its address to device memory,
many programs copy the payload with 8-byte loads, and one program
publishes (a system-scope fence, then release stores). The waits poll
with ld.acquire.sys and the publishes use st.release.sys: plain loads and
posted writes over PCIe. Atomic operations on host memory would be PCIe
atomics, which hang on machines without them.
"""

import torch

from . import kernels

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton: no GPU end
    triton = None


class PinnedRing:
    """A ring in pinned host memory that a GPU and the CPU share: `slots`
    slots of `slot_bytes` payload. The CPU's end is the native code (ring_echo
    and the step host); put() and get() are the GPU's."""

    def __init__(self, slots, slot_bytes, device):
        k = kernels.load()
        layout = k.ring_layout()
        self.slots, self.slot_bytes = slots, slot_bytes
        self.region = torch.zeros(k.ring_bytes(slots, slot_bytes), dtype=torch.uint8).pin_memory()
        k.ring_init(self.region, slots, slot_bytes)
        from .stepgraph import _Driver
        self.address = _Driver().device_address(self.region)  # the region as the GPU addresses it
        self.control, self.head, self.tail = layout["control"], layout["head"], layout["tail"]
        self.seq, self.bytes_field, self.header = layout["seq"], layout["bytes"], layout["slot_header"]
        self.stride = self.header + (slot_bytes + 63) // 64 * 64
        self.device = torch.device(device)
        # device-side state: the next message's sequence number, and the current slot's payload address
        self.put_count = torch.zeros(1, dtype=torch.int64, device=device)
        self.get_count = torch.zeros(1, dtype=torch.int64, device=device)
        self.put_slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.get_slot = torch.zeros(1, dtype=torch.int64, device=device)

    def put(self, src):
        """Send src's bytes (a contiguous CUDA tensor) as the next message,
        waiting on the GPU while the ring is full. Capturable."""
        data = src.contiguous().view(torch.uint8).view(-1)
        n = data.numel()
        if n > self.slot_bytes:
            raise ValueError(f"a message of {n} bytes does not fit a ring slot of {self.slot_bytes}")
        with torch.cuda.device(self.device):
            _wait_space[(1,)](self.address, self.put_count, self.put_slot, self.slots, self.stride,
                              CTRL=self.control, TAIL=self.tail, SH=self.header)
            _copy_to_slot[(max(1, triton.cdiv(n, _CHUNK)),)](self.put_slot, data, n, CHUNK=_CHUNK)
            _publish[(1,)](self.address, self.put_count, n, self.slots, self.stride, CTRL=self.control,
                           HEAD=self.head, SEQ=self.seq, BYTES=self.bytes_field)

    def get(self, dst):
        """Receive the next message into dst (a contiguous CUDA tensor of the
        message's size), waiting on the GPU until it arrives. Capturable."""
        data = dst.view(torch.uint8).view(-1)
        n = data.numel()
        with torch.cuda.device(self.device):
            _wait_message[(1,)](self.address, self.get_count, self.get_slot, self.slots, self.stride,
                                CTRL=self.control, SEQ=self.seq, SH=self.header)
            _copy_from_slot[(max(1, triton.cdiv(n, _CHUNK)),)](self.get_slot, data, n, CHUNK=_CHUNK)
            _release[(1,)](self.address, self.get_count, TAIL=self.tail)


_CHUNK = 8192  # bytes a copy program moves

if triton is not None:
    @triton.jit
    def _ld_acquire_sys(addr):
        return tl.inline_asm_elementwise("ld.acquire.sys.global.u64 $0, [$1];", "=l,l", [addr], dtype=tl.int64,
                                         is_pure=False, pack=1)

    @triton.jit
    def _st_release_sys(addr, v):
        return tl.inline_asm_elementwise("st.release.sys.global.u64 [$1], $2; mov.u32 $0, 0;", "=r,l,l", [addr, v],
                                         dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _fence_sys(dep):
        """Every write this kernel's stream made before (earlier kernels' copies included) is visible to the
        CPU before any later write of this thread. (dep: any value, as the inline assembly needs an input.)"""
        return tl.inline_asm_elementwise("fence.sc.sys; mov.u32 $0, 0;", "=r,l", [dep], dtype=tl.int32,
                                         is_pure=False, pack=1)

    @triton.jit
    def _wait_space(ring, count_ptr, slot_ptr, slots, stride, CTRL: tl.constexpr, TAIL: tl.constexpr,
                    SH: tl.constexpr):
        n = tl.load(count_ptr)
        while n - _ld_acquire_sys(ring + TAIL) >= slots:  # the ring is full
            pass
        tl.store(slot_ptr, ring + CTRL + (n % slots) * stride + SH)

    @triton.jit
    def _wait_message(ring, count_ptr, slot_ptr, slots, stride, CTRL: tl.constexpr, SEQ: tl.constexpr,
                      SH: tl.constexpr):
        n = tl.load(count_ptr)
        slot = ring + CTRL + (n % slots) * stride
        while _ld_acquire_sys(slot + SEQ) != n + 1:  # not arrived yet
            pass
        tl.store(slot_ptr, slot + SH)

    @triton.jit
    def _copy_to_slot(slot_ptr, src_ptr, nbytes, CHUNK: tl.constexpr):
        dst = tl.load(slot_ptr).to(tl.pointer_type(tl.uint8))
        _copy(src_ptr, dst, nbytes, CHUNK)

    @triton.jit
    def _copy_from_slot(slot_ptr, dst_ptr, nbytes, CHUNK: tl.constexpr):
        src = tl.load(slot_ptr).to(tl.pointer_type(tl.uint8))
        _copy(src, dst_ptr, nbytes, CHUNK)

    @triton.jit
    def _copy(src, dst, nbytes, CHUNK: tl.constexpr):
        """This program's CHUNK bytes: 8-byte words, then the bytes past the last whole word."""
        base = tl.program_id(0) * CHUNK
        w = base // 8 + tl.arange(0, CHUNK // 8)
        words = nbytes // 8
        v = tl.load(src.to(tl.pointer_type(tl.int64)) + w, mask=w < words, volatile=True)
        tl.store(dst.to(tl.pointer_type(tl.int64)) + w, v, mask=w < words)
        if base + CHUNK >= nbytes:  # the last program: the tail
            t = words * 8 + tl.arange(0, 8)
            tl.store(dst + t, tl.load(src + t, mask=t < nbytes, volatile=True), mask=t < nbytes)

    @triton.jit
    def _publish(ring, count_ptr, nbytes, slots, stride, CTRL: tl.constexpr, HEAD: tl.constexpr,
                 SEQ: tl.constexpr, BYTES: tl.constexpr):
        n = tl.load(count_ptr)
        slot = ring + CTRL + (n % slots) * stride
        tl.store((slot + BYTES).to(tl.pointer_type(tl.int64)), nbytes.to(tl.int64))
        _fence_sys(n)
        _st_release_sys(slot + SEQ, n + 1)
        _st_release_sys(ring + HEAD, n + 1)
        tl.store(count_ptr, n + 1)

    @triton.jit
    def _release(ring, count_ptr, TAIL: tl.constexpr):
        n = tl.load(count_ptr)
        _fence_sys(n)
        _st_release_sys(ring + TAIL, n + 1)
        tl.store(count_ptr, n + 1)
