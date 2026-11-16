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

"""Tests for native message rings between processes (csrc/ring.h, rings.py)."""

import multiprocessing as mp
import os
import platform
import signal
import stat
import tempfile
import time
import unittest
import torch

from . import kernels
from .rings import Ring


def _get_cpu_name() -> str:
    """Read the CPU model name on Linux, or fall back to platform info."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, UnicodeDecodeError):
        pass
    return platform.processor() or platform.machine() or "unknown cpu"


def _shm_dir() -> str | None:
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


def _reader_order_proc(path: str, count: int, queue: mp.Queue) -> None:
    ring = Ring.attach(path, role="reader")
    results: list[bytes] = []
    w_times: list[int] = []
    r_times: list[int] = []
    for _ in range(count):
        payload, w_ns, r_ns = ring.read()
        results.append(payload)
        w_times.append(w_ns)
        r_times.append(r_ns)
    queue.put((results, w_times, r_times))


def _slow_reader_proc(path: str, count: int, queue: mp.Queue) -> None:
    ring = Ring.attach(path, role="reader")
    results: list[bytes] = []
    for _ in range(count):
        time.sleep(0.002)
        payload, _, _ = ring.read()
        results.append(payload)
    queue.put(results)


def _echo_proc(in_path: str, out_path: str, messages: int) -> None:
    ring_in = Ring.attach(in_path, role="reader")
    ring_out = Ring.attach(out_path, role="writer")
    ring_in.echo(ring_out, messages)


def _reader_first_proc(path: str, result_q: mp.Queue) -> None:
    ring = Ring.attach(path, role="reader")
    payload, _, _ = ring.read()
    result_q.put(payload)


def _reader_die_target(path: str, ready_q: mp.Queue) -> None:
    Ring.attach(path, role="reader")
    ready_q.put(os.getpid())
    while True:
        time.sleep(1)


def _writer_die_target(path: str, ready_q: mp.Queue) -> None:
    ring = Ring.attach(path, role="writer")
    ring.write(b"first")
    ready_q.put(os.getpid())
    while True:
        time.sleep(1)


class RingTest(unittest.TestCase):

    def setUp(self) -> None:
        if kernels.load() is None:
            self.skipTest("native kernels unavailable")

    def test_file_permissions_and_header_reading(self) -> None:
        """Creating a ring sets 0600 mode; attach reads dimensions from header."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "perm_test.shm")
            ring = Ring.create(path, 4, 128)
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600)

            attached = Ring.attach(path)
            self.assertEqual(attached.slots, 4)
            self.assertEqual(attached.slot_bytes, 128)
            attached.close(unlink=True)
            self.assertFalse(os.path.exists(path))

    def test_size_and_bounds_validation(self) -> None:
        """Bounds checks reject negative, zero, overflowing sizes, or truncated files."""
        k = kernels.load()
        assert k is not None

        # Negative or zero slots/slot_bytes rejected
        with self.assertRaises((ValueError, RuntimeError)):
            Ring.bytes_for(0, 64)
        with self.assertRaises((ValueError, RuntimeError)):
            Ring.bytes_for(-1, 64)
        with self.assertRaises((ValueError, RuntimeError)):
            Ring.bytes_for(4, 0)
        with self.assertRaises((ValueError, RuntimeError)):
            Ring.bytes_for(4, -1)

        # Truncated tensor smaller than RingControl or ring_bytes rejected
        small = torch.zeros(16, dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            k.ring_init(small, 4, 64)

        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "bounds.shm")
            Ring.create(path, 4, 64)
            with open(path, "r+b") as f:
                f.truncate(100)
            with self.assertRaises(RuntimeError):
                Ring.attach(path)

    def test_echo_destination_bounds(self) -> None:
        """Echo refuses messages that exceed the destination slot bytes."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            in_path = os.path.join(tmpdir, "in.shm")
            out_path = os.path.join(tmpdir, "out.shm")
            ring_in = Ring.create(in_path, 2, 128)
            ring_out = Ring.create(out_path, 2, 32)

            ring_in.write(b"a" * 64)
            with self.assertRaises(RuntimeError) as cm:
                ring_in.echo(ring_out, 1)
            self.assertIn("message exceeds destination slot_bytes", str(cm.exception))

    def test_two_processes_order_and_wrapping(self) -> None:
        """Every message arrives intact and in order when the ring wraps."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "ring.shm")
            slots = 4
            slot_bytes = 64
            count = 500

            ring = Ring.create(path, slots, slot_bytes, role="writer")

            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            proc = ctx.Process(target=_reader_order_proc, args=(path, count, queue))
            proc.start()

            for i in range(count):
                msg = f"order-msg-{i:06d}".encode()
                ring.write(msg)

            results, w_times, r_times = queue.get(timeout=10)
            proc.join(timeout=5)

            self.assertEqual(len(results), count)
            for i in range(count):
                expected = f"order-msg-{i:06d}".encode()
                self.assertEqual(results[i], expected)
                self.assertGreaterEqual(r_times[i], w_times[i])

    def test_reader_slower_than_writer(self) -> None:
        """When the reader is slower than the writer, messages arrive intact and in order."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "slow_ring.shm")
            slots = 2
            slot_bytes = 64
            count = 30

            ring = Ring.create(path, slots, slot_bytes, role="writer")

            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            proc = ctx.Process(target=_slow_reader_proc, args=(path, count, queue))
            proc.start()

            for i in range(count):
                msg = f"slow-msg-{i:04d}".encode()
                ring.write(msg)

            results = queue.get(timeout=10)
            proc.join(timeout=5)

            self.assertEqual(len(results), count)
            for i in range(count):
                self.assertEqual(results[i], f"slow-msg-{i:04d}".encode())

    def test_reader_starts_before_writer(self) -> None:
        """A reader that starts before its writer waits without error until writer arrives."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "reader_first.shm")
            Ring.create(path, 4, 64)

            ctx = mp.get_context("spawn")
            result_q = ctx.Queue()
            proc = ctx.Process(target=_reader_first_proc, args=(path, result_q))
            proc.start()

            # Writer arrives later with no synchronization or ready queue.
            time.sleep(0.05)
            writer = Ring.attach(path, role="writer")
            writer.write(b"hello from later writer")

            received = result_q.get(timeout=5)
            proc.join(timeout=5)
            self.assertEqual(received, b"hello from later writer")

    def test_round_trip_measurement(self) -> None:
        """Measure round trip between two processes and print the timing."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            in_path = os.path.join(tmpdir, "rtt_in.shm")
            out_path = os.path.join(tmpdir, "rtt_out.shm")
            slots = 4
            slot_bytes = 64
            messages = 2000

            out_ring = Ring.create(in_path, slots, slot_bytes, role="writer")
            in_ring = Ring.create(out_path, slots, slot_bytes, role="reader")

            ctx = mp.get_context("spawn")
            proc = ctx.Process(target=_echo_proc, args=(in_path, out_path, messages))
            proc.start()

            rtts = out_ring.ping(in_ring, messages=messages, bytes_count=slot_bytes)
            proc.join(timeout=5)

            self.assertEqual(len(rtts), messages)
            sorted_rtts = sorted(rtts)
            median_us = sorted_rtts[len(sorted_rtts) // 2] / 1e3
            p99_us = sorted_rtts[int(len(sorted_rtts) * 0.99)] / 1e3
            cpu = _get_cpu_name()

            print(f"\n[ring_test] round trip on {cpu}: median {median_us:.2f} us, p99 {p99_us:.2f} us ({messages} messages of {slot_bytes} bytes)")

    def test_reader_timeout_on_empty_ring(self) -> None:
        """A reader waiting on an empty ring times out when no writer publishes within timeout."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "orphan.shm")
            Ring.create(path, 2, 64)
            reader = Ring.attach(path, role="reader")
            with self.assertRaises(RuntimeError) as cm:
                reader.read(timeout_ns=50_000_000)  # 50 ms
            self.assertIn("writer timed out", str(cm.exception))

    def test_reader_dies_writer_fails(self) -> None:
        """A writer waiting on a dead reader fails promptly with a clear message."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "dead_reader.shm")
            ring = Ring.create(path, 2, 64, role="writer")

            ctx = mp.get_context("spawn")
            ready_q = ctx.Queue()
            proc = ctx.Process(target=_reader_die_target, args=(path, ready_q))
            proc.start()

            reader_pid = ready_q.get(timeout=5)

            # Fill the 2-slot ring so the next write must wait for space.
            ring.write(b"slot-0")
            ring.write(b"slot-1")

            # Kill the reader process.
            os.kill(reader_pid, signal.SIGKILL)
            time.sleep(0.01)

            # Writer must fail promptly and not spin forever.
            with self.assertRaises(RuntimeError) as cm:
                ring.write(b"slot-overflow")
            self.assertIn("reader process died", str(cm.exception))
            proc.join(timeout=5)

    def test_writer_dies_reader_fails(self) -> None:
        """A reader waiting on a dead writer fails promptly with a clear message."""
        with tempfile.TemporaryDirectory(dir=_shm_dir()) as tmpdir:
            path = os.path.join(tmpdir, "dead_writer.shm")
            ring = Ring.create(path, 2, 64, role="reader")

            ctx = mp.get_context("spawn")
            ready_q = ctx.Queue()
            proc = ctx.Process(target=_writer_die_target, args=(path, ready_q))
            proc.start()

            writer_pid = ready_q.get(timeout=5)

            # Read the one message that was published.
            payload, _, _ = ring.read()
            self.assertEqual(payload, b"first")

            # Kill the writer process while the ring is empty.
            os.kill(writer_pid, signal.SIGKILL)
            time.sleep(0.01)

            # Reader must fail promptly and not spin forever.
            with self.assertRaises(RuntimeError) as cm:
                ring.read()
            self.assertIn("writer process died", str(cm.exception))
            proc.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
