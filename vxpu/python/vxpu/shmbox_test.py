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

"""GPU-free tests and round-trip benchmarks for the shared-memory control plane (#539)."""

import multiprocessing as mp
import os
import struct
import time
import unittest

from .shmbox import (
    BROADCAST,
    COORDINATOR,
    DEFAULT_SLOT_SIZE,
    FLAG_JSON,
    FLAG_RAW,
    MAGIC,
    VERSION,
    AcceptanceControl,
    ControlPlane,
    RankStatus,
    StepControl,
)


def _worker_rank_barrier(r: int, num_ranks: int, rounds: int, shm_name: str) -> None:
    cp = ControlPlane.attach(name=shm_name, rank=r)
    try:
        for _ in range(rounds):
            cp.rank_barrier(timeout=5.0)
    finally:
        cp.close()


def _worker_all_barrier(p: int, num_ranks: int, rounds: int, shm_name: str) -> None:
    rank = COORDINATOR if p == num_ranks else p
    cp = ControlPlane.attach(name=shm_name, rank=rank)
    try:
        for _ in range(rounds):
            cp.barrier(timeout=5.0)
    finally:
        cp.close()


def _fake_rank_step_loop(r: int, num_ranks: int, n_steps: int, shm_name: str) -> None:
    cp = ControlPlane.attach(name=shm_name, rank=r)
    try:
        for step in range(1, n_steps + 1):
            # 1. Receive coordinator step broadcast
            step_ctrl = cp.read_step(expected_step_id=step, timeout=5.0)
            if step_ctrl.step_id != step:
                raise ValueError(f"Rank {r}: expected step {step}, got {step_ctrl.step_id}")

            # 2. Rank 0 publishes acceptance / draft info, peers consume
            if r == 0:
                cp.publish_acceptance(step_id=step, accepted_len=4, accepted_tokens=(100, 101, 102, 103))
            else:
                acc = cp.read_acceptance(expected_step_id=step, timeout=5.0)
                if acc.step_id != step:
                    raise ValueError(f"Rank {r}: expected acceptance step {step}, got {acc.step_id}")

            # 3. Step barrier across the 4 card ranks
            cp.rank_barrier(timeout=5.0)

            # 4. Report status to coordinator
            cp.post_status(step_id=step, status_code=0)
    finally:
        cp.close()


class ControlPlaneTest(unittest.TestCase):
    """Tests for shared-memory mailboxes, sequence numbers, and barriers."""

    def test_total_bytes_and_layout(self) -> None:
        total = ControlPlane.total_bytes(num_ranks=4, slot_size=DEFAULT_SLOT_SIZE)
        self.assertGreater(total, 0)
        self.assertEqual(total % 4096, 0)

    def test_create_and_header(self) -> None:
        cp = ControlPlane.create(num_ranks=4, slot_size=1024)
        try:
            magic = bytes(cp.buf[0:4])
            self.assertEqual(magic, MAGIC)
            version, n_ranks, slot_sz, _ = struct.unpack_from("<IIIQ", cp.buf, 4)
            self.assertEqual(version, VERSION)
            self.assertEqual(n_ranks, 4)
            self.assertEqual(slot_sz, 1024)
        finally:
            cp.close()

    def test_send_recv_raw(self) -> None:
        cp_coord = ControlPlane.create(num_ranks=2, slot_size=512, rank=COORDINATOR)
        cp_rank0 = ControlPlane.from_buffer(cp_coord.buf, rank=0, num_ranks=2, slot_size=512)
        try:
            seq1 = cp_coord.send(dst=0, payload=b"hello-rank-0", flags=FLAG_RAW)
            self.assertEqual(seq1, 1)

            msg = cp_rank0.recv(src=COORDINATOR, timeout=1.0)
            self.assertEqual(msg.seq, 1)
            self.assertEqual(msg.payload, b"hello-rank-0")
            self.assertEqual(msg.sender, COORDINATOR)
            self.assertEqual(msg.destination, 0)

            # Reply back to coordinator
            seq2 = cp_rank0.send(dst=COORDINATOR, payload=b"reply-to-coord", flags=FLAG_RAW)
            self.assertEqual(seq2, 1)

            reply = cp_coord.recv(src=0, timeout=1.0)
            self.assertEqual(reply.seq, 1)
            self.assertEqual(reply.payload, b"reply-to-coord")
            self.assertEqual(reply.sender, 0)
        finally:
            cp_coord.close()

    def test_send_recv_json(self) -> None:
        cp_coord = ControlPlane.create(num_ranks=2, slot_size=1024, rank=COORDINATOR)
        cp_rank1 = ControlPlane.from_buffer(cp_coord.buf, rank=1, num_ranks=2, slot_size=1024)
        try:
            data = {"step_id": 42, "tokens": [1, 2, 3], "layout": "b7"}
            cp_coord.send(dst=1, payload=data)

            msg = cp_rank1.recv(src=COORDINATOR, timeout=1.0)
            self.assertEqual(msg.flags, FLAG_JSON)
            got = msg.json()
            self.assertEqual(got, data)
        finally:
            cp_coord.close()

    def test_step_control_pack_unpack(self) -> None:
        step = StepControl(step_id=10, layout_id=128, draft_ids=(1, 2, 3), masks=(1, 1, 0))
        packed = step.pack()
        unpacked = StepControl.unpack(packed)
        self.assertEqual(unpacked, step)

    def test_acceptance_control_pack_unpack(self) -> None:
        acc = AcceptanceControl(step_id=10, accepted_len=3, accepted_tokens=(101, 102, 103))
        packed = acc.pack()
        unpacked = AcceptanceControl.unpack(packed)
        self.assertEqual(unpacked, acc)

    def test_rank_status_pack_unpack(self) -> None:
        status = RankStatus(step_id=10, rank=2, status_code=0)
        packed = status.pack()
        unpacked = RankStatus.unpack(packed)
        self.assertEqual(unpacked, status)

    def test_send_payload_exceeds_slot_size(self) -> None:
        cp = ControlPlane.create(num_ranks=2, slot_size=128)
        try:
            with self.assertRaises(ValueError):
                cp.send(dst=0, payload=b"x" * 256)
        finally:
            cp.close()

    def test_recv_timeout(self) -> None:
        cp = ControlPlane.create(num_ranks=2, slot_size=128, rank=0)
        try:
            with self.assertRaises(TimeoutError):
                cp.recv(src=COORDINATOR, timeout=0.01)
        finally:
            cp.close()

    def test_named_shm_create_attach_unlink(self) -> None:
        name = f"vxpu_test_shm_{os.getpid()}_{time.time_ns()}"
        cp = ControlPlane.create(name=name, num_ranks=4, slot_size=1024)
        try:
            attached = ControlPlane.attach(name=name, rank=0)
            self.assertEqual(attached.num_ranks, 4)
            self.assertEqual(attached.slot_size, 1024)
            attached.close()
        finally:
            cp.close()
            cp.unlink()

    def test_rank_barrier_multiprocess(self) -> None:
        num_ranks = 4
        rounds = 50
        name = f"vxpu_test_rb_{os.getpid()}_{time.time_ns()}"
        cp = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=1024)
        try:
            procs = [
                mp.Process(target=_worker_rank_barrier, args=(r, num_ranks, rounds, name))
                for r in range(num_ranks)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive(), f"Rank process {p.pid} hung")
                self.assertEqual(p.exitcode, 0)
        finally:
            cp.close()
            cp.unlink()

    def test_all_barrier_multiprocess(self) -> None:
        num_ranks = 4
        total_participants = num_ranks + 1
        rounds = 50
        name = f"vxpu_test_ab_{os.getpid()}_{time.time_ns()}"
        cp = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=1024)
        try:
            procs = [
                mp.Process(target=_worker_all_barrier, args=(p, num_ranks, rounds, name))
                for p in range(total_participants)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive(), f"Participant process {p.pid} hung")
                self.assertEqual(p.exitcode, 0)
        finally:
            cp.close()
            cp.unlink()

    def test_fake_processes_step_round_trip_benchmark(self) -> None:
        """Measures full step control round trip across coordinator and 4 ranks."""
        num_ranks = 4
        n_steps = 1000
        name = f"vxpu_test_bench_{os.getpid()}_{time.time_ns()}"
        cp = ControlPlane.create(name=name, num_ranks=num_ranks, slot_size=4096, rank=COORDINATOR)
        try:
            procs = [
                mp.Process(target=_fake_rank_step_loop, args=(r, num_ranks, n_steps, name))
                for r in range(num_ranks)
            ]
            for p in procs:
                p.start()

            # Warmup
            time.sleep(0.05)
            t0 = time.perf_counter()
            for step in range(1, n_steps + 1):
                # Coordinator publishes step message
                cp.publish_step(step_id=step, layout_id=128, draft_ids=(101, 102, 103), masks=(1, 1, 1))

                # Coordinator gathers acks from all 4 ranks
                for r in range(num_ranks):
                    status = cp.read_status(rank=r, expected_step_id=step, timeout=5.0)
                    self.assertEqual(status.status_code, 0)
            t1 = time.perf_counter()

            for p in procs:
                p.join(timeout=10.0)
                self.assertFalse(p.is_alive())
                self.assertEqual(p.exitcode, 0)

            dt_us = (t1 - t0) / n_steps * 1e6
            print(f"[vxpu] control plane round trip: {dt_us:.2f} us per step across coordinator and {num_ranks} ranks")
            # Issue target is under 50 us on c3d
            self.assertLess(dt_us, 50.0, f"Round trip time {dt_us:.2f} us exceeded target of 50 us")
        finally:
            cp.close()
            cp.unlink()


if __name__ == "__main__":
    unittest.main()
