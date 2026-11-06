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

"""The control plane in shared memory for process-per-rank execution (#539).

This library coordinates N executor card processes and a coordinator
across a process boundary. It provides fixed-size slots, monotonic
sequence numbers, and two-phase barriers aligned to CPU cache lines.

The layout reserves dedicated slots for each sender and destination pair.
A participant writes only to slots where it is the sender. Readers poll
monotonic sequence numbers, avoiding locks or kernel crossings during
steady-state decode steps. Control messages carry draft token identifiers,
attention masks, acceptance choices, layout decisions, and step counters.
"""

from dataclasses import dataclass
import json
import mmap
from multiprocessing import shared_memory
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

MAGIC = b"VXMB"
VERSION = 1
CACHELINE = 64
DEFAULT_SLOT_SIZE = 4096

COORDINATOR = -1
BROADCAST = -2

FLAG_RAW = 0
FLAG_JSON = 1
FLAG_STEP = 2
FLAG_ACCEPT = 3
FLAG_STATUS = 4


@dataclass(frozen=True)
class StepControl:
    """Binary step parameters published by the coordinator."""

    step_id: int
    layout_id: int
    draft_ids: Tuple[int, ...] = ()
    masks: Tuple[int, ...] = ()

    def pack(self) -> bytes:
        """Packs step parameters into little-endian binary bytes."""
        n = len(self.draft_ids)
        hdr = struct.pack("<QII", self.step_id, self.layout_id, n)
        d_bytes = struct.pack(f"<{n}q", *self.draft_ids) if n else b""
        m_bytes = bytes(self.masks) if self.masks else b""
        return hdr + d_bytes + m_bytes

    @classmethod
    def unpack(cls, buf: bytes) -> "StepControl":
        """Unpacks step parameters from binary bytes."""
        step_id, layout_id, n = struct.unpack_from("<QII", buf, 0)
        drafts = struct.unpack_from(f"<{n}q", buf, 16) if n else ()
        masks_offset = 16 + n * 8
        masks = tuple(buf[masks_offset : masks_offset + n]) if len(buf) >= masks_offset + n else ()
        return cls(step_id=step_id, layout_id=layout_id, draft_ids=drafts, masks=masks)


@dataclass(frozen=True)
class AcceptanceControl:
    """Binary acceptance choices published by rank 0."""

    step_id: int
    accepted_len: int
    accepted_tokens: Tuple[int, ...] = ()

    def pack(self) -> bytes:
        """Packs acceptance choices into little-endian binary bytes."""
        n = len(self.accepted_tokens)
        hdr = struct.pack("<QII", self.step_id, self.accepted_len, n)
        t_bytes = struct.pack(f"<{n}q", *self.accepted_tokens) if n else b""
        return hdr + t_bytes

    @classmethod
    def unpack(cls, buf: bytes) -> "AcceptanceControl":
        """Unpacks acceptance choices from binary bytes."""
        step_id, accepted_len, n = struct.unpack_from("<QII", buf, 0)
        tokens = struct.unpack_from(f"<{n}q", buf, 16) if n else ()
        return cls(step_id=step_id, accepted_len=accepted_len, accepted_tokens=tokens)


@dataclass(frozen=True)
class RankStatus:
    """Binary status returned by an executor rank to the coordinator."""

    step_id: int
    rank: int
    status_code: int = 0

    def pack(self) -> bytes:
        """Packs rank status into little-endian binary bytes."""
        return struct.pack("<QII", self.step_id, self.rank, self.status_code)

    @classmethod
    def unpack(cls, buf: bytes) -> "RankStatus":
        """Unpacks rank status from binary bytes."""
        step_id, rank, status_code = struct.unpack_from("<QII", buf, 0)
        return cls(step_id=step_id, rank=rank, status_code=status_code)


class Message:
    """A control message retrieved from a shared-memory slot."""

    def __init__(self, seq: int, payload: bytes, flags: int, sender: int, destination: int):
        self.seq = seq
        self.payload = payload
        self.flags = flags
        self.sender = sender
        self.destination = destination

    @property
    def text(self) -> str:
        """Decodes the payload as a UTF-8 string."""
        return self.payload.decode("utf-8")

    def json(self) -> Any:
        """Parses the payload as JSON data."""
        return json.loads(self.text)

    def as_step(self) -> StepControl:
        """Unpacks binary step control data."""
        return StepControl.unpack(self.payload)

    def as_acceptance(self) -> AcceptanceControl:
        """Unpacks binary acceptance control data."""
        return AcceptanceControl.unpack(self.payload)

    def as_status(self) -> RankStatus:
        """Unpacks binary rank status data."""
        return RankStatus.unpack(self.payload)

    def __bytes__(self) -> bytes:
        return self.payload

    def __len__(self) -> int:
        return len(self.payload)

    def __repr__(self) -> str:
        return (f"Message(seq={self.seq}, len={len(self.payload)}, "
                f"flags={self.flags}, sender={self.sender}, dst={self.destination})")


class ControlPlane:
    """The shared-memory control plane coordinating rank processes.

    The memory segment contains a 64-byte file header, barrier state arrays,
    and fixed-size message slots.
    """

    def __init__(self, buf: Any, rank: int, num_ranks: int, slot_size: int = DEFAULT_SLOT_SIZE,
                 shm_resource: Optional[Any] = None, is_owner: bool = False):
        self.buf = buf
        self.rank = rank
        self.num_ranks = num_ranks
        self.slot_size = slot_size
        self._shm_resource = shm_resource
        self._is_owner = is_owner

        self.num_senders = num_ranks + 1
        self.num_dests = num_ranks + 2
        self.num_slots = self.num_senders * self.num_dests

        self.header_size = 64
        self.rb_arrival = self.header_size
        self.rb_departure = self.rb_arrival + num_ranks * CACHELINE
        self.ab_arrival = self.rb_departure + num_ranks * CACHELINE
        self.ab_departure = self.ab_arrival + self.num_senders * CACHELINE

        barriers_end = self.ab_departure + self.num_senders * CACHELINE
        self.slots_offset = (barriers_end + CACHELINE - 1) & ~(CACHELINE - 1)
        self.slot_stride = 64 + slot_size

        self._out_seq: Dict[Tuple[int, int], int] = {}
        self._in_seq: Dict[Tuple[int, int], int] = {}
        self._rb_round = 0
        self._ab_round = 0

    @classmethod
    def total_bytes(cls, num_ranks: int, slot_size: int = DEFAULT_SLOT_SIZE) -> int:
        """Calculates total shared-memory bytes needed for N ranks and slot size."""
        num_senders = num_ranks + 1
        num_dests = num_ranks + 2
        num_slots = num_senders * num_dests
        header_size = 64
        rb_size = 2 * num_ranks * CACHELINE
        ab_size = 2 * num_senders * CACHELINE
        barriers_end = header_size + rb_size + ab_size
        slots_offset = (barriers_end + CACHELINE - 1) & ~(CACHELINE - 1)
        total = slots_offset + num_slots * (64 + slot_size)
        return (total + 4095) & ~4095

    @classmethod
    def create(cls, name: Optional[str] = None, num_ranks: int = 4,
               slot_size: int = DEFAULT_SLOT_SIZE, rank: int = COORDINATOR) -> "ControlPlane":
        """Creates a new shared-memory control plane segment.

        When name is None, allocates anonymous shared memory suitable for
        forked processes. When name is provided, allocates a named POSIX
        shared-memory block that independent processes attach to.
        """
        size = cls.total_bytes(num_ranks, slot_size)
        if name is not None:
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
            buf = shm.buf
            resource = shm
        else:
            buf = mmap.mmap(-1, size, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
            resource = buf

        # Initialize header: magic (4B), version (4B), num_ranks (4B), slot_size (4B), size (8B)
        buf[0:4] = MAGIC
        struct.pack_into("<IIIQ", buf, 4, VERSION, num_ranks, slot_size, size)

        # Zero barrier areas and slot headers
        barriers_end = 64 + 2 * num_ranks * CACHELINE + 2 * (num_ranks + 1) * CACHELINE
        buf[64:barriers_end] = b"\x00" * (barriers_end - 64)

        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size,
                   shm_resource=resource, is_owner=True)

    @classmethod
    def attach(cls, name: str, rank: int) -> "ControlPlane":
        """Attaches to an existing named shared-memory control plane segment."""
        shm = shared_memory.SharedMemory(create=False, name=name)
        buf = shm.buf

        magic = bytes(buf[0:4])
        if magic != MAGIC:
            shm.close()
            raise ValueError(f"Invalid shared memory magic: expected {MAGIC!r}, got {magic!r}")

        version, num_ranks, slot_size, _ = struct.unpack_from("<IIIQ", buf, 4)
        if version != VERSION:
            shm.close()
            raise ValueError(f"Unsupported control plane version: expected {VERSION}, got {version}")

        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size,
                   shm_resource=shm, is_owner=False)

    @classmethod
    def from_buffer(cls, buf: Any, rank: int, num_ranks: int,
                    slot_size: int = DEFAULT_SLOT_SIZE) -> "ControlPlane":
        """Wraps an existing buffer or memoryview directly."""
        return cls(buf=buf, rank=rank, num_ranks=num_ranks, slot_size=slot_size)

    def slot_offset(self, src: int, dst: int) -> int:
        """Computes the byte offset of the slot for a (src, dst) pair."""
        s_idx = self.num_ranks if src == COORDINATOR else src
        d_idx = self.num_ranks if dst == COORDINATOR else (self.num_ranks + 1 if dst == BROADCAST else dst)
        k = s_idx * self.num_dests + d_idx
        return self.slots_offset + k * self.slot_stride

    def send(self, dst: int, payload: Union[bytes, str, Dict[str, Any], List[Any], StepControl, AcceptanceControl, RankStatus],
             flags: int = FLAG_RAW, seq: Optional[int] = None) -> int:
        """Sends a control message to destination dst.

        Encodes strings or dictionaries when provided. Writes metadata and
        payload first, then commits the sequence number to notify readers.
        """
        if isinstance(payload, StepControl):
            data = payload.pack()
            flags = FLAG_STEP
        elif isinstance(payload, AcceptanceControl):
            data = payload.pack()
            flags = FLAG_ACCEPT
        elif isinstance(payload, RankStatus):
            data = payload.pack()
            flags = FLAG_STATUS
        elif isinstance(payload, (dict, list)):
            data = json.dumps(payload).encode("utf-8")
            if flags == FLAG_RAW:
                flags = FLAG_JSON
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            data = bytes(payload)
        else:
            raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

        if len(data) > self.slot_size:
            raise ValueError(f"Payload size {len(data)} exceeds slot size {self.slot_size}")

        off = self.slot_offset(self.rank, dst)
        k = (self.rank, dst)
        if seq is None:
            self._out_seq[k] = self._out_seq.get(k, 0) + 1
            seq = self._out_seq[k]
        else:
            self._out_seq[k] = seq

        n = len(data)
        # Write payload first
        self.buf[off + 64 : off + 64 + n] = data
        # Write metadata header at off + 8: length, flags, sender, destination
        struct.pack_into("<IIii", self.buf, off + 8, n, flags, self.rank, dst)
        # Commit sequence number last at off + 0 to publish message atomically
        struct.pack_into("<Q", self.buf, off, seq)
        return seq

    def recv(self, src: int, dst: Optional[int] = None,
             expected_seq: Optional[int] = None, timeout: Optional[float] = None) -> Message:
        """Waits for and reads a control message from sender src.

        Spins for low latency on fast steps, yielding periodically to avoid
        hogging CPU during longer pauses. Raises TimeoutError when timeout expires.
        """
        if dst is None:
            dst = self.rank
        off = self.slot_offset(src, dst)
        k = (src, dst)
        if expected_seq is None:
            expected_seq = self._in_seq.get(k, 0) + 1

        t0 = time.perf_counter()
        spin = 0
        while True:
            cur_seq = struct.unpack_from("<Q", self.buf, off)[0]
            if cur_seq >= expected_seq:
                break
            spin += 1
            if (spin & 1023) == 0:
                if timeout is not None and (time.perf_counter() - t0) > timeout:
                    raise TimeoutError(f"recv timed out waiting for seq {expected_seq} from {src} (slot {k})")
                if spin > 50000:
                    time.sleep(0.0001)

        n, flags, sender, destination = struct.unpack_from("<IIii", self.buf, off + 8)
        payload = bytes(self.buf[off + 64 : off + 64 + n])
        self._in_seq[k] = cur_seq
        return Message(seq=cur_seq, payload=payload, flags=flags, sender=sender, destination=destination)

    def rank_barrier(self, timeout: Optional[float] = None) -> None:
        """Synchronizes card ranks (0 through num_ranks-1) across a step boundary."""
        if self.rank < 0 or self.rank >= self.num_ranks:
            raise ValueError(f"Participant rank {self.rank} is not a worker rank (0..{self.num_ranks - 1})")
        self._rb_round += 1
        rnd = self._rb_round
        r = self.rank

        # Phase 1: Signal arrival
        struct.pack_into("<Q", self.buf, self.rb_arrival + r * CACHELINE, rnd)
        t0 = time.perf_counter()
        for i in range(self.num_ranks):
            off = self.rb_arrival + i * CACHELINE
            spin = 0
            while struct.unpack_from("<Q", self.buf, off)[0] < rnd:
                spin += 1
                if (spin & 1023) == 0:
                    if timeout is not None and (time.perf_counter() - t0) > timeout:
                        raise TimeoutError(f"rank_barrier arrival timed out at round {rnd}")
                    if spin > 50000:
                        time.sleep(0.0001)

        # Phase 2: Signal departure
        struct.pack_into("<Q", self.buf, self.rb_departure + r * CACHELINE, rnd)
        for i in range(self.num_ranks):
            off = self.rb_departure + i * CACHELINE
            spin = 0
            while struct.unpack_from("<Q", self.buf, off)[0] < rnd:
                spin += 1
                if (spin & 1023) == 0:
                    if timeout is not None and (time.perf_counter() - t0) > timeout:
                        raise TimeoutError(f"rank_barrier departure timed out at round {rnd}")
                    if spin > 50000:
                        time.sleep(0.0001)

    def barrier(self, timeout: Optional[float] = None) -> None:
        """Synchronizes all participants, including the coordinator and ranks."""
        self._ab_round += 1
        rnd = self._ab_round
        p_idx = self.num_ranks if self.rank == COORDINATOR else self.rank

        # Phase 1: Signal arrival
        struct.pack_into("<Q", self.buf, self.ab_arrival + p_idx * CACHELINE, rnd)
        t0 = time.perf_counter()
        for i in range(self.num_senders):
            off = self.ab_arrival + i * CACHELINE
            spin = 0
            while struct.unpack_from("<Q", self.buf, off)[0] < rnd:
                spin += 1
                if (spin & 1023) == 0:
                    if timeout is not None and (time.perf_counter() - t0) > timeout:
                        raise TimeoutError(f"barrier arrival timed out at round {rnd}")
                    if spin > 50000:
                        time.sleep(0.0001)

        # Phase 2: Signal departure
        struct.pack_into("<Q", self.buf, self.ab_departure + p_idx * CACHELINE, rnd)
        for i in range(self.num_senders):
            off = self.ab_departure + i * CACHELINE
            spin = 0
            while struct.unpack_from("<Q", self.buf, off)[0] < rnd:
                spin += 1
                if (spin & 1023) == 0:
                    if timeout is not None and (time.perf_counter() - t0) > timeout:
                        raise TimeoutError(f"barrier departure timed out at round {rnd}")
                    if spin > 50000:
                        time.sleep(0.0001)

    # ---- High-level control message helpers ----

    def publish_step(self, step_id: int, layout_id: int = 0,
                     draft_ids: Optional[Union[List[int], Tuple[int, ...]]] = None,
                     masks: Optional[Union[List[int], Tuple[int, ...]]] = None) -> int:
        """Publishes step parameters from the coordinator to all ranks."""
        ctrl = StepControl(
            step_id=step_id,
            layout_id=layout_id,
            draft_ids=tuple(draft_ids) if draft_ids else (),
            masks=tuple(masks) if masks else ()
        )
        return self.send(dst=BROADCAST, payload=ctrl, seq=step_id)

    def read_step(self, expected_step_id: Optional[int] = None,
                  timeout: Optional[float] = None) -> StepControl:
        """Reads the step broadcast published by the coordinator."""
        msg = self.recv(src=COORDINATOR, dst=BROADCAST, expected_seq=expected_step_id, timeout=timeout)
        return msg.as_step()

    def publish_acceptance(self, step_id: int, accepted_len: int,
                           accepted_tokens: Optional[Union[List[int], Tuple[int, ...]]] = None) -> int:
        """Publishes acceptance results from rank 0 to peer ranks."""
        ctrl = AcceptanceControl(
            step_id=step_id,
            accepted_len=accepted_len,
            accepted_tokens=tuple(accepted_tokens) if accepted_tokens else ()
        )
        return self.send(dst=BROADCAST, payload=ctrl, seq=step_id)

    def read_acceptance(self, expected_step_id: Optional[int] = None,
                        timeout: Optional[float] = None) -> AcceptanceControl:
        """Reads acceptance results published by rank 0."""
        msg = self.recv(src=0, dst=BROADCAST, expected_seq=expected_step_id, timeout=timeout)
        return msg.as_acceptance()

    def post_status(self, step_id: int, status_code: int = 0) -> int:
        """Posts rank execution status back to the coordinator."""
        status = RankStatus(step_id=step_id, rank=self.rank, status_code=status_code)
        return self.send(dst=COORDINATOR, payload=status, seq=step_id)

    def read_status(self, rank: int, expected_step_id: Optional[int] = None,
                    timeout: Optional[float] = None) -> RankStatus:
        """Reads status from an executor rank at the coordinator."""
        msg = self.recv(src=rank, dst=COORDINATOR, expected_seq=expected_step_id, timeout=timeout)
        return msg.as_status()

    # ---- Resource cleanup ----

    def close(self) -> None:
        """Closes the shared-memory buffer or mapping."""
        if hasattr(self.buf, "close"):
            self.buf.close()
        elif self._shm_resource is not None and hasattr(self._shm_resource, "close"):
            self._shm_resource.close()

    def unlink(self) -> None:
        """Unlinks named shared memory if this instance owns the allocation."""
        if self._is_owner and self._shm_resource is not None and hasattr(self._shm_resource, "unlink"):
            try:
                self._shm_resource.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "ControlPlane":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
        if self._is_owner:
            self.unlink()


ShmMailbox = ControlPlane
