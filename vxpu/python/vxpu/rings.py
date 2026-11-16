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

"""A message ring in shared memory (csrc/ring.h).

Two processes on one machine exchange fixed-size messages through shared
memory with no lock and no system call on the fast path. Every message records
when it was written and when it was read, timing every crossing directly.
"""

# pyright: strict

import os
from typing import Any
import torch

from . import kernels


def _require_kernels() -> Any:
    k: Any = kernels.load()
    if k is None:
        raise RuntimeError("native kernels unavailable")
    return k


class Ring:
    """A single-producer, single-consumer message ring in shared memory."""

    def __init__(self, region: torch.Tensor, path: str | None = None) -> None:
        self.region: torch.Tensor = region
        self.path: str | None = path
        self._k: Any = _require_kernels()
        # Validate that region holds a valid ring header and sufficient bytes.
        self._k.ring_slots(self.region)

    @property
    def slots(self) -> int:
        """The number of message slots in the ring."""
        return int(self._k.ring_slots(self.region))

    @property
    def slot_bytes(self) -> int:
        """The maximum payload bytes per slot."""
        return int(self._k.ring_slot_bytes(self.region))

    @classmethod
    def bytes_for(cls, slots: int, slot_bytes: int) -> int:
        """The total number of bytes needed for a ring region."""
        k = _require_kernels()
        return int(k.ring_bytes(slots, slot_bytes))

    @classmethod
    def create(
        cls,
        path: str,
        slots: int,
        slot_bytes: int,
        role: str | None = None,
        peer_pid: int = 0,
    ) -> "Ring":
        """Create and initialize a new ring backed by a shared memory file."""
        k = _require_kernels()
        nbytes = int(k.ring_bytes(slots, slot_bytes))
        fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.truncate(nbytes)
        region: torch.Tensor = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
        k.ring_init(region, slots, slot_bytes)
        ring = cls(region, path=path)
        ring._apply_role(role, peer_pid)
        return ring

    @classmethod
    def attach(
        cls,
        path: str,
        role: str | None = None,
        peer_pid: int = 0,
        slots: int | None = None,
        slot_bytes: int | None = None,
    ) -> "Ring":
        """Attach to an existing ring backed by a shared memory file."""
        file_size = os.path.getsize(path)
        region: torch.Tensor = torch.from_file(path, shared=True, size=file_size, dtype=torch.uint8)
        ring = cls(region, path=path)
        if slots is not None and ring.slots != slots:
            raise ValueError(f"slots mismatch: header has {ring.slots}, caller passed {slots}")
        if slot_bytes is not None and ring.slot_bytes != slot_bytes:
            raise ValueError(f"slot_bytes mismatch: header has {ring.slot_bytes}, caller passed {slot_bytes}")
        ring._apply_role(role, peer_pid)
        return ring

    def _apply_role(self, role: str | None, peer_pid: int) -> None:
        if role == "writer":
            self._k.ring_attach_writer(self.region, peer_pid)
        elif role == "reader":
            self._k.ring_attach_reader(self.region, peer_pid)
        elif peer_pid > 0:
            raise ValueError("specifying peer_pid requires specifying role ('writer' or 'reader')")

    def write(self, data: bytes, timeout_ns: int = -1) -> None:
        """Write a message into the ring, waiting if full."""
        self._k.ring_write(self.region, data, timeout_ns)

    def read(self, timeout_ns: int = -1) -> tuple[bytes, int, int]:
        """Read the next message, returning (payload, written_ns, read_ns)."""
        res: tuple[bytes, int, int] = self._k.ring_read(self.region, timeout_ns)
        return res

    def set_writer_pid(self, pid: int) -> None:
        """Set writer PID for peer liveness monitoring."""
        self._k.ring_set_writer_pid(self.region, pid)

    def set_reader_pid(self, pid: int) -> None:
        """Set reader PID for peer liveness monitoring."""
        self._k.ring_set_reader_pid(self.region, pid)

    def unlink(self) -> None:
        """Remove the backing file if this ring was created or attached with a path."""
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def close(self, unlink: bool = False) -> None:
        """Close the ring, notifying waiting peers, and optionally unlink the file."""
        self._k.ring_close(self.region)
        if unlink:
            self.unlink()

    def echo(self, out_ring: "Ring", messages: int, timeout_ns: int = -1) -> None:
        """Answer messages from this ring with the same payload on out_ring."""
        self._k.ring_echo(self.region, out_ring.region, messages, timeout_ns)

    def ping(self, in_ring: "Ring", messages: int, bytes_count: int, timeout_ns: int = -1) -> list[int]:
        """Send messages on this ring and time replies on in_ring, returning round trips in ns."""
        res: list[int] = self._k.ring_ping(self.region, in_ring.region, messages, bytes_count, timeout_ns)
        return res
