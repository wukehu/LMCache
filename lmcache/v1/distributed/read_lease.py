# SPDX-License-Identifier: Apache-2.0
"""Per-object read reservations and completion-scoped pins.

The containing L1 manager must serialize all calls with its existing mutex.
Create a new ledger for every object incarnation. Never key completions solely
by a content hash: a replaced allocation may have the same hash.
"""

# Standard
from dataclasses import dataclass, field
from typing import Hashable
import math


@dataclass(frozen=True, eq=False)
class ReadClaim:
    """Opaque identity for one reader's claim on one object incarnation."""

    owner: Hashable
    reader: Hashable


@dataclass
class _Reservation:
    readers: int
    deadline: float
    claimed: set[Hashable] = field(default_factory=set)
    completed: set[Hashable] = field(default_factory=set)
    cancelled: bool = False


class ReadLeaseLedger:
    """Track exact ownership independently of a shared expiring counter.

    A pending reservation expires for eviction purposes, but its owner record
    remains until explicit cleanup. Acquiring after expiry requires the caller
    to validate that this object's committed allocation is still resident.
    Active copies stay pinned until their exact completion token is returned.
    """

    def __init__(self) -> None:
        self._owners: dict[Hashable, _Reservation] = {}
        self._active: dict[ReadClaim, _Reservation] = {}

    def reserve(self, owner: Hashable, readers: int, deadline: float) -> None:
        """Register a unique lookup generation; reject ambiguous replacement."""
        if readers <= 0:
            raise ValueError("readers must be positive")
        if owner in self._owners:
            raise ValueError("lookup generation already reserved")
        self._owners[owner] = _Reservation(readers, deadline)

    def acquire(self, owner: Hashable, reader: Hashable) -> ReadClaim | None:
        """Pin one validated resident object, once per reader and generation.

        The L1 manager must check committed/resident state under the same mutex
        immediately before this call. An active pin is not an expiring lease.
        """
        reservation = self._owners.get(owner)
        if (
            reservation is None
            or reservation.cancelled
            or reader in reservation.claimed
            or len(reservation.claimed) >= reservation.readers
        ):
            return None
        claim = ReadClaim(owner, reader)
        reservation.claimed.add(reader)
        self._active[claim] = reservation
        return claim

    def renew(self, owner: Hashable, deadline: float) -> bool:
        """Extend an existing pending owner without adding or reviving readers.

        The caller authenticates the Session and validates its bounded future
        deadline. This ledger belongs to one allocation incarnation only.
        """
        if not math.isfinite(deadline):
            raise ValueError("read lease deadline must be finite")
        reservation = self._owners.get(owner)
        if (reservation is None or reservation.cancelled
                or len(reservation.completed) >= reservation.readers):
            return False
        reservation.deadline = max(reservation.deadline, deadline)
        return True

    def complete(self, claim: ReadClaim) -> bool:
        """Release exactly this pin; stale, forged or duplicate tokens do nothing."""
        reservation = self._active.pop(claim, None)
        if reservation is None:
            return False
        reservation.completed.add(claim.reader)
        if reservation.cancelled:
            self.forget(claim.owner)
        return True

    def abandon(self, owner: Hashable, reader: Hashable) -> bool:
        """Consume an unused reader slot without releasing an active transfer."""
        reservation = self._owners.get(owner)
        if (
            reservation is None
            or reservation.cancelled
            or reader in reservation.claimed
            or len(reservation.claimed) >= reservation.readers
        ):
            return False
        reservation.claimed.add(reader)
        reservation.completed.add(reader)
        return True

    def cancel(self, owner: Hashable) -> bool:
        """Discard unconsumed claims while keeping already queued GPU copies pinned."""
        reservation = self._owners.get(owner)
        if reservation is not None:
            reservation.cancelled = True
            return True
        return False

    def forget(self, owner: Hashable) -> bool:
        """Remove a finished/cancelled owner only after every active copy completes."""
        reservation = self._owners.get(owner)
        if reservation is None:
            return True
        if any(value is reservation for value in self._active.values()):
            return False
        if not reservation.cancelled and len(reservation.completed) < reservation.readers:
            return False
        del self._owners[owner]
        return True

    def outstanding(self, owner: Hashable) -> int:
        """Return remaining consumer claims, including active transfers."""
        reservation = self._owners.get(owner)
        if reservation is None:
            return 0
        if reservation.cancelled:
            return sum(value is reservation for value in self._active.values())
        return reservation.readers - len(reservation.completed)

    def is_pinned(self, now: float) -> bool:
        """Protect active copies and unexpired unconsumed lookup reservations."""
        return bool(self._active) or any(
            not r.cancelled and now < r.deadline and len(r.completed) < r.readers
            for r in self._owners.values()
        )

    def expired_by(self, owner: Hashable, now: float) -> float | None:
        """Return elapsed seconds past this owner's deadline, or None if unknown."""
        reservation = self._owners.get(owner)
        return None if reservation is None else now - reservation.deadline
