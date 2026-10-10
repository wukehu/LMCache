# SPDX-License-Identifier: Apache-2.0
"""Local owned-read lease primitive; transport/L1 integration is still pending."""
from dataclasses import dataclass, field
import itertools
import threading
import uuid

from lmcache.lmcache_native import TTLLock


class ReadLeaseCapacityError(RuntimeError):
    """A cache object has reached its bounded reservation capacity."""


@dataclass(frozen=True)
class ReadLeaseToken:
    lock_generation: str
    reservation: int


@dataclass
class _Reservation:
    remaining: int
    claims: dict[str, int] = field(default_factory=dict)


class OwnedReadLock:
    """Separate explicit read ownership from the legacy anonymous TTL counter.

    This token is local to one MP process and one object/lock incarnation.
    There is deliberately no time-based release of owned shares: the caller
    must prove that a share was not handed to a GPU, or that the GPU completed.
    """

    def __init__(self, ttl_seconds: int = 300, max_leases: int = 1024):
        if isinstance(max_leases, bool) or not isinstance(max_leases, int) or max_leases <= 0:
            raise ValueError('max_leases must be a positive integer')
        self._legacy = TTLLock(ttl_seconds)
        self._generation = uuid.uuid4().hex
        self._sequence = itertools.count()
        self._leases: dict[ReadLeaseToken, _Reservation] = {}
        self._max_leases = max_leases
        self._guard = threading.Lock()

    @staticmethod
    def _validate_count(count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 128:
            raise ValueError('read count must be an integer in [1,128]')

    def reserve(self, count: int = 1) -> ReadLeaseToken:
        self._validate_count(count)
        with self._guard:
            if len(self._leases) >= self._max_leases:
                raise ReadLeaseCapacityError('Owned read reservations exhausted')
            token = ReadLeaseToken(self._generation, next(self._sequence))
            self._leases[token] = _Reservation(count)
            return token

    def adopt_anonymous(self) -> ReadLeaseToken | None:
        """Replace one live anonymous share with a non-expiring owned share.

        The guard excludes anonymous renewal between checking and consuming
        the legacy share. The L1 caller must also exclude object replacement.
        """
        with self._guard:
            if not self._legacy.is_locked():
                return None
            if len(self._leases) >= self._max_leases:
                raise ReadLeaseCapacityError('Owned read reservations exhausted')
            token = ReadLeaseToken(self._generation, next(self._sequence))
            self._leases[token] = _Reservation(1)
            self._legacy.unlock()
            return token

    def release(self, token: ReadLeaseToken, claim_id: str, count: int = 1) -> bool:
        """Release a consumer's share once; stale tokens never affect new leases."""
        self._validate_count(count)
        if not isinstance(token, ReadLeaseToken):
            raise TypeError('Expected ReadLeaseToken')
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError('claim_id must be a nonempty string')
        with self._guard:
            reservation = self._leases.get(token)
            if reservation is None:
                return False
            previous = reservation.claims.get(claim_id)
            if previous is not None:
                if previous != count:
                    raise ValueError('Repeated release changes reader count')
                return False
            if count > reservation.remaining:
                raise ValueError('Release exceeds remaining owned readers')
            reservation.claims[claim_id] = count
            reservation.remaining -= count
            if reservation.remaining == 0:
                del self._leases[token]
            return True

    def owns(self, token: ReadLeaseToken) -> bool:
        with self._guard:
            return token in self._leases

    def snapshot(self) -> dict[str, int]:
        with self._guard:
            return {'active_leases': len(self._leases),
                    'owned_readers': sum(v.remaining for v in self._leases.values()),
                    'max_leases': self._max_leases}

    def lock(self) -> None:
        """Legacy anonymous reservation; cannot mutate owned shares."""
        with self._guard:
            self._legacy.lock()

    def unlock(self) -> None:
        with self._guard:
            self._legacy.unlock()

    def reset(self) -> None:
        """Reset only anonymous state; explicit owned shares stay pinned."""
        with self._guard:
            self._legacy.reset()

    def anonymous_is_locked(self) -> bool:
        with self._guard:
            return self._legacy.is_locked()

    def is_locked(self) -> bool:
        with self._guard:
            return bool(self._leases) or self._legacy.is_locked()
