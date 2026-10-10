"""Explicit exclusive writer ownership layered over legacy TTL behavior.

Used by the staged L1 candidate. Finishing this lock alone is not a storage
commit; L1 must validate object identity and publish under its manager lock.
"""
from dataclasses import dataclass
import threading
import uuid
from lmcache.lmcache_native import TTLLock

@dataclass(frozen=True)
class WriteLeaseToken:
    lock_generation: str
    reservation: str

class OwnedWriteLock:
    def __init__(self, ttl_seconds=300):
        self._legacy=TTLLock(ttl_seconds)
        self._generation=uuid.uuid4().hex
        self._owner=None
        self._guard=threading.Lock()

    def lock(self):
        with self._guard:
            if self._owner is not None or self._legacy.is_locked():
                raise RuntimeError('writer already active')
            self._legacy.lock()

    def reserve(self):
        """Acquire a new owned writer without a lock-to-pin TTL window."""
        token=WriteLeaseToken(self._generation,uuid.uuid4().hex)
        with self._guard:
            if self._owner is not None or self._legacy.is_locked():
                raise RuntimeError('writer already active')
            self._legacy.lock()
            self._owner=token
            return token

    def pin(self):
        # Allocate the token before publishing ownership.
        token=WriteLeaseToken(self._generation,uuid.uuid4().hex)
        with self._guard:
            if self._owner is not None or not self._legacy.is_locked():
                raise RuntimeError('pin requires an unowned active writer')
            self._owner=token
            return token

    def owns(self, token):
        with self._guard:
            return self._owner is not None and self._owner==token

    def has_owner(self):
        with self._guard:
            return self._owner is not None

    def finish(self, token):
        if not isinstance(token,WriteLeaseToken):
            raise TypeError('expected WriteLeaseToken')
        with self._guard:
            if self._owner!=token:
                return False
            # The caller owns the exclusive writer, including its legacy bit.
            self._legacy.reset()
            self._owner=None
            return True

    def unlock(self):
        with self._guard:
            if self._owner is not None:
                return False
            self._legacy.unlock()
            return True

    def reset(self):
        with self._guard:
            if self._owner is not None:
                return False
            self._legacy.reset()
            return True

    def is_locked(self):
        with self._guard:
            return self._owner is not None or self._legacy.is_locked()
