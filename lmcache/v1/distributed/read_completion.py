# SPDX-License-Identifier: Apache-2.0
"""Process-local receipt storage for serialized GPU completion callbacks."""

# Standard
from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")


class ReadCompletionRegistry(Generic[T]):
    """Keep live allocations local; only an opaque string crosses the dispatcher.

    A receipt is retired only after its completion handler succeeds. Duplicate
    or concurrently delivered callbacks cannot invoke the handler twice.
    """

    def __init__(self, finish: Callable[[T], None]) -> None:
        self._lock = Lock()
        self._receipts: dict[str, T] = {}
        self._finish = finish

    def register(self, receipt: T) -> str:
        """Retain a receipt until a stream-ordered completion consumes its handle."""
        handle = uuid4().hex
        with self._lock:
            self._receipts[handle] = receipt
        return handle

    def complete(self, handle: str) -> bool:
        """Consume exactly one handle; unknown or delayed duplicates are harmless."""
        with self._lock:
            if handle not in self._receipts:
                return False
            self._finish(self._receipts[handle])
            del self._receipts[handle]
            return True

    def pending_count(self) -> int:
        """Return retained receipts for diagnostics and orderly teardown checks."""
        with self._lock:
            return len(self._receipts)
