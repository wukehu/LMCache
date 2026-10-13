# SPDX-License-Identifier: Apache-2.0
"""Read-only filesystem inventory, independent of cache lifecycle counters."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import os
import stat
import threading
import time


@dataclass(frozen=True)
class InventorySnapshot:
    """A completed directory pass, not an atomic filesystem snapshot.

    Logical bytes are file lengths. Allocated bytes are filesystem blocks,
    or None where the platform cannot report them. Neither proves valid KV
    contents, memory footprint after deserialization, nor enforced quota.
    """

    objects: int
    logical_bytes: int
    allocated_bytes: int | None
    completed_at: float
    duration_seconds: float


class InventoryIncomplete(RuntimeError):
    """A bounded scan could not produce a complete directory observation."""


def scan_inventory(
    path: str,
    accepts_name: Callable[[str], bool],
    *,
    max_entries: int = 100_000,
    max_seconds: float = 2.0,
    cancelled: threading.Event | None = None,
) -> InventorySnapshot:
    """Read one directory without following symlinks or reading file contents.

    Args:
        path: Effective adapter directory; callers supply its filename parser.
        accepts_name: True for committed cache object filenames only.
        max_entries: Maximum entries inspected, including rejected names.
        max_seconds: Cooperative wall-clock budget between filesystem calls.
        cancelled: Optional shutdown signal checked during traversal.

    Returns:
        One complete pass. Empty directories produce a genuine zero value.

    Raises:
        ValueError: Invalid limits.
        InventoryIncomplete: Budget, cancellation, or filesystem failure.

    A blocking filesystem syscall cannot be interrupted by this budget. A
    concurrently disappearing object invalidates the pass rather than turning
    a partial sum into a complete inventory. Writes concurrent with a completed
    scan still mean this is a sampled observation, not a transactional snapshot.
    """
    if max_entries < 1 or not 0 < max_seconds < float("inf"):
        raise ValueError("positive finite scan limits required")
    started = time.monotonic()
    objects = logical = allocated = entries = 0
    has_allocated = True

    def check_budget() -> None:
        if cancelled is not None and cancelled.is_set():
            raise InventoryIncomplete("cancelled")
        if time.monotonic() - started > max_seconds:
            raise InventoryIncomplete("time limit")

    try:
        check_budget()
        with os.scandir(path) as directory:
            for entry in directory:
                check_budget()
                entries += 1
                if entries > max_entries:
                    raise InventoryIncomplete("entry limit")
                if not accepts_name(entry.name):
                    continue
                # DirEntry.stat may cache enumeration metadata on Windows.
                # A fresh lstat also detects a file removed during the pass.
                info = os.stat(entry.path, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                if info.st_size < 0:
                    raise InventoryIncomplete("negative file size")
                objects += 1
                logical += info.st_size
                blocks = getattr(info, "st_blocks", None)
                if blocks is None:
                    has_allocated = False
                elif blocks < 0:
                    raise InventoryIncomplete("negative allocated blocks")
                else:
                    allocated += blocks * 512
        check_budget()
    except OSError as exc:
        raise InventoryIncomplete(type(exc).__name__) from exc
    return InventorySnapshot(
        objects, logical, allocated if has_allocated else None,
        time.time(), time.monotonic() - started,
    )


class InventoryObservation:
    """Thread-safe last-complete value and latest-attempt status.

    A failed scan retains the last complete snapshot and marks success false.
    No complete value is represented by None, never a fabricated zero.
    The owner schedules scans outside scrape callbacks and joins its worker.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: InventorySnapshot | None = None
        self._success = False

    def sample(
        self, path: str, accepts_name: Callable[[str], bool], *,
        max_entries: int = 100_000, max_seconds: float = 2.0,
        cancelled: threading.Event | None = None,
    ) -> bool:
        """Attempt one scan; return success, retaining previous data on failure."""
        try:
            snapshot = scan_inventory(
                path, accepts_name, max_entries=max_entries,
                max_seconds=max_seconds, cancelled=cancelled,
            )
        except InventoryIncomplete:
            with self._lock:
                self._success = False
            return False
        with self._lock:
            self._snapshot = snapshot
            self._success = True
        return True

    def read(self) -> tuple[InventorySnapshot | None, bool]:
        """Return immutable last-complete data and latest-attempt success."""
        with self._lock:
            return self._snapshot, self._success


class InventorySampler:
    """One read-only worker; scrape callbacks only read cached observations.

    Targets map adapter IDs to effective paths. Path changes discard the old
    snapshot, as do removed targets. A failed target refresh marks existing
    observations unsuccessful rather than claiming current data. The worker
    never calls adapter read/write/lock or lifecycle notification methods.
    """

    def __init__(
        self, targets: Callable[[], dict[str, str]],
        accepts_name: Callable[[str], bool], *, interval: float = 30.0,
    ) -> None:
        if not 0 < interval < float("inf"):
            raise ValueError("positive finite interval required")
        self._targets = targets
        self._accepts_name = accepts_name
        self._interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._values: dict[
            tuple[str, str], tuple[InventorySnapshot | None, bool]
        ] = {}
        self._thread = threading.Thread(
            target=self._run, name="lmcache-fs-inventory", daemon=True,
        )

    def start(self) -> None:
        """Start once, after the adapter registry has been initialized."""
        self._thread.start()

    def close(self, timeout: float = 3.0) -> bool:
        """Cancel traversal and join; False reports a still-blocked syscall."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def read(self) -> dict[str, tuple[InventorySnapshot | None, bool]]:
        """Return cached per-adapter values without filesystem access."""
        with self._lock:
            return {key[0]: value for key, value in self._values.items()}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                targets = self._targets()
                with self._lock:
                    previous = dict(self._values)
                values = {}
                for adapter_id, path in targets.items():
                    if self._stop.is_set():
                        return
                    key = (adapter_id, path)
                    try:
                        snapshot = scan_inventory(
                            path, self._accepts_name, cancelled=self._stop,
                        )
                        values[key] = (snapshot, True)
                    except InventoryIncomplete:
                        values[key] = (previous.get(key, (None, False))[0], False)
                with self._lock:
                    self._values = values
            except Exception:
                # A telemetry error must not kill the worker or serving path.
                with self._lock:
                    self._values = {key: (value[0], False)
                                    for key, value in self._values.items()}
            self._stop.wait(self._interval)
