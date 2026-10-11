# SPDX-License-Identifier: Apache-2.0
"""Request-local physical KV provenance, independent of the lossy EventBus.

Objects/bytes are physical KV shards, never logical tokens or wire bytes.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
import json
import math
import threading

SOURCES = frozenset({"local_cpu", "local_disk", "remote_store", "peer", "unknown"})


def payload_sizes(layouts: Mapping[int, Any]) -> dict[int, int]:
    """Compute uncompressed KV bytes per registered group from shapes/dtypes.

    Allocation alignment and backend serialization are intentionally excluded.
    ``layouts`` contains MemoryLayoutDesc-compatible objects. Mismatched shape
    and dtype counts raise ValueError (also rejected by MemoryLayoutDesc).
    """
    return {
        group: sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
        )
        for group, layout in layouts.items()
    }


def parse_source_mapping(value: str) -> dict[str, str]:
    """Parse declared physical sources keyed by ``l1`` or ``l2:<adapter-id>``.

    Raises ValueError for malformed JSON, keys or source names. An empty JSON
    object is valid; undeclared L2 media remain unknown (including NFS mounts).
    """
    mapping = json.loads(value)
    if not isinstance(mapping, dict):
        raise ValueError("source mapping must be a JSON object")
    for key, source in mapping.items():
        valid_key = key == "l1" or (
            key.startswith("l2:") and key[3:].isascii() and key[3:].isdigit()
        )
        if not valid_key or not isinstance(source, str) or source not in SOURCES:
            raise ValueError(f"Invalid source mapping: {key!r}: {source!r}")
    return mapping


@dataclass(frozen=True)
class Source:
    """Configured origin; adapter IDs distinguish instances of the same backend."""

    source: str
    tier: str
    backend: str
    adapter: str


@dataclass(frozen=True)
class SourceObject:
    """Successful object origin and uncompressed KV payload size in bytes."""

    source: Source
    size_bytes: int | None


class SourceMetrics:
    """Synchronous OTel counters; no request IDs, hashes or cache salts as labels.

    Pass an OTel meter for tests, otherwise use the process MeterProvider.
    Counters follow the provider lifetime and reset on process restart.
    """

    def __init__(self, meter: Any = None) -> None:
        if meter is None:
            # Third Party
            from opentelemetry import metrics

            meter = metrics.get_meter("lmcache.source")
        self._objects = meter.create_counter(
            "lmcache_mp.source_objects",
            unit="objects",
            description="Successful physical KV objects by origin and stage",
        )
        self._bytes = meter.create_counter(
            "lmcache_mp.source_bytes",
            unit="By",
            description="Uncompressed KV payload bytes by origin and stage",
        )
        self._unknown_sizes = meter.create_counter(
            "lmcache_mp.source_unknown_size_objects",
            unit="objects",
            description="Retained objects missing provenance; bytes are incomplete",
        )

    def record(self, stage: str, mode: str, objects: Iterable[SourceObject]) -> None:
        """Add an already-deduplicated batch at loaded or retained boundary.

        ``loaded`` includes successful L2 reads later discarded by prefix trim.
        ``retained`` means available to a lookup caller, not consumed by a GPU.
        """
        totals: dict[Source, list[int]] = defaultdict(lambda: [0, 0, 0])
        for obj in objects:
            totals[obj.source][0] += 1
            if obj.size_bytes is None:
                totals[obj.source][2] += 1
            else:
                totals[obj.source][1] += obj.size_bytes
        for source, (count, size, unknown) in totals.items():
            attrs = {
                "source": source.source,
                "tier": source.tier,
                "backend": source.backend,
                "adapter": source.adapter,
                "stage": stage,
                "mode": mode,
            }
            self._objects.add(count, attrs)
            if count > unknown:
                self._bytes.add(size, attrs)
            if unknown:
                self._unknown_sizes.add(unknown, attrs)


class SourceLedger:
    """One prefetch attempt's provenance; retained accounting is idempotent.

    The manager and controller share this object in-process only. It carries
    no locks/buffers and never changes storage ownership. Retried attempts get
    a new ledger; loaded metrics intentionally include repeated physical I/O.
    """

    def __init__(
        self,
        metrics: SourceMetrics,
        l1_source: Source,
        mapping: Mapping[str, str],
        mode: str,
        group_sizes: Mapping[int, int] | None = None,
    ) -> None:
        self.l1_source = l1_source
        self.mode = mode
        self._mapping = dict(mapping)
        self._group_sizes = dict(group_sizes or {})
        self._metrics = metrics
        self._objects: dict[Hashable, SourceObject] = {}
        self._loaded: set[Hashable] = set()
        self._finished = False
        self._lock = threading.Lock()

    def payload_bytes(self, group: int) -> int:
        """Return group payload bytes; unregistered groups raise KeyError."""
        return self._group_sizes[group]

    def resident(self, objects: Iterable[tuple[Hashable, int]]) -> None:
        """Record objects successfully read-locked in L1, before final trimming."""
        with self._lock:
            for key, size in objects:
                self._objects[key] = SourceObject(self.l1_source, size)

    def loaded(
        self, adapter: int, backend: str, objects: Iterable[tuple[Hashable, int]]
    ) -> None:
        """Record only successful L2 load results, once per object in this attempt."""
        source = Source(
            self._mapping.get(f"l2:{adapter}", "unknown"),
            "l2",
            backend,
            str(adapter),
        )
        with self._lock:
            new = []
            for key, size in objects:
                if key not in self._loaded:
                    obj = SourceObject(source, size)
                    self._objects[key] = obj
                    self._loaded.add(key)
                    new.append(obj)
            self._metrics.record("loaded", self.mode, new)

    def finish(self, retained: Iterable[Hashable]) -> None:
        """Count final policy-retained objects once; warmups do not claim reuse.

        Missing provenance counts as unknown with an explicit missing-size
        counter. An observability gap must not fail the serving request.
        """
        with self._lock:
            if self._finished:
                return
            unknown = SourceObject(
                Source("unknown", "unknown", "unknown", "unknown"), None
            )
            objects = [self._objects.get(key, unknown) for key in set(retained)]
            if self.mode == "lookup":
                self._metrics.record("retained", self.mode, objects)
            self._finished = True

    def snapshot(self) -> dict[Hashable, SourceObject]:
        """Return a detached source map for request diagnostics (not metric labels)."""
        with self._lock:
            return dict(self._objects)
