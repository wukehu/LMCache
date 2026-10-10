# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional, overload
import threading
import sys
from lmcache.v1.multiprocess.resource_budget import Usage
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    token_ids: list[int] = field(default_factory=list)
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)
    lookup_ipc_key: Optional[IPCCacheServerKey] = None
    prefetch_hit_chunks: int = -1
    prefetch_locked_gids: tuple = ()
    prefetch_group_windows: tuple[int, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
    _lookup_generation: int = field(default=0, repr=False)
    _retired: bool = field(default=False, repr=False)
    _read_leases: dict | None = field(default=None, repr=False)
    _read_release_check: Any = field(default=None, repr=False)
    _read_cleanup_retry: Any = field(default=None, repr=False)
    _read_retry_after: float = field(default=0.0, repr=False)
    _read_retry_delay: float = field(default=1.0, repr=False)
    _retired_unhanded_plan: tuple | None = field(default=None, repr=False)
    _scheduler_released_objects: set[Any] = field(default_factory=set, repr=False)
    _worker_object_claims: dict[Any, set[int]] = field(default_factory=dict, repr=False)
    _retrieve_results: dict[tuple, tuple[Any, Future]] = field(default_factory=dict, repr=False)
    _external_completions: list[Future] = field(default_factory=list, repr=False)
    _completion_events: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _failed_retrieve_releases: set[tuple[int, int, int, int, int]] = field(
        default_factory=set, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    _budget_account: Any = field(default=None, repr=False)
    _token_reservation: Any = field(default=None, repr=False)
    _operation_reservations: dict = field(default_factory=dict, repr=False)

    def _release_resource_budget(self):
        with self._lock:
            if self._token_reservation is not None:
                self._token_reservation.release()
                self._token_reservation = None
            for reservation in self._operation_reservations.values():
                reservation.release()
            self._operation_reservations.clear()

    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        with self._lock:
            if self._budget_account is None:
                self.token_ids = full_token_ids
                return
            if self._retired:
                raise ValueError("Cannot update tokens on a retired session")
            # Snapshot caller-owned mutable input; account its retained Python
            # list and integer sizes. Transport/decode peaks are separate.
            tokens = list(full_token_ids)
            if any(type(token) is not int for token in tokens):
                raise ValueError("token IDs must be integers")
            usage = Usage(metadata_bytes=sys.getsizeof(tokens)+sum(sys.getsizeof(t) for t in tokens), operations=1)
            if self._token_reservation is None:
                reservation = self._budget_account.reserve(usage)
                self._token_reservation = reservation
            else:
                self._token_reservation.resize(usage)
            self.token_ids = tokens

    @overload
    def get_hashes(self, start: int, end: int) -> list: ...

    @overload
    def get_hashes(self, start: int) -> list: ...

    def get_hashes(self, start: int, end: int | None = None) -> list:
        """Compute and return chunk hashes for the [start, end) token range.

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks.

        Two calling conventions are supported (declared via ``@overload``)::

            get_hashes(start, end)  # explicit end
            get_hashes(start)       # end = last full-chunk boundary

        Args:
            start: Start token index (must be aligned to chunk_size).
            end: End token index (must be aligned to chunk_size).
                When omitted (``None``), automatically set to the last
                full-chunk boundary of the current token sequence.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).

        Raises:
            ValueError: If an explicit ``end`` exceeds the current tokens
                (hashing past them yields valid-looking garbage).
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size

        with self._lock:
            if end is not None and end > len(self.token_ids):
                raise ValueError(
                    f"get_hashes end ({end}) exceeds the session's "
                    f"{len(self.token_ids)} token(s); the session may have "
                    "been recreated after request cleanup"
                )
            if end is None:
                # No explicit end: use the last full-chunk boundary.
                # Lock must be held here because `self.token_ids` may be
                # concurrently replaced by `set_tokens` from another thread.
                end = len(self.token_ids) - (len(self.token_ids) % chunk_size)
            assert end % chunk_size == 0, (
                f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
            )
            end_chunk = end // chunk_size
            self._compute_hash(end_chunk)
            return self.chunk_hashes[start_chunk:end_chunk]

    def _compute_hash(self, end_chunk: int) -> None:
        """Compute rolling hashes up to end_chunk.

        Uses cached state to skip already-computed chunks.

        Args:
            end_chunk: Compute hashes up to (but not including) this chunk.
        """
        chunk_size = self.hasher.chunk_size

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * chunk_size
            ce = cs + chunk_size
            chunk = self.token_ids[cs:ce]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else self.hasher.none_hash
            )
            h = self.hasher.hash_tokens(chunk, prefix)
            self.last_prefix_hash = h
            self.chunk_hashes.append(h)
            self.num_chunks_processed += 1

    def begin_lookup(
        self,
        key: IPCCacheServerKey,
        group_windows: tuple[int, ...],
    ) -> bool:
        """Admit a lookup once; tagged generations require separate sessions."""
        with self._lock:
            if self._retired:
                raise RuntimeError("Cannot begin lookup on a retired session")
            previous = self.lookup_ipc_key
            if previous is not None and (previous.lookup_id or key.lookup_id):
                if (previous.lookup_id == key.lookup_id and previous == key
                        and previous.num_kv_readers == key.num_kv_readers
                        and previous.lookup_worker_ids == key.lookup_worker_ids
                        and previous.request_configs == key.request_configs):
                    return False
                raise RuntimeError("End the existing lookup session before changing generation or identity")
            if self._read_leases is not None:
                raise RuntimeError("Owned lookup requires a separate session generation")
            self.lookup_ipc_key = key
            self.prefetch_hit_chunks = -1
            self.prefetch_locked_gids = ()
            self.prefetch_group_windows = group_windows
            self._lookup_generation += 1
            self._failed_retrieve_releases.clear()
            self._scheduler_released_objects.clear()
            self._worker_object_claims.clear()
            self._retrieve_results.clear()
            return True

    def prepare_scheduler_release(self, key: IPCCacheServerKey):
        """Snapshot a completed scheduler lookup without inventing ownership."""
        with self._lock:
            if not self._matches_scheduler_release(key):
                return None
            return (self.prefetch_hit_chunks, self.prefetch_locked_gids,
                    self.prefetch_group_windows, self._lookup_generation)

    def _matches_scheduler_release(self, key: IPCCacheServerKey) -> bool:
        """Caller holds _lock; worker transfer releases use their own protocol."""
        lookup = self.lookup_ipc_key
        return bool(
            not self._retired and lookup is not None and key.worker_id is None
            and self.prefetch_hit_chunks >= 0
            and key.request_id == self.request_id
            and key.model_name == lookup.model_name
            and key.world_size == lookup.world_size
            and key.token_ids == lookup.token_ids
            and key.cache_salt == lookup.cache_salt
            and key.lookup_id == lookup.lookup_id
            and key.lookup_worker_ids == lookup.lookup_worker_ids
            and key.num_kv_readers == lookup.num_kv_readers
            and key.start >= lookup.start and key.end <= lookup.end
        )

    def claim_scheduler_release(self, key: IPCCacheServerKey, generation: int,
                                object_keys: list) -> list:
        """Claim each object once, including partially overlapping ranges.

        Claims precede storage mutation. A partial storage failure must not be
        blindly retried; retain the claim and let the failure remain visible.
        """
        with self._lock:
            if generation != self._lookup_generation or not self._matches_scheduler_release(key):
                return []
            claimed = []
            for obj in object_keys:
                if (obj not in self._scheduler_released_objects
                        and not self._worker_object_claims.get(obj)):
                    self._scheduler_released_objects.add(obj)
                    claimed.append(obj)
            return claimed

    def claim_worker_read(self, key: IPCCacheServerKey, instance_id: int,
                          generation: int, object_keys: list) -> bool:
        """Reserve one reader share per object before a tagged transfer.

        This is a lifecycle claim, not an extra storage read lock. Keep it after
        completion so a delayed scheduler FREE cannot release that share again.
        The caller must pair duplicate claims with the original completion;
        False alone is not a safe substitute for an in-flight GPU event.
        """
        state = self.prepare_failed_retrieve_release(key)
        if state is None or state[3] != generation or not key.lookup_id:
            return False
        keys = set(object_keys)
        with self._lock:
            if self._retired or generation != self._lookup_generation:
                return False
            if self._read_leases is not None and not keys <= self._read_leases.keys():
                return False
            readers = self.lookup_ipc_key.num_kv_readers
            for obj in keys:
                owners = self._worker_object_claims.get(obj, set())
                if (obj in self._scheduler_released_objects
                        or instance_id in owners or len(owners) >= readers):
                    return False
            for obj in keys:
                self._worker_object_claims.setdefault(obj, set()).add(instance_id)
            return True

    def begin_retrieve(self, key: IPCCacheServerKey, instance_id: int,
                       signature: tuple) -> tuple[bool, Future]:
        """Elect one submitter and share its actual IPC completion result.

        Caller retains this Session until the elected operation publishes its
        result. Completion means the RPC has an event handle, not that GPU DMA
        has finished; every duplicate must still wait on that same event.
        """
        operation = (key.lookup_id, instance_id, key.worker_id, key.start, key.end)
        state = self.prepare_failed_retrieve_release(key)
        with self._lock:
            existing = self._retrieve_results.get(operation)
            if existing is not None:
                old_signature, future = existing
                if old_signature != signature:
                    raise ValueError("Duplicate RETRIEVE changes transfer parameters")
                return False, future
            if self._retired or state is None or state[3] != self._lookup_generation:
                raise ValueError("RETRIEVE no longer owns this lookup generation")
            future = Future()
            result = (True, future)
            entry = (signature, future)
            reservation = None
            if self._budget_account is not None:
                reservation = self._budget_account.reserve(Usage(
                    metadata_bytes=sys.getsizeof(future)+sys.getsizeof(operation)+sys.getsizeof(signature),
                    operations=1))
            try:
                if reservation is not None:
                    self._operation_reservations[operation] = reservation
                self._retrieve_results[operation] = entry
            except BaseException:
                self._operation_reservations.pop(operation, None)
                if reservation is not None:
                    reservation.release()
                raise
            return result

    def find_retrieve(self, key: IPCCacheServerKey, instance_id: int, signature: tuple):
        operation = (key.lookup_id, instance_id, key.worker_id, key.start, key.end)
        with self._lock:
            existing = self._retrieve_results.get(operation)
            if existing is None:
                return None
            if existing[0] != signature:
                raise ValueError("Duplicate RETRIEVE changes transfer parameters")
            return existing[1]

    def retain_completion_event(self, backend, event, device) -> None:
        with self._lock:
            self._completion_events.append((backend, event, device))

    def reserve_direct_submission(self, completion, backend, event, device, payload_bytes):
        """Reserve before enqueue; host receipt and device event both pin retirement."""
        with self._lock:
            if self._retired:
                raise ValueError("Cannot submit direct work on a retired session")
            if self._budget_account is None:
                raise ValueError("Direct admission requires a resource account")
            entry = (backend, event, device)
            external_size = len(self._external_completions)
            event_size = len(self._completion_events)
            reservation = self._budget_account.reserve(Usage(
                metadata_bytes=sys.getsizeof(completion)+sys.getsizeof(entry),
                payload_bytes=payload_bytes, operations=1))
            try:
                self._operation_reservations[completion] = reservation
                self._external_completions.append(completion)
                self._completion_events.append(entry)
            except BaseException:
                del self._external_completions[external_size:]
                del self._completion_events[event_size:]
                self._operation_reservations.pop(completion, None)
                reservation.release()
                raise

    def retain_external_completion(self, completion: Future) -> None:
        """Keep ownership while an external SHM consumer may access the buffer."""
        with self._lock:
            self._external_completions.append(completion)

    def completions_finished(self) -> bool:
        with self._lock:
            if any(not future.done() or future.cancelled()
                   or future.exception() is not None
                   for future in self._external_completions):
                return False
            # Cancellation describes the RPC observer, not device completion.
            # Future.exception() raises on cancelled futures; letting that
            # escape would also stop reclamation of unrelated retired sessions.
            if any(not future.done() or future.cancelled()
                   for _, future in self._retrieve_results.values()):
                return False
            if any(future.exception() is not None for _, future in self._retrieve_results.values()):
                return False
            events = list(self._completion_events)
        try:
            return all(backend.query_event(event) for backend, event, _ in events)
        except Exception:
            logger.exception("Cannot confirm device completion for retired session %s", self.request_id)
            return False

    def compact_retired_payload(self) -> bool:
        """Release completed data occupancy while retaining replay metadata.

        Retired sessions cannot submit new work. Completion and read-release
        checks are therefore monotonic once true. Keep uncertain/cancelled
        device work charged, and keep operation records for duplicate replies.
        """
        with self._lock:
            if not self._retired:
                return False
        if not self.completions_finished() or not self.owned_reads_released():
            return False
        with self._lock:
            for reservation in self._operation_reservations.values():
                reservation.release_payload()
        return True

    def compact_retired_claims(self) -> None:
        """Discard per-object ledgers after retirement and confirmed completion."""
        with self._lock:
            if self._retired:
                # Retirement planning needs the original worker share counts.
                if self._read_leases is not None and self._retired_unhanded_plan is None:
                    return
                self._worker_object_claims.clear()
                self._scheduler_released_objects.clear()
                self._failed_retrieve_releases.clear()

    def begin_owned_submission(self, key, *, payload_bytes=0):
        """Pin session metadata while storage may acquire unreported leases."""
        with self._lock:
            if self._retired or not key.lookup_id or self.lookup_ipc_key != key:
                return None
            completion = Future()
            reservation = None
            if self._budget_account is not None:
                reservation = self._budget_account.reserve(Usage(
                    metadata_bytes=sys.getsizeof(completion),
                    payload_bytes=payload_bytes, operations=1))
            try:
                if reservation is not None:
                    self._operation_reservations[completion] = reservation
                self._external_completions.append(completion)
            except BaseException:
                self._operation_reservations.pop(completion, None)
                if reservation is not None:
                    reservation.release()
                raise
            return completion

    def adopt_failed_submission(self, key, leases, release_check, cleanup_retry):
        """Take ownership of unsubmitted reads, including after a racing END.

        Caller must hold an unfinished begin_owned_submission completion so
        retirement cannot discard this session before adoption is complete.
        No hit count is published and no worker can consume these leases.
        """
        with self._lock:
            if self.lookup_ipc_key != key or self._read_leases is not None:
                raise RuntimeError("Failed submission does not match unpublished lookup")
            self._read_leases = dict(leases)
            self._read_release_check = release_check
            self._read_cleanup_retry = cleanup_retry
            self._retired_unhanded_plan = None

    def record_owned_prefetch_result(self, key, hit_chunks, locked_gids, leases, release_check=None, cleanup_retry=None) -> bool:
        """Publish completed prefetch ownership before exposing its hit count.

        A racing retirement wins under the same lock: the caller must retain
        and clean up its prefetch handle if this returns False.
        """
        with self._lock:
            lookup = self.lookup_ipc_key
            if (self._retired or lookup is None or not key.lookup_id
                    or key.lookup_id != lookup.lookup_id
                    or key.request_id != self.request_id or key != lookup):
                return False
            if self._read_leases is not None:
                raise RuntimeError("Owned prefetch result already published")
            self._read_leases = dict(leases)
            self._read_release_check = release_check
            self._read_cleanup_retry = cleanup_retry
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids
            return True

    def retry_retired_read_cleanup(self) -> None:
        """Retry only exact unhanded reservations, with bounded retry frequency."""
        now = time.monotonic()
        with self._lock:
            callback = self._read_cleanup_retry
            if not self._retired or callback is None or now < self._read_retry_after:
                return
            self._read_retry_after = now + self._read_retry_delay
            self._read_retry_delay = min(60.0, self._read_retry_delay * 2)
        try:
            callback()
        except Exception:
            logger.exception("Retired read cleanup retry failed for %s", self.request_id)

    def owned_reads_released(self) -> bool:
        """Do not discard ownership merely because RPC/device completion is done."""
        with self._lock:
            if not self._read_leases:
                return True
            leases = dict(self._read_leases)
            check = self._read_release_check
        if check is None:
            return False
        try:
            return not check(leases)
        except Exception:
            logger.exception("Cannot confirm owned read release for %s", self.request_id)
            return False

    def read_lease_snapshot(self, object_keys):
        """Keep exact tokens available to an already admitted worker after END."""
        with self._lock:
            if self._read_leases is None:
                return None
            return {obj: self._read_leases[obj] for obj in object_keys}

    def completed_worker_read_plan(self) -> tuple:
        """Recover failed release callbacks only after all consumers complete.

        Retirement prevents new admissions. Every admitted operation must have
        a completed result and all retained device/external fences must pass.
        Unknown or failed completion evidence leaves the buffers pinned.
        """
        with self._lock:
            if not self._retired or self._read_leases is None or not self._retrieve_results:
                return ()
        if not self.completions_finished():
            return ()
        with self._lock:
            return tuple((obj, self._read_leases[obj], instance)
                         for obj, owners in self._worker_object_claims.items()
                         if obj in self._read_leases for instance in owners)

    def retired_unhanded_read_plan(self) -> tuple:
        """Replayable release plan excluding every share handed to a worker.

        The caller releases using each exact token and a stable claim id;
        repeating this plan cannot decrement another reservation. Worker
        completion is independent and may happen after session retirement.
        """
        with self._lock:
            if not self._retired:
                raise RuntimeError("Unhanded retirement requires a retired session")
            if self._retired_unhanded_plan is not None:
                return self._retired_unhanded_plan
            if self._read_leases is None:
                return ()
            readers = self.lookup_ipc_key.num_kv_readers
            plan = []
            for obj, token in self._read_leases.items():
                remaining = readers - len(self._worker_object_claims.get(obj, ()))
                if remaining > 0:
                    plan.append((obj, token, remaining))
            self._retired_unhanded_plan = tuple(plan)
            return self._retired_unhanded_plan

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
    ) -> None:
        """Record the lock set acquired by the current lookup."""
        with self._lock:
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids

    def prepare_failed_retrieve_release(
        self,
        key: IPCCacheServerKey,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Return a stable snapshot for a failed worker's lock release."""
        if key.worker_id is None:
            return None

        with self._lock:
            lookup_key = self.lookup_ipc_key
            hit_chunks = self.prefetch_hit_chunks
            if self._retired or lookup_key is None or hit_chunks < 0:
                return None

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.request_id == self.request_id
                and key.world_size == lookup_key.world_size
                and key.token_ids == lookup_key.token_ids
                and key.cache_salt == lookup_key.cache_salt
                and key.lookup_id == lookup_key.lookup_id
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
                and 0 <= key.worker_id < lookup_key.world_size
                and (lookup_key.lookup_worker_ids is None
                     or key.worker_id in lookup_key.lookup_worker_ids)
                and (not key.lookup_id or key.num_kv_readers == lookup_key.num_kv_readers)
            )
            if not same_lookup:
                return None

            return (
                hit_chunks,
                self.prefetch_locked_gids,
                self.prefetch_group_windows,
                self._lookup_generation,
            )

    def claim_failed_retrieve_release(
        self,
        instance_id: int,
        key: IPCCacheServerKey,
        lookup_generation: int,
    ) -> bool:
        """Atomically claim one failed worker's prepared lock release.

        A scheduler lookup acquires read locks for every KV worker (or one
        reader count per TP worker for MLA), while RETRIEVE responses are
        per worker instance.  When a worker has lost its GPU registration,
        only that instance's share may be released.  Claiming after key
        resolution makes duplicate failed RETRIEVEs idempotent without
        consuming the claim when resolution itself fails.

        ``False`` is also returned when the session cannot prove ownership of
        the requested range.  In that case it is safer to leave the lock to
        its TTL than to decrement a concurrent request's anonymous L1 count.
        """
        if key.worker_id is None:
            return False

        with self._lock:
            lookup_key = self.lookup_ipc_key
            if lookup_key is None or lookup_generation != self._lookup_generation:
                return False

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_ids == lookup_key.token_ids
                and key.cache_salt == lookup_key.cache_salt
                and key.lookup_id == lookup_key.lookup_id
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return False

            owner = (
                lookup_generation,
                instance_id,
                key.worker_id,
                key.start,
                key.end,
            )
            if owner in self._failed_retrieve_releases:
                return False
            self._failed_retrieve_releases.add(owner)
            return True


class SessionCapacityError(RuntimeError):
    """Admission rejected while live or replay-protected sessions fill the budget."""


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes
    DEFAULT_REPLAY_TTL = 1800  # Covers the deployed 900-second RPC budget.
    DEFAULT_CLEANUP_INTERVAL = 60.0

    def __init__(
        self,
        hasher: TokenHasher,
        ttl: float = DEFAULT_SESSION_TTL,
        cleanup_interval: float | None = DEFAULT_CLEANUP_INTERVAL,
        replay_ttl: float | None = None,
        max_sessions: int = 65536,
        resource_budget=None,
        request_limits=None,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        self._resource_budget = resource_budget
        self.request_limits = request_limits
        self._max_sessions = max_sessions
        self._admission_rejections = 0
        self._hasher = hasher
        self._ttl = ttl
        self._replay_ttl = max(ttl, self.DEFAULT_REPLAY_TTL) if replay_ttl is None else replay_ttl
        if self._replay_ttl < 0:
            raise ValueError("replay_ttl must be nonnegative")
        self._sessions: dict[str, Session] = {}
        self._retired_sessions: dict[tuple[str, str], tuple[Session, float]] = {}
        self._lock = threading.Lock()
        #: Callbacks invoked for every destroyed session, so owners of
        #: per-request state (e.g. entries in ``Session.extras``) can release
        #: it however the request ended. See :meth:`_notify_destroyed`.
        self.destroy_listeners: list[Callable[[Session], None]] = []
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: PeriodicThread | None = None
        if cleanup_interval is not None and cleanup_interval > 0:
            self._cleanup_thread = self._create_cleanup_thread(cleanup_interval)
            self._cleanup_thread.start()

    def _check_admission_locked(self) -> None:
        # Moving active -> retired never frees a slot. Completion and replay
        # expiry together determine when its slot becomes reusable.
        if len(self._sessions) + len(self._retired_sessions) >= self._max_sessions:
            self._admission_rejections += 1
            raise SessionCapacityError("Session ownership budget exhausted")

    def admission_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"active": len(self._sessions), "retired": len(self._retired_sessions),
                    "limit": self._max_sessions, "rejections": self._admission_rejections}

    def get_or_create(self, request_id: str) -> Session:
        """Get existing session or create a new one.

        Args:
            request_id: Unique request identifier.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            if request_id not in self._sessions:
                self._check_admission_locked()
                self._sessions[request_id] = Session(
                    request_id=request_id, hasher=self._hasher,
                    _budget_account=self._resource_budget.account() if self._resource_budget else None
                )
                logger.debug("Created session for request_id=%s", request_id)
            return self._sessions[request_id]

    def get(self, request_id: str) -> Optional[Session]:
        """Return an existing session without creating ownership state."""
        with self._lock:
            return self._sessions.get(request_id)

    def get_or_create_lookup(self, key: IPCCacheServerKey) -> Session:
        """Reject retired tagged LOOKUP replays during the retention window."""
        with self._lock:
            if key.lookup_id and (key.request_id, key.lookup_id) in self._retired_sessions:
                raise ValueError("LOOKUP generation has already ended")
            if key.request_id not in self._sessions:
                self._check_admission_locked()
                self._sessions[key.request_id] = Session(key.request_id, self._hasher, _budget_account=self._resource_budget.account() if self._resource_budget else None)
            return self._sessions[key.request_id]

    def get_for_replay(self, request_id: str, lookup_id: str) -> Optional[Session]:
        with self._lock:
            current = self._sessions.get(request_id)
            if current is not None and current.lookup_ipc_key is not None:
                if current.lookup_ipc_key.lookup_id == lookup_id:
                    return current
            retired = self._retired_sessions.get((request_id, lookup_id))
            return retired[0] if retired is not None else None

    def remove(self, request_id: str) -> Optional[Session]:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            if request_id not in self._sessions:
                return None
            session = self._sessions[request_id]
            with session._lock:
                session._retired = True
                del self._sessions[request_id]
                self._retain_retired_locked(session)
            logger.debug("Removed session for request_id=%s", request_id)
        self._notify_destroyed(session)
        return session

    def remove_if_lookup(self, request_id: str, lookup_id: str) -> Optional[Session]:
        """Compare lookup ownership and unlink under manager/session locks.

        An old caller holding the detached Session may not begin another lookup
        on it. Listeners run outside both locks, and receive this exact session.
        """
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None:
                return None
            with session._lock:
                key = session.lookup_ipc_key
                current = key.lookup_id if key is not None else ""
                if current != lookup_id:
                    return None
                session._retired = True
                del self._sessions[request_id]
                self._retain_retired_locked(session)
        self._notify_destroyed(session)
        return session

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired: list[Session] = []
        with self._lock:
            for session in self._sessions.values():
                if now - session.created_at > self._ttl:
                    expired.append(session)
            for session in expired:
                with session._lock:
                    session._retired = True
                    del self._sessions[session.request_id]
                    self._retain_retired_locked(session)

        for session in expired:
            self._notify_destroyed(session)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        self.cleanup_retired()
        return len(expired)

    def _retain_retired_locked(self, session: Session) -> None:
        key = session.lookup_ipc_key
        if key is not None and key.lookup_id:
            self._retired_sessions.setdefault((session.request_id, key.lookup_id),
                                             (session, time.monotonic()))

        elif session._budget_account is not None:
            # Untagged generations need distinct retirement identities too.
            self._retired_sessions.setdefault((session.request_id, id(session)),
                                             (session, time.monotonic()))

    @property
    def replay_ttl_seconds(self) -> float:
        return self._replay_ttl

    def cleanup_retired(self) -> int:
        """Keep replay state for its retention window and device completion."""
        now = time.monotonic()
        with self._lock:
            candidates = list(self._retired_sessions.items())
        removed = 0
        for key, entry in candidates:
            session, retired_at = entry
            if not session.owned_reads_released():
                session.retry_retired_read_cleanup()
            if not session.completions_finished() or not session.owned_reads_released():
                continue
            session.compact_retired_claims()
            session.compact_retired_payload()
            if now - retired_at < self._replay_ttl:
                continue
            with self._lock:
                if self._retired_sessions.get(key) is entry:
                    session._release_resource_budget()
                    del self._retired_sessions[key]
                    removed += 1
        return removed

    def _notify_destroyed(self, session: Session) -> None:
        """Run every destroy listener for one removed session.

        Called without the manager lock held: a listener may take other locks
        (e.g. the storage manager's). A listener that raises is logged and
        skipped so destruction always completes.

        Args:
            session: The session that was just removed.
        """
        for listener in list(self.destroy_listeners):
            try:
                listener(session)
            except Exception:
                logger.exception(
                    "Session destroy listener failed for request_id=%s",
                    session.request_id,
                )
        # Normal END should not hold completed payload until either the next
        # periodic sweep or replay expiry. Listeners have first released any
        # unhanded reads. Unfinished work remains charged for the later sweep.
        try:
            session.compact_retired_payload()
        except Exception:
            logger.exception("Cannot compact retired payload for %s", session.request_id)

    def active_count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Number of currently tracked sessions.
        """
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        """Stop the background cleanup thread and unregister it."""
        if self._cleanup_thread is None:
            return

        PeriodicThreadRegistry.get_instance().unregister(self._cleanup_thread.name)
        self._cleanup_thread.stop()
        self._cleanup_thread = None

    def _create_cleanup_thread(self, cleanup_interval: float) -> PeriodicThread:
        def execute_cleanup() -> ThreadRunSummary:
            removed = self.cleanup_expired()
            return ThreadRunSummary(
                success=True,
                message=f"Removed {removed} expired sessions",
                extra_info={"removed_sessions": str(removed)},
            )

        return create_periodic_thread(
            name=f"SessionManager-cleanup-thread-{id(self):x}",
            interval=cleanup_interval,
            execute_fn=execute_cleanup,
            level=ThreadLevel.MEDIUM,
        )
