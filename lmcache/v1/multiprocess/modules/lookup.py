# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass, field, replace
from concurrent.futures import Future
from functools import partial
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_ranked
from lmcache.v1.distributed.storage_manager import UnsubmittedPrefetchError
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, LookupSessionRef
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


def planned_payload_bytes(object_layouts):
    """Logical uncompressed bytes, counted once per selected cache object.

    Python integer multiplication cannot wrap. This is a work reservation,
    not physical memory usage or compressed transfer bytes.
    """
    total = 0
    for layout in object_layouts.values():
        if len(layout.shapes) != len(layout.dtypes):
            raise ValueError("Mismatched layout shapes and dtypes")
        for shape, dtype in zip(layout.shapes, layout.dtypes):
            elements = 1
            for dim in shape:
                if type(dim) is not int or dim < 0:
                    raise ValueError("Invalid layout dimension")
                elements *= dim
            itemsize = dtype.itemsize
            if type(itemsize) is not int or itemsize <= 0:
                raise ValueError("Invalid layout dtype itemsize")
            total += elements * itemsize
    return total


def resolve_prefetched_obj_keys(
    ctx: MPCacheServerContext,
    key: IPCCacheServerKey,
    hit_chunks: int,
    locked_gids: tuple,
    group_windows: tuple[int, ...] | None = None,
) -> list[ObjectKey]:
    """Resolve the subset of a request range that lookup actually locked.

    ``key.worker_id=None`` resolves every KV rank for scheduler-owned cleanup.
    A worker-specific key resolves only that worker's shard (or one MLA reader
    share), which is required for per-instance RETRIEVE failure cleanup.
    """
    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(
        list(key.token_ids), start=key.start, end=key.end
    )
    if not chunk_hashes:
        return []

    start_chunk = key.start // ctx.chunk_size
    end_chunk = start_chunk + len(chunk_hashes)
    if group_windows is None:
        group_windows = tuple(
            ctx.layout_desc_registry.find_attn_desc(
                key.model_name, key.world_size
            ).num_chunks_in_sw
        )

    obj_keys: list[ObjectKey] = []
    for group_idx, window in enumerate(group_windows):
        if locked_gids and group_idx not in locked_gids:
            continue
        if hit_chunks < 0:
            if window >= 0:
                continue
            lo, hi = start_chunk, end_chunk
        else:
            # Locked range per ``unfold``: the whole hit prefix for full
            # attention, its trailing ``window`` chunks otherwise.
            lo = 0 if window < 0 else max(0, hit_chunks - window)
            lo = max(lo, start_chunk)
            hi = min(hit_chunks, end_chunk)
        if lo >= hi:
            continue
        group_hashes = chunk_hashes[lo - start_chunk : hi - start_chunk]
        obj_keys.extend(ipc_key_to_object_keys(key, group_hashes, [group_idx])[0])
    return obj_keys


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 on the early-exit paths (see
    # ``early_exit_reason``).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    num_object_groups: int = 1
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""
    # Names the ``lookup()`` branch that returned before submitting a prefetch
    # task; empty on the normal path.
    early_exit_reason: str = ""
    object_keys: tuple[ObjectKey, ...] = ()
    num_kv_readers: int = 1
    guard: threading.Lock = field(default_factory=threading.Lock, repr=False)
    submission_completion: Future | None = field(default=None, repr=False)
    # Retain destructive query results until handoff or cancellation finishes.
    completed_result: object | None = field(default=None, repr=False)
    handoff_completed: bool = False
    cleanup_failed: bool = False
    cleanup_keys: tuple[ObjectKey, ...] | None = None
    cleanup_retry_after: float = 0.0
    cleanup_retry_delay: float = 0.1
    lookup_id: str = ""


class LookupModule:
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS,
    QUERY_PREFETCH_LOOKUP_HITS, FREE_LOOKUP_LOCKS, and END_SESSION
    request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        self._cancelled_jobs: dict[int, _PrefetchJob] = {}
        self._cleanup_condition = threading.Condition(self._prefetch_job_lock)
        self._cleanup_thread: threading.Thread | None = None
        self._closed = False
        ctx.session_manager.destroy_listeners.append(self._session_destroyed)
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        admission = self._ctx.session_manager.admission_snapshot()
        with self._prefetch_job_lock:
            return {
                **{f"sessions_{name}": value for name, value in admission.items()},
                "active_prefetch_jobs": len(self._prefetch_jobs) + len(self._cancelled_jobs),
                "cancelled_prefetch_jobs": len(self._cancelled_jobs),
                "failed_prefetch_cleanups": sum(job.cleanup_failed for job in self._cancelled_jobs.values()),
            }

    def close(self) -> None:
        """Stop cleanup before context storage closes; never free pending DMA."""
        listeners = self._ctx.session_manager.destroy_listeners
        if self._session_destroyed in listeners:
            listeners.remove(self._session_destroyed)
        with self._prefetch_job_lock:
            request_ids = list(self._prefetch_jobs)
        for request_id in request_ids:
            self._cancel_prefetch(request_id)
        with self._cleanup_condition:
            self._closed = True
            self._cleanup_condition.notify()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join()

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    def lookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        limits = self._ctx.session_manager.request_limits
        if limits is not None:
            limits.validate_key(key)
            limits.validate_expansion(
                key, self._ctx.chunk_size,
                self._ctx.layout_desc_registry.find_attn_desc(
                    key.model_name, key.world_size
                ).num_object_groups,
            )
        model_name, world_size = key.model_name, key.world_size
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        worker_layouts = self._ctx.layout_desc_registry.find_worker_group_layout_descs(
            model_name, world_size
        )
        missing_workers = (
            set(key.lookup_worker_ids) - worker_layouts.keys()
            if key.lookup_worker_ids is not None
            else set()
        )
        layout_desc = self._ctx.layout_desc_registry.find(model_name, world_size)
        if layout_desc is None or missing_workers:
            logger.error(
                "Incomplete KV layout registration for model %s, world size %d, "
                "missing PP workers=%s",
                model_name,
                world_size,
                sorted(missing_workers),
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_found_indices=(),
                        l1_hit_chunks=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    lookup_id=key.lookup_id,
                    early_exit_reason=("missing_pp_layout" if missing_workers else "no_gpu_context"),
                )
            )
            return

        num_kv_readers = key.require_num_kv_readers()

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_found_indices=(),
                        l1_hit_chunks=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    lookup_id=key.lookup_id,
                    early_exit_reason="empty_chunk_hashes",
                )
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._ctx.event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # Lay keys out chunk-major across object groups (see
        # _chunk_major_object_keys); pass the windows to the prefetch policy.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        attn_desc = replace(attn_desc, world_size=key.lookup_world_size)
        session = self._ctx.session_manager.get_or_create_lookup(key)
        if not session.begin_lookup(key, tuple(attn_desc.num_chunks_in_sw)):
            return
        session.set_tokens(list(key.token_ids))
        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)

        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(
            model_name, world_size
        )
        if not group_layout_descs:
            logger.error(
                "No group layout descs found for model %s with world size %d "
                "during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_found_indices=(),
                        l1_hit_chunks=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    lookup_id=key.lookup_id,
                    early_exit_reason="no_group_layout_descs",
                )
            )
            return

        ranked_layouts = {
            ObjectKey.ComputeKVRank(world_size, worker, world_size, worker): layouts
            for worker, layouts in worker_layouts.items()
        }
        object_layouts = {
            obj: ranked_layouts.get(obj.kv_rank, group_layout_descs)[obj.object_group_id]
            for obj in obj_keys
        }
        spec = PrefetchRequestSpec(
            keys=obj_keys,
            group_layout_descs=group_layout_descs,
            object_layout_descs=object_layouts,
            read_leases={} if key.lookup_id else None,
            num_kv_readers=num_kv_readers,
            attn_desc=attn_desc,
        )
        # Allocate the task, guard and key snapshot before storage can accept
        # asynchronous work. The placeholder handle never escapes this method.
        job = _PrefetchJob(
            handle=PrefetchHandle(-1, key.request_id, (), 0, len(obj_keys), time.monotonic()),
            world_size=key.lookup_world_size,
            request_id=key.request_id,
            requested_tokens=requested_tokens,
            num_object_groups=attn_desc.num_object_groups,
            attn_desc=attn_desc,
            model_name=model_name,
            cache_salt=key.cache_salt,
            lookup_id=key.lookup_id,
            object_keys=tuple(obj_keys),
            num_kv_readers=num_kv_readers,
        )
        # Keep accounting optional until all direct/legacy paths and config
        # have explicit admission coverage. Reserve before native submission.
        payload_bytes = (
            planned_payload_bytes(object_layouts)
            if session._budget_account is not None else 0
        )
        submission = session.begin_owned_submission(
            key, payload_bytes=payload_bytes
        ) if key.lookup_id else None
        if key.lookup_id and submission is None:
            return
        job.submission_completion = submission
        try:
            handle = self._ctx.storage_manager.submit_prefetch_task(
                spec, external_request_id=key.request_id,
            )
        except UnsubmittedPrefetchError:
            if submission is not None:
                try:
                    session.adopt_failed_submission(
                        key, spec.read_leases,
                        self._ctx.storage_manager.owned_reads_pending,
                        lambda: self._release_retired_reads(session),
                    )
                    self._ctx.session_manager.remove_if_lookup(key.request_id, key.lookup_id)
                finally:
                    submission.set_result(None)
            raise
        # Successful submission stays pinned until ownership is published to
        # its session or cancelled cleanup is confirmed. Unknown exceptions
        # keep the admission slot rather than assume asynchronous work stopped.
        job.handle = handle
        try:
            self._register_prefetch_job(job)
        except Exception:
            # Registration may fail before or after publication, or race with
            # query/retirement. Serialize with result handoff and retain the
            # exact accepted task until asynchronous completion is confirmed.
            with job.guard:
                if not job.handoff_completed and (submission is None or not submission.done()):
                    self._queue_cancelled_job(job, detached=True)
            self._ctx.session_manager.remove_if_lookup(key.request_id, key.lookup_id)
            raise

    def query_prefetch_lookup_hits(
        self,
        request_id: str | LookupSessionRef,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        ref = request_id
        expected_lookup_id = ref.lookup_id if isinstance(ref, LookupSessionRef) else ""
        request_id = ref.request_id if isinstance(ref, LookupSessionRef) else ref
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None or job.lookup_id != expected_lookup_id:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        # Result is already in chunk-level units (l1_hit_chunks + l2_hit_chunks).
        return self._ctx.storage_manager.query_prefetch_lookup_hits(job.handle)

    def query_prefetch_status(
        self,
        request_id: str | LookupSessionRef,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        ref = request_id
        expected_lookup_id = ref.lookup_id if isinstance(ref, LookupSessionRef) else ""
        request_id = ref.request_id if isinstance(ref, LookupSessionRef) else ref
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None or job.lookup_id != expected_lookup_id:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        with job.guard:
            with self._prefetch_job_lock:
                if self._prefetch_jobs.get(request_id) is not job:
                    return 0
            # A TTL cleanup may already have removed the session. Never
            # resurrect it merely because a result arrived late.
            session = self._ctx.session_manager.get(request_id)
            if session is None and job.object_keys:
                self._queue_cancelled_job(job)
                return 0
            return self._consume_prefetch_status(job, session)

    def _query_job_result(self, job):
        if job.completed_result is None:
            job.completed_result = self._ctx.storage_manager.query_prefetch_status(job.handle)
        return job.completed_result

    def _consume_prefetch_status(self, job, session) -> int | None:
        request_id = job.request_id
        found = self._query_job_result(job)
        if found is None:
            return None

        stride = job.attn_desc.num_object_groups * job.world_size
        num_chunks = job.handle.total_requested_keys // stride
        found_count, _retain = fold_unfold_ranked(
            found,
            num_chunks,
            job.world_size,
            job.attn_desc.num_chunks_in_sw,
        )

        # Record the model-wide hit length on the session so a later
        # free_lookup_locks can reconstruct which keys the prefetch
        # read-locked (see ``unfold``: full-attention groups lock the whole
        # hit prefix, sliding-window groups only its in-window suffix).
        if session is not None:
            leases = job.handle.read_leases
            if leases is not None:
                lookup_key = session.lookup_ipc_key
                if (lookup_key is None or lookup_key.lookup_id != job.lookup_id
                        or not session.record_owned_prefetch_result(
                            lookup_key, found_count,
                            tuple(range(job.attn_desc.num_object_groups)), leases,
                            release_check=self._ctx.storage_manager.owned_reads_pending,
                            cleanup_retry=lambda: self._release_retired_reads(session))):
                    self._queue_cancelled_job(job)
                    return 0
            else:
                session.record_prefetch_result(
                    found_count,
                    tuple(range(job.attn_desc.num_object_groups)),
                )

        # Commit the ownership handoff before any metrics or logging. Once
        # the session owns these leases, observer failures must not leave a
        # pending submission or let a retry publish the same leases again.
        job.handoff_completed = True
        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)
        if job.submission_completion is not None:
            job.submission_completion.set_result(None)

        try:
            # L2 contributes only the prefix beyond the L1 hit boundary.
            l1_chunks = job.handle.l1_hit_chunks
            if l1_chunks > found_count:
                logger.error(
                    "L1 hit chunks exceed total hit chunks: l1=%d total=%d request=%s",
                    l1_chunks, found_count, request_id,
                )
                l1_chunks = found_count
            l2_chunks = found_count - l1_chunks
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP_PREFETCH_END,
                    session_id=job.request_id,
                    metadata={
                        "found_count": found_count,
                        "requested_tokens": job.requested_tokens,
                        "hit_tokens": found_count * self._ctx.chunk_size,
                        "l1_hit_tokens": l1_chunks * self._ctx.chunk_size,
                        "l2_hit_tokens": l2_chunks * self._ctx.chunk_size,
                        "early_exit_reason": job.early_exit_reason,
                        "model_name": job.model_name,
                        "cache_salt": job.cache_salt,
                    },
                )
            )
        except Exception:
            try:
                logger.exception("Prefetch completion event failed: %s", request_id)
            except Exception:
                # A failed log sink cannot reverse a committed handoff.
                pass

        return found_count

    def wait_prefetch_status(
        self,
        request_id: str | LookupSessionRef,
        timeout: float,
    ) -> int | None:
        """Block until a prefetch job completes, then return its chunk count.

        Like query_prefetch_status, but waits for the daemon to publish the
        result instead of returning None while the prefetch is still in
        progress, so the caller does not have to busy-poll. The job entry is
        removed once a non-None result is returned (exactly-once semantics).

        Args:
            request_id: The external request ID passed in the lookup key.
            timeout: Maximum number of seconds to wait for the prefetch.

        Returns:
            Chunk count (int) when done, None if the wait timed out, 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        ref = request_id
        expected_lookup_id = ref.lookup_id if isinstance(ref, LookupSessionRef) else ""
        request_id = ref.request_id if isinstance(ref, LookupSessionRef) else ref
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None or job.lookup_id != expected_lookup_id:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if not self._ctx.storage_manager.wait_prefetch_status(job.handle, timeout):
            return None
        return self.query_prefetch_status(ref)

    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Only the keys the prefetch actually read-locked are released.

        Releases the same per-object count the lookup reserved
        (``key.num_kv_readers``).

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        if key.start >= key.end:
            return

        session = self._ctx.session_manager.get(key.request_id)
        if session is None:
            return
        state = session.prepare_scheduler_release(key)
        if state is None:
            return
        hit_chunks, locked_gids, group_windows, generation = state
        obj_keys = resolve_prefetched_obj_keys(
            self._ctx, key, hit_chunks, locked_gids, group_windows
        )
        obj_keys = session.claim_scheduler_release(key, generation, obj_keys)

        if not obj_keys:
            return

        leases = session.read_lease_snapshot(obj_keys)
        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys, read_locks=key.require_num_kv_readers(),
            **({"leases": leases, "claim_id": "scheduler-unhanded"}
               if leases is not None else {})
        )

    def end_session(self, request_id: str | LookupSessionRef) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        ref = request_id
        expected_lookup_id = ref.lookup_id if isinstance(ref, LookupSessionRef) else ""
        request_id = ref.request_id if isinstance(ref, LookupSessionRef) else ref
        session = self._ctx.session_manager.remove_if_lookup(request_id, expected_lookup_id)
        # Early-exit lookups may have a job without a session. Match the nonce
        # before cancelling; a delayed END must never touch a newer generation.
        self._cancel_prefetch(request_id, expected_lookup_id)
        if session is None:
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch",
                request_id,
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = self._chunk_major_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _chunk_major_object_keys(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> list[ObjectKey]:
        """Resolve the flat object-key list across all object groups,
        chunk-major.

        The object-group count is read from the layout registry for
        ``key``'s ``(model_name, world_size)``. The keys are ordered
        ``chunk -> object group -> kv_rank`` so that all keys belonging to one
        chunk are contiguous; a leading-ones prefix over the flat list then maps
        directly to a whole-chunk hit count. Callers that need the full key set
        regardless of order (lock release, touch) use this too.

        Example (2 chunks ``c0,c1``; 2 groups ``g0,g1``; 2 kv_ranks ``r0,r1``)::

            [c0g0r0, c0g0r1, c0g1r0, c0g1r1,   # chunk 0: all groups, all ranks
             c1g0r0, c1g0r1, c1g1r0, c1g1r1]   # chunk 1: ...

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.

        Returns:
            The chunk-major flattened list of object keys across all groups.
        """
        num_groups = self._ctx.layout_desc_registry.find_attn_desc(
            key.model_name, key.world_size
        ).num_object_groups
        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))
        if num_groups == 1:
            return per_group[0]
        # Each per-group list is chunk-major / rank-minor of length
        # len(chunk_hashes) * num_ranks; recover num_ranks to slice per chunk.
        num_ranks = len(per_group[0]) // len(chunk_hashes) if chunk_hashes else 0
        obj_keys: list[ObjectKey] = []
        for chunk_idx in range(len(chunk_hashes)):
            lo = chunk_idx * num_ranks
            hi = lo + num_ranks
            for group_keys in per_group:
                obj_keys.extend(group_keys[lo:hi])
        return obj_keys

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        session = self._ctx.session_manager.get(job.request_id)
        key = session.lookup_ipc_key if session is not None else None
        stale = bool(job.object_keys) and (key is None or key.lookup_id != job.lookup_id)
        with self._prefetch_job_lock:
            occupied = job.request_id in self._prefetch_jobs
            if not stale and not occupied:
                self._prefetch_jobs[job.request_id] = job
        if stale or occupied:
            # Storage may already hold reservations. Retire this exact task
            # without overwriting or cancelling the currently registered task.
            with job.guard:
                self._queue_cancelled_job(job, detached=True)
            return
        # Session expiry may race with storage submission, before this entry
        # was visible to the session destroy listener.
        current = self._ctx.session_manager.get(job.request_id)
        if job.object_keys and current is not session:
            self._cancel_prefetch(job.request_id, job.lookup_id)

    def _session_destroyed(self, session) -> None:
        key = session.lookup_ipc_key
        self._cancel_prefetch(session.request_id, key.lookup_id if key else "")
        self._release_retired_reads(session)

    def _release_retired_reads(self, session) -> None:
        for obj, token, count in session.retired_unhanded_read_plan():
            self._ctx.storage_manager.finish_read_prefetched(
                [obj], read_locks=count, leases={obj: token},
                claim_id="scheduler-unhanded",
            )

        for obj, token, instance in session.completed_worker_read_plan():
            self._ctx.storage_manager.finish_read_prefetched(
                [obj], leases={obj: token}, claim_id=f"worker:{instance}",
            )

    def _cancel_prefetch(self, request_id: str, expected_lookup_id: str | None = None) -> None:
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is not None:
            with job.guard:
                if expected_lookup_id is not None and job.lookup_id != expected_lookup_id:
                    return
                self._queue_cancelled_job(job)

    def _queue_cancelled_job(self, job, detached: bool = False) -> None:
        # Caller owns job.guard, excluding normal result consumption.
        with self._cleanup_condition:
            if not detached and self._prefetch_jobs.get(job.request_id) is not job:
                return
            if self._prefetch_jobs.get(job.request_id) is job:
                del self._prefetch_jobs[job.request_id]
            self._cancelled_jobs[id(job)] = job
            if self._cleanup_thread is None:
                self._cleanup_thread = threading.Thread(
                    target=self._drain_cancelled, name="lmcache-prefetch-cleanup",
                    daemon=True,
                )
                self._cleanup_thread.start()
            self._cleanup_condition.notify()

    def _drain_cancelled(self) -> None:
        while True:
            with self._cleanup_condition:
                if not self._cancelled_jobs and not self._closed:
                    self._cleanup_condition.wait()
                jobs = list(self._cancelled_jobs.values())
            for job in jobs:
                if job.cleanup_failed or time.monotonic() < job.cleanup_retry_after:
                    continue
                try:
                    if job.cleanup_keys is None:
                        found = self._query_job_result(job)
                        if found is None:
                            continue
                        # The result may be consumed on query. Retain its exact
                        # keys before release so an owned retry never re-queries.
                        job.cleanup_keys = tuple(
                            key for i, key in enumerate(job.object_keys) if found.test(i)
                        )
                    keys = list(job.cleanup_keys)
                    if keys:
                        self._ctx.storage_manager.finish_read_prefetched(
                            keys, read_locks=job.num_kv_readers,
                            **({"leases": job.handle.read_leases,
                                "claim_id": "cancelled-prefetch-unhanded"}
                               if job.handle.read_leases is not None else {})
                        )
                    if (job.handle.read_leases is not None
                            and self._ctx.storage_manager.owned_reads_pending(job.handle.read_leases)):
                        raise RuntimeError("Cancelled prefetch still owns read leases")
                    with self._cleanup_condition:
                        self._cancelled_jobs.pop(id(job), None)
                    if job.submission_completion is not None:
                        job.submission_completion.set_result(None)
                except Exception:
                    if job.handle.read_leases is None:
                        # Anonymous partial decrements cannot be replayed safely.
                        job.cleanup_failed = True
                    else:
                        job.cleanup_retry_after = time.monotonic() + job.cleanup_retry_delay
                        job.cleanup_retry_delay = min(60.0, job.cleanup_retry_delay * 2)
                    logger.exception("Cancelled prefetch cleanup failed: %s", job.request_id)
            with self._cleanup_condition:
                if self._closed:
                    if self._cancelled_jobs:
                        logger.warning("Closing with %d unconfirmed prefetch cleanups; buffers not released early", len(self._cancelled_jobs))
                    return
                if self._cancelled_jobs:
                    self._cleanup_condition.wait(timeout=0.01)

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs) + len(self._cancelled_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
