# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

# Local
from .server_transfer import (
    TransferStrategy,
    OwnedShmRead,
    finish_owned_shm_read,
    UncertainShmCommit,
    OwnedShmWrite,
    OwnedLegacyShmRead,
    ShmTransferStrategy,
    create_transfer_strategy,
)
from lmcache.v1.multiprocess.shm_wire import ShmTransferRef, ShmRegistrationIdentity

logger = init_logger(__name__)


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
        last_seen: ``time.monotonic()`` of the most recent activity from this
            instance (register, PING, prepare/commit). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window. Latched only by PING.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    transfer_strategy: TransferStrategy | None = None
    resources_released: bool = False


class EngineDrivenTransferModule(InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext, max_shm_transfer_records: int = 65536) -> None:
        self._ctx = ctx
        self._engine_driven_contexts: dict[int, EngineDrivenContextEntry] = {}
        self._strategies: dict[int, TransferStrategy] = {}
        # Guards _engine_driven_contexts and _strategies together (the reaper
        # mutates them off the MQ main loop). Lock order is engine registration
        # lock, then pending SHM lock, then record-table lock.
        self._lock = threading.Lock()
        from lmcache.v1.multiprocess.shm_records import ShmRecords
        self._shm_records = ShmRecords(max_shm_transfer_records)
        self._pending_shm_writes = self._shm_records.writes
        self._pending_shm_reads = self._shm_records.reads
        self._pending_shm_lock = threading.Lock()
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        with self._lock:
            entries = dict(self._engine_driven_contexts)
        for instance_id, entry in entries.items():
            registered_non_cuda_ids.append(instance_id)
            non_cuda_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "block_size": entry.metadata.block_size,
                "use_mla": entry.metadata.use_mla,
            }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
            "shm_transfer_records": self._shm_records.snapshot(),
        }

    def close(self) -> None:
        """Release resources owned by this module."""
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()

    def touch_instance(self, instance_id: int) -> bool:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True
                return True
            return False

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered non-GPU instances."""
        with self._lock:
            return len(self._engine_driven_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap non-GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one that
        has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, EngineDrivenContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._engine_driven_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                entry = self._engine_driven_contexts.pop(iid)
                strategy = self._strategies.pop(iid, None)
                if hasattr(strategy, "retire_owned_writes"):
                    strategy.retire_owned_writes()
                reaped.append((iid, entry))
        for iid, entry in reaped:
            self._release_entry(iid, entry)
            logger.warning(
                "Reaped non-GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
        return [iid for iid, _ in reaped]

    def _resolve_for_transfer(
        self, instance_id: int
    ) -> tuple[EngineDrivenContextEntry, TransferStrategy]:
        """Return (entry, strategy) for a transfer, refreshing last_seen.

        Pair-atomicity guarantees the entry exists whenever the strategy
        does. Refreshes last_seen (no latch) so an active worker is not
        reaped mid-transfer.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry and its transfer strategy.

        Raises:
            ValueError: If the instance is not registered (or was reaped).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            strategy = self._strategies.get(instance_id)
            if entry is None or strategy is None:
                raise ValueError(
                    "non-GPU context not registered (or reaped) for "
                    f"instance ID {instance_id}"
                )
            entry.last_seen = now
            return entry, strategy

    def _release_entry(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        # Serialize layout reference retirement against replacement registration.
        # Delayed/repeated cleanup must not decrement a newer registration.
        with self._lock:
            if entry.resources_released:
                return
            self._release_entry_resources(instance_id, entry)
            entry.resources_released = True

    def _release_entry_resources(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        """Release resources for a popped entry (run outside the lock).

        Quarantines legacy transfers: losing a context does not prove that
        a client has stopped accessing exported SHM slots.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
        with self._pending_shm_lock:
            for pending_map in (self._pending_shm_writes, self._pending_shm_reads):
                for key, reservation in pending_map.items():
                    if key[0] == instance_id and isinstance(reservation, (OwnedShmWrite, OwnedLegacyShmRead)):
                        strategy = getattr(entry, "transfer_strategy", None)
                        if reservation.owner is getattr(strategy, "_registration_token", None) and reservation.state != "committed":
                            reservation.state = "quarantined"
                    elif key[0] == instance_id and not isinstance(reservation, (OwnedShmRead, UncertainShmCommit)):
                        pending_map[key] = UncertainShmCommit(tuple(reservation))

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple:
        # Cache equality deliberately ignores request/lookup identity. Pending
        # transport operations must not: another request can reuse the same KV.
        return (instance_id, key.request_id, key.lookup_id, key)

    def _resolve_single_group_obj_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
        """Resolve object keys for the single object group used by
        non-GPU transfers."""
        return self._ctx.resolve_obj_keys(key, [0])[0]

    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                num_physical_slots).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        num_physical_slots = payload.num_physical_slots
        if num_physical_slots is None:
            # Compatibility with clients from before the physical-slot field
            # was added. Those clients require one slot per logical token.
            num_physical_slots = self._ctx.chunk_size
        elif num_physical_slots <= 0:
            raise ValueError(
                f"num_physical_slots must be positive, got {num_physical_slots}"
            )

        shape = (
            torch.Size(
                [payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        now = time.monotonic()
        with self._lock:
            existing = self._engine_driven_contexts.get(payload.instance_id)
            if existing is not None:
                if (
                    existing.model_name != payload.model_name
                    or existing.world_size != payload.world_size
                    or existing.metadata != metadata
                ):
                    raise ValueError("registration metadata mismatch for existing instance")
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered (non-GPU); refreshing liveness",
                    payload.instance_id,
                )
                return RegisterEngineDrivenContextResponse(
                    shm_name=shm_name, pool_size=pool_size
                )
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SYNC-serialized, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            last_seen=now,
            has_liveness_signal=False,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
            shm_name=shm_name,
            pool_size=pool_size,
            pending_writes=self._pending_shm_writes,
            pending_reads=self._pending_shm_reads,
            pending_lock=self._pending_shm_lock,
            transfer_key_factory=self._make_transfer_key,
        )
        if isinstance(strategy, ShmTransferStrategy):
            from lmcache.v1.multiprocess.shm_sequenced_store import SequencedShmStore
            strategy.sequence_store = SequencedShmStore(strategy,self._shm_records.limit)
        with self._lock:
            previous_strategy = self._strategies.get(payload.instance_id)
            if hasattr(previous_strategy, "retire_owned_writes"):
                previous_strategy.retire_owned_writes()
            entry.transfer_strategy = strategy
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, layout_desc
            )
            self._engine_driven_contexts[payload.instance_id] = entry
            self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
        )

        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name, pool_size=pool_size
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        with self._lock:
            entry = self._engine_driven_contexts.pop(instance_id, None)
            if entry is not None:
                strategy = self._strategies.pop(instance_id, None)
                if hasattr(strategy, "retire_owned_writes"):
                    strategy.retire_owned_writes()
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._release_entry(instance_id, entry)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    def _validate_request_limits(self, key):
        limits = self._ctx.session_manager.request_limits
        if limits is not None:
            limits.validate_key(key)
            # Engine-driven transfer currently resolves a single object group.
            limits.validate_expansion(key, self._ctx.chunk_size, 1)

    def _sequenced_store(self, instance_id):
        entry,strategy=self._resolve_for_transfer(instance_id)
        if not isinstance(strategy,ShmTransferStrategy):
            raise ValueError('sequenced SHM requires a shared-memory registration')
        return entry,strategy.sequence_store

    def shm_v2_handshake(self,instance_id: int) -> ShmRegistrationIdentity:
        _,store=self._sequenced_store(instance_id)
        return store.handshake()

    def shm_v2_prepare_store(self,key: IPCCacheServerKey,instance_id: int,ref: ShmTransferRef) -> PrepareStoreResponse:
        self._validate_request_limits(key)
        entry,store=self._sequenced_store(instance_id)
        return store.prepare(key,instance_id,ref,entry.metadata,self._resolve_single_group_obj_keys)

    def shm_v2_commit_store(self,instance_id: int,ref: ShmTransferRef) -> bool:
        _,store=self._sequenced_store(instance_id)
        return store.commit(instance_id,ref)

    def shm_v2_prepare_retrieve(self,key: IPCCacheServerKey,instance_id: int,ref: ShmTransferRef) -> PrepareRetrieveResponse:
        self._validate_request_limits(key)
        _,store=self._sequenced_store(instance_id)
        return store.prepare_read(key,instance_id,ref,
            lambda:self.prepare_retrieve(key,instance_id),
            lambda:self.commit_retrieve(key,instance_id))

    def shm_v2_commit_retrieve(self,instance_id: int,ref: ShmTransferRef) -> bool:
        _,store=self._sequenced_store(instance_id)
        return store.commit(instance_id,ref,kind='retrieve')

    def shm_v2_abort_store(self,instance_id: int,ref: ShmTransferRef) -> bool:
        _,store=self._sequenced_store(instance_id)
        return store.abort_store(instance_id,ref)

    def shm_v2_ack(self,instance_id: int,ref: ShmTransferRef) -> int:
        _,store=self._sequenced_store(instance_id)
        return store.acknowledge(instance_id,ref)

    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for pickle mode.
        """
        self._validate_request_limits(key)
        entry, strategy = self._resolve_for_transfer(instance_id)
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
        return response

    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit serialized CPU chunks to storage.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Pickled list of CPU tensors produced by the worker.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        self._validate_request_limits(key)
        entry, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("store_start_time", None)
        result = strategy.commit_store(
            key=key,
            instance_id=instance_id,
            cpu_data=cpu_data,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        if st is not None and result:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

    def _replay_engine_retrieve(self, key, instance_id, result):
        response = result.result()
        if response.success and response.context.get("slots"):
            # Session replay retains the PREPARE result for COMMIT idempotency.
            # A cached descriptor is exportable only while its exact owned
            # receipt still protects the SHM range. Completed reads must miss.
            transfer_key = ShmTransferStrategy._owned_transfer_key(key, instance_id)
            with self._pending_shm_lock:
                pending = self._pending_shm_reads.get(transfer_key)
                if (not isinstance(pending, OwnedShmRead)
                        or pending.response is not response
                        or pending.commit_started.is_set()
                        or pending.completion.done()):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
        return response

    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Retrieve prefetched chunks and return serialized CPU tensors.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        self._validate_request_limits(key)
        if key.lookup_id:
            session = self._ctx.session_manager.get_for_replay(key.request_id, key.lookup_id)
            if session is None:
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            signature = ("engine-driven", key.model_name, key.world_size,
                         key.token_ids, key.cache_salt, key.num_kv_readers)
            previous = session.find_retrieve(key, instance_id, signature)
            if previous is not None:
                return self._replay_engine_retrieve(key, instance_id, previous)
            state = session.prepare_failed_retrieve_release(key)
            if state is None:
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            obj_keys = self._resolve_single_group_obj_keys(key)
            try:
                leases = session.read_lease_snapshot(obj_keys)
            except KeyError:
                # A completed lookup can retain only a prefix (or no keys).
                # Never fall back to an unowned read beyond that coverage.
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            if leases is not None:
                _, strategy = self._resolve_for_transfer(instance_id)
                owner, result = session.begin_retrieve(key, instance_id, signature)
                if not owner:
                    return self._replay_engine_retrieve(key, instance_id, result)
                try:
                    if not session.claim_worker_read(key, instance_id, state[3], obj_keys):
                        response = PrepareRetrieveResponse(success=False, data=b"", context={})
                    else:
                        response = strategy.prepare_retrieve(
                            key=key, instance_id=instance_id,
                            resolve_obj_keys=lambda _: obj_keys,
                            leases=leases, claim_id=f"worker:{instance_id}",
                        )
                    if response.success and isinstance(strategy, ShmTransferStrategy):
                        transfer_key = strategy._owned_transfer_key(key, instance_id)
                        with self._pending_shm_lock:
                            pending = self._pending_shm_reads.get(transfer_key)
                        if pending is not None:
                            session.retain_external_completion(pending.completion)
                    result.set_result(response)
                    return response
                except BaseException as exc:
                    result.set_exception(exc)
                    raise
        _, strategy = self._resolve_for_transfer(instance_id)
        response = strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
        return response

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Args:
            key: Cache key (unused for pickle).
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            Always ``True``.
        """
        self._validate_request_limits(key)
        if key.lookup_id:
            transfer_key = ShmTransferStrategy._owned_transfer_key(key, instance_id)
            with self._pending_shm_lock:
                pending = self._pending_shm_reads.get(transfer_key)
            if isinstance(pending, OwnedShmRead):
                # Context removal is not completion; an exact COMMIT can still
                # release its reservation after the instance registry was reaped.
                finish_owned_shm_read(self._ctx.storage_manager, pending)
                with self._pending_shm_lock:
                    if self._pending_shm_reads.get(transfer_key) is pending:
                        del self._pending_shm_reads[transfer_key]
                return True
            session = self._ctx.session_manager.get_for_replay(key.request_id, key.lookup_id)
            if session is None:
                return False
            if session.read_lease_snapshot([]) is not None:
                # Pickle finished synchronously, or this COMMIT was already
                # processed. Do not fall through to anonymous SHM reservations.
                signature = ("engine-driven", key.model_name, key.world_size,
                             key.token_ids, key.cache_salt, key.num_kv_readers)
                completed = session.find_retrieve(key, instance_id, signature)
                return bool(completed is not None and completed.done()
                            and not completed.cancelled()
                            and completed.exception() is None
                            and completed.result().success)
        _, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result
