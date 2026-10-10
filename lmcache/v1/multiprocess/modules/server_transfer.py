# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field
from concurrent.futures import Future
from threading import Lock, Event
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


def create_transfer_strategy(
    storage_manager: "StorageManager",
    *,
    shm_name: str,
    pool_size: int,
    pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_lock: LockType,
    transfer_key_factory: Callable[
        [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
    ],
) -> "TransferStrategy":
    """Create the non-GPU transfer strategy for a registered context.

    Args:
        storage_manager: Storage manager used by the selected strategy.
        shm_name: Shared-memory pool name advertised to workers.
        pool_size: Shared-memory pool size in bytes.
        pending_writes: Map of pending SHM write reservations keyed by transfer key.
        pending_reads: Map of pending SHM read reservations keyed by transfer key.
        pending_lock: Lock guarding shared pending SHM reservation state.
        transfer_key_factory: Factory that builds the `(instance_id, key)` lookup key
            used in the pending SHM reservation maps.

    Returns:
        ``ShmTransferStrategy`` when SHM is configured with a non-empty pool name and
        positive pool size, otherwise ``PickleTransferStrategy``.
    """
    if shm_name and pool_size > 0:
        logger.info("Using shm non-GPU transfer strategy")
        return ShmTransferStrategy(
            storage_manager=storage_manager,
            pending_writes=pending_writes,
            pending_reads=pending_reads,
            pending_lock=pending_lock,
            transfer_key_factory=transfer_key_factory,
            fallback_strategy=PickleTransferStrategy(storage_manager),
        )

    logger.info("Using pickle non-GPU transfer strategy")
    return PickleTransferStrategy(storage_manager)


class TransferStrategy(abc.ABC):
    """Contract for non-GPU transport backends used by the server.

    Implementations encapsulate the transport-specific prepare/commit lifecycle for
    store and retrieve operations, allowing the server to use either pickle-based or
    shared-memory-based transfers behind a common interface.
    """

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific store preparation response.
        """

    @abc.abstractmethod
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific retrieve preparation response.
        """

    @abc.abstractmethod
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.

        Returns:
            ``True`` when retrieve finalization succeeds.
        """


class PickleTransferStrategy(TransferStrategy):
    """Pickle-based transport for non-GPU transfer requests.

    This is the default transport when SHM is unavailable, and it is also used as a
    fallback by the SHM strategy when the worker sends an inline serialized payload.
    ``prepare_store`` returns an empty context, while ``commit_store`` deserializes
    the pickle payload and writes the resulting tensors into reserved objects.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
    ) -> None:
        """Initialize pickle transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
        """
        self._storage_manager = storage_manager

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Return empty store context for pickle mode.

        Pickle transport does not pre-allocate SHM slots during prepare.
        """
        return PrepareStoreResponse(context={})

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        obj_keys = resolve_obj_keys(key)
        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
        reserved_dict = self._storage_manager.reserve_write(
            obj_keys, context.layout_desc, "new"
        )
        written_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                if obj_key not in reserved_dict:
                    continue
                if idx >= len(chunks):
                    logger.error(
                        "Engine-driven pickle store is missing chunk %d "
                        "(instance_id=%d, object_keys=%d, chunks=%d)",
                        idx,
                        instance_id,
                        len(obj_keys),
                        len(chunks),
                    )
                    continue
                memory_obj = reserved_dict[obj_key]
                if memory_obj.tensor is None:
                    logger.error(
                        "Engine-driven pickle store reserved an object without "
                        "a tensor (instance_id=%d, chunk_index=%d)",
                        instance_id,
                        idx,
                    )
                    continue
                chunk_cpu = chunks[idx]
                if chunk_cpu.shape != memory_obj.tensor.shape:
                    logger.error(
                        "Engine-driven pickle store chunk shape mismatch "
                        "(instance_id=%d, chunk_index=%d, chunk_shape=%s, "
                        "object_shape=%s)",
                        instance_id,
                        idx,
                        tuple(chunk_cpu.shape),
                        tuple(memory_obj.tensor.shape),
                    )
                    continue
                memory_obj.tensor.copy_(chunk_cpu)
                written_keys.append(obj_key)
        finally:
            if written_keys:
                self._storage_manager.finish_write(written_keys)

        success = len(written_keys) == len(reserved_dict)
        if not success:
            logger.error(
                "Engine-driven pickle store incomplete (instance_id=%d, "
                "object_keys=%d, reserved=%d, chunks=%d, written=%d)",
                instance_id,
                len(obj_keys),
                len(reserved_dict),
                len(chunks),
                len(written_keys),
            )
        return success

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        leases: dict | None = None,
        claim_id: str = "",
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        if leases is not None and not claim_id:
            raise ValueError("Owned pickle transfer requires a completion claim")
        obj_keys = resolve_obj_keys(key)
        prefetched_keys: list[ObjectKey] = []
        try:
            read_ctx = self._storage_manager.read_prefetched_results(
                obj_keys, **({"leases": leases} if leases is not None else {})
            )
            with read_ctx as maybe_memory_objs:
                if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                prefetched_keys = obj_keys[: len(maybe_memory_objs)]
                chunks = []
                for memory_obj in maybe_memory_objs:
                    if memory_obj.tensor is None:
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    chunks.append(memory_obj.tensor.cpu().clone())
                return PrepareRetrieveResponse(
                    success=True, data=pickle.dumps(chunks), context={}
                )
        finally:
            if leases is not None:
                # CPU clone/serialization finishes synchronously; no source
                # address escapes this call, including its failure paths.
                self._storage_manager.finish_read_prefetched(
                    obj_keys, leases=leases, claim_id=claim_id,
                )
            elif prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """No-op for pickle mode; data was already copied during prepare."""
        return True


@dataclass(frozen=True)
class OwnedShmRead:
    keys: tuple
    leases: dict
    claim_id: str
    response: PrepareRetrieveResponse
    completion: Future = field(default_factory=Future, compare=False)
    commit_lock: Any = field(default_factory=Lock, compare=False)
    commit_started: Any = field(default_factory=Event, compare=False)
    commit_failed: Any = field(default_factory=Event, compare=False)


def finish_owned_shm_read(storage_manager, pending):
    """One completion owner; unknown release outcomes must remain pinned."""
    with pending.commit_lock:
        if pending.commit_failed.is_set():
            raise RuntimeError("owned SHM read completion outcome unknown")
        if pending.completion.done():
            return pending.completion.result()
        pending.commit_started.set()
        try:
            storage_manager.finish_read_prefetched(
                list(pending.keys), leases=pending.leases, claim_id=pending.claim_id,
            )
        except BaseException:
            pending.commit_failed.set()
            raise
        pending.completion.set_result(True)
        return True


from lmcache.v1.multiprocess.shm_records import ShmRecordCapacityError


class ShmStoreAdmissionRejected(ShmRecordCapacityError):
    """Record admission failed before ownership or descriptors were created."""


@dataclass
class OwnedShmWrite:
    leases: dict = field(default_factory=dict)
    keys: tuple = ()
    response: Any = None
    state: str = "preparing"
    owner: Any = None


@dataclass
class OwnedLegacyShmRead:
    leases: dict = field(default_factory=dict)
    keys: tuple = ()
    response: Any = None
    state: str = "preparing"
    owner: Any = None


@dataclass
class UncertainShmCommit:
    """Retain exact keys when finish may have partially changed ownership."""
    keys: tuple


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory transport for non-GPU transfer requests.

    This strategy exposes SHM slot descriptors during ``prepare_store`` and
    ``prepare_retrieve`` so workers can access storage buffers directly. It tracks
    pending SHM reservations until the matching commit step releases them, and it
    falls back to pickle-based commit handling when ``cpu_data`` is non-empty.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_lock: LockType,
        transfer_key_factory: Callable[
            [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
        ],
        fallback_strategy: PickleTransferStrategy,
    ) -> None:
        """Initialize SHM transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
            pending_writes: Shared pending SHM write reservations map.
            pending_reads: Shared pending SHM read reservations map.
            pending_lock: Lock guarding shared pending SHM maps.
            transfer_key_factory: Factory to build `(instance_id, key)` transfer keys.
            fallback_strategy: Pickle fallback for non-empty ``cpu_data`` payloads.
        """
        self._storage_manager = storage_manager
        self._pending_writes = pending_writes
        self._pending_reads = pending_reads
        self._pending_lock = pending_lock
        self._transfer_key_factory = transfer_key_factory
        self._fallback_strategy = fallback_strategy
        self._retired = False
        self._registration_token = object()

    @staticmethod
    def _compact_completed(receipt):
        # Call only after storage completion returned successfully. Keep replay
        # identity/state, not the allocation, exported slots or strategy graph.
        receipt.leases = {}
        receipt.keys = ()
        receipt.response = None
        receipt.state = "committed"

    def retire_owned_writes(self):
        """Fence this registration before removing its engine entry."""
        with self._pending_lock:
            self._retired = True
            for pending in (self._pending_writes, self._pending_reads):
                for receipt in pending.values():
                    if isinstance(receipt, (OwnedShmWrite, OwnedLegacyShmRead)) and receipt.owner is self._registration_token and receipt.state not in ("committed", "aborted"):
                        receipt.state = "quarantined"

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Context with ``slots`` and ``chunk_indices``.
        """
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            if self._retired:
                raise RuntimeError("SHM write context retired")
            previous = self._pending_writes.get(transfer_key)
            if isinstance(previous, OwnedShmWrite) and previous.owner is self._registration_token and previous.state == "prepared":
                return previous.response
            if previous is not None:
                raise RuntimeError("SHM write identity already used or quarantined")
            receipt = OwnedShmWrite(owner=self._registration_token)
            try:
                self._pending_writes[transfer_key] = receipt
            except ShmRecordCapacityError as exc:
                raise ShmStoreAdmissionRejected(str(exc)) from exc
        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        reserved_keys: list[ObjectKey] = []
        try:
            obj_keys = resolve_obj_keys(key)
            reserved = self._storage_manager.reserve_write(
                obj_keys, context.layout_desc, "new", write_leases=receipt.leases
            )
            for idx, obj_key in enumerate(obj_keys):
                memory_obj = reserved.get(obj_key)
                if memory_obj is None or memory_obj.tensor is None:
                    continue
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                chunk_indices.append(idx)
                reserved_keys.append(obj_key)
            reserved_keys_set = set(reserved_keys)
            unused = {k: v for k, v in receipt.leases.items() if k not in reserved_keys_set}
            if unused:
                # No descriptors have been published, so no worker can write these.
                self._storage_manager.abort_write({}, write_leases=unused)
            response = PrepareStoreResponse(context={"slots": slots, "chunk_indices": chunk_indices})
            with self._pending_lock:
                if receipt.state != "preparing":
                    raise RuntimeError("SHM context retired during prepare")
                receipt.keys = tuple(reserved_keys)
                receipt.response = response
                receipt.state = "prepared"
            return response
        except BaseException:
            with self._pending_lock:
                receipt.state = "quarantined"
            raise

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize SHM store write locks or fallback to pickle commit.

        Returns:
            ``True`` when pending SHM reservation is committed successfully.
        """
        if cpu_data != b"":
            return self._fallback_strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
            )
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            if self._retired:
                raise RuntimeError("SHM write context retired")
            receipt = self._pending_writes.get(transfer_key)
            if receipt is None:
                return False
            if not isinstance(receipt, OwnedShmWrite):
                raise RuntimeError("SHM write receipt is not owned")
            if receipt.owner is not self._registration_token:
                raise RuntimeError("SHM write context retired or generation mismatch")
            if receipt.state == "committed":
                return True
            if receipt.state != "prepared":
                raise RuntimeError("SHM write outcome unknown or context retired")
            receipt.state = "committing"
        try:
            self._storage_manager.finish_write(list(receipt.keys), write_leases=receipt.leases)
        except BaseException:
            with self._pending_lock:
                receipt.state = "quarantined"
            raise
        with self._pending_lock:
            self._compact_completed(receipt)
        # Retain a tombstone: the wire COMMIT has no reservation nonce. Reusing
        # this transfer identity could let an old COMMIT publish a newer write.
        return True

    def abort_store(self, key, instance_id):
        """Caller must have stopped all access to the exported SHM slots."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            receipt = self._pending_writes.get(transfer_key)
            if self._retired or not isinstance(receipt, OwnedShmWrite) or receipt.owner is not self._registration_token:
                raise RuntimeError("SHM abort registration or ownership mismatch")
            if receipt.state == "aborted":return True
            if receipt.state != "prepared":
                raise RuntimeError("SHM abort requires a prepared, uncommitted write")
            receipt.state = "aborting"
        try:
            self._storage_manager.abort_write({}, write_leases=receipt.leases, strict=True)
        except BaseException:
            with self._pending_lock:receipt.state = "quarantined"
            raise
        with self._pending_lock:
            self._compact_completed(receipt)
            receipt.state = "aborted"
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        leases: dict | None = None,
        claim_id: str = "",
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        if leases is not None:
            return self._prepare_owned_retrieve(key, instance_id, resolve_obj_keys, leases, claim_id)
        return self._prepare_legacy_owned_read(key, instance_id, resolve_obj_keys)

    def _prepare_legacy_owned_read(self, key, instance_id, resolve_obj_keys):
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            if self._retired:
                raise RuntimeError("SHM read context retired")
            previous = self._pending_reads.get(transfer_key)
            if isinstance(previous, OwnedLegacyShmRead) and previous.owner is self._registration_token and previous.state == "prepared":
                return previous.response
            if previous is not None:
                raise RuntimeError("SHM read identity already used or quarantined")
            receipt = OwnedLegacyShmRead(owner=self._registration_token)
            self._pending_reads[transfer_key] = receipt
        try:
            obj_keys = resolve_obj_keys(key)
            keys, objects = self._storage_manager.adopt_legacy_reads(obj_keys, receipt.leases)
            receipt.keys = tuple(keys)
            valid = bool(objects) and len(keys) == len(obj_keys) and all(obj.tensor is not None for obj in objects)
            if not valid:
                self._storage_manager.finish_read_prefetched(list(receipt.leases), leases=receipt.leases, claim_id="unexported")
                response = PrepareRetrieveResponse(success=False, data=b"", context={})
            else:
                slots = [ShmSlotDescriptor(offset=obj.shm_offset, length=obj.shm_byte_length,
                    shape=list(obj.tensor.shape), dtype=_dtype_to_name(obj.tensor.dtype)).to_dict() for obj in objects]
                response = PrepareRetrieveResponse(success=True, data=b"", context={"slots": slots})
            with self._pending_lock:
                if receipt.state != "preparing":
                    raise RuntimeError("SHM read context retired during prepare")
                receipt.response = response
                receipt.state = "prepared" if valid else "committed"
                if not valid:
                    self._compact_completed(receipt)
            return response
        except BaseException:
            with self._pending_lock:
                receipt.state = "quarantined"
            raise

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        owned: bool = False,
    ) -> bool:
        """Release pending SHM read locks for the completed retrieve request."""
        if key.lookup_id:
            transfer_key = self._owned_transfer_key(key, instance_id)
            with self._pending_lock:
                pending = self._pending_reads.get(transfer_key)
            if pending is not None:
                finish_owned_shm_read(self._storage_manager, pending)
                with self._pending_lock:
                    if self._pending_reads.get(transfer_key) is pending:
                        del self._pending_reads[transfer_key]
                return True
        if owned:
            return False
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            receipt = self._pending_reads.get(transfer_key)
            if receipt is None:
                return True
            if self._retired or not isinstance(receipt, OwnedLegacyShmRead) or receipt.owner is not self._registration_token:
                raise RuntimeError("SHM read context retired or generation mismatch")
            if receipt.state == "committed":
                return True
            if receipt.state != "prepared":
                raise RuntimeError("SHM read completion unknown")
            receipt.state = "committing"
        try:
            self._storage_manager.finish_read_prefetched(list(receipt.keys), leases=receipt.leases, claim_id="shm-client-complete")
        except BaseException:
            with self._pending_lock:
                receipt.state = "quarantined"
            raise
        with self._pending_lock:
            self._compact_completed(receipt)
        return True

    @staticmethod
    def _reject_uncertain(pending_map, transfer_key):
        if isinstance(pending_map.get(transfer_key), UncertainShmCommit):
            raise RuntimeError("SHM commit outcome unknown; refusing unsafe retry")

    def _commit_legacy(self, pending_map, transfer_key, finish, *, missing_result):
        # Publish the in-flight receipt against duplicate COMMIT/reaper, then
        # call storage without the pending-map lock (observers may reenter).
        with self._pending_lock:
            keys = pending_map.get(transfer_key)
            if keys is None:
                return missing_result
            if isinstance(keys, UncertainShmCommit):
                raise RuntimeError("SHM commit outcome unknown; refusing unsafe retry")
            receipt = UncertainShmCommit(tuple(keys))
            pending_map[transfer_key] = receipt
        # On any failure retain the receipt; never unlock twice on retry.
        if keys:
            finish(keys)
        with self._pending_lock:
            if pending_map.get(transfer_key) is receipt:
                del pending_map[transfer_key]
        return True

    @staticmethod
    def _owned_transfer_key(key, instance_id):
        return (instance_id, "owned", key.request_id, key.lookup_id, key)

    def _prepare_owned_retrieve(self, key, instance_id, resolve_obj_keys, leases, claim_id):
        if not key.lookup_id or not claim_id:
            raise ValueError("Owned SHM transfer requires lookup identity and completion claim")
        transfer_key = self._owned_transfer_key(key, instance_id)
        # Serialize prepare/commit publication. Existing identical PREPARE
        # returns the same descriptors without acquiring another reader share.
        with self._pending_lock:
            existing = self._pending_reads.get(transfer_key)
            if existing is not None:
                if existing.leases != leases or existing.claim_id != claim_id:
                    raise ValueError("SHM replay changed ownership")
                return existing.response
            obj_keys = resolve_obj_keys(key)
            published = False
            try:
                found, objects = self._storage_manager.unsafe_read(obj_keys, leases=leases)
                if len(found) != len(obj_keys) or not objects:
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                slots = []
                for obj in objects:
                    if obj.tensor is None:
                        return PrepareRetrieveResponse(success=False, data=b"", context={})
                    slots.append(ShmSlotDescriptor(
                        offset=obj.shm_offset, length=obj.shm_byte_length,
                        shape=list(obj.tensor.shape), dtype=_dtype_to_name(obj.tensor.dtype),
                    ).to_dict())
                response = PrepareRetrieveResponse(success=True, data=b"", context={"slots": slots})
                self._pending_reads[transfer_key] = OwnedShmRead(
                    tuple(obj_keys), dict(leases), claim_id, response,
                )
                published = True
                return response
            finally:
                if not published:
                    # Descriptors never escaped; synchronous cleanup is safe.
                    self._storage_manager.finish_read_prefetched(
                        obj_keys, leases=leases, claim_id=claim_id,
                    )
