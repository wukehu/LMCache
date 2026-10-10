# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from dataclasses import dataclass, field, replace
from typing import TypedDict
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.gpu_connector.gds_context import (
    get_gds_context,
    initialize_gds_context,
)
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class ShmPoolInfo(TypedDict):
    """Shared-memory pool metadata returned during registration."""

    shm_name: str
    pool_size: int


@dataclass
class _WorkerLayoutEntry:
    """One worker id's per-group layouts and its active registration count."""

    group_layout_descs: dict[int, MemoryLayoutDesc]
    ref_count: int


@dataclass
class _LayoutDescEntry:
    """Stored layout descriptor and its active registration count."""

    layout_desc: MemoryLayoutDesc
    ref_count: int
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    """Cross-chunk attention windows of all object groups, in object-group
    order. Defaults to a single full-attention group."""
    group_layout_descs: dict[int, MemoryLayoutDesc] | None = None
    """Per-group layout descriptors, or ``None`` if all share ``layout_desc``."""
    worker_entries: dict[int, _WorkerLayoutEntry] = field(default_factory=dict)
    """Per-group layouts of each worker id that registered with one. Every
    worker registration is also counted in ``ref_count``, so these entries
    never outlive the pair's entry."""


class LayoutDescRegistry:
    """Thread-safe registry mapping (model_name, world_size) to MemoryLayoutDesc.

    Modules write to this registry when KV caches are registered.
    Consumers (e.g. LookupModule) read from it to find layout descriptors
    for prefetch tasks. Multiple worker instances can share the same
    ``(model_name, world_size)`` entry, so the registry keeps the descriptor
    until the last matching registration is unregistered.

    Workers of one pair can hold different layouts: pipeline-parallel stages
    host different numbers of layers, so their objects of one object group
    differ in size. A registration that names its worker id is therefore also
    kept per worker id; the pair-wide descriptor is the latest registration's
    and serves the workers that never named theirs.
    """

    def __init__(self) -> None:
        # Key: (model_name, world_size) -> layout descriptor entry
        self._registry: dict[tuple[str, int], _LayoutDescEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        group_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        worker_id: int | None = None,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Re-registering the same pair increments the active registration
        count. The latest descriptor is retained for lookups, both pair-wide
        and for ``worker_id``.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order. Defaults to a single full-attention group.
            group_layout_descs: Maps object_group_id to that group's layout
                (each group is a separate keyed allocation); ``None`` derives
                one entry per object group in ``attn_desc``, all sharing
                ``layout_desc``.
            worker_id: The worker id (``IPCCacheServerKey.worker_id``) whose
                KV shard these layouts describe, or ``None`` when the
                registering client did not name it. A named worker's layouts
                are returned by :meth:`find_worker_group_layout_descs`.
        """
        key = (model_name, world_size)
        attn_desc = replace(attn_desc, world_size=world_size)
        if group_layout_descs is None:
            group_layout_descs = {
                gid: layout_desc for gid in range(attn_desc.num_object_groups)
            }
        with self._lock:
            entry = self._registry.get(key)
            if worker_id is not None:
                if not 0 <= worker_id < world_size:
                    raise ValueError("Layout worker is outside global world size")
                previous = entry.worker_entries.get(worker_id) if entry else None
                if previous is not None and previous.group_layout_descs != group_layout_descs:
                    raise ValueError("Conflicting layouts for the same global KV shard")
            if entry is None:
                entry = _LayoutDescEntry(
                    layout_desc=layout_desc,
                    ref_count=1,
                    attn_desc=attn_desc,
                    group_layout_descs=group_layout_descs,
                )
                self._registry[key] = entry
            else:
                entry.layout_desc = layout_desc
                entry.attn_desc = attn_desc
                entry.group_layout_descs = group_layout_descs
                entry.ref_count += 1

            if worker_id is None:
                return
            worker_entry = entry.worker_entries.get(worker_id)
            if worker_entry is None:
                entry.worker_entries[worker_id] = _WorkerLayoutEntry(
                    group_layout_descs=group_layout_descs, ref_count=1
                )
            else:
                worker_entry.group_layout_descs = group_layout_descs
                worker_entry.ref_count += 1

    def unregister(
        self, model_name: str, world_size: int, worker_id: int | None = None
    ) -> None:
        """Unregister one layout descriptor registration for a pair.

        The descriptor is removed only when the last active registration for
        the pair is unregistered; a worker's layouts likewise stay until the
        last registration naming that worker is unregistered.

        Args:
            model_name: The model name.
            world_size: The world size.
            worker_id: The worker id the registration was made with (the
                ``worker_id`` passed to :meth:`register`).
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                return

            if entry.ref_count <= 1:
                self._registry.pop(key)
                return

            entry.ref_count -= 1
            if worker_id is None:
                return
            worker_entry = entry.worker_entries[worker_id]
            if worker_entry.ref_count <= 1:
                del entry.worker_entries[worker_id]
            else:
                worker_entry.ref_count -= 1

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by (model_name, world_size).

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.layout_desc

    def find_group_layout_descs(
        self, model_name: str, world_size: int
    ) -> dict[int, MemoryLayoutDesc] | None:
        """Look up per-object-group layout descriptors, or ``None``."""
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.group_layout_descs

    def find_worker_group_layout_descs(
        self, model_name: str, world_size: int
    ) -> dict[int, dict[int, MemoryLayoutDesc]]:
        """Look up the per-group layouts of each worker id of a pair.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            Maps worker id to that worker's ``{object_group_id: layout}``,
            for the worker ids registered by :meth:`register` with a
            ``worker_id``. Empty when the pair is unregistered or no
            registration named its worker; workers missing here use
            :meth:`find_group_layout_descs`.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return {}
            return {
                worker_id: worker_entry.group_layout_descs
                for worker_id, worker_entry in entry.worker_entries.items()
            }

    def find_attn_desc(self, model_name: str, world_size: int) -> AttnWindowDesc:
        """Look up the attention-window descriptor for a pair.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The :class:`AttnWindowDesc` registered for the pair.

        Raises:
            ValueError: If no descriptor is registered for the pair. Callers
                must register the KV cache (which establishes the pair) before
                looking up its windows.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                raise ValueError(
                    f"No attention-window descriptor registered for model "
                    f"{model_name!r} with world size {world_size}"
                )
            return entry.attn_desc


class MPCacheServerContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Chunk size for KV cache operations.
        hash_algorithm: Hash algorithm for token hashing.
        separate_object_groups: Whether to split kernel groups into one object
            group per sliding-window size at KV-cache registration. Default
            False.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ) -> None:
        self._chunk_size = chunk_size
        self._separate_object_groups = separate_object_groups
        self._full_sw_kv = full_sw_kv

        # Initialize the process-global GDS context.
        # No-op when GDS L1 is disabled (config is None).
        initialize_gds_context(storage_manager_config.l1_manager_config.gds_l1_config)

        self.shm_pool_info: ShmPoolInfo = self._compute_shm_pool_info(
            storage_manager_config
        )
        self._storage_manager = StorageManager(storage_manager_config)
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._session_manager = SessionManager(self._token_hasher)
        self._event_bus = get_event_bus()
        self._layout_desc_registry = LayoutDescRegistry()

    def close(self) -> None:
        """
        Tear down the session manager, storage manager, and the process-global
        GDS context.
        """
        self._session_manager.close()
        self._storage_manager.close()
        # Tear down the GDS cuFile context (the shared slab + its handle).
        get_gds_context().close()

    @property
    def chunk_size(self) -> int:
        """Chunk size for KV cache operations."""
        return self._chunk_size

    @property
    def separate_object_groups(self) -> bool:
        """Whether to split kernel groups into per-sliding-window object groups."""
        return self._separate_object_groups

    @property
    def full_sw_kv(self) -> bool:
        """Whether sliding-window groups cache full per-chunk KV (no window cutting)."""
        return self._full_sw_kv

    @property
    def storage_manager(self) -> StorageManager:
        """The storage manager instance."""
        return self._storage_manager

    @property
    def token_hasher(self) -> TokenHasher:
        """The token hasher for computing chunk hashes."""
        return self._token_hasher

    @property
    def session_manager(self) -> SessionManager:
        """The session manager for request lifecycle tracking."""
        return self._session_manager

    @property
    def event_bus(self) -> EventBus:
        """The event bus for observability events."""
        return self._event_bus

    @property
    def layout_desc_registry(self) -> LayoutDescRegistry:
        """Registry mapping (model_name, world_size) to MemoryLayoutDesc."""
        return self._layout_desc_registry

    def resolve_obj_keys(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> list[list[ObjectKey]]:
        """Resolve per-object-group object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.
            object_group_ids: Object group ids to produce keys for.

        Returns:
            The i-th element is the list of ObjectKeys for
            ``object_group_ids[i]``.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        if session.lookup_ipc_key is None:
            session.lookup_ipc_key = key.no_worker_id_version()
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes, object_group_ids)

    @staticmethod
    def _compute_shm_pool_info(
        storage_manager_config: StorageManagerConfig,
    ) -> ShmPoolInfo:
        """Compute normalized SHM pool metadata from storage config.

        Returns an empty pool (disabled SHM transport) when ``shm_name`` is
        empty or lazy memory mode is enabled. Otherwise strips any leading ``/``
        and ensures the name starts with ``lmcache_l1_pool_``.
        """
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        shm_name = mem_cfg.shm_name or ""
        if not shm_name or mem_cfg.use_lazy or mem_cfg.devdax_path:
            return {"shm_name": "", "pool_size": 0}
        bare = shm_name.lstrip("/")
        if not bare.startswith("lmcache_l1_pool_"):
            shm_name = f"lmcache_l1_pool_{bare}"
        return {"shm_name": shm_name, "pool_size": mem_cfg.size_in_bytes}
