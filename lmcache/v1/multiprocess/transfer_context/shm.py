# SPDX-License-Identifier: Apache-2.0
"""Shared-memory EngineDrivenContext implementation for multiprocess mode."""

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from typing import Any
import ctypes
import threading
from lmcache.v1.multiprocess.shm_sequence_client import ShmSequenceClient,STORE_FEATURE,ABORT_FEATURE

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


@dataclass(frozen=True)
class ShmSlotDescriptor:
    """Describe one tensor slot in the shared-memory pool.

    Args:
        offset: Byte offset into the shared-memory pool.
        length: Byte length of the slot.
        shape: Logical tensor shape to view at the slot.
        dtype: Torch dtype attribute name, such as ``"bfloat16"``.
    """

    offset: int
    length: int
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the slot descriptor into the MQ context schema.

        Returns:
            Dict payload shared between the server and worker for one SHM slot.
        """
        return {
            "offset": self.offset,
            "length": self.length,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShmSlotDescriptor":
        """Parse a slot descriptor from the MQ context schema.

        Args:
            d: Mapping containing ``offset``, ``length``, ``shape``, and
                ``dtype`` fields.

        Returns:
            Parsed immutable slot descriptor.

        Raises:
            KeyError: If any required field is missing.
            TypeError: If ``shape`` cannot be converted with ``list(...)``.
            ValueError: If numeric fields cannot be coerced to integers.
        """
        return cls(
            offset=int(d["offset"]),
            length=int(d["length"]),
            shape=list(d["shape"]),
            dtype=str(d["dtype"]),
        )


class EngineDrivenContextShm(EngineDrivenContext):
    """Shared-memory implementation of :class:`EngineDrivenContext`."""

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        req_client: RequestClient,
        mq_timeout: float,
        shm_name: str,
        pool_size: int,
        *, recovery_enabled: bool = True,
    ) -> None:
        super().__init__(metadata, req_client, mq_timeout)
        self._sequence_clients={}
        self._sequence_lock=threading.Lock()
        self._recovery_stop=threading.Event()
        self._recovery_thread=None
        self._recovery_cursor=0
        if not shm_name or pool_size <= 0:
            raise ValueError("shm_name must be non-empty and pool_size must be > 0")

        self._shm_name = shm_name
        self._pool_size = pool_size
        self._shm: shared_memory.SharedMemory | None = None
        self._shm_buffer: memoryview | None = None
        self._pinned = False
        self._pinned_ptr = 0
        self._pinned_size = 0
        try:
            self._shm = shared_memory.SharedMemory(
                name=shm_name.lstrip("/"), create=False
            )
            # The SHM segment is owned by the server process. Unregister it
            # from this worker's resource tracker so that Python does not
            # unlink the segment when this worker exits.
            unregister(f"/{self._shm.name}", "shared_memory")
            self._shm_buffer = self._shm.buf
            # pin memory is per process
            # the shm might be pinned on lmcache server side already
            # pin memory here is for worker side for fast DMA copy
            self._pin_shm_buffer()
            logger.info("SHM pinned=%s for shm_name=%s", self._pinned, self._shm_name)
        except Exception:
            self._shm = None
            self._shm_buffer = None
            raise

        if recovery_enabled:
            self._recovery_thread=threading.Thread(target=self._recover_loop,name='lmcache-shm-recovery',daemon=True)
            self._recovery_thread.start()

    def _recover_loop(self):
        while not self._recovery_stop.wait(.25):
            if not self._sequence_lock.acquire(blocking=False):continue
            try:clients=tuple(c for c in self._sequence_clients.values() if c is not None)
            finally:self._sequence_lock.release()
            if not clients:continue
            budget=4
            start=self._recovery_cursor%len(clients)
            for offset in range(min(16,len(clients))):
                if self._recovery_stop.is_set() or budget<=0:break
                client=clients[(start+offset)%len(clients)]
                self._recovery_cursor=start+offset+1
                try:
                    budget-=client.recover_pending(limit=1,timeout=.25)
                except Exception:
                    # Keep ownership intact and report unexpected recovery faults.
                    logger.exception('SHM background recovery failed')
                    budget-=1

    def _make_tensor_view(
        self,
        offset: int,
        length: int,
        shape: list[int],
        dtype_str: str,
    ) -> torch.Tensor:
        """Create a tensor view over a SHM slot via ``torch.frombuffer``."""
        dtype = getattr(torch, dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype string: {dtype_str}")
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize <= 0:
            raise ValueError(f"Invalid dtype size for {dtype_str}")
        count = length // itemsize
        if self._shm_buffer is None:
            raise RuntimeError(
                f"Shared memory buffer not initialized for shm_name={self._shm_name}"
            )
        tensor_1d = torch.frombuffer(
            self._shm_buffer, dtype=dtype, count=count, offset=offset
        )
        return tensor_1d.view(torch.Size(shape))

    def _build_slot_tensors(self, slots: list[dict[str, Any]]) -> list[torch.Tensor]:
        descriptors = [ShmSlotDescriptor.from_dict(slot) for slot in slots]
        return [
            self._make_tensor_view(
                offset=descriptor.offset,
                length=descriptor.length,
                shape=descriptor.shape,
                dtype_str=descriptor.dtype,
            )
            for descriptor in descriptors
        ]

    def _sequence_client(self,instance_id):
        with self._sequence_lock:
            if instance_id in self._sequence_clients:
                return self._sequence_clients[instance_id]
            result=None
            if hasattr(self.req_client,'shm_v2_handshake'):
                capabilities=self.req_client.get_experimental().result(timeout=self.mq_timeout)
                if STORE_FEATURE in capabilities:
                    identity=self.req_client.shm_v2_handshake(instance_id).result(timeout=self.mq_timeout)
                    result=ShmSequenceClient(self.req_client,instance_id,self.mq_timeout,identity,supports_abort=ABORT_FEATURE in capabilities)
            self._sequence_clients[instance_id]=result
            return result

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        sequenced=self._sequence_client(instance_id)
        if sequenced is not None:
            try:response=sequenced.prepare(key)
            except TimeoutError as exc:
                raise LMCacheTimeoutError(
                    f"Sequenced PREPARE_STORE timed out for instance_id={instance_id}",
                    session_id=key.request_id,
                ) from exc
            context=response.context
            slots=context.get('slots',[])
            return (self._build_slot_tensors(slots),context['chunk_indices']) if slots else ([],[])
        future = self.req_client.prepare_store(key, instance_id)
        # wait() first so a timeout raises exactly one LMCacheTimeoutError
        # (one event); result() then returns without its own timeout.
        if not future.wait(timeout=self.mq_timeout):
            raise LMCacheTimeoutError(
                f"PREPARE_STORE timed out for instance_id={instance_id} "
                f"after {self.mq_timeout}s",
                session_id=key.request_id,
            )
        response = future.result()
        context = response.context if isinstance(response.context, dict) else {}
        slots = context.get("slots")
        if not isinstance(slots, list):
            return None
        if not slots:
            # Server explicitly signals all chunks are already cached.
            return [], []
        chunk_indices: list[int] = context["chunk_indices"]
        return self._build_slot_tensors(slots), chunk_indices

    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, _chunks: list[torch.Tensor]
    ) -> bool:
        sequenced=self._sequence_client(instance_id)
        if sequenced is not None:
            try:return sequenced.commit(key)
            except TimeoutError:return False
        future = self.req_client.commit_store(key, instance_id, b"")
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        sequenced=self._sequence_client(instance_id)
        if sequenced is not None and 'retrieve' in sequenced.identity.operations:
            try:
                response=sequenced.prepare(key,kind='retrieve')
                if not response.success:
                    # No memory was exported; retire the miss without requiring
                    # callers to COMMIT a None result. Keep retry state on loss.
                    sequenced.commit(key,kind='retrieve')
                    return None
            except TimeoutError as exc:
                raise LMCacheTimeoutError('Sequenced PREPARE_RETRIEVE timed out',session_id=key.request_id) from exc
            slots=response.context.get('slots',[])
            return self._build_slot_tensors(slots) if slots else None
        future = self.req_client.prepare_retrieve(key, instance_id)
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success:
            return None
        slots = response.context.get("slots", [])
        return self._build_slot_tensors(slots) if slots else None

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        sequenced=self._sequence_client(instance_id)
        if sequenced is not None and 'retrieve' in sequenced.identity.operations:
            try:return sequenced.commit(key,kind='retrieve')
            except TimeoutError:return False
        future = self.req_client.commit_retrieve(key, instance_id)
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def abort_store(self,key,instance_id,*,access_complete):
        sequenced=self._sequence_client(instance_id)
        if sequenced is None:raise RuntimeError('sequenced SHM abort unavailable')
        try:return sequenced.abort_store(key,access_complete=access_complete)
        except TimeoutError:return False

    def abandon_retrieve(self,key,instance_id,*,access_complete):
        sequenced=self._sequence_client(instance_id)
        if sequenced is None:raise RuntimeError('sequenced SHM read abandon unavailable')
        try:return sequenced.abandon_retrieve(key,access_complete=access_complete)
        except TimeoutError:return False

    def close(self) -> None:
        self._recovery_stop.set()
        if self._recovery_thread is not None:
            self._recovery_thread.join(timeout=2)
            if self._recovery_thread.is_alive():
                raise RuntimeError('SHM recovery still active; retaining mapping')
        if self._shm is None:
            return
        self._unpin_shm_buffer()
        try:
            self._shm.close()
        finally:
            self._shm = None
            self._shm_buffer = None

    def _pin_shm_buffer(self) -> None:
        """Pin the SHM buffer as page-locked host memory via cudaHostRegister.

        Enables faster async D2H CUDA copies to the SHM region. If pinning is
        not available or fails, logs a warning and continues without pinning.
        """
        if self._shm_buffer is None or not torch_dev.is_available():
            return
        try:
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm_buffer))
        except Exception as exc:
            logger.warning(
                "Failed to get pointer for shm_name=%s: %r; "
                "D2H copies will be synchronous",
                self._shm_name,
                exc,
            )
            return
        if current_device_spec.pin_memory(ptr, self._pool_size):
            self._pinned = True
            self._pinned_ptr = ptr
            self._pinned_size = self._pool_size
        else:
            logger.warning(
                "pin_memory failed for shm_name=%s ptr=%#x size=%d; "
                "D2H copies will be synchronous",
                self._shm_name,
                ptr,
                self._pool_size,
            )

    def _unpin_shm_buffer(self) -> None:
        """Unpin the SHM buffer if it was previously pinned via cudaHostRegister."""
        if not self._pinned or self._pinned_ptr == 0:
            return
        current_device_spec.unpin_memory(self._pinned_ptr)
        self._pinned = False
        self._pinned_ptr = 0
        self._pinned_size = 0
