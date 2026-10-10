# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
import enum
import inspect
import itertools
import queue
import threading
import weakref

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.platform import EventNotifier, create_event_notifier

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int

# Invalid MessagePack opcode prevents collision with a serialized payload.
RPC_ERROR_FRAME = b"\xc1lmcache.rpc_error_v1"
RPC_ERROR_CAPABILITY = "rpc_error_v1"


class RemoteRequestError(RuntimeError):
    """The remote handler failed; successful response types are unchanged."""

class RequestFrameCapacityError(RuntimeError):
    """Serialized request exceeds configured pre-decoder admission."""

@dataclass
class _RequestAdmission:
    reservation: Any
    transferred: bool = False
    response_reservation: Any = None

    def release(self):
        if self.reservation is not None:
            self.reservation.release()
        if self.response_reservation is not None:
            self.response_reservation.release()


@dataclass(frozen=True)
class ResponseLimits:
    max_response_bytes: int
    total_response_bytes: int
    max_responses: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.max_response_bytes, self.total_response_bytes, self.max_responses
        )) or self.max_response_bytes > self.total_response_bytes:
            raise ValueError("invalid response byte/count limits")


@dataclass
class _QueuedResponse:
    frames: list[bytes]
    admission: Any = None


@dataclass(frozen=True)
class RequestFrameLimits:
    max_frame_bytes: int
    max_request_bytes: int
    max_frames: int
    max_header_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.max_frame_bytes, self.max_request_bytes,
            self.max_frames, self.max_header_bytes,
        )):
            raise ValueError("frame limits must be positive integers")

    def safe_response_prefix(self, frames):
        return len(frames) >= 3 and all(len(frames[i]) <= self.max_header_bytes for i in range(3))

    def validate(self, frames):
        if len(frames) > self.max_frames:
            raise RequestFrameCapacityError("request frame count exceeded")
        total = 0
        for frame in frames:
            if len(frame) > self.max_frame_bytes:
                raise RequestFrameCapacityError("request frame bytes exceeded")
            total += len(frame)
            if total > self.max_request_bytes:
                raise RequestFrameCapacityError("request total bytes exceeded")


def _response_prefix(frames: list[bytes]) -> list[bytes]:
    return frames[:-1] if frames[-1] == RPC_ERROR_FRAME else frames


# Helper functions
def encode_request_uid(uid: RequestUID) -> bytes:
    return msgspec.msgpack.encode(uid)


def decode_request_uid(b_uid: bytes) -> RequestUID:
    return msgspec.msgpack.decode(b_uid, type=RequestUID)


def unwrap_request_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> list[Any]:
    if len(b_payloads) != len(payload_clss):
        raise ValueError("Payload count does not match expected count")

    decoded_payloads = [
        msgspec_decode(payload, cls=cls)
        for payload, cls in zip(b_payloads, payload_clss, strict=False)
    ]
    return decoded_payloads


_SPECIAL_ENCODER_DECODERS = {
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    MemoryLayoutDesc: (
        get_customized_encoder(MemoryLayoutDesc),
        get_customized_decoder(MemoryLayoutDesc),
    ),
    dict[int, MemoryLayoutDesc]: (
        get_customized_encoder(dict[int, MemoryLayoutDesc]),
        get_customized_decoder(dict[int, MemoryLayoutDesc]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    # Defensive guard: coerce obj to the declared cls so that
    # e.g. a bool passed as int (or vice-versa) is encoded in the
    # wire format that msgspec_decode expects for that cls.
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    # Defensive guard: msgspec strict-validates wire format
    # (bool ≠ int in msgpack), but runtime type may not match
    # declared cls. Decode untyped, then coerce.
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


# Shared polling loop for MessageQueueClient instances


class _OpKind(enum.Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"


@dataclass
class _PollOp:
    kind: _OpKind
    client: "MessageQueueClient"
    done: threading.Event


class ClientPollingLoop:
    """Singleton polling loop shared by all MessageQueueClient instances.

    Instead of each client running its own daemon thread and zmq.Poller,
    a single loop polls all clients' DEALER sockets and dispatches
    inbound/outbound work.

    Use ``get_instance()`` / ``release_instance()`` for lifecycle
    management — the loop starts lazily on first client and stops
    automatically when the last client releases.
    """

    _instance: "ClientPollingLoop | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._is_finished = threading.Event()
        self._notifier: EventNotifier = create_event_notifier()
        self._ops_queue: queue.Queue[_PollOp] = queue.Queue()
        self._poller = zmq.Poller()
        self._poller.register(self._notifier.fileno(), zmq.POLLIN)
        self._socket_to_client: dict[zmq.Socket, "MessageQueueClient"] = {}
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-client-shared-loop"
        )
        self._thread.start()

    @classmethod
    def get_instance(cls) -> "ClientPollingLoop":
        """Get or create the singleton, incrementing the ref count.

        Returns:
            ClientPollingLoop: The shared polling loop instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ClientPollingLoop()
            cls._instance._ref_count += 1
            return cls._instance

    @classmethod
    def release_instance(cls) -> None:
        """Decrement the ref count; tear down the loop when it reaches 0."""
        with cls._instance_lock:
            inst = cls._instance
            if inst is None:
                return
            inst._ref_count -= 1
            if inst._ref_count > 0:
                return
            inst._is_finished.set()
            inst._notifier.notify()
            cls._instance = None

        inst._thread.join()
        inst._notifier.close()
        logger.debug("ClientPollingLoop shut down")

    def register(self, client: "MessageQueueClient") -> None:
        """Register a client's DEALER socket with the shared poller.

        Blocks until the loop thread has completed the registration.

        Args:
            client: The MessageQueueClient to register.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.REGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def unregister(self, client: "MessageQueueClient") -> None:
        """Unregister a client's DEALER socket from the shared poller.

        Blocks until the loop thread has completed the unregistration.

        Args:
            client: The MessageQueueClient to unregister.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.UNREGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def notify(self) -> None:
        """Wake the polling loop to process outbound tasks."""
        self._notifier.notify()

    def _process_ops(self) -> None:
        """Drain the ops queue and apply register/unregister to the poller."""
        try:
            while True:
                op = self._ops_queue.get_nowait()
                if op.kind is _OpKind.REGISTER:
                    self._poller.register(op.client.socket, zmq.POLLIN)
                    self._socket_to_client[op.client.socket] = op.client
                    logger.debug("Registered client socket %s", op.client.socket)
                elif op.kind is _OpKind.UNREGISTER:
                    self._poller.unregister(op.client.socket)
                    self._socket_to_client.pop(op.client.socket, None)
                    logger.debug("Unregistered client socket %s", op.client.socket)
                op.done.set()
        except queue.Empty:
            pass

    def _main_loop(self) -> None:
        """Unified poll loop for all registered clients."""
        notifier_fd = self._notifier.fileno()

        while not self._is_finished.is_set():
            self._process_ops()

            socks = dict(self._poller.poll(1000))

            # Outbound: shared notifier woke us — drain it, then flush
            # all clients' output queues.
            if socks.get(notifier_fd) and socks[notifier_fd] & zmq.POLLIN:
                self._notifier.consume()
                for client in self._socket_to_client.values():
                    client.process_outbound_task()

            # Inbound: dispatch each ready DEALER socket to its client.
            for sock, event in socks.items():
                if sock is notifier_fd:
                    continue
                if event & zmq.POLLIN:
                    owner = self._socket_to_client.get(sock)
                    if owner is not None:
                        try:
                            owner.process_inbound()
                        except Exception:
                            # This loop is shared by every client in the
                            # process, so one undecodable or unexpected frame
                            # must not terminate it: that would strand all
                            # other clients' pending and future requests.
                            logger.exception(
                                "process_inbound failed; dropping this frame "
                                "and continuing the shared polling loop"
                            )

        # Drain remaining ops so any waiting threads unblock.
        self._process_ops()


# Main classes
# These messages export no device or IPC resource. Abandoning a reply does not
# cancel the remote handler or release its cache locks; server lifecycle cleanup
# is still required after an unconfirmed lookup/status request.
_ABANDONABLE_REQUESTS = frozenset({
    RequestType.PING, RequestType.GET_EXPERIMENTAL, RequestType.GET_CHUNK_SIZE,
    RequestType.LOOKUP, RequestType.QUERY_PREFETCH_STATUS,
    RequestType.WAIT_PREFETCH_STATUS, RequestType.QUERY_PREFETCH_LOOKUP_HITS,
    RequestType.FREE_LOOKUP_LOCKS, RequestType.END_SESSION,
})


class _AbandonableMessagingFuture(MessagingFuture):
    def __init__(self, callback):
        super().__init__()
        self._abandon_callback = callback
        self._abandon_lock = threading.Lock()
        self._abandon_requested = False

    def abandon(self) -> bool:
        with self._abandon_lock:
            if self.query() or self._abandon_requested:
                return True
            if self._retained_references:
                return False
            self._abandon_requested = True
            return self._abandon_callback(self)

    def retain_reference(self, value: object) -> None:
        with self._abandon_lock:
            if self._abandon_requested:
                raise RuntimeError("Cannot retain resources on an abandoned request")
            super().retain_reference(value)


def abandon_messaging_future(future: MessagingFuture[Any] | None) -> bool:
    """Discard an unneeded control reply, never a device-operation future."""
    return isinstance(future, _AbandonableMessagingFuture) and future.abandon()


@dataclass
class _AbandonedRequest:
    request_uid: RequestUID
    future: MessagingFuture


class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any]
        request_type: RequestType
        request_payloads: list[Any]

    def __init__(self, server_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.DEALER)
        if server_url.startswith("tcp://"):
            # Application RPC timeouts do not close a half-open TCP pipe.
            # Let ZMTP detect absent traffic independently of request handlers
            # so a replaced Kubernetes Service endpoint can be reconnected.
            # Reconnection does not confirm or replay any pending operation;
            # its future and owned device resources keep their existing state.
            self.socket.setsockopt(zmq.HEARTBEAT_IVL, 1000)
            self.socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 5000)
        self.socket.connect(server_url)

        # Input queue
        self.input_queue: queue.Queue = queue.Queue()

        # Pending job's futures
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}
        self._rpc_errors = False
        self._closed = False
        self._submission_lock = threading.Lock()

        # Register with the shared polling loop
        self._polling_loop = ClientPollingLoop.get_instance()
        self._polling_loop.register(self)

    def process_outbound_task(self):
        try:
            while wrapped_request := self.input_queue.get_nowait():
                request_uid = wrapped_request.request_uid
                if isinstance(wrapped_request, _AbandonedRequest):
                    if self.pending_futures.get(request_uid) is wrapped_request.future:
                        self.pending_futures.pop(request_uid)
                        wrapped_request.future.set_exception(TimeoutError("Control reply abandoned"))
                    continue
                if getattr(wrapped_request.future, "_abandon_requested", False):
                    wrapped_request.future.set_exception(TimeoutError("Control request abandoned before send"))
                    continue
                try:
                    b_request_uid = msgspec_encode(request_uid, cls=RequestUID)
                    b_request_type = msgspec_encode(
                        wrapped_request.request_type, cls=RequestType
                    )
                    payload_classes = get_payload_classes(wrapped_request.request_type)
                    if len(payload_classes) != len(wrapped_request.request_payloads):
                        expected_classes = [cls.__name__ for cls in payload_classes]
                        actual_classes = [
                            type(payload).__name__
                            for payload in wrapped_request.request_payloads
                        ]
                        raise ValueError(
                            f"Payload count mismatch for request "
                            f"{wrapped_request.request_type}: "
                            f"expected {len(payload_classes)} payloads "
                            f"{expected_classes}, "
                            f"got {len(wrapped_request.request_payloads)} payloads "
                            f"{actual_classes}. "
                            f"This is likely caused by a version mismatch between "
                            f"the lmcache client and lmcache server."
                        )

                    b_payloads = [
                        msgspec_encode(payload, cls=cls)
                        for payload, cls in zip(
                            wrapped_request.request_payloads,
                            payload_classes,
                            strict=False,
                        )
                    ]
                    self.pending_futures[request_uid] = wrapped_request.future
                    self.socket.send_multipart(
                        [b_request_uid, b_request_type] + b_payloads
                        + ([RPC_ERROR_FRAME] if self._rpc_errors else [])
                    )
                except Exception as exc:
                    self.pending_futures.pop(request_uid, None)
                    wrapped_request.future.set_exception(exc)
        except queue.Empty:
            pass

    def process_inbound(self) -> None:
        """Process one inbound response from the server.

        Called by the shared ClientPollingLoop when the DEALER socket
        is readable.  Only touches ``pending_futures``, which is
        exclusively accessed from the loop thread.
        """
        msg = self.socket.recv_multipart()
        if not msg:
            logger.error("Empty response: no request UID to resolve")
            return
        try:
            request_uid = msgspec_decode(msg[0], cls=RequestUID)
        except Exception:
            logger.exception("Invalid response request UID; cannot resolve a future")
            return
        future = self.pending_futures.pop(request_uid, None)
        if future is None:
            return  # Late or duplicate response; no waiting request owns it.
        try:
            if len(msg) < 2:
                raise ValueError("Malformed response: missing request type")
            _, b_request_type, *b_response = msg
            request_type = msgspec_decode(b_request_type, cls=RequestType)
            response_cls = get_response_class(request_type)
            if b_response and b_response[0] == RPC_ERROR_FRAME:
                if not self._rpc_errors or len(b_response) != 2:
                    raise ValueError("Unexpected RPC error envelope")
                error = msgspec.msgpack.decode(b_response[1], type=list[str])
                if len(error) != 2:
                    raise ValueError("Malformed RPC error envelope")
                raise RemoteRequestError(f"{error[0]}: {error[1]}")
            if len(b_response) > 1:
                raise ValueError("Malformed response: extra payload frames")
            if b_response:
                response = msgspec_decode(b_response[0], cls=response_cls)
                if request_type == RequestType.GET_EXPERIMENTAL:
                    self._rpc_errors = RPC_ERROR_CAPABILITY in response
                future.set_result(response)
            else:
                future.set_result(None)
        except Exception as exc:
            # Once removed from pending_futures, every path must resolve it.
            # Otherwise a decode error strands the caller even after timeout.
            future.set_exception(exc)

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
        response_cls: Optional[T] = None,
    ) -> MessagingFuture[T]:
        """Submit a request to the server.

        Args:
            request_type (RequestType): The type of the request.
            request_payloads (list[Any]): The payloads of the request.
            response_cls (Optional[T]): The expected response class.
                This should be get from `get_response_class(request_type)`.

        Returns:
            MessagingFuture[T]: A future that will hold the response.
        """
        with self._submission_lock:
            if self._closed:
                raise RuntimeError("Message queue client is closed")
            request_uid = next(self._request_counter)
            if request_type in _ABANDONABLE_REQUESTS:
                owner = weakref.ref(self)
                def discard(future):
                    client = owner()
                    if client is None:
                        return False
                    with client._submission_lock:
                        if client._closed:
                            return False
                        client.input_queue.put(_AbandonedRequest(request_uid, future))
                        client._polling_loop.notify()
                    return True
                future = _AbandonableMessagingFuture(discard)
            else:
                future = MessagingFuture()
            self.input_queue.put(MessageQueueClient.WrappedRequest(
                request_uid=request_uid, future=future,
                request_type=request_type, request_payloads=request_payloads,
            ))
            self._polling_loop.notify()
        return future

    def close(self) -> None:
        with self._submission_lock:
            if self._closed:
                return
            self._closed = True
        self._polling_loop.unregister(self)
        # Unregister is a barrier: the polling thread no longer owns this map.
        for uid, future in list(self.pending_futures.items()):
            if isinstance(future, _AbandonableMessagingFuture) and not future._retained_references:
                self.pending_futures.pop(uid)
                future.set_exception(RuntimeError("Message queue client closed"))
        # Queued requests have not been sent and cannot be using remote resources.
        while True:
            try:
                queued = self.input_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(queued, _AbandonedRequest):
                queued.future.set_exception(RuntimeError("Message queue client closed before send"))
        ClientPollingLoop.release_instance()
        self.socket.close()


ResponseType = TypeVar("ResponseType", covariant=True)
StateType = TypeVar("StateType", covariant=True)


class RequestHandlerBase(Generic[ResponseType]):
    def __call__(self, payloads: list[bytes]):
        raise NotImplementedError

    def get_response_class(self) -> ResponseType:
        raise NotImplementedError

    def get_handler_type(self) -> HandlerType:
        raise NotImplementedError


class SyncRequestHandler(RequestHandlerBase[ResponseType]):
    """
    The handler for those "fast" functions that can be executed in the main loop
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.payload_clss = payload_clss
        self.response_cls = response_cls
        self.handler = handler

    def __call__(self, payloads: list[bytes]) -> ResponseType:
        return self.handler(*unwrap_request_payloads(payloads, self.payload_clss))

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.SYNC


class BlockingRequestHandler(RequestHandlerBase[ResponseType]):
    """
    Returns the future of the response.

    The ``executor`` field is initially ``None`` and must be assigned via
    :meth:`MessageQueueServer.add_normal_thread_pool` or
    :meth:`MessageQueueServer.add_affinity_thread_pool` before the server
    is started.
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.executor: ThreadPoolExecutor | AffinityThreadPool | None = None
        self.payload_clss = payload_clss
        self.handler = handler
        self.response_cls = response_cls

    def __call__(
        self, payloads: list[bytes], affinity_key: int = 0, admission=None,
        retain_admission: bool = False,
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "BlockingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        if admission is not None:
            def invoke():
                try:
                    return self.handler(*decoded_payloads)
                finally:
                    if not retain_admission:
                        admission.release()

            def cancelled(future):
                if future.cancelled() and not retain_admission:
                    admission.release()

            # A submit exception can occur after queue insertion/thread start.
            # Never refund on that uncertainty; execution/cancellation owns it.
            admission.transferred = True
            if isinstance(self.executor, AffinityThreadPool):
                future = self.executor.submit(invoke, affinity_key=affinity_key)
            else:
                future = self.executor.submit(invoke)
            future.add_done_callback(cancelled)
            return future
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler, *decoded_payloads, affinity_key=affinity_key
            )
        return self.executor.submit(self.handler, *decoded_payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.BLOCKING


class NonBlockingRequestHandler(Generic[ResponseType, StateType]):
    """
    The handler for the "fire and probe" functions that launch async tasks
    and have special mechanism to probe the task status.

    It requires 2 callables as the input:
    - the first one is to launch the async task. This function should return
        a 'state handle' that can be used to probe the task status later.
    - the second one is to probe the task status and get the return value
        with the 'state handle' returned by the first function.
    """

    # TODO: implement this in the future versions if needed
    pass


class MessageQueueServer:
    def _reserve_response(self):
        from lmcache.v1.multiprocess.resource_budget import Usage
        return self.response_budget.account().reserve(Usage(
            metadata_bytes=self.response_limits.max_response_bytes, operations=1))

    def _admit_request(self, frames):
        from lmcache.v1.multiprocess.resource_budget import Usage
        if self.request_budget is None and self.response_budget is None:
            return None
        admission = _RequestAdmission(None)
        try:
            if self.request_budget is not None:
                admission.reservation = self.request_budget.account().reserve(
                    Usage(metadata_bytes=sum(map(len, frames)), operations=1))
            if self.response_budget is not None:
                admission.response_reservation = self._reserve_response()
            return admission
        except Exception:
            admission.release()
            raise

    def _queue_response(self, frames, admission=None):
        with self._response_lock:
            if self._responses_closed:
                # The handler has finished; only its transport admission is
                # released here. Closing is not remote-operation cancellation.
                if admission is not None:
                    admission.release()
                return
            if admission is not None:
                admission.transferred = True
            if self.response_limits is not None:
                if admission is None or admission.response_reservation is None:
                    raise RuntimeError("response published without pre-dispatch capacity")
                if sum(map(len, frames)) > self.response_limits.max_response_bytes:
                    # Handler effects may already be committed. Do not replace
                    # this with an RPC failure or refund and admit more work.
                    # Keep only the bounded admission record until close.
                    self._oversized_responses.append(admission)
                    logger.error("Oversized RPC response quarantined; result unknown to peer")
                    return
            self.output_queue.put(_QueuedResponse(frames, admission))
            self._output_efd.notify()

    def _send_error(self, prefix_frames: list[bytes], exc: Exception, admission=None) -> None:
        # Opt-in is per request, so legacy clients never see an error envelope.
        if prefix_frames[-1] != RPC_ERROR_FRAME:
            if admission is not None:
                admission.release()
            return
        if self.response_budget is not None and admission is None:
            from lmcache.v1.multiprocess.resource_budget import ResourceCapacityError
            try:
                admission = _RequestAdmission(None, response_reservation=self._reserve_response())
            except ResourceCapacityError:
                # No handler was dispatched by this rejection path. Do not
                # create an unbounded secondary queue of capacity errors.
                self.dropped_admission_errors += 1
                return
        payload = msgspec.msgpack.encode([type(exc).__name__, str(exc)[:1024]])
        self._queue_response(_response_prefix(prefix_frames) + [RPC_ERROR_FRAME, payload], admission)

    def _reap_native_responses(self):
        pending = []
        for tracker, admission in self._native_responses:
            if tracker.done:
                admission.release()
            else:
                pending.append((tracker, admission))
        self._native_responses = pending

    def _send_tracked_response(self, response):
        # Explicit Frames avoid pyzmq's small-message copy threshold. Track
        # every multipart frame, not merely send_multipart's last tracker.
        frames = [zmq.Frame(part, copy=False, track=True) for part in response.frames]
        tracker = zmq.MessageTracker(*(frame.tracker for frame in frames))
        # Publish ownership before entering native transport. An exception
        # after an effect must not leave close() treating this as unsent.
        self._native_responses.append((tracker, response.admission))
        try:
            self.socket.send_multipart(frames, flags=zmq.DONTWAIT, copy=False)
        except zmq.Again:
            self._native_responses.pop()
            raise
        except zmq.ZMQError as exc:
            if exc.errno == zmq.EHOSTUNREACH:
                self._native_responses.pop()
            else:
                self._pending_response = None
            raise
        except BaseException:
            self._pending_response = None
            raise
        return tracker

    def _drain_responses(self):
        # The polling thread exclusively owns the popped response. Keep it
        # reachable on EAGAIN or an uncertain transport exception.
        for _ in range(128):
            if self._pending_response is None:
                try:
                    self._pending_response = self.output_queue.get_nowait()
                except queue.Empty:
                    return
            response = self._pending_response
            try:
                if self.response_budget is not None:
                    tracker = self._send_tracked_response(response)
                    self._pending_response = None
                    self._reap_native_responses()
                    continue
                self.socket.send_multipart(response.frames, flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            except zmq.ZMQError as exc:
                if exc.errno != zmq.EHOSTUNREACH:
                    raise
                # ROUTER_MANDATORY reports an absent route before enqueue.
                # Discard only the serialized response: the handler may have
                # committed, so this is NOT cancellation or a failed RPC reply.
                self.undeliverable_responses += 1
                logger.warning("Discarding RPC response for disconnected route")
            if response.admission is not None:
                response.admission.release()
            self._pending_response = None
        if not self.output_queue.empty():
            self._output_efd.notify()

    def __init__(self, bind_url: str, context: zmq.Context, *, frame_limits=None, request_budget=None, response_limits=None):
        # Socket
        self.ctx = context
        self.frame_limits = frame_limits
        self.request_budget = request_budget
        self.response_limits = response_limits
        self.response_budget = None
        if response_limits is not None:
            from lmcache.v1.multiprocess.resource_budget import ResourceBudget, Usage
            self.response_budget = ResourceBudget(
                Usage(metadata_bytes=response_limits.max_response_bytes, operations=1),
                Usage(metadata_bytes=response_limits.total_response_bytes,
                      operations=response_limits.max_responses))
        self.socket = self.ctx.socket(zmq.ROUTER)
        # Without mandatory routing, ROUTER silently drops at HWM or when the
        # identity is absent, defeating the owned pending-response path.
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(bind_url)
        # Use a cross-platform Notifier instead of zmq PUSH/PULL sockets
        # because blocking handler callbacks run on ThreadPoolExecutor
        # threads, and zmq sockets are not thread-safe. Notifier.notify()
        # is atomic (eventfd on Linux, self-pipe elsewhere).
        self._output_efd = create_event_notifier()
        self.output_queue: queue.Queue = queue.Queue()
        self._pending_response = None
        self._native_responses = []
        self._response_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._responses_closed = False
        self.undeliverable_responses = 0
        self.dropped_admission_errors = 0
        self._oversized_responses = []

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self._output_efd.fileno(), zmq.POLLIN)

        # Main loop thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-server-thread"
        )

        # Registered handlers: request_type -> (payload_cls, handler)
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}

        # Thread pools assigned via add_normal_thread_pool / add_affinity_thread_pool
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []

    def _publish_handler_response(self, response, response_cls, prefix_frames, admission):
        prefix = _response_prefix(prefix_frames)
        if self.response_limits is None:
            encoded = msgspec_encode(response, cls=response_cls)
            self._queue_response(prefix + [encoded] if response is not None else prefix, admission)
            return
        try:
            from lmcache.v1.multiprocess.response_snapshot import snapshot_response
            if response is None:
                frames = prefix
            else:
                if response_cls in _SPECIAL_ENCODER_DECODERS:
                    raise ValueError("custom response encoder needs bounded snapshot adapter")
                value = response_cls(response) if response_cls in (bool, int) else response
                snapshot, _ = snapshot_response(
                    value, self.response_limits.max_response_bytes - sum(map(len, prefix)))
                encoded = msgspec_encode(snapshot, cls=response_cls)
                frames = prefix + [encoded]
        except Exception:
            # The handler returned successfully: its effects may be committed.
            # Keep a bounded receipt, never send a false operation-failure reply.
            with self._response_lock:
                admission.transferred = True
                if self._responses_closed:
                    admission.release()
                else:
                    self._oversized_responses.append(admission)
            logger.error("RPC response preparation quarantined; result unknown to peer")
            return
        self._queue_response(frames, admission)

    def _call_sync_handler(
        self,
        handler_entry: SyncRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
        admission=None,
    ) -> Any:
        """
        Call the sync handler and send the response back to the client.

        Args:
            handler_entry (SyncRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
        """
        response = handler_entry(payloads)
        response_cls = handler_entry.get_response_class()
        self._publish_handler_response(response, response_cls, prefix_frames, admission)

    def _call_blocking_handler(
        self,
        handler_entry: BlockingRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
        admission=None,
    ) -> Any:
        """
        Call the blocking handler in a separate thread and send the response
        back to the client.

        Args:
            handler_entry (BlockingRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
                prefix_frames[0] is the zmq identity used as affinity key.
        """
        affinity_key = hash(prefix_frames[0])
        future = handler_entry(payloads, affinity_key=affinity_key, admission=admission,
                               retain_admission=True)

        def _notify_response(fut: Future):
            try:
                response = fut.result()
                response_cls = handler_entry.get_response_class()

            except Exception as exc:
                logger.exception("Error in blocking handler")
                self._send_error(prefix_frames, exc, admission)
                return
            # Publication is outside the handler-error boundary. If notify
            # fails after insertion, never enqueue a second response owner.
            self._publish_handler_response(response, response_cls, prefix_frames, admission)

        future.add_done_callback(_notify_response)

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
        admission=None,
    ) -> Any:
        match handler_entry.get_handler_type():
            case HandlerType.SYNC:
                assert isinstance(handler_entry, SyncRequestHandler)
                self._call_sync_handler(handler_entry, payloads, prefix_frames, admission)
            case HandlerType.BLOCKING:
                assert isinstance(handler_entry, BlockingRequestHandler)
                self._call_blocking_handler(handler_entry, payloads, prefix_frames, admission)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError("Unknown handler type")

    def _main_loop(self):
        output_fd = self._output_efd.fileno()
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            self._reap_native_responses()
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(output_fd, None)

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                if len(msg) < 3:
                    logger.error("Request missing identity, UID or request type")
                    continue

                # recv_multipart has already allocated raw frames. These caps
                # bound decoder input, not libzmq buffers or queued requests.
                if self.frame_limits is not None and not self.frame_limits.safe_response_prefix(msg):
                    logger.error("Dropping RPC with oversized response header")
                    continue
                identity, b_request_uid, b_request_type = msg[:3]
                prefix_frames = [identity, b_request_uid, b_request_type]
                error_opt_in = len(msg) > 3 and msg[-1] == RPC_ERROR_FRAME
                if error_opt_in:
                    prefix_frames.append(RPC_ERROR_FRAME)
                admission = None
                try:
                    if self.frame_limits is not None:
                        self.frame_limits.validate(msg)
                    admission = self._admit_request(msg)
                    payloads = msg[3:-1] if error_opt_in else msg[3:]
                    request_type = msgspec_decode(b_request_type, cls=RequestType)
                    if handler_entry := self.handlers.get(request_type):
                        self._call_handler(
                            handler_entry=handler_entry,
                            payloads=payloads,
                            prefix_frames=prefix_frames,
                            admission=admission,
                        )
                    else:
                        raise ValueError(f"No handler registered for {request_type}")
                except Exception as exc:
                    logger.exception("Error handling RPC request")
                    # A submit failure after ownership transfer is uncertain:
                    # only its eventual callback may release the reservation.
                    if admission is None or not admission.transferred:
                        self._send_error(prefix_frames, exc, admission)
                finally:
                    if admission is not None and not admission.transferred:
                        admission.release()

            # Send the responses
            if outbound_state and outbound_state & zmq.POLLIN:
                # Consume the notifier counter (resets atomically)
                self._output_efd.consume()

                self._drain_responses()
            elif self._pending_response is not None:
                self._drain_responses()

    def _inspect_handler_signature(self, request_type: RequestType, handler) -> bool:
        """Inspect the handler signature to ensure it matches the expected
        payload classes.

        Args:
            handler (callable): The handler function.

        Returns:
            bool: True if the signature matches, False otherwise.
        """

        def same_type(a, b) -> bool:
            if a is None:
                a = type(None)
            if b is None:
                b = type(None)
            return a == b

        sig = inspect.signature(handler)
        hints = get_type_hints(handler)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        payload_clss = get_payload_classes(request_type)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d arguments, but got %d",
                request_type,
                len(payload_clss),
                len(params),
            )
            return False

        for i, (param, expected_cls) in enumerate(
            zip(params, payload_clss, strict=False)
        ):
            ann = hints.get(param.name, param.annotation)
            if not same_type(ann, expected_cls):
                logger.error(
                    "Handler for %s argument %d expects type %s, but got %s",
                    request_type,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(request_type)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return type %s, but got %s",
                request_type,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler,
    ) -> None:
        """Register a handler for a specific request type.

        Args:
            request_type (RequestType): The type of the request to handle.
            payload_clss (list[Any]): The expected payload classes for the request.
                This should be get from `get_payload_classes(request_type)`.
            handler (callable): The handler function that takes the payloads
                as arguments.
        """
        if not self._inspect_handler_signature(request_type, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {request_type}"
            )

        match handler_type:
            case HandlerType.SYNC:
                self.add_sync_handler(request_type, payload_clss, handler)
            case HandlerType.BLOCKING:
                self.add_blocking_handler(request_type, payload_clss, handler)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = SyncRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_blocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = BlockingRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        request_types: list[RequestType],
        method_name: str,
    ) -> None:
        """Validate that all request types are registered BlockingRequestHandlers."""
        for request_type in request_types:
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(
                    f"No handler registered for request type: {request_type}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {request_type} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler. "
                    f"Only blocking handlers can use thread pools."
                )

    def add_normal_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign a ThreadPoolExecutor to specific request types.

        Use this for non-GPU blocking handlers (e.g. LOOKUP, END_SESSION).

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().  Each request_type must
        already be registered as a BlockingRequestHandler; otherwise a
        ValueError or TypeError is raised.

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_normal_thread_pool")
        if not request_types:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign an AffinityThreadPool to specific request types.

        Use this for GPU-bound blocking handlers (e.g. STORE, RETRIEVE).
        Requests from the same zmq client identity are always dispatched
        to the same worker thread, eliminating the need for per-instance
        GPU transfer locks.

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_affinity_thread_pool")
        if not request_types:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def start(self):
        # Validate all blocking handlers have an executor assigned
        for rt, handler in self.handlers.items():
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    f"assigned. Call add_normal_thread_pool or "
                    f"add_affinity_thread_pool before start()."
                )
        self.worker_thread.start()

    def close(self) -> None:
        with self._close_lock:
            if self._responses_closed:
                return
            self.is_finished.set()
            self._output_efd.notify()
            if self.worker_thread.is_alive():
                self.worker_thread.join()
            # The polling thread no longer owns the socket/pending response.
            # Serialize late callback publication with notifier destruction.
            with self._response_lock:
                self._responses_closed = True
                self.socket.close(linger=0)
                # Closing a ROUTER does not prove an inproc peer released its
                # borrowed frames. Unfinished sends stay charged/quarantined.
                self._reap_native_responses()
                if self._pending_response is not None:
                    if self._pending_response.admission is not None:
                        self._pending_response.admission.release()
                    self._pending_response = None
                while True:
                    try:
                        response = self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                    if response.admission is not None:
                        response.admission.release()
                self._output_efd.close()
                for admission in self._oversized_responses:
                    admission.release()
                self._oversized_responses.clear()
            for pool in self.extra_pools:
                pool.shutdown(wait=False)
