# SPDX-License-Identifier: Apache-2.0
"""Real ZMQ serialization/dispatch for session liveness, without GPU transfer."""

# Standard
from types import SimpleNamespace
import uuid

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess import session as session_module
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.zmq_impl.client import ZmqMultiprocessClient
from lmcache.v1.multiprocess.transport.zmq_impl.server import (
    add_handler_helper,
    build_zmq_request_server,
    get_zmq_handler_specs,
)


class LookupWithoutGPU(LookupModule):
    """Replace only prefetch submission; keep real liveness and END handlers."""

    def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        session = self.context.session_manager.get(key.request_id)
        session.bind_owned_read([])


def test_real_zmq_liveness_retains_owner_and_end_rejects_late_credentials(monkeypatch) -> None:
    clock = {"mono": 10.0}
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock["mono"])
    sessions = SessionManager(
        TokenHasher(chunk_size=4, hash_algorithm="blake3"), cleanup_interval=None
    )
    cancelled = []
    module = LookupWithoutGPU(SimpleNamespace(
        session_manager=sessions,
        storage_manager=SimpleNamespace(
            supports_owned_reads=lambda: True,
            cancel_owned_read=lambda keys, owner: cancelled.append(owner),
        ),
        event_bus=SimpleNamespace(publish=lambda event: None),
    ))
    # Exercise the actual factory's SYNC/NORMAL partition, not only manual
    # registration below. Inserting a SYNC handler in NORMAL must fail here.
    factory_server = build_zmq_request_server(
        [module], MPServerConfig(host="127.0.0.1", port=0)
    )
    try:
        factory_server.start()
    finally:
        factory_server.close()
    context = zmq.Context.instance()
    url = "inproc://session-liveness-" + uuid.uuid4().hex
    server = MessageQueueServer(url, context)
    for spec in get_zmq_handler_specs(module):
        if spec.request_type in {
            RequestType.LOOKUP_LEASED, RequestType.RENEW_SESSIONS,
            RequestType.END_SESSION,
        }:
            add_handler_helper(server, spec.request_type, spec.handler)
    server.add_normal_thread_pool(
        [RequestType.LOOKUP_LEASED, RequestType.END_SESSION], max_workers=2
    )
    server.start()
    client = ZmqMultiprocessClient(MessageQueueClient(url, context))
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, None, [1, 2, 3, 4], end=4, request_id="queued"
    )
    try:
        client.lookup_leased(key, 1, "incarnation-a").result(timeout=5)
        original = sessions.get("queued")
        owner = original.owned_read_state()[0]
        for age in (500, 1000, 1500):
            clock["mono"] = 10 + age
            assert client.renew_sessions(
                [("queued", "incarnation-a")]
            ).result(timeout=5) == ["queued"]
            assert sessions.cleanup_expired() == 0
            assert original.owned_read_state()[0] == owner
        client.end_session("queued").result(timeout=5)
        assert cancelled == [owner]
        assert sessions.active_count() == 0
        assert client.renew_sessions(
            [("queued", "incarnation-a")]
        ).result(timeout=5) == []
        client.lookup_leased(key, 1, "incarnation-b").result(timeout=5)
        assert sessions.get("queued") is not original
        assert client.renew_sessions(
            [("queued", "incarnation-a")]
        ).result(timeout=5) == []
        clock["mono"] += 601
        assert sessions.cleanup_expired() == 1
        assert len(cancelled) == 2 and cancelled[1] != owner
    finally:
        client.close()
        server.close()
        sessions.close()
