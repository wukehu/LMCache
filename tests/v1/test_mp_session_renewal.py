# SPDX-License-Identifier: Apache-2.0
"""Scheduler liveness uses per-request credentials on every backing MP."""

# Standard
from typing import Any, Callable, ClassVar
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_module
from lmcache.v1.multiprocess.futures import MessagingFuture

# Local
from test_mp_nonblocking_lookup import AdapterFactory, Client, ready
from test_mp_nonblocking_lookup import make_adapter as make_adapter


class ManualHeartbeat:
    """Expose periodic ticks without timing-dependent background threads."""

    instances: ClassVar[list["ManualHeartbeat"]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.callback = lambda: True
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def register_periodic_callback(self, callback: Callable[[], bool]) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def tick(self) -> bool:
        assert self.started and not self.stopped
        return self.callback()


@pytest.fixture
def leased(monkeypatch: pytest.MonkeyPatch, make_adapter: AdapterFactory) -> AdapterFactory:
    monkeypatch.setenv("LMCACHE_MP_OWNED_READ_LEASE", "1")
    ManualHeartbeat.instances = []
    monkeypatch.setattr(adapter_module, "HeartbeatThread", ManualHeartbeat)

    def lookup(client: Client, key: Any, tp_size: int, credential: str) -> MessagingFuture[Any]:
        client.credential = credential
        client.renewals = []
        return client.lookup(key, tp_size)

    def renew(client: Client, credentials: list[tuple[str, str]]) -> MessagingFuture[Any]:
        client.renewals.append(credentials)
        return ready([request_id for request_id, _ in credentials])

    monkeypatch.setattr(Client, "lookup_leased", lookup, raising=False)
    monkeypatch.setattr(Client, "renew_sessions", renew, raising=False)
    return make_adapter


def test_two_mp_renew_only_after_lookup_ack_and_keep_after_result_cleanup(leased: AdapterFactory) -> None:
    adapter, clients = leased(count=2)
    adapter.maybe_submit_lookup_request("queued", list(range(256)))
    assert len(ManualHeartbeat.instances) == 2
    assert clients[0].credential == clients[1].credential
    for heartbeat, client in zip(ManualHeartbeat.instances, clients, strict=True):
        assert heartbeat.tick()
        assert client.renewals == []
    for client in clients:
        client.ack.set_result(None)
    adapter.cleanup_lookup_result("queued")
    for heartbeat, client in zip(ManualHeartbeat.instances, clients, strict=True):
        assert heartbeat.tick()
        assert client.renewals == [[("queued", client.credential)]]
    adapter.end_session("queued")
    for heartbeat, client in zip(ManualHeartbeat.instances, clients, strict=True):
        assert heartbeat.tick()
        assert len(client.renewals) == 1
        assert client.ends == ["queued"]


def test_new_session_uses_new_credential_and_same_live_request_keeps_deadline(leased: AdapterFactory) -> None:
    adapter, clients = leased()
    adapter.maybe_submit_lookup_request("r", list(range(128)))
    first = clients[0].credential
    clients[0].ack.set_result(None)
    adapter.cleanup_lookup_result("r")
    adapter.maybe_submit_lookup_request("r", list(range(256)))
    assert clients[0].credential == first
    adapter.cleanup_lookup_result("r")
    adapter.end_session("r")
    adapter.cleanup_lookup_result("r")
    adapter.maybe_submit_lookup_request("r", list(range(128)))
    assert clients[0].credential != first


def test_finish_before_first_renewal_does_not_leave_liveness(leased: AdapterFactory) -> None:
    adapter, clients = leased()
    adapter.maybe_submit_lookup_request("r", list(range(128)))
    clients[0].ack.set_result(None)
    adapter.cleanup_lookup_result("r")
    adapter.end_session("r")
    assert ManualHeartbeat.instances[0].tick()
    assert clients[0].renewals == []


def test_shutdown_stops_threads_before_transport_close_and_never_restarts(leased: AdapterFactory) -> None:
    adapter, clients = leased()
    adapter.maybe_submit_lookup_request("r", list(range(128)))
    clients[0].ack.set_result(None)
    order = []
    clients[0].close = lambda: order.append(
        all(heartbeat.stopped for heartbeat in ManualHeartbeat.instances)
    )
    adapter.shutdown()
    assert order == [True]
    adapter.maybe_submit_lookup_request("after-close", list(range(128)))
    assert len(ManualHeartbeat.instances) == 1


def test_late_periodic_callback_cannot_publish_healthy_after_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "send_ping", lambda *args, **kwargs: True)
    health = threading.Event()
    entered, release = threading.Event(), threading.Event()
    heartbeat = adapter_module.HeartbeatThread(
        req_client=None, health_event=health, interval=10
    )

    def renew() -> bool:
        entered.set()
        assert release.wait(3)
        return True

    heartbeat.register_periodic_callback(renew)
    try:
        heartbeat.start()
        assert entered.wait(3)
        heartbeat.stop(timeout=0.01)
        release.set()
        deadline = time.monotonic() + 3
        while heartbeat.total_runs == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.total_runs == 1
        assert not health.is_set()
    finally:
        release.set()
        heartbeat.stop(timeout=3)
