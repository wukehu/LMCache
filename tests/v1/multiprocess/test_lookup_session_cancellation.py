# SPDX-License-Identifier: Apache-2.0
"""Cancellation must retire lookup bookkeeping and reject late publication."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess import session as session_module
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def cache_key() -> IPCCacheServerKey:
    """Return a one-chunk scheduler lookup with explicit reader count."""
    return IPCCacheServerKey.from_token_ids(
        "model", 1, None, [1, 2, 3, 4], end=4, request_id="request",
        cache_salt="cancel-test", num_kv_readers=1,
    )


class ControlledStorage:
    """Gate storage completion while exercising the real lookup/session APIs."""

    def __init__(self) -> None:
        self.before_query: Callable[[], None] | None = None
        self.before_submit: Callable[[], None] | None = None
        self.before_wait: Callable[[], None] | None = None
        self.cancelled: list[str] = []

    def supports_owned_reads(self) -> bool:
        """Advertise the owned-read protocol used by this integration branch."""
        return True

    def submit_prefetch_task(self, spec: Any, external_request_id: str,
                             read_owner: str) -> PrefetchHandle:
        """Return a completed one-key L1 handle without allocating GPU memory."""
        if self.before_submit is not None:
            self.before_submit()
        return PrefetchHandle(-1, external_request_id, (0,), 1, 1, time.monotonic())

    def wait_prefetch_status(self, handle: PrefetchHandle, timeout: float) -> bool:
        """Report readiness; the subsequent query retains the same captured handle."""
        if self.before_wait is not None:
            self.before_wait()
        return True

    def query_prefetch_status(self, handle: PrefetchHandle) -> Bitmap:
        """Wait on the optional barrier, then publish a real native bitmap."""
        hook, self.before_query = self.before_query, None
        if hook is not None:
            hook()
        result = Bitmap(1)
        result.set(0)
        return result

    def cancel_owned_read(self, keys: Any, owner: str) -> None:
        """Record cancellation without simulating completion of GPU copies."""
        self.cancelled.append(owner)

    def touch_l1_keys(self, keys: Any) -> None:
        """Accept the end-session cache-recency update."""


@pytest.fixture
def lookup(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a real lookup module and manager with controlled storage only."""
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.lookup.register_gauge",
        lambda *args, **kwargs: None,
    )
    storage = ControlledStorage()
    sessions = SessionManager(TokenHasher(4, "blake3"), cleanup_interval=None)
    registry = MagicMock()
    registry.find.return_value = SimpleNamespace(dtypes=(), shapes=())
    registry.find_attn_desc.return_value = AttnWindowDesc([-1])
    registry.find_group_layout_descs.return_value = {0: object()}
    bus = MagicMock()
    bus.has_subscribers.return_value = False
    ctx = SimpleNamespace(storage_manager=storage, session_manager=sessions,
                          token_hasher=TokenHasher(4, "blake3"), chunk_size=4,
                          layout_desc_registry=registry, event_bus=bus)
    return SimpleNamespace(module=LookupModule(ctx), sessions=sessions,
                           storage=storage, bus=bus)


def test_normal_completion_consumes_job_and_removes_session(lookup: SimpleNamespace) -> None:
    """Control: ordinary lookup/query/end consumes both public lifecycle objects."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    assert lookup.module.query_prefetch_status("request") == 1
    lookup.module.end_session("request")
    assert lookup.sessions.get("request") is None
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


def test_end_before_poll_retires_prefetch_job(lookup: SimpleNamespace) -> None:
    """An abandoned request must not need another status query for cleanup."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    lookup.module.end_session("request")
    assert lookup.sessions.get("request") is None
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


@pytest.mark.parametrize("reuse_request_id", [False, True])
@pytest.mark.parametrize("status_api", ["query", "wait"])
def test_late_status_cannot_revive_or_mutate_new_session(
    lookup: SimpleNamespace, reuse_request_id: bool, status_api: str,
) -> None:
    """A completion crossing END may only act on its original incarnation."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    entered, resume = Event(), Event()

    def block() -> None:
        """Pause after lookup has captured its old job, before result publication."""
        entered.set()
        assert resume.wait(5)

    lookup.storage.before_query = block
    with ThreadPoolExecutor(max_workers=1) as pool:
        method = (lookup.module.query_prefetch_status if status_api == "query"
                  else lambda rid: lookup.module.wait_prefetch_status(rid, 1))
        pending = pool.submit(method, "request")
        try:
            assert entered.wait(5)
            lookup.module.end_session("request")
            if reuse_request_id:
                lookup.module.lookup_leased(cache_key(), 1, "new")
                replacement = lookup.sessions.get("request")
                assert replacement is not None
                assert replacement.prefetch_hit_chunks == -1
        finally:
            resume.set()
        pending.result(timeout=5)
    if reuse_request_id:
        assert lookup.sessions.get("request") is replacement
        assert replacement.prefetch_hit_chunks == -1, "old completion mutated new session"
        assert lookup.module.report_status()["active_prefetch_jobs"] == 1
    else:
        assert lookup.sessions.get("request") is None, "late status revived ended session"


def test_idle_expiration_retires_unconsumed_job(
    lookup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded liveness expiration uses the same exact-lifetime cleanup as END."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    later = time.monotonic() + 601
    monkeypatch.setattr(session_module.time, "monotonic", lambda: later)
    assert lookup.sessions.cleanup_expired() == 1
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


def test_cancel_during_submission_cannot_register_late_job(lookup: SimpleNamespace) -> None:
    """Removal during reservation publication must not leave an unreachable job."""
    entered, resume, removed = Event(), Event(), Event()

    def block() -> None:
        """Hold the normal session reservation scope until removal has started."""
        entered.set()
        assert resume.wait(5)

    lookup.storage.before_submit = block
    lookup.sessions.destroy_listeners.insert(0, lambda session: removed.set())
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(lookup.module.lookup_leased, cache_key(), 1, "old")
        try:
            assert entered.wait(5)
            ending = pool.submit(lookup.module.end_session, "request")
            assert removed.wait(5)
        finally:
            resume.set()
        pending.result(timeout=5)
        ending.result(timeout=5)
    assert lookup.sessions.get("request") is None
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


def test_two_status_readers_consume_owned_result_once(lookup: SimpleNamespace) -> None:
    """A second result reader must not publish or consume a replacement job."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    entered, resume = Event(), Event()

    def block() -> None:
        """Pause one reader while another consumes the same public lookup job."""
        entered.set()
        assert resume.wait(5)

    lookup.storage.before_query = block
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lookup.module.query_prefetch_status, "request")
        try:
            assert entered.wait(5)
            second = pool.submit(lookup.module.query_prefetch_status, "request")
            assert second.result(timeout=5) == 1
        finally:
            resume.set()
        assert first.result(timeout=5) == 0
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


@pytest.mark.parametrize(
    "reason", ["no_gpu_context", "empty_chunk_hashes", "no_group_layout_descs"]
)
def test_end_retires_early_exit_job(lookup: SimpleNamespace, reason: str) -> None:
    """Even a lookup with no usable cache context belongs to its original session."""
    key = cache_key()
    if reason == "no_gpu_context":
        lookup.module.context.layout_desc_registry.find.return_value = None
    elif reason == "empty_chunk_hashes":
        key = replace(key, token_ids=(), end=0)
    else:
        lookup.module.context.layout_desc_registry.find_group_layout_descs.return_value = {}
    lookup.module.lookup_leased(key, 1, "old")
    assert lookup.module.report_status()["active_prefetch_jobs"] == 1
    lookup.module.end_session("request")
    assert lookup.sessions.get("request") is None
    assert lookup.module.report_status()["active_prefetch_jobs"] == 0


def test_old_wait_does_not_query_replacement_job(lookup: SimpleNamespace) -> None:
    """A waiting handler must query its captured handle, not refetch by request ID."""
    lookup.module.lookup_leased(cache_key(), 1, "old")
    entered, resume = Event(), Event()

    def block() -> None:
        """Pause the readiness wait before a result query can start."""
        entered.set()
        assert resume.wait(5)

    lookup.storage.before_wait = block
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(lookup.module.wait_prefetch_status, "request", 1)
        try:
            assert entered.wait(5)
            lookup.module.end_session("request")
            lookup.module.lookup_leased(cache_key(), 1, "new")
            replacement = lookup.sessions.get("request")
        finally:
            resume.set()
        assert old.result(timeout=5) == 0
    assert replacement.prefetch_hit_chunks == -1
    assert lookup.module.report_status()["active_prefetch_jobs"] == 1
