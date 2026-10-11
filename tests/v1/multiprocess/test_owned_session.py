# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def make_session():
    session = Session("request", TokenHasher(chunk_size=4, hash_algorithm="blake3"))
    key = SimpleNamespace(model_name="m", world_size=1, worker_id=0,
                          token_ids=(1, 2, 3, 4), cache_salt="s", start=0, end=4)
    session.begin_lookup(key, (-1,))
    return session, key


def test_destroyed_session_cannot_publish_new_reservation():
    session, _ = make_session()
    session.bind_owned_read([])
    cancelled = []
    session.cancel_owned_reads(lambda keys, owner: cancelled.append(owner), final=True)
    assert len(cancelled) == 1
    assert session.owned_read_state() is None
    with pytest.raises(RuntimeError):
        session.bind_owned_read([])


def test_destruction_waits_for_inflight_reservation_publication():
    session, _ = make_session()
    owner = session.bind_owned_read([])
    entered, finish, destroyed = Event(), Event(), Event()
    order = []
    def publish():
        with session.owned_reservation_scope(owner):
            entered.set()
            assert finish.wait(3)
            order.append("publish")
    def destroy():
        session.cancel_owned_reads(lambda keys, token: order.append("cancel"), final=True)
        destroyed.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish)
        assert entered.wait(3)
        second = pool.submit(destroy)
        assert not destroyed.is_set()
        finish.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert order == ["publish", "cancel"]


def test_unknown_prefetch_and_foreign_scope_fail_closed():
    session, key = make_session()
    owner = session.bind_owned_read([])
    assert session.owned_read_for(key) is None
    session.record_prefetch_result(1, (0,))
    assert session.owned_read_for(key) == owner
    foreign = SimpleNamespace(**{**vars(key), "cache_salt": "other"})
    assert session.owned_read_for(foreign) is None


def test_same_request_new_session_never_reuses_owner():
    first, _ = make_session()
    second, _ = make_session()
    assert first.bind_owned_read([]) != second.bind_owned_read([])
