# SPDX-License-Identifier: Apache-2.0
"""Bounded, incarnation-scoped scheduler ownership across long queue waits."""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import session as session_module
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Advance wall/monotonic clocks independently without real waiting."""
    values = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr(session_module.time, "time", lambda: values["wall"])
    monkeypatch.setattr(session_module.time, "monotonic", lambda: values["mono"])
    return values


def manager(ttl: float = 600, maximum: float = 14400) -> SessionManager:
    """Construct a manager with explicit bounds and no background thread."""
    return SessionManager(
        TokenHasher(chunk_size=4, hash_algorithm="blake3"),
        ttl=ttl, cleanup_interval=None, max_lease_lifetime=maximum,
    )


def test_live_request_survives_creation_ttl_without_new_read_owner(clock: dict[str, float]) -> None:
    sessions = manager()
    original = sessions.get_or_create("queued")
    original.created_at = clock["wall"]
    owner = original.bind_owned_read([])
    assert sessions.bind_liveness("queued", "lookup-1")
    for age in (500, 1000, 1500):
        clock.update(wall=1000 + age, mono=10 + age)
        assert sessions.renew_sessions([("queued", "lookup-1")]) == ["queued"]
        assert sessions.cleanup_expired() == 0
        assert sessions.get("queued") is original
        assert original.owned_read_state()[0] == owner


def test_idle_expiry_cancels_exact_owner_once(clock: dict[str, float]) -> None:
    sessions = manager()
    original = sessions.get_or_create("queued")
    owner = original.bind_owned_read([])
    cancelled = []
    sessions.destroy_listeners.append(
        lambda item: item.cancel_owned_reads(
            lambda keys, token: cancelled.append(token), final=True
        )
    )
    assert sessions.bind_liveness("queued", "lookup-1")
    clock["mono"] += 601
    assert sessions.renew_sessions([("queued", "lookup-1")]) == []
    assert sessions.cleanup_expired() == 1
    assert sessions.cleanup_expired() == 0
    assert cancelled == [owner]
    assert sessions.active_count() == 0


def test_heartbeat_cannot_extend_absolute_lifetime(clock: dict[str, float]) -> None:
    sessions = manager(ttl=600, maximum=1000)
    sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "lookup-1")
    for age in (500, 999):
        clock["mono"] = 10 + age
        assert sessions.renew_sessions([("queued", "lookup-1")]) == ["queued"]
    clock["mono"] = 1011
    assert sessions.renew_sessions([("queued", "lookup-1")]) == []
    assert sessions.bind_liveness("queued", "lookup-1") is False
    assert sessions.cleanup_expired() == 1


def test_removed_or_recreated_request_rejects_late_renewal(clock: dict[str, float]) -> None:
    sessions = manager()
    old = sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "lookup-1")
    assert sessions.remove("queued") is old
    assert sessions.renew_sessions([("queued", "lookup-1")]) == []
    assert sessions.get("queued") is None
    replacement = sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "lookup-2")
    clock["mono"] += 500
    assert sessions.renew_sessions([("queued", "lookup-1")]) == []
    assert sessions.bind_liveness("queued", "lookup-1") is False
    clock["mono"] += 101
    assert sessions.cleanup_expired() == 1
    assert sessions.get("queued") is None
    assert replacement is not old


def test_wrong_owner_and_missing_session_do_not_create_state(clock: dict[str, float]) -> None:
    sessions = manager()
    assert sessions.bind_liveness("absent", "owner") is False
    sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "") is False
    assert sessions.bind_liveness("queued", "lookup-1")
    assert sessions.renew_sessions([("queued", "wrong"), ("absent", "owner")]) == []
    assert sessions.active_count() == 1


def test_leased_deadline_ignores_wall_clock_jumps(clock: dict[str, float]) -> None:
    sessions = manager()
    sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "lookup-1")
    clock["wall"] += 100000
    clock["mono"] += 5
    assert sessions.cleanup_expired() == 0
    assert sessions.renew_sessions([("queued", "lookup-1")]) == ["queued"]


def test_repeated_binding_does_not_renew_and_legacy_age_is_unchanged(clock: dict[str, float]) -> None:
    sessions = manager()
    legacy = sessions.get_or_create("legacy")
    legacy.created_at = clock["wall"]
    sessions.get_or_create("leased")
    assert sessions.bind_liveness("leased", "lookup-1")
    clock["mono"] += 500
    assert sessions.bind_liveness("leased", "lookup-1")
    clock["mono"] += 101
    clock["wall"] += 601
    assert sessions.bind_liveness("legacy", "late") is False
    assert sessions.cleanup_expired() == 2


def test_concurrent_end_and_renew_never_resurrect(clock: dict[str, float]) -> None:
    sessions = manager()
    sessions.get_or_create("queued")
    assert sessions.bind_liveness("queued", "lookup-1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(sessions.renew_sessions, [("queued", "lookup-1")])
                for _ in range(20)]
        removed = pool.submit(sessions.remove, "queued")
        for job in jobs:
            job.result(timeout=3)
        assert removed.result(timeout=3) is not None
    assert sessions.get("queued") is None
    assert sessions.renew_sessions([("queued", "lookup-1")]) == []


@pytest.mark.parametrize("ttl,maximum", [(0, 10), (10, 0), (-1, 10),
                                          (float("nan"), 10),
                                          (10, float("inf"))])
def test_deadlines_must_be_positive_and_finite(ttl: float, maximum: float) -> None:
    with pytest.raises(ValueError):
        manager(ttl, maximum)
