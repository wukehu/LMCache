# SPDX-License-Identifier: Apache-2.0
"""Two MP scopes with pipeline groups; CPU ownership, not GPU topology proof."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.read_completion import ReadCompletionRegistry
from lmcache.v1.distributed.read_lease import ReadLeaseLedger
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@pytest.mark.parametrize("cleanup", ["remove", "expire"])
@pytest.mark.parametrize("readers", [1, 2, 8])
def test_two_mp_pipeline_cleanup_and_late_callbacks(cleanup, readers):
    """Retiring one MP session cannot free other scopes or an in-flight copy."""
    managers, ledgers, registries, owners, handles = [], [], [], [], []
    key = SimpleNamespace(model_name="m", world_size=readers * 2,
                          worker_id=None, token_ids=(1, 2, 3, 4),
                          cache_salt="s", start=0, end=4)
    for mp in range(2):
        manager = SessionManager(TokenHasher(chunk_size=4, hash_algorithm="blake3"),
                                 ttl=0 if cleanup == "expire" else 600,
                                 cleanup_interval=None)
        ledger = ReadLeaseLedger()
        registry = ReadCompletionRegistry(ledger.complete)
        manager.destroy_listeners.append(
            lambda session, l=ledger: session.cancel_owned_reads(
                lambda keys, owner: l.cancel(owner), final=True))
        session = manager.get_or_create("same-request")
        session.begin_lookup(key, (mp,))
        owner = session.bind_owned_read([])
        ledger.reserve(owner, readers, deadline=1)
        assert not ledger.is_pinned(now=2)
        claim = ledger.acquire(owner, mp * readers)
        assert claim is not None
        handle = registry.register(claim)
        managers.append(manager)
        ledgers.append(ledger)
        registries.append(registry)
        owners.append(owner)
        handles.append(handle)
    assert owners[0] != owners[1]
    # Receipt delivery to the wrong MP must not consume either MP's pin.
    assert not registries[1].complete(handles[0])
    assert [r.pending_count() for r in registries] == [1, 1]
    if cleanup == "remove":
        managers[0].remove("same-request")
    else:
        assert managers[0].cleanup_expired() == 1
    assert managers[0].active_count() == 0
    assert managers[1].active_count() == 1
    assert ledgers[0].outstanding(owners[0]) == 1
    assert ledgers[1].outstanding(owners[1]) == readers
    # The same request ID can be reused while its prior GPU callback is late.
    replacement = managers[0].get_or_create("same-request")
    replacement.begin_lookup(key, (0,))
    new_owner = replacement.bind_owned_read([])
    assert new_owner not in owners
    ledgers[0].reserve(new_owner, readers, deadline=1)
    new_claim = ledgers[0].acquire(new_owner, 0)
    new_handle = registries[0].register(new_claim)
    assert registries[0].complete(handles[0])
    assert not registries[0].complete(handles[0])
    assert ledgers[0].outstanding(new_owner) == readers
    assert ledgers[0].is_pinned(now=1000)
    assert ledgers[1].is_pinned(now=1000)
    managers[0].remove("same-request")
    managers[1].remove("same-request")
    assert registries[0].complete(new_handle)
    assert registries[1].complete(handles[1])
    assert [r.pending_count() for r in registries] == [0, 0]
    assert all(not l.is_pinned(now=1000) for l in ledgers)
    assert all(m.active_count() == 0 for m in managers)
