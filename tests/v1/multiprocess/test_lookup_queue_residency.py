# SPDX-License-Identifier: Apache-2.0
"""Queued leased lookups must preserve promised resident data, not just owners.

Uses real lookup/session/storage/L1/native locks on CPU. CUDA copies and the
eviction policy's selection algorithm are outside this test's scope.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc, MemoryLayoutDesc, ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig, L1ManagerConfig, L1MemoryManagerConfig, StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.v1.multiprocess.session as session_module


@pytest.fixture
def queued_lookup(monkeypatch: pytest.MonkeyPatch):
    """Create one stored CPU chunk and the actual leased-lookup call chain."""
    monkeypatch.setenv('LMCACHE_MP_OWNED_READ_LEASE','1')
    monkeypatch.setenv('LMCACHE_MP_SOURCE_MAPPING','{}')
    storage = StorageManager(StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=16 << 20,use_lazy=False,shm_name=''),
            read_ttl_seconds=1,
        ),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
        eviction_config=EvictionConfig(eviction_policy='LRU'),
    ))
    hasher = TokenHasher(4,'blake3')
    sessions = SessionManager(hasher,cleanup_interval=None)
    key = IPCCacheServerKey.from_token_ids(
        'queue-residency',1,None,[1,2,3,4],end=4,
        request_id='queued',cache_salt='isolated-residency',num_kv_readers=1,
    )
    hashes = hasher.compute_chunk_hashes(list(key.token_ids))
    keys = ipc_key_to_object_keys(key,hashes,[0])[0]
    layout = MemoryLayoutDesc(shapes=[torch.Size([2048])],dtypes=[torch.int32])
    registry = MagicMock()
    registry.find.return_value = layout
    registry.find_attn_desc.return_value = AttnWindowDesc([-1])
    registry.find_group_layout_descs.return_value = {0:layout}
    bus = MagicMock()
    bus.has_subscribers.return_value = False
    context = SimpleNamespace(storage_manager=storage,session_manager=sessions,
                              token_hasher=hasher,chunk_size=4,
                              layout_desc_registry=registry,event_bus=bus)
    module = LookupModule(context)
    try:
        objects = storage.reserve_write(keys,layout,mode='new')
        assert len(objects) == 1
        objects[keys[0]].tensor.fill_(37)
        storage.finish_write(keys)
        yield SimpleNamespace(module=module,storage=storage,sessions=sessions,
                              key=key,keys=keys)
    finally:
        module.end_session(key.request_id)
        sessions.close()
        storage.close()


def test_live_leased_lookup_retains_data_across_read_ttl(queued_lookup) -> None:
    """An authenticated live queue must not lose data already promised by lookup."""
    case = queued_lookup
    case.module.lookup_leased(case.key,1,'lookup-incarnation')
    assert case.module.query_prefetch_status('queued') == 1
    original = case.sessions.get('queued')
    owner = original.owned_read_state()[0]
    assert case.storage.delete_l1_keys(case.keys) == (0,1)
    time.sleep(1.1)
    # Initial publication must cover the heartbeat interval, too.
    assert case.storage.delete_l1_keys(case.keys) == (0,1)
    assert case.module.renew_sessions([('queued','lookup-incarnation')]) == ['queued']
    assert case.sessions.get('queued') is original
    assert original.owned_read_state()[0] == owner
    assert case.storage.delete_l1_keys(case.keys) == (0,1)
    result = case.storage.begin_owned_read(case.keys,owner,0)
    assert result is not None
    completion,objects = result
    assert torch.all(objects[0].tensor == 37).item()
    case.storage.finish_owned_read(completion)


def test_unleased_pending_lookup_keeps_existing_expiry(queued_lookup) -> None:
    """A nonleased lookup retains the existing bounded reservation policy."""
    case = queued_lookup
    case.module.lookup(case.key,1)
    assert case.module.query_prefetch_status('queued') == 1
    time.sleep(1.1)
    assert case.storage.delete_l1_keys(case.keys) == (1,0)


def test_session_deadline_is_bounded_and_stale_session_cannot_renew(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(session_module, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], time=time.time))
    manager = SessionManager(TokenHasher(4,'blake3'), ttl=10,
                             max_lease_lifetime=25, cleanup_interval=None)
    try:
        original = manager.get_or_create('bounded')
        assert manager.read_lease_deadline(original) is None
        assert manager.bind_liveness('bounded','generation-1')
        assert manager.read_lease_deadline(original) == 110
        for now, expected in [(109,119),(118,125),(124,125)]:
            clock[0] = now
            assert manager.renew_sessions([('bounded','generation-1')]) == ['bounded']
            assert manager.read_lease_deadline(original) == expected
        clock[0] = 126
        assert manager.renew_sessions([('bounded','generation-1')]) == []
        with pytest.raises(RuntimeError):
            manager.read_lease_deadline(original)
        assert manager.cleanup_expired() == 1
        replacement = manager.get_or_create('bounded')
        assert manager.bind_liveness('bounded','generation-2')
        assert manager.renew_sessions([('bounded','generation-1')]) == []
        with pytest.raises(RuntimeError):
            manager.read_lease_deadline(original)
        assert manager.read_lease_deadline(replacement) == 136
        with pytest.raises(RuntimeError):
            manager.read_lease_deadline(replacement, credential='generation-1')
    finally:
        manager.close()


def test_wrong_credential_cannot_extend_idle_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(session_module, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], time=time.time))
    manager = SessionManager(TokenHasher(4,'blake3'), ttl=10, cleanup_interval=None)
    try:
        original = manager.get_or_create('idle')
        assert manager.bind_liveness('idle','right')
        clock[0] = 109
        assert manager.renew_sessions([('idle','wrong')]) == []
        assert manager.read_lease_deadline(original) == 110
        clock[0] = 111
        assert manager.renew_sessions([('idle','right')]) == []
        assert manager.cleanup_expired() == 1
    finally:
        manager.close()


def test_replacement_between_authentication_and_storage_renewal_is_rejected(
    queued_lookup, monkeypatch
) -> None:
    case = queued_lookup
    case.module.lookup_leased(case.key,1,'old-generation')
    assert case.module.query_prefetch_status('queued') == 1
    original = case.sessions.get('queued')
    renew = case.sessions.renew_sessions

    def replace_after_authentication(credentials):
        accepted = renew(credentials)
        case.module.end_session('queued')
        case.module.lookup_leased(case.key,1,'new-generation')
        assert case.module.query_prefetch_status('queued') == 1
        return accepted

    monkeypatch.setattr(case.sessions, 'renew_sessions', replace_after_authentication)
    assert case.module.renew_sessions([('queued','old-generation')]) == []
    replacement = case.sessions.get('queued')
    assert replacement is not original
    assert case.storage.delete_l1_keys(case.keys) == (0,1)
    result = case.storage.begin_owned_read(
        case.keys,replacement.owned_read_state()[0],0)
    assert result is not None
    completion, objects = result
    assert torch.all(objects[0].tensor == 37).item()
    case.storage.finish_owned_read(completion)


def test_cancelled_leased_lookup_releases_pending_data(queued_lookup) -> None:
    """Ending a live leased lookup releases unused data without waiting for TTL."""
    case = queued_lookup
    case.module.lookup_leased(case.key,1,'lookup-incarnation')
    assert case.module.query_prefetch_status('queued') == 1
    assert case.module.renew_sessions([('queued','lookup-incarnation')]) == ['queued']
    case.module.end_session('queued')
    assert case.sessions.get('queued') is None
    assert case.storage.delete_l1_keys(case.keys) == (1,0)
