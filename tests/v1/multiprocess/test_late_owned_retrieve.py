# SPDX-License-Identifier: Apache-2.0
"""Late owned Retrieve cannot resurrect a cancelled Session or touch a replacement."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import LMCacheDrivenTransferModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def setup():
    manager = SessionManager(TokenHasher(4, "blake3"), cleanup_interval=None)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, [1, 2, 3, 4], end=4,
                                         request_id="request", cache_salt="late", num_kv_readers=1)
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = SimpleNamespace(session_manager=manager,
        storage_manager=SimpleNamespace(supports_owned_reads=lambda: True),
        resolve_obj_keys=MagicMock(), event_bus=MagicMock())
    entry = SimpleNamespace(cache_context=SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(num_object_groups=1)),
        model_name="model", event_backend=object())
    module.get_and_touch_context_entry = MagicMock(return_value=entry)
    return manager, key, module


def test_cancelled_retrieve_does_not_recreate_session_or_submit_gpu_work():
    manager, key, module = setup()
    manager.get_or_create(key.request_id)
    manager.remove(key.request_id)
    assert module.retrieve(key, 1, [[1]], b"unused") == (b"", False)
    assert manager.get(key.request_id) is None
    assert manager.active_count() == 0
    module.get_and_touch_context_entry.assert_not_called()
    module._ctx.resolve_obj_keys.assert_not_called()
    module._ctx.event_bus.publish.assert_not_called()


@pytest.mark.parametrize("replace", [False, True])
def test_cancel_during_key_resolution_uses_captured_session_and_rejects_late_work(replace):
    manager, key, module = setup()
    original = manager.get_or_create(key.request_id)
    replacement = None

    def resolve(given_key, groups, *, session):
        nonlocal replacement
        assert session is original
        manager.remove(key.request_id)
        if replace:
            replacement = manager.get_or_create(key.request_id)
        # Execute the real resolver after END. It must not consult get-or-create.
        return MPCacheServerContext.resolve_obj_keys(module._ctx, given_key, groups, session=session)

    module._ctx.resolve_obj_keys.side_effect = resolve
    assert module.retrieve(key, 1, [[1]], b"unused") == (b"", False)
    assert manager.get(key.request_id) is replacement
    assert manager.active_count() == int(replace)
    module._ctx.event_bus.publish.assert_not_called()


def test_default_key_resolution_preserves_legacy_create_behavior():
    manager, key, module = setup()
    keys = MPCacheServerContext.resolve_obj_keys(module._ctx, key, [0])
    assert len(keys) == 1 and len(keys[0]) == 1
    assert manager.get(key.request_id) is not None
