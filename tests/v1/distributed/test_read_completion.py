# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.read_completion import ReadCompletionRegistry


def test_serialized_duplicate_completion_consumes_once():
    completed = []
    registry = ReadCompletionRegistry(completed.append)
    receipt = object()
    handle = json.loads(json.dumps(registry.register(receipt)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(registry.complete, [handle] * 8))
    assert sum(results) == 1
    assert completed == [receipt]
    assert registry.pending_count() == 0


def test_late_handle_never_aliases_new_receipt():
    completed = []
    registry = ReadCompletionRegistry(completed.append)
    old = registry.register("old")
    assert registry.complete(old)
    new = registry.register("new")
    assert old != new
    assert not registry.complete(old)
    assert registry.pending_count() == 1
    assert registry.complete(new)
    assert completed == ["old", "new"]


def test_failed_completion_retains_receipt_for_safe_retry():
    calls = []
    def finish(receipt):
        calls.append(receipt)
        if len(calls) == 1:
            raise RuntimeError("completion interrupted")
    registry = ReadCompletionRegistry(finish)
    handle = registry.register("allocation")
    with pytest.raises(RuntimeError):
        registry.complete(handle)
    assert registry.pending_count() == 1
    assert registry.complete(handle)
    assert registry.pending_count() == 0


def test_opt_in_rejects_unported_l2_before_allocating(monkeypatch):
    from lmcache.v1.distributed.storage_manager import StorageManager
    monkeypatch.setenv("LMCACHE_MP_OWNED_READ_LEASE", "1")
    config = SimpleNamespace(l2_adapter_config=SimpleNamespace(adapters=[object()]))
    with pytest.raises(ValueError, match="L1-only"):
        StorageManager(config)
