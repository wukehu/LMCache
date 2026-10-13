# SPDX-License-Identifier: Apache-2.0
"""L1 lifecycle tests with real native TTL locks and a CPU-only allocator double.

These test ownership, eviction and completion ordering, not CUDA transfer.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager


@pytest.fixture
def manager():
    allocator = MagicMock()
    allocator.allocate.side_effect = lambda layout, count: (
        L1Error.SUCCESS, [MagicMock() for _ in range(count)]
    )
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=1048576, use_lazy=False),
        read_ttl_seconds=1,
        write_ttl_seconds=1,
    )
    with patch("lmcache.v1.distributed.l1_manager.L1MemoryManager", return_value=allocator):
        with patch("lmcache.v1.distributed.l1_manager.register_gauge"):
            mgr = L1Manager(config)
    yield mgr
    mgr.close()


def stored(manager, index=1):
    key = ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(index), model_name="lease", kv_rank=0)
    assert manager.reserve_write([key], [False], MagicMock())[key][0] == L1Error.SUCCESS
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    return key


def test_expired_resident_read_pins_until_exact_completion(manager):
    key = stored(manager)
    assert manager.reserve_owned_read([key], "A", 1)[key][0] == L1Error.SUCCESS
    time.sleep(1.05)
    receipts = manager.acquire_owned_read([key], "A", 0)
    assert receipts is not None
    time.sleep(1.05)
    assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
    assert manager.reserve_write([key], [False], MagicMock())[key][0] == L1Error.KEY_NOT_WRITABLE
    manager.clear()
    assert manager.get_object_state(key) is not None
    manager.finish_owned_read(list(receipts.values()))
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_late_and_duplicate_callbacks_preserve_other_owner(manager):
    key = stored(manager)
    manager.reserve_owned_read([key], "A", 1)
    old = manager.acquire_owned_read([key], "A", 0)
    manager.reserve_owned_read([key], "B", 1)
    new = manager.acquire_owned_read([key], "B", 0)
    manager.cancel_owned_read([key], "A")
    manager.cancel_owned_read([key], "B")
    manager.finish_owned_read(list(old.values()))
    manager.finish_owned_read(list(old.values()))
    assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
    manager.finish_owned_read(list(new.values()))
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_expired_write_is_not_committed_data(manager):
    key = ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(1), model_name="lease", kv_rank=0)
    manager.reserve_write([key], [False], MagicMock())
    time.sleep(1.05)
    assert manager.reserve_owned_read([key], "A", 1)[key][0] == L1Error.KEY_NOT_READABLE
    assert manager.acquire_owned_read([key], "A", 0) is None


def test_partial_failure_releases_only_new_pins(manager):
    one, two = stored(manager, 1), stored(manager, 2)
    manager.reserve_owned_read([one, two], "A", 1)
    time.sleep(1.05)
    assert manager.delete([two])[two] == L1Error.SUCCESS
    assert manager.acquire_owned_read([one, two], "A", 0) is None
    assert manager.delete([one])[one] == L1Error.SUCCESS


def test_replaced_allocation_and_unknown_owner_fail_closed(manager):
    key = stored(manager)
    manager.reserve_owned_read([key], "A", 1)
    time.sleep(1.05)
    manager.delete([key])
    stored(manager)
    manager.reserve_owned_read([key], "B", 1)
    assert manager.acquire_owned_read([key], "A", 0) is None
    assert manager.acquire_owned_read([key], "unknown", 0) is None
    assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED


def test_duplicate_concurrent_reader_gets_exactly_one_claim(manager):
    key = stored(manager)
    manager.reserve_owned_read([key], "A", 8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: manager.acquire_owned_read([key], "A", 0), range(8)))
    acquired = [result for result in results if result is not None]
    assert len(acquired) == 1
    manager.cancel_owned_read([key], "A")
    assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
    manager.finish_owned_read(list(acquired[0].values()))
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_legacy_completion_cannot_release_owned_pin(manager):
    key = stored(manager)
    manager.reserve_read([key], read_locks=2)
    manager.reserve_owned_read([key], "A", 1)
    receipts = manager.acquire_owned_read([key], "A", 0)
    manager.cancel_owned_read([key], "A")
    assert manager.finish_read([key], read_locks=2)[key] == L1Error.SUCCESS
    assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
    manager.finish_owned_read(list(receipts.values()))
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_duplicate_batch_rejected_without_partial_reservation(manager):
    key = stored(manager)
    with pytest.raises(ValueError):
        manager.reserve_owned_read([key, key], "A", 1)
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_partial_tp_completion_does_not_reappear_after_expiry(manager):
    key = stored(manager)
    manager.reserve_owned_read([key], ("request", 1), 4)
    first = manager.acquire_owned_read([key], ("request", 1), 0)
    manager.finish_owned_read(list(first.values()))
    time.sleep(1.05)
    assert manager.acquire_owned_read([key], ("request", 1), 0) is None
    for rank in (1, 2, 3):
        receipts = manager.acquire_owned_read([key], ("request", 1), rank)
        assert receipts is not None
        manager.finish_owned_read(list(receipts.values()))
    assert manager.delete([key])[key] == L1Error.SUCCESS


@pytest.mark.parametrize("readers", [2, 8])
def test_owned_partial_prefix_tail_preserves_common_and_other_owner(manager, readers):
    common = [stored(manager, 10), stored(manager, 11)]
    tail = [stored(manager, 12), stored(manager, 13)]
    owner, other = ("partial", 1), ("other", 1)
    manager.reserve_owned_read(common + tail, owner, readers)
    manager.reserve_owned_read(tail, other, 1)

    # The adapter's over-hit interval is cancelled twice to also check that
    # duplicate control messages cannot consume another owner's reservation.
    manager.cancel_owned_read(tail, owner)
    manager.cancel_owned_read(tail, owner)
    for key in common:
        assert manager.get_object_state(key).read_leases.outstanding(owner) == readers
    for key in tail:
        leases = manager.get_object_state(key).read_leases
        assert leases.outstanding(owner) == 0
        assert leases.outstanding(other) == 1
        assert manager.acquire_owned_read([key], owner, 0) is None
        assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED

    receipts = manager.acquire_owned_read(tail, other, 0)
    assert receipts is not None
    manager.finish_owned_read(list(receipts.values()))
    assert all(value == L1Error.SUCCESS for value in manager.delete(tail).values())

    for rank in range(readers):
        receipts = manager.acquire_owned_read(common, owner, rank)
        assert receipts is not None
        manager.finish_owned_read(list(receipts.values()))
        manager.finish_owned_read(list(receipts.values()))
        expected = readers - rank - 1
        for key in common:
            assert manager.get_object_state(key).read_leases.outstanding(owner) == expected
        if expected:
            assert all(value == L1Error.KEY_IS_LOCKED for value in manager.delete(common).values())
    assert all(value == L1Error.SUCCESS for value in manager.delete(common).values())
