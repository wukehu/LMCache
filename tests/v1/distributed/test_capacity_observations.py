# SPDX-License-Identifier: Apache-2.0
"""Capacity telemetry separates adapter payload/quota from backing filesystem."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import is_fs_object_filename
from lmcache.v1.mp_observability.fs_inventory import InventorySnapshot


def test_inventory_targets_use_effective_adapter_path(monkeypatch):
    sm = manager(monkeypatch, [snapshot(7, "fs", adapter(path="/effective")),
                               snapshot(9, "mooncake", adapter(path="/ignored"))])
    assert sm.get_fs_inventory_targets() == {"7": "/effective"}


def test_inventory_scrape_reads_cache_only(monkeypatch):
    sm = StorageManager.__new__(StorageManager)
    cached = InventorySnapshot(5, 140, None, 1000.0, 0.2)
    sm._fs_inventory = SimpleNamespace(read=lambda: {"7": (cached, False),
                                                    "8": (None, False)})
    monkeypatch.setattr("os.scandir", lambda path: pytest.fail("scrape scanned FS"))
    assert sm.get_fs_inventory_observations("logical_bytes") == [
        (140, {"adapter_id": "7", "l2_name": "fs"})]
    assert sm.get_fs_inventory_observations("allocated_bytes") == []
    assert [value for value, _ in sm.get_fs_inventory_observations("success")] == [0, 0]
    assert sm.get_fs_inventory_observations("completed_at")[0][0] == 1000
    with pytest.raises(ValueError):
        sm.get_fs_inventory_observations("quota")


def test_inventory_uses_real_fs_filename_parser():
    chunk = "ab" * 32
    assert is_fs_object_filename(f"model@0x0@0@{chunk}.data")
    assert is_fs_object_filename(f"model@0x0@0@{chunk}@salt.data")
    for name in ("stray.data", f"model@nothex@0@{chunk}.data", "a.tmp"):
        assert not is_fs_object_filename(name)


def manager(monkeypatch, adapters):
    """Supply public adapter snapshots without starting GPU controllers."""
    result = StorageManager.__new__(StorageManager)
    monkeypatch.setattr(result, "_snapshot_adapters", lambda: adapters)
    return result


def adapter(used=17, capacity=0, path="/cache", fail=False):
    """Provide the adapter's public status and usage interface."""
    def usage():
        if fail:
            raise OSError("unavailable")
        return SimpleNamespace(total_bytes_used=used, total_capacity_bytes=capacity)

    return SimpleNamespace(
        get_usage=usage,
        report_status=lambda: {"base_path": path, "serde_wrapped": True},
    )


def snapshot(number, kind, instance):
    return number, SimpleNamespace(type_name=kind), instance


def test_unknown_quota_is_distinct_from_empty_usage(monkeypatch):
    sm = manager(monkeypatch, [snapshot(7, "fs", adapter())])
    labels = {"adapter_id": "7", "l2_name": "fs", "quota_limited": "false"}
    assert sm.get_l2_capacity_observations("capacity") == [(0, labels)]
    assert sm.get_l2_capacity_observations("usage") == [(17, labels)]


def test_identity_and_effective_quota(monkeypatch):
    sm = manager(monkeypatch, [
        snapshot(1, "fs", adapter(0, 128)),
        snapshot(9, "fs", adapter(32, 256)),
    ])
    assert [value for value, _ in sm.get_l2_capacity_observations("capacity")] == [128, 256]
    assert [labels["adapter_id"] for _, labels in sm.get_l2_capacity_observations("usage")] == ["1", "9"]
    assert all(labels["quota_limited"] == "true" for _, labels in sm.get_l2_capacity_observations("usage"))


@pytest.mark.parametrize("bad", [adapter(fail=True), adapter(-1), adapter(capacity=-1)])
def test_unavailable_or_invalid_usage_is_absent(monkeypatch, bad):
    sm = manager(monkeypatch, [snapshot(0, "fs", bad)])
    assert sm.get_l2_capacity_observations("usage") == []


def test_filesystem_uses_effective_public_path_and_reserved_space(monkeypatch):
    calls = []

    def statvfs(path):
        calls.append(path)
        return SimpleNamespace(f_blocks=100, f_bfree=40, f_bavail=30, f_frsize=4096)

    monkeypatch.setattr("os.statvfs", statvfs, raising=False)
    sm = manager(monkeypatch, [
        snapshot(2, "fs", adapter(path="/effective/path")),
        snapshot(3, "mooncake", adapter()),
    ])
    assert sm.get_fs_capacity_observations("capacity")[0][0] == 409600
    assert sm.get_fs_capacity_observations("used")[0][0] == 245760
    assert sm.get_fs_capacity_observations("available")[0][0] == 122880
    assert calls == ["/effective/path"] * 3
    assert sm.get_l2_capacity_observations("usage")[0][0] == 17


@pytest.mark.parametrize("path", [None, ""])
def test_missing_path_is_absent(monkeypatch, path):
    sm = manager(monkeypatch, [snapshot(0, "fs", adapter(path=path))])
    assert sm.get_fs_capacity_observations("used") == []


def test_missing_mount_is_absent(monkeypatch):
    def fail(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr("os.statvfs", fail, raising=False)
    sm = manager(monkeypatch, [snapshot(0, "fs", adapter())])
    assert sm.get_fs_capacity_observations("used") == []


@pytest.mark.parametrize("free,available", [(101, 40), (40, 41), (-1, 0)])
def test_invalid_filesystem_snapshot_is_absent(monkeypatch, free, available):
    monkeypatch.setattr("os.statvfs", lambda path: SimpleNamespace(
        f_blocks=100, f_bfree=free, f_bavail=available, f_frsize=4096,
    ), raising=False)
    sm = manager(monkeypatch, [snapshot(0, "fs", adapter())])
    assert sm.get_fs_capacity_observations("used") == []


def test_real_filesystem_observation(monkeypatch, tmp_path):
    sm = manager(monkeypatch, [snapshot(0, "fs", adapter(path=str(tmp_path)))])
    capacity = sm.get_fs_capacity_observations("capacity")
    assert len(capacity) == 1 and capacity[0][0] > 0
    assert 0 <= sm.get_fs_capacity_observations("used")[0][0] <= capacity[0][0]


def test_unknown_field_rejected(monkeypatch):
    sm = manager(monkeypatch, [])
    with pytest.raises(ValueError):
        sm.get_fs_capacity_observations("quota")
    with pytest.raises(ValueError):
        sm.get_l2_capacity_observations("free")
