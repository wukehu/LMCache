# SPDX-License-Identifier: Apache-2.0
"""Persistent inventory is distinct from process counters and failed scans."""

# Standard
from pathlib import Path
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

# First Party
from lmcache.v1.mp_observability.fs_inventory import (
    InventoryIncomplete,
    InventoryObservation,
    InventorySampler,
    scan_inventory,
)


class InventoryTests(unittest.TestCase):
    """Exercise actual files and incomplete observations without GPU imports."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.accepts = lambda name: name.endswith(".data")

    def test_existing_files_counted_without_store_events(self) -> None:
        (self.path / "a.data").write_bytes(b"123")
        (self.path / "b.data").write_bytes(b"45678")
        (self.path / "a.tmp").write_bytes(b"excluded")
        (self.path / "nested").mkdir()
        (self.path / "nested" / "c.data").write_bytes(b"excluded")
        got = scan_inventory(str(self.path), self.accepts)
        self.assertEqual((got.objects, got.logical_bytes), (2, 8))
        self.assertGreater(got.completed_at, 0)

    def test_empty_success_is_different_from_no_snapshot(self) -> None:
        observation = InventoryObservation()
        self.assertEqual(observation.read(), (None, False))
        self.assertTrue(observation.sample(str(self.path), self.accepts))
        got, success = observation.read()
        self.assertTrue(success)
        self.assertEqual((got.objects, got.logical_bytes), (0, 0))

    def test_failed_scan_keeps_last_complete_value(self) -> None:
        (self.path / "a.data").write_bytes(b"123")
        observation = InventoryObservation()
        self.assertTrue(observation.sample(str(self.path), self.accepts))
        previous = observation.read()[0]
        self.assertFalse(observation.sample(str(self.path / "missing"), self.accepts))
        self.assertEqual(observation.read(), (previous, False))

    def test_entry_budget_counts_rejected_names(self) -> None:
        for number in range(3):
            (self.path / f"{number}.tmp").touch()
        with self.assertRaises(InventoryIncomplete):
            scan_inventory(str(self.path), self.accepts, max_entries=2)

    def test_cancellation_prevents_empty_success(self) -> None:
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InventoryIncomplete):
            scan_inventory(str(self.path), self.accepts, cancelled=stop)

    def test_permission_failure_does_not_become_zero(self) -> None:
        with patch("os.scandir", side_effect=PermissionError):
            with self.assertRaises(InventoryIncomplete):
                scan_inventory(str(self.path), self.accepts)

    def test_symlink_not_followed(self) -> None:
        target = self.path / "target.tmp"
        target.write_bytes(b"excluded")
        try:
            (self.path / "link.data").symlink_to(target)
        except OSError:
            self.skipTest("host cannot create symlinks")
        got = scan_inventory(str(self.path), self.accepts)
        self.assertEqual((got.objects, got.logical_bytes), (0, 0))

    def test_concurrent_disappearance_invalidates_snapshot(self) -> None:
        file = self.path / "a.data"
        file.touch()

        def remove_before_stat(name: str) -> bool:
            file.unlink()
            return True

        with self.assertRaises(InventoryIncomplete):
            scan_inventory(str(self.path), remove_before_stat)

    def test_time_budget(self) -> None:
        with patch("time.monotonic", side_effect=[0, 0, 3]):
            with self.assertRaises(InventoryIncomplete):
                scan_inventory(str(self.path), self.accepts, max_seconds=2)

    def test_logical_and_allocated_bytes_are_separate(self) -> None:
        file = self.path / "a.data"
        file.write_bytes(b"123")
        got = scan_inventory(str(self.path), self.accepts)
        self.assertEqual(got.logical_bytes, 3)
        info = os.stat(file)
        expected = info.st_blocks * 512 if hasattr(info, "st_blocks") else None
        self.assertEqual(got.allocated_bytes, expected)

    def test_invalid_limits(self) -> None:
        for limits in ({"max_entries": 0}, {"max_seconds": 0},
                       {"max_seconds": float("nan")},
                       {"max_seconds": float("inf")}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                scan_inventory(str(self.path), self.accepts, **limits)

    def wait_for(self, condition) -> None:
        """Bound background tests without assuming thread scheduling order."""
        deadline = time.monotonic() + 2
        while not condition():
            if time.monotonic() > deadline:
                self.fail("sampler did not publish expected observation")
            time.sleep(0.005)

    def test_background_sampling_read_is_io_free_and_close_joins(self) -> None:
        (self.path / "a.data").write_bytes(b"123")
        sampler = InventorySampler(lambda: {"7": str(self.path)}, self.accepts,
                                   interval=0.02)
        sampler.start()
        self.addCleanup(sampler.close)
        self.wait_for(lambda: sampler.read().get("7", (None, False))[1])
        with patch("os.scandir", side_effect=AssertionError("scrape did IO")):
            snapshot, success = sampler.read()["7"]
            self.assertEqual(snapshot.logical_bytes, 3)
            self.assertTrue(success)
        self.assertTrue(sampler.close())
        self.assertTrue(sampler.close())

    def test_background_failure_preserves_snapshot_and_recovers(self) -> None:
        file = self.path / "a.data"
        file.write_bytes(b"123")
        sampler = InventorySampler(lambda: {"7": str(self.path)}, self.accepts,
                                   interval=0.01)
        sampler.start()
        self.addCleanup(sampler.close)
        self.wait_for(lambda: sampler.read().get("7", (None, False))[1])
        previous = sampler.read()["7"][0]
        with patch("os.scandir", side_effect=PermissionError):
            self.wait_for(lambda: not sampler.read()["7"][1])
            self.assertEqual(sampler.read()["7"][0].logical_bytes,
                             previous.logical_bytes)
        self.wait_for(lambda: sampler.read()["7"][1])

    def test_changed_path_discards_old_snapshot_and_removal_clears_target(self) -> None:
        targets = {"7": str(self.path)}
        sampler = InventorySampler(lambda: dict(targets), self.accepts, interval=0.01)
        sampler.start()
        self.addCleanup(sampler.close)
        self.wait_for(lambda: sampler.read().get("7", (None, False))[1])
        targets["7"] = str(self.path / "missing")
        self.wait_for(lambda: sampler.read()["7"] == (None, False))
        targets.clear()
        self.wait_for(lambda: not sampler.read())

    def test_target_refresh_failure_marks_snapshot_unsuccessful(self) -> None:
        fail = threading.Event()

        def targets() -> dict[str, str]:
            if fail.is_set():
                raise OSError("adapter unavailable")
            return {"7": str(self.path)}

        sampler = InventorySampler(targets, self.accepts, interval=0.01)
        sampler.start()
        self.addCleanup(sampler.close)
        self.wait_for(lambda: sampler.read().get("7", (None, False))[1])
        fail.set()
        self.wait_for(lambda: not sampler.read()["7"][1])
        self.assertIsNotNone(sampler.read()["7"][0])


if __name__ == "__main__":
    unittest.main()
