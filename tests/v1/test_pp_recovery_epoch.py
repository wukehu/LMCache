# SPDX-License-Identifier: Apache-2.0
"""A successful ping must not hide a restarted MP server's lost registrations."""

# Standard
from unittest.mock import MagicMock, PropertyMock, patch
import threading
import unittest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter
from lmcache.v1.multiprocess.modules.management import ManagementModule


class RecoveryEpochTest(unittest.TestCase):
    def test_server_epoch_is_stable_until_server_recreation(self) -> None:
        first = ManagementModule(MagicMock())
        second = ManagementModule(MagicMock())
        before = adapter.get_server_epoch(set(first.get_experimental()))
        self.assertIsNotNone(before)
        self.assertEqual(before, adapter.get_server_epoch(set(first.get_experimental())))
        self.assertNotEqual(before, adapter.get_server_epoch(set(second.get_experimental())))

    def make_heartbeat(self, epoch: str | None = "server_epoch_v1:before"):
        event = threading.Event()
        event.set()
        heartbeat = adapter.HeartbeatThread(
            MagicMock(), event, instance_id=17, server_epoch=epoch
        )
        return heartbeat, event

    def test_restart_without_failed_ping_recovers_once(self) -> None:
        heartbeat, event = self.make_heartbeat()

        def recover() -> bool:
            self.assertFalse(event.is_set())
            return True

        callback = MagicMock(side_effect=recover)
        heartbeat.register_recover_callback(callback)
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental", return_value={"server_epoch_v1:after"}
        ):
            heartbeat._execute()
            heartbeat._execute()
        callback.assert_called_once()
        self.assertTrue(event.is_set())

    def test_failed_recovery_retries_without_publishing_health(self) -> None:
        heartbeat, event = self.make_heartbeat()
        callback = MagicMock(side_effect=[False, True])
        heartbeat.register_recover_callback(callback)
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental", return_value={"server_epoch_v1:after"}
        ):
            heartbeat._execute()
            self.assertFalse(event.is_set())
            heartbeat._execute()
        self.assertEqual(callback.call_count, 2)
        self.assertTrue(event.is_set())

    def test_epoch_query_failure_does_not_publish_health(self) -> None:
        heartbeat, event = self.make_heartbeat()
        callback = MagicMock(return_value=True)
        heartbeat.register_recover_callback(callback)
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental", side_effect=TimeoutError
        ):
            heartbeat._execute()
        callback.assert_not_called()
        self.assertFalse(event.is_set())

    def test_legacy_server_keeps_ping_only_behavior(self) -> None:
        heartbeat, event = self.make_heartbeat(None)
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental"
        ) as query:
            heartbeat._execute()
        query.assert_not_called()
        self.assertTrue(event.is_set())

    def test_missing_epoch_after_negotiation_is_unhealthy(self) -> None:
        heartbeat, event = self.make_heartbeat()
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental", return_value=set()
        ):
            heartbeat._execute()
        self.assertFalse(event.is_set())

    def test_stop_during_recovery_does_not_restore_health(self) -> None:
        heartbeat, event = self.make_heartbeat()
        callback = MagicMock(return_value=True)
        heartbeat.register_recover_callback(callback)
        with patch.object(adapter, "send_ping", return_value=True), patch.object(
            adapter, "get_experimental", return_value={"server_epoch_v1:after"}
        ), patch.object(
            type(heartbeat), "stop_requested", new_callable=PropertyMock,
            side_effect=[False, True],
        ):
            heartbeat._execute()
        callback.assert_called_once()
        self.assertFalse(event.is_set())


if __name__ == "__main__":
    unittest.main()
