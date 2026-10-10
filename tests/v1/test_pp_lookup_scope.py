# SPDX-License-Identifier: Apache-2.0
"""Scoped lookup must address exactly the same objects as global workers."""

# Standard
from dataclasses import replace
import unittest

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


class PPLookupScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.key = IPCCacheServerKey.from_token_ids(
            "pp-test", 4, None, list(range(32)), end=32, num_kv_readers=4
        )

    def test_selected_objects_equal_worker_store_objects(self) -> None:
        hashes = [b"first", b"second"]
        scoped = replace(self.key, lookup_worker_ids=(2, 3))
        actual = ipc_key_to_object_keys(scoped, hashes, [0, 1])
        for group in (0, 1):
            expected = []
            for chunk in hashes:
                for worker in (2, 3):
                    key = replace(self.key, worker_id=worker)
                    expected.extend(ipc_key_to_object_keys(key, [chunk], [group])[0])
            self.assertEqual(actual[group], expected)
        self.assertEqual(scoped.world_size, 4)
        self.assertEqual(scoped.lookup_world_size, 2)

    def test_wire_roundtrip_and_cleanup_preserve_scope(self) -> None:
        key = replace(self.key, lookup_worker_ids=(1, 3))
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(key), type=IPCCacheServerKey
        )
        self.assertEqual(decoded.lookup_worker_ids, (1, 3))
        self.assertEqual(decoded.no_worker_id_version().lookup_worker_ids, (1, 3))
        self.assertEqual(decoded.require_num_kv_readers(), 4)

    def test_old_wire_payload_remains_unscoped(self) -> None:
        payload = msgspec.to_builtins(self.key)
        del payload["lookup_worker_ids"]
        key = msgspec.msgpack.decode(
            msgspec.msgpack.encode(payload), type=IPCCacheServerKey
        )
        self.assertIsNone(key.lookup_worker_ids)
        self.assertEqual(key.lookup_world_size, 4)
        self.assertEqual(len(ipc_key_to_object_keys(key, [b"c"], [0])[0]), 4)

    def test_invalid_scope_is_rejected(self) -> None:
        for scope in ((), (1, 1), (3, 1), (-1,), (4,)):
            with self.assertRaises(ValueError):
                replace(self.key, lookup_worker_ids=scope)
        with self.assertRaises(ValueError):
            replace(self.key, worker_id=0, lookup_worker_ids=(0,))


if __name__ == "__main__":
    unittest.main()
