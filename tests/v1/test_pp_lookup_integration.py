# SPDX-License-Identifier: Apache-2.0
"""PP lookup uses real registry/hash/session logic with a storage boundary spy."""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import time
import unittest

# Third Party
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


class PPLookupIntegrationTest(unittest.TestCase):
    def tearDown(self) -> None:
        self.module.close()
        self.ctx.session_manager.close()

    def setUp(self) -> None:
        self.ctx = MagicMock()
        self.ctx.chunk_size = 16
        self.ctx.token_hasher = TokenHasher(chunk_size=16)
        self.ctx.session_manager = SessionManager(self.ctx.token_hasher)
        self.ctx.layout_desc_registry = LayoutDescRegistry()
        self.small = MemoryLayoutDesc([torch.Size([30, 16, 8])], [torch.bfloat16])
        self.large = MemoryLayoutDesc([torch.Size([31, 16, 8])], [torch.bfloat16])
        self.ctx.layout_desc_registry.register("pp", 4, self.small, worker_id=2)
        self.module = LookupModule(self.ctx)
        self.key = replace(
            IPCCacheServerKey.from_token_ids(
                "pp", 4, None, list(range(32)), end=32, request_id="lookup", num_kv_readers=4
            ),
            lookup_worker_ids=(2, 3),
        )

    def test_missing_stage_returns_zero_without_storage_lookup(self) -> None:
        self.module.lookup(self.key, 4)
        self.ctx.storage_manager.submit_prefetch_task.assert_not_called()
        self.ctx.storage_manager.query_prefetch_status.return_value = Bitmap(0)
        self.assertEqual(self.module.query_prefetch_status("lookup"), 0)

    def test_scoped_lookup_delivers_distinct_layouts_and_local_fanout(self) -> None:
        self.ctx.layout_desc_registry.register("pp", 4, self.large, worker_id=3)
        self.ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
            prefetch_request_id=1,
            external_request_id="lookup",
            l1_found_indices=(),
            l1_hit_chunks=0,
            total_requested_keys=4,
            submit_time=time.monotonic(),
        )
        self.module.lookup(self.key, 4)
        spec = self.ctx.storage_manager.submit_prefetch_task.call_args.args[0]
        self.assertEqual(spec.attn_desc.world_size, 2)
        self.assertEqual(spec.num_kv_readers, 4)
        self.assertEqual(len(spec.keys), 4)
        self.assertEqual(
            [spec.layout_for_key(key) for key in spec.keys],
            [self.small, self.large, self.small, self.large],
        )


if __name__ == "__main__":
    unittest.main()
