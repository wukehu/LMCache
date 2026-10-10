# SPDX-License-Identifier: Apache-2.0
"""Different PP stages retain distinct layouts and registration lifetimes."""

# Standard
import unittest

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, PrefetchRequestSpec
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry


class PPLayoutRegistryTest(unittest.TestCase):
    def test_prefetch_layouts_resolve_each_stage_and_reject_missing_layout(self) -> None:
        small = MemoryLayoutDesc([torch.Size([30, 256, 576])], [torch.bfloat16])
        large = MemoryLayoutDesc([torch.Size([31, 256, 576])], [torch.bfloat16])
        keys = [
            ObjectKey(b"chunk", "r1", ObjectKey.ComputeKVRank(2, rank, 2, rank), 0)
            for rank in (0, 1)
        ]
        spec = PrefetchRequestSpec(
            keys, {0: small}, object_layout_descs={keys[0]: small, keys[1]: large}
        )
        self.assertEqual(spec.layout_for_key(keys[0]), small)
        self.assertEqual(spec.layout_for_key(keys[1]), large)
        with self.assertRaises(ValueError):
            PrefetchRequestSpec(keys, {0: small}, object_layout_descs={keys[0]: small})

    def test_unequal_stages_and_shared_replica_lifetimes(self) -> None:
        registry = LayoutDescRegistry()
        small = MemoryLayoutDesc([torch.Size([30, 256, 576])], [torch.bfloat16])
        large = MemoryLayoutDesc([torch.Size([31, 256, 576])], [torch.bfloat16])
        registry.register("r1", 2, small, worker_id=0)
        registry.register("r1", 2, large, worker_id=1)
        registry.register("r1", 2, small, worker_id=0)
        self.assertEqual(
            registry.find_worker_group_layout_descs("r1", 2),
            {0: {0: small}, 1: {0: large}},
        )
        registry.unregister("r1", 2, worker_id=0)
        self.assertIn(0, registry.find_worker_group_layout_descs("r1", 2))
        registry.unregister("r1", 2, worker_id=0)
        self.assertEqual(registry.find_worker_group_layout_descs("r1", 2), {1: {0: large}})
        registry.unregister("r1", 2, worker_id=1)
        self.assertEqual(registry.find_worker_group_layout_descs("r1", 2), {})

    def test_conflicting_registration_does_not_change_existing_layout(self) -> None:
        registry = LayoutDescRegistry()
        small = MemoryLayoutDesc([torch.Size([30, 256, 576])], [torch.bfloat16])
        large = MemoryLayoutDesc([torch.Size([31, 256, 576])], [torch.bfloat16])
        registry.register("r1", 2, small, worker_id=0)
        with self.assertRaises(ValueError):
            registry.register("r1", 2, large, worker_id=0)
        self.assertEqual(registry.find_worker_group_layout_descs("r1", 2), {0: {0: small}})
        registry.unregister("r1", 2, worker_id=0)
        self.assertIsNone(registry.find("r1", 2))


if __name__ == "__main__":
    unittest.main()
