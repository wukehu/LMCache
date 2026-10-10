# SPDX-License-Identifier: Apache-2.0
"""Real filesystem-to-L1 reload for PP stages with unequal object sizes."""

# Standard
import tempfile
import time
import unittest

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey, PrefetchRequestSpec
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2Adapter, FSL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.prefetch_controller import PrefetchController
from lmcache.v1.distributed.storage_controllers.prefetch_policy import DefaultPrefetchPolicy
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.platform import current_device_spec


class PPFilesystemReloadTest(unittest.TestCase):
    def test_reload_unequal_stage_bytes_into_correct_allocations(self) -> None:
        keys = [
            ObjectKey(ObjectKey.IntHash2Bytes(1), "pp", ObjectKey.ComputeKVRank(2, r, 2, r), 0)
            for r in (0, 1)
        ]
        tensors = [torch.full((layers, 16, 8), r + 1, dtype=torch.bfloat16) for r, layers in enumerate((30, 31))]
        layouts = {key: MemoryLayoutDesc([t.shape], [t.dtype]) for key, t in zip(keys, tensors, strict=True)}
        objects = [
            TensorMemoryObj(
                raw_data=t,
                metadata=MemoryObjMetadata(shape=t.shape, dtype=t.dtype, address=0, phy_size=t.numel() * t.element_size(), ref_count=0),
                parent_allocator=None,
            )
            for t in tensors
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = FSL2AdapterConfig(base_path=directory)
            adapter = FSL2Adapter(config)
            l1 = L1Manager(L1ManagerConfig(memory_config=L1MemoryManagerConfig(
                size_in_bytes=16 * 1024 * 1024,
                init_size_in_bytes=8 * 1024 * 1024,
                align_bytes=4096,
                use_lazy=current_device_spec.is_pin_supported,
            )))
            controller = PrefetchController(
                l1_manager=l1, l2_adapters=[adapter],
                adapter_descriptors=[AdapterDescriptor(index=0, config=config)],
                policy=DefaultPrefetchPolicy(),
            )
            controller.start()
            try:
                store = adapter.submit_store_task(keys, objects)
                deadline = time.monotonic() + 10
                while store not in adapter.pop_completed_store_tasks():
                    self.assertLess(time.monotonic(), deadline, "filesystem store timed out")
                    time.sleep(0.02)
                request = controller.submit_prefetch_request(PrefetchRequestSpec(
                    keys=keys, group_layout_descs={0: layouts[keys[0]]},
                    object_layout_descs=layouts, num_kv_readers=4,
                    attn_desc=AttnWindowDesc([-1], world_size=2),
                ))
                deadline = time.monotonic() + 10
                result = None
                while result is None:
                    self.assertLess(time.monotonic(), deadline, "filesystem reload timed out")
                    result = controller.query_prefetch_result(request)
                    time.sleep(0.02)
                self.assertEqual(result.count_leading_ones(), 2)
                loaded = l1.unsafe_read(keys)
                for key, expected in zip(keys, tensors, strict=True):
                    error, obj = loaded[key]
                    self.assertEqual(error, L1Error.SUCCESS)
                    self.assertIsNotNone(obj)
                    self.assertEqual(obj.get_shapes(), [expected.shape])
                    self.assertTrue(torch.equal(obj.get_tensor(0), expected))
                l1.finish_read(keys, read_locks=4)
            finally:
                controller.stop()
                adapter.close()
                l1.close()


if __name__ == "__main__":
    unittest.main()
