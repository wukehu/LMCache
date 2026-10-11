# SPDX-License-Identifier: Apache-2.0
"""CPU native storage regression for source attribution and owned leases."""

# Standard
import time

# Third Party
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, PrefetchRequestSpec
from lmcache.v1.distributed.config import (
    EvictionConfig, L1ManagerConfig, L1MemoryManagerConfig, StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.source_attribution import SourceMetrics


@pytest.mark.parametrize("owned", [False, True])
def test_source_accounting_and_resident_read(monkeypatch, owned):
    """Retained counts finalize once; owned reads survive reservation expiry."""
    monkeypatch.setenv("LMCACHE_MP_OWNED_READ_LEASE", "1" if owned else "0")
    monkeypatch.setenv("LMCACHE_MP_SOURCE_MAPPING", "{}")
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metric_sink = SourceMetrics(provider.get_meter("coexistence"))
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.SourceMetrics", lambda: metric_sink
    )
    manager = StorageManager(StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=16 << 20, use_lazy=False, shm_name=""
            ), read_ttl_seconds=1,
        ),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    ))
    try:
        keys = [ObjectKey(chunk_hash=i.to_bytes(8, "big"),
                          model_name="owned-source", kv_rank=0) for i in range(2)]
        layout = MemoryLayoutDesc(shapes=[torch.Size([2048])], dtypes=[torch.int32])
        allocated = manager.reserve_write(keys, layout, mode="new")
        assert len(allocated) == 2
        for i, key in enumerate(keys):
            allocated[key].tensor.fill_(i + 7)
        manager.finish_write(keys)
        handle = manager.submit_prefetch_task(
            PrefetchRequestSpec(keys, {0: layout}),
            read_owner="request-A" if owned else None,
        )
        assert manager.query_prefetch_status(handle).get_indices_list() == [0, 1]
        assert manager.query_prefetch_status(handle).get_indices_list() == [0, 1]
        if owned:
            time.sleep(1.1)
            result = manager.begin_owned_read(keys, "request-A", 0)
            assert result is not None
            completion, objects = result
            manager.cancel_owned_read(keys, "request-A")
            for i, obj in enumerate(objects):
                assert torch.all(obj.tensor == i + 7).item()
            manager.finish_owned_read(completion)
            manager.finish_owned_read(completion)
        else:
            with manager.read_prefetched_results(keys) as objects:
                assert objects is not None
                for i, obj in enumerate(objects):
                    assert torch.all(obj.tensor == i + 7).item()
            manager.finish_read_prefetched(keys)
        totals = {}
        for resource in reader.get_metrics_data().resource_metrics:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    for point in metric.data.data_points:
                        assert point.attributes["source"] == "local_cpu"
                        assert point.attributes["stage"] == "retained"
                        totals[metric.name] = totals.get(metric.name, 0) + point.value
        assert totals["lmcache_mp.source_objects"] == 2
        assert totals["lmcache_mp.source_bytes"] == 16384
    finally:
        manager.close()
        provider.shutdown()
