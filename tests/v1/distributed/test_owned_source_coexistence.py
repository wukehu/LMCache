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
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.source_attribution import (
    SourceMetrics,
    TokenConsumptionLedger,
    merge_token_sources,
    source_token_attribution_enabled,
)


@pytest.mark.parametrize("ranks", [1, 2, 16])
def test_logical_source_tokens_do_not_multiply_with_tp(ranks):
    spans = merge_token_sources(
        [[(0, 256, "local_cpu"), (256, 512, "remote_store")]] * ranks, 17, 511
    )
    ledger = TokenConsumptionLedger(spans, 17, 511)
    assert ledger.consumed(511) == {}  # Lookup alone cannot prove a load.
    assert ledger.loaded() == {"local_cpu": 239, "remote_store": 255}
    assert ledger.loaded() == {}
    assert ledger.consumed(256) == {"local_cpu": 239}
    assert ledger.consumed(512) == {"remote_store": 255}
    assert ledger.consumed(1024) == {}


def test_source_intersection_handles_mixed_missing_and_overlapping_shards():
    assert merge_token_sources(
        [
            [(0, 256, "local_cpu"), (256, 512, "local_disk")],
            [(0, 256, "remote_store"), (256, 384, "local_disk")],
        ],
        0,
        512,
    ) == [(0, 256, "mixed"), (256, 384, "local_disk"), (384, 512, "unknown")]
    assert merge_token_sources(
        [[(0, 256, "local_cpu"), (128, 256, "local_cpu")]], 0, 256
    ) == [(0, 128, "local_cpu"), (128, 256, "unknown")]
    assert merge_token_sources([[(0, 256, "local_cpu")], []], 0, 256) == [
        (0, 256, "unknown")
    ]


@pytest.mark.parametrize("loaded_first", [False, True])
def test_invalidated_allocation_ignores_late_load_and_model_callbacks(loaded_first):
    ledger = TokenConsumptionLedger([(0, 512, "local_disk")], 16, 511)
    if loaded_first:
        assert ledger.loaded() == {"local_disk": 495}
    ledger.invalidate()
    assert ledger.loaded() == {}
    assert ledger.consumed(512) == {}


@pytest.mark.parametrize("flag", [None, False, "false", "0"])
def test_source_token_switch_explicit_opt_out(flag):
    assert not source_token_attribution_enabled(
        {"lmcache.mp.source_token_attribution": flag}
    )


@pytest.mark.parametrize("config", [None, {}, {"unrelated": False}])
def test_source_token_switch_defaults_to_enabled(config):
    assert source_token_attribution_enabled(config)


@pytest.mark.parametrize("flag", [True, 1, "true", "1", "yes", "on"])
def test_source_token_switch_explicit_enable(flag):
    assert source_token_attribution_enabled(
        {"lmcache.mp.source_token_attribution": flag}
    )


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
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 << 20, use_lazy=False, shm_name=""
                ),
                read_ttl_seconds=1,
            ),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
    )
    try:
        keys = [
            ObjectKey(
                chunk_hash=i.to_bytes(8, "big"), model_name="owned-source", kv_rank=0
            )
            for i in range(2)
        ]
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


@pytest.mark.parametrize("config", [None, {}, {"unrelated": True}])
def test_default_attribution_does_not_block_unsupported_engine(config):
    assert not source_token_attribution_enabled(config, supported=False)


@pytest.mark.parametrize("flag", [False, "false", "0"])
def test_explicit_opt_out_remains_valid_on_unsupported_engine(flag):
    assert not source_token_attribution_enabled(
        {"lmcache.mp.source_token_attribution": flag}, supported=False
    )


@pytest.mark.parametrize("flag", [True, "true", "1"])
def test_explicit_attribution_requires_supported_engine(flag):
    with pytest.raises(ValueError, match="unsupported"):
        source_token_attribution_enabled(
            {"lmcache.mp.source_token_attribution": flag}, supported=False
        )


@pytest.mark.parametrize("load_first", [False, True])
def test_async_source_ledger_joins_load_and_execution_once(load_first):
    ledger = TokenConsumptionLedger([(0, 512, "local_cpu")], 16, 511)
    if load_first:
        ledger.loaded()
        assert ledger.completed(511) == {"local_cpu": 495}
    else:
        assert ledger.completed(511) == {}
        ledger.loaded()
        assert ledger.completed() == {"local_cpu": 495}
    assert ledger.completed(1024) == {}
    ledger.invalidate()
    assert ledger.completed(2048) == {}


@pytest.mark.parametrize("old_frame", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_async_connector_rejects_old_allocation_and_invalid_load(old_frame, invalid):
    from types import SimpleNamespace
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPConnectorMetadata

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._source_tokens_enabled = True
    connector._source_token_stats = SimpleNamespace(data={})
    tracker = SimpleNamespace(
        source_allocation_id="new", source_tokens=TokenConsumptionLedger(
            [(0, 512, "local_cpu")], 16, 511
        ), source_loaded_tokens=TokenConsumptionLedger(
            [(0, 512, "local_cpu")], 0, 512
        ), allocated_block_ids={0: [7]},
    )
    connector.request_trackers = {"req": tracker}
    frame = LMCacheMPConnectorMetadata()
    frame.source_allocations = {"req": "old" if old_frame else "new"}
    output = SimpleNamespace(finished_recving={"req"}, invalid_block_ids={7} if invalid else set())
    connector.on_model_step_completed_with_metadata("req", 511, frame, output)
    counts = dict(connector._source_token_stats.data.get("reused", {}))
    assert counts == ({} if old_frame or invalid else {"local_cpu": 495})
    connector.on_model_step_completed_with_metadata("req", 511, frame, output)
    assert connector._source_token_stats.data.get("reused", {}) == counts
    # Cancellation removes the tracker; a late completion cannot recreate it.
    connector.request_trackers.clear()
    connector.on_model_step_completed_with_metadata("req", 511, frame, output)
    assert connector._source_token_stats.data.get("reused", {}) == counts
