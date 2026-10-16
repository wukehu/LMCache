# SPDX-License-Identifier: Apache-2.0
"""Source classification contracts independent of native storage kernels."""

# Standard
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.source_attribution import Source, SourceLedger


@pytest.mark.parametrize("adapter", [2, 3, 42])
@pytest.mark.parametrize("override", [None, "unknown", "remote_store", "peer"])
def test_p2p_source_survives_dynamic_ids_and_honors_override(
    adapter: int, override: str | None
) -> None:
    metrics = Mock()
    mapping = {} if override is None else {f"l2:{adapter}": override}
    ledger = SourceLedger(
        metrics, Source("local_cpu", "l1", "dram", "l1"), mapping, "lookup"
    )
    ledger.loaded(adapter, "p2p", [("key", 1024)])
    ledger.finish(["key"])
    expected = Source(override or "peer", "l2", "p2p", str(adapter))
    assert ledger.snapshot()["key"].source == expected
    assert [call.args[0] for call in metrics.record.call_args_list] == [
        "loaded",
        "retained",
    ]
    for call in metrics.record.call_args_list:
        assert call.args[2][0].source == expected
        assert call.args[2][0].size_bytes == 1024


@pytest.mark.parametrize("backend", ["fs", "mooncake_store", "unrecognized"])
def test_undeclared_storage_media_remain_unknown(backend: str) -> None:
    ledger = SourceLedger(Mock(), Source("local_cpu", "l1", "dram", "l1"), {}, "lookup")
    ledger.loaded(2, backend, [("key", 1024)])
    assert ledger.snapshot()["key"].source.source == "unknown"
