# SPDX-License-Identifier: Apache-2.0
"""Public adapter contract for global PP pieces and local consumers."""

# Standard
from collections import Counter
import unittest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy


class PPParallelStrategyTest(unittest.TestCase):
    def test_local_consumers_and_one_writer_per_piece(self) -> None:
        for mla in (False, True):
            for tp, pp, servers in ((8, 2, 2), (4, 4, 2), (8, 2, 4)):
                consumers: Counter = Counter()
                writers: Counter = Counter()
                for rank in range(tp * pp):
                    strategy = ParallelStrategy(mla, tp * pp, rank, tp, pp, servers)
                    server = rank // (tp * pp // servers)
                    key = (server, strategy.kv_worker_id)
                    consumers[key] += 1
                    writers[key] += int(strategy.is_kv_writer)
                    self.assertEqual(strategy.kv_world_size, pp if mla else tp * pp)
                    self.assertEqual(
                        strategy.kv_worker_id, rank // tp if mla else rank
                    )
                self.assertEqual(set(writers.values()), {1})
                self.assertEqual(set(consumers.values()), {strategy.num_kv_readers})

    def test_pp1_contract_unchanged(self) -> None:
        for mla in (False, True):
            for rank in range(8):
                strategy = ParallelStrategy(mla, 8, rank, 8, 1, 2)
                self.assertEqual(strategy.kv_world_size, 1 if mla else 4)
                self.assertEqual(strategy.kv_worker_id, 0 if mla else rank % 4)
                self.assertEqual(strategy.num_kv_readers, 4 if mla else 1)
                self.assertEqual(strategy.is_kv_writer, rank % 4 == 0 if mla else True)

    def test_reject_unsupported_pp_geometry(self) -> None:
        for strategy in (
            ParallelStrategy(True, 16, 0, 8, 2, 2, dcp_size=2),
            ParallelStrategy(True, 8, 0, 8, 2, 2),
            ParallelStrategy(True, 16, 16, 8, 2, 2),
        ):
            with self.assertRaises(ValueError):
                _ = strategy.kv_world_size


if __name__ == "__main__":
    unittest.main()
