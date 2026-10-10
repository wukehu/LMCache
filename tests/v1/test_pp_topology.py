# SPDX-License-Identifier: Apache-2.0
"""Executable ownership contracts; no GPU or inference model is required."""

# Standard
from collections import Counter
import unittest

# First Party
from lmcache.integration.vllm.pp_topology import KVParallelTopology


class PPTopologyTests(unittest.TestCase):
    def test_deepseek_tp8_pp2_two_nodes(self):
        topology = KVParallelTopology(8, 2, 2, True)
        self.assertEqual(topology.server_shards(0), (0,))
        self.assertEqual(topology.server_shards(1), (1,))
        self.assertEqual(topology.readers_per_shard, 8)
        self.assertEqual(
            [r for r in range(16) if topology.is_writer(r)], [0, 8]
        )

    def test_two_stages_on_each_node(self):
        topology = KVParallelTopology(4, 4, 2, True)
        self.assertEqual(topology.server_shards(0), (0, 1))
        self.assertEqual(topology.server_shards(1), (2, 3))
        self.assertEqual(topology.readers_per_shard, 4)
        self.assertEqual(
            [r for r in range(16) if topology.is_writer(r)], [0, 4, 8, 12]
        )

    def test_tp_group_spans_nodes(self):
        topology = KVParallelTopology(8, 2, 4, True)
        self.assertEqual(
            [topology.server_shards(s) for s in range(4)],
            [(0,), (0,), (1,), (1,)],
        )
        self.assertEqual(topology.readers_per_shard, 4)
        self.assertEqual(
            [r for r in range(16) if topology.is_writer(r)], [0, 4, 8, 12]
        )

    def test_non_mla_shards_are_not_renumbered_per_node(self):
        topology = KVParallelTopology(2, 2, 2, False)
        self.assertEqual(topology.server_shards(0), (0, 1))
        self.assertEqual(topology.server_shards(1), (2, 3))
        self.assertEqual(topology.readers_per_shard, 1)

    def test_p2p_identity_survives_server_count_change(self):
        for mla in (False, True):
            a = KVParallelTopology(4, 4, 2, mla)
            b = KVParallelTopology(4, 4, 4, mla)
            self.assertEqual(a.shard_count, b.shard_count)
            self.assertEqual(
                [a.shard_id(r) for r in range(16)],
                [b.shard_id(r) for r in range(16)],
            )

    def test_reader_and_writer_balance_for_supported_geometries(self):
        for mla in (False, True):
            for tp in (1, 2, 4, 8):
                for pp in (1, 2, 3, 4):
                    for servers in range(1, tp * pp + 1):
                        try:
                            topology = KVParallelTopology(tp, pp, servers, mla)
                        except ValueError:
                            continue
                        consumers = Counter()
                        writers = Counter()
                        for rank in range(topology.world_size):
                            key = (topology.server_id(rank), topology.shard_id(rank))
                            consumers[key] += 1
                            if topology.is_writer(rank):
                                writers[key] += 1
                        self.assertEqual(set(consumers), set(writers))
                        self.assertEqual(set(writers.values()), {1})
                        self.assertEqual(
                            set(consumers.values()), {topology.readers_per_shard}
                        )

    def test_reject_invalid_or_misaligned_geometry(self):
        for args in ((0, 2, 1), (4, 4, 3), (4, 3, 2), (16, 16, 2)):
            with self.assertRaises(ValueError):
                KVParallelTopology(*args, mla=True)
        topology = KVParallelTopology(4, 2, 2, True)
        for rank in (-1, 8):
            with self.assertRaises(ValueError):
                topology.shard_id(rank)
        with self.assertRaises(ValueError):
            topology.server_shards(2)


if __name__ == "__main__":
    unittest.main()
