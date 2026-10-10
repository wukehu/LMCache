# SPDX-License-Identifier: Apache-2.0
"""Global KV shard identity and node-local ownership for TP x PP, DP=DCP=1."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class KVParallelTopology:
    """Map contiguous engine ranks to servers without renumbering KV objects.

    Args:
        tp_size: Tensor-parallel ranks per pipeline stage.
        pp_size: Pipeline stages in one engine replica.
        n_servers: Node-local servers, in engine node-rank order.
        mla: Whether TP ranks replicate each stage's KV bytes.

    Raises:
        ValueError: If sizes are nonpositive, exceed the existing object-key
            encoding, or server boundaries bisect unequal portions of stages.

    Compatible replicas use the same shard IDs independently of server count.
    Replica/session identity must not be added to these IDs. DCP and layout
    conversion are intentionally outside this geometry contract.
    """

    tp_size: int
    pp_size: int
    n_servers: int
    mla: bool

    def __post_init__(self) -> None:
        if min(self.tp_size, self.pp_size, self.n_servers) < 1:
            raise ValueError("TP, PP and server counts must be positive")
        if self.world_size > 255:
            raise ValueError("TP x PP exceeds the current 8-bit rank encoding")
        if self.world_size % self.n_servers:
            raise ValueError("TP x PP must be divisible by the server count")
        local = self.ranks_per_server
        if local % self.tp_size and self.tp_size % local:
            raise ValueError("Server rank blocks must align with TP stages")

    @property
    def world_size(self) -> int:
        """Return the engine rank count, excluding independent replicas."""
        return self.tp_size * self.pp_size

    @property
    def ranks_per_server(self) -> int:
        """Return the size of each contiguous server rank block."""
        return self.world_size // self.n_servers

    @property
    def shard_count(self) -> int:
        """Return distinct KV pieces in one global token chunk."""
        return self.pp_size if self.mla else self.world_size

    def shard_id(self, rank: int) -> int:
        """Return the global KV piece held by a validated engine rank."""
        self.validate_rank(rank)
        return rank // self.tp_size if self.mla else rank

    def server_id(self, rank: int) -> int:
        """Return the server assigned to a validated engine rank."""
        self.validate_rank(rank)
        return rank // self.ranks_per_server

    def server_shards(self, server_id: int) -> tuple[int, ...]:
        """Return precisely the global pieces a server must find for a hit."""
        if not 0 <= server_id < self.n_servers:
            raise ValueError("Server index out of range")
        start = server_id * self.ranks_per_server
        return tuple(
            dict.fromkeys(
                self.shard_id(rank)
                for rank in range(start, start + self.ranks_per_server)
            )
        )

    @property
    def readers_per_shard(self) -> int:
        """Return actual consumers of each local object for one request.

        Alignment validation makes this count uniform across every local
        piece. Locks are per request; concurrent replicas reserve separately.
        """
        return min(self.tp_size, self.ranks_per_server) if self.mla else 1

    def is_writer(self, rank: int) -> bool:
        """Select exactly one writer per (server, global shard)."""
        self.validate_rank(rank)
        if not self.mla:
            return True
        local_rank = rank % self.ranks_per_server
        return local_rank % self.tp_size == 0

    def validate_rank(self, rank: int) -> None:
        """Raise ValueError when a rank is outside this engine replica."""
        if not 0 <= rank < self.world_size:
            raise ValueError("Engine rank out of range")
