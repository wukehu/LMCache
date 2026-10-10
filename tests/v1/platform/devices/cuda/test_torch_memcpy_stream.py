# SPDX-License-Identifier: Apache-2.0
"""CUDA stream ordering of the Torch fallback's public pointer-copy operation."""

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.platform import torch_ops

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
@pytest.mark.parametrize("offset,nbytes", [(0, 65536), (31, 131101)])
def test_pointer_copy_orders_with_current_stream(
    direction: TransferDirection, offset: int, nbytes: int
) -> None:
    """A pending writer or reader must not race a pointer copy on another stream."""
    size = offset + nbytes + 31
    gpu = torch.zeros(size, dtype=torch.uint8, device="cuda")
    host = torch.full((size,), 7, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()
    observed = torch.empty_like(gpu)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        # A short GPU delay guarantees outstanding work without blocking the CPU.
        torch.cuda._sleep(50000000)
        if direction == TransferDirection.D2H:
            gpu.fill_(7)
            host.fill_(99)
            dest, src = host.data_ptr(), gpu.data_ptr()
        else:
            gpu.fill_(99)
            dest, src = gpu.data_ptr(), host.data_ptr()
        torch_ops.lmcache_memcpy_async(
            dest + offset, src + offset, nbytes, direction, offset, 65536
        )
        if direction == TransferDirection.H2D:
            observed.copy_(gpu)
    stream.synchronize()
    result = host if direction == TransferDirection.D2H else observed.cpu()
    assert torch.all(result[offset : offset + nbytes] == 7)
    assert torch.all(result[:offset] == 99)
    assert torch.all(result[offset + nbytes :] == 99)