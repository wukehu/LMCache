# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC registration must work from a background recovery thread."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import worker_transfer


class CompletedRegistration:
    def result(self, timeout):
        return None


class RegistrationClient:
    def register_kv_cache(self, *args):
        return CompletedRegistration()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class BackgroundRegistrationTest(unittest.TestCase):
    def test_register_initializes_background_cuda_context(self):
        from cuda.bindings import driver

        cache = {"layer": torch.empty(1024, device="cuda:0")}
        client = RegistrationClient()

        def export(caches):
            result, _, _ = driver.cuMemGetAddressRange(caches["layer"].data_ptr())
            self.assertEqual(result, driver.CUresult.CUDA_SUCCESS)
            return caches

        def register():
            context = worker_transfer.LMCacheDrivenTransferContext(1, client)
            context.register(cache, "test-model", 1, 1, 5.0)

        with patch.object(worker_transfer, "wrap_kv_caches", side_effect=export):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(register).result(timeout=15)


if __name__ == "__main__":
    unittest.main()

