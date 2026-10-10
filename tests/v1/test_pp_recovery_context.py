# SPDX-License-Identifier: Apache-2.0
"""Recovery must bind the KV device on the thread exporting IPC handles."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import threading
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod


class PPRecoveryContextTest(unittest.TestCase):
    def test_cuda_recovery_binds_device_before_register(self) -> None:
        self.check_recovery("cuda:1", False)

    def test_cpu_recovery_does_not_initialize_cuda(self) -> None:
        self.check_recovery("cpu", False)

    def test_device_failure_keeps_worker_unhealthy(self) -> None:
        self.check_recovery("cuda:0", True)

    def check_recovery(self, device_name: str, bind_fails: bool) -> None:
        """Exercise registration and the heartbeat's published callback."""
        device = torch.device(device_name)
        thread_state = threading.local()
        registered_threads: list[int] = []
        heartbeats: list[MagicMock] = []
        main_thread = threading.get_ident()

        def bind(actual: torch.device) -> None:
            self.assertEqual(actual, device)
            if bind_fails:
                raise RuntimeError("device unavailable")
            thread_state.bound = actual

        def register(*args: object, **kwargs: object) -> None:
            if threading.get_ident() != main_thread and device.type == "cuda":
                self.assertEqual(getattr(thread_state, "bound", None), device)
            registered_threads.append(threading.get_ident())

        def heartbeat_factory(**kwargs: object) -> MagicMock:
            heartbeat = MagicMock()
            heartbeat.health_event = kwargs["health_event"]
            heartbeat.stop_requested = False
            heartbeats.append(heartbeat)
            return heartbeat

        context = MagicMock()
        context.register.side_effect = register
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter_mod.RequestClientFactory, "create"))
            stack.enter_context(
                patch.object(adapter_mod, "get_lmcache_chunk_size", return_value=256)
            )
            stack.enter_context(
                patch.object(adapter_mod, "get_experimental", return_value=set())
            )
            stack.enter_context(
                patch.object(
                    adapter_mod, "HeartbeatThread", side_effect=heartbeat_factory
                )
            )
            stack.enter_context(
                patch.object(adapter_mod, "create_transfer_context", return_value=context)
            )
            set_device = stack.enter_context(
                patch.object(adapter_mod.torch_dev, "set_device", side_effect=bind)
            )
            adapter = adapter_mod.LMCacheMPWorkerAdapter(
                server_url="tcp://127.0.0.1:0",
                context=MagicMock(),
                model_name="recovery-test",
                vllm_block_size=16,
                parallel_strategy=adapter_mod.ParallelStrategy(False, 2, 1, 1, 2, 1),
            )
            tensor = MagicMock()
            tensor.device = device
            adapter.register_kv_caches({"layer.0": tensor}, layout_hints={})
            adapter.submit_store_request(
                "request",
                adapter_mod.LoadStoreOp(
                    token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4
                ),
                MagicMock(),
            )
            heartbeat = heartbeats[0]
            heartbeat.health_event.clear()
            callback = heartbeat.register_recover_callback.call_args.args[0]
            with ThreadPoolExecutor(max_workers=1) as pool:
                recovered = pool.submit(callback).result(timeout=5)
            self.assertEqual(recovered, not bind_fails)
            self.assertEqual(len(registered_threads), 1 if bind_fails else 2)
            if not bind_fails:
                self.assertNotEqual(registered_threads[0], registered_threads[1])
            self.assertEqual(set_device.call_count, int(device.type != "cpu"))
            if bind_fails:
                self.assertFalse(adapter.is_healthy)
            adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
