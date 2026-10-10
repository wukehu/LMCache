# SPDX-License-Identifier: Apache-2.0
"""Node order follows engine ranks, never hostname sorting or another replica."""

# Standard
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch
import json
import unittest
import urllib.error

# First Party
from lmcache.integration.vllm.kubernetes_pp import discover_pp_endpoints, pp_node_endpoints


class PPNodeEndpointsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoints = {"z-node": "tcp://10.96.1.1:5555", "a-node": "tcp://10.96.1.2:5555"}
        self.pods = []
        for role, index, node, ip in (("ldr", 0, "z-node", "192.0.2.40"), ("wkr", 0, "a-node", "192.0.2.39")):
            self.pods.append({"metadata": {"labels": {
                "grove.io/podcliquescalinggroup": "service-decode",
                "grove.io/podcliquescalinggroup-replica-index": "0",
                "grove.io/podclique": "service-" + role,
                "grove.io/podclique-pod-index": str(index),
            }}, "spec": {"nodeName": node}, "status": {"hostIP": ip}})

    def test_rank_order_and_replica_isolation(self) -> None:
        other = deepcopy(self.pods[0])
        other["metadata"]["labels"]["grove.io/podcliquescalinggroup-replica-index"] = "1"
        expected = ["tcp://10.96.1.1:5555", "tcp://10.96.1.2:5555"]
        for current in self.pods:
            self.assertEqual(pp_node_endpoints(current, [other, *reversed(self.pods)], 2, self.endpoints), expected)

    def test_missing_duplicate_or_same_node_is_rejected(self) -> None:
        for peers in (self.pods[:1], self.pods + self.pods[:1]):
            with self.assertRaises(ValueError):
                pp_node_endpoints(self.pods[0], peers, 2, self.endpoints)
        peers = deepcopy(self.pods)
        peers[1]["spec"] = peers[0]["spec"]
        with self.assertRaises(ValueError):
            pp_node_endpoints(peers[0], peers, 2, self.endpoints)

    def test_discovery_uses_injected_endpoints_without_service_permission(self) -> None:
        urls = []

        def request(req, **kwargs):
            urls.append(req.full_url)
            body = {"items": self.pods} if "labelSelector=" in req.full_url else self.pods[0]
            return BytesIO(json.dumps(body).encode())

        env = {"HOSTNAME": "pod-0", "KUBERNETES_SERVICE_HOST": "10.96.0.1"}
        with patch.dict("os.environ", env), patch("pathlib.Path.read_text", return_value="test"), patch("ssl.create_default_context"), patch("urllib.request.urlopen", side_effect=request):
            endpoints = discover_pp_endpoints("mp", 2, node_endpoints=self.endpoints)
        self.assertEqual(endpoints, ["tcp://10.96.1.1:5555", "tcp://10.96.1.2:5555"])
        self.assertEqual(len(urls), 2)
        self.assertTrue(all("/services/" not in url for url in urls))

    def test_permission_denial_fails_without_retry(self) -> None:
        env = {"HOSTNAME": "pod-0", "KUBERNETES_SERVICE_HOST": "10.96.0.1"}
        denied = urllib.error.HTTPError("https://cluster", 403, "Forbidden", {}, None)
        with patch.dict("os.environ", env), patch("pathlib.Path.read_text", return_value="test"), patch("ssl.create_default_context"), patch("urllib.request.urlopen", side_effect=denied) as request:
            with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
                discover_pp_endpoints("mp", 2, node_endpoints=self.endpoints)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
