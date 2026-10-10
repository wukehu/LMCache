# SPDX-License-Identifier: Apache-2.0
"""Resolve PP server order from observed engine Pod placement."""

# Standard
from typing import Any
from pathlib import Path
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request


def discover_pp_endpoints(service_name: str, members: int, timeout: float = 120, node_endpoints: dict[str, str] | None = None) -> list[str]:
    """Resolve this replica's node-local MP endpoints from Kubernetes.

    Uses the mounted service-account token/CA and namespace. Requires get/list
    Pods. Stable per-node Service endpoints must be injected by the deployer.
    Credentials are never logged. A 401
    or 403 fails immediately; scheduling/rollout incompleteness retries up to
    timeout seconds. Raises RuntimeError or TimeoutError on failure.
    """
    if not service_name or "/" in service_name or timeout <= 0 or not node_endpoints:
        raise ValueError("A local Service name and positive timeout are required")
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = account.joinpath("namespace").read_text().strip()
    pod_name = os.environ.get("POD_NAME") or os.environ["HOSTNAME"]
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    host = f"[{host}]" if ":" in host else host
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    base = f"https://{host}:{port}/api/v1/namespaces/{urllib.parse.quote(namespace, safe='')}"
    context = ssl.create_default_context(cafile=str(account / "ca.crt"))
    deadline = time.monotonic() + timeout

    def read(path: str) -> dict[str, Any]:
        token = account.joinpath("token").read_text().strip()
        request = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, context=context, timeout=min(10, max(0.1, deadline - time.monotonic()))) as response:
            return json.load(response)

    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            current = read("/pods/" + urllib.parse.quote(pod_name, safe=""))
            labels = current["metadata"].get("labels", {})
            if "grove.io/podcliquescalinggroup" in labels:
                selector = "grove.io/podcliquescalinggroup=" + labels["grove.io/podcliquescalinggroup"]
            elif "leaderworkerset.sigs.k8s.io/group-key" in labels:
                selector = "leaderworkerset.sigs.k8s.io/group-key=" + labels["leaderworkerset.sigs.k8s.io/group-key"]
            elif members == 1:
                return pp_node_endpoints(current, [current], members, node_endpoints)
            else:
                raise RuntimeError("Unknown multi-node engine group identity")
            pods = read("/pods?" + urllib.parse.urlencode({"labelSelector": selector}))["items"]
            return pp_node_endpoints(current, pods, members, node_endpoints)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(f"Kubernetes denied PP endpoint discovery: HTTP {exc.code}") from exc
            last_error = exc
        except (ValueError, KeyError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"PP endpoint discovery did not converge: {last_error}")


def pp_node_endpoints(
    current: dict[str, Any], pods: list[dict[str, Any]], members: int, node_endpoints: dict[str, str]
) -> list[str]:
    """Return node-bound Service endpoints in engine node-rank order.

    Supports observed Dynamo Grove and LWS member labels. A single-member
    engine needs only its own Pod. Excludes other replicas and terminating
    Pods. Missing, duplicate, unplaced or unrecognized members raise ValueError;
    callers can retry while Kubernetes scheduling is still converging.
    """
    if members < 1 or not node_endpoints:
        raise ValueError("Invalid member count or node RPC port")
    labels = current["metadata"].get("labels", {})
    if members == 1:
        selected = [(0, current)]
    elif "grove.io/podcliquescalinggroup" in labels:
        scope = (
            "grove.io/podcliquescalinggroup",
            "grove.io/podcliquescalinggroup-replica-index",
        )
        if any(key not in labels for key in scope):
            raise ValueError("Incomplete Grove replica identity")
        selected = []
        for pod in pods:
            meta = pod["metadata"]
            peer = meta.get("labels", {})
            if meta.get("deletionTimestamp") or any(peer.get(k) != labels[k] for k in scope):
                continue
            clique = peer.get("grove.io/podclique", "")
            index = int(peer.get("grove.io/podclique-pod-index", "-1"))
            if clique.endswith("-ldr") and index == 0:
                rank = 0
            elif clique.endswith("-wkr") and index >= 0:
                rank = index + 1
            else:
                raise ValueError("Unrecognized Grove PP member rank")
            selected.append((rank, pod))
    elif "leaderworkerset.sigs.k8s.io/group-key" in labels:
        group = labels["leaderworkerset.sigs.k8s.io/group-key"]
        selected = [
            (int(p["metadata"].get("labels", {}).get("leaderworkerset.sigs.k8s.io/worker-index", "-1")), p)
            for p in pods
            if not p["metadata"].get("deletionTimestamp")
            and p["metadata"].get("labels", {}).get("leaderworkerset.sigs.k8s.io/group-key") == group
        ]
    else:
        raise ValueError("Unknown multi-node engine group identity")
    selected.sort(key=lambda entry: entry[0])
    if [rank for rank, _ in selected] != list(range(members)):
        raise ValueError("Engine node ranks are incomplete or duplicated")
    nodes: set[str] = set()
    endpoints = []
    for _, pod in selected:
        node = pod.get("spec", {}).get("nodeName", "")
        if not node or node in nodes or node not in node_endpoints:
            raise ValueError("Engine members require distinct nodes with assigned MP Services")
        endpoint = node_endpoints[node]
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "tcp" or not parsed.hostname or parsed.port != 5555:
            raise ValueError("Invalid node MP Service endpoint")
        endpoints.append(endpoint)
        nodes.add(node)
    return endpoints
