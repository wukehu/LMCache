# SPDX-License-Identifier: Apache-2.0
"""Versioned PP cache identity shared by compatible independent replicas."""

# Standard
from copy import deepcopy
from typing import Any
import hashlib
import json

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


PP_SHARED_CAPABILITY = "pp_shared_v1"


def pp_cache_namespace(model_name: str, layout: dict[str, Any]) -> str:
    """Return a versioned model key for a JSON-compatible global layout.

    The caller supplies model revision/configuration, TP/PP sizes, dtype,
    cache dtype, block geometry and layer partition. Node addresses, worker
    rank, request IDs and replica IDs must not appear in this descriptor.
    DeepSeek's runtime RoPE aliases are canonicalized without mutating the
    descriptor. Scaling parameters remain part of the identity.
    Invalid/non-JSON descriptors raise TypeError or ValueError.
    """
    layout = deepcopy(layout)
    model_config = layout.get("model_config", {})
    if model_config.get("model_type") in ("deepseek_v2", "deepseek_v3"):
        rope = model_config.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("rope_type", "default") != "default":
            # vLLM DeepseekV2Attention/DeepseekV2MLAAttention applies this
            # rewrite during model construction, after the scheduler forks.
            rope["rope_type"] = (
                "deepseek_yarn"
                if rope.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:32]
    namespace = f"{model_name}--{PP_SHARED_CAPABILITY}-{digest}"
    components = {
        key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
        for key, value in layout.items()
    }
    config_components = {
        key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
        for key, value in layout.get("model_config", {}).items()
    }
    logger.info("PP cache namespace=%s components=%s config_components=%s", namespace, components, config_components)
    return namespace
