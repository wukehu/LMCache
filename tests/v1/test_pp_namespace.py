# SPDX-License-Identifier: Apache-2.0
"""Compatible replicas share namespace; incompatible global layouts do not."""

# Standard
from copy import deepcopy
import unittest

# First Party
from lmcache.integration.vllm.pp_namespace import pp_cache_namespace


class PPNamespaceTest(unittest.TestCase):
    def test_deepseek_rope_runtime_normalization_preserves_identity(self) -> None:
        scheduler = {"model_config": {
            "model_type": "deepseek_v3",
            "rope_parameters": {"rope_type": "yarn", "factor": 40},
        }, "tp": 8, "pp": 2}
        worker = deepcopy(scheduler)
        worker["model_config"]["rope_parameters"]["rope_type"] = "deepseek_yarn"
        self.assertEqual(
            pp_cache_namespace("model", scheduler),
            pp_cache_namespace("model", worker),
        )
        self.assertEqual(scheduler["model_config"]["rope_parameters"]["rope_type"], "yarn")
        for field, value in (("factor", 8), ("apply_yarn_scaling", False), ("rope_type", "default")):
            changed = deepcopy(scheduler)
            changed["model_config"]["rope_parameters"][field] = value
            self.assertNotEqual(
                pp_cache_namespace("model", scheduler),
                pp_cache_namespace("model", changed),
            )

    def test_other_architectures_keep_distinct_rope_types(self) -> None:
        original = {"model_config": {
            "model_type": "qwen3", "rope_parameters": {"rope_type": "yarn", "factor": 4}
        }}
        changed = deepcopy(original)
        changed["model_config"]["rope_parameters"]["rope_type"] = "deepseek_yarn"
        self.assertNotEqual(
            pp_cache_namespace("model", original), pp_cache_namespace("model", changed)
        )

    def test_mapping_order_does_not_change_identity(self) -> None:
        self.assertEqual(
            pp_cache_namespace("model", {"tp": 8, "pp": 2}),
            pp_cache_namespace("model", {"pp": 2, "tp": 8}),
        )

    def test_layout_changes_isolate_cache(self) -> None:
        layout = {"tp": 8, "pp": 2, "dtype": "bfloat16", "cache_dtype": "auto", "layer_partition": "30,31", "revision": "r1", "block_size": 16}
        baseline = pp_cache_namespace("model", layout)
        for field, value in (("tp", 4), ("pp", 4), ("dtype", "float16"), ("cache_dtype", "fp8"), ("layer_partition", "31,30"), ("revision", "r2"), ("block_size", 32)):
            self.assertNotEqual(baseline, pp_cache_namespace("model", {**layout, field: value}))
        self.assertNotEqual(baseline, pp_cache_namespace("other-model", layout))


if __name__ == "__main__":
    unittest.main()
