from pprint import pprint

import pytest

from src.exp_context import ExpContext
from src.pipeline_factory import PipelineFactory
from config.config_manager import ConfigManager
from standards.pipelines import EXPECTED_PIPELINES
from tests.utils import compare_pipelines


def test_all_pipelines():
    cfg = ConfigManager.load_yaml("./config/config.yaml")
    dataset_name = "cybersecurity"

    dataset_cfg = cfg.datasets[dataset_name]
    feature_cfg = cfg.features[dataset_name]

    nominal_features = feature_cfg["nominal_features"]
    text_features = feature_cfg.get("text_features", [])

    succeeded = 0
    total = len(EXPECTED_PIPELINES)

    print("\n=== PIPELINE STRUCTURE TEST ===\n")

    for method_key, expected_pipeline in EXPECTED_PIPELINES.items():
        print(f"🔹 Method: {method_key}")

        try:
            ctx = ExpContext(
                method_key=method_key,
                dataset_name=dataset_name,
                cfg=cfg,
            )

            ctx.nominal_features = nominal_features
            ctx.text_features = text_features

            ctx.numerical_features = ["num_1", "num_2"]

            strategy = PipelineFactory.get_strategy(ctx)

            pipeline = strategy.build(ctx)

            if compare_pipelines(pipeline, expected_pipeline):
                print("✅ Pipeline is correct.")
                succeeded += 1
            else:
                print("❌ Pipeline mismatch!")
                print("\nGenerated pipeline:")
                pprint(pipeline)
                print("\nExpected pipeline:")
                pprint(expected_pipeline)

        except Exception as e:
            print(f"❌ Failed to build pipeline for {method_key}: {e}")

        print("\n" + "-" * 80 + "\n")

    assert succeeded == total, f"{succeeded}/{total} pipelines passed"
