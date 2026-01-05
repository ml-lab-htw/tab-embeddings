from pprint import pprint

import pytest

from config.config_manager import ConfigManager
from src.exp_context import ExpContext
from src.param_grid_factory import ParamGridFactory
from standards.grids import EXPECTED_GRIDS


def test_all_param_grids():
    cfg = ConfigManager.load_yaml("./config/config.yaml")
    dataset_name = "cybersecurity"

    succeeded = 0
    total = len(EXPECTED_GRIDS)

    print("\n=== PARAM GRID STRUCTURE TEST ===\n")

    for method_key, expected_grid in EXPECTED_GRIDS.items():
        print(f"🔹 Method: {method_key}")

        try:
            ctx = ExpContext(
                method_key=method_key,
                dataset_name=dataset_name,
                cfg=cfg,
                embedding_key="DUMMY" if "_te" in method_key else None,
            )

            grid = ParamGridFactory.get_strategy(ctx).build(ctx)

            if grid == expected_grid:
                print("✅ Grid is correct.")
                succeeded += 1
            else:
                print("❌ Grid mismatch!")
                print("\nGenerated grid:")
                pprint(grid)
                print("\nExpected grid:")
                pprint(expected_grid)

        except Exception as e:
            print(f"❌ Failed to build grid for {method_key}: {e}")

        print("\n" + "-" * 80 + "\n")

    assert succeeded == total, f"{succeeded}/{total} grids passed"
