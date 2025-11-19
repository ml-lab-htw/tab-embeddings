from source.helpers.run_setup import build_param_grid
from tests.standards.grids import EXPECTED_GRIDS


def test_param_grids():
    for method_key, expected_grid in EXPECTED_GRIDS.items():
        generated_grid = build_param_grid(method_key)

        if generated_grid != expected_grid:
            print(f"❌ Grid mismatch for method: {method_key}")
            print("Generated:")
            print(generated_grid)
            print("Expected:")
            print(expected_grid)
            print()
        else:
            print(f"✅ {method_key} grid OK")


if __name__ == "__main__":
    print("-" * 100)
    print("Run param_grid tests...")
    print("-" * 100)
    test_param_grids()
