import pandas as pd

from source.helpers.feature_preparer import prepare_features


# --- Dummy test data ---
DUMMY_DATA = {
    "X_train": pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4], "m1": [0.1, 0.2], "m2": [0.3, 0.4]}),
    "X_test": pd.DataFrame({"feat1": [5, 6], "feat2": [7, 8], "m1": [0.5, 0.6], "m2": [0.7, 0.8]}),
    "X_metr_train": pd.DataFrame({"m1": [0.1, 0.2], "m2": [0.3, 0.4]}),
    "X_metr_test": pd.DataFrame({"m1": [0.5, 0.6], "m2": [0.7, 0.8]}),
    "summaries_train": ["text a", "text b", "nom a", "nom b"],
    "summaries_test": ["text c", "text d", "nom c", "nom d"],
    "nom_summaries_train": ["nom a", "nom b"],
    "nom_summaries_test": ["nom c", "nom d"],
}

EXPECTED_OUTCOMES = {
    # Ohne Embeddings
    "lr": ("dataframe", ["feat1", "feat2", "m1", "m2"]),
    "gbdt": ("dataframe", ["feat1", "feat2", "m1", "m2"]),

    # RTE
    "lr_rte": ("dataframe", ["feat1", "feat2", "m1", "m2"]),
    "lr_rte_conc": ("dataframe", ["feat1", "feat2", "m1", "m2"]),
    "gbdt_rte_conc": ("dataframe", ["feat1", "feat2", "m1", "m2"]),
    "gbdt_rte": ("dataframe", ["feat1", "feat2", "m1", "m2"]),

    # Nur Text-Embeddings → Liste
    "lr_te": ("list", None),
    "gbdt_te": ("list", None),

    # Concatenation-Varianten
    "lr_conc1_te": ("dataframe", ["text"]),
    "gbdt_conc1_te": ("dataframe", ["text"]),
    "lr_conc2_te": ("dataframe", ["text", "m1", "m2"]),
    "gbdt_conc2_te": ("dataframe", ["text", "m1", "m2"]),
    "lr_conc3_te": ("dataframe", ["nom_text", "m1", "m2"]),
    "gbdt_conc3_te": ("dataframe", ["nom_text", "m1", "m2"]),
}


def test_prepare_features():
    passed, failed = 0, 0
    failed_tests = []

    for method_key, (expected_type, expected_columns) in EXPECTED_OUTCOMES.items():
        print(f"\n🔹 Testing {method_key} ...")

        try:
            X_train, X_test = prepare_features(
                method_key=method_key,
                data=DUMMY_DATA,
                # text_embedding_dict=DUMMY_EMB
            )
        except Exception as e:
            print(f"❌ {method_key}: Error during execution → {e}")
            failed += 1
            failed_tests.append((method_key, f"Execution error: {e}"))
            continue

        # --- Type Check ---
        if expected_type == "dataframe":
            if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
                print(f"❌ {method_key}: Expected DataFrame, got {type(X_train)} / {type(X_test)}")
                failed += 1
                failed_tests.append((method_key, "Wrong return type"))
                continue
        elif expected_type == "list":
            if not isinstance(X_train, list) or not isinstance(X_test, list):
                print(f"❌ {method_key}: Expected list, got {type(X_train)} / {type(X_test)}")
                failed += 1
                failed_tests.append((method_key, "Wrong return type"))
                continue

        # --- Column Check ---
        if expected_columns and isinstance(X_train, pd.DataFrame):
            got_cols = list(X_train.columns)
            if sorted(got_cols) != sorted(expected_columns):
                print(f"❌ {method_key}: Column mismatch")
                print("Expected:", expected_columns)
                print("Got:", got_cols)
                failed += 1
                failed_tests.append((method_key, "Column mismatch"))
                continue

        # --- Passed ---
        print(f"✅ {method_key} passed all checks.")
        if isinstance(X_train, pd.DataFrame):
            print("→ X_train.head(2):")
            print(X_train.head(2))
        elif isinstance(X_train, list):
            print(f"→ Sample train embeddings: {X_train[:2]}")
        passed += 1

    # --- Summary ---
    total = passed + failed
    print("\n" + "=" * 100)
    print("🧪 TEST SUMMARY")
    print("=" * 100)
    print(f"Total tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for method, reason in failed_tests:
            print(f" - {method}: {reason}")

    print("=" * 100)


if __name__ == "__main__":
    print("-" * 100)
    print("Running prepare_features() tests...")
    print("-" * 100)
    test_prepare_features()
