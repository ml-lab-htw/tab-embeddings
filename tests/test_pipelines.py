from pprint import pprint

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from source.config.config import METHOD_KEYS, FEATURES, DATASETS
from source.utils.load_data import load_dataset
from source.helpers.run_setup import build_pipeline
from standards.pipelines import EXPECTED_PIPELINES


def compare_pipelines(p1, p2):
    """Compare two sklearn pipelines structurally (steps and names)."""
    if type(p1) != type(p2):
        return False
    if isinstance(p1, Pipeline):
        if len(p1.steps) != len(p2.steps):
            return False
        for (n1, s1), (n2, s2) in zip(p1.steps, p2.steps):
            if n1 != n2:
                return False
            if not compare_pipelines(s1, s2):
                return False
    elif isinstance(p1, ColumnTransformer):
        if len(p1.transformers) != len(p2.transformers):
            return False
        for (n1, t1, _), (n2, t2, _) in zip(p1.transformers, p2.transformers):
            if n1 != n2:
                return False
            if not compare_pipelines(t1, t2):
                return False
    return True


def test_all_pipelines():
    dataset_name = "cybersecurity"
    cfg = DATASETS[dataset_name]

    nominal_features = FEATURES[dataset_name]["nominal_features"]
    text_features = ["text"]

    dataset = load_dataset(cfg, use_cache=False)
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    numerical_features = [c for c in X_train.columns if c not in nominal_features]
    non_text_columns = nominal_features + numerical_features
    all_columns = text_features + non_text_columns

    print("=== PIPELINE STRUCTURE TEST ===\n")
    total = len(METHOD_KEYS)
    succeeded = 0

    for mk in METHOD_KEYS:
        print(f"🔹 Method: {mk}")
        try:
            pipeline = build_pipeline(
                method_key=mk,
                dataset_name=dataset_name,
                nominal_features=nominal_features,
                numerical_features=numerical_features,
                text_features=text_features,
                feature_extractor="dummy_extractor"
            )
            expected_pipeline = EXPECTED_PIPELINES.get(mk)
            if expected_pipeline and compare_pipelines(pipeline, expected_pipeline):
                print("✅ Pipeline is correct.")
                #print("Generated pipeline:")
                #pprint(pipeline)
                #print("Expected pipeline:")
                #pprint(expected_pipeline)
                succeeded += 1
            else:
                print("❌ Pipeline mismatch!")
                print("Generated pipeline:")
                pprint(pipeline)
                print("Expected pipeline:")
                pprint(expected_pipeline)
        except Exception as e:
            print(f"❌ Failed to build pipeline for {mk}: {e}")

        print("\n" + "-" * 100 + "\n")

    print(f"Summary: {succeeded} out of {total} pipelines passed.")


if __name__ == "__main__":
    print("-" * 100)
    print("Run pipeline tests...")
    print("-" * 100)
    test_all_pipelines()
