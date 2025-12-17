import pandas as pd
import numpy as np

from source.config.config import DATASETS, FEATURES
from source.helpers.feature_preparer import prepare_features


def prepare_data_for_experiment(method_key, dataset_name, data, text_embedding_dict=None, verbose=True):
    """
    Loads and inspects helpers for experiment.
    Returns X_train, X_test, y_train, y_test, and feature type lists.
    """
    cfg = DATASETS[dataset_name]

    # --- Get features and targets ---
    X_train, X_test = prepare_features(method_key, data, text_embedding_dict)
    y_train = data.get("y_train")
    y_test = data.get("y_test")

    def _quick_preview(name, df):
        print(f"\n{name}:")
        if isinstance(df, pd.DataFrame):
            print(f"Shape={df.shape}, Cols={list(df.columns)}")
            print(df.head(2))
        elif isinstance(df, (pd.Series, np.ndarray, list)):
            print(f"Len={len(df)}, First 2={df[:2]}")
        else:
            print(f"Type={type(df)}")

    _quick_preview("quick preview: X_train", X_train)
    _quick_preview("quick preview: X_test", X_test)
    _quick_preview("quick preview: y_train", y_train)
    _quick_preview("quick preview: y_test", y_test)

    if isinstance(X_train, pd.DataFrame):
        print("'prepare_data_for_experiment' - columns extraction: X_train is df")
        all_columns = X_train.columns.tolist()

        # Try to use definitions from config.FEATURES if available
        feature_cfg = FEATURES.get(dataset_name, {})
        predefined_text_cols = feature_cfg.get("text_features", [])
        predefined_nominal_cols = feature_cfg.get("nominal_features", [])

        # Use predefined lists if present, otherwise infer automatically
        if predefined_nominal_cols or predefined_text_cols:
            text_cols = [c for c in predefined_text_cols if c in all_columns]
            nominal_cols = [c for c in predefined_nominal_cols if c in all_columns]
            numerical_cols = [
                c for c in all_columns if c not in text_cols + nominal_cols
            ]
            print(f"[INFO] Using FEATURE definitions from config for '{dataset_name}'")
        else:
            print(f"[INFO] No FEATURE definitions found for '{dataset_name}', inferring automatically...")
            text_cols = [c for c in all_columns if "text" in c.lower()]
            nominal_cols = [
                c for c in all_columns if X_train[c].dtype == "object" and c not in text_cols
            ]
            numerical_cols = [c for c in all_columns if c not in text_cols + nominal_cols]
    else:
        # Fallback for embeddings or text-only lists
        print("'prepare_data_for_experiment' - columns extraction: X_train is NOT df")
        text_cols, nominal_cols, numerical_cols = [], [], []

    if verbose:
        print(f"\n[INFO] Dataset: {dataset_name}, Method: {method_key}")
        print(f"  Text cols: {text_cols}")
        print(f"  Nominal cols: {nominal_cols}")
        print(f"  Numerical cols: {numerical_cols}")

    return X_train, X_test, y_train, y_test, text_cols, nominal_cols, numerical_cols, cfg
