import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ====================
# Main loader function
# ====================

_CACHE_KEYS = [
    "X_train", "X_test", "y_train", "y_test",
    "X_metr_train", "X_metr_test",
    "summaries_train", "summaries_test",
    "nom_summaries_train", "nom_summaries_test",
]


def _preview_data(name, obj):
    """Print concise info: shape, columns, and head(2) for DataFrames."""
    print(f"\n🔹 {name}:")
    if isinstance(obj, pd.DataFrame):
        print(f"Shape: {obj.shape}")
        print(f"Columns: {list(obj.columns)}")
        print(f"Head(2):\n{obj.head(2)}")
    elif isinstance(obj, (list, np.ndarray)):
        print(f"Type: {type(obj)}, Length: {len(obj)}")
        preview = obj[:2] if len(obj) > 2 else obj
        print(f"First 2 items: {preview}")
    else:
        print(f"Type: {type(obj)} — not previewed")


def load_dataset(cfg, test_mode=False, cache_root="./cache", use_cache=True):
    """
    Loads a dataset according to config, with optional caching.
    Returns a dict with standardized keys.
    """
    # base = cfg["base_dir"]
    # dataset_name = os.path.basename(os.path.normpath(base))
    # cache_dir = os.path.join(cache_root, dataset_name)
    # os.makedirs(cache_dir, exist_ok=True)

    # --- if cache exists, load it ---
    # if use_cache and _is_cache_complete(cache_dir):
    #    print(f"Loading cached dataset from {cache_dir}")
    #    return {k: joblib.load(os.path.join(cache_dir, f"{k}.pkl")) for k in _CACHE_KEYS}

    # --- otherwise, load from scratch ---
    data = _load_dataset_fresh(cfg, test_mode=test_mode)
    print("\n✅ Dataset loaded successfully. Preview of contents:")
    for k, v in data.items():
        _preview_data(k, v)
    # --- save to cache ---
    #if use_cache:
    #     for k, v in data.items():
    #        joblib.dump(v, os.path.join(cache_dir, f"{k}.pkl"))
    #    print(f"Saved dataset cache to {cache_dir}")

    return data


# ====================
# Internal helper: fresh load
# ====================

def _load_dataset_fresh(cfg, test_mode=False):
    base = cfg["base_dir"]

    # --- multi_file (needs splitting) ---
    if cfg["type"] == "multi_file":
        files = cfg["files"]

        X = load_features(os.path.join(base, files["X"]), test_mode=test_mode)
        y = load_labels(os.path.join(base, files["y"]), test_mode=test_mode)

        X_metr = load_features(os.path.join(base, files["X_metr"]), test_mode=test_mode)
        summaries = load_summaries(os.path.join(base, files["summaries"]), test_mode=test_mode)
        nom_summaries = load_summaries(os.path.join(base, files["nom_summaries"]), test_mode=test_mode)

        # consistent split
        split_params = cfg.get("split_params", {})
        stratify_y = y if split_params.get("stratify", False) else None

        X_train, X_test, y_train, y_test, idx_train, idx_test = _split_with_indices(
            X, y,
            test_size=split_params.get("test_size", 0.2),
            random_state=split_params.get("random_state", 42),
            stratify=stratify_y
        )

        data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_metr_train": X_metr.iloc[idx_train].reset_index(drop=True),
            "X_metr_test": X_metr.iloc[idx_test].reset_index(drop=True),
            "summaries_train": [summaries[i] for i in idx_train],
            "summaries_test": [summaries[i] for i in idx_test],
            "nom_summaries_train": [nom_summaries[i] for i in idx_train],
            "nom_summaries_test": [nom_summaries[i] for i in idx_test],
        }

    # --- split (already divided) ---
    elif cfg["type"] == "split":
        train = cfg["train_files"]
        test = cfg["test_files"]

        data = {
            "X_train": load_features(os.path.join(base, train["X"]), test_mode=test_mode),
            "X_test": load_features(os.path.join(base, test["X"]), test_mode=test_mode),
            "y_train": load_labels(os.path.join(base, train["y"]), test_mode=test_mode),
            "y_test": load_labels(os.path.join(base, test["y"]), test_mode=test_mode),
            "X_metr_train": load_features(os.path.join(base, train["X_metr"]), test_mode=test_mode),
            "X_metr_test": load_features(os.path.join(base, test["X_metr"]), test_mode=test_mode),
            "summaries_train": load_summaries(os.path.join(base, train["summaries"]), test_mode=test_mode),
            "summaries_test": load_summaries(os.path.join(base, test["summaries"]), test_mode=test_mode),
            "nom_summaries_train": load_summaries(os.path.join(base, train["nom_summaries"]), test_mode=test_mode),
            "nom_summaries_test": load_summaries(os.path.join(base, test["nom_summaries"]), test_mode=test_mode),
        }

    else:
        raise ValueError(f"Unknown dataset type: {cfg['type']}")

    return data


# ====================
# Helper functions
# ====================
def load_features(file_path, delimiter=",", test_mode=False):
    if test_mode:
        data = pd.read_csv(file_path, delimiter=delimiter, nrows=220)
    else:
        data = pd.read_csv(file_path, delimiter=delimiter)
    print(f"Loaded features from {file_path} with shape {data.shape}")
    return data


def load_labels(file_path, delimiter=",", test_mode=False):
    if test_mode:
        data = pd.read_csv(file_path, delimiter=delimiter, nrows=220)
    else:
        data = pd.read_csv(file_path, delimiter=delimiter)

    y = data.values.ravel()
    y = pd.Series(y)

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding values: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.to_numpy()

    return y


def load_summaries(file_path, test_mode=False):
    if not os.path.exists(file_path):
        print(f"Summaries file not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        summaries_list = [line.strip() for line in f.readlines()]
    if test_mode:
        summaries_list = summaries_list[:220]
    print(f"Loaded {len(summaries_list)} summaries from {file_path}")
    return summaries_list


# ====================
# Internal helpers
# ====================

def _split_with_indices(X, y, **kwargs):
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, **kwargs)
    return (
        X.iloc[idx_train].reset_index(drop=True),
        X.iloc[idx_test].reset_index(drop=True),
        y[idx_train],
        y[idx_test],
        idx_train,
        idx_test,
    )


def _is_cache_complete(cache_dir):
    return all(os.path.exists(os.path.join(cache_dir, f"{k}.pkl")) for k in _CACHE_KEYS)
