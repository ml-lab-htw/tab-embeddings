import pandas as pd
import os


def save_to_csv(data_dict: dict, llm=None):
    """
    Saves experiment results (train/test metrics) to structured CSV files.

    Args:
        data_dict (dict): Returned from run_experiment(), must include:
            - dataset
            - method
            - train_metrics
            - test_metrics
            - best_params
            :param data_dict:
            :param llm:
    """

    dataset_name = data_dict["dataset"]
    method_key = data_dict["method"]

    conc_method = ""

    # --- Extract info from method_key ---
    # ML method
    if "lr" in method_key:
        ml_method = "lr"
    elif "gbdt" in method_key:
        ml_method = "gbdt"
    else:
        ml_method = "unknown"

    # Embedding method
    if llm and "_te" in method_key:
        emb_method = llm
    elif "rte" in method_key:
        emb_method = "RTE"
    else:
        emb_method = "none"

    # Concatenation type
    if "conc1" in method_key:
        concatenation = "conc 1"
        conc_method = "conc_1"
    elif "conc2" in method_key:
        concatenation = "conc 2"
        conc_method = "conc_2"
    elif "conc3" in method_key:
        concatenation = "conc 3"
        conc_method = "conc_3"
    else:
        concatenation = ""

    # PCA
    if "pca" in method_key:
        pca = "PCA"
    else:
        pca = ""
    # --- Prepare output directories ---
    output_base = os.path.join("../../outputs", dataset_name)
    train_dir = os.path.join(output_base, "train")
    test_dir = os.path.join(output_base, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # --- Prepare file paths ---
    train_file = fr"{train_dir}\{dataset_name}_{emb_method}_{ml_method}_{pca}{conc_method}train.csv"
    test_file = fr"{test_dir}\{dataset_name}_{emb_method}_{ml_method}_{pca}{conc_method}test.csv"

    # --- Create DataFrames for saving ---
    train_df = pd.DataFrame([{
        "dataset": dataset_name,
        "method": method_key,
        "concatenation": concatenation,
        "pca": pca,
        "ml_method": ml_method,
        "emb_method": emb_method,
        "best_params": str(data_dict.get("best_params", {})),
        **{f"train_{k}": v for k, v in data_dict.get("train_metrics", {}).items()}
    }])

    test_df = pd.DataFrame([{
        "dataset": dataset_name,
        "method": method_key,
        "concatenation": concatenation,
        "pca": pca,
        "ml_method": ml_method,
        "emb_method": emb_method,
        "best_params": str(data_dict.get("best_params", {})),
        **{f"test_{k}": v for k, v in data_dict.get("test_metrics", {}).items()}
    }])

    # --- Save to CSV ---
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"✅ Saved train/test results for '{method_key}'")
    print(f"  → {train_file}")
    print(f"  → {test_file}")
