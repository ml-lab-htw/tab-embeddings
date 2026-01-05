import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.exp_context import ExpContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

def save_to_csv(data_dict: dict, ctx: ExpContext) -> None:
    """
    Saves experiment results (train/test metrics) to structured CSV files.
    """
    if data_dict is None:
        raise TypeError("Saving results as .csv failed. Result dict must not be None")
    if ctx is None:
        raise TypeError("Saving results as .csv failed. Experiment context must not be None")

    method_id = ctx.experiment_id
    # --- Prepare output directories ---
    base_results_dir = Path(ctx.cfg.globals["results_dir"])
    # todo: create results_datetime folder!
    train_dir = base_results_dir / "train" / ctx.dataset_name
    test_dir = base_results_dir / "test" / ctx.dataset_name

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare file paths ---
    train_file = train_dir / f"{method_id}_train.csv"
    test_file = test_dir / f"{method_id}_test.csv"

    # --- Create DataFrames for saving ---

    train_df = make_df(data_dict=data_dict, ctx=ctx, mode="train")
    test_df = make_df(data_dict=data_dict, ctx=ctx, mode="test")

    # --- Save to CSV ---
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"✅ Saved train/test results for '{method_id}'")
    logger.info(f"Train scores saved to → {train_file}")
    logger.info(f"Test scores saved to → {test_file}")


def make_df(data_dict: dict, ctx: ExpContext, mode: str = "train") -> pd.DataFrame:
    """
    Convert the metrics dict for a given mode ('train' or 'test') into a DataFrame.
    """
    logger.debug(f"(Data dict: {data_dict})")

    metrics = data_dict.get(f"{mode}_metrics", {})

    return pd.DataFrame([{
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ctx.dataset_name,
        "method": ctx.method_key,
        "concatenation": "conc 1" if ctx.flags.conc1 else "conc 2" if ctx.flags.conc2 else "conc 3" if ctx.flags.conc3 else "",
        "pca": "PCA" if ctx.flags.has_pca else "",
        "ml_method": "GBDT" if ctx.flags.is_gbdt else "LogReg" if ctx.flags.is_lr else "Undefined",
        "emb_method": "RTE" if ctx.flags.has_rte else ctx.embedding_key or "none",
        "BestParams": json.dumps(data_dict.get("best_params", {})),
        **{f"{mode}_{k}": v for k, v in metrics.items()}
    }])

