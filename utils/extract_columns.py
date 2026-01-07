# src/data_preparation/csv_feature_splitter.py

from pathlib import Path
import pandas as pd

from config.config_manager import ConfigManager


class CSVFeatureSplitter:
    """
    Splits a dataset CSV into nominal and numerical feature CSVs
    using config-defined feature groups.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    def split_and_save(
        self,
        dataset_name: str,
        input_csv: str | Path,
        output_dir: str | Path,
        *,
        index: bool = False,
    ) -> None:
        input_csv = Path(input_csv)
        output_dir = Path(output_dir)

        df = self._load(input_csv)

        nominal_cols = self._config.features[dataset_name]["nominal_features"]

        self._validate_columns(df, nominal_cols)

        numerical_cols = self._compute_numerical_columns(
            df.columns, nominal_cols
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save(df[nominal_cols], output_dir / f"X_{dataset_name}_nom.csv", index)
        self._save(df[numerical_cols], output_dir / f"X_{dataset_name}_metrics.csv", index)

    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

    @staticmethod
    def _compute_numerical_columns(
        all_columns,
        nominal_columns
    ) -> list[str]:
        excluded = set(nominal_columns)
        return [col for col in all_columns if col not in excluded]

    @staticmethod
    def _save(df: pd.DataFrame, path: Path, index: bool) -> None:
        df.to_csv(path, index=index)
