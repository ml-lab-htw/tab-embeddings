# src/data_preparation/csv_extractors.py

from pathlib import Path
import pandas as pd


class CSVColumnExtractor:
    """
    Utility class for extracting and saving CSV column subsets.
    """

    @classmethod
    def extract_and_save(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        columns: list[str],
        *,
        index: bool = False
    ) -> None:
        df = cls._load(input_path)
        df = cls._select_columns(df, columns)
        cls._save(df, output_path, index=index)

    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        return df[columns]

    @staticmethod
    def _save(df: pd.DataFrame, path: str | Path, index: bool) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)
