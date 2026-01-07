from enum import Enum
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


class Classification(Enum):
    VERY_LOW = "very low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very high"


class TabularSummaryGenerator:
    """
    Generates natural-language summaries from tabular datasets.
    """

    def __init__(
        self,
        *,
        categorical_values: Optional[Dict[str, Dict]] = None,
        classify_numeric: bool = False,
        subject_name: str = "sample",
    ) -> None:
        self.categorical_values = categorical_values or {}
        self.classify_numeric = classify_numeric
        self.subject_name = subject_name

    # -------------------------
    # Public API
    # -------------------------

    def generate(
        self,
        input_csv: str | Path,
        output_file: str | Path | None = None,
    ) -> list[str]:
        df = self._load(input_csv)

        numeric_cols = self._get_numeric_columns(df)
        stats = self._compute_numeric_stats(df, numeric_cols) if self.classify_numeric else {}

        summaries = [
            self._summarize_row(idx + 1, row, df.columns, numeric_cols, stats)
            for idx, row in df.iterrows()
        ]

        if output_file is not None:
            self._write(summaries, output_file)

        return summaries

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    @staticmethod
    def _compute_numeric_stats(df: pd.DataFrame, numeric_cols: list[str]) -> Dict[str, Dict]:
        return {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std(),
            }
            for col in numeric_cols
        }

    def _summarize_row(
        self,
        prefix: str | None,
        row_number: int,
        row: pd.Series,
        all_columns: list[str],
        numeric_cols: list[str],
        stats: Dict[str, Dict],
    ) -> str:
        prefix = f"{prefix}. " if prefix else ""
        begin = f"{prefix}The following is the data for {self.subject_name} number {row_number}. "
        details = []

        for col in all_columns:
            value = row[col]

            if self._skip_value(value):
                continue

            rendered = self._render_value(col, value, numeric_cols, stats)
            details.append(f"{col} is {rendered}")

        return begin + "; ".join(details) + "."

    @staticmethod
    def _skip_value(value) -> bool:
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def _render_value(
        self,
        column: str,
        value,
        numeric_cols: list[str],
        stats: Dict[str, Dict],
    ) -> str:
        if column in self.categorical_values:
            return self._render_categorical(column, value)

        if self.classify_numeric and column in numeric_cols:
            return self._classify_number(value, stats[column])

        return str(value)

    def _render_categorical(self, column: str, value) -> str:
        mapping = self.categorical_values[column]
        try:
            return mapping.get(int(value), str(value))
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _classify_number(value, stat: Dict) -> str:
        mean = stat["mean"]
        std = stat["std"]

        if value < mean - 2 * std:
            return Classification.VERY_LOW.value
        if value < mean - std:
            return Classification.LOW.value
        if value < mean + std:
            return Classification.NORMAL.value
        if value < mean + 2 * std:
            return Classification.HIGH.value
        return Classification.VERY_HIGH.value

    @staticmethod
    def _write(summaries: list[str], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for s in summaries:
                f.write(s + "\n")
