import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.exp_context import ExpContext


class DataPreparer:
    def __init__(self, ctx: ExpContext):
        self.ctx = ctx
        self.resolver = DataResolver(ctx)
        self.X = None
        self.y = None

    def _validate_lengths(self, X, y, summaries = None):
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )
        if summaries is not None and len(summaries) != len(X):
            raise ValueError(
                f"Summaries length mismatch: {len(summaries)} vs {len(X)}"
            )

    def prepare(self):
        paths = self.resolver.resolve()

        X = DataLoader.load_features(
            path=paths["X"],
            nrows=self._nrows()
        )

        y = DataLoader.load_labels(
            path=paths["y"],
            nrows=self._nrows()
        )
        summaries = None
        if self.ctx.flags.has_text:
            summaries = DataLoader.load_summaries(paths["summaries"])

        X, y, summaries = self._drop_unlabeled(X, y, summaries)

        self._validate_lengths(X, y, summaries)

        if summaries is not None:
            X = self._concatenate(X, summaries)

        self.ctx.update_numerical_features(X)
        self.ctx.update_nominal_indices(list(X.columns))

        return self._split(X, y)

    def _concatenate(self, X, summaries):
        X = X.copy()
        X["text"] = summaries
        return X

    def _drop_unlabeled(self, X, y, summaries=None):
        mask = ~pd.isna(y)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask]
        if summaries is not None:
            summaries = [s for i, s in enumerate(summaries) if mask.iloc[i]]
        return X, y, summaries

    def _split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.ctx.cfg.globals["test_size"],
            stratify=y,
            random_state=self.ctx.cfg.globals["random_state"]
        )


class DataLoader:
    """
    loads .csv files
    loads features (X.csv) as dataframe
    loads labels (y.csv) as np.array
    """
    @staticmethod
    def load_summaries(path):
        summaries_list = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} has not been was found.")
        with open(path, "r") as file:
            summaries_list = [line.strip() for line in file.readlines()]
        return summaries_list

    @staticmethod
    def load_features(path, nrows, delimiter=","):
        X = pd.read_csv(path, delimiter=delimiter, nrows=nrows)
        print(f"Loaded tabular features from {path} with shape {X.shape}")
        return X

    @staticmethod
    def load_labels(path, nrows=None, delimiter=","):
        """
        Load y.csv labels as np.array
        Encode categories if necessary
        """
        data = pd.read_csv(path, delimiter=delimiter, nrows=nrows)
        if data.empty:
            raise ValueError(f"Labels file is empty: {path}")

        if data.shape[1] != 1:
            raise ValueError(
                f"Labels file must contain exactly one column, "
                f"found {data.shape[1]} columns in {path}"
            )
        y = data.values.ravel()
        y = pd.Series(y)
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy()
        print(f"Loaded labels from {path} with shape {len(y)}")
        return y


class DataResolver:
    def __init__(self, ctx: ExpContext):
        self.ctx = ctx
        self.dataset_cfg = ctx.cfg.datasets[ctx.dataset_name]

        self.base_path = Path(self.dataset_cfg["path"])

    def resolve(self) -> Dict[str, Any]:
        """
        Returns a dict with paths needed for the experiment.
        Keys may include:
        - X_path
        - y_path
        - summaries_path
        """
        resolved = {
            "X": self._resolve_X(),
            "y": self._resolve_y(),
        }

        if self.ctx.flags.has_text:
            resolved["summaries"] = self._resolve_summaries()

        return resolved

    def _resolve_X(self) -> Path:
        return self.base_path / self.dataset_cfg["X"]

    def _resolve_y(self) -> Path:
        return self.base_path / self.dataset_cfg["y"]

    def _resolve_summaries(self) -> Path | list[Path]:
        """
        Handles:
        - pure text experiments
        - concatenated experiments
        - different concat modes (conc1/2/3)
        """
        if self.ctx.flags.has_text or self.ctx.flags.conc1 or self.ctx.flags.conc2:
            return self.base_path / self.dataset_cfg["summaries"]

        if self.ctx.flags.conc3:
            return self.base_path / self.dataset_cfg["nom_summaries"]

        raise ValueError(
            f"Unknown concatenation mode for method '{self.ctx.method_key}'"
        )
