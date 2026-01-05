import os
import logging

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.exp_context import ExpContext


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

class DataPreparer:
    """
    Delivers ready to use train and test data.
    """
    def __init__(self, ctx: ExpContext):
        self.ctx = ctx
        self.resolver = DataResolver(ctx)
        #self.X = None
        #self.y = None

    def _nrows(self):
        if self.ctx.cfg.test_mode:
            return self.ctx.cfg.test_samples
        return None

    def _validate_lengths(self, y, X = None, summaries = None):
        # todo: should summaries be mentioned here if before X=summaries?
        logger.debug(
            f"Validating lengths: X={len(X) if X is not None else 'None'}, y={len(y)}, summaries={len(summaries) if summaries else 'None'}")

        if X is not None and len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )
        if summaries is not None and len(summaries) != len(y):
            raise ValueError(
                f"Summaries length mismatch: {len(summaries)} vs {len(y)}"
            )

    def prepare(self):
        paths = self.resolver.resolve()

        y = DataLoader.load_labels(
            path=paths["y"],
            nrows=self._nrows()
        )
        X = None
        summaries = None

        if self.ctx.flags.has_text and not self.ctx.flags.is_concat:
            summaries = DataLoader.load_summaries(path=paths["summaries"], nrows=self._nrows())
            _, y, summaries = self._drop_unlabeled(y=y, summaries=summaries)
            self._validate_lengths(X=X, y=y, summaries=summaries)
            X = pd.DataFrame({"text": summaries})

        else:
            X = DataLoader.load_features(
                path=paths["X"],
                nrows=self._nrows()
            )
            if self.ctx.flags.has_text:
                summaries = DataLoader.load_summaries(path=paths["summaries"], nrows=self._nrows())

            X, y, summaries = self._drop_unlabeled(X=X, y=y, summaries=summaries)
            self._validate_lengths(X=X, summaries=summaries, y=y)

            if summaries is not None:
                X = self._concatenate(X, summaries)

        self.ctx.update_numerical_features(X)
        self.ctx.update_nominal_indices(list(X.columns))

        return self._split(X, y)

    def _concatenate(self, X, summaries):
        logger.debug("Concatenating text summaries with features")
        X = X.copy()
        X["text"] = summaries
        return X

    def _drop_unlabeled(self, y, X=None, summaries=None):
        mask = ~pd.isna(y)

        y = y.loc[mask].reset_index(drop=True)

        if X is not None:
            X = X.loc[mask].reset_index(drop=True)

        if summaries is not None:
            summaries = [s for i, s in enumerate(summaries) if mask.iloc[i]]

        num_dropped = (~mask).sum()
        if num_dropped > 0:
            logger.info(f"Dropped {num_dropped} samples with missing labels.")

        return X, y, summaries

    def _split(self, X, y):
        logger.info(
            f"Splitting data: {len(X)} samples, test size={self.ctx.cfg.globals['test_size']}"
        )
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
    def load_summaries(path, nrows) -> list[str]:
        summaries = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}.")
        with open(path, "r") as file:
            if nrows is None:
                summaries = [line.strip() for line in file.readlines()]
            else:
                for _, line in zip(range(nrows), file.readlines()):
                    summaries.append(line.strip())
        if not summaries:
            raise ValueError(f"Summaries file is empty: {path}")
        logger.info(f"Loaded text summaries from {path} with length {len(summaries)}")
        return summaries

    @staticmethod
    def load_features(path, nrows, delimiter=",") -> pd.DataFrame:
        X = pd.read_csv(path, delimiter=delimiter, nrows=nrows)
        if X.empty:
            raise ValueError(f"Features file is empty: {path}")
        logger.info(f"Loaded tabular features from {path} with shape {X.shape}")
        return X

    @staticmethod
    def load_labels(path, nrows, delimiter=",") -> pd.Series:
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

        y = data.iloc[:, 0]

        if not np.issubdtype(y.dtype, np.number):
            y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

        logger.info(f"Loaded labels from {path} with shape {y.shape}")
        return y.reset_index(drop=True)


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

        logger.info(f"Resolved paths: {resolved}")

        return resolved

    def _resolve_X(self) -> Path:
        return self.base_path / self.dataset_cfg["X"]

    def _resolve_y(self) -> Path:
        return self.base_path / self.dataset_cfg["y"]

    def _resolve_summaries(self) -> Path:
        """
        Handles:
        - pure text experiments
        - concatenated experiments
        - different concat modes (conc1/2/3)
        """
        if self.ctx.flags.conc1 or self.ctx.flags.conc2:
            return self.base_path / self.dataset_cfg["summaries"]

        elif self.ctx.flags.conc3:
            return self.base_path / self.dataset_cfg["nom_summaries"]

        elif self.ctx.flags.has_text:
            return self.base_path / self.dataset_cfg["summaries"]

        raise ValueError(
            f"Unknown concatenation mode for method '{self.ctx.method_key}'"
        )
