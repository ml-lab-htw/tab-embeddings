import logging

import pandas as pd

from config.config_manager import ConfigManager
from dataclasses import dataclass
from typing import Any


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentFlags:
    is_lr: bool
    is_gbdt: bool
    has_text: bool
    has_rte: bool
    is_concat: bool
    has_pca: bool
    conc1: bool
    conc2: bool
    conc3: bool


# todo: init the paths to the files depending on the method key here? And build a delegate method in data_prep?
class ExpContext:
    def __init__(self,
                 method_key: str,
                 dataset_name: str,
                 cfg: ConfigManager,
                 embedding_key: str | None = None,
                 *,
                 validate: bool = True,
                 feature_extractor: Any | None = None,
                 ):
        self.method_key = method_key
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.embedding_key = embedding_key

        # --------------------------------------------------
        # Flags derived from method_key
        # --------------------------------------------------
        self.flags = ExperimentFlags(
            is_lr=method_key.startswith("lr"),
            is_gbdt=method_key.startswith("gbdt"),
            has_text="_te" in method_key,
            has_rte="_rte" in method_key,
            is_concat="_conc" in method_key,
            has_pca="_pca" in method_key,
            conc1="_conc1" in method_key,
            conc2="_conc2" in method_key,
            conc3="_conc3" in method_key,
        )

        # --------------------------------------------------
        # Feature config (static)
        # --------------------------------------------------
        feat_cfg = cfg.features[dataset_name]
        self.nominal_features = feat_cfg.get("nominal_features", [])
        # self.nominal_features = self._nominal_features(feat_cfg=feat_cfg)
        self.text_features = feat_cfg.get("text_features", [])

        # --------------------------------------------------
        # Feature config (runtime)
        # --------------------------------------------------
        self.numerical_features: list[str] = []
        self.nominal_indices: list[int] = []
        self.non_text_columns: list[str] = []

        # --------------------------------------------------
        # Data config (runtime)
        # --------------------------------------------------
        self.features: list[str] = []
        self.label: list[int] = [] # todo: might be different if we have different tasks

        # --------------------------------------------------
        # Text Embeddings
        # --------------------------------------------------

        self.feature_extractor = feature_extractor

        if validate:
            self._validate()

    def _validate(self):
        if self.flags.has_text:
            if not self.embedding_key:
                raise ValueError(
                    f"Experiment '{self.method_key}' requires text embeddings "
                    f"but no embedding_key was provided."
                )
            if self.feature_extractor is None:
                raise ValueError(
                    f"Experiment '{self.method_key}' requires a feature_extractor "
                    f"but none was provided."
                )
        if self.flags.has_text and "text" not in self.text_features:
            raise ValueError(
                f"Experiment '{self.method_key}' requires text features, "
                f"but 'text' is not listed in text_features "
                f"for dataset '{self.dataset_name}'."
            )

    def update_numerical_features(self, X):
        #if self.nominal_features:
        self.numerical_features = [
            c for c in X.columns
            if c not in self.nominal_features
            and c not in self.text_features
        ]

    def update_nominal_indices(self, all_columns: list[str]):
        if not self.nominal_features:
            self.nominal_indices = []
            return

        self.nominal_indices = [
                i for i, c in enumerate(all_columns)
                if c in self.nominal_features
            ]

    def update_non_text_columns(self, X_tabular: pd.DataFrame):
        self.non_text_columns = [
            col for col in X_tabular.columns
            if col not in self.text_features
        ]

    @property
    def categorical_features_for_classifier(self):
        """
        Implemented to pass the cat features to HIstGRadientBoostingClassifier
        in a correct format, depending on the method key.
        """
        if not self.flags.is_gbdt:
            return None

        if self.flags.has_text and not self.flags.is_concat:
            # gbdt_te, gbdt_te_pca
            return None

        if self.flags.has_rte and not self.flags.is_concat:
            # gbdt_rte
            return None

        if self.flags.is_concat:
            # concatenated → numeric matrix → indices
            logging.debug(f"CTX returnes nominal indices for gbdt concat: {self.nominal_indices}")
            return self.nominal_indices

        # plain gbdt
        logging.debug(f"CTX returnes nominal features for plain gbdt: {self.nominal_indices}")
        return self.nominal_features

    @property
    def requires_categorical_indices(self) -> bool:
        return (
            self.flags.is_gbdt and
            self.flags.is_concat and
            (self.flags.has_text or self.flags.has_rte)
        )

    @property
    def experiment_id(self) -> str:
        """Unique identifier for this experiment."""
        if self.embedding_key:
            return f"{self.dataset_name}_{self.method_key}_{self.embedding_key}"
        return f"{self.dataset_name}_{self.method_key}"
