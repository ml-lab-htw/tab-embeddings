from config.config_manager import ConfigManager
from src.llm_related.llm_registry import FeatureExtractorRegistry
from dataclasses import dataclass
from typing import Any

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


# todo (for GBDT Classifier): nom_feat = none is _te and not conc
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
        self.text_features = feat_cfg.get("text_features", [])

        # --------------------------------------------------
        # Feature config (runtime)
        # --------------------------------------------------
        self.numerical_features: list[str] = []
        self.nominal_indices: list[int] = []

        # --------------------------------------------------
        # Data config (runtime)
        # --------------------------------------------------
        self.features: list[str] = []
        self.label: list[int] = [] # todo: might be different if we have different tasks

        # --------------------------------------------------
        # Text Embeddings
        # --------------------------------------------------
        if validate:
            self._validate()

        self.feature_extractor = feature_extractor

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

    def update_numerical_features(self, X):
        self.numerical_features = [
            c for c in X.columns
            if c not in self.nominal_features and c not in self.text_features
        ]

    def update_nominal_indices(self, all_columns: list[str]):
        self.nominal_indices = [
            i for i, c in enumerate(all_columns) if c in self.nominal_features
        ]

    '''
    def assign_correct_files(self):
        if self.flags.has_text:
            if self.flags.is_concat:
                if self.flags.conc1:
                    pass
                elif self.flags.conc2:
                    pass
                elif self.flags.conc3:
                    pass
            else:
                # just summaries
                pass
        else:
            # just X.csv
            pass

    def check_if_file_exists(self):
        """
        Before assigning correct files, check if file exists.
        """
        pass
    '''
