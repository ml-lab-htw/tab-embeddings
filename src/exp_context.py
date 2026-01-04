from config.config_manager import ConfigManager
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
        # self.nominal_features = feat_cfg.get("nominal_features", [])
        self.nominal_features = self._nominal_features(feat_cfg=feat_cfg)
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

    def _nominal_features(self, feat_cfg):
        if (
            self.flags.is_gbdt
            and (self.flags.has_text or self.flags.has_rte)
            and not self.flags.is_concat
        ):
            return []
        return feat_cfg.get("nominal_features", []) or []

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
