from pathlib import Path
from typing import Dict, Any

import yaml


class ConfigManager:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @classmethod
    def load_yaml(cls, path: str | Path) -> "ConfigManager":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Top-level YAML config must be a mapping")

        return cls(config)

    def _require_section(self, key: str) -> Any:
        """
        Require that a top-level key exists in the config and return its value.
        """
        if key not in self._config:
            raise KeyError(f"Missing required config section: {key}"
                           # todo: here also provide paths
                           f"Make sure the section {key} exists in the config file as well as implemented in config_manager.")
        return self._config[key]

    @property
    def test_mode(self) -> bool:
        return bool(self._require_section("TEST_MODE"))

    @property
    def test_samples(self) -> int:
        return int(self._require_section("TEST_SAMPLES"))

    @property
    def globals(self) -> Dict[str, Any]:
        return self._require_section("GLOBAL")

    @property
    def datasets(self) -> Dict[str, Any]:
        return self._require_section("DATASETS")

    @property
    def features(self) -> Dict[str, Any]:
        return self._require_section("FEATURES")

    @property
    def model_cfg(self) -> Dict[str, Any]:
        return self._require_section("MODEL_CONFIGS")

    @property
    def hyperparams_cfg(self) -> Dict[str, Any]:
        return self._require_section("HYPERPARAMETERS")

    @property
    def data_prep(self) -> Dict[str, Any]:
        return self._require_section("DATA_PREP_CONFIG")

    @property
    def experiments(self, test_mode=False) -> Dict[str, Any]:
        if test_mode: # todo: we have a self._test_mode
            return self._require_section("EXPERIMENTS_TEST")
        return self._require_section("EXPERIMENTS")

    @property
    def llms(self, test_mode=False) -> Dict[str, Any]:
        return self._require_section("LLMS")
