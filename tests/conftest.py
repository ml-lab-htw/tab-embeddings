import pytest
from config.config_manager import ConfigManager
from src.exp_context import ExpContext
from src.llm_related.llm_registry import FeatureExtractorRegistry


@pytest.fixture(scope="session")
def cfg():
    return ConfigManager.load_yaml("config/config.yaml")


@pytest.fixture
def ctx_factory(cfg):
    def _factory(method_key: str, dataset_name: str = "cybersecurity"):
        return ExpContext(
            method_key=method_key,
            dataset_name=dataset_name,
            cfg=cfg
        )
    return _factory


@pytest.fixture(scope="session", autouse=True)
def register_dummy_feature_extractor():
    """
    Register a lightweight dummy feature extractor for pipeline tests.
    """
    FeatureExtractorRegistry.register(
        "DUMMY",
        lambda: lambda texts: [[0.0] * 10 for _ in texts]  # fake embeddings
    )
