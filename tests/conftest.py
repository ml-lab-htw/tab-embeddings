import pytest
from config.config_manager import ConfigManager
from src.exp_context import ExpContext


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
