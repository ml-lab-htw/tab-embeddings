from typing import Any, Dict, Callable

#from src.helpers.feature_extractor_creators import create_gen_feature_extractor, create_gte_feature_extractor


class FeatureExtractorRegistry:
    _REGISTRY: Dict[str, Callable[[], Any]] = {}

    @classmethod
    def register(cls, key: str, factory: Callable[[], Any]):
        cls._REGISTRY[key] = factory

    @classmethod
    def create(cls, key:str) -> Any:
        if key not in cls._REGISTRY:
            raise KeyError(
                f"Unknown LLM key '{key}'."
                f"Available keys are {list(cls._REGISTRY.keys())}"
            )
        return cls._REGISTRY[key]()


"""FeatureExtractorRegistry.register(
    "E5_SMALL_V2",
    lambda: create_gen_feature_extractor("intfloat/e5-small-v2")
)

FeatureExtractorRegistry.register(
    "E5_BASE_V2",
    lambda: create_gen_feature_extractor("intfloat/e5-base-v2")
)

FeatureExtractorRegistry.register(
    "E5_LARGE_V2",
    lambda: create_gen_feature_extractor("intfloat/e5-large-v2")
)

FeatureExtractorRegistry.register(
    "BGE_SMALL_EN_V1_5",
    lambda: create_gen_feature_extractor("BAAI/bge-small-en-v1.5")
)

FeatureExtractorRegistry.register(
    "BGE_BASE_EN_V1_5",
    lambda: create_gen_feature_extractor("BAAI/bge-base-en-v1.5")
)

FeatureExtractorRegistry.register(
    "BGE_LARGE_EN_V1_5",
    lambda: create_gen_feature_extractor("BAAI/bge-large-en-v1.5")
)

FeatureExtractorRegistry.register(
    "GIST_SMALL_EMBEDDING_V0",
    lambda: create_gen_feature_extractor("avsolatorio/GIST-small-Embedding-v0")
)

FeatureExtractorRegistry.register(
    "GIST_EMBEDDING_V0",
    lambda: create_gen_feature_extractor("avsolatorio/GIST-Embedding-v0")
)

FeatureExtractorRegistry.register(
    "GIST_LARGE_EMBEDDING_V0",
    lambda: create_gen_feature_extractor("avsolatorio/GIST-large-Embedding-v0")
)

FeatureExtractorRegistry.register(
    "GTE_SMALL",
    lambda: create_gen_feature_extractor("thenlper/gte-small")
)

FeatureExtractorRegistry.register(
    "GTE_BASE",
    lambda: create_gen_feature_extractor("thenlper/gte-base")
)

FeatureExtractorRegistry.register(
    "GTE_BASE_EN_V1_5",
    lambda: create_gte_feature_extractor("Alibaba-NLP/gte-base-en-v1.5")
)

FeatureExtractorRegistry.register(
    "GTE_LARGE",
    lambda: create_gen_feature_extractor("thenlper/gte-large")
)

FeatureExtractorRegistry.register(
    "STELLA_EN_400M_V5",
    lambda: create_gen_feature_extractor("dunzhang/stella_en_400M_v5")
)

FeatureExtractorRegistry.register(
    "EMBER_V1",
    lambda: create_gen_feature_extractor('llmrails/ember-v1')
)

FeatureExtractorRegistry.register(
    "ALL_MINILM_L6_V2",
    lambda: create_gen_feature_extractor('sentence-transformers/all-MiniLM-L6-v2')
)"""
