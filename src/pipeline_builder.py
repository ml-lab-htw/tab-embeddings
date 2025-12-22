from abc import ABC, abstractmethod

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.exp_context import ExpContext
from src.helpers import build_tabular_transformer, select_classifier, build_feature_union


class PipelineStrategy(ABC):
    @abstractmethod
    def build(self, ctx: ExpContext) -> Pipeline:
        raise RuntimeError(
            f"{self.__class__.__name__}.build() must be implemented"
        )


class BasicPipeline(PipelineStrategy):
    """
    Baseline pipeline builder.
    Blabb blubb.
    """
    def build(self, ctx: ExpContext) -> Pipeline:
        if ctx.flags.is_gbdt:
            return Pipeline([
                ("classifier", select_classifier(ctx, ctx.cfg))
            ])

        return Pipeline([
            ("transformer", build_tabular_transformer(ctx)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])


class RTEPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("transformer", build_tabular_transformer(ctx=ctx)),
            ("embedding", RandomTreesEmbedding(
                random_state=ctx.cfg.globals["random_state"]
            )),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])


class TextPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        steps = [
            ("aggregator", EmbeddingAggregator(
                feature_extractor=ctx.feature_extractor
            )),
        ]

        if ctx.flags.has_pca:
            steps.extend([
                ("numerical_scaler", StandardScaler()),
                ("pca", PCA(n_components=ctx.pca_components)),
            ])
        else:
            steps.append(("numerical_scaler", MinMaxScaler()))

        steps.append(("classifier", select_classifier(ctx, ctx.cfg)))
        return Pipeline(steps)



class ConcatTextPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("transformer", build_tabular_transformer(ctx, include_text=True)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])


class ConcatRTEPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("feature_combiner", build_feature_union(ctx)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])
