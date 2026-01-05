from abc import ABC, abstractmethod

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.llm_related.embedding_aggregator import EmbeddingAggregator
from src.exp_context import ExpContext
from src.helpers.pipeline_helpers import build_tabular_transformer, select_classifier, build_feature_union, \
    build_text_pipeline_steps, build_raw_branch


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

        elif ctx.flags.is_lr:
            return Pipeline([
            ("transformer", build_tabular_transformer(ctx=ctx, include_text=False, scale=True)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])

        else:
            raise NotImplementedError(
                # todo: print the concrete flag
                f"Pipeline not implemented for {ctx.flags}"
            )


class RTEPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("transformer", build_tabular_transformer(ctx=ctx, include_text=False, scale=False)),
            ("embedding", RandomTreesEmbedding(
                random_state=ctx.cfg.globals["random_state"]
            )),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])


class TextPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        steps = build_text_pipeline_steps(ctx=ctx)
        steps.append(("classifier", select_classifier(ctx, ctx.cfg)))
        return Pipeline(steps)


class ConcatTextPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        if ctx.flags.is_gbdt:
            return Pipeline([
                ("transformer", build_raw_branch(ctx=ctx)),
                ("classifier", select_classifier(ctx, ctx.cfg))
            ])

        elif ctx.flags.is_lr:
            return Pipeline([
            ("transformer", build_tabular_transformer(ctx=ctx, include_text=True, scale=True)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])

        else:
            raise NotImplementedError(
                # todo: print the concrete flag
                f"Pipeline not implemented for {ctx.flags}"
            )


class ConcatRTEPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("feature_combiner", build_feature_union(ctx)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])
