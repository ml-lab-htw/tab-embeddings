from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier
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
                random_state=ctx.cfg.globals["random_state"],
                sparse_output=False
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
            # todo: there is a problem here
            return Pipeline([
                #("transformer", build_raw_branch(ctx=ctx)),

                ("transformer", ColumnTransformer([
                ("numerical", "passthrough", ctx.non_text_columns),
                ("text", Pipeline([
                    ("embedding_aggregator", EmbeddingAggregator(
                        feature_extractor=ctx.feature_extractor,
                        is_sentence_transformer=False)),
                    ("numerical_scaler", MinMaxScaler())
                ]), ctx.text_features)
            ])),
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
                f"Pipeline not yet implemented for this ml model."
            )


class ConcatRTEPipeline(PipelineStrategy):
    def build(self, ctx: ExpContext) -> Pipeline:
        return Pipeline([
            ("feature_combiner", build_feature_union(ctx)),
            ("classifier", select_classifier(ctx, ctx.cfg)),
        ])


from sklearn.base import BaseEstimator, TransformerMixin


class PipelineDebugger(BaseEstimator, TransformerMixin):
    def __init__(self, name="DebugStep"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If X is a sparse matrix (common in text), convert temporarily to check
        if hasattr(X, "toarray"):
            sample = X.toarray()[5]
        else:
            sample = X[5]

        print(f"\n--- DEBUG: {self.name} ---")
        print(f"Shape of data reaching classifier: {X.shape}")
        print(f"First row (first 15 columns): {sample[:15]}")
        print(f"Indices [1, 2, 4, 5] values: {[sample[i] for i in [1, 2, 4, 5]]}")
        print("-" * 30)
        return X


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FormatDebugger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("\n" + "=" * 50)
        print("SCIENTIFIC DATA FORMAT CHECK")
        print("=" * 50)
        print(f"Container Type: {type(X)}")

        if isinstance(X, pd.DataFrame):
            print("Format: Pandas DataFrame")
            print(f"Dtypes of first 5 columns:\n{X.dtypes.head()}")
        elif isinstance(X, np.ndarray):
            print("Format: NumPy Array")
            print(f"Array Dtype: {X.dtype}")
            if X.dtype == object:
                print("!!! WARNING: Object Array detected (Strings mixed with Floats) !!!")

        # Check specific categorical index (e.g., Index 1)
        sample_val = X.iloc[0, 1] if hasattr(X, 'iloc') else X[0, 1]
        print(f"Value at Index 1: {sample_val} (Type: {type(sample_val)})")
        print("=" * 50 + "\n")
        return X
