from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomTreesEmbedding
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from config.config_manager import ConfigManager
from src.embedding_aggregator import EmbeddingAggregator
from src.exp_context import ExpContext


def build_nominal_pipeline(ctx: ExpContext) -> Pipeline:
    encoder = OneHotEncoder(
        handle_unknown=ctx.handle_unknown,
        drop=ctx.drop
    )

    return Pipeline([
        ("nominal_imputer", SimpleImputer(strategy=ctx.strategy)),
        ("nominal_encoder", encoder),
    ])


def build_numerical_pipeline(ctx: ExpContext, scale: bool) -> Pipeline:
    steps = [
        ("numerical_imputer", IterativeImputer(max_iter=ctx.imp_max_iter)),
    ]

    if scale:
        steps.append(("numerical_scaler", MinMaxScaler()))

    return Pipeline(steps)


def build_text_pipeline(ctx: ExpContext, with_pca: bool) -> Pipeline:
    steps = [
        ("aggregator", EmbeddingAggregator(
            feature_extractor=ctx.feature_extractor
        )),
    ]

    if with_pca:
        steps.append([
            ("numerical_scaler", StandardScaler()),
            ("pca", PCA(n_components=ctx.pca_components)),
        ])
    else:
        steps.append(("numerical_scaler", MinMaxScaler()))

    return Pipeline(steps)


def build_tabular_transformer(ctx: ExpContext, *, include_text: bool) -> ColumnTransformer:
    transformers = []

    if ctx.numerical_features:
        transformers.append((
            "numerical",
            build_numerical_pipeline(ctx),
            ctx.numerical_features
        ))

    if ctx.nominal_features:
        transformers.append((
            "nominal",
            build_nominal_pipeline(ctx),
            ctx.nominal_features
        ))

    if include_text and ctx.text_features:
        transformers.append((
            "text",
            build_text_pipeline(ctx, with_pca=ctx.flags.has_pca),
            ctx.text_features
        ))

    if not transformers:
        raise ValueError("No transformers defined for ColumnTransformer")

    return ColumnTransformer(transformers)


def build_feature_union(ctx: ExpContext) -> FeatureUnion:
    return FeatureUnion([
        ("raw", build_tabular_transformer(ctx)),
        ("embeddings", Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal",
                 build_nominal_pipeline(ctx),
                 ctx.nominal_features),
                ("numerical",
                 Pipeline([("numerical_imputer", IterativeImputer(max_iter=ctx.imp_max_iter))]),
                 ctx.numerical_features),
            ], remainder="passthrough")),
            ("embedding", RandomTreesEmbedding(
                random_state=ctx.cfg.globals["random_state"]
            )),
        ]))
    ])


def select_classifier(ctx: ExpContext, cfg: ConfigManager):
    random_state = cfg.globals.get("random_state")

    if ctx.flags.is_lr:
        lr_cfg = cfg.model_cfg["lr"]
        return LogisticRegression(
            penalty=lr_cfg["penalty"],
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            random_state=random_state,
        )

    if ctx.flags.is_gbdt:
        return HistGradientBoostingClassifier(
            random_state=random_state,
            # todo: not always/if no just set empty list in ctx?
            # todo: features or indices?
            # todo: cfg vs ctx? clear difference
            categorical_features=ctx.nominal_features,
        )

    raise NotImplementedError(
        f"Classifier not implemented for method_key='{ctx.method_key}'"
    )
