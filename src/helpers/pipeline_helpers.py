import logging

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomTreesEmbedding
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from config.config_manager import ConfigManager
from src.llm_related.embedding_aggregator import EmbeddingAggregator
from src.exp_context import ExpContext


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

def build_nominal_pipeline(ctx: ExpContext) -> Pipeline:
    encoder = OneHotEncoder(
        handle_unknown=ctx.cfg.data_prep["OHE"]["handle_unknown"],
        drop=ctx.cfg.data_prep["OHE"]["drop"]
    )
    imputer = SimpleImputer(strategy=ctx.cfg.data_prep["simp_imp"]["strategy"])

    return Pipeline([
        ("nominal_imputer", imputer),
        ("nominal_encoder", encoder),
    ])


def build_numerical_pipeline(ctx: ExpContext, scale: bool) -> Pipeline:
    max_iter=ctx.cfg.globals["imp_max_iter"]
    steps = [
        ("numerical_imputer", IterativeImputer(max_iter=max_iter)),
        #("numerical_scaler", StandardScaler()),
    ]

    if scale:
        steps.append(("numerical_scaler", MinMaxScaler()))

    return Pipeline(steps)


def build_text_pipeline_steps(ctx: ExpContext) -> list:
    dataset = ctx.dataset_name
    pca_components = ctx.cfg.datasets[dataset]["pca_components"]
    steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=ctx.feature_extractor
        ))
    ]
    if ctx.flags.has_pca:
        steps.append(("numerical_scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=pca_components)))
    elif ctx.flags.is_lr:
        steps.append(("numerical_scaler", MinMaxScaler()))
    return steps


def build_tabular_transformer(ctx: ExpContext, *, include_text: bool, scale: bool) -> ColumnTransformer:
    transformers = []

    if ctx.nominal_features:
        transformers.append((
            "nominal",
            build_nominal_pipeline(ctx),
            ctx.nominal_features
        ))

    if ctx.numerical_features:
        transformers.append((
            "numerical",
            build_numerical_pipeline(ctx=ctx, scale=scale),
            ctx.numerical_features
        ))

    if include_text and ctx.text_features:
        transformers.append((
            "text",
            Pipeline(steps=build_text_pipeline_steps(ctx)),
            ctx.text_features
        ))

    if not transformers:
        raise ValueError("No transformers defined for ColumnTransformer")

    return ColumnTransformer(transformers)


def build_feature_union(ctx: ExpContext) -> FeatureUnion:
    # include_text is set to False, because build_feature_union
    # has been only created for RTE experiments
    scale_before_rte = False
    return FeatureUnion([
        ("raw", build_raw_branch(ctx)),
        ("embeddings", Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal",
                 build_nominal_pipeline(ctx),
                 ctx.nominal_features),
                ("numerical",
                 build_numerical_pipeline(ctx=ctx, scale=scale_before_rte),
                 ctx.numerical_features),
            ]#, remainder="passthrough"
            )),
            ("embedding", RandomTreesEmbedding(
                sparse_output=False,
                random_state=ctx.cfg.globals["random_state"]
            )),
        ]))
    ])

def build_raw_branch(ctx: ExpContext) -> ColumnTransformer | str:
    """
    Preprocessing of the raw features as expected by the
    downstream classifier.
    """
    if ctx.flags.is_gbdt:
        if ctx.flags.has_rte:
            return "passthrough"
        elif ctx.flags.has_text:
            logging.debug(f"Non text columns: {ctx.non_text_columns}")
            # todo: there is a problem here!
            text_steps = build_text_pipeline_steps(ctx)
            return ColumnTransformer([
                ('numerical', 'passthrough', ctx.non_text_columns),
                ('text', Pipeline(text_steps), ctx.text_features)])
                #('text', Pipeline(
                #    [
                #        ("embedding_aggregator", EmbeddingAggregator(
                #            feature_extractor=ctx.feature_extractor,
                #            is_sentence_transformer=False)),
                #        ("numerical_scaler", MinMaxScaler())
                #    ]
                #), ctx.text_features)])

    elif ctx.flags.is_lr:
        return build_tabular_transformer(
            ctx=ctx,
            include_text=False,
            scale=True
        )

    raise NotImplementedError(
        f"Raw branch not defined for method_key='{ctx.method_key}'"
    )


def select_classifier(ctx: ExpContext, cfg: ConfigManager):
    random_state = cfg.globals.get("random_state")

    if ctx.flags.is_lr:
        lr_cfg = cfg.model_cfg["lr"]
        return LogisticRegression(
            l1_ratio=lr_cfg["l1_ratio"],
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            random_state=random_state,
        )

    if ctx.flags.is_gbdt:
        logger.debug(f"Select classifier -> cat features: {ctx.categorical_features_for_classifier}")
        return HistGradientBoostingClassifier(
            random_state=random_state,
            categorical_features=ctx.categorical_features_for_classifier,
        )

    raise NotImplementedError(
        f"Classifier not implemented for method_key='{ctx.method_key}'"
    )
