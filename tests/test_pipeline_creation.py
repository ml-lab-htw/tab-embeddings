import pytest

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from src.pipeline_factory import PipelineFactory
from tests.utils.pipeline_assertions import assert_pipeline_steps, assert_column_transformer


@pytest.mark.parametrize("method_key", [
    "lr",
    "lr_rte",
    "lr_rte_conc",
    "lr_te",
    "lr_te_pca",
    "lr_conc1_te",
    "lr_conc2_te",
    "lr_conc3_te",
    "lr_conc1_te_pca",
    "lr_conc2_te_pca",
    "lr_conc3_te_pca",
    "gbdt",
    "gbdt_rte",
    "gbdt_rte_conc",
    "gbdt_te",
    "gbdt_te_pca",
    "gbdt_conc1_te",
    "gbdt_conc2_te",
    "gbdt_conc3_te",
    "gbdt_conc1_te_pca",
    "gbdt_conc2_te_pca",
    "gbdt_conc3_te_pca",
])

def test_pipeline_is_created(ctx_factory, method_key):
    ctx = ctx_factory(method_key)
    pipeline = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    assert isinstance(pipeline, Pipeline)

def test_lr_pipeline_structure(ctx_factory):
    ctx = ctx_factory("lr")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    assert_pipeline_steps(
        pipe,
        ["transformer", "classifier"]
    )

def test_lr_rte_pipeline_structure(ctx_factory):
    ctx = ctx_factory("lr_rte")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    assert_pipeline_steps(
        pipe,
        ["transformer", "embedding", "classifier"]
    )

def test_lr_rte_concat_pipeline(ctx_factory):
    ctx = ctx_factory("lr_rte_conc")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    assert_pipeline_steps(
        pipe,
        ["feature_combiner", "classifier"]
    )

def test_lr_column_transformer(ctx_factory):
    ctx = ctx_factory("lr")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    transformer = dict(pipe.steps)["transformer"]
    assert isinstance(transformer, ColumnTransformer)

    assert_column_transformer(
        transformer,
        ["numerical", "nominal"]
    )

def test_lr_text_concat_transformer(ctx_factory):
    ctx = ctx_factory("lr_conc1_te")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    transformer = dict(pipe.steps)["transformer"]

    assert_column_transformer(
        transformer,
        ["numerical", "nominal", "text"]
    )

def test_lr_classifier_config(ctx_factory, cfg):
    ctx = ctx_factory("lr")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    clf = dict(pipe.steps)["classifier"]
    assert isinstance(clf, LogisticRegression)
    assert clf.max_iter == cfg.model_cfg["lr"]["max_iter"]

def test_gbdt_classifier(ctx_factory):
    ctx = ctx_factory("gbdt")
    pipe = PipelineFactory.get_strategy(ctx.flags).build(ctx)

    clf = dict(pipe.steps)["classifier"]
    assert isinstance(clf, HistGradientBoostingClassifier)
