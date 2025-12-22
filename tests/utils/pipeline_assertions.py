from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion


def assert_pipeline_steps(pipeline: Pipeline, expected_steps: list[str]):
    assert isinstance(pipeline, Pipeline)
    actual = [name for name, _ in pipeline.steps]
    assert actual == expected_steps


def assert_step_type(pipeline: Pipeline, step_name: str, expected_type: type):
    step = dict(pipeline.steps)[step_name]
    assert isinstance(step, expected_type)


def assert_column_transformer(ct: ColumnTransformer, expected_names: list[str]):
    actual = [name for name, _, _ in ct.transformers]
    assert actual == expected_names


def assert_feature_union(fu: FeatureUnion, expected_names: list[str]):
    actual = [name for name, _ in fu.transformer_list]
    assert actual == expected_names
