from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer


def compare_pipelines(p1, p2) -> bool:
    """Structural comparison of sklearn pipelines."""

    if type(p1) is not type(p2):
        return False

    # ---- Pipeline ----
    if isinstance(p1, Pipeline):
        if len(p1.steps) != len(p2.steps):
            return False
        for (n1, s1), (n2, s2) in zip(p1.steps, p2.steps):
            if n1 != n2:
                return False
            if not compare_pipelines(s1, s2):
                return False
        return True

    # ---- ColumnTransformer ----
    if isinstance(p1, ColumnTransformer):
        if len(p1.transformers) != len(p2.transformers):
            return False
        for (n1, t1, cols1), (n2, t2, cols2) in zip(
            p1.transformers, p2.transformers
        ):
            if n1 != n2:
                return False
            if cols1 != cols2:
                return False
            if not compare_pipelines(t1, t2):
                return False
        return True

    # ---- FeatureUnion ----
    if isinstance(p1, FeatureUnion):
        if len(p1.transformer_list) != len(p2.transformer_list):
            return False
        for (n1, t1), (n2, t2) in zip(
            p1.transformer_list, p2.transformer_list
        ):
            if n1 != n2:
                return False
            if not compare_pipelines(t1, t2):
                return False
        return True

    # ---- Leaf estimator ----
    return type(p1) is type(p2)
