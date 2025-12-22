import sys

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding, HistGradientBoostingClassifier

from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

from old_source.config.config import DATASETS, GLOBAL, DS_MODELS, RTE_PARAMS
from src.embedding_aggregator import EmbeddingAggregator


# ======================================
# Build Pipeline
# ======================================
# todo: exceptions hinzufügen
def build_pipeline(
    method_key: str,
    dataset_name: str,
    text_features=None,
    nominal_features=None,
    numerical_features=None,
    feature_extractor=None,
    imp_max_iter: int = 30,
):
    """
    Build fully expanded pipeline for any method_key.
    """

    # Flags
    has_pca = "pca" in method_key
    has_text = "_te" in method_key
    has_rte = "rte" in method_key
    is_concat = "conc" in method_key
    is_lr = "lr" in method_key
    is_gbdt = "gbdt" in method_key

    model_cfg = DS_MODELS["lr"] if is_lr else DS_MODELS["gbdt"]

    cfg = DATASETS[dataset_name]
    n_components = cfg.get("pca_components")
    lr_max_iter = model_cfg.get("max_iter")
    if has_pca and (n_components is None):
        print(f"[ERROR] Missing 'pca_components' in config for dataset '{dataset_name}'. Please define it before "
              f"executing this program.")
        sys.exit(1)

    # todo: move all variables to config
    # === COMMON PIPELINES ===
    nominal_pipeline = Pipeline([
        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])

    num_steps = [("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))]
    if not has_rte:
        num_steps.append(("numerical_scaler", StandardScaler() if has_pca else MinMaxScaler()))
    numerical_pipeline = Pipeline(num_steps)

    if has_text and feature_extractor:
        text_steps = [("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor))]
        if has_pca:
            text_steps += [("numerical_scaler", StandardScaler()), ("pca", PCA(n_components=n_components))]
        elif not has_pca and is_lr:
            text_steps.append(("numerical_scaler", MinMaxScaler()))
    else:
        text_steps = None

    # === BUILD PIPELINE ===
    if not has_text and not has_rte and not is_concat:
        # Simple LR / GBDT
        if is_lr:
            classifier = LogisticRegression(
                penalty="l2", solver="saga", max_iter=lr_max_iter,
                random_state=GLOBAL["random_state"]
            )
            steps = [("transformer", ColumnTransformer([
                                    ("nominal", nominal_pipeline, nominal_features),
                                    ("numerical", numerical_pipeline, numerical_features)
                ]))]
        elif is_gbdt:
            classifier = HistGradientBoostingClassifier(
                random_state=GLOBAL["random_state"],
                categorical_features=nominal_features
            )
            steps = []
        else:
            return "Unknown classifier"
        steps.append(("classifier", classifier))
        pipeline = Pipeline(steps=steps)

        return pipeline

    if has_rte and not is_concat and not has_text:
        # LR/GBDT + RTE
        if is_lr:
            classifier = LogisticRegression(
                penalty="l2", solver="saga", max_iter=lr_max_iter,
                random_state=GLOBAL["random_state"]
            )
        elif is_gbdt:
            classifier = HistGradientBoostingClassifier(random_state=GLOBAL["random_state"])
        else:
            # todo: add error message
            return
        pipeline = Pipeline([
            ("transformer", ColumnTransformer([
                ("nominal", nominal_pipeline, nominal_features),
                ("numerical", Pipeline([("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))]),
                 numerical_features)
            ])),
            ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=GLOBAL["random_state"])),
            ("classifier", classifier)
        ])
        return pipeline

    if has_text and not is_concat:
        # LR/GBDT text only
        if is_lr:
            classifier = LogisticRegression(
                penalty="l2", solver="saga", max_iter=lr_max_iter,
                random_state=GLOBAL["random_state"]
            )
        else:
            classifier = HistGradientBoostingClassifier(random_state=GLOBAL["random_state"])
        text_steps.append(("classifier", classifier))
        pipeline = Pipeline(
            text_steps
        )
        return pipeline

    if is_concat and has_rte:
        # Concatenated pipelines
        transformers = [("raw", ColumnTransformer([
            ("nominal", nominal_pipeline, nominal_features),
            ("numerical", numerical_pipeline, numerical_features)
        ], remainder="passthrough"))]

        if has_rte:
            rte_branch = Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                    ]), nominal_features),
                    ("numerical", Pipeline([("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))]),
                     numerical_features)
                ], remainder="passthrough")),
                ("embedding", RandomTreesEmbedding(random_state=GLOBAL["random_state"]))
            ])
            transformers.append(("embeddings", rte_branch))

        if has_text:
            transformers.append(("text", text_steps))

        if is_lr:
            classifier = LogisticRegression(
                penalty="l2", solver="saga", max_iter=lr_max_iter,
                random_state=GLOBAL["random_state"]
            )
        else:
            classifier = HistGradientBoostingClassifier(random_state=GLOBAL["random_state"],
                                                        categorical_features=nominal_features)

        pipeline = Pipeline([
            ("feature_combiner", FeatureUnion(transformer_list=transformers)),
            ("classifier", classifier)
        ])
        return pipeline
    if is_concat and has_text and is_lr:
        # ---------- conc_lr_te ----------
        pipeline = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
                ]), nominal_features),
                ("text", Pipeline(text_steps), text_features)
            ])),
            ("classifier", LogisticRegression(
                penalty="l2", solver="saga", max_iter=lr_max_iter, random_state=GLOBAL["random_state"]))
        ])
        return pipeline

    if is_concat and has_text and is_gbdt:
        # ---------- conc_hgbc_te ----------
        if not has_pca:
            text_steps.append(('numerical_scaler', MinMaxScaler()))
        pipeline = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", numerical_features),
                ("text", Pipeline(text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(
                random_state=GLOBAL["random_state"], categorical_features=nominal_features))
        ])
        return pipeline

    if is_concat and has_rte and is_lr:
        # ---------- conc_lr_rte ----------
        num_pipeline_steps = [
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))
        ]
        if has_pca:
            num_pipeline_steps.append(("scaler", StandardScaler()))

        pipeline = Pipeline([
            ("feature_combiner", FeatureUnion([
                ("raw", ColumnTransformer([
                    ("nominal", nominal_pipeline, nominal_features),
                    ("numerical", numerical_pipeline, numerical_features)
                ], remainder="passthrough")),
                ("embeddings", Pipeline([
                    ("transformer", ColumnTransformer([
                        ("nominal", Pipeline([
                            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                            ("nominal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                        ]), nominal_features),
                        ("numerical", Pipeline(steps=num_pipeline_steps), numerical_features)
                    ], remainder="passthrough")),
                    ("embedding", RandomTreesEmbedding(random_state=GLOBAL["random_state"])),
                    ("pca", PCA(n_components=n_components)) if has_pca else ("noop", "passthrough")
                ]))
            ])),
            ("classifier", LogisticRegression(
                penalty="l2", solver="saga", max_iter=model_cfg.get("max_iter"), random_state=GLOBAL["random_state"]))
        ])
        return pipeline

    if is_concat and has_rte and is_gbdt:
        # ---------- conc_hgbc_rte ----------
        num_pipeline_steps = [
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))
        ]
        if has_pca:
            num_pipeline_steps.append(("scaler", StandardScaler()))

        pipeline = Pipeline([
            ("feature_combiner", FeatureUnion([
                ("raw", "passthrough"),
                ("embeddings", Pipeline([
                    ("transformer", ColumnTransformer([
                        ("nominal", nominal_pipeline, nominal_features),
                        ("numerical", Pipeline(steps=num_pipeline_steps), numerical_features)
                    ])),
                    ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=GLOBAL["random_state"])),
                    ("pca", PCA(n_components=n_components)) if has_pca else ("noop", "passthrough")
                ]))
            ])),
            ("classifier", HistGradientBoostingClassifier(
                random_state=GLOBAL["random_state"], categorical_features=nominal_features))
        ])
        return pipeline
    raise ValueError(f"Unsupported pipeline configuration: {method_key}")


# ======================================
# Build Parameter Grid
# In Ordnung
# ======================================
def build_param_grid(method_key):
    """
    Builds a parameter grid for GridSearchCV depending on the model and text usage.
    """

    is_lr = "lr" in method_key
    is_gbdt = "gbdt" in method_key
    has_rte = "rte" in method_key

    param_grid = {}
    model_cfg = DS_MODELS["lr"] if is_lr else DS_MODELS["gbdt"] # todo: if new method?
    rte_cfg = RTE_PARAMS["rte_params"]


    lr_class_c = model_cfg.get("C")
    gbdt_min_samples_leaf = model_cfg.get("min_samples_leaf")

    gbdt_n_estimators = model_cfg.get("n_estimators")
    gbdt_max_depth = model_cfg.get("max_depth")

    rte_params = rte_cfg

    if is_lr and (lr_class_c is None):
        print(f"[ERROR] Missing 'C' in config for downstream model Logistic Regression. Please define it before "
              f"executing this program.")
        sys.exit(1)

    if is_gbdt and (gbdt_min_samples_leaf is None):
        print(f"[ERROR] Missing 'C' in config for downstream model Logistic Regression. Please define it before "
              f"executing this program.")
        sys.exit(1)

    #################
    # ON TABLE DATA #
    #################
    if is_lr:
        param_grid.update({
            "classifier__C": lr_class_c,
        })

    elif is_gbdt:
        param_grid.update({
            "classifier__min_samples_leaf": gbdt_min_samples_leaf
        })

    ######################
    # ON TEXT EMBEDDINGS #
    ######################
    if "_te" in method_key:
        # todo: to config
        embedding_methods = [
            "embedding_cls",
            "embedding_mean_with_cls_and_sep",
            "embedding_mean_without_cls_and_sep"
        ]
        if "conc" not in method_key:
            param_grid.update({
                "aggregator__method": embedding_methods
            })
        else:
            param_grid.update({
                # "transformer__text__embedding_aggregator__method": embedding_methods
                "transformer__text__aggregator__method": embedding_methods
            })

    ##########
    # ON RTE #
    ##########
    elif has_rte:
        rte_params = rte_params
        if "conc" not in method_key:
            param_grid.update({
                # f"embedding__{key}":
                f"{key}":
                    value for key, value in rte_params.items()
            })
        else:
            param_grid.update({
                f"feature_combiner__embeddings__{key}":
                    value for key, value in rte_params.items()
            })

    return param_grid