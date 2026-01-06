import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sympy import nroots

from config.config_manager import ConfigManager
from src.data_prep import DataLoader
from src.dummy import run
from src.exp_context import ExpContext
from src.llm_related.embedding_aggregator import EmbeddingAggregator
from src.llm_related.llm_registry import FeatureExtractorRegistry
from src.pipeline_builder import ConcatTextPipeline
from src.pipeline_factory import PipelineFactory
from src.run_experiments import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tab-embedding experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(config_path=args.config)
    runner.run()

    """
    llm_key="E5_SMALL_V2"
    cfg = ConfigManager.load_yaml("config/config.yaml")
    ctx = ExpContext(
        method_key="lr_conc1_te",
        dataset_name="cybersecurity",
        cfg=cfg,
        embedding_key=llm_key,
        feature_extractor=FeatureExtractorRegistry.create(llm_key)
    )
    strategy = PipelineFactory.get_strategy(ctx=ctx) # same as you pass to GridSearchCV
    pipeline = strategy.build(ctx)
    data_loader = DataLoader()
    X, y = (data_loader.load_features(path="data/cybersecurity/X_cybersecurity.csv", nrows=200),
            data_loader.load_labels("data/cybersecurity/y_cybersecurity.csv", nrows=200))
    summaries = data_loader.load_labels("data/cybersecurity/cybersecurity_summaries.txt", nrows=200)
    X["text"] = pd.Series(summaries).fillna("").astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    concat_hgbc_txt_emb(dataset_name="cybersecurity", text_feature_column_name="text",
                        nominal_features=ctx.nominal_features, raw_text_summaries=summaries,
                        X_tabular=X, y=y, feature_extractor=ctx.feature_extractor)"""

def concat_hgbc_txt_emb(dataset_name,
                        X_tabular, y, text_feature_column_name, feature_extractor,
                        nominal_features, raw_text_summaries):
    dataset = dataset_name
    n_splits = 5
    n_components = 50
    n_repeats = 1

    metrics_per_fold = []
    text_features = [text_feature_column_name]

    X_tabular[text_feature_column_name] = raw_text_summaries
    non_text_columns = list(set(X_tabular.columns) -
                            set(text_features))
    nominal_feature_indices = [
        non_text_columns.index(col)
        for col in nominal_features if col in non_text_columns
    ]

    pca=False

    is_sentence_transformer = False


    pipeline_text_steps = [
        ("embedding_aggregator", EmbeddingAggregator(
            feature_extractor=feature_extractor,
            is_sentence_transformer=is_sentence_transformer)),
        # ("debug_text", DebugTransformer(name="Text Debug"))
    ]
    if pca:
        pipeline_text_steps.append(("numerical_scaler", StandardScaler()))
        pipeline_text_steps.append(("pca", PCA(n_components=n_components)))
    else:
        pipeline_text_steps.append(("numerical_scaler", MinMaxScaler()))

    search = GridSearchCV(
        estimator=Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", non_text_columns),
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_feature_indices))
        ]),
        param_grid={
            "classifier__min_samples_leaf": [5, 10, 15, 20],
            "transformer__text__embedding_aggregator__method": [
                "embedding_cls",
                "embedding_mean_with_cls_and_sep",
                "embedding_mean_without_cls_and_sep"
            ]
        },
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats),
        #n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(X_tabular, y, test_size=0.2, random_state=42)
    search.fit(X_train, y_train)

    y_test_pred = search.predict(X_test)
    y_test_pred_proba = search.predict_proba(X_test)[:, 1]

    best_param = f"Best params for this fold: {search.best_params_}"

    y_train_pred = search.predict(X_train)
    y_train_pred_proba = search.predict_proba(X_train)[:, 1]


    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {best_params}")
    print(f"Test metrics per fold: {y_test_pred_proba}")


if __name__ == "__main__":
    # main()
    run()
