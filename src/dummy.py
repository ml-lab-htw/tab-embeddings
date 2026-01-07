import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModel

from config.config_manager import ConfigManager
from src.data_prep import DataPreparer
from src.exp_context import ExpContext
from src.llm_related.embedding_aggregator import EmbeddingAggregator
from src.llm_related.llm_registry import FeatureExtractorRegistry
from src.param_grid_factory import ParamGridFactory
from src.pipeline_builder import ConcatTextPipeline
from src.pipeline_factory import PipelineFactory


def load_features(file_path, delimiter=',', n_samples=200):
    data = pd.read_csv(file_path, delimiter=delimiter)
    if n_samples:
        data = data.head(n_samples)  # Take only the first n_samples rows
    print(f"features: {data}")
    return data


def load_labels(file_path, delimiter=',', n_samples=200):
    # Load the labels
    data = pd.read_csv(file_path, delimiter=delimiter)
    if n_samples:
        data = data.head(n_samples)
    y = data.values.ravel()

    y = pd.Series(y)

    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.to_numpy()
    return y


def load_summaries(file_name, n_samples=200):
    if not os.path.exists(file_name):
        print("File not found")
        return []
    with open(file_name, "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    if n_samples:
        summaries_list = summaries_list[:n_samples]
    return summaries_list

"""
def create_gen_feature_extractor(model_name):
    
    '''
    Creates a feature extractor pipeline for a given model.
    Compatible with: CL, Bert, Electra, SimSce, BGE, some GTE(thenlper), tbc
    '''
    print(f"Starting to create a feature extractor{model_name}.")
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Selected device: {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda:0" if device == 0 else "cpu")
    print("Finished creating a feature extractor.")
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=device)
"""

def run():
    """
    Diagnosis:
    1. replace data loading - seems clear
    2. replace feature creator - seems clear
    3. replace where I take features names from - seems clear
    4. replace where I take pipeline from -> here is the problem
    5. replace where I take a grid from seems clear
    """
    #y = load_labels("data/lung_disease/y_lung_disease.csv", n_samples=200)
    #X = load_features("data/lung_disease/X_lung_disease.csv", n_samples=200)
    #summaries = load_summaries("data/lung_disease/lung_disease_summaries.txt", n_samples=200)
    cfg = ConfigManager.load_yaml("config/config.yaml")
    #feature_extractor= create_gen_feature_extractor("intfloat/e5-small-v2")
    feature_extractor = FeatureExtractorRegistry.create("E5_SMALL_V2")

    ctx = ExpContext(
        method_key="gbdt_conc1_te",
        dataset_name="lung_disease",
        cfg=cfg,
        embedding_key="intfloat/e5-small-v2",
        feature_extractor=feature_extractor
    )
    data_preparer = DataPreparer(ctx)
    X_train, X_test, y_train, y_test = data_preparer.prepare()


    concat_hgbc_txt_emb(X_train=X_train, X_test=X_test, y_test=y_test, y_train=y_train,
                        # X_tabular=X, y=y, raw_text_summaries=summaries,
                        feature_extractor=ctx.feature_extractor,
                        nominal_indices=ctx.categorical_features_for_classifier,
                        non_text_columns=ctx.non_text_columns, ctx=ctx)


def concat_hgbc_txt_emb(X_train, X_test, y_train, y_test,
                        #X_tabular, y, raw_text_summaries,
                        feature_extractor,
                        nominal_indices,
                        non_text_columns,
                        ctx: ExpContext):
    n_splits = 5
    n_repeats = 1

    """pipeline = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", non_text_columns),
                ("text", Pipeline([
                    ("embedding_aggregator", EmbeddingAggregator(
                        feature_extractor=feature_extractor,
                        is_sentence_transformer=False)),
                    ("numerical_scaler", MinMaxScaler())
                ]), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_indices))
        ])
    """

    pipeline=PipelineFactory.get_strategy(ctx).build(ctx)
    grid=ParamGridFactory.get_strategy(ctx).build(ctx)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        #{
        #    "classifier__min_samples_leaf": [5, 10, 15, 20],
        #    "transformer__text__embedding_aggregator__method": [
        #        "embedding_cls",
        #        "embedding_mean_with_cls_and_sep",
        #        "embedding_mean_without_cls_and_sep"
        #    ]
        #}
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats),
    )

    # === Evaluation ===
    #X_train, X_test, y_train, y_test = train_test_split(X_tabular, y, test_size=0.2, random_state=42)

    search.fit(X_train, y_train)

    y_test_pred = search.predict(X_test)
    y_test_pred_proba = search.predict_proba(X_test)[:, 1]

    y_train_pred = search.predict(X_train)
    y_train_pred_proba = search.predict_proba(X_train)[:, 1]

    best_params = f"{search.best_params_}"
    print(f"Best hyperparameters: {best_params}")
