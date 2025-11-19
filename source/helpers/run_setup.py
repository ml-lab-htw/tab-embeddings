import time

from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score, balanced_accuracy_score, \
    precision_score, average_precision_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from source.config.config import GLOBAL
from source.helpers.data_preparer import prepare_data_for_experiment
from source.helpers.pipeline_builder import build_pipeline, build_param_grid


#
def run_experiment(method_key, dataset_name, data, feature_extractor=None, text_embedding_dict=None, rte_dict=None,
                   custom_config=False, verbose=True):
    """
    Orchestrates the experiment: prepare data, build pipeline, train + evaluate.
    """
    # --- Step 1: prepare data---
    X_train, X_test, y_train, y_test, text_cols, nominal_cols, numerical_cols, cfg = \
        prepare_data_for_experiment(method_key, dataset_name, data, text_embedding_dict, verbose)
    print("Run exp. debug:")
    print(f"Nom columns: {nominal_cols}")
    print(f"Num columns: {numerical_cols}")
    print(f"Text columns: {text_cols}")

    # --- Step 2: build pipeline & grid ---
    pipeline = build_pipeline(method_key=method_key,
                              dataset_name=dataset_name,
                              feature_extractor=feature_extractor,
                              text_features=text_cols,
                              nominal_features=nominal_cols,
                              numerical_features=numerical_cols)
    param_grid = build_param_grid(method_key)

    # --- Step 3: train + evaluate ---
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=RepeatedStratifiedKFold(
            n_splits=cfg.get("splits"),
            n_repeats=cfg.get("n_repeats"),
            random_state=GLOBAL["random_state"],
        ),
    )

    start = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start

    # --- Step 4: test + evaluate ---
    y_test_pred = search.predict(X_test)
    y_test_proba = search.predict_proba(X_test)[:, 1]

    test_metrics = calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_proba)

    y_train_pred = search.predict(X_train)
    y_train_proba = search.predict_proba(X_train)[:, 1]

    train_metrics = calc_metrics(y=y_train, y_pred=y_train_pred, y_pred_proba=y_train_proba)

    if verbose:
        print(f"\n [FINISHED] {dataset_name} – {method_key}")
        print(f"\n Duration: {duration:.2f}s")

    return {
        "dataset": dataset_name,
        "method": method_key,
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        # "duration_sec": duration,
    }


def calc_metrics(y, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "AUC": roc_auc_score(y, y_pred_proba),
        "AP": average_precision_score(y, y_pred_proba),
        "Sensitivity": recall_score(y, y_pred, pos_label=1),
        "Specificity": specificity,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, average='macro'),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred)
    }

    return metrics
