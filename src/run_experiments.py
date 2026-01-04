import time
from datetime import datetime
import logging
from pprint import pprint

from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score, balanced_accuracy_score, \
    precision_score, average_precision_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from config.config_manager import ConfigManager
from pathlib import Path

from src.data_prep import DataPreparer
from src.exp_context import ExpContext
from src.helpers.csv_creator import save_to_csv
from src.llm_related.llm_registry import FeatureExtractorRegistry
from src.param_grid_factory import ParamGridFactory
from src.pipeline_factory import PipelineFactory

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.cfg = ConfigManager.load_yaml(config_path)
        # todo: self.ctx ???
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.results_dir = Path(self.cfg.globals["results_dir"]) / f"results_{self.run_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._feature_extractors: dict[str, object] = {}

    def _get_feature_extractor(self, llm_key: str):
        if llm_key not in self._feature_extractors:
            self._feature_extractors[llm_key] = FeatureExtractorRegistry.create(llm_key)
        return self._feature_extractors[llm_key]

    def run(self):
        """
        Iterates through all datasets defined in config.yaml
        """
        logger.debug(f"Starting run {self.run_id}")
        for dataset_name in self.cfg.datasets:
            self.run_dataset(dataset_name)

    def run_dataset(self, dataset_name: str):
        """
        Iterates through all method_keys defined in config.yaml
        """
        logger.debug(f"\nDataset: {dataset_name}")
        for method_key in self.cfg.experiments:
            self.run_method(dataset_name, method_key)

    def run_method(self, dataset_name: str, method_key: str):
        """
        Iterates through LLMs defined in config.yaml (if needed)
        """
        probe_ctx = ExpContext(
            method_key=method_key,
            dataset_name=dataset_name,
            cfg=self.cfg,
            validate=False
        )
        if probe_ctx.flags.has_text:
            for llm_key in self.cfg.llm_keys:
                logger.debug(f"  LLM key: {llm_key}")
                self.run_experiment(dataset_name, method_key, llm_key)
        else:
            self.run_experiment(dataset_name, method_key, None)

    def run_experiment(
            self,
            dataset_name: str,
            method_key: str,
            llm_key: str | None,
    ):
        feature_extractor = None
        if llm_key:
            feature_extractor = self._get_feature_extractor(llm_key)
        print("-"*50)
        logger.debug(f"  LLM key: {llm_key}, feature_extractor: {feature_extractor}")
        ctx = ExpContext(
            method_key=method_key,
            dataset_name=dataset_name,
            cfg=self.cfg,
            embedding_key=llm_key,
            feature_extractor=feature_extractor
        )
        exp_id = ctx.experiment_id
        logger.debug(f"  Running: {exp_id}")

        # ---- data ----
        data_preparer = DataPreparer(ctx)
        X_train, X_test, y_train, y_test = data_preparer.prepare()

        logger.debug(f"    X_train: {X_train.shape}")
        logger.debug(f"    X_test : {X_test.shape}")

        # ---- pipeline + grid ----
        pipeline = PipelineFactory.get_strategy(ctx).build(ctx)
        param_grid = ParamGridFactory.get_strategy(ctx).build(ctx)

        logger.debug(f"Pipeline structure for {exp_id}:")
        pprint(pipeline.steps)

        logger.debug(f"Parameter grid for {exp_id}:")
        pprint(param_grid)

        # ---- grid search ----
        cv = RepeatedStratifiedKFold(
            n_splits=self.cfg.datasets[dataset_name]["n_splits"],
            n_repeats=self.cfg.datasets[dataset_name]["n_repeats"],
            random_state=self.cfg.globals["random_state"],
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.cfg.globals["grid_search_scoring"],
            cv=cv,
            n_jobs=self.cfg.globals.get("grid_search_n_jobs", -1),
            verbose=1,
        )

        logger.debug(
            f"X_train columns: {X_train.columns} "
        )

        start = time.time()
        search.fit(X_train, y_train)
        duration = time.time() - start

        y_test_pred = search.predict(X_test)
        y_test_pred_proba = search.predict_proba(X_test)[:, 1]

        logger.debug(f"Best hyperparameters: {search.best_params_}")

        test_metrics = calc_metrics(y=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_proba)

        # train metrics
        y_train_pred = search.predict(X_train)
        y_train_pred_proba = search.predict_proba(X_train)[:, 1]
        train_metrics = calc_metrics(y=y_train, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba)

        logger.debug(
            f"Finished {exp_id} in {duration:.2f}s "
            f"(Test metrics: {test_metrics})"
            f"(Train metrics: {train_metrics})"
        )

        # ---- save results ----
        # todo save_to_csv()
        results_dict = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        save_to_csv(
            data_dict=results_dict,
            ctx=ctx
        )

    def info(self):
        logger.debug("Datasets:", list(self.cfg.datasets.keys()))
        logger.debug("Experiments:", self.cfg.experiments)
        logger.debug("LLMs:", self.cfg.llm_keys)

"""
def run_experiment(method_key, dataset_name, data, feature_extractor=None, text_embedding_dict=None, rte_dict=None,
                   custom_config=False, verbose=True):
    '''
    Orchestrates the experiment: prepare data, build pipeline, train + evaluate.
    '''
    
    # --- Step 1: prepare data---
    print("Run exp. debug:")
    print(f"Nom columns: {nominal_cols}")
    print(f"Num columns: {numerical_cols}")
    print(f"Text columns: {text_cols}")

    # --- Step 2: build pipeline & grid ---
    pipeline = None
    param_grid = None

    # --- Step 3: train ---
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
    }"""


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
