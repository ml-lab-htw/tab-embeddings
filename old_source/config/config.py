# config.py
# ===============================================================
# Global experiment configuration for ML method runs
# ===============================================================
import os

from old_source.helpers.llm_loader import feature_extractor_e5_small_v2, feature_extractor_all_minilm_l6_v2

"""
from models import feature_extractor_bge_base_en_v1_5, feature_extractor_e5_small_v2, feature_extractor_e5_base_v2, \
    feature_extractor_e5_large_v2, feature_extractor_bge_small_en_v1_5, feature_extractor_bge_large_en_v1_5, \
    feature_extractor_gist_embedding_v0, feature_extractor_gist_small_embedding_v0, \
    feature_extractor_gist_large_embedding_v0, feature_extractor_gte_small, feature_extractor_gte_base_en_v1_5, \
    feature_extractor_gte_base, feature_extractor_gte_large, feature_extractor_stella_en_400M_v5, \
    feature_extractor_all_minilm_l6_v2, feature_extractor_ember_v1
"""

# TODO (BIG): check all params
# === GLOBAL OPTIONS ===
GLOBAL = {
    "random_state": 42,
    "test_size": 0.2,
    # todo: need these?
    "results_dir": "./results",
    "cache_dir": "./cache",
}

# todo: code zur Erstellung von summaries und metrics hinzufügen
# todo: in readme benutzung erkären
# todo: in readme beschreiben wie hinzufüge ich einen neuen Datensatz, neue Methode, neue llm ...
# === DATASETS ===
DATASETS = {
    "cybersecurity": {
        "type": "multi_file",
        "base_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/")),
        "files": {
            "X": "X_cybersecurity.csv",
            "y": "y_cybersecurity.csv",
            "X_metr": "X_cybersecurity_metrics.csv",
            "summaries": "cybersecurity_summaries.txt",
            "nom_summaries": "cybersecurity_nom_summaries.txt",
        },
        "label_col": "label",
        "pca_components": 50,
        "n_repeats": 1,
        "splits": 5
    },
    #"mimic": {
    #    "type": "split",
    #    "base_dir": "./data/mimic/",
    #    "train_files": {
    #        "X": "X_train.csv",
    #        "y": "y_train.csv",
    #        "X_metr": "X_metr_train.csv",
    #        "summaries": "summaries_train.csv",
    #        "nom_summaries": "nom_summaries_train.csv",
    #    },
    #    "test_files": {
    #        "X": "X_test.csv",
    #        "y": "y_test.csv",
    #        "X_metr": "X_metr_test.csv",
    #        "summaries": "summaries_test.csv",
    #        "nom_summaries": "nom_summaries_test.csv",
    #    },
    #    "label_col": "label",
    #    "pca_components": 50,
    #    "n_repeats": 1,
    #    "splits": 5
    #},

}
"""
"lungdisease": {
    "type": "multi_file",
    "base_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "data/")),
    "files": {
        "X": "X_lung_disease.csv",
        "y": "y_lung_disease.csv",
        "X_metr": "X_lung_disease_metrics.csv",
        "summaries": "lung_disease_summaries.txt",
        "nom_summaries": "lung_disease_nom_summaries.txt",
    },
    "label_col": "target",
    "split_params": {"test_size": 0.2, "random_state": 42},
    "pca_components": 50,
    "n_repeats": 1,
    "splits": 5
},
"""
# TODO
# === MODEL DEFAULT PARAMS ===
DS_MODELS = {
    "lr": {"C": [2, 10], "max_iter": 10000},
    "gbdt": {"min_samples_leaf": [5, 10, 15, 20]},
}

RTE_PARAMS = {
    "rte_params": {
        "embedding__n_estimators": [10, 100],
        "embedding__max_depth": [2, 5]
    }
}

# === FEATURES ===
FEATURES = {
    "cybersecurity": {
        "nominal_features": ['encryption_used',
                             'browser_type',
                             'protocol_type',
                             'unusual_time_access'],
        "text_features": ["text", "nom_text"],
    },
    "lungdisease": {
        "nominal_features": ['Gender',
                             'Smoking Status',
                             'Disease Type',
                             'Treatment Type'
                             ],
        "text_features": ["text", "nom_text"],
    },
    "mimic": {
        "nominal_features": [],
        "text_features": ["text", "nom_text"],
    },
}


# === ALL POSSIBLE METHODS ===
METHOD_KEYS = [
    # Logistic Regression
    "lr",
    "lr_te",
    "lr_te_pca",
    "lr_conc1_te",
    "lr_conc2_te",
    "lr_conc3_te",
    "lr_conc1_pca_te",
    "lr_conc2_pca_te",
    "lr_conc3_pca_te",
    "lr_rte",
    "lr_rte_conc",
    # GBDT
    "gbdt", "gbdt_te", "gbdt_te_pca",
    "gbdt_conc1_te",
    "gbdt_conc2_te",
    "gbdt_conc3_te",
    "gbdt_conc1_pca_te",
    "gbdt_conc2_pca_te",
    "gbdt_conc3_pca_te",
    "gbdt_rte", "gbdt_rte_conc",
]


LLMS_TO_USE = {
    # === MiniLM ===
    #"all_minilm_l6_v2": feature_extractor_all_minilm_l6_v2,
    #"e5_base_v2": feature_extractor_e5_base_v2,
    #"e5_large_v2": feature_extractor_e5_large_v2,

    # === BGE Models ===
    #"bge_small_en_v1_5": feature_extractor_bge_small_en_v1_5,
    #"bge_base_en_v1_5": feature_extractor_bge_base_en_v1_5,
    #"bge_large_en_v1_5": feature_extractor_bge_large_en_v1_5,

    # === GIST Models ===
    #"gist_small_embedding_v0": feature_extractor_gist_small_embedding_v0,
    #"gist_embedding_v0": feature_extractor_gist_embedding_v0,
    #"gist_large_embedding_v0": feature_extractor_gist_large_embedding_v0,

    # === GTE Models ===
    #"gte_small": feature_extractor_gte_small,
    #"gte_base": feature_extractor_gte_base,
    #"gte_base_en_v1_5": feature_extractor_gte_base_en_v1_5,
    #"gte_large": feature_extractor_gte_large,

    # === STELLA ===
    #"stella_en_400M_v5": feature_extractor_stella_en_400M_v5,

    # === Ember ===
    #"ember_v1": feature_extractor_ember_v1,

    # === E5 Models ===
    "e5_small_v2": feature_extractor_e5_small_v2,
}

# todo: möglicherweise methods auf subgroups teilen (nur lr, alles mit nur einer llm, etc)
# === METHODS TO RUN THIS TIME ===
# pick subset or full list
METHODS_TO_RUN = [
    # for quick testing
    #"lr"
    #"lr_conc1_te",
    #"lr_conc2_te",
    #"lr_conc3_te",
    "gbdt_conc1_te",
    "gbdt_conc2_te",
    "gbdt_conc3_te",
]
