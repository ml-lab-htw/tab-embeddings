EXPECTED_GRIDS = {}

lr_classifier = {"classifier__C": [2, 10]}
gbdt_classifier = {"classifier__min_samples_leaf": [5, 10, 15, 20]}

# === LR baseline ===
EXPECTED_GRIDS["lr"] = lr_classifier


# === LR + RTE ===
EXPECTED_GRIDS["lr_rte"] = {
    **lr_classifier,
    "embedding__n_estimators": [10, 100],
    "embedding__max_depth": [2, 5],
}

# === LR + Text ===
EXPECTED_GRIDS["lr_te"] = {
    **lr_classifier,
    "aggregator__method": [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]
}

# === LR + Text (concatenated) ===
EXPECTED_GRIDS["lr_conc1_te"] = {
    **lr_classifier,
    "transformer__text__aggregator__method": [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]
}

# === LR + RTE (concatenated) ===
EXPECTED_GRIDS["lr_rte_conc"] = {
    **lr_classifier,
    "feature_combiner__embeddings__embedding__n_estimators": [10, 100],
    "feature_combiner__embeddings__embedding__max_depth": [2, 5],
}

# === GBDT baseline ===
EXPECTED_GRIDS["gbdt"] = gbdt_classifier

# === GBDT + RTE ===
EXPECTED_GRIDS["gbdt_rte"] = {
    **gbdt_classifier,
    "embedding__n_estimators": [10, 100],
    "embedding__max_depth": [2, 5],
}

# === GBDT + Text ===
EXPECTED_GRIDS["gbdt_te"] = {
    **gbdt_classifier,
    "aggregator__method": [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]
}

# === GBDT + Text (concatenated) ===
EXPECTED_GRIDS["gbdt_conc1_te"] = {
    **gbdt_classifier,
    "transformer__text__aggregator__method": [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep"
    ]
}

# === GBDT + RTE (concatenated) ===
EXPECTED_GRIDS["gbdt_rte_conc"] = {
    **gbdt_classifier,
    "feature_combiner__embeddings__embedding__n_estimators": [10, 100],
    "feature_combiner__embeddings__embedding__max_depth": [2, 5],
}
