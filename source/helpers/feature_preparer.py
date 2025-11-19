import pandas as pd


def prepare_features(method_key, data, text_embedding_dict=None):
    """
    Prepares helpers for a given method_key.
    Returns a pandas dataframe or list of strings
    """
    # TODO: will it work if the text column contains embeddings not text?
    print(f"Preparing helpers for method {method_key}.")
    # --- NO EMB OR RTE ---
    if "rte" in method_key or "_te" not in method_key:
        return data["X_train"], data["X_test"]  # DataFrame

    # --- Text embeddings only ---
    if "te" in method_key and "conc" not in method_key:
        summaries_train = _get_text_features(data["summaries_train"], text_embedding_dict)
        summaries_test = _get_text_features(data["summaries_test"], text_embedding_dict)
        return summaries_train, summaries_test

    X_train_list, X_test_list = [], []

    # --- Concatenations ---
    if "conc1" in method_key and "rte" not in method_key:
        summaries_train = _get_text_features(data["summaries_train"], text_embedding_dict)
        summaries_test = _get_text_features(data["summaries_test"], text_embedding_dict)
        X_train_list.append(pd.DataFrame({"text": summaries_train}))
        X_test_list.append(pd.DataFrame({"text": summaries_test}))

        X_train_list.append(data["X_train"].copy())
        X_test_list.append(data["X_test"].copy())
    if "conc2" in method_key:
        summaries_train = _get_text_features(data["summaries_train"], text_embedding_dict)
        summaries_test = _get_text_features(data["summaries_test"], text_embedding_dict)
        X_train_list.append(pd.DataFrame({"text": summaries_train}))
        X_test_list.append(pd.DataFrame({"text": summaries_test}))

        # Metric/tabular helpers
        X_train_list.append(data["X_metr_train"].copy())
        X_test_list.append(data["X_metr_test"].copy())

    if "conc3" in method_key:
        nom_train = _get_text_features(data["nom_summaries_train"], text_embedding_dict)
        nom_test = _get_text_features(data["nom_summaries_test"], text_embedding_dict)
        X_train_list.append(pd.DataFrame({"nom_text": nom_train}))
        X_test_list.append(pd.DataFrame({"nom_text": nom_test}))

        # Metric/tabular helpers
        X_train_list.append(data["X_metr_train"].copy())
        X_test_list.append(data["X_metr_test"].copy())

    # --- Concatenate ---
    # todo: check if we have column names
    X_train = pd.concat(X_train_list, axis=1)
    X_test = pd.concat(X_test_list, axis=1)
    print(f"→ Final concatenated shapes: train={X_train.shape}, test={X_test.shape}")
    print(f"Train columns: {list(X_train.columns)}")
    print(f"Test columns: {list(X_test.columns)}")
    print(f"X_train head(). {X_train.head(2)}")
    print(f"X_test head(). {X_test.head(2)}")

    return X_train, X_test


# --- Helpers ---
# TODO: works correctly with pca?
def _get_text_features(text_list, embedding_dict=None):
    if embedding_dict is not None:
        return embedding_dict
    else:
        return text_list


def _to_numpy(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.to_numpy()
    return x
