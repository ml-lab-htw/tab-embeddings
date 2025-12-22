import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.exp_context import ExpContext
from tests.standards.pipelines import dataset


class DataLoader:
    """
    loads .csv files
    loads features (X.csv) as dataframe
    loads labels (y.csv) as np.array
    """
    def __init__(self, ctx: ExpContext):
        self._ctx = ctx

    def load_summaries(self):
        # todo: adjust extracting path
        dataset = self._ctx.dataset_name
        dataset_cfg = self._ctx.cfg.datasets.get(dataset)
        # todo: where does that check happen, which summaries are needed here? ExpContext or here?
        summaries = []

        return summaries

    def load_features(self):
        test_mode = self._ctx.cfg.test_mode
        test_samples = self._ctx.cfg.test_samples if test_mode else None

        delimiter = self._ctx.cfg.delimiter # todo

        X = pd.read_csv(file_path, delimiter=delimiter, nrows=test_samples)
        print(f"Loaded tabular features from {file_path} with shape {X.shape}")

        return X

    def load_labels(self):
        """
        Load y.csv labels as np.array
        Check for empty values
        If there are empty values, save indices and remove them
        Encode categories if necessary
        """
        test_mode = self._ctx.cfg.test_mode
        test_samples = self._ctx.cfg.test_samples if test_mode else None

        data = pd.read_csv(file_path, delimiter=delimiter, nrows=test_samples)

        y = data.values.ravel()
        y = pd.Series(y)

        if not np.issubdtype(y.dtype, np.number):
            print(f"Label encoding values: {y.unique()}")
            # todo: add exception
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.to_numpy()

        print(f"Loaded labels from {file_path} with shape {y.shape}")
        return y

    def ensure_same_length(self):
        """
        ensure that the features and labels have same length
        """
        pass

    def concatenate_data(self):
        """
        If experiment requires tabular and text features to be
        concatenated, add text features as a new column to the table.
        """
        pass

    def train_test_split(self):
        """

        """
        pass

    # todo: or just reinit data?
    def return_data(self):
        """
        returns suitable data in a suitable format.
        """
        pass
