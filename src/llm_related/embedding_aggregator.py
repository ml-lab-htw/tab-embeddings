import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class EmbeddingAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor, method="embedding_cls"):
        self.method = method
        self.feature_extractor = feature_extractor

    # Embeddings based on [CLS] token
    def _embedding_cls(self, text_features):
        print(f"Length of text_features: {len(text_features)}")
        embeddings = []
        summary_ = text_features[0]
        for summary in text_features:
            embedding = self.feature_extractor(summary)[0][0]
            embeddings.append(embedding)
        print(len(embeddings))
        return np.array(embeddings)

    # Create mean embedding excluding [CLS] and [SEP] tokens
    def _embedding_mean_without_cls_and_sep(self, text_features):
        embeddings = []
        for summary in text_features:
            embedding = self.feature_extractor(summary)[0][1:-1]
            embeddings.append(np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    # Create mean embedding including [CLS] and [SEP] tokens
    def _embedding_mean_with_cls_and_sep(self, text_features):
        embeddings = []
        for summary in text_features:
            embedding = self.feature_extractor(summary)[0][:]
            embeddings.append(np.mean(embedding, axis=0))
            print("Embedding cls_and_sep dimension" + np.mean(embedding, axis=0))
        print(len(embeddings))
        return np.array(embeddings)

    def fit(self, X, y=None):  # X - summaries
        return self

    def transform(self, X_text):
        if isinstance(X_text, pd.DataFrame):
            X_text = X_text.iloc[:, 0].tolist()

        if not all(isinstance(x, str) for x in X_text):
            print(f"X_text: {X_text}")
            raise ValueError("All inputs must be strings.")

        print("Using token-level model (e.g., BERT-style)")
        if self.method == "embedding_cls":
            return self._embedding_cls(X_text)

        elif self.method == "embedding_mean_with_cls_and_sep":
            return self._embedding_mean_with_cls_and_sep(X_text)

        elif self.method == "embedding_mean_without_cls_and_sep":
            return self._embedding_mean_without_cls_and_sep(X_text)

        else:
            raise ValueError("Invalid aggregation method")
