from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomTreesEmbedding

from config.config_manager import ConfigManager
from src.llm_related.embedding_aggregator import EmbeddingAggregator


cfg = ConfigManager.load_yaml("./config/config.yaml")
dataset_name = "cybersecurity"

dataset_cfg = cfg.datasets[dataset_name]
feature_cfg = cfg.features[dataset_name]

nominal_features = feature_cfg["nominal_features"]
numerical_features = ["num_1", "num_2"]
text_features = feature_cfg.get("text_features", [])

non_text_columns = nominal_features + numerical_features
all_columns = text_features + non_text_columns


imp_max_iter = 30
class_max_iter = 10000
feature_extractor = None
pca_transformer = PCA(n_components=50)
random_state = 42

EXPECTED_PIPELINES = {}

# === LR === yes
EXPECTED_PIPELINES["lr"] = (
    Pipeline([
        ("transformer", ColumnTransformer([
            ("nominal", Pipeline([
                ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
            ]), nominal_features),
            ("numerical", Pipeline([
                ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                ("numerical_scaler", MinMaxScaler())
            ]), numerical_features),
        ])),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter))
    ]))

# === LR + RTE === yes
EXPECTED_PIPELINES["lr_rte"] = (
    Pipeline([
        ("transformer", ColumnTransformer([
            ("nominal", Pipeline([
                ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
            ]), nominal_features),
            ("numerical", Pipeline([
                ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))
            ]), numerical_features),
        ])),
        ("embedding", RandomTreesEmbedding(random_state=random_state)),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter))
    ]))

# === LR Conc + RTE === yes
EXPECTED_PIPELINES["lr_rte_conc"] = (
    Pipeline([
        ("feature_combiner", FeatureUnion([
            ("raw", ColumnTransformer([
                ("nominal", Pipeline([
                    ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                    ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
                ]), nominal_features),
                ("numerical", Pipeline([
                    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ("numerical_scaler", MinMaxScaler())
                ]), numerical_features),
            ], remainder="passthrough")),
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
                        #("nominal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]), nominal_features),
                    ("numerical", Pipeline(steps=[
                        ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
                    ]), numerical_features),
                ], remainder="passthrough")),
                ("embedding", RandomTreesEmbedding(random_state=random_state))
            ]))
        ])),
        ("classifier", LogisticRegression(penalty="l2", solver="saga", max_iter=class_max_iter, random_state=42))
    ]))

# === LR Text === yes
EXPECTED_PIPELINES["lr_te"] = Pipeline([
    ("aggregator", EmbeddingAggregator(
        feature_extractor=feature_extractor)),
    ("numerical_scaler", MinMaxScaler()),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text + PCA === yes
EXPECTED_PIPELINES["lr_te_pca"] = Pipeline([
    ("aggregator", EmbeddingAggregator(
        feature_extractor=feature_extractor)),
    ("numerical_scaler", StandardScaler()),
    ("pca", pca_transformer),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text Conc1 === yes
EXPECTED_PIPELINES["lr_conc1_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", MinMaxScaler())
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text Conc2 === yes
EXPECTED_PIPELINES["lr_conc2_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", MinMaxScaler())
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text Conc3 === yes
EXPECTED_PIPELINES["lr_conc3_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", MinMaxScaler())
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text PCA Conc ===
EXPECTED_PIPELINES["lr_conc1_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", StandardScaler()),
            ("pca", pca_transformer)
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text PCA Conc2 ===
EXPECTED_PIPELINES["lr_conc2_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", StandardScaler()),
            ("pca", pca_transformer)
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === LR Text PCA Conc3 ===
EXPECTED_PIPELINES["lr_conc3_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
        ]), nominal_features),
        ("numerical", Pipeline([
            ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter)),
            ("numerical_scaler", MinMaxScaler())
        ]), numerical_features),
        ("text", Pipeline([
            ("aggregator", EmbeddingAggregator(
                feature_extractor=feature_extractor)),
            ("numerical_scaler", StandardScaler()),
            ("pca", pca_transformer)
        ]), text_features)
    ])),
    ("classifier", LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=class_max_iter))
])

# === GBDT ===
EXPECTED_PIPELINES["gbdt"] = (
    Pipeline([
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features))
        ]
    ))

# === GBDT + RTE ===
EXPECTED_PIPELINES["gbdt_rte"] = (
    Pipeline([
        ("transformer", ColumnTransformer([
            ("nominal", Pipeline([
                ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
            ]), nominal_features),
            ("numerical", Pipeline([
                ("numerical_imputer", IterativeImputer(max_iter=30))
            ]), numerical_features),
        ])),
        ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=random_state)),
        ("classifier", HistGradientBoostingClassifier())
    ]))

# === GBDT Conc + RTE === yes
num_pipeline_steps = [
    ("numerical_imputer", IterativeImputer(max_iter=imp_max_iter))
]
EXPECTED_PIPELINES["gbdt_rte_conc"] = (
    Pipeline([
        ("feature_combiner", FeatureUnion([
            ("raw", "passthrough"),
            ("embeddings", Pipeline([
                ("transformer", ColumnTransformer([
                    ("nominal", Pipeline([
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
                    ]), nominal_features),
                    ("numerical", Pipeline(
                        steps=num_pipeline_steps
                    ), numerical_features)
                ])),
                ("embedding", RandomTreesEmbedding(sparse_output=False, random_state=random_state)),
            ]))
        ])),
        ("classifier", HistGradientBoostingClassifier(random_state=random_state, categorical_features=nominal_features))
    ]))

# === GBDT Text ===
EXPECTED_PIPELINES["gbdt_te"] = (
    Pipeline([
    ("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor)),
    ("classifier", HistGradientBoostingClassifier(random_state=random_state))
]))

# === GBDT Text + PCA ===
EXPECTED_PIPELINES["gbdt_te_pca"] = Pipeline([
    ("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor)),
    ("numerical_scaler", StandardScaler()),
    ("pca", pca_transformer),
    ("classifier", HistGradientBoostingClassifier(random_state=random_state))
])

# === GBDT Text Conc === yes
pipeline_text_steps = [("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor)),
                       #("numerical_scaler", MinMaxScaler()) todo: not needed for gbdt?
                       ]

EXPECTED_PIPELINES["gbdt_conc1_te"] = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", numerical_features),
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
        ])
# === GBDT Text Conc === yes

EXPECTED_PIPELINES["gbdt_conc2_te"] = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", numerical_features),
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
        ])
# === GBDT Text Conc === yes

EXPECTED_PIPELINES["gbdt_conc3_te"] = Pipeline([
            ("transformer", ColumnTransformer([
                ("numerical", "passthrough", numerical_features),
                ("text", Pipeline(pipeline_text_steps), text_features)
            ])),
            ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
        ])

# === GBDT Text + PCA Conc ===
pipeline_text_steps_pca = [
    ("aggregator", EmbeddingAggregator(feature_extractor=feature_extractor)),
    ("numerical_scaler", StandardScaler()),
    ("pca", pca_transformer)
]

EXPECTED_PIPELINES["gbdt_conc1_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("numerical", "passthrough", numerical_features),
        ("text", Pipeline(pipeline_text_steps_pca), text_features)
    ])),
    ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
])

# === GBDT Text + PCA Conc ===
EXPECTED_PIPELINES["gbdt_conc2_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("numerical", "passthrough", numerical_features),
        ("text", Pipeline(pipeline_text_steps_pca), text_features)
    ])),
    ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
])

# === GBDT Text + PCA Conc ===
EXPECTED_PIPELINES["gbdt_conc3_pca_te"] = Pipeline([
    ("transformer", ColumnTransformer([
        ("numerical", "passthrough", numerical_features),
        ("text", Pipeline(pipeline_text_steps_pca), text_features)
    ])),
    ("classifier", HistGradientBoostingClassifier(categorical_features=nominal_features, random_state=42))
])
