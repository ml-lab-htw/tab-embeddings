# Tab-embeddings

This project is designed to train machine learning models on tabular data, RTE embeddings, text embeddings, and concatenations of tabular data with embeddings.

The current experiments include:
- 2 downstream models: Logistic Regression and HistGradientBoostingClassifier
- 16 large language models for embeddings
- RTE embeddings
- 4 concatenation approaches: 
  - Random trees embeddings + tabular data 
  - Text embeddings + tabular data
  - Text embeddings + metrical features
  - Nominal text embeddings + metrical tabular data

## Installation
1. Open a command prompt and navigate to the directory where you want to store your Python projects.
2. Clone the repository: git clone https://github.com/ml-lab-htw/tab-embeddings.git
3. Install the required dependencies: pip install -r requirements.txt

## Data preprocessing
This project can be applied to custom datasets in addition to the provided examples. To do so, you must register the 
dataset and perform the required preprocessing steps as outlined below.
1. Add the dataset
Create a new directory under data/ and name it after your dataset (e.g., cybersecurity, bank_churn):<br>
data/<br>
└── bank_churn/ <br>
----├── X_bank_churn.csv<br>
----└── y_bank_churn.csv<br>
2. Register the dataset in the configuration file
Update config/config.yaml by adding entries under both DATASETS and FEATURES. Follow the structure of existing datasets.

DATASETS<br>
  bank_churn:<br>
    path: ./data/bank_churn<br>
    X: "X_bank_churn.csv"<br>
    y: "y_bank_churn.csv"<br>
    X_metr: "X_bank_churn_metrics.csv"<br>
    X_nom: "X_bank_churn_nom.csv"<br>
    summaries: "bank_churn_summaries.txt"<br>
    nom_summaries: "bank_churn_nom_summaries.txt"<br>
    pca_components: 50<br>
    n_splits: 5<br>
    n_repeats: 1<br>

FEATURES:<br>
  bank_churn:<br>
    nominal_features: [<br>
        'nom_feat_1',<br>
        'nom_feat_2',<br>
        # ... <br>
        'nom_feat_n'<br>
    ]<br>
    text_features: ["text"]<br>

3. Generate derived data files<br>
For each dataset, the following derived files must be created:

* X_<dataset>_metrics.csv – numerical features only
* X_<dataset>_nom.csv – nominal (categorical) features only
* <dataset>_summaries.txt – summaries from the full dataset
* <dataset>_nom_summaries.txt – summaries from nominal features only

These files are required for downstream experiments.

4. Run pre-processing commands<br>
4.1 Navigate to the project directory: cd tab-embeddings<br>
4.2 Activate the virtual environment: <br>
source venv/bin/activate (macOS/Linux) <br>
venv/scripts/activate (Windows)<br>
4.3 Split numerical and nominal features: python -m src.main --config config/config.yaml split --dataset <dataset><br>
4.4 Generate summaries from the full dataset: python -m src.main --config config/config.yaml summaries --dataset <dataset> --scope full<br>
4.5 Generate summaries from nominal features only: python -m src.main --config config/config.yaml summaries --dataset <dataset> --scope nominal<br>

Once these steps are completed, the data set is fully prepared and can be used in the experiment pipeline.

## Usage
You can create your own configuration file by strictly following the structure of config/config.py, which is included in this project.

If you do not want to run all experiments or all LLMs at once, you can comment them out in the configuration file.

If you want to only test the code, you should set TEST_MODE to True in the config file. Then, TEST_LLM_KEYS and TEST_EXPERIMENTS will be used. 
You should then also set an amount of samples to be used for testing, f.e. TEST_SAMPLES: 200

Additional LLMs or machine learning methods can be added. Instructions for this will be provided in the future.

When running the project with your own datasets, make sure the data is available in the required formats for the experiments you want to run.
For example, features and targets should be stored in separate files (e.g., X.csv and y.csv).

To start the project, run the following commands:

1. Navigate to the project directory: cd tab-embeddings

2. Activate the virtual environment:
* source venv/bin/activate (macOS/Linux) 
* venv/scripts/activate (Windows)

3. Run the project: python src/main.py --config config/config.yaml  (If you encounter a file not found error, try: python -m src.main --config config/config.yaml
## License
This project is licensed under the MIT License.