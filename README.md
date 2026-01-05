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

## Usage
You can create your own configuration file by following the structure of config/config.py, which is included in this project.

If you do not want to run all experiments or all LLMs at once, you can comment them out in the configuration file.

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