import argparse

from config.config_manager import ConfigManager
from src.run_experiments import ExperimentRunner
from utils.extract_columns import CSVFeatureSplitter
from utils.summaries_generator import TabularSummaryGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tab-embedding experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(config_path=args.config)
    runner.run()

    #cfg = ConfigManager.load_yaml("config/config.yaml")
    #splitter = CSVFeatureSplitter(config=cfg)
    #splitter.split_and_save(
    #    dataset_name="cybersecurity",
    #    input_csv="data/cybersecurity/X_cybersecurity.csv",
    #    output_dir="data/cybersecurity/split"
    #)
    # todo: test summaries creator
    '''generator = TabularSummaryGenerator(
        categorical_values={
            "gender": {0: "male", 1: "female"},
            "smoker": {0: "no", 1: "yes"},
        },
        classify_numeric=True,
        subject_name="patient",
    )

    generator.generate(
        input_csv="X_cybersecurity.csv",
        output_file="cybersecurity_summaries.txt",
    )'''


if __name__ == "__main__":
    main()
