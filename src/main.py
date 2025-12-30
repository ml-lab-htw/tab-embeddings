import argparse

from src.run_experiments import ExperimentRunner


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


if __name__ == "__main__":
    main()
