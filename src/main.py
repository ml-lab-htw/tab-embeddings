import argparse

from config.config_manager import ConfigManager
from src.run_experiments import ExperimentRunner
from utils.extract_columns import CSVFeatureSplitter
from utils.path_resolver import resolve_dataset_path
from utils.summaries_generator import TabularSummaryGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tab-embedding experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run tab-embedding experiments"
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Splits numerical and nominal features of a dataset into 2 separate .csv files."
    )

    split_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key as defined in FEATURES and DATASETS",
    )

    summ_parser = subparsers.add_parser(
        "summaries",
        help="Creates text summaries from a dataset."
    )

    summ_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key as defined in FEATURES and DATASETS",
    )

    summ_parser.add_argument(
        "--scope",
        choices=["full", "nominal"],
        help="Generate summaries from full dataset or nominal-only dataset",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigManager.load_yaml(args.config)

    if args.command == "run":
        runner = ExperimentRunner(config_path=args.config)
        runner.run()
    elif args.command == "split":
        dataset = args.dataset
        dataset_cfg = cfg.datasets[dataset]
        splitter = CSVFeatureSplitter(config=cfg)
        splitter.split_and_save(
            dataset_name=dataset,
            input_csv = resolve_dataset_path(dataset_cfg, "X"),
            output_dir=resolve_dataset_path(dataset_cfg)
        )
    elif args.command == "summaries":
        dataset = args.dataset
        dataset_cfg = cfg.datasets[dataset]
        feature_cfg = cfg.features.get(dataset, {})
        scope = args.scope
        if dataset not in cfg.datasets:
            raise KeyError(f"Unknown dataset '{dataset}'")

        nominal_features = feature_cfg.get("nominal_features", [])

        if scope == "full":
            input_csv = resolve_dataset_path(dataset_cfg, "X")
            output_file = resolve_dataset_path(dataset_cfg, "summaries")
            classify_numeric = True
            categorical_columns = nominal_features

        else:
            input_csv = resolve_dataset_path(dataset_cfg, "X_nom")
            output_file = resolve_dataset_path(dataset_cfg, "nom_summaries")
            classify_numeric = False
            categorical_columns = None

        generator = TabularSummaryGenerator(
            categorical_columns=categorical_columns,
            classify_numeric=classify_numeric,
            subject_name="sample",
        )

        generator.generate(
            input_csv=input_csv,
            output_file=output_file
        )
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
