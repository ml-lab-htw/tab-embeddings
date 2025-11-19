import importlib.util
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from source.utils.csv_creator import save_to_csv
from source.utils.load_data import load_dataset
from source.helpers.run_setup import run_experiment


if __name__ == "__main__":
    print("-" * 100)
    print("Running all experiments...")
    print("-" * 100)

    custom_config = False

    # === Ask user for custom config path ===
    user_input = input(
        "If you have your own config file, paste its full path.\n"
        "If you want to proceed with the default config file, just press ENTER:\n> "
    ).strip()

    if user_input:
        if os.path.isfile(user_input):
            print(f"\n✅ Using custom config file: {user_input}")
            spec = importlib.util.spec_from_file_location("custom_config", user_input)
            custom_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config)

            DATASETS = custom_config.DATASETS
            METHOD_KEYS = custom_config.METHOD_KEYS
            LLMS_TO_USE = getattr(custom_config, "LLMS_TO_USE", {})

            custom_config = True
        else:
            print(f"\n❌ File not found at: {user_input}")
            sys.exit(1)
    else:
        print("\n✅ Using default config file.")
        from source.config.config import DATASETS, LLMS_TO_USE, METHODS_TO_RUN

    methods = METHODS_TO_RUN
    datasets = DATASETS

    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        data = load_dataset(DATASETS[dataset_name])

        for method_key in methods:
            if "_te" in method_key:  # text embedding method
                for llm_name, llm_model in LLMS_TO_USE.items():
                    print(f"\nRunning method '{method_key}' with LLM '{llm_name}'...")
                    result = run_experiment(
                        method_key=method_key,
                        dataset_name=dataset_name,
                        data=data,
                        feature_extractor=llm_model,
                        custom_config=custom_config
                    )
                    save_to_csv(result, llm=llm_name)
            else:
                # Non-embedding or RTE methods
                print(f"\nRunning method '{method_key}' (no LLM)...")
                result = run_experiment(
                    method_key=method_key,
                    dataset_name=dataset_name,
                    data=data,
                    custom_config=custom_config
                )
                save_to_csv(result)

    print("\n✅ All experiments finished.")
