from config.config_manager import ConfigManager
from src.exp_context import ExpContext
from src.pipeline_factory import PipelineFactory

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







    cfg = ConfigManager()
    experiments = cfg.experiments
    embedding_keys = cfg._require_section("TEXT_EMBEDDINGS")

    for method_key in experiments:
        flags = Flags.from_method_key(method_key)

        if flags.has_text:
            for emb_key in embedding_keys:
                ctx = ExpContext(
                    method_key=method_key,
                    dataset_name=dataset_name,
                    cfg=cfg,
                    embedding_key=emb_key,
                )
                pipeline = PipelineFactory.get_strategy(ctx).build(ctx)
                run_experiment(ctx, pipeline)
        else:
            ctx = ExpContext(
                method_key=method_key,
                dataset_name=dataset_name,
                cfg=cfg,
            )
            pipeline = PipelineFactory.get_strategy(ctx).build(ctx)
            run_experiment(ctx, pipeline)


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
