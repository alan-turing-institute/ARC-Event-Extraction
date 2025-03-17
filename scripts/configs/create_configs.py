from __future__ import annotations

import argparse
import itertools
import os

import yaml

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_settings(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_yaml(
    exp_name: str,
    model_kwargs: dict,
    data_kwargs: dict,
    prompt_kwargs: dict,
    output_path: str,
) -> None:
    # Create the YAML data
    data = {
        "experiment": exp_name,
        "model_kwargs": model_kwargs,
        "data_kwargs": data_kwargs,
        "prompt_kwargs": prompt_kwargs,
    }

    # Write to a YAML file
    with open(output_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

    print(f"Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load args for creating config files.")
    parser.add_argument("settings_file", type=str, help="Path to settings file.")
    args = parser.parse_args()
    settings = load_settings(args.settings_file)

    MODELS = settings["models"]
    ADAPTERS = settings.get("adapters", [""])
    MODEL_TYPE = settings["model_type"]
    BATCH_SIZE = settings["batch_size"]
    MODEL_INIT_KWARGS = settings["model_init_kwargs"]
    LARGE_MODELS = settings["large_models"]
    DATASET = settings["dataset"]
    SPLITS = settings["splits"]
    PROMPTS = settings["prompts"]
    PROMPTS_RANDOM_EXAMPLES = settings["prompts_random_examples"]
    PROMPTS_SIMILAR_EXAMPLES = settings["prompts_similar_examples"]
    PROMPTS_ARG_EXTRACTION = settings["prompts_arg_extraction"]
    SENTENCE_EMBEDDINGS = settings["sentence_embeddings"]
    NUM_EXAMPLES = settings["num_examples"]
    SEED = settings["seed"]
    CONFIG_FOLDER = settings["config_folder"]
    OUTPUT_FORMAT = settings["output_format"]

    os.makedirs(f"scripts/configs/{CONFIG_FOLDER}", exist_ok=True)
    os.makedirs(f"scripts/slurm/{CONFIG_FOLDER}", exist_ok=True)

    large_model_filenames = []
    small_model_filenames = []
    for model, adapter, split, prompt, output_format in itertools.product(
        MODELS, ADAPTERS, SPLITS, PROMPTS, OUTPUT_FORMAT
    ):
        model_short_name = MODELS[model]
        prompt_short_name = PROMPTS[prompt]
        output_format_short_name = OUTPUT_FORMAT[output_format]
        # Initialise experiment name
        exp_name_init = (
            f"{DATASET}_split{split}_{model_short_name}_{prompt_short_name}_"
            f"{output_format_short_name}"
        )
        if MODEL_TYPE == "guide":
            if output_format == "json":
                model_type = "guide_json"
            elif output_format == "json_schema":
                model_type = "guide_schema"
            else:
                model_type = "guide_free"
        else:
            model_type = MODEL_TYPE

        # DEFINE MODEL KWARGS
        model_short_name = MODELS[model]
        model_kwargs = {
            "model_id": model,
            "model_type": model_type,
            "batch_size": BATCH_SIZE,
            "model_init_kwargs": {},
        }

        # Set model init kwargs for ollama models
        if MODEL_TYPE == "ollama":
            num_ctx = 16384
            if model == "gemma3:27b":
                num_ctx = 8192
            model_kwargs["model_init_kwargs"]["output_format"] = output_format
            model_kwargs["model_init_kwargs"]["options"] = {
                "num_ctx": num_ctx,
                "temperature": 0,
            }
        # And for guidance models
        elif MODEL_TYPE == "guide":
            model_kwargs["model_init_kwargs"] = MODEL_INIT_KWARGS
            model_kwargs["model_init_kwargs"]["adapter_id"] = adapter
            # Update exp name to include apater short name
            exp_name_init = f"{exp_name_init}_{ADAPTERS[adapter]}"

        # DEFINE DATA KWARGS
        data_path = f"data/{DATASET}/output/split{split}/test.jsonlines"
        schema_path = f"data/{DATASET}/schema/event_type_short_definitions.json"
        if prompt in PROMPTS_ARG_EXTRACTION:
            gold_path = f"data/{DATASET}/gold/gold_label_args.jsonl"
        else:
            gold_path = f"data/{DATASET}/gold/gold_label_triggers.jsonl"
        if "EAE" in prompt:
            schema_path = f"data/{DATASET}/schema/pattern.json"
        data_kwargs = {
            "data_path": data_path,
            "schema_path": schema_path,
            "gold_path": gold_path,
        }

        prompt_kwargs = {"prompt_cls": prompt}
        if prompt in PROMPTS_RANDOM_EXAMPLES or prompt in PROMPTS_SIMILAR_EXAMPLES:
            for num_examples in NUM_EXAMPLES:
                prompt_init_kwargs = {
                    "example_data_path": (
                        f"data/{DATASET}/output/split{split}/train.jsonlines"
                    ),
                    "num_examples": num_examples,
                }
                if prompt in PROMPTS_RANDOM_EXAMPLES:
                    exp_name = f"{exp_name_init}_{num_examples}_rand{SEED}"
                    prompt_init_kwargs["seed"] = SEED
                    prompt_kwargs["prompt_init_kwargs"] = prompt_init_kwargs

                    filename = f"scripts/configs/{CONFIG_FOLDER}/{exp_name}.yaml"
                    if model in LARGE_MODELS:
                        large_model_filenames.append(filename)
                    else:
                        small_model_filenames.append(filename)
                    save_yaml(
                        exp_name,
                        model_kwargs,
                        data_kwargs,
                        prompt_kwargs,
                        filename,
                    )
                elif prompt in PROMPTS_SIMILAR_EXAMPLES:
                    for sentence_embedding in SENTENCE_EMBEDDINGS:
                        sentence_embedding_short_name = SENTENCE_EMBEDDINGS[
                            sentence_embedding
                        ]
                        exp_name = (
                            f"{exp_name_init}_{num_examples}"
                            f"_{sentence_embedding_short_name}"
                        )
                        prompt_init_kwargs["sentence_transformer_model"] = (
                            sentence_embedding
                        )
                        prompt_kwargs["prompt_init_kwargs"] = prompt_init_kwargs

                        filename = f"scripts/configs/{CONFIG_FOLDER}/{exp_name}.yaml"
                        if model in LARGE_MODELS:
                            large_model_filenames.append(filename)
                        else:
                            small_model_filenames.append(filename)
                        save_yaml(
                            exp_name,
                            model_kwargs,
                            data_kwargs,
                            prompt_kwargs,
                            filename,
                        )
        else:
            exp_name = exp_name_init
            filename = f"scripts/configs/{CONFIG_FOLDER}/{exp_name}.yaml"
            if model in LARGE_MODELS:
                large_model_filenames.append(filename)
            else:
                small_model_filenames.append(filename)
            save_yaml(
                exp_name,
                model_kwargs,
                data_kwargs,
                prompt_kwargs,
                filename,
            )

    if len(small_model_filenames) > 0:
        filename = f"scripts/slurm/{CONFIG_FOLDER}/small_models.txt"
        with open(filename, "w") as file:
            for name in small_model_filenames:
                file.write(f"{name}\n")

    if len(large_model_filenames) > 0:
        filename = f"scripts/slurm/{CONFIG_FOLDER}/large_models.txt"
        with open(filename, "w") as file:
            for name in large_model_filenames:
                file.write(f"{name}\n")


if __name__ == "__main__":
    main()
