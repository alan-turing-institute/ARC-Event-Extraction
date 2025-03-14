from __future__ import annotations

import argparse
import itertools

import yaml

MODELS = {
    # "llama3.1:8b": "llama8b",
    # "gemma2:27b": "gemma27b",
    "mistral-small:24b": "mistral24b",
    # "phi4": "phi4",
    # "mistral-large:123b": "mistral123b",
    # "llama3.3:70b": "llama70b",
}
LARGE_MODELS = ["mistral-large:123b", "llama3.3:70b"]
DATASET = "ACE05"
SPLITS = [1]
PROMPTS = {
    "EDPrompter1": "ED1",
    # "EDPrompter2": "ED2",
    # "EDPrompter3": "ED3",
    "EDPrompter4": "ED4",
    "EAEPrompter1": "EAE1",
    # "EAEPrompter2": "EAE2",
    # "EAEPrompter3": "EAE3",
    # "EAEPrompter4": "EAE4",
    # "EAEPrompter5": "EAE5",
    "EAEPrompter6": "EAE6",
}
PROMPTS_RANDOM_EXAMPLES = ["EDPrompter3", "EAEPrompter3", "EAEPrompter4"]
PROMPTS_SIMILAR_EXAMPLES = ["EDPrompter4", "EAEPrompter5", "EAEPrompter6"]
SENTENCE_EMBEDDINGS = {
    # "all-mpnet-base-v2": "mpnetbase",
    "intfloat/multilingual-e5-small": "e5small",
    # "intfloat/e5-small-v2": "e5smallv2",
}
# NUM_EXAMPLES = [3, 5, 10]
NUM_EXAMPLES = [10]
SEED = 42
CONFIG_FOLDER = "ollamaOutputFormat"
OUTPUT_FORMAT = {"null": "unstructured", "json": "json", "json_schema": "schema"}


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
    LARGE_MODELS = settings["large_models"]
    DATASET = settings["dataset"]
    SPLITS = settings["splits"]
    PROMPTS = settings["prompts"]
    PROMPTS_RANDOM_EXAMPLES = settings["prompts_random_examples"]
    PROMPTS_SIMILAR_EXAMPLES = settings["prompts_similar_examples"]
    SENTENCE_EMBEDDINGS = settings["sentence_embeddings"]
    NUM_EXAMPLES = settings["num_examples"]
    SEED = settings["seed"]
    CONFIG_FOLDER = settings["config_folder"]
    OUTPUT_FORMAT = settings["output_format"]

    large_model_filenames = []
    small_model_filenames = []
    for model, split, prompt, output_format in itertools.product(
        MODELS, SPLITS, PROMPTS, OUTPUT_FORMAT
    ):
        model_short_name = MODELS[model]
        prompt_short_name = PROMPTS[prompt]
        output_format_short_name = OUTPUT_FORMAT[output_format]
        # Initialise experiment name
        exp_name_init = (
            f"{DATASET}_split{split}_{model_short_name}_{prompt_short_name}_"
            f"{output_format_short_name}"
        )

        # DEFINE MODEL KWARGS
        model_short_name = MODELS[model]
        num_ctx = 16384
        if model == "gemma3:27b":
            num_ctx = 8192
        model_kwargs = {
            "model_id": model,
            "model_type": "ollama",
            "batch_size": 1,
            "model_init_kwargs": {
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": 0,
                },
                "output_format": output_format if output_format != "null" else None,
            },
        }

        # DEFINE DATA KWARGS
        data_path = f"data/{DATASET}/output/split{split}/test.jsonlines"
        schema_path = f"data/{DATASET}/schema/event_type_short_definitions.json"
        if "EAE" in prompt:
            schema_path = f"data/{DATASET}/schema/pattern.json"
        data_kwargs = {
            "data_path": data_path,
            "schema_path": schema_path,
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

    filename = f"scripts/slurm/{CONFIG_FOLDER}/small_models.txt"
    with open(filename, "w") as file:
        for name in small_model_filenames:
            file.write(f"{name}\n")

    filename = f"scripts/slurm/{CONFIG_FOLDER}/large_models.txt"
    with open(filename, "w") as file:
        for name in large_model_filenames:
            file.write(f"{name}\n")


if __name__ == "__main__":
    main()
