from __future__ import annotations

import itertools

import yaml

MODELS = {
    "llama3.1:8b": "llama8b",
    "gemma2:27b": "gemma27b",
    "mistral-small:24b": "mistral24b",
    "phi4": "phi4",
    "mistral-large:123b": "mistral123b",
    "llama3.3:70b": "llama70b",
}
DATASET = "ACE05"
SPLITS = [1]
PROMPTS = [
    "EDPrompter1",
    "EDPrompter2",
    "EDPrompter3",
    "EDPrompter4",
    "EAEPrompter1",
    "EAEPrompter2",
    "EAEPrompter3",
    "EAEPrompter4",
    "EAEPrompter5",
    "EAEPrompter6",
]
PROMPTS_RANDOM_EXAMPLES = ["EDPrompter3", "EAEPrompter3", "EAEPrompter4"]
PROMPTS_SIMILAR_EXAMPLES = ["EDPrompter4", "EAEPrompter5", "EAEPrompter6"]
SENTENCE_EMBEDDINGS = {
    "all-mpnet-base-v2": "all-mpnet-base",
    "intfloat/multilingual-e5-small": "e5small",
}
NUM_EXAMPLES = [3, 5, 10]
SEED = 42


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


config_id = 0
large_model_config_id = 0
for model, split, prompt in itertools.product(MODELS, SPLITS, PROMPTS):
    model_short_name = MODELS[model]
    # Initialise experiment name
    exp_name_init = f"{DATASET}_split{split}_{model_short_name}_{prompt}"

    # DEFINE MODEL KWARGS
    model_short_name = MODELS[model]
    num_ctx = 16384
    if model == "gemma2:27b":
        num_ctx = 8192
    model_kwargs = {
        "model_id": model,
        "model_type": "ollama",
        "batch_size": 1,
        "model_init_kwargs": {"options": {"num_ctx": num_ctx, "temperature": 0}},
    }

    # DEFINE DATA KWARGS
    data_path = f"data/{DATASET}/split{split}/test.jsonlines"
    schema_path = f"data/{DATASET}/event_type_short_definitions.json"
    argument_extraction = False
    if "EAE" in prompt:
        argument_extraction = True
        schema_path = f"data/{DATASET}/pattern.json"
    data_kwargs = {
        "data_path": data_path,
        "schema_path": schema_path,
        "argument_extraction": argument_extraction,
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
                exp_name = f"{exp_name_init}_{num_examples}_{SEED}"
                prompt_init_kwargs["seed"] = SEED
            elif prompt in PROMPTS_SIMILAR_EXAMPLES:
                for sentence_embedding in SENTENCE_EMBEDDINGS:
                    sentence_embedding_short_name = SENTENCE_EMBEDDINGS[
                        sentence_embedding
                    ]
                    exp_name = (
                        f"{exp_name_init}_{num_examples}"
                        "_{sentence_embedding_short_name}"
                    )
                    prompt_init_kwargs["sentence_transformer_model"] = (
                        sentence_embedding
                    )
            prompt_kwargs["prompt_init_kwargs"] = prompt_init_kwargs
            config_id += 1
            save_yaml(
                exp_name,
                model_kwargs,
                data_kwargs,
                prompt_kwargs,
                f"scripts/configs/ollama_group1/{exp_name}.yaml",
            )
    else:
        exp_name = exp_name_init
        config_id += 1
        save_yaml(
            exp_name,
            model_kwargs,
            data_kwargs,
            prompt_kwargs,
            f"scripts/configs/ollama_group1/{exp_name}.yaml",
        )
