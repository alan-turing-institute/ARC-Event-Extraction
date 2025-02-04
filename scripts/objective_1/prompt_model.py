from __future__ import annotations

import argparse

from arcee.ee_models.ee_models import EEOllamaModel
from arcee.prompts.prompts import EAEPromptNoExample_01


def parse_args():
    parser = argparse.ArgumentParser(description="Get arguments for prompting a model")
    parser.add_argument(
        "--predict_dataset_filepath",
        required=True,
        help="The path and file of the dataset to be predicted",
    )
    parser.add_argument(
        "--pattern_filepath",
        required=True,
        help="The path and file of the pattern (schema) to be used.",
    )
    parser.add_argument(
        "--ee_model",
        required=True,
        type=str,
        help="Either 'ollama' or 'hf', depending on the implementation.",
    )
    parser.add_argument(
        "--llm",
        required=True,
        type=str,
        help=(
            "The name of the llm to use for prompting, e.g.'gemma2:27b' "
            "or 'llama3.2:1b'"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predict_dataset = args.predict_dataset_filepath
    pattern = args.pattern_filepath
    llm = args.llm

    if args.ee_model == "ollama":
        ee_model = EEOllamaModel(llm)
    else:
        err_msg = "Only valid EE model is ollama"
        raise NotImplementedError(err_msg)

    prompt_obj = EAEPromptNoExample_01()

    ee_model.predict_eae(predict_dataset, pattern, prompt_obj)


if __name__ == "__main__":
    main()
