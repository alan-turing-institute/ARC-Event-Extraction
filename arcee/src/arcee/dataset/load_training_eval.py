from __future__ import annotations

import json

from datasets import Dataset, load_dataset

import arcee.prompts.prompts as prompts
from arcee.dataset.schemas import data_to_argument_schema, data_to_ed_schema


def load_training_eval_data(
    training_data_path: str,
    eval_data_path: str,
    prompt_cls: str,
    schema_path: str,
    arg_extraction: bool = False,
) -> Dataset:
    """
    This function loads the training and evaluation data from the specified locations
    and formats it for training an event extraction model

    Args:
    training_data_path: str
        The file path where the training data is stored, this should include the file
        name and extension, e.g. "data/RAMS/output/splits/split1/train.json"
    eval_data_path: str
        The file path where the evaluation data is stored, this should include the file
        name and extension, e.g. "data/RAMS/output/splits/split1/dev.json"
    prompt_cls: str
        The class name of the prompt constructor to use for formatting the data, e.g.
        'EDPrompter1'
    schema_path: str
        The file path where the schema is stored for the dataset
    arg_extraction: bool
        If set to True, then it will load the records so that there is one event type
        per record (in order to do argument extraction per event type)
    """
    dataset = load_dataset(
        "json", data_files={"train": training_data_path, "eval": eval_data_path}
    )
    if arg_extraction:
        dataset = dataset.filter(lambda x: len(x["event_mentions"]) > 0)
    with open(schema_path) as file:
        schema = json.load(file)
    # AT THE MOMENT, ONLY WORKS WITH A PROMPT WITHOUT ANY EXAMPLES
    prompt_constructor = getattr(prompts, prompt_cls)(schema)

    formatted_dataset: Dataset = dataset.map(
        format_chat_message,
        fn_kwargs={
            "prompt_constructor": prompt_constructor,
            "schema": schema,
            "arg_extraction": arg_extraction,
        },
        remove_columns=dataset["train"].column_names,
    )
    return formatted_dataset


def format_chat_message(
    data_record: dict[str, str | int | list],
    prompt_constructor: prompts.EEPrompt,
    schema: dict,
    arg_extraction: bool,
) -> dict:
    prompt, _, _, _ = prompt_constructor.get_prompt(data_record)
    if arg_extraction:
        # Doing argument extraction
        response = data_to_argument_schema([dict(data_record)], schema=schema)[0][
            "output"
        ]
    else:
        # Doing event trigger detection
        response = data_to_ed_schema([data_record])[0]["output"]
    json_response = "```json\n" + json.dumps(response, indent=2) + "\n```"

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(json_response)},
        ]
    }
