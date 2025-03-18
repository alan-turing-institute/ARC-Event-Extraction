from __future__ import annotations

import json

from datasets import Dataset, load_dataset

import arcee.prompts.prompts as prompts
from arcee.dataset.schemas import data_to_argument_schema, data_to_ed_schema


def expand_record(record: dict) -> list[dict]:
    """
    Takes a record from the dataset and expands it so that it returns one record for
    each event type

    Args:
        record (dict): The record to expand

    Returns:
        list[dict]: A list of the expanded records
    """
    num_mentions = record.get("event_mentions", 1)
    if num_mentions > 1:
        expanded_records = []
        for mention in range(num_mentions):
            new_record = record.copy()
            new_record["event_mentions"] = [new_record["event_mentions"][mention]]
            expanded_records.append(new_record)
        return expanded_records
    return [record]


def load_training_eval_data(
    training_data_path: str,
    eval_data_path: str,
    prompt_cls: str,
    schema_path: str,
    only_events: bool = False,
) -> Dataset:
    """
    This function loads the training and evaluation data from the specified locations
    and formats it for training an event extraction model
    TODO: Update function so that it can work on prompts with examples

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
    only_events: bool
        If set to True, then it will load the records so that there is one event type
        per record (might be wanted for training event detection models)

    Returns:
    Dataset
        A dataset object with the formatted training and evaluation data
    """
    dataset = load_dataset(
        "json", data_files={"train": training_data_path, "eval": eval_data_path}
    )
    # if arg_extraction:
    #     dataset = dataset.filter(lambda x: len(x["event_mentions"]) > 0)
    with open(schema_path) as file:
        schema = json.load(file)
    # TODO: Update so that it can take more kwargs for the prompt constructor,
    # e.g., to construct examples for a prompt.
    prompt_constructor = getattr(prompts, prompt_cls)(schema)
    arg_extraction = prompt_constructor.get_is_arg_extraction()

    if only_events or arg_extraction:
        # If doing argument extraction, then we only want to keep records that have
        # events. If doing event extraction, this might sometimes be wanted.
        dataset = dataset.filter(lambda x: len(x["event_mentions"]) > 0)
        if arg_extraction:
            dataset = {
                split: dataset[split].map(expand_record, batched=False)
                for split in dataset
            }

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
    """
    Takes a data record, prompt and schema and returns messages to be used when
    training a model. i.e., a question and an expected response.

    Args:
        data_record: dict[str, str  |  int  |  list])
            A dictionary containing the input text and details of events (if they exist
            in the text)
        prompt_constructor: prompts.EEPrompt
            An instance of EEPrompt
        schema: dict
            A dictionary containing the schema of event types and argument roles
        arg_extraction: bool
            If set to True, then it will load the records so that there is one event
            type per record (in order to do argument extraction per event type)

    Returns:
        dict: A dictionary containing the formatted chat message.
    """
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
