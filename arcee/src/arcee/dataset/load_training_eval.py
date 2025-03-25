from __future__ import annotations

import json

import numpy as np
from datasets import Dataset, DatasetDict

import arcee.prompts.prompts as prompts
from arcee.dataset.load import load_data
from arcee.dataset.schemas import data_to_argument_schema, data_to_ed_schema


def sample_and_shuffle_null_events(
    dataset: Dataset, split: str, rng: np.random.default_rng, null_event_ratio: float
) -> Dataset:
    positive_events = dataset.filter(lambda x: len(x["event_mentions"]) > 0)
    num_positive = len(positive_events[split])

    if null_event_ratio > 0:
        null_events = dataset.filter(lambda x: len(x["event_mentions"]) == 0)
        num_null_samples = int(num_positive * null_event_ratio)

        if num_null_samples > 0 and len(null_events[split]) > 0:
            sampled_indices = rng.choice(
                len(null_events[split]),
                size=min(num_null_samples, len(null_events[split])),
                replace=False,
            )
            sampled_null_events = null_events[split].select(sampled_indices)
            combined_records = (
                positive_events[split].to_list() + sampled_null_events.to_list()
            )
            shuffled_indices = rng.permutation(len(combined_records))
            return Dataset.from_list([combined_records[i] for i in shuffled_indices])
        shuffled_indices = rng.permutation(len(positive_events[split]))
        return Dataset.from_list([positive_events[split][i] for i in shuffled_indices])
    shuffled_indices = rng.permutation(len(positive_events[split]))
    return Dataset.from_list([positive_events[split][i] for i in shuffled_indices])


def load_training_eval_data(
    training_data_path: str,
    eval_data_path: str,
    prompt_cls: str,
    schema_path: str,
    only_events: bool = False,
    null_event_ratio: float = 0,
    null_event_seed: int = 42,
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
    null_event_ratio: float
        The ratio of null events to include in the dataset, this is expected to be
        between 0 and 1 and will only be considered if only_events is set to True. The
        ratio is the fraction of null events, with respect to the number of positive
        events to include in the dataset.
    null_event_seed: int
        The seed to use for sampling the null events

    Returns:
    Dataset
        A dataset object with the formatted training and evaluation data
    """
    with open(schema_path) as file:
        schema = json.load(file)

    prompt_constructor = getattr(prompts, prompt_cls)(schema)
    arg_extraction = prompt_constructor.get_is_arg_extraction()

    train_dataset = load_data(training_data_path, arg_extraction)
    eval_dataset = load_data(eval_data_path, arg_extraction)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_dataset),
            "eval": Dataset.from_list(eval_dataset),
        }
    )

    if only_events:
        # If doing event detection, this might sometimes be wanted as there may be
        # an imbalanced dataset.
        rng = np.random.default_rng(null_event_seed)

        dataset["train"] = sample_and_shuffle_null_events(
            dataset, "train", rng, null_event_ratio
        )
        dataset["eval"] = sample_and_shuffle_null_events(
            dataset, "eval", rng, null_event_ratio
        )

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
