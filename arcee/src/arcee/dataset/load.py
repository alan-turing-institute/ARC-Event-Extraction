from __future__ import annotations

import copy
import json
from typing import Any


def load_data(dataset_file_path: str, record_per_event: bool = False) -> list[Any]:
    """
    This function loads a dataset from the specified location
    The dataset is expected to be in JSON format

    Parameters
    ----------
    dataset_file_path: str
        The file path where the json files are stored for the data, this should
        include the file name and extension,
        e.g. "data/RAMS/output/splits/split1/test.json"
    record_per_event: bool
        If set to True, then it will load the records so that there is one event
        type per record (in order to do argument extraction per event type)

    Returns
    -------
    list[Any]
        A list of dictionaries, with each dictionary representing a record in
        the dataset
    """
    with open(dataset_file_path) as file:
        json_lines = [json.loads(line) for line in file]

    if not record_per_event:
        return json_lines

    new_lines: list[Any] = []
    for line in json_lines:
        # Make one record per event mention, ignores any records without events
        new_lines.extend(
            {**copy.deepcopy(line), "event_mentions": [event_mention]}
            for event_mention in line["event_mentions"]
        )

    return new_lines


def load_json(pattern_file_path: str) -> dict:
    """
    This function loads a json file from the specified location
    The file is expected to be in JSON format

    Parameters
    ----------
    pattern_file_path: str
        The file path where the pattern file is stored for the dataset, this should
        include the file name and extension, expected to be JSON format
        e.g. "data/RAMS/patterns/pattern.json"

    Returns
    -------
    dict
        A dictionary containing the schema of event types and argument roles.
    """
    with open(pattern_file_path) as file:
        return json.load(file)
