from __future__ import annotations

import json
from typing import Any


def load_data(dataset_file_path: str) -> list[Any]:
    """
    This function loads a dataset from the specified location
    The dataset is expected to be in JSON format

    Parameters
    ----------
    dataset_file_path: str
        The file path where the json files are stored for the data, this should
        include the file name and extension,
        e.g. "data/RAMS/output/splits/split1/test.json"

    Returns
    -------
    list[Any]
        A list of dictionaries, with each dictionary representing a record in
        the dataset
    """
    with open(dataset_file_path) as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]


def load_pattern(pattern_file_path: str) -> dict:
    """
    This function loads a schema for a given dataset
    The file is expected to be in txt format

    Parameters
    ----------
    pattern_file_path: str
        The file path where the pattern file is stored for the dataset, this should
        include the file name and extension,
        e.g. "data/RAMS/patterns/pattern.txt"

    Returns
    -------
    dict
        A dictionary containing the schema of event types and argument roles.
    """
    with open(pattern_file_path) as file:
        return json.load(file)
