from __future__ import annotations

import json
from ast import literal_eval
from typing import Any


def load_data(dataset_file_path: str) -> list[Any]:
    """
    This function loads either the train, dev or test dataset from the specified
    location
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


def load_pattern(pattern_file_path: str, dataset_name: str) -> dict:
    """
    To do
    """
    with open(pattern_file_path) as file:
        data = file.read()
    return literal_eval(data)[dataset_name]
