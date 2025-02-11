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
        include the file name and extension, expected to be JSON format
        e.g. "data/RAMS/patterns/pattern.josn"

    Returns
    -------
    dict
        A dictionary containing the schema of event types and argument roles.
    """
    with open(pattern_file_path) as file:
        return json.load(file)


def load_schema(pattern_file_path: str) -> dict:
    """
    Converts the JSON in the pattern file into a dictionary representing a schema for
    the dataset. The dictionary will contain keys that are the event type names where
    the words in the hierarchy are split by an underscore '_'. The values are dicts
    where the keys are the role names and the values are the permitted types, which
    will either by a 'str | None' or a tuple of 'str | None' values.

    Parameters
    ----------
    pattern_file_path: str
        The file path where the pattern file is stored for the dataset, this should
        include the file name and extension, expected to be JSON format
        e.g. "data/RAMS/patterns/pattern.json"

    Returns
    -------
    dict
        A dictionary containing the schema
    """
    pattern = load_pattern(pattern_file_path)

    # Split the words making up the event type hierarchy with an underscore '_'
    # and capitalise each first letter
    pattern_formatted = {
        "_".join(
            word.capitalize().replace("N/a", "NA").replace("-", "")
            for word in key.replace(":", ".").split(".")
        ): pattern[key]
        for key in pattern
    }

    # Create a schema dictionary - each key is the class name and the values
    # are dictionaries with the key being the role name and the vlaue being
    # the expected type (a 'str' or a tuple of 'str')
    schema = {}  # type: dict[str, dict]
    for key in pattern_formatted:
        schema[key] = {}
        roles = pattern_formatted[key]["roles"]
        for role in set(roles):
            count_role = roles.count(role)
            if count_role == 1:
                schema[key][role] = "str | None"
            else:
                str_types = "str | None"
                while count_role > 1:
                    str_types += ", str | None"
                    count_role -= 1
                schema[key][role] = "tuple[" + str_types + "]"

    return schema
