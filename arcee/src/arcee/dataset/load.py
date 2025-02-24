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


def data_to_argument_schema(
    event_records: list[dict[str, Any]], json_schema: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Transforms a list of event records into the structure defined by a JSON schema
    for event types and arguments. This will be used to transform the gold labelled
    data into the same format as output by the LLMs.

    This function takes a list of event records and a dictionary representing the
    JSON schema. Each record is expected to be in the format of the data processed
    by TextEE with an "event_mentions" key.
    Each "event_mentions" value contains an "event_type" and a list of "arguments"
    (each with a role and text)

    For each property in the schema:
      - If the property is defined as an array (e.g., the "participant" field), the
        function collects all argument texts with a matching role.
      - If the property is a scalar (string or null), the function assigns the text
        of the first matching argument, or None if there is no match.

    The function outputs list of records whose keys correspond to the properties
    defined in the JSON schema for that event type.

    Parameters
    ----------
    event_records : List[Dict[str, Any]]
        A list of event records. Each record is expected to have the form:
            {
                "event_type": "<event type>",
                "trigger": {"text": <str>, "start": <int>, "end": <int>},
                "arguments": [
                    {"entity_id": <str>, "role": <str>, "text": <str>, "start": <int>,
                    "end": <int>},
                    ...
                ]
            }
    json_schema : Dict[str, Any]
        A dictionary representing the JSON schema. The keys in this dictionary are event
        type names and each value is a JSON schema definition (including a "properties"
        field) for that event type.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries formatted according to the JSON schema for the
        corresponding event types.
        For example:
            [
                {
                    "participant": [
                        "Alice",
                        "Bob"
                    ],
                    "place": null
                },
                {
                    "target": null,
                    "attacker": "she",
                    "instrument": "intervention",
                    "place": null
                },
            ]

    Raises
    ------
    ValueError
        If an event type in a data record does not match an event type in the given
        schema.
    """
    # List to store the transformed event arguments
    transformed_records = []

    for record in event_records:
        event_mentions = record["event_mentions"]
        # There may be multiple event mentions - one for each event type in the text
        for event_mention in event_mentions:
            event_type = event_mention.get("event_type")
            # Have to format the event type to make sure it matches the  schema
            schema_def = json_schema.get(event_type)
            if schema_def is None:
                # If the event_type is not defined in the schema, raise error
                err_msg = event_type + " is not defined in the given schema."
                raise ValueError(err_msg)
            # Get definition of event type schema
            properties = schema_def.get("properties", {})
            new_record = {}
            # Populate new record with event arguments according to schema
            for prop, prop_def in properties.items():
                if prop_def.get("type") == "array":
                    # Collect all matching texts for array-type properties.
                    values = [
                        arg.get("text")
                        for arg in event_mention.get("arguments", [])
                        if arg.get("role") == prop
                    ]
                    new_record[prop] = values if values else None
                else:
                    # For scalar properties, take the first matching argument or None.
                    value = next(
                        (
                            arg.get("text")
                            for arg in event_mention.get("arguments", [])
                            if arg.get("role") == prop
                        ),
                        None,
                    )
                    new_record[prop] = value

            transformed_records.append(new_record)

    return transformed_records


def data_to_ed_schema(event_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    This takes a list of event records and transforms them according to a basic
    schema where the key is the event type and the value is the trigger word(s).

    Parameters
    ----------
    event_records : List[Dict[str, Any]]
        A list of event records. Each record is expected to have the form:
            {
                "event_type": "<event type>",
                "trigger": {"text": <str>, "start": <int>, "end": <int>},
                "arguments": [
                    {"entity_id": <str>, "role": <str>, "text": <str>, "start": <int>,
                    "end": <int>},
                    ...
                ]
            }

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries formatted according to the JSON schema for the
        corresponding event types.
        For example:
            [
                {
                    "Transport:Movement": "drove",
                    "Life:Die": "died"
                }
                {
                    "Transaction:Transfer-Money": "gave"
                }
                ...
            ]
    """
    # List to store the transformed event types
    transformed_records = []
    # For each record in the list, add the event types and triggers to a new dict
    # and append to list of transformed records.
    for record in event_records:
        event_mentions = record["event_mentions"]
        new_record = {}
        for event_mention in event_mentions:
            event_type = event_mention.get("event_type")
            trigger = event_mention["trigger"]["text"]
            new_record[event_type] = trigger
        transformed_records.append(new_record)

    return transformed_records
