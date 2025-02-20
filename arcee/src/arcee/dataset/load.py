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

    new_lines = []
    for line in json_lines:
        if not line["event_mentions"]:
            # Keep records with empty event_mentions as-is
            new_lines.append(line)
        else:
            # Make one record per event mention
            new_lines.extend(
                {**copy.deepcopy(line), "event_mentions": [event_mention]}
                for event_mention in line["event_mentions"]
            )

    return new_lines


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


def format_event_type(event_type: str) -> str:
    """
    Formats an event type name into a valid Python class name. The event type name
    is expected to be hierarchical with each level in the hierarchy delimited by
    a '.' or ':'

    Parameters
    ----------
    event_type : str
        The original event type string, which may contain colons (':'), periods ('.'),
        hyphens ('-'), or 'N/a' values.

    Returns
    -------
    str
        A formatted string where:
        - Words are capitalized only on the first letter.
        - 'N/a' is replaced with 'NA'.
        - Hyphens are removed.
        - Colons (':') are replaced with periods ('.'), then split on '.' and
            joined with '_'.

    Examples
    --------
    >>> format_event_type("Event:Attack")
    'Event_Attack'

    >>> format_event_type("naturaldisaster.n/a")
    'NaturalDisaster_NA'
    """
    return "_".join(
        (word[0].upper() + word[1:]).replace("N/a", "NA").replace("-", "")
        for word in event_type.replace(":", ".").split(".")
    )


def pattern_to_class_schema(pattern_file_path: str) -> dict:
    """
    Converts a pattern JSON file into a schema dictionary suitable for Pydantic class
    generation.

    Parameters
    ----------
    pattern_file_path : str
        The file path to the JSON pattern file containing event types and role
        information.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are formatted event type names (valid Python class names).
        - Values are dictionaries mapping role names to their expected data types.
          Each role is assigned either:
            - `"str | None"` for a single occurrence.
            - `"tuple[str | None, str | None, ...]"` for multiple occurrences.

    Notes
    -----
    - The function ensures that event type names are formatted to be valid Python class
        names.
    - Role occurrences determine whether a value is a single string or a tuple of
        strings.

    Examples
    --------
    Given a pattern file containing:

    ```json
    {
        "Event:Attack": {
            "roles": ["Attacker", "Target", "Target"]
        }
    }
    ```

    Running:

    >>> pattern_to_class_schema("pattern.json")

    Returns:

    ```
    {
        "EventAttack": {
            "Attacker": "str | None",
            "Target": "tuple[str | None, str | None]"
        }
    }
    ```
    """
    pattern = load_pattern(pattern_file_path)

    # Recreate the pattern dictionary with the event type names formatted
    # to make them valid as Python Class names (i.e., without ':' or '.'
    # in the name)
    pattern_formatted = {format_event_type(key): pattern[key] for key in pattern}

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
            schema_def = json_schema.get(format_event_type(event_type))
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
