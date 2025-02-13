from __future__ import annotations

import json
from typing import Any


def format_event_type(event_type: str) -> str:
    """
    This function takes a event type that is defined in a hierarchy where the
    different levels of the hierarchy are divided by either '.' or ':'. It transforms
    the event type string so that each level of the hierarchy is divided by '_' and
    so that any mention of 'N/a' is replaced with 'NA'

    Parameters
    ----------
    event_type: str
        The name of the event type from the dataset

    Returns
    -------
    str
        The re-formatted event type string
    """
    return "_".join(
        word.capitalize().replace("N/a", "NA").replace("-", "")
        for word in event_type.replace(":", ".").split(".")
    )


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
