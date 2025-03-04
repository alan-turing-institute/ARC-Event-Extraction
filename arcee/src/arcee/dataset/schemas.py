from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from arcee.dataset.load import load_json


def get_ed_schema(schema: dict) -> dict:
    """
    Returns a JSON schema given a dictionary where all the possible
    event types are the keys.

    Parameters
    ----------
    schema : dict
        A dictionary where the keys are the event types and the values are
        descriptions of the event types.

    Returns
    -------
    dict
        A JSON schema that represents the event types.
    """

    return {
        "type": "array",
        "items": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": False,
            "properties": {key: {"type": "string"} for key in schema},
        },
        "minItems": 0,
        "description": (
            "An array of objects, where each object contains exactly one key-value "
            "pair with predefined keys."
        ),
    }


def get_arg_roles(schema: dict, event_type: str) -> list[str]:
    """
    Retrieves the list of argument roles associated with a given event type.

    Parameters
    ----------
    schema : dict
        A dictionary mapping event types to their respective schemas,
        including argument roles.
    event_type : str
        The event type whose argument roles should be retrieved.

    Returns
    -------
    list of str
        A list of argument roles associated with the specified event type.
        If there are duplicate roles, then the roles are renamed to ensure
        uniqueness, .e.g. "participant_1", "participant_2", etc.

    Raises
    ------
    KeyError
        If the given event_type is not found in the schema.
    """
    if event_type not in schema:
        err_msg = f"Event type '{event_type}' not found in schema."
        raise KeyError(err_msg)

    arg_roles = schema[event_type]["roles"]

    unique_arg_roles = set(arg_roles)
    if len(arg_roles) != len(unique_arg_roles):
        new_arg_roles = []
        for role in unique_arg_roles:
            if arg_roles.count(role) == 1:
                new_arg_roles.append(role)
            else:
                for i in range(arg_roles.count(role)):
                    new_arg_roles.append(role + "_" + str(i + 1))

        return new_arg_roles
    return arg_roles


def pattern_to_json_schema(event_type: str, pattern_record: list[str]) -> dict:
    """
    Converts a pattern record, which is a list of strings of arguments, into a
    JSON schema dictionary.

    Parameters
    ----------
    event_type: str
        The name of the event type
    pattern_record : list[str]
        The pattern record from a pattern file, which is a list of arguments that
        are associated with an event type.

    Returns
    -------
    dict
        A dictionary that represents a valid JSON schema for the event type.

    Notes
    -----
    - The function ensures that event type names are formatted to be valid Python class
        names.
    - Role occurrences determine whether a value is a single string or a tuple of
        strings.

    Examples
    --------
    Given a pattern record:

    ```
    ["participant", "participant", "place"]
    ```

    Running:

    >>> pattern_to_class_schema(pattern_record)

    Returns:

    ```
    {
        "properties": {
            "participant": {
                "maxItems": 2,
                "minItems": 2,
                "prefixItems": [
                    {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    {"anyOf": [{"type": "string"}, {"type": "null"}]},
                ],
                "title": "Participant",
                "type": "array",
            },
            "place": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Place",
            },
        },
        "required": ["participant", "place"],
        "title": event_type,
        "type": "object",
    }
    ```
    """

    role_dict: dict[str, Any] = {}
    for role in set(pattern_record):
        count_role = pattern_record.count(role)
        if count_role == 1:
            role_dict[role] = str | None
        else:
            role_dict[role] = tuple[*(str | None,) * count_role]  # type: ignore[misc]

    attributes = {"__annotations__": role_dict}

    dynamic_class = type(event_type, (BaseModel,), attributes)
    # Cast to Type[BaseModel] to stop mypy complaining
    ModelClass = cast(type[BaseModel], dynamic_class)

    return ModelClass.model_json_schema()


def data_to_argument_schema(
    event_records: list[dict[str, Any]],
    pattern_file_path: str | None = None,
    schema: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Transforms event records into structured argument schemas based on a predefined
    pattern.

    This function processes event records by mapping event mentions to argument roles
    specified in an external pattern schema. It renames duplicate argument roles,
    ensures event arguments are structured according to the schema, and extracts
    relevant arguments for each event mention.

    Args:
        event_records : list of dict
            A list of dictionaries, where each dictionary represents an event record.
            Each record contains various attributes, including `event_mentions`,
            which lists events occurring in the text.
        pattern_file_path : str, optional
            The file path to a JSON file containing predefined event argument schemas.
            The schema maps event types to argument roles and their corresponding
            templates.
            This argument is optional if `schema` is provided.
        schema : dict, optional
            A dictionary containing predefined event argument schemas. This argument is
            optional if `pattern_file_path` is provided.

    Returns:
        list of dict
            A list of transformed event argument records. Each dictionary contains:
            - `evt_id` (str): The event mention ID.
            - `output` (dict): A dictionary mapping argument roles to their respective
            vals.

    Raises:
        TypeError
            If `event_records`  contains non-dictionary elements.
        KeyError
            If an event mention lacks the "event_type" key or if the event type is
            missing from the schema.
    """
    # Validate event_records type
    if not all(isinstance(record, dict) for record in event_records):
        err_msg = "Each element in `event_records` must be a dictionary."
        raise TypeError(err_msg)

    if pattern_file_path is None:
        if schema is None:
            err_msg = "Either `pattern_file_path` or `schema` must be provided."
            raise ValueError(err_msg)
    else:
        schema = load_json(pattern_file_path)

    # List to store the transformed event arguments
    transformed_records = []

    for record in event_records:
        event_mentions = record["event_mentions"]
        # There may be multiple event mentions - one for each event type in the text
        for event_mention in event_mentions:
            event_type = event_mention.get("event_type")
            if event_type is None:
                err_msg = "Missing 'event_type' in event mention."
                raise KeyError(err_msg)
            if event_type not in schema:
                err_msg = f"Event type '{event_type}' not found in schema."
                raise KeyError(err_msg)

            unique_arg_roles = get_arg_roles(schema, event_type)
            original_arg_roles = schema[event_type]["roles"]
            for role in set(original_arg_roles):
                # Rename the argument roles in the event mention if
                # they are duplicated in the schema
                if original_arg_roles.count(role) > 1:
                    role_ind = 1
                    for arg in event_mention["arguments"]:
                        if arg["role"] == role:
                            arg["role"] = role + "_" + str(role_ind)
                            role_ind += 1
            json_schema = pattern_to_json_schema(event_type, unique_arg_roles)
            # Get definition of event type schema
            properties = json_schema.get("properties", {})
            new_record = {
                "evt_id": event_mention.get("id"),
                "output": {},
            }
            # Populate new record with event arguments according to schema
            for prop, _ in properties.items():
                # All properties should be scalar (not arrays),
                # take the first matching argument or None.
                value = next(
                    (
                        arg.get("text")
                        for arg in event_mention.get("arguments", [])
                        if arg.get("role") == prop
                    ),
                    None,
                )
                new_record["output"][prop] = value

            transformed_records.append(new_record)

    return transformed_records


def data_to_ed_schema(event_records: list[dict[str, Any]]) -> list[dict]:
    """
    This takes a list of event records and transforms them according to a basic
    schema where the key is the event type and the value is the trigger word(s).

    Parameters
    ----------
    event_records : List[Dict[str, Any]]
        A list of event records. Each record is expected to have the form:
            {
                "wnd_id": "wnd1",
                "event_mentions": [
                    {
                        "event_type": "<event type>",
                        "trigger": {"text": <str>, "start": <int>, "end": <int>},
                        "arguments": [
                            {"entity_id": <str>, "role": <str>, "text": <str>,
                            "start": <int>, "end": <int>},
                            ...
                        ]
                    },
                    ...
                ]
            }

    list of dict
        A list of transformed event argument records. Each dictionary contains:
        - `wnd_id` (str): The window ID of the event record.
        - `evt_id` (str): The event mention ID.
        - `output` (dict): A dictionary mapping event types to trigger words.
    """
    # List to store the transformed event types
    transformed_records = []
    # For each record in the list, add the event types and triggers to a new dict
    # and append to list of transformed records.
    for record in event_records:
        event_mentions = record["event_mentions"]
        output = []
        for event_mention in event_mentions:
            event_type = event_mention.get("event_type")
            trigger = event_mention["trigger"]["text"]
            output.append({event_type: trigger})
        transformed_records.append(
            {
                "wnd_id": record["wnd_id"],
                "output": output,
            }
        )

    return transformed_records


def get_ed_json_schema(schema: dict) -> dict:
    """
    Returns a JSON schema given a dictionary where all the possible
    event types are the keys.

    Parameters
    ----------
    schema : dict
        A dictionary where the keys are the event types and the values are
        descriptions of the event types.

    Returns
    -------
    dict
        A JSON schema that represents the event types.
    """

    return {
        "type": "array",
        "items": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": False,
            "properties": {key: {"type": "string"} for key in schema},
        },
        "minItems": 0,
        "description": (
            "An array of objects, where each object contains exactly one key-value "
            "pair with predefined keys."
        ),
    }
