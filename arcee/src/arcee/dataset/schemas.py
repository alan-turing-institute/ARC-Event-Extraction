from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel


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


def pattern_to_json_schema(event_type: str, pattern_record: list[str]) -> dict:
    """
    Converts a pattern record, which is a list of strings of arguments, into a
    JSON schema dictionary

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
