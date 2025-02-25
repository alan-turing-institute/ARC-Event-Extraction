from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel


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
