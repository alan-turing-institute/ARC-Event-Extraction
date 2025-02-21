from __future__ import annotations

from typing import cast

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

    role_dict = {}
    for role in set(pattern_record):
        count_role = pattern_record.count(role)
        role_dict[role] = create_fixed_length_tuple_type(count_role)

    attributes = {"__annotations__": role_dict}

    dynamic_class = type(event_type, (BaseModel,), attributes)
    # Cast to Type[BaseModel] to stop mypy complaining
    ModelClass = cast(type[BaseModel], dynamic_class)

    return ModelClass.model_json_schema()


def create_fixed_length_tuple_type(count: int) -> object:
    """
    Dynamically construct a fixed-length tuple type consisting of `count` elements,
    where each element is `str | None`.

    Parameters
    ----------
    count : int
        The number of elements in the tuple. Each element is `str | None`.

    Returns
    -------
    object
        If `count` is 1, returns the union type `str | None`.
        Otherwise returns a built-in tuple type, e.g. `tuple[str | None, str | None]`.

    Raises
    ------
    ValueError
        If ``count < 1``, indicating an invalid tuple size.

    Examples
    --------
    >>> create_fixed_length_tuple_type(1)
    str | None

    >>> create_fixed_length_tuple_type(2)
    tuple[str | None, str | None]

    Notes
    -----
    - This function uses `eval` internally to build the type string, which is
      generally safe as long as `count` remains an integer from a trusted source.
    - Mypy and other static type checkers usually cannot infer types from dynamically
      created type objects.
    """
    if count < 1:
        err_msg = ValueError(f"'count' must be >= 1, got {count}.")
        raise err_msg

    if count == 1:
        return str | None

    # Build something like "tuple[str | None, str | None, str | None]" for count=3
    joined = ", ".join(["str | NoneType"] * count)
    type_str = f"tuple[{joined}]"

    # Evaluate the constructed string in a controlled environment
    local_ns = {
        "str": str,
        "NoneType": type(None),
    }

    return eval(type_str, {}, local_ns)
