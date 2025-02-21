from __future__ import annotations

import pytest

from arcee.dataset.schemas import (
    create_fixed_length_tuple_type,
    format_event_type,
    pattern_to_json_schema,
)


@pytest.mark.parametrize(
    ("input_event_type", "expected"),
    [
        ("Event:Attack", "Event_Attack"),
        ("naturaldisaster.n/a", "Naturaldisaster_NA"),
        ("some.event.hierarchy", "Some_Event_Hierarchy"),
        ("NaturalDisaster.n/a", "NaturalDisaster_NA"), 
        ("multi.colon:event-type", "Multi_Colon_Eventtype"),
    ],
)
def test_format_event_type(input_event_type, expected):
    """
    Test various transformations in format_event_type.
    """
    result = format_event_type(input_event_type)
    assert result == expected


def test_create_fixed_length_tuple_type_count_less_than_one():
    """
    Ensure create_fixed_length_tuple_type raises ValueError for invalid count < 1.
    """
    with pytest.raises(ValueError, match="must be >= 1") as exc_info:
        create_fixed_length_tuple_type(0)  # count < 1
    assert "must be >= 1" in str(exc_info.value)


def test_create_fixed_length_tuple_type_count_eq_one():
    """
    For count == 1, it should return the union type: str | None.
    """
    result = create_fixed_length_tuple_type(1)
    assert str(result) == "str | None"


def test_create_fixed_length_tuple_type_count_eq_two():
    """
    For count == 2, returns tuple[str | None, str | None].
    """
    result = create_fixed_length_tuple_type(2)
    # Check the string representation
    # In Python 3.12.8, str(result) might be "tuple[str | NoneType, str | NoneType]"
    # We do a partial substring check:
    assert "tuple" in str(result)
    assert "str | None" in str(result)


def test_pattern_to_class_schema_single_role():
    """
    Test pattern_to_class_schema with a single role in the pattern record.
    """
    schema = pattern_to_json_schema("TestEvent", ["place"])
    assert schema["title"] == "TestEvent"
    assert schema["type"] == "object"

    # "place" should be a field with anyOf -> [string, null]
    props = schema["properties"]
    assert "place" in props
    assert "anyOf" in props["place"]
    assert {"type": "string"} in props["place"]["anyOf"]
    assert {"type": "null"} in props["place"]["anyOf"]

    # "place" should be required
    assert "required" in schema
    assert "place" in schema["required"]


def test_pattern_to_json_schema_duplicated_roles():
    """
    Test pattern_to_class_schema with repeated roles.
    """
    schema = pattern_to_json_schema("MyEvent", ["participant", "participant", "place"])
    assert schema["title"] == "MyEvent"
    assert schema["type"] == "object"

    props = schema["properties"]
    # We expect 2 properties: "participant" and "place"
    assert len(props) == 2

    # participant should be an array with exactly 2 items
    participant_prop = props["participant"]
    assert participant_prop["type"] == "array"
    # minItems and maxItems should be 2
    assert participant_prop["minItems"] == 2
    assert participant_prop["maxItems"] == 2
    # prefixItems should have 2 elements
    prefix_items = participant_prop["prefixItems"]
    assert len(prefix_items) == 2
    # each prefixItem -> anyOf with [string, null]
    for item in prefix_items:
        assert "anyOf" in item
        assert {"type": "string"} in item["anyOf"]
        assert {"type": "null"} in item["anyOf"]

    # place should be anyOf -> [string, null]
    place_prop = props["place"]
    assert "anyOf" in place_prop
    assert {"type": "string"} in place_prop["anyOf"]
    assert {"type": "null"} in place_prop["anyOf"]

    # required fields
    required_fields = schema["required"]
    # We expect participant and place to both be required
    assert set(required_fields) == {"participant", "place"}

