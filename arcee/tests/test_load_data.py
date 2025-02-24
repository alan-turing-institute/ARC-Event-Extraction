from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest

from arcee.dataset.load import load_data, load_json


def write_temp_json_file(data: list[dict[str, Any]]) -> str:
    """Helper function to write test data to a temporary JSON file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    with open(temp_file.name, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")  # Write each dictionary as a JSON line
    return temp_file.name


def test_load_data(test_event_records):
    """
    Test that load_data correctly loads JSON data when record_per_event=False.
    i.e., one record is loaded per record in the original dataset
    """
    file_path = write_temp_json_file(test_event_records)
    loaded_data = load_data(file_path, record_per_event=False)

    assert isinstance(loaded_data, list)
    assert len(loaded_data) == len(
        test_event_records
    )  # Should load 3 JSON objects as-is

    for original, loaded in zip(test_event_records, loaded_data, strict=False):
        assert original == loaded  # Ensure the loaded data matches the input


def test_load_data_one_record_per_event(test_event_records):
    """
    Test that load_data correctly splits event_mentions when record_per_event=True.
    i.e., one record is loaded per event and there may be multiple or no events in
    each input record - if there are no events then this should still be loaded as
    a single record.
    """
    file_path = write_temp_json_file(test_event_records)
    loaded_data = load_data(file_path, record_per_event=True)

    # Expected number of records:
    # - doc_001 (0, not included)
    # - doc_002 (1 event mention → unchanged)
    # - doc_003 (2 event mentions → split into 2 records)
    # - doc_004 (0, not included)
    # - doc_005 (1 event mention → unchanged)
    expected_num_records = 0 + 1 + 2 + 0 + 1  # Total: 6
    assert isinstance(loaded_data, list)
    assert len(loaded_data) == expected_num_records

    # Ensure first doc is not loaded and the second doc is loaded first
    assert loaded_data[0]["doc_id"] == "doc_002"
    assert len(loaded_data[0]["event_mentions"]) == 1

    # Ensure second and third records are correctly split
    assert loaded_data[1]["doc_id"] == "doc_003"
    assert len(loaded_data[1]["event_mentions"]) == 1
    assert loaded_data[1]["event_mentions"][0]["event_subtype"] == "Earthquake"

    assert loaded_data[2]["doc_id"] == "doc_003"
    assert len(loaded_data[2]["event_mentions"]) == 1
    assert loaded_data[2]["event_mentions"][0]["event_subtype"] == "Collapse"


def test_load_json_valid(pattern_data):
    """Test that load_json correctly loads a valid JSON file."""
    file_path = write_temp_json_file(pattern_data)
    loaded_data = load_json(file_path)

    assert isinstance(loaded_data, dict)
    assert loaded_data == pattern_data[0]  # Ensure the loaded data matches the input


def test_load_json_empty():
    """Test that load_json correctly handles an empty JSON file."""
    file_path = write_temp_json_file([{}])  # Write an empty dictionary
    loaded_data = load_json(file_path)

    assert isinstance(loaded_data, dict)
    assert loaded_data == {}  # Expect an empty dictionary


def test_load_json_malformed():
    """Test that load_json raises JSONDecodeError for a malformed JSON file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    with open(temp_file.name, "w") as f:
        f.write("{invalid_json: true")  # Write malformed JSON

    with pytest.raises(json.JSONDecodeError):
        load_json(temp_file.name)


def test_load_json_file_not_found():
    """Test that load_json raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_json("non_existent_file.json")
