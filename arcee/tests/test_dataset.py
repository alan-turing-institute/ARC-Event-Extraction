from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from arcee.dataset.dataset import EEDataset


@pytest.fixture
def mock_data():
    """A small sample list of dicts to represent event data."""
    return [
        {"id": 1, "text": "Event text 1", "event_type": "type_A"},
        {"id": 2, "text": "Event text 2", "event_type": "type_B"},
    ]


@pytest.fixture
def mock_prompt_constructor():
    """
    Create a mock or stub EEPrompt-like object with a known return value.
    The get_prompt method returns a tuple of (prompt, output_format, record_ids,
    input_text).
    """
    mock_prompt = MagicMock()
    mock_prompt.get_prompt.side_effect = lambda data: (
        f"Prompt for data ID {data['id']}",
        "mock_output_format",
        {"record_id": data["id"]},
        f"Input text: {data['text']}",
    )
    return mock_prompt


class TestEEDataset:
    def test_init(self, mock_data, mock_prompt_constructor):
        """
        Test that EEDataset is initialized with given data and prompt_constructor.
        """
        dataset = EEDataset(data=mock_data, prompt_constuctor=mock_prompt_constructor)
        assert len(dataset.data) == len(mock_data)
        assert dataset.prompt_constuctor is mock_prompt_constructor

    def test_len(self, mock_data, mock_prompt_constructor):
        """
        Test that __len__ returns the length of the internal data list.
        """
        dataset = EEDataset(data=mock_data, prompt_constuctor=mock_prompt_constructor)
        assert len(dataset) == len(mock_data)

    def test_getitem(self, mock_data, mock_prompt_constructor):
        """
        Test that __getitem__ returns the correctly shaped tuple and that
        prompt_constructor.get_prompt is called with the correct piece of data.
        """
        dataset = EEDataset(data=mock_data, prompt_constuctor=mock_prompt_constructor)
        item = dataset[0]
        assert isinstance(item, tuple), "getitem should return a tuple"
        assert len(item) == 4, "getitem should return a 4-tuple"

        user_message, output_format, record_ids, input_text = item
        assert user_message["role"] == "user"
        assert "content" in user_message
        assert output_format == "mock_output_format"
        assert record_ids == {"record_id": mock_data[0]["id"]}
        assert "Event text 1" in input_text

        mock_prompt_constructor.get_prompt.assert_any_call(data=mock_data[0])

    def test_from_file_integration(self, mock_prompt_constructor):
        """
        Create a temporary JSONL file, write test data to it, and confirm that
        from_file properly loads the data into EEDataset.
        """
        # Example data to be written to the JSONL file
        temp_data = [
            {"id": 100, "text": "Integration test event 1", "event_type": "type_X"},
            {"id": 101, "text": "Integration test event 2", "event_type": "type_Y"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path like "dummy/path.jsonl" in the temp directory
            data_path = os.path.join(tmpdir, "dummy.jsonl")

            # Write each record as a JSON line
            with open(data_path, "w", encoding="utf-8") as f:
                for record in temp_data:
                    f.write(json.dumps(record) + "\n")

            # Instantiate EEDataset using from_file
            dataset = EEDataset.from_file(
                data_path=data_path,
                prompt_constuctor=mock_prompt_constructor,
                only_events=False,
            )

            # Verify we got the expected number of records
            assert len(dataset) == len(temp_data)

            # Spot-check that the dataset content matches what we wrote
            # (the prompt_constructor is a mock, so we just check data correctness here)
            for i in range(len(dataset)):
                user_message, output_format, record_ids, input_text = dataset[i]
                assert record_ids["record_id"] == temp_data[i]["id"]
                assert temp_data[i]["text"] in input_text

            # Check that our prompt_constructor was indeed used
            assert mock_prompt_constructor.get_prompt.call_count == len(temp_data)
