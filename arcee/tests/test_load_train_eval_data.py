from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import DatasetDict

from arcee.dataset.load_training_eval import (
    format_chat_message,
    load_training_eval_data,
)
from arcee.prompts.prompts import EAEPrompter1, EDPrompter1


@pytest.mark.parametrize(
    ("arg_extraction", "prompter_class"),
    [
        (False, EDPrompter1),  # event detection mode
        (True, EAEPrompter1),  # argument extraction mode
    ],
    ids=["event_detection", "argument_extraction"],
)
def test_format_chat_message(test_event_records, arg_extraction, prompter_class):
    """
    Test that format_chat_message produces the expected structure
    (a list of messages with 'user' and 'assistant') and that the
    JSON response is well-formed for ED (event detection) or EA (argument extraction).
    """

    if arg_extraction:
        fake_schema = {
            "Conflict:Demonstrate": {"roles": ["Demonstrator", "Audience"]},
            "Disaster:Earthquake": {"roles": ["Location"]},
            "Disaster:Collapse": {"roles": ["Location"]},
            "Transaction:TransferOwnership": {"roles": ["Giver", "Receiver", "Item"]},
        }
    else:
        fake_schema = {
            "Conflict:Demonstrate": "a demonstration",
            "Disaster:Earthquake": "an earthquake",
            "Disaster:Collapse": "a collapse",
            "Transaction:TransferOwnership": "a transfer of ownership",
        }

    # Instantiate the prompt constructor class
    prompter = prompter_class(fake_schema)

    # Get a test record
    record = test_event_records[2]

    result = format_chat_message(
        data_record=record,
        prompt_constructor=prompter,
        schema=fake_schema,
        arg_extraction=arg_extraction,
    )

    # Verify the structure of returned messages
    assert "messages" in result
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"

    # Check that "assistant" message is JSON enclosed in backticks:
    assistant_content = result["messages"][1]["content"]
    # Expecting: ```json\n{...}\n```
    stripped_json = assistant_content.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(stripped_json)  # Raises if invalid
    if arg_extraction:
        assert isinstance(
            parsed, dict
        ), "Expected a JSON dict object in the assistant message."
    else:
        assert isinstance(
            parsed, list
        ), "Expected a JSON array object in the assistant message."


@pytest.fixture
def fake_ed_schema_json(tmp_path) -> Path:
    """
    Creates a temporary JSON schema file and returns its Path.
    """
    schema_path = tmp_path / "ed_schema.json"
    fake_schema = {
        "Conflict:Demonstrate": "a demonstration",
        "Disaster:Earthquake": "an earthquake",
        "Disaster:Collapse": "a collapse",
        "Transaction:TransferOwnership": "a transfer of ownership",
    }
    schema_path.write_text(json.dumps(fake_schema))
    return schema_path


@pytest.fixture
def fake_eae_schema_json(tmp_path) -> Path:
    """
    Creates a temporary JSON schema file and returns its Path.
    """
    schema_path = tmp_path / "eae_schema.json"
    fake_schema = {
        "Conflict:Demonstrate": {"roles": ["Demonstrator", "Audience"]},
        "Disaster:Earthquake": {"roles": ["Location"]},
        "Disaster:Collapse": {"roles": ["Location"]},
        "Transaction:TransferOwnership": {"roles": ["Giver", "Receiver", "Item"]},
    }
    schema_path.write_text(json.dumps(fake_schema))
    return schema_path


@pytest.fixture
def fake_data_files(tmp_path, test_event_records):
    """
    Creates two temporary JSON files (train.json, eval.json) containing
    the test_event_records. Returns their paths.
    """
    train_path = tmp_path / "train.json"
    eval_path = tmp_path / "eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(test_event_records, f, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(test_event_records, f, indent=2)

    return train_path, eval_path


@pytest.mark.parametrize(
    ("arg_extraction", "only_events", "prompt_cls_str"),
    [
        (False, False, "EDPrompter1"),  # event detection
        (False, True, "EDPrompter1"),  # event detection
        (True, False, "EAEPrompter1"),  # argument extraction
    ],
    ids=["event_detection", "event_detection_only_events", "argument_extraction"],
)
def test_load_training_eval_data(
    fake_data_files,
    fake_ed_schema_json,
    fake_eae_schema_json,
    arg_extraction,
    only_events,
    prompt_cls_str,
):
    """
    Test that load_training_eval_data returns a DatasetDict with train/eval splits
    and that records are transformed into the expected messages structure.
    """
    train_path, eval_path = fake_data_files

    if arg_extraction:
        schema_path = str(fake_eae_schema_json)
    else:
        schema_path = str(fake_ed_schema_json)

    dataset = load_training_eval_data(
        training_data_path=str(train_path),
        eval_data_path=str(eval_path),
        prompt_cls=prompt_cls_str,
        schema_path=schema_path,
        only_events=only_events,
    )

    # Check that a DatasetDict with "train" and "eval" splits was returned
    assert isinstance(dataset, DatasetDict)
    assert set(dataset.keys()) == {"train", "eval"}

    # Each entry in each split should now contain 'messages'
    for split_name in ["train", "eval"]:
        for item in dataset[split_name]:
            assert "messages" in item
            assert len(item["messages"]) == 2
            assert item["messages"][0]["role"] == "user"
            assert item["messages"][1]["role"] == "assistant"

            # Optionally parse the JSON to ensure it's valid
            assistant_content = item["messages"][1]["content"]
            stripped_json = (
                assistant_content.replace("```json", "").replace("```", "").strip()
            )
            parsed = json.loads(stripped_json)
            if arg_extraction:
                assert isinstance(
                    parsed, dict
                ), "Expected a JSON object from the prompt."
            else:
                assert isinstance(
                    parsed, list
                ), "Expected a JSON array from the prompt."
