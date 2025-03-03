from __future__ import annotations

import json
import tempfile

import pytest

from arcee.dataset.schemas import get_arg_roles, pattern_to_json_schema
from arcee.prompts.prompts import (
    EAEPrompter1,
    EAEPrompter2,
    EDPrompter1,
    EDPrompter2,
    EDPrompter3,
    EDPrompter4,
)

DATA = {
    "doc_id": "doc1",
    "wnd_id": "wnd1",
    "text": "The explosion destroyed the building.",
    "tokens": ["The", "explosion", "destroyed", "the", "building", "."],
    "event_mentions": [
        {"id": "ev1", "trigger": {"start": 2, "end": 3}, "event_type": "Explosion"}
    ],
}

ED_SCHEMA = {
    "Explosion": "An event where something bursts violently.",
    "Attack": "A violent physical act causing harm or damage.",
}
JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "minProperties": 1,
        "maxProperties": 1,
        "additionalProperties": False,
        "properties": {key: {"type": "string"} for key in ED_SCHEMA},
    },
    "minItems": 0,
    "description": (
        "An array of objects, where each object contains exactly one key-value "
        "pair with predefined keys."
    ),
}
EAE_SCHEMA = {
    "Explosion": {"roles": ["Attacker", "Victim", "Location"]},
    "Attack": {"roles": ["Attacker", "Target", "Location"]},
}

EXAMPLE_RECORDS = [
    {
        "doc_id": "doc1",
        "wnd_id": "wnd1",
        "text": "The explosion destroyed the building.",
        "tokens": ["The", "explosion", "destroyed", "the", "building", "."],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "explosion", "start": 2, "end": 3},
                "event_type": "Explosion",
            }
        ],
    },
    {
        "doc_id": "doc2",
        "wnd_id": "wnd2",
        "text": "A fire broke out in the warehouse.",
        "tokens": ["A", "fire", "broke", "out", "in", "the", "warehouse", "."],
        "event_mentions": [
            {
                "id": "ev2",
                "trigger": {"text": "fire", "start": 1, "end": 2},
                "event_type": "Fire",
            }
        ],
    },
    {
        "doc_id": "doc3",
        "wnd_id": "wnd3",
        "text": "The hurricane caused severe damage.",
        "tokens": ["The", "hurricane", "caused", "severe", "damage", "."],
        "event_mentions": [
            {
                "id": "ev3",
                "trigger": {"text": "hurricane", "start": 1, "end": 2},
                "event_type": "Hurricane",
            }
        ],
    },
    {
        "doc_id": "doc4",
        "wnd_id": "wnd4",
        "text": "An earthquake shook the city last night.",
        "tokens": ["An", "earthquake", "shook", "the", "city", "last", "night", "."],
        "event_mentions": [
            {
                "id": "ev4",
                "trigger": {"text": "earthquake", "start": 1, "end": 2},
                "event_type": "Earthquake",
            }
        ],
    },
    {
        "doc_id": "doc5",
        "wnd_id": "wnd5",
        "text": "The protest escalated into a riot.",
        "tokens": ["The", "protest", "escalated", "into", "a", "riot", "."],
        "event_mentions": [
            {
                "id": "ev5",
                "trigger": {"text": "protest", "start": 1, "end": 2},
                "event_type": "Protest",
            }
        ],
    },
]

### TESTING get_arg_roles ###


def test_get_arg_roles_valid():
    """Test _get_arg_roles returns correct argument roles."""
    assert get_arg_roles(EAE_SCHEMA, "Explosion") == [
        "Attacker",
        "Victim",
        "Location",
    ]


def test_get_arg_roles_invalid():
    """Test _get_arg_roles raises KeyError for unknown event type."""
    with pytest.raises(KeyError, match="Event type 'Flood' not found in schema."):
        get_arg_roles(EAE_SCHEMA, "Flood")


### TESTING EDPrompter1 ###
class TestEDPrompter1:
    def setup_method(self):
        """Setup a subclass instance for testing since EEPrompt is abstract"""
        self.prompter = EDPrompter1()

    def test_get_system_content(self):
        """Test _get_system_content returns expected system message."""
        expected_content = (
            "You are an assistant that extracts event triggers and their associated "
            "event types from input text. Many triggers are verbs and represent a "
            "change of state or action in the text.\n"
            "---\n"
        )
        assert self.prompter._get_system_content() == expected_content

    def test_get_event_trigger_valid(self):
        """Test _get_event_trigger extracts correct trigger details."""
        event_mention = {"trigger": {"start": 5, "end": 10}, "event_type": "Explosion"}
        assert self.prompter._get_event_trigger(event_mention) == (5, 10, "Explosion")

    def test_get_event_trigger_invalid_format(self):
        """Test _get_event_trigger raises TypeError for invalid format."""
        invalid_event_mention = {"trigger": "not_a_dict", "event_type": 123}
        with pytest.raises(TypeError, match="Invalid event_mention format"):
            self.prompter._get_event_trigger(invalid_event_mention)

    def test_get_tagged_text_valid(self):
        """Test _get_tagged_text properly tags trigger words."""
        tokens = ["The", "bomb", "exploded", "suddenly"]
        trig_start, trig_end = 2, 3
        expected_output = "The bomb <trigger>exploded</trigger> suddenly"
        assert (
            self.prompter._get_tagged_text(tokens, trig_start, trig_end)
            == expected_output
        )

    def test_get_tagged_text_invalid_indices(self):
        """Test _get_tagged_text raises ValueError for invalid indices."""
        tokens = ["A", "small", "fire"]
        with pytest.raises(ValueError, match="Invalid trigger indices"):
            self.prompter._get_tagged_text(tokens, 3, 4)

    def test_construct_prompt(self):
        """Test _construct_prompt returns expected format and output."""
        prompt, output_format, ids = self.prompter._construct_prompt(DATA, ED_SCHEMA)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}

    def test_get_prompt(self):
        """Test get_prompt calls _construct_prompt and returns expected values."""
        prompt, output_format, ids = self.prompter.get_prompt(DATA, ED_SCHEMA)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}


### TESTING EDPrompter2 ###
class TestEDPrompter2:
    def setup_method(self):
        """Setup an instance of EDPrompter2 for testing."""
        self.prompter = EDPrompter2()

    def test_construct_prompt(self):
        """Test _construct_prompt for EDPrompter2 returns expected format and output."""
        prompt, output_format, ids = self.prompter._construct_prompt(DATA, ED_SCHEMA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)  # Should return a JSON schema
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}

    def test_get_prompt(self):
        """Test get_prompt for EDPrompter2 works correctly."""
        prompt, output_format, ids = self.prompter.get_prompt(DATA, ED_SCHEMA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}


### TESTING EDPrompter3 ###
class TestEDPrompter3:
    def setup_method(self):
        """Setup an instance of EDPrompter3 for testing."""
        self.prompter = EDPrompter3()

    def test_get_examples(self):
        """Test _get_examples returns expected examples."""

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        example_kwargs = {"seed": 42}
        selected_examples = self.prompter._get_examples(
            temp_file_path, num_examples=3, **example_kwargs
        )

        assert len(selected_examples) == 3, "Should select exactly 3 examples"
        assert all(
            example in EXAMPLE_RECORDS for example in selected_examples
        ), "Selected examples should be from input data"

        # Ensure reproducibility
        selected_examples_again = self.prompter._get_examples(
            temp_file_path, num_examples=3, **example_kwargs
        )
        assert (
            selected_examples == selected_examples_again
        ), "Results should be reproducible with the same seed"

    def test_construct_examples(self):
        """Test _construct_examples returns expected format and output."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        example_kwargs = {"seed": 42}
        examples_text = self.prompter._construct_examples(
            temp_file_path, num_examples=3, **example_kwargs
        )
        assert "### Example 1" in examples_text
        assert "### Example 2" in examples_text
        assert "### Example 3" in examples_text
        assert EXAMPLE_RECORDS[0]["text"] in examples_text
        assert EXAMPLE_RECORDS[3]["text"] in examples_text
        assert EXAMPLE_RECORDS[4]["text"] in examples_text
        assert "Expected Response (JSON):" in examples_text


### TESTING EDPrompter4 ###
class TestEDPrompter4:
    def setup_method(self):
        """Setup an instance of EDPrompter4 for testing."""
        self.prompter = EDPrompter4()

    def test_get_examples(self):
        """Test _get_examples returns expected examples."""

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        target_text = "An example of a target sentence."
        example_kwargs = {
            "sentence_transformer_model": model_name,
            "target_text": target_text,
        }
        selected_examples = self.prompter._get_examples(
            temp_file_path, num_examples=3, **example_kwargs
        )

        assert len(selected_examples) == 3, "Should select exactly 3 examples"
        assert all(
            example in EXAMPLE_RECORDS for example in selected_examples
        ), "Selected examples should be from input data"

        # Ensure reproducibility
        selected_examples_again = self.prompter._get_examples(
            temp_file_path, num_examples=3, **example_kwargs
        )
        assert (
            selected_examples == selected_examples_again
        ), "Results should be reproducible with the same settings"


### TESTING EAEPrompter1 ###
class TestEAEPrompter1:
    def setup_method(self):
        """Setup an instance of EAEPrompter1 for testing."""
        self.prompter = EAEPrompter1()

    def test_construct_prompt(self):
        """
        Test _construct_prompt for EAEPrompter1 returns expected format and output.
        """
        prompt, output_format, ids = self.prompter._construct_prompt(DATA, EAE_SCHEMA)

        assert isinstance(prompt, str)
        assert output_format == "json"
        assert "Attacker" in prompt
        assert "Victim" in prompt
        assert "Location" in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"evt_id": "ev1"}

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter1 works correctly."""
        prompt, output_format, ids = self.prompter.get_prompt(DATA, EAE_SCHEMA)

        assert isinstance(prompt, str)
        assert output_format == "json"
        assert "Attacker" in prompt
        assert "Victim" in prompt
        assert "Location" in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"evt_id": "ev1"}


### TESTING EAEPrompter2 ###
class TestEAEPrompter2:
    def setup_method(self):
        """Setup an instance of EAEPrompter2 for testing."""
        self.prompter = EAEPrompter2()

    def test_construct_prompt(self):
        """
        Test _construct_prompt for EAEPrompter2 returns expected format and output.
        """
        prompt, output_format, ids = self.prompter._construct_prompt(DATA, EAE_SCHEMA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)  # Should return a JSON schema
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert "Attacker" in prompt
        assert "Victim" in prompt
        assert "Location" in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"evt_id": "ev1"}

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter2 works correctly."""
        prompt, output_format, ids = self.prompter.get_prompt(DATA, EAE_SCHEMA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert "Attacker" in prompt
        assert "Victim" in prompt
        assert "Location" in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"evt_id": "ev1"}
