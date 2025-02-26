from __future__ import annotations

import pytest

from arcee.dataset.schemas import get_arg_roles, pattern_to_json_schema
from arcee.prompts.prompts import EAEPrompter1, EAEPrompter2, EDPrompter1, EDPrompter2

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
