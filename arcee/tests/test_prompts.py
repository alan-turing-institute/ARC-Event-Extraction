from __future__ import annotations

import pytest

from arcee.prompts.prompts import EDPrompter1

DATA = {
    "doc_id": "doc1",
    "wnd_id": "wnd1",
    "text": "The explosion destroyed the building.",
    "tokens": ["The", "explosion", "destroyed", "the", "building", "."],
    "event_mentions": [
        {"id": "ev1", "trigger": {"start": 2, "end": 3}, "event_type": "Explosion"}
    ],
}

SCHEMA = {
    "Explosion": "An event where something bursts violently.",
    "Attack": "A violent physical act causing harm or damage.",
}


class TestEEPrompt:
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
        prompt, output_format, ids = self.prompter._construct_prompt(DATA, SCHEMA)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"doc_id": "doc1", "wnd_id": "wnd1"}

    def test_get_prompt(self):
        """Test get_prompt calls _construct_prompt and returns expected values."""
        prompt, output_format, ids = self.prompter.get_prompt(DATA, SCHEMA)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"doc_id": "doc1", "wnd_id": "wnd1"}


# class TestEAEPrompter1:
#     def setup_method(self):
#         """Setup an instance of EAEPrompter1 for testing."""
#         self.prompter = EAEPrompter1()

#     def test_get_system_content(self):
#         """Test _get_system_content returns expected system message."""
#         expected_content = (
#             "You are an assistant that extracts event arguments from input text. "
#             "Event arguments are the key entities, participants, and attributes "
#             "associated with an event in a text. They provide context and specific "
#             "details about what happened, who was involved, and where it occurred. "
#             "\n"
#             "---\n"
#         )
#         assert self.prompter._get_system_content() == expected_content

#     def test_get_arg_roles_valid(self):
#         """
#         Test _get_arg_roles returns correct argument roles for a given
#         event type.
#        """
#         schema = {
#             "Explosion": {"roles": ["Attacker", "Victim", "Location"]},
#             "Fire": {"roles": ["Cause", "Damage", "Location"]},
#         }
#         assert self.prompter._get_arg_roles(schema, "Explosion") == [
#             "Attacker",
#             "Victim",
#             "Location",
#         ]

#     def test_get_arg_roles_invalid_event_type(self):
#         """Test _get_arg_roles raises KeyError when event type is not in schema."""
#         schema = {"Explosion": {"roles": ["Attacker", "Victim"]}}
#         with pytest.raises(KeyError, match="Event type 'Flood' not found in schema."):
#             self.prompter._get_arg_roles(schema, "Flood")

#     def test_construct_prompt_valid(self):
#         """
#         Test _construct_prompt returns a valid prompt and expected output format.
#         """
#         data = {
#             "text": "A bomb exploded in the city center.",
#             "event_mentions": [
#                 {"trigger": {"start": 2, "end": 3}, "event_type": "Explosion"}
#             ],
#             "tokens": ["A", "bomb", "exploded", "in", "the", "city", "center", "."],
#         }
#         schema = {
#             "Explosion": {"roles": ["Attacker", "Victim", "Location"]},
#             "Fire": {"roles": ["Cause", "Damage", "Location"]},
#         }

#         prompt, output_format = self.prompter._construct_prompt(data, schema)

#         assert output_format == "json"
#         assert "An event type of Explosion has been detected" in prompt
#         assert "<trigger>exploded</trigger>" in prompt
#         assert "Attacker\nVictim\nLocation" in prompt
#         assert "Format the output strictly as JSON" in prompt

#     def test_construct_prompt_missing_event_type(self):
#         """
#         Test _construct_prompt raises KeyError if event type is missing from
#         schema.
#         """
#         data = {
#             "text": "A bomb exploded in the city center.",
#             "event_mentions": [
#                 {"trigger": {"start": 2, "end": 3}, "event_type": "Flood"}
#             ],
#             "tokens": ["A", "bomb", "exploded", "in", "the", "city", "center", "."],
#         }
#         schema = {
#             "Explosion": {"roles": ["Attacker", "Victim", "Location"]},
#             "Fire": {"roles": ["Cause", "Damage", "Location"]},
#         }

#         with pytest.raises(KeyError, match="Event type 'Flood' not found in schema."):
#             self.prompter._construct_prompt(data, schema)

#     def test_construct_prompt_with_no_arguments(self):
#         """Test _construct_prompt when an event has no associated arguments."""
#         data = {
#             "text": "An earthquake shook the city.",
#             "event_mentions": [
#                 {"trigger": {"start": 1, "end": 2}, "event_type": "Earthquake"}
#             ],
#             "tokens": ["An", "earthquake", "shook", "the", "city", "."],
#         }
#         schema = {
#             "Earthquake": {"roles": []},  # No argument roles
#         }

#         prompt, output_format = self.prompter._construct_prompt(data, schema)

#         assert output_format == "json"
#         assert "An event type of Earthquake has been detected" in prompt
#         assert "<trigger>earthquake</trigger>" in prompt
#         assert "Format the output strictly as JSON" in prompt
#         assert (
#             "What are the words in the input text that represent the following "
#             "argument roles?" not in prompt
#         )

#     def test_get_prompt(self):
#         """Test get_prompt calls _construct_prompt and returns expected values."""
#         data = {
#             "text": "A car crash injured several people.",
#             "event_mentions": [
#                 {"trigger": {"start": 2, "end": 3}, "event_type": "Accident"}
#             ],
#             "tokens": ["A", "car", "crash", "injured", "several", "people", "."],
#         }
#         schema = {
#             "Accident": {"roles": ["Driver", "Victim", "Location"]},
#         }

#         prompt, output_format = self.prompter.get_prompt(data, schema)

#         assert isinstance(prompt, str)
#         assert isinstance(output_format, str)
#         assert output_format == "json"
#         assert "A car crash injured several people." in prompt
#         assert "<trigger>crash</trigger>" in prompt
#         assert "Driver\nVictim\nLocation" in prompt
#         assert "Format the output strictly as JSON" in prompt
