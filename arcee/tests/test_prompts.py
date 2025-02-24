from __future__ import annotations

import pytest

from arcee.prompts.prompts import EDPrompter1


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

    def test_get_arg_schema_description_single_args(self):
        """Test _get_arg_schema_description with unique roles."""
        arg_roles = ["Attacker", "Victim"]
        expected_output = (
            "There will be at most one argument for each role "
            "and there may be no arguments for a given role. "
        )
        assert self.prompter._get_arg_schema_description(arg_roles) == expected_output

    def test_get_arg_schema_description_multiple_roles(self):
        """Test _get_arg_schema_description with repeated roles."""
        arg_roles = ["participant", "participant", "place"]
        expected_output = [
            "There may be between zero and 2 arguments for role participant. ",
            "There may be between zero and 1 arguments for role place. ",
        ]
        assert expected_output[0] in self.prompter._get_arg_schema_description(
            arg_roles
        )
        assert expected_output[1] in self.prompter._get_arg_schema_description(
            arg_roles
        )

    def test_construct_prompt(self):
        """Test _construct_prompt returns expected format and output."""
        data = {"text": "The explosion destroyed the building."}
        schema = {
            "Explosion": "An event where something bursts violently.",
            "Attack": "A violent physical act causing harm or damage.",
        }
        prompt, output_format = self.prompter._construct_prompt(data, schema)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt

    def test_get_prompt(self):
        """Test get_prompt calls _construct_prompt and returns expected values."""
        data = {"text": "A bomb detonated downtown."}
        schema = {"Explosion": "An event where something bursts violently."}
        prompt, output_format = self.prompter.get_prompt(data, schema)

        assert isinstance(prompt, str)
        assert isinstance(output_format, str)
        assert output_format == "json"
        assert "A bomb detonated downtown." in prompt
        assert "Explosion: An event where something bursts violently." in prompt
