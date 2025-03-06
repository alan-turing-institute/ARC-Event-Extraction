from __future__ import annotations

import json
import tempfile

import pytest

from arcee.dataset.schemas import get_arg_roles, pattern_to_json_schema
from arcee.prompts.prompts import (
    EAEPrompter1,
    EAEPrompter2,
    EAEPrompter3,
    EAEPrompter4,
    EAEPrompter5,
    EAEPrompter6,
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
        {"id": "ev1", "trigger": {"start": 1, "end": 2}, "event_type": "Explosion"}
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
    "Fire": {"roles": ["Agent", "Location"]},
    "Hurricane": {"roles": ["Location"]},
    "Earthquake": {"roles": ["Location"]},
    "Protest": {"roles": ["Location"]},
}

EXAMPLE_RECORDS = [
    {
        "doc_id": "doc1",
        "wnd_id": "wnd1",
        "text": "The explosion destroyed the car.",
        "tokens": ["The", "explosion", "destroyed", "the", "car", "."],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "explosion", "start": 1, "end": 2},
                "event_type": "Explosion",
                "arguments": [{"role": "Victim", "text": "car"}],
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
                "arguments": [{"role": "Location", "text": "warehouse"}],
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
                "arguments": [{"role": "Location", "text": None}],
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
                "arguments": [{"role": "Location", "text": "city"}],
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
                "arguments": [{"role": "Location", "text": None}],
            }
        ],
    },
]

EXAMPLE_RECORDS_SAME_TYPE = [
    {
        "doc_id": "doc1",
        "wnd_id": "wnd1",
        "text": "The explosion destroyed the car.",
        "tokens": ["The", "explosion", "destroyed", "the", "car", "."],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "explosion", "start": 1, "end": 2},
                "event_type": "Explosion",
                "arguments": [{"role": "Victim", "text": "car"}],
            }
        ],
    },
    {
        "doc_id": "doc2",
        "wnd_id": "wnd2",
        "text": "The explosion made it hard to hear the TV.",
        "tokens": [
            "The",
            "explosion",
            "made",
            "it",
            "hard",
            "to",
            "hear",
            "the",
            "TV",
            ".",
        ],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "explosion", "start": 1, "end": 2},
                "event_type": "Explosion",
                "arguments": [{}],
            }
        ],
    },
    {
        "doc_id": "doc3",
        "wnd_id": "wnd3",
        "text": "In the middle of the ocean there was an explosion.",
        "tokens": [
            "In",
            "the",
            "middle",
            "of",
            "the",
            "ocean",
            "there",
            "was",
            "an",
            "explosion",
            ".",
        ],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "explosion", "start": 9, "end": 10},
                "event_type": "Explosion",
                "arguments": [{"role": "Location", "text": "ocean"}],
            }
        ],
    },
    {
        "doc_id": "doc1",
        "wnd_id": "wnd1",
        "text": "The bomb went off at the start of the attack in the city.",
        "tokens": [
            "The",
            "bomb",
            "went",
            "off",
            "at",
            "the",
            "start",
            "of",
            "the",
            "attack",
            "in",
            "the",
            "city",
            ".",
        ],
        "event_mentions": [
            {
                "id": "ev1",
                "trigger": {"text": "attack", "start": 9, "end": 210},
                "event_type": "Attack",
                "arguments": [],
            },
            {
                "id": "ev2",
                "trigger": {"text": "went off", "start": 2, "end": 4},
                "event_type": "Explosion",
                "arguments": [{"role": "Location", "text": "city"}],
            },
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
        self.prompter = EDPrompter1(ED_SCHEMA)

    def test_get_system_content(self):
        """Test _get_system_content returns expected system message."""
        expected_content = (
            "You are an assistant that extracts event triggers and their associated "
            "event types from input text. Many triggers are verbs and represent a "
            "change of state or action in the text.\n"
            "---\n"
        )
        assert self.prompter._get_system_content() == expected_content

    def test_get_prompt(self):
        """Test get_prompt calls _construct_prompt and returns expected values."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert output_format == "json"
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}
        assert input_text == DATA["text"]


### TESTING EDPrompter2 ###
class TestEDPrompter2:
    def setup_method(self):
        """Setup an instance of EDPrompter2 for testing."""
        self.prompter = EDPrompter2(ED_SCHEMA)

    def test_get_prompt(self):
        """Test get_prompt for EDPrompter2 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}
        assert input_text == DATA["text"]


### TESTING EDPrompter3 ###
class TestEDPrompter3:
    def setup_method(self):
        """Setup an instance of EDPrompter3 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs = {"example_data_path": temp_file_path, "num_examples": 3, "seed": 42}
        self.prompter = EDPrompter3(ED_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EDPrompter3 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert EXAMPLE_RECORDS[0]["text"] in prompt
        assert EXAMPLE_RECORDS[1]["text"] not in prompt
        assert EXAMPLE_RECORDS[2]["text"] not in prompt
        assert EXAMPLE_RECORDS[3]["text"] in prompt
        assert EXAMPLE_RECORDS[4]["text"] in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}
        assert input_text == DATA["text"]


### TESTING EDPrompter4 ###
class TestEDPrompter4:
    def setup_method(self):
        """Setup an instance of EDPrompter4 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs = {
            "example_data_path": temp_file_path,
            "num_examples": 3,
            "sentence_transformer_model": (
                "sentence-transformers-testing/stsb-bert-tiny-safetensors"
            ),
        }
        self.prompter_no_events = EDPrompter4(ED_SCHEMA, **kwargs)

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS_SAME_TYPE:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs["example_data_path"] = temp_file_path
        kwargs["num_examples"] = 2
        self.prompter_same_events = EDPrompter4(ED_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EDPrompter4 works correctly."""
        # The 'prompter_no_events' doesn't have access to enough events of the same type
        # as the input data, so it should just use random examples, regardless of type
        prompt, output_format, ids, input_text = self.prompter_no_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert EXAMPLE_RECORDS[0]["text"] in prompt
        assert EXAMPLE_RECORDS[1]["text"] in prompt
        assert EXAMPLE_RECORDS[2]["text"] not in prompt
        assert EXAMPLE_RECORDS[3]["text"] in prompt
        assert EXAMPLE_RECORDS[4]["text"] not in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}
        assert input_text == DATA["text"]

        prompt, output_format, ids, input_text = self.prompter_same_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == JSON_SCHEMA
        assert "Explosion: An event where something bursts violently." in prompt
        assert "Attack: A violent physical act causing harm or damage." in prompt
        assert "The explosion destroyed the building." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" not in prompt
        assert "### Example 4" not in prompt
        assert EXAMPLE_RECORDS_SAME_TYPE[0]["text"] in prompt
        assert EXAMPLE_RECORDS_SAME_TYPE[1]["text"] not in prompt
        assert EXAMPLE_RECORDS_SAME_TYPE[2]["text"] in prompt
        assert EXAMPLE_RECORDS_SAME_TYPE[3]["text"] not in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"wnd_id": "wnd1"}
        assert input_text == DATA["text"]


### TESTING EAEPrompter1 ###
class TestEAEPrompter1:
    def setup_method(self):
        """Setup an instance of EAEPrompter1 for testing."""
        self.prompter = EAEPrompter1(EAE_SCHEMA)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter1 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert isinstance(prompt, str)
        assert output_format == "json"
        assert "Attacker" in prompt
        assert "Victim" in prompt
        assert "Location" in prompt
        assert "Format the output strictly as JSON" in prompt
        assert ids == {"evt_id": "ev1"}
        assert input_text == DATA["text"]

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


### TESTING EAEPrompter2 ###
class TestEAEPrompter2:
    def setup_method(self):
        """Setup an instance of EAEPrompter2 for testing."""
        self.prompter = EAEPrompter2(EAE_SCHEMA)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter2 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

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
        assert input_text == DATA["text"]


### TESTING EAEPrompter3 ###
class TestEAEPrompter3:
    def setup_method(self):
        """Setup an instance of EAEPrompter3 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs = {"example_data_path": temp_file_path, "num_examples": 3, "seed": 42}
        self.prompter = EAEPrompter3(EAE_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter3 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert "A <trigger>fire</trigger> broke out in the warehouse ." not in prompt
        assert "The <trigger>hurricane</trigger> caused severe damage ." not in prompt
        assert "An <trigger>earthquake</trigger> shook the city last night ." in prompt
        assert "The <trigger>protest</trigger> escalated into a riot ." in prompt
        assert '{"evt_id": "ev1", "output": ' in prompt
        assert '{"evt_id": "ev4", "output": {"Location": "city"}}' in prompt
        assert '{"evt_id": "ev5", "output": {"Location": null}}' in prompt
        assert input_text == DATA["text"]


### TESTING EAEPrompter4 ###
class TestEAEPrompter4:
    def setup_method(self):
        """Setup an instance of EAEPrompter4 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        # Set up one prompt where the example data doesn't have enough of the same
        # event type as the input data
        kwargs = {"example_data_path": temp_file_path, "num_examples": 3, "seed": 42}
        self.prompter_no_events = EAEPrompter4(EAE_SCHEMA, **kwargs)

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS_SAME_TYPE:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        # Set up one prompt where the example data doesn't have enough of the same
        # event type as the input data
        kwargs = {"example_data_path": temp_file_path, "num_examples": 2, "seed": 42}
        self.prompter_same_events = EAEPrompter4(EAE_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter4 works correctly."""
        # The 'prompter_no_events' doesn't have access to enough events of the same type
        # as the input data, so it should just use random examples, regardless of type
        # (i.e., the same as EAEPrompter3)
        prompt, output_format, ids, input_text = self.prompter_no_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert "A <trigger>fire</trigger> broke out in the warehouse ." not in prompt
        assert "The <trigger>hurricane</trigger> caused severe damage ." not in prompt
        assert "An <trigger>earthquake</trigger> shook the city last night ." in prompt
        assert "The <trigger>protest</trigger> escalated into a riot ." in prompt
        assert '{"evt_id": "ev1", "output": ' in prompt
        assert '{"evt_id": "ev4", "output": {"Location": "city"}}' in prompt
        assert '{"evt_id": "ev5", "output": {"Location": null}}' in prompt
        assert input_text == DATA["text"]

        prompt, output_format, ids, input_text = self.prompter_same_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" not in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert (
            "The bomb <trigger>went off</trigger> at the start of the attack in the "
            "city ." in prompt
        )
        assert input_text == DATA["text"]


### TESTING EAEPrompter5 ###
class TestEAEPrompter5:
    def setup_method(self):
        """Setup an instance of EAEPrompter5 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs = {
            "example_data_path": temp_file_path,
            "num_examples": 3,
            "sentence_transformer_model": (
                "sentence-transformers-testing/stsb-bert-tiny-safetensors"
            ),
        }
        self.prompter = EAEPrompter5(EAE_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter5 works correctly."""
        prompt, output_format, ids, input_text = self.prompter.get_prompt(DATA)

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert "A <trigger>fire</trigger> broke out in the warehouse ." in prompt
        assert "The <trigger>hurricane</trigger> caused severe damage ." not in prompt
        assert "An <trigger>earthquake</trigger> shook the city last night ." in prompt
        assert "The <trigger>protest</trigger> escalated into a riot ." not in prompt
        assert '{"evt_id": "ev1", "output": ' in prompt
        assert '{"evt_id": "ev2", "output": ' in prompt
        assert '{"evt_id": "ev4", "output": {"Location": "city"}}' in prompt
        assert input_text == DATA["text"]


### TESTING EAEPrompter6 ###
class TestEAEPrompter6:
    def setup_method(self):
        """Setup an instance of EAEPrompter6 for testing."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs = {
            "example_data_path": temp_file_path,
            "num_examples": 3,
            "sentence_transformer_model": (
                "sentence-transformers-testing/stsb-bert-tiny-safetensors"
            ),
        }
        self.prompter_no_events = EAEPrompter6(EAE_SCHEMA, **kwargs)

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        ) as temp_file:
            for record in EXAMPLE_RECORDS_SAME_TYPE:
                temp_file.write(json.dumps(record) + "\n")
            temp_file_path = temp_file.name

        kwargs["example_data_path"] = temp_file_path
        kwargs["num_examples"] = 2
        self.prompter_same_events = EAEPrompter6(EAE_SCHEMA, **kwargs)

    def test_get_prompt(self):
        """Test get_prompt for EAEPrompter5 works correctly."""
        prompt, output_format, ids, input_text = self.prompter_no_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert "A <trigger>fire</trigger> broke out in the warehouse ." in prompt
        assert "The <trigger>hurricane</trigger> caused severe damage ." not in prompt
        assert "An <trigger>earthquake</trigger> shook the city last night ." in prompt
        assert "The <trigger>protest</trigger> escalated into a riot ." not in prompt
        assert '{"evt_id": "ev1", "output": ' in prompt
        assert '{"evt_id": "ev2", "output": ' in prompt
        assert '{"evt_id": "ev4", "output": {"Location": "city"}}' in prompt
        assert input_text == DATA["text"]

        prompt, output_format, ids, input_text = self.prompter_same_events.get_prompt(
            DATA
        )

        assert isinstance(prompt, str)
        assert isinstance(output_format, dict)
        assert output_format == pattern_to_json_schema(
            "Explosion", EAE_SCHEMA["Explosion"]["roles"]
        )
        assert ids == {"evt_id": "ev1"}

        assert "The <trigger>explosion</trigger> destroyed the building ." in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt
        assert "### Example 3" not in prompt
        assert "### Example 4" not in prompt
        assert "The <trigger>explosion</trigger> destroyed the car ." in prompt
        assert (
            "In the middle of the ocean there was an <trigger>explosion</trigger> ."
            in prompt
        )
        assert input_text == DATA["text"]
