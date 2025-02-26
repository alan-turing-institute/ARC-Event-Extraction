from __future__ import annotations

from pathlib import Path

import pytest

from arcee.config.experiment import ExperimentConfig

TEST_DIR = Path(__file__, "..").resolve()
TEST_CONFIG_DIR = TEST_DIR / "configs"
TEST_DATA_DIR = TEST_DIR / "data"


@pytest.fixture(scope="module")
def test_tokens() -> list[str]:
    return [
        "The",
        "university",
        "hired",
        "Dr",
        ".",
        "Smith",
        "to",
        "lead",
        "the",
        "biology",
        "department",
        ".",
    ]


@pytest.fixture(scope="module")
def dummy_experment_config():
    return ExperimentConfig.from_yaml(TEST_CONFIG_DIR / "dummy_experiment.yaml")


@pytest.fixture(scope="module")
def test_event_mentions() -> list:
    return [{"event_type": "hire", "trigger": {"text": "hired", "start": 2, "end": 3}}]


@pytest.fixture(scope="module")
def test_schema() -> dict:
    return {"hire": {"roles": ["object", "subject"]}}


@pytest.fixture(scope="module")
def test_batch_prompts() -> list[dict[str, str]]:
    prompt1 = (
        "An event type of hire has been detected in the input text. "
        "The trigger word is identified with a <trigger> tag.\n "
        "Input text:\n```The university <trigger>hired</trigger> Dr . "
        "Smith to lead the biology department .```\n What are the words "
        "in the input text that represent the argument role ```object```? "
        "Provide your response in JSON format with the "
        "key object and the value as the words in the input text "
        "that represent the argument."
    )
    prompt2 = (
        "An event type of hire has been detected in the input text. "
        "The trigger word is identified with a <trigger> tag.\n "
        "Input text:\n```The university <trigger>hired</trigger> Dr . "
        "Jones to lead the physics department .```\n What are the words "
        "in the input text that represent the argument role ```object```? "
        "Provide your response in JSON format with the "
        "key object and the value as the words in the input text "
        "that represent the argument."
    )
    return [{"role": "user", "content": prompt1}, {"role": "user", "content": prompt2}]


@pytest.fixture(scope="module")
def test_event_records():
    """Fixture providing a list of five event records with varying structures."""
    return [
        # Record with an empty event_mentions list
        {
            "doc_id": "doc_001",
            "wnd_id": "doc_001-1",
            "text": "The sun is shining brightly today.",
            "lang": "en",
            "tokens": ["The", "sun", "is", "shining", "brightly", "today", "."],
            "entity_mentions": [],
            "event_mentions": [],
            "relation_mentions": [],
        },
        # Record with one event mention
        {
            "doc_id": "doc_002",
            "wnd_id": "doc_002-1",
            "text": "The protest escalated into a riot.",
            "lang": "en",
            "tokens": ["The", "protest", "escalated", "into", "a", "riot", "."],
            "entity_mentions": [
                {
                    "text": "protest",
                    "id": "doc_002-E1",
                    "start": 1,
                    "end": 2,
                    "entity_type": "Event",
                    "entity_subtype": "Protest",
                    "mention_type": "NAM",
                }
            ],
            "event_mentions": [
                {
                    "id": "doc_002-EV1",
                    "event_main_type": "Conflict",
                    "event_subtype": "Demonstrate",
                    "event_type": "Conflict:Demonstrate",
                    "trigger": {"text": "protest", "start": 1, "end": 2},
                    "arguments": [],
                }
            ],
            "relation_mentions": [],
        },
        # Record with multiple event mentions
        {
            "doc_id": "doc_003",
            "wnd_id": "doc_003-1",
            "text": "An earthquake struck the city, causing buildings to collapse.",
            "lang": "en",
            "tokens": [
                "An",
                "earthquake",
                "struck",
                "the",
                "city",
                ",",
                "causing",
                "buildings",
                "to",
                "collapse",
                ".",
            ],
            "entity_mentions": [
                {
                    "text": "earthquake",
                    "id": "doc_003-E1",
                    "start": 1,
                    "end": 2,
                    "entity_type": "Disaster",
                    "entity_subtype": "Natural",
                    "mention_type": "NAM",
                },
                {
                    "text": "city",
                    "id": "doc_003-E2",
                    "start": 4,
                    "end": 5,
                    "entity_type": "GPE",
                    "entity_subtype": "City",
                    "mention_type": "NAM",
                },
            ],
            "event_mentions": [
                {
                    "id": "doc_003-EV1",
                    "event_main_type": "Disaster",
                    "event_subtype": "Earthquake",
                    "event_type": "Disaster:Earthquake",
                    "trigger": {"text": "earthquake", "start": 1, "end": 2},
                    "arguments": [],
                },
                {
                    "id": "doc_003-EV2",
                    "event_main_type": "Disaster",
                    "event_subtype": "Collapse",
                    "event_type": "Disaster:Collapse",
                    "trigger": {"text": "collapse", "start": 9, "end": 10},
                    "arguments": [],
                },
            ],
            "relation_mentions": [],
        },
        # Record with an entity mention but no event mentions
        {
            "doc_id": "doc_004",
            "wnd_id": "doc_004-1",
            "text": "The president gave a speech about economic policies.",
            "lang": "en",
            "tokens": [
                "The",
                "president",
                "gave",
                "a",
                "speech",
                "about",
                "economic",
                "policies",
                ".",
            ],
            "entity_mentions": [
                {
                    "text": "president",
                    "id": "doc_004-E1",
                    "start": 1,
                    "end": 2,
                    "entity_type": "PER",
                    "entity_subtype": "Individual",
                    "mention_type": "NAM",
                }
            ],
            "event_mentions": [],
            "relation_mentions": [],
        },
        # Record with a relation mention between entities
        {
            "doc_id": "doc_005",
            "wnd_id": "doc_005-1",
            "text": "The company acquired a startup in a multimillion-dollar deal.",
            "lang": "en",
            "tokens": [
                "The",
                "company",
                "acquired",
                "a",
                "startup",
                "in",
                "a",
                "multimillion",
                "-",
                "dollar",
                "deal",
                ".",
            ],
            "entity_mentions": [
                {
                    "text": "company",
                    "id": "doc_005-E1",
                    "start": 1,
                    "end": 2,
                    "entity_type": "ORG",
                    "entity_subtype": "Commercial",
                    "mention_type": "NAM",
                },
                {
                    "text": "startup",
                    "id": "doc_005-E2",
                    "start": 4,
                    "end": 5,
                    "entity_type": "ORG",
                    "entity_subtype": "Commercial",
                    "mention_type": "NAM",
                },
            ],
            "event_mentions": [
                {
                    "id": "doc_005-EV1",
                    "event_main_type": "Transaction",
                    "event_subtype": "TransferOwnership",
                    "event_type": "Transaction:TransferOwnership",
                    "trigger": {"text": "acquired", "start": 2, "end": 3},
                    "arguments": [],
                }
            ],
            "relation_mentions": [
                {
                    "id": "doc_005-R1",
                    "relation_type": "ORG-AFF",
                    "relation_subtype": "Ownership",
                    "arguments": [
                        {
                            "entity_id": "doc_005-E1",
                            "role": "Arg-1",
                            "text": "company",
                            "start": 1,
                            "end": 2,
                        },
                        {
                            "entity_id": "doc_005-E2",
                            "role": "Arg-2",
                            "text": "startup",
                            "start": 4,
                            "end": 5,
                        },
                    ],
                }
            ],
        },
    ]


@pytest.fixture(scope="module")
def pattern_data():
    """Fixture providing a valid JSON pattern."""
    return [
        {
            "Life:Marry": {
                "template": "<arg1> married <arg2> in <arg3> place",
                "roles": ["Person", "Person", "Place"],
            },
            "Movement:Transport": {
                "template": (
                    "<arg1> transported <arg2> in <arg3> vehicle from <arg4> "
                    "place to <arg5> place"
                ),
                "roles": ["Agent", "Artifact", "Vehicle", "Origin", "Destination"],
            },
        }
    ]
